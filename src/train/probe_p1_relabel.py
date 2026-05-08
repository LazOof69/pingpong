"""
P1: Estimated-test-marginal plug-in re-fit (Alexandari 2020 bias-corrected pipeline).

Pipeline:
1. Load bag2 OOF + test predictions (raw, post-α-blend, pre-bias)
2. Temperature scale OOF logits to fix calibration bias (Alexandari 2020)
3. Estimate q_test(y) via stable Saerens EM with class masking + smoothing
4. Re-fit 29-param additive bias using q-weighted macro-F1 objective on OOF
5. Apply new bias to test → submission CSV
6. Verification: held-out OOF half, weighted-F1 lift check

Output:
- artifacts/probe_p1_relabel.npz (q estimates, biases, F1 metrics)
- submissions/submission_v5z_md_p1_relabel.csv (new prediction)
"""
import os, sys, time
os.environ["V5Z_MATCH_DISJOINT"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.optimize import minimize_scalar

from train_v5z import prepare_samples, train_df, N_ACTION, N_POINT, N_FOLDS, log


def reindex_for_seed(seed, train_samples, uid_list, uid_arr, match_arr, uid2idx):
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    s2o = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for _, va_uidx in sgkf.split(uid_list, uid_arr, groups=match_arr):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                s2o[i] = cur; cur += 1
    return s2o


def temp_scale_nll(T, probs, labels):
    """NLL on labels under temperature-scaled probs."""
    log_p = np.log(probs + 1e-12) / T
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    p_T = np.exp(log_p)
    p_T = p_T / p_T.sum(axis=1, keepdims=True)
    return -np.log(p_T[np.arange(len(labels)), labels] + 1e-12).mean()


def apply_temp(probs, T):
    log_p = np.log(probs + 1e-12) / T
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    p_T = np.exp(log_p)
    return p_T / p_T.sum(axis=1, keepdims=True)


def stable_saerens(P_test, p_train, valid_classes, n_iter=200, smooth=0.02, tol=1e-7):
    """Saerens EM with class masking + p_OOF floor + Dirichlet smoothing."""
    K = P_test.shape[1]
    mask = np.zeros(K, dtype=bool); mask[valid_classes] = True
    p_train_safe = np.maximum(p_train, 0.005)
    p_train_safe[~mask] = 0
    p_train_norm = p_train.copy(); p_train_norm[~mask] = 0
    p_train_norm = p_train_norm / max(p_train_norm.sum(), 1e-12)

    P_test_m = P_test.copy(); P_test_m[:, ~mask] = 0
    P_test_m = P_test_m / np.maximum(P_test_m.sum(axis=1, keepdims=True), 1e-12)

    q = p_train_norm.copy()
    for it in range(n_iter):
        ratio = q / np.maximum(p_train_safe, 1e-3)
        ratio[~mask] = 0
        P_adj = P_test_m * ratio[None, :]
        P_adj = P_adj / np.maximum(P_adj.sum(axis=1, keepdims=True), 1e-12)
        q_new = P_adj.mean(axis=0)
        q_new[~mask] = 0
        q_new = (1 - smooth) * q_new + smooth * p_train_norm
        q_new[~mask] = 0
        q_sum = q_new.sum()
        if q_sum > 0:
            q_new = q_new / q_sum
        diff = np.abs(q_new - q).max()
        q = q_new
        if diff < tol: break
    return q, it+1


def fit_plugin_qweighted(probs, labels, q, p_oof, n_class, valid_classes,
                         grid_step=0.05, grid_range=2.5, n_rounds=4):
    """Re-fit 29-param additive bias with q-weighted macro-F1 objective."""
    log_p = np.log(probs + 1e-12)
    # Per-sample weight: q(y_true) / p_OOF(y_true), capped
    sample_w = np.clip(q[labels] / np.maximum(p_oof[labels], 0.005), 0.1, 10.0)
    sample_w = sample_w / sample_w.mean()

    def weighted_macro_f1(b):
        pred = (log_p + b).argmax(1)
        f1s = []
        for c in valid_classes:
            tp_mask = (pred == c) & (labels == c)
            fp_mask = (pred == c) & (labels != c)
            fn_mask = (pred != c) & (labels == c)
            tp = sample_w[tp_mask].sum()
            fp = sample_w[fp_mask].sum()
            fn = sample_w[fn_mask].sum()
            denom = tp + 0.5 * (fp + fn)
            f1s.append(tp / denom if denom > 0 else 0.0)
        return float(np.mean(f1s))

    b = np.zeros(n_class)
    best_f1 = weighted_macro_f1(b)
    for r in range(n_rounds):
        improved = False
        for c in valid_classes:
            best_c = b[c]; best_at_c = best_f1
            for d in np.arange(-grid_range, grid_range + 1e-6, grid_step):
                b_try = b.copy(); b_try[c] = d
                f1 = weighted_macro_f1(b_try)
                if f1 > best_at_c:
                    best_at_c = f1; best_c = d
            if best_at_c > best_f1:
                best_f1 = best_at_c
                b[c] = best_c
                improved = True
        if not improved: break
    return b, best_f1


def fit_plugin_standard(probs, labels, n_class, valid_classes,
                        grid_step=0.05, grid_range=2.5, n_rounds=4):
    """Standard plug-in (current ship's objective)."""
    log_p = np.log(probs + 1e-12)
    def macro_f1(b):
        pred = (log_p + b).argmax(1)
        return f1_score(labels, pred, average='macro', zero_division=0,
                        labels=valid_classes)
    b = np.zeros(n_class)
    best_f1 = macro_f1(b)
    for r in range(n_rounds):
        improved = False
        for c in valid_classes:
            best_c = b[c]; best_at_c = best_f1
            for d in np.arange(-grid_range, grid_range + 1e-6, grid_step):
                b_try = b.copy(); b_try[c] = d
                f1 = macro_f1(b_try)
                if f1 > best_at_c:
                    best_at_c = f1; best_c = d
            if best_at_c > best_f1:
                best_f1 = best_at_c
                b[c] = best_c
                improved = True
        if not improved: break
    return b, best_f1


def main():
    log("="*78)
    log("P1: Estimated-test-marginal plug-in re-fit")
    log("="*78)

    # ============================================================
    # Load artifacts
    # ============================================================
    oof_s42 = np.load('artifacts/v5z_md_full_s42_oof.npz', allow_pickle=True)
    oof_s1337 = np.load('artifacts/v5z_md_full_s1337_oof.npz', allow_pickle=True)
    test_s42 = np.load('artifacts/v5z_md_full_s42_test.npz', allow_pickle=True)
    test_s1337 = np.load('artifacts/v5z_md_full_s1337_test.npz', allow_pickle=True)
    advcal = np.load('artifacts/v5z_md_advcal_bag2_params.npz', allow_pickle=True)
    a_alpha = float(advcal['a_alpha'])
    p_alpha = float(advcal['p_alpha'])
    bias_a_curr = advcal['bias_a']
    bias_p_curr = advcal['bias_p']
    log(f"Current ship: a_alpha={a_alpha}, p_alpha={p_alpha}")

    # Reindex OOF to sample order
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_arr = np.array([uid_to_match[u] for u in uid_list])
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)

    s2o_42 = reindex_for_seed(42, train_samples, uid_list, uid_arr, match_arr, uid2idx)
    s2o_1337 = reindex_for_seed(1337, train_samples, uid_list, uid_arr, match_arr, uid2idx)

    # Bag2 raw probs (sample order)
    def get_can(d, s2o):
        return {k: d[k][s2o] for k in ['lstm_a','lstm_p','lgb_a','lgb_p','la','lp']}
    c42 = get_can(oof_s42, s2o_42); c13 = get_can(oof_s1337, s2o_1337)
    assert (c42['la'] == c13['la']).all()
    la = c42['la']; lp = c42['lp']

    oof_a_raw = a_alpha * (c42['lstm_a'] + c13['lstm_a']) / 2 + \
                (1 - a_alpha) * (c42['lgb_a'] + c13['lgb_a']) / 2
    oof_p_raw = p_alpha * (c42['lstm_p'] + c13['lstm_p']) / 2 + \
                (1 - p_alpha) * (c42['lgb_p'] + c13['lgb_p']) / 2

    test_a_raw = a_alpha * (test_s42['lstm_a'] + test_s1337['lstm_a']) / 2 + \
                 (1 - a_alpha) * (test_s42['lgb_a'] + test_s1337['lgb_a']) / 2
    test_p_raw = p_alpha * (test_s42['lstm_p'] + test_s1337['lstm_p']) / 2 + \
                 (1 - p_alpha) * (test_s42['lgb_p'] + test_s1337['lgb_p']) / 2

    # Sanity: current ship OOF F1
    pred_a_curr = (np.log(oof_a_raw + 1e-12) + bias_a_curr).argmax(1)
    pred_p_curr = (np.log(oof_p_raw + 1e-12) + bias_p_curr).argmax(1)
    f1a_curr = f1_score(la, pred_a_curr, average='macro', zero_division=0)
    f1p_curr = f1_score(lp, pred_p_curr, average='macro', zero_division=0)
    score_curr = 0.4*f1a_curr + 0.4*f1p_curr + 0.2
    log(f"\n[Sanity] Current ship OOF: F1_a={f1a_curr:.4f}, F1_p={f1p_curr:.4f}, Score={score_curr:.4f}")

    # ============================================================
    # Step 1: Temperature scaling (Alexandari 2020 bias correction)
    # ============================================================
    log("\n" + "─"*78)
    log("Step 1: Temperature scaling on OOF (Alexandari 2020)")
    log("─"*78)
    res_a = minimize_scalar(temp_scale_nll, args=(oof_a_raw, la),
                            bounds=(0.1, 10.0), method='bounded')
    T_a = res_a.x
    res_p = minimize_scalar(temp_scale_nll, args=(oof_p_raw, lp),
                            bounds=(0.1, 10.0), method='bounded')
    T_p = res_p.x
    log(f"  T_action = {T_a:.3f} (NLL: {res_a.fun:.4f})")
    log(f"  T_point  = {T_p:.3f} (NLL: {res_p.fun:.4f})")

    oof_a_T = apply_temp(oof_a_raw, T_a)
    oof_p_T = apply_temp(oof_p_raw, T_p)
    test_a_T = apply_temp(test_a_raw, T_a)
    test_p_T = apply_temp(test_p_raw, T_p)

    # ============================================================
    # Step 2: Saerens EM (stable)
    # ============================================================
    log("\n" + "─"*78)
    log("Step 2: Saerens EM for q_test estimate")
    log("─"*78)
    p_oof_a = np.bincount(la, minlength=N_ACTION) / len(la)
    p_oof_p = np.bincount(lp, minlength=N_POINT) / len(lp)

    # action: classes 0-14 valid (15-18 never as next-stroke target per CLAUDE.md §10)
    valid_a = list(range(15))
    valid_p = list(range(N_POINT))

    q_a, it_a = stable_saerens(test_a_T, p_oof_a, valid_a)
    q_p, it_p = stable_saerens(test_p_T, p_oof_p, valid_p)
    log(f"  Saerens converged: action {it_a} iter, point {it_p} iter")

    # Show q vs p_OOF
    log(f"\n  Action q_test vs p_OOF (active classes 0-14):")
    log(f"  {'cls':<5}{'p_OOF':<10}{'q_test':<10}{'q-p':<10}{'q/p':<10}")
    for c in valid_a:
        log(f"  {c:<5}{p_oof_a[c]:<10.4f}{q_a[c]:<10.4f}{q_a[c]-p_oof_a[c]:<+10.4f}{q_a[c]/max(p_oof_a[c],1e-6):<10.3f}")
    log(f"\n  Point q_test vs p_OOF:")
    for c in valid_p:
        log(f"  {c:<5}{p_oof_p[c]:<10.4f}{q_p[c]:<10.4f}{q_p[c]-p_oof_p[c]:<+10.4f}{q_p[c]/max(p_oof_p[c],1e-6):<10.3f}")

    # ============================================================
    # Step 3: Re-fit plug-in with q-weighted macro-F1
    # ============================================================
    log("\n" + "─"*78)
    log("Step 3: Re-fit plug-in with q-weighted macro-F1 objective")
    log("─"*78)
    log("  Fitting action plug-in (this may take 1-2 min)...")
    t0 = time.time()
    bias_a_p1, wf1_a = fit_plugin_qweighted(oof_a_T, la, q_a, p_oof_a, N_ACTION, valid_a)
    log(f"  Action: weighted F1 = {wf1_a:.4f} ({time.time()-t0:.1f}s)")
    log("  Fitting point plug-in...")
    t0 = time.time()
    bias_p_p1, wf1_p = fit_plugin_qweighted(oof_p_T, lp, q_p, p_oof_p, N_POINT, valid_p)
    log(f"  Point: weighted F1 = {wf1_p:.4f} ({time.time()-t0:.1f}s)")

    # Evaluate on STANDARD macro-F1 (vs current ship)
    pred_a_p1 = (np.log(oof_a_T + 1e-12) + bias_a_p1).argmax(1)
    pred_p_p1 = (np.log(oof_p_T + 1e-12) + bias_p_p1).argmax(1)
    f1a_p1 = f1_score(la, pred_a_p1, average='macro', zero_division=0)
    f1p_p1 = f1_score(lp, pred_p_p1, average='macro', zero_division=0)
    score_p1 = 0.4*f1a_p1 + 0.4*f1p_p1 + 0.2

    log(f"\n  P1 OOF (standard macro-F1):")
    log(f"    F1_a = {f1a_p1:.4f} (current {f1a_curr:.4f}, Δ={f1a_p1-f1a_curr:+.4f})")
    log(f"    F1_p = {f1p_p1:.4f} (current {f1p_curr:.4f}, Δ={f1p_p1-f1p_curr:+.4f})")
    log(f"    Score = {score_p1:.4f} (current {score_curr:.4f}, Δ={score_p1-score_curr:+.4f})")
    log(f"\n  Note: standard OOF F1 may be LOWER for P1 because it's optimized for q_test, not p_OOF.")
    log(f"  The hope is that under TEST distribution (q_test), P1 macro-F1 > current ship.")

    # ============================================================
    # Step 4: Held-out verification
    # ============================================================
    log("\n" + "─"*78)
    log("Step 4: Held-out OOF verification")
    log("─"*78)
    np.random.seed(42)
    n_oof = len(la)
    perm = np.random.permutation(n_oof)
    half = n_oof // 2
    fit_idx = perm[:half]; val_idx = perm[half:]

    # Re-fit on first half using q
    log("  Fitting P1 plug-in on FIT half...")
    bias_a_v, _ = fit_plugin_qweighted(oof_a_T[fit_idx], la[fit_idx], q_a, p_oof_a, N_ACTION, valid_a)
    bias_p_v, _ = fit_plugin_qweighted(oof_p_T[fit_idx], lp[fit_idx], q_p, p_oof_p, N_POINT, valid_p)

    # Evaluate on VAL half: weighted-F1
    log_p_a_v = np.log(oof_a_T[val_idx] + 1e-12)
    log_p_p_v = np.log(oof_p_T[val_idx] + 1e-12)
    sample_w_a_v = np.clip(q_a[la[val_idx]] / np.maximum(p_oof_a[la[val_idx]], 0.005), 0.1, 10.0)
    sample_w_a_v /= sample_w_a_v.mean()
    sample_w_p_v = np.clip(q_p[lp[val_idx]] / np.maximum(p_oof_p[lp[val_idx]], 0.005), 0.1, 10.0)
    sample_w_p_v /= sample_w_p_v.mean()

    def w_f1(log_p, b, labels, sample_w, valid):
        pred = (log_p + b).argmax(1)
        f1s = []
        for c in valid:
            tp = sample_w[(pred==c)&(labels==c)].sum()
            fp = sample_w[(pred==c)&(labels!=c)].sum()
            fn = sample_w[(pred!=c)&(labels==c)].sum()
            d = tp + 0.5*(fp+fn)
            f1s.append(tp/d if d > 0 else 0.0)
        return float(np.mean(f1s))

    wf1_a_p1_val = w_f1(log_p_a_v, bias_a_v, la[val_idx], sample_w_a_v, valid_a)
    wf1_a_curr_val = w_f1(log_p_a_v, bias_a_curr, la[val_idx], sample_w_a_v, valid_a)
    wf1_p_p1_val = w_f1(log_p_p_v, bias_p_v, lp[val_idx], sample_w_p_v, valid_p)
    wf1_p_curr_val = w_f1(log_p_p_v, bias_p_curr, lp[val_idx], sample_w_p_v, valid_p)

    log(f"\n  Held-out (VAL half) weighted macro-F1:")
    log(f"    Action: P1 bias = {wf1_a_p1_val:.4f}  vs current bias = {wf1_a_curr_val:.4f}  Δ={wf1_a_p1_val-wf1_a_curr_val:+.4f}")
    log(f"    Point:  P1 bias = {wf1_p_p1_val:.4f}  vs current bias = {wf1_p_curr_val:.4f}  Δ={wf1_p_p1_val-wf1_p_curr_val:+.4f}")
    log(f"    Avg Δ:  {((wf1_a_p1_val-wf1_a_curr_val) + (wf1_p_p1_val-wf1_p_curr_val))/2:+.4f}")

    # Standard F1 on val half (sanity)
    pred_a_v_p1 = (log_p_a_v + bias_a_v).argmax(1)
    pred_a_v_curr = (log_p_a_v + bias_a_curr).argmax(1)
    f1a_v_p1 = f1_score(la[val_idx], pred_a_v_p1, average='macro', zero_division=0)
    f1a_v_curr = f1_score(la[val_idx], pred_a_v_curr, average='macro', zero_division=0)
    log(f"\n  Held-out STANDARD macro-F1 (action):")
    log(f"    P1 bias = {f1a_v_p1:.4f}  vs current = {f1a_v_curr:.4f}  Δ={f1a_v_p1-f1a_v_curr:+.4f}")

    # ============================================================
    # Step 5: Generate test submission
    # ============================================================
    log("\n" + "─"*78)
    log("Step 5: Generate test submission")
    log("─"*78)
    test_pred_a = (np.log(test_a_T + 1e-12) + bias_a_p1).argmax(1)
    test_pred_p = (np.log(test_p_T + 1e-12) + bias_p_p1).argmax(1)
    test_sgp = test_s42['test_sgp']
    test_uids = test_s42['test_uids']

    sub_df = pd.DataFrame({
        'rally_uid': test_uids,
        'serverGetPoint': test_sgp,
        'actionId': test_pred_a,
        'pointId': test_pred_p,
    })
    sub_path = 'submissions/submission_v5z_md_p1_relabel.csv'
    sub_df.to_csv(sub_path, index=False)
    log(f"  Submission saved: {sub_path}")
    log(f"  Test action distribution: {dict(zip(*np.unique(test_pred_a, return_counts=True)))}")
    log(f"  Test point distribution:  {dict(zip(*np.unique(test_pred_p, return_counts=True)))}")

    # Compare with current ship submission
    test_pred_a_curr = (np.log(test_a_raw + 1e-12) + bias_a_curr).argmax(1)
    test_pred_p_curr = (np.log(test_p_raw + 1e-12) + bias_p_curr).argmax(1)
    diff_a = (test_pred_a != test_pred_a_curr).sum()
    diff_p = (test_pred_p != test_pred_p_curr).sum()
    log(f"\n  Test predictions diff from current ship:")
    log(f"    Action: {diff_a}/{len(test_pred_a)} ({100*diff_a/len(test_pred_a):.1f}%) different")
    log(f"    Point:  {diff_p}/{len(test_pred_p)} ({100*diff_p/len(test_pred_p):.1f}%) different")

    # Save artifacts
    np.savez('artifacts/probe_p1_relabel.npz',
             T_a=T_a, T_p=T_p, q_a=q_a, q_p=q_p,
             bias_a_p1=bias_a_p1, bias_p_p1=bias_p_p1,
             bias_a_curr=bias_a_curr, bias_p_curr=bias_p_curr,
             f1a_curr=f1a_curr, f1p_curr=f1p_curr,
             f1a_p1=f1a_p1, f1p_p1=f1p_p1,
             wf1_a_p1_val=wf1_a_p1_val, wf1_a_curr_val=wf1_a_curr_val,
             wf1_p_p1_val=wf1_p_p1_val, wf1_p_curr_val=wf1_p_curr_val)
    log(f"\n  Artifacts saved: artifacts/probe_p1_relabel.npz")

    # ============================================================
    # Decision summary
    # ============================================================
    log("\n" + "="*78)
    log("DECISION SUMMARY")
    log("="*78)
    val_lift = ((wf1_a_p1_val - wf1_a_curr_val) + (wf1_p_p1_val - wf1_p_curr_val)) / 2
    log(f"\n  Held-out weighted macro-F1 Δ: {val_lift:+.4f}")
    if val_lift > 0.005:
        log(f"  ✅ PROCEED — held-out lift {val_lift:.4f} > +0.005")
        log(f"  → Submit submissions/submission_v5z_md_p1_relabel.csv to LB")
    elif val_lift > 0.0:
        log(f"  ⚠️  MARGINAL — held-out lift {val_lift:.4f} ∈ [0, +0.005]")
        log(f"  → Borderline; could submit if LB sub budget available")
    else:
        log(f"  ❌ KILL — held-out lift {val_lift:.4f} ≤ 0")
        log(f"  → P1 mechanism not delivering; do not submit")


if __name__ == "__main__":
    main()
