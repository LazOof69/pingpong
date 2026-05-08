"""
P1 v2: Use m_hat directly as q estimate (avoid Saerens EM degeneracy).

Saerens EM v1 result was degenerate (q concentrated on rare classes).
This v2 uses Alexandari 2020's simpler approach: temperature-scale model
predictions, then m_hat = mean softmax on test serves as q_test estimate
(biased by model miscalibration, but we've fixed that with temp scaling).

Also adds: clamp q to [0.5*p_OOF, 2.0*p_OOF] to prevent any single class
from dominating; this is "shrinkage on q" not "shrinkage on p_OOF".

Output:
- artifacts/probe_p1_v2.npz
- submissions/submission_v5z_md_p1_v2.csv (only if held-out STANDARD macro-F1 improves)
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
    log_p = np.log(probs + 1e-12) / T
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    p_T = np.exp(log_p); p_T = p_T / p_T.sum(axis=1, keepdims=True)
    return -np.log(p_T[np.arange(len(labels)), labels] + 1e-12).mean()


def apply_temp(probs, T):
    log_p = np.log(probs + 1e-12) / T
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    p_T = np.exp(log_p); return p_T / p_T.sum(axis=1, keepdims=True)


def fit_plugin_qweighted(probs, labels, q, p_oof, n_class, valid_classes,
                         grid_step=0.05, grid_range=2.5, n_rounds=4):
    log_p = np.log(probs + 1e-12)
    sample_w = np.clip(q[labels] / np.maximum(p_oof[labels], 0.005), 0.1, 10.0)
    sample_w = sample_w / sample_w.mean()

    def weighted_macro_f1(b):
        pred = (log_p + b).argmax(1)
        f1s = []
        for c in valid_classes:
            tp = sample_w[(pred==c)&(labels==c)].sum()
            fp = sample_w[(pred==c)&(labels!=c)].sum()
            fn = sample_w[(pred!=c)&(labels==c)].sum()
            d = tp + 0.5*(fp+fn)
            f1s.append(tp/d if d > 0 else 0.0)
        return float(np.mean(f1s))

    b = np.zeros(n_class)
    best_f1 = weighted_macro_f1(b)
    for r in range(n_rounds):
        improved = False
        for c in valid_classes:
            best_c = b[c]; best_at_c = best_f1
            for d in np.arange(-grid_range, grid_range+1e-6, grid_step):
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


def main():
    log("="*78)
    log("P1 v2: m_hat-direct q estimate (avoid Saerens EM degeneracy)")
    log("="*78)

    oof_s42 = np.load('artifacts/v5z_md_full_s42_oof.npz', allow_pickle=True)
    oof_s1337 = np.load('artifacts/v5z_md_full_s1337_oof.npz', allow_pickle=True)
    test_s42 = np.load('artifacts/v5z_md_full_s42_test.npz', allow_pickle=True)
    test_s1337 = np.load('artifacts/v5z_md_full_s1337_test.npz', allow_pickle=True)
    advcal = np.load('artifacts/v5z_md_advcal_bag2_params.npz', allow_pickle=True)
    a_alpha = float(advcal['a_alpha']); p_alpha = float(advcal['p_alpha'])
    bias_a_curr = advcal['bias_a']; bias_p_curr = advcal['bias_p']

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

    def get_can(d, s2o):
        return {k: d[k][s2o] for k in ['lstm_a','lstm_p','lgb_a','lgb_p','la','lp']}
    c42 = get_can(oof_s42, s2o_42); c13 = get_can(oof_s1337, s2o_1337)
    la = c42['la']; lp = c42['lp']

    oof_a_raw = a_alpha * (c42['lstm_a']+c13['lstm_a'])/2 + (1-a_alpha) * (c42['lgb_a']+c13['lgb_a'])/2
    oof_p_raw = p_alpha * (c42['lstm_p']+c13['lstm_p'])/2 + (1-p_alpha) * (c42['lgb_p']+c13['lgb_p'])/2
    test_a_raw = a_alpha * (test_s42['lstm_a']+test_s1337['lstm_a'])/2 + (1-a_alpha) * (test_s42['lgb_a']+test_s1337['lgb_a'])/2
    test_p_raw = p_alpha * (test_s42['lstm_p']+test_s1337['lstm_p'])/2 + (1-p_alpha) * (test_s42['lgb_p']+test_s1337['lgb_p'])/2

    p_oof_a = np.bincount(la, minlength=N_ACTION) / len(la)
    p_oof_p = np.bincount(lp, minlength=N_POINT) / len(lp)
    valid_a = list(range(15)); valid_p = list(range(N_POINT))

    # Sanity baselines
    pred_a_curr = (np.log(oof_a_raw + 1e-12) + bias_a_curr).argmax(1)
    pred_p_curr = (np.log(oof_p_raw + 1e-12) + bias_p_curr).argmax(1)
    f1a_curr = f1_score(la, pred_a_curr, average='macro', zero_division=0)
    f1p_curr = f1_score(lp, pred_p_curr, average='macro', zero_division=0)
    log(f"Current ship OOF F1: action={f1a_curr:.4f}, point={f1p_curr:.4f}")

    # Step 1: Temperature scaling
    res_a = minimize_scalar(temp_scale_nll, args=(oof_a_raw, la), bounds=(0.1, 10.0), method='bounded')
    T_a = res_a.x
    res_p = minimize_scalar(temp_scale_nll, args=(oof_p_raw, lp), bounds=(0.1, 10.0), method='bounded')
    T_p = res_p.x
    log(f"\nTemperature scaling: T_a={T_a:.3f}, T_p={T_p:.3f}")

    oof_a_T = apply_temp(oof_a_raw, T_a); oof_p_T = apply_temp(oof_p_raw, T_p)
    test_a_T = apply_temp(test_a_raw, T_a); test_p_T = apply_temp(test_p_raw, T_p)

    # Step 2: q = m_hat (direct, with mask + clamp)
    log("\n" + "─"*78)
    log("Step 2: q estimate via m_hat (mean test softmax) with clamping")
    log("─"*78)

    def m_hat_q(P_test, p_train, valid_classes, clamp_lo=0.5, clamp_hi=2.0):
        K = P_test.shape[1]
        mask = np.zeros(K, dtype=bool); mask[valid_classes] = True
        # Renormalize test probs over valid classes only
        P = P_test.copy(); P[:, ~mask] = 0
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        m_hat = P.mean(axis=0)
        # Renormalize p_train over valid classes
        p_t = p_train.copy(); p_t[~mask] = 0
        p_t = p_t / max(p_t.sum(), 1e-12)
        # Clamp q to [clamp_lo * p_t, clamp_hi * p_t] per class (Alexandari-style shrinkage)
        q = np.clip(m_hat, clamp_lo * p_t, clamp_hi * p_t)
        q[~mask] = 0
        q = q / q.sum()
        return q, m_hat

    q_a, m_hat_a = m_hat_q(test_a_T, p_oof_a, valid_a)
    q_p, m_hat_p = m_hat_q(test_p_T, p_oof_p, valid_p)

    log(f"\n  Action q (clamped to [0.5×, 2.0×] of p_OOF):")
    log(f"  {'cls':<5}{'p_OOF':<10}{'m_hat':<10}{'q_clamp':<10}{'q-p':<10}{'q/p':<10}")
    for c in valid_a:
        log(f"  {c:<5}{p_oof_a[c]:<10.4f}{m_hat_a[c]:<10.4f}{q_a[c]:<10.4f}{q_a[c]-p_oof_a[c]:<+10.4f}{q_a[c]/max(p_oof_a[c],1e-6):<10.3f}")
    log(f"\n  Point q:")
    for c in valid_p:
        log(f"  {c:<5}{p_oof_p[c]:<10.4f}{m_hat_p[c]:<10.4f}{q_p[c]:<10.4f}{q_p[c]-p_oof_p[c]:<+10.4f}{q_p[c]/max(p_oof_p[c],1e-6):<10.3f}")

    # Step 3: Re-fit plug-in
    log("\n" + "─"*78)
    log("Step 3: Re-fit plug-in with q-weighted macro-F1")
    log("─"*78)
    t0 = time.time()
    bias_a_p1, wf1_a = fit_plugin_qweighted(oof_a_T, la, q_a, p_oof_a, N_ACTION, valid_a)
    log(f"  Action: weighted F1 = {wf1_a:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    bias_p_p1, wf1_p = fit_plugin_qweighted(oof_p_T, lp, q_p, p_oof_p, N_POINT, valid_p)
    log(f"  Point: weighted F1 = {wf1_p:.4f} ({time.time()-t0:.1f}s)")

    # Step 4: STANDARD macro-F1 sanity (the LB metric)
    pred_a_p1 = (np.log(oof_a_T + 1e-12) + bias_a_p1).argmax(1)
    pred_p_p1 = (np.log(oof_p_T + 1e-12) + bias_p_p1).argmax(1)
    f1a_p1 = f1_score(la, pred_a_p1, average='macro', zero_division=0)
    f1p_p1 = f1_score(lp, pred_p_p1, average='macro', zero_division=0)
    score_p1 = 0.4*f1a_p1 + 0.4*f1p_p1 + 0.2
    score_curr = 0.4*f1a_curr + 0.4*f1p_curr + 0.2
    log(f"\n  P1 v2 OOF (standard macro-F1, the LB metric):")
    log(f"    F1_a = {f1a_p1:.4f} (current {f1a_curr:.4f}, Δ={f1a_p1-f1a_curr:+.4f})")
    log(f"    F1_p = {f1p_p1:.4f} (current {f1p_curr:.4f}, Δ={f1p_p1-f1p_curr:+.4f})")
    log(f"    Score = {score_p1:.4f} (current {score_curr:.4f}, Δ={score_p1-score_curr:+.4f})")

    # Step 5: Held-out verification
    log("\n" + "─"*78)
    log("Step 5: Held-out verification (STANDARD macro-F1)")
    log("─"*78)
    np.random.seed(42)
    perm = np.random.permutation(len(la))
    half = len(la) // 2
    fit_idx, val_idx = perm[:half], perm[half:]
    bias_a_v, _ = fit_plugin_qweighted(oof_a_T[fit_idx], la[fit_idx], q_a, p_oof_a, N_ACTION, valid_a)
    bias_p_v, _ = fit_plugin_qweighted(oof_p_T[fit_idx], lp[fit_idx], q_p, p_oof_p, N_POINT, valid_p)

    pred_a_v_p1 = (np.log(oof_a_T[val_idx] + 1e-12) + bias_a_v).argmax(1)
    pred_a_v_curr = (np.log(oof_a_T[val_idx] + 1e-12) + bias_a_curr).argmax(1)
    pred_p_v_p1 = (np.log(oof_p_T[val_idx] + 1e-12) + bias_p_v).argmax(1)
    pred_p_v_curr = (np.log(oof_p_T[val_idx] + 1e-12) + bias_p_curr).argmax(1)

    f1a_v_p1 = f1_score(la[val_idx], pred_a_v_p1, average='macro', zero_division=0)
    f1a_v_curr = f1_score(la[val_idx], pred_a_v_curr, average='macro', zero_division=0)
    f1p_v_p1 = f1_score(lp[val_idx], pred_p_v_p1, average='macro', zero_division=0)
    f1p_v_curr = f1_score(lp[val_idx], pred_p_v_curr, average='macro', zero_division=0)

    log(f"\n  Held-out STANDARD macro-F1:")
    log(f"    Action: P1 v2 = {f1a_v_p1:.4f}, current = {f1a_v_curr:.4f}, Δ={f1a_v_p1-f1a_v_curr:+.4f}")
    log(f"    Point:  P1 v2 = {f1p_v_p1:.4f}, current = {f1p_v_curr:.4f}, Δ={f1p_v_p1-f1p_v_curr:+.4f}")
    avg_delta = ((f1a_v_p1-f1a_v_curr) + (f1p_v_p1-f1p_v_curr)) / 2
    log(f"    Avg Δ:  {avg_delta:+.4f}")

    # Step 6: Submission (only if held-out STANDARD F1 improves OR ties)
    log("\n" + "─"*78)
    log("Step 6: Decision")
    log("─"*78)
    if avg_delta >= 0:
        log(f"\n  ✅ PROCEED — held-out standard F1 lift {avg_delta:+.4f}")
        test_pred_a = (np.log(test_a_T + 1e-12) + bias_a_p1).argmax(1)
        test_pred_p = (np.log(test_p_T + 1e-12) + bias_p_p1).argmax(1)
        sub_df = pd.DataFrame({
            'rally_uid': test_s42['test_uids'],
            'serverGetPoint': test_s42['test_sgp'],
            'actionId': test_pred_a,
            'pointId': test_pred_p,
        })
        sub_path = 'submissions/submission_v5z_md_p1_v2.csv'
        sub_df.to_csv(sub_path, index=False)
        log(f"  Submission saved: {sub_path}")
        log(f"  Test action dist: {dict(zip(*np.unique(test_pred_a, return_counts=True)))}")
        log(f"  Test point dist:  {dict(zip(*np.unique(test_pred_p, return_counts=True)))}")
        # Diff vs current ship
        test_pred_a_curr = (np.log(test_a_raw + 1e-12) + bias_a_curr).argmax(1)
        test_pred_p_curr = (np.log(test_p_raw + 1e-12) + bias_p_curr).argmax(1)
        log(f"  Diff vs current ship: action {(test_pred_a!=test_pred_a_curr).sum()}/1236, point {(test_pred_p!=test_pred_p_curr).sum()}/1236")
    else:
        log(f"\n  ❌ KILL — held-out standard F1 regressed {avg_delta:+.4f}")
        log(f"  → P1 v2 fails LB-metric test; do not submit")
        log(f"  → m_hat as q estimate also doesn't deliver lift on this dataset")

    np.savez('artifacts/probe_p1_v2.npz',
             T_a=T_a, T_p=T_p, q_a=q_a, q_p=q_p,
             m_hat_a=m_hat_a, m_hat_p=m_hat_p,
             bias_a_p1=bias_a_p1, bias_p_p1=bias_p_p1,
             f1a_v_p1=f1a_v_p1, f1a_v_curr=f1a_v_curr,
             f1p_v_p1=f1p_v_p1, f1p_v_curr=f1p_v_curr,
             avg_delta=avg_delta)


if __name__ == "__main__":
    main()
