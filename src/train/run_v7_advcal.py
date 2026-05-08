#!/usr/bin/env python3
"""
V7 Advanced Calibration
=======================
In V7 we used: ens = w*LSTM + (1-w)*LGB,  then argmax(log(ens) + bias_c)
                      └─ scalar w per task                └─ per-class bias

This script tests two extensions:
  A) **Per-class ensemble weight** w_c per class:
         ens_c = w_c * LSTM_c + (1 - w_c) * LGB_c
     Followed by plug-in additive bias per class.
     Same OOF-safe structure; just more expressive (19 params for action, 10 for point).
  B) **3-way ensemble** with V7 P2 LGB as third member:
         ens = α*LSTM + β*V5_LGB + (1 - α - β)*V7P2_LGB
     Scalar α, β per task; then plug-in bias.
  C) **Both**: per-class α_c, β_c + plug-in bias.

Writes:
  artifacts/v7_advcal_params.npz    (best scheme's params)
  submissions/submission_v7_advcal.csv
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src/train")
from train_v5 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    test_df, prepare_samples,
    RallyDataset, PingPongModel, log,
)
from run_v7_p1 import macro_f1_fast, plugin_add


SEEDS = [42, 1337, 2024]
LSTM_PATHS = {
    42:   "models/v5/model_v5_lstm_fold{fold}.pt",
    1337: "models/v7/model_v7_seed1337_fold{fold}.pt",
    2024: "models/v7/model_v7_seed2024_fold{fold}.pt",
}


# ============================================================
# Calibration primitives
# ============================================================
def plugin_per_class_w(lstm, lgb, labels, n_class, n_rounds=5, step=0.05):
    """Per-class w_c ensemble + subsequent plug-in additive bias."""
    w_grid = np.arange(0.0, 1.0001, step)
    w = np.full(n_class, 0.5, dtype=np.float64)
    ens = w[None, :] * lstm + (1 - w[None, :]) * lgb
    best_f1 = macro_f1_fast(ens.argmax(1), labels, n_class)

    for rnd in range(n_rounds):
        improved = False
        for c in range(n_class):
            col_l, col_g = lstm[:, c], lgb[:, c]
            best_w_c, best_local = w[c], best_f1
            for w_try in w_grid:
                ens[:, c] = w_try * col_l + (1 - w_try) * col_g
                f1 = macro_f1_fast(ens.argmax(1), labels, n_class)
                if f1 > best_local + 1e-6:
                    best_local, best_w_c = f1, w_try
            w[c] = best_w_c
            ens[:, c] = best_w_c * col_l + (1 - best_w_c) * col_g
            if best_local > best_f1 + 1e-6:
                best_f1 = best_local
                improved = True
        if not improved:
            break
    return w, ens, best_f1


def search_3way(lstm, v5, v7p2, labels, n_class, step=0.1):
    """Scalar α*LSTM + β*V5 + (1-α-β)*V7P2, grid search simplex."""
    best = (-1.0, 0.0, 0.0, None)
    for a in np.arange(0.0, 1.0001, step):
        for b in np.arange(0.0, 1.0001 - a, step):
            c = 1.0 - a - b
            ens = a * lstm + b * v5 + c * v7p2
            bias, f1 = plugin_add(ens, labels, n_class, n_rounds=3)
            if f1 > best[0]:
                best = (f1, float(a), float(b), bias)
    return best


def plugin_per_class_3way(lstm, v5, v7p2, labels, n_class, n_rounds=5, step=0.1):
    """Per-class α_c, β_c 3-way + plug-in bias."""
    grid = np.arange(0.0, 1.0001, step)
    alpha = np.full(n_class, 0.4, dtype=np.float64)
    beta = np.full(n_class, 0.3, dtype=np.float64)
    ens = alpha[None, :] * lstm + beta[None, :] * v5 + (1 - alpha[None, :] - beta[None, :]) * v7p2
    best_f1 = macro_f1_fast(ens.argmax(1), labels, n_class)

    for rnd in range(n_rounds):
        improved = False
        for c in range(n_class):
            best_a, best_b, best_local = alpha[c], beta[c], best_f1
            l, v, p = lstm[:, c], v5[:, c], v7p2[:, c]
            for a_try in grid:
                for b_try in grid:
                    if a_try + b_try > 1.0001:
                        continue
                    c_try = 1.0 - a_try - b_try
                    ens[:, c] = a_try * l + b_try * v + c_try * p
                    f1 = macro_f1_fast(ens.argmax(1), labels, n_class)
                    if f1 > best_local + 1e-6:
                        best_local, best_a, best_b = f1, a_try, b_try
            alpha[c], beta[c] = best_a, best_b
            ens[:, c] = best_a * l + best_b * v + (1 - best_a - best_b) * p
            if best_local > best_f1 + 1e-6:
                best_f1 = best_local
                improved = True
        if not improved:
            break
    return alpha, beta, ens, best_f1


def score(fa, fp):
    return 0.4 * fa + 0.4 * fp + 0.2


# ============================================================
def load_test_lstm_preds(test_samples):
    n_test = len(test_samples)
    lstm_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lstm_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)
    test_dl = DataLoader(RallyDataset(test_samples, is_train=False), BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)
    n_models = len(SEEDS) * N_FOLDS
    log("  inferring LSTM on test (15 models)...")
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            path = LSTM_PATHS[seed].format(fold=fold + 1)
            model = PingPongModel(player_drop_p=0).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            offset = 0
            with torch.no_grad():
                for batch in test_dl:
                    sc = batch["seq_cat"].to(DEVICE)
                    sn = batch["seq_num"].to(DEVICE)
                    ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                    st = batch["static"].to(DEVICE)
                    lens = batch["length"]
                    nsns = batch["next_sn"]
                    a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                         next_sns=nsns, apply_mask=True)
                    bs = sc.size(0)
                    lstm_test_a[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / n_models
                    lstm_test_p[offset:offset+bs] += F.softmax(p_log, dim=-1).cpu().numpy() / n_models
                    offset += bs
    return lstm_test_a, lstm_test_p


def main():
    log("=" * 70)
    log("V7 Advanced Calibration")
    log("=" * 70)

    # ---- load OOFs ----
    v7 = np.load("artifacts/v7_oof.npz")
    p2 = np.load("artifacts/v7_p2_oof.npz")
    lstm_a, lstm_p = v7["lstm_a"], v7["lstm_p"]          # OOF order
    v5_a, v5_p = v7["lgb_a"], v7["lgb_p"]                # OOF order (V5 LGB)
    v7p2_a_smp, v7p2_p_smp = p2["lgb_a"], p2["lgb_p"]    # sample order
    la = v7["la"].astype(np.int64)
    lp = v7["lp"].astype(np.int64)

    # p2 arrays are in sample order; need to realign to OOF order to compare with lstm_a/v5_a
    # p2 also includes la/lp; verify alignment by label recovery
    la_p2 = p2["la"].astype(np.int64)  # sample order
    lp_p2 = p2["lp"].astype(np.int64)
    sample_fold = p2["sample_fold"]

    # Rebuild sample→OOF permutation
    # For this we need prepare_samples-order labels too — rebuild from train_df
    # Shortcut: find a permutation π such that la_p2[π] == la (element-wise)
    # But la may have ties; safer to reconstruct via the CV split.
    from train_v5 import train_df
    log("Rebuilding sample→OOF permutation for P2 → OOF reindex...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    sample_idx_to_oof_idx = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for fold, (_, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_idx_to_oof_idx[i] = cur
                cur += 1

    # sample_idx_to_oof_idx[i] = OOF position of sample i
    # We want OOF-order array for v7p2: out[oof_pos] = v7p2[i where sample_idx_to_oof_idx[i] == oof_pos]
    # Inverse: oof_to_sample = argsort(sample_idx_to_oof_idx) — but it's a permutation, so inverse is straightforward:
    oof_to_sample = np.empty_like(sample_idx_to_oof_idx)
    oof_to_sample[sample_idx_to_oof_idx] = np.arange(len(sample_idx_to_oof_idx))
    v7p2_a = v7p2_a_smp[oof_to_sample]  # now OOF order
    v7p2_p = v7p2_p_smp[oof_to_sample]

    # sanity: labels should match
    assert np.array_equal(la_p2[oof_to_sample], la), "la misaligned"
    assert np.array_equal(lp_p2[oof_to_sample], lp), "lp misaligned"
    log(f"  OOF samples: {len(la)}  (LSTM, V5 LGB, V7 P2 LGB, all aligned)")

    # Base V7 score
    from run_v7_p1 import plugin_add as pa
    log("\n【Baseline V7】scalar w + plug-in bias (2-way LSTM+V5LGB)")
    def search_w(lstm, lgb, labels, n_class, step=0.05, n_rounds=5):
        best = (-1.0, 0.0, None)
        for w in np.arange(0.0, 1.0001, step):
            ens = w * lstm + (1 - w) * lgb
            bias, f1 = pa(ens, labels, n_class, n_rounds=n_rounds)
            if f1 > best[0]:
                best = (f1, float(w), bias)
        return best
    t0 = time.time()
    fa_v7, wa_v7, bia_v7 = search_w(lstm_a, v5_a, la, N_ACTION)
    fp_v7, wp_v7, bip_v7 = search_w(lstm_p, v5_p, lp, N_POINT)
    log(f"  action  w_a={wa_v7:.2f} F1_a={fa_v7:.4f}")
    log(f"  point   w_p={wp_v7:.2f} F1_p={fp_v7:.4f}")
    log(f"  V7 CV:  {score(fa_v7, fp_v7):.4f}  ({time.time()-t0:.1f}s)")

    # A: Per-class w (2-way LSTM + V5 LGB)
    log(f"\n{'─'*70}")
    log("【A】Per-class w_c (2-way LSTM + V5 LGB) + plug-in bias")
    t0 = time.time()
    wa_pc, ens_a, _ = plugin_per_class_w(lstm_a, v5_a, la, N_ACTION)
    bia_pc, fa_A = plugin_add(ens_a, la, N_ACTION, n_rounds=5)
    log(f"  action  F1_a={fa_A:.4f}  w_a range=[{wa_pc.min():.2f},{wa_pc.max():.2f}] ({time.time()-t0:.1f}s)")
    t0 = time.time()
    wp_pc, ens_p, _ = plugin_per_class_w(lstm_p, v5_p, lp, N_POINT)
    bip_pc, fp_A = plugin_add(ens_p, lp, N_POINT, n_rounds=5)
    log(f"  point   F1_p={fp_A:.4f}  w_p range=[{wp_pc.min():.2f},{wp_pc.max():.2f}] ({time.time()-t0:.1f}s)")
    log(f"  A CV:   {score(fa_A, fp_A):.4f}  (Δ vs V7={score(fa_A, fp_A)-score(fa_v7, fp_v7):+.4f})")

    # B: 3-way scalar
    log(f"\n{'─'*70}")
    log("【B】3-way scalar α*LSTM + β*V5 + (1-α-β)*V7P2 + plug-in bias")
    t0 = time.time()
    fa_B, aa_B, ab_B, bia_B = search_3way(lstm_a, v5_a, v7p2_a, la, N_ACTION, step=0.1)
    log(f"  action  α={aa_B:.2f} β={ab_B:.2f} γ={1-aa_B-ab_B:.2f}  F1_a={fa_B:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    fp_B, ap_B, bp_B, bip_B = search_3way(lstm_p, v5_p, v7p2_p, lp, N_POINT, step=0.1)
    log(f"  point   α={ap_B:.2f} β={bp_B:.2f} γ={1-ap_B-bp_B:.2f}  F1_p={fp_B:.4f} ({time.time()-t0:.1f}s)")
    log(f"  B CV:   {score(fa_B, fp_B):.4f}  (Δ vs V7={score(fa_B, fp_B)-score(fa_v7, fp_v7):+.4f})")

    # C: per-class 3-way
    log(f"\n{'─'*70}")
    log("【C】Per-class α_c, β_c 3-way + plug-in bias  (heavy)")
    t0 = time.time()
    alpha_a, beta_a, ens_a3, _ = plugin_per_class_3way(lstm_a, v5_a, v7p2_a, la, N_ACTION, step=0.1)
    bia_C, fa_C = plugin_add(ens_a3, la, N_ACTION, n_rounds=5)
    log(f"  action  F1_a={fa_C:.4f}  α∈[{alpha_a.min():.2f},{alpha_a.max():.2f}] β∈[{beta_a.min():.2f},{beta_a.max():.2f}] ({time.time()-t0:.1f}s)")
    t0 = time.time()
    alpha_p, beta_p, ens_p3, _ = plugin_per_class_3way(lstm_p, v5_p, v7p2_p, lp, N_POINT, step=0.1)
    bip_C, fp_C = plugin_add(ens_p3, lp, N_POINT, n_rounds=5)
    log(f"  point   F1_p={fp_C:.4f}  α∈[{alpha_p.min():.2f},{alpha_p.max():.2f}] β∈[{beta_p.min():.2f},{beta_p.max():.2f}] ({time.time()-t0:.1f}s)")
    log(f"  C CV:   {score(fa_C, fp_C):.4f}  (Δ vs V7={score(fa_C, fp_C)-score(fa_v7, fp_v7):+.4f})")

    # ---- pick winner + save/submit ----
    results = {
        "V7_baseline":     (fa_v7, fp_v7, "baseline"),
        "A_perclass2":     (fa_A, fp_A, "A"),
        "B_3way_scalar":   (fa_B, fp_B, "B"),
        "C_perclass3":     (fa_C, fp_C, "C"),
    }
    log(f"\n{'='*70}\n【Summary】")
    log(f"{'variant':<20} {'F1_a':>8} {'F1_p':>8} {'Score':>8}")
    log("─" * 70)
    best_name, best_score, best_a, best_p = None, -1.0, None, None
    for name, (fa, fp, _) in results.items():
        sc = score(fa, fp)
        marker = ""
        if sc > best_score:
            best_score, best_name = sc, name
            best_a, best_p = fa, fp
            marker = " ←"
        log(f"{name:<20} {fa:>8.4f} {fp:>8.4f} {sc:>8.4f}{marker}")
    log(f"\n🏆 Best: {best_name}  CV={best_score:.4f}  (Δ vs V7={best_score-score(fa_v7, fp_v7):+.4f})")

    # Save params for best scheme
    log("\nSaving params + generating submission for best scheme...")
    params = {"method": best_name}
    if best_name == "V7_baseline":
        params.update(dict(kind="w+bias_2way",
                           w_a=np.float64(wa_v7), w_p=np.float64(wp_v7),
                           bias_a=bia_v7, bias_p=bip_v7))
    elif best_name == "A_perclass2":
        params.update(dict(kind="perclass_w_2way",
                           w_a=wa_pc, w_p=wp_pc,
                           bias_a=bia_pc, bias_p=bip_pc))
    elif best_name == "B_3way_scalar":
        params.update(dict(kind="scalar_3way",
                           a_a=np.float64(aa_B), b_a=np.float64(ab_B),
                           a_p=np.float64(ap_B), b_p=np.float64(bp_B),
                           bias_a=bia_B, bias_p=bip_B))
    elif best_name == "C_perclass3":
        params.update(dict(kind="perclass_3way",
                           alpha_a=alpha_a, beta_a=beta_a,
                           alpha_p=alpha_p, beta_p=beta_p,
                           bias_a=bia_C, bias_p=bip_C))
    np.savez("artifacts/v7_advcal_params.npz", **params)

    # ---- Apply best scheme to test ----
    # Need LSTM test preds, V5 LGB test preds, V7 P2 LGB test preds.
    # Load existing v7_p2_test.npz (has lstm + P2 LGB + V5 LGB):
    test_art = np.load("artifacts/v7_p2_test.npz")
    lstm_test_a = test_art["lstm_a"]  # this IS the 15-model avg (same)
    lstm_test_p = test_art["lstm_p"]
    v7p2_test_a = test_art["lgb_a"]
    v7p2_test_p = test_art["lgb_p"]
    # V5 LGB test preds need to be regenerated (v7_p2_test.lgb is V7P2's LGB).
    # Faster: retrain V5 LGB is ~5 min. Or: load submission_v7.csv won't give probs.
    # Best: quickly reproduce V5 LGB test via re-running fold LGB on base 49 features.
    # Cheap shortcut: predict_v7.py outputs the V5 LGB test in its run, but we didn't save.
    # Re-run LGB folds with base features (same as predict_v7.py did).
    log("  Retraining V5 LGB folds (base 49 features) for test preds...")
    from train_v5 import train_lgb_fold, build_lgb_features, train_df as _td
    t0 = time.time()
    lgb_X_train, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(_td, is_train=True, augment=True)
    lgb_X_test, _, _, test_uids, test_sgp = build_lgb_features(test_df, is_train=False)
    log(f"    features built ({time.time()-t0:.1f}s)")
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)
    kf2 = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    v5_test_a = np.zeros((len(lgb_X_test), N_ACTION))
    v5_test_p = np.zeros((len(lgb_X_test), N_POINT))
    kf2_splits = list(kf2.split(uid_list, uid_arr))
    for fold, (tr_uidx, va_uidx) in enumerate(kf2_splits):
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        lgb_tr = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        Xt, Xv = lgb_X_train.iloc[lgb_tr], lgb_X_train.iloc[lgb_va]
        ya_t = [lgb_ya[i] for i in lgb_tr]; ya_v = [lgb_ya[i] for i in lgb_va]
        yp_t = [lgb_yp[i] for i in lgb_tr]; yp_v = [lgb_yp[i] for i in lgb_va]
        ma, _, _ = train_lgb_fold(Xt, ya_t, Xv, ya_v, N_ACTION, "actionId")
        mp, _, _ = train_lgb_fold(Xt, yp_t, Xv, yp_v, N_POINT, "pointId")
        v5_test_a += ma.predict(lgb_X_test) / N_FOLDS
        v5_test_p += mp.predict(lgb_X_test) / N_FOLDS
        log(f"    fold {fold+1}/{N_FOLDS} done")

    # Apply best scheme
    def apply_scheme(lstm, v5, v7p2):
        if best_name == "V7_baseline":
            ens = params["w_a"] * lstm + (1 - params["w_a"]) * v5 \
                if lstm.shape[1] == N_ACTION else params["w_p"] * lstm + (1 - params["w_p"]) * v5
            return ens
        if best_name == "A_perclass2":
            w = params["w_a"] if lstm.shape[1] == N_ACTION else params["w_p"]
            return w[None, :] * lstm + (1 - w[None, :]) * v5
        if best_name == "B_3way_scalar":
            if lstm.shape[1] == N_ACTION:
                return params["a_a"] * lstm + params["b_a"] * v5 + (1 - params["a_a"] - params["b_a"]) * v7p2
            else:
                return params["a_p"] * lstm + params["b_p"] * v5 + (1 - params["a_p"] - params["b_p"]) * v7p2
        if best_name == "C_perclass3":
            if lstm.shape[1] == N_ACTION:
                a, b = params["alpha_a"], params["beta_a"]
            else:
                a, b = params["alpha_p"], params["beta_p"]
            return a[None, :] * lstm + b[None, :] * v5 + (1 - a[None, :] - b[None, :]) * v7p2
        raise ValueError(best_name)

    ens_a_test = apply_scheme(lstm_test_a, v5_test_a, v7p2_test_a)
    ens_p_test = apply_scheme(lstm_test_p, v5_test_p, v7p2_test_p)
    log_a = np.log(ens_a_test + 1e-12) + params["bias_a"]
    log_p = np.log(ens_p_test + 1e-12) + params["bias_p"]
    pred_action = log_a.argmax(1)
    pred_point = log_p.argmax(1)

    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    })
    sub = sub.sort_values("rally_uid").reset_index(drop=True)
    sub.to_csv("submissions/submission_v7_advcal.csv", index=False)
    log(f"  → submissions/submission_v7_advcal.csv ({len(sub)} 筆)")


if __name__ == "__main__":
    main()
