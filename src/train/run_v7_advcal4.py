#!/usr/bin/env python3
"""
V7 Advanced Cal v2: 4-way ensemble search.
    ens = α·LSTM + β·V5_LGB + γ·V7P2_LGB + (1-α-β-γ)·V7P3_LGB
Scalar per task, simplex grid step=0.1 (O(10^3) per task), then plug-in bias.

Compares vs V7 (CV=0.4883) and V7 advcal 3-way (CV=0.4893).
Saves winning params + submission if it beats 3-way.
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
    train_df, build_lgb_features, train_lgb_fold,
)
from run_v7_p1 import macro_f1_fast, plugin_add


def search_4way(lstm, v5, p2, p3, labels, n_class, step=0.1):
    """Grid simplex (α, β, γ) with δ = 1-α-β-γ."""
    best = (-1.0, None)
    grid = np.arange(0.0, 1.0001, step)
    for a in grid:
        for b in grid:
            if a + b > 1.0001: continue
            for g in grid:
                if a + b + g > 1.0001: continue
                d = 1.0 - a - b - g
                ens = a * lstm + b * v5 + g * p2 + d * p3
                bias, f1 = plugin_add(ens, labels, n_class, n_rounds=3)
                if f1 > best[0]:
                    best = (f1, (float(a), float(b), float(g), float(d), bias))
    return best


def score(fa, fp):
    return 0.4 * fa + 0.4 * fp + 0.2


def main():
    log("=" * 70)
    log("V7 Advanced Calibration v2 — 4-way ensemble")
    log("=" * 70)

    v7 = np.load("artifacts/v7_oof.npz")
    p2 = np.load("artifacts/v7_p2_oof.npz")
    p3 = np.load("artifacts/v7_p3_oof.npz")
    lstm_a, lstm_p = v7["lstm_a"], v7["lstm_p"]
    v5_a, v5_p = v7["lgb_a"], v7["lgb_p"]
    la = v7["la"].astype(np.int64)
    lp = v7["lp"].astype(np.int64)

    # P2/P3 arrays are in sample order; realign to OOF order
    # Use sample_fold & CV split to derive inverse permutation
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
    oof_to_sample = np.empty_like(sample_idx_to_oof_idx)
    oof_to_sample[sample_idx_to_oof_idx] = np.arange(len(sample_idx_to_oof_idx))

    p2_a = p2["lgb_a"][oof_to_sample]
    p2_p = p2["lgb_p"][oof_to_sample]
    p3_a = p3["lgb_a"][oof_to_sample]
    p3_p = p3["lgb_p"][oof_to_sample]

    # sanity
    assert np.array_equal(p2["la"].astype(np.int64)[oof_to_sample], la)
    assert np.array_equal(p3["la"].astype(np.int64)[oof_to_sample], la)

    log(f"OOF samples: {len(la)}")

    # Raw component F1
    from sklearn.metrics import f1_score
    for name, arr in [("LSTM_bag", lstm_a), ("V5", v5_a), ("P2", p2_a), ("P3", p3_a)]:
        f = f1_score(la, arr.argmax(1), average="macro", zero_division=0)
        log(f"  raw action {name}: F1_a={f:.4f}")
    for name, arr in [("LSTM_bag", lstm_p), ("V5", v5_p), ("P2", p2_p), ("P3", p3_p)]:
        f = f1_score(lp, arr.argmax(1), average="macro", zero_division=0)
        log(f"  raw point  {name}: F1_p={f:.4f}")

    # ---- 4-way search ----
    log(f"\n{'─'*70}")
    log("【D】4-way scalar (LSTM + V5 + P2 + P3) + plug-in bias  step=0.1")
    t0 = time.time()
    fa_D, (aa, ab, ag, ad, bia_D) = search_4way(lstm_a, v5_a, p2_a, p3_a, la, N_ACTION, step=0.1)
    log(f"  action  α={aa:.2f} β={ab:.2f} γ={ag:.2f} δ={ad:.2f}  F1_a={fa_D:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    fp_D, (ap_, bp_, gp_, dp_, bip_D) = search_4way(lstm_p, v5_p, p2_p, p3_p, lp, N_POINT, step=0.1)
    log(f"  point   α={ap_:.2f} β={bp_:.2f} γ={gp_:.2f} δ={dp_:.2f}  F1_p={fp_D:.4f} ({time.time()-t0:.1f}s)")
    sc_D = score(fa_D, fp_D)
    log(f"  D CV:   {sc_D:.4f}  (vs V7 advcal 0.4893: {sc_D-0.4893:+.4f})")

    if sc_D <= 0.4893 + 1e-5:
        log(f"\n4-way did not beat 3-way. V7 advcal stays final.")
        return

    log(f"\n🏆 4-way beat 3-way. Saving params + generating submission.")
    # Load test preds (from p2_test and p3_test; both have LSTM test avg — use p3's for freshness)
    t2 = np.load("artifacts/v7_p2_test.npz")
    t3 = np.load("artifacts/v7_p3_test.npz")
    lstm_test_a = t3["lstm_a"]
    lstm_test_p = t3["lstm_p"]
    p2_test_a = t2["lgb_a"]
    p2_test_p = t2["lgb_p"]
    p3_test_a = t3["lgb_a"]
    p3_test_p = t3["lgb_p"]

    # Rebuild V5 LGB test preds (5 fold × 2 targets)
    log("  Rebuilding V5 LGB test preds (base 49 features)...")
    lgb_X_train, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(train_df, is_train=True, augment=True)
    lgb_X_test, _, _, test_uids, test_sgp = build_lgb_features(test_df, is_train=False)
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)
    kf2 = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    v5_test_a = np.zeros((len(lgb_X_test), N_ACTION))
    v5_test_p = np.zeros((len(lgb_X_test), N_POINT))
    for fold, (tr_uidx, va_uidx) in enumerate(kf2.split(uid_list, uid_arr)):
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
        log(f"    fold {fold+1}/{N_FOLDS}")

    # Apply
    ens_a = aa * lstm_test_a + ab * v5_test_a + ag * p2_test_a + ad * p3_test_a
    ens_p = ap_ * lstm_test_p + bp_ * v5_test_p + gp_ * p2_test_p + dp_ * p3_test_p
    log_a = np.log(ens_a + 1e-12) + bia_D
    log_p = np.log(ens_p + 1e-12) + bip_D
    pred_action = log_a.argmax(1); pred_point = log_p.argmax(1)

    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    }).sort_values("rally_uid").reset_index(drop=True)
    sub.to_csv("submissions/submission_v7_advcal4.csv", index=False)
    np.savez("artifacts/v7_advcal4_params.npz",
             method="4way_scalar",
             a_a=aa, b_a=ab, g_a=ag, d_a=ad, bias_a=bia_D,
             a_p=ap_, b_p=bp_, g_p=gp_, d_p=dp_, bias_p=bip_D,
             cv_score=sc_D)
    log(f"  → submissions/submission_v7_advcal4.csv")


if __name__ == "__main__":
    main()
