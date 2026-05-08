#!/usr/bin/env python3
"""
V7 P3: TS-lens LGB feature expansion
====================================
Focused feature set (vs P2's broader set that regressed):
  1. **Lag features** (5): prev_prev_actionId, prev_prev_pointId, prev_prev_handId,
     prev_prev_spinId, prev3_actionId. Plus prev3_action_group.
  2. **Transition TE** on key=last_actionId (19 cells, ~3670 samples/cell):
     - P(next_action | last_action) — 19 cols
     - P(next_point  | last_action) — 10 cols
     - α=30 Bayesian smoothing, 5-fold OOF.
Total new features: 6 + 19 + 10 = 35 (vs base 49).

Outputs v7_p3_oof.npz, v7_p3_test.npz for later ensemble.
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
from sklearn.metrics import f1_score

sys.path.insert(0, "src/train")
from train_v5 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    ACTION_GROUPS,
    train_df, test_df, prepare_samples, build_lgb_features,
    RallyDataset, PingPongModel, train_lgb_fold, log,
)

TE_ALPHA = 30.0


def lag_features(rows_ctx):
    """Return dict of lag-2 and lag-3 features from ctx rows."""
    k = len(rows_ctx)
    # lag-2 (prev_prev): rows_ctx[-2] is lag-1, [-3] is lag-2
    if k >= 3:
        pp = rows_ctx[-3]
        pp_action = pp["actionId"]
        pp_point = pp["pointId"]
        pp_hand = pp["handId"]
        pp_spin = pp["spinId"]
    else:
        pp_action = -1; pp_point = -1; pp_hand = -1; pp_spin = -1
    # lag-3 (prev_prev_prev): rows_ctx[-4]
    if k >= 4:
        p3 = rows_ctx[-4]
        p3_action = p3["actionId"]
        p3_ag = ACTION_GROUPS.get(p3["actionId"], 0)
    else:
        p3_action = -1; p3_ag = -1
    return {
        "lag2_actionId": pp_action,
        "lag2_pointId": pp_point,
        "lag2_handId": pp_hand,
        "lag2_spinId": pp_spin,
        "lag3_actionId": p3_action,
        "lag3_action_group": p3_ag,
    }


def build_lag_df(df, is_train=True, augment=True):
    feats = []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        if is_train:
            start_k = 1 if augment else max(1, N - 1)
            for k in range(start_k, N):
                feats.append(lag_features(rows[:k]))
        else:
            feats.append(lag_features(rows))
    return pd.DataFrame(feats)


def compute_te_train(keys, y, n_class, sample_fold, n_folds, alpha=TE_ALPHA):
    y = np.asarray(y, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.int64)
    sample_fold = np.asarray(sample_fold, dtype=np.int64)
    N = len(y)
    te = np.zeros((N, n_class), dtype=np.float32)
    global_p = np.bincount(y, minlength=n_class).astype(np.float64) / N
    for f in range(n_folds):
        tr_mask = sample_fold != f
        va_mask = sample_fold == f
        sums, counts = {}, {}
        for k, yi in zip(keys[tr_mask], y[tr_mask]):
            if k not in sums:
                sums[k] = np.zeros(n_class, dtype=np.float64)
                counts[k] = 0
            sums[k][yi] += 1
            counts[k] += 1
        for idx in np.where(va_mask)[0]:
            k = int(keys[idx])
            n = counts.get(k, 0)
            te[idx] = global_p if n == 0 else (sums[k] + alpha * global_p) / (n + alpha)
    return te


def compute_te_test(keys_test, keys_train, y_train, n_class, alpha=TE_ALPHA):
    y_train = np.asarray(y_train, dtype=np.int64)
    keys_train = np.asarray(keys_train, dtype=np.int64)
    keys_test = np.asarray(keys_test, dtype=np.int64)
    global_p = np.bincount(y_train, minlength=n_class).astype(np.float64) / len(y_train)
    sums, counts = {}, {}
    for k, yi in zip(keys_train, y_train):
        if k not in sums:
            sums[k] = np.zeros(n_class, dtype=np.float64)
            counts[k] = 0
        sums[k][yi] += 1
        counts[k] += 1
    N = len(keys_test)
    te = np.zeros((N, n_class), dtype=np.float32)
    for i in range(N):
        k = int(keys_test[i])
        n = counts.get(k, 0)
        te[i] = global_p if n == 0 else (sums[k] + alpha * global_p) / (n + alpha)
    return te


def main():
    t_all = time.time()
    log("=" * 70)
    log("V7 P3 · TS-lens LGB (lag + transition TE)")
    log("=" * 70)

    # ---- samples + CV split ----
    log("\n[1] prepare_samples ...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"    train={len(train_samples)} test={len(test_samples)} ({time.time()-t0:.1f}s)")

    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    sample_fold = np.empty(len(train_samples), dtype=np.int64)
    for fold, (_, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_fold[i] = fold

    # ---- base LGB features ----
    log("\n[2] base LGB features ...")
    t0 = time.time()
    lgb_base_train, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_base_test, _, _, test_uids, test_sgp = build_lgb_features(
        test_df, is_train=False)
    log(f"    base: train={lgb_base_train.shape} test={lgb_base_test.shape} ({time.time()-t0:.1f}s)")

    # ---- lag features ----
    log("\n[3] lag features ...")
    t0 = time.time()
    lag_train = build_lag_df(train_df, is_train=True, augment=True)
    lag_test = build_lag_df(test_df, is_train=False)
    log(f"    lag: train={lag_train.shape} test={lag_test.shape} ({time.time()-t0:.1f}s)")

    # ---- transition TE on last_action ----
    log("\n[4] transition TE (last_action → next_action / next_point) ...")
    t0 = time.time()
    key_train = lgb_base_train["last_actionId"].values.astype(np.int64)
    key_test = lgb_base_test["last_actionId"].values.astype(np.int64)
    y_a = np.array(lgb_ya, dtype=np.int64)
    y_p = np.array(lgb_yp, dtype=np.int64)
    trans_a_tr = compute_te_train(key_train, y_a, N_ACTION, sample_fold, N_FOLDS)
    trans_p_tr = compute_te_train(key_train, y_p, N_POINT, sample_fold, N_FOLDS)
    trans_a_te = compute_te_test(key_test, key_train, y_a, N_ACTION)
    trans_p_te = compute_te_test(key_test, key_train, y_p, N_POINT)
    log(f"    TE shape: action={trans_a_tr.shape} point={trans_p_tr.shape} ({time.time()-t0:.1f}s)")

    # ---- assemble ----
    def _pd(arr, prefix, n):
        return pd.DataFrame(arr, columns=[f"{prefix}_{c}" for c in range(n)])
    tr_cols = [lgb_base_train.reset_index(drop=True),
               lag_train.reset_index(drop=True),
               _pd(trans_a_tr, "trans_a", N_ACTION),
               _pd(trans_p_tr, "trans_p", N_POINT)]
    te_cols = [lgb_base_test.reset_index(drop=True),
               lag_test.reset_index(drop=True),
               _pd(trans_a_te, "trans_a", N_ACTION),
               _pd(trans_p_te, "trans_p", N_POINT)]
    X_train = pd.concat(tr_cols, axis=1)
    X_test = pd.concat(te_cols, axis=1)
    log(f"    enhanced: train={X_train.shape} test={X_test.shape}")

    # ---- LSTM test preds (reused 15 models) ----
    log("\n[5] LSTM test preds (15 models) ...")
    n_test = len(test_samples)
    lstm_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lstm_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)
    test_dl = DataLoader(RallyDataset(test_samples, is_train=False), BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)
    SEEDS = [42, 1337, 2024]
    paths = {
        42: "models/v5/model_v5_lstm_fold{fold}.pt",
        1337: "models/v7/model_v7_seed1337_fold{fold}.pt",
        2024: "models/v7/model_v7_seed2024_fold{fold}.pt",
    }
    n_models = len(SEEDS) * N_FOLDS
    t0 = time.time()
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            p = paths[seed].format(fold=fold + 1)
            model = PingPongModel(player_drop_p=0).to(DEVICE)
            model.load_state_dict(torch.load(p, map_location=DEVICE))
            model.eval()
            offset = 0
            with torch.no_grad():
                for batch in test_dl:
                    sc = batch["seq_cat"].to(DEVICE); sn = batch["seq_num"].to(DEVICE)
                    ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                    st = batch["static"].to(DEVICE); lens = batch["length"]; nsns = batch["next_sn"]
                    a_log, p_log = model(sc, sn, ps, pr, pn, st, lens, next_sns=nsns, apply_mask=True)
                    bs = sc.size(0)
                    lstm_test_a[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / n_models
                    lstm_test_p[offset:offset+bs] += F.softmax(p_log, dim=-1).cpu().numpy() / n_models
                    offset += bs
    log(f"    LSTM test preds done ({time.time()-t0:.1f}s)")

    # ---- 5-fold LGB ----
    log("\n[6] Train 5-fold LGB (enhanced feats) ...")
    lgb_oof_a = np.zeros((len(train_samples), N_ACTION), dtype=np.float64)
    lgb_oof_p = np.zeros((len(train_samples), N_POINT), dtype=np.float64)
    lgb_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lgb_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)

    for fold in range(N_FOLDS):
        va_idx = np.where(sample_fold == fold)[0]
        tr_idx = np.where(sample_fold != fold)[0]
        log(f"\n  fold {fold+1}/{N_FOLDS}  tr={len(tr_idx)} va={len(va_idx)}")

        Xt = X_train.iloc[tr_idx]; Xv = X_train.iloc[va_idx]
        ya_t = y_a[tr_idx].tolist(); ya_v = y_a[va_idx].tolist()
        yp_t = y_p[tr_idx].tolist(); yp_v = y_p[va_idx].tolist()

        t0 = time.time()
        ma, _, _ = train_lgb_fold(Xt, ya_t, Xv, ya_v, N_ACTION, "actionId")
        lgb_oof_a[va_idx] = ma.predict(Xv)
        lgb_test_a += ma.predict(X_test) / N_FOLDS
        log(f"    action LGB ({time.time()-t0:.1f}s)")
        t0 = time.time()
        mp, _, _ = train_lgb_fold(Xt, yp_t, Xv, yp_v, N_POINT, "pointId")
        lgb_oof_p[va_idx] = mp.predict(Xv)
        lgb_test_p += mp.predict(X_test) / N_FOLDS
        log(f"    point  LGB ({time.time()-t0:.1f}s)")

    fa_raw = f1_score(y_a, lgb_oof_a.argmax(1), average="macro", zero_division=0)
    fp_raw = f1_score(y_p, lgb_oof_p.argmax(1), average="macro", zero_division=0)
    log(f"\n[7] Raw OOF:  P3 LGB F1_a={fa_raw:.4f} F1_p={fp_raw:.4f}")
    log(f"   V5 LGB base: F1_a=0.3903 F1_p=0.2523")

    # ---- save ----
    np.savez("artifacts/v7_p3_oof.npz",
             lgb_a=lgb_oof_a, lgb_p=lgb_oof_p,
             la=y_a, lp=y_p,
             sample_fold=sample_fold)
    np.savez("artifacts/v7_p3_test.npz",
             lgb_a=lgb_test_a, lgb_p=lgb_test_p,
             lstm_a=lstm_test_a, lstm_p=lstm_test_p)
    log(f"\n→ saved v7_p3_oof.npz + v7_p3_test.npz")
    log(f"TOTAL: {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
