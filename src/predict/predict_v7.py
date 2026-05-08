#!/usr/bin/env python3
"""
V7 submission：
- 15 LSTM models (seeds 42,1337,2024 × 5 folds) avg on test
- LGB retrain with real CV split (same as predict_v6_p1)
- v7_final_params.npz calibration (additive bias + per-task w)
輸出 submission_v7.csv
"""
import os, sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT / "src" / "train"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import time

from train_v5 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    train_df, test_df, prepare_samples, build_lgb_features,
    RallyDataset, PingPongModel, train_lgb_fold, log,
)

SEEDS = [42, 1337, 2024]
MODEL_PATHS = {
    42:   "models/v5/model_v5_lstm_fold{fold}.pt",
    1337: "models/v7/model_v7_seed1337_fold{fold}.pt",
    2024: "models/v7/model_v7_seed2024_fold{fold}.pt",
}


def main():
    log("=" * 60)
    log("V7 推論：3-seed LSTM bagging + LGB + V7 calibration")
    log("=" * 60)

    params = np.load("artifacts/v7_final_params.npz")
    bias_a = params["bias_a"]
    bias_p = params["bias_p"]
    w_a = float(params["w_a"])
    w_p = float(params["w_p"])
    log(f"params: w_a={w_a:.2f} w_p={w_p:.2f}")
    log(f"  bias_a: {np.round(bias_a, 2).tolist()}")
    log(f"  bias_p: {np.round(bias_p, 2).tolist()}")

    log("\n準備 samples + features...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"  train={len(train_samples)}, test={len(test_samples)} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    lgb_X_train, lgb_ya, lgb_yp, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_X_test, _, _, _, _ = build_lgb_features(test_df, is_train=False)
    log(f"  LGB features: train={lgb_X_train.shape} test={lgb_X_test.shape} ({time.time()-t0:.1f}s)")

    n_test = len(test_samples)
    lstm_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lstm_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)
    lgb_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lgb_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)

    test_dl = DataLoader(RallyDataset(test_samples, is_train=False), BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)

    log(f"\n1) LSTM 推論 (3 seeds × 5 folds = 15 models)...")
    n_models = len(SEEDS) * N_FOLDS
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            path = MODEL_PATHS[seed].format(fold=fold + 1)
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
            log(f"  seed={seed} fold {fold+1} done")

    log(f"\n2) LGB fold 重訓並推論...")
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        log(f"\n  fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        lgb_tr = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        X_tr, X_va = lgb_X_train.iloc[lgb_tr], lgb_X_train.iloc[lgb_va]
        y_a_tr = [lgb_ya[i] for i in lgb_tr]
        y_a_va = [lgb_ya[i] for i in lgb_va]
        y_p_tr = [lgb_yp[i] for i in lgb_tr]
        y_p_va = [lgb_yp[i] for i in lgb_va]

        a_model, _, _ = train_lgb_fold(X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "actionId")
        p_model, _, _ = train_lgb_fold(X_tr, y_p_tr, X_va, y_p_va, N_POINT, "pointId")
        lgb_test_a += a_model.predict(lgb_X_test) / N_FOLDS
        lgb_test_p += p_model.predict(lgb_X_test) / N_FOLDS

    log(f"\n3) 組合 + plug-in 校準...")
    ens_a = w_a * lstm_test_a + (1 - w_a) * lgb_test_a
    ens_p = w_p * lstm_test_p + (1 - w_p) * lgb_test_p
    log_a = np.log(ens_a + 1e-12) + bias_a
    log_p = np.log(ens_p + 1e-12) + bias_p
    pred_action = log_a.argmax(1)
    pred_point = log_p.argmax(1)

    test_sgp = [s["sgp_label"] for s in test_samples]
    test_uids = [s["uid"] for s in test_samples]

    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    })
    sub = sub.sort_values("rally_uid").reset_index(drop=True)
    sub.to_csv("submissions/submission_v7.csv", index=False)

    log(f"\n{'='*60}")
    log(f"submission_v7.csv ({len(sub)} 筆)")
    log(f"  actionId: {dict(sorted(Counter(pred_action).items()))}")
    log(f"  pointId:  {dict(sorted(Counter(pred_point).items()))}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
