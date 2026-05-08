#!/usr/bin/env python3
"""
V6 Phase 1 submission：使用 V5 模型 + Phase 1 參數（per-task weights + plug-in bias）
生成 submission_v6.csv。不重訓 LSTM，只重訓 LGB 以取得 fold-averaged test probs。
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


def main():
    log("=" * 60)
    log("V6 Phase 1 推論：V5 模型 + plug-in bias + per-task weights")
    log("=" * 60)

    params = np.load("artifacts/v6_phase1_params.npz")
    bias_a = params["bias_a"]
    bias_p = params["bias_p"]
    w_a = float(params["w_a"])
    w_p = float(params["w_p"])
    log(f"載入 params: w_a={w_a:.2f} w_p={w_p:.2f}")
    log(f"  bias_a: {np.round(bias_a, 2).tolist()}")
    log(f"  bias_p: {np.round(bias_p, 2).tolist()}")

    log("\n準備 samples + features...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"  train samples: {len(train_samples)}, test samples: {len(test_samples)} ({time.time()-t0:.1f}s)")

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

    log(f"\n1) LSTM fold 推論（載入 model_v5_lstm_fold*.pt）...")
    for fold in range(N_FOLDS):
        model = PingPongModel(player_drop_p=0).to(DEVICE)
        model.load_state_dict(torch.load(
            f"models/v5/model_v5_lstm_fold{fold+1}.pt", map_location=DEVICE))
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
                lstm_test_a[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / N_FOLDS
                lstm_test_p[offset:offset+bs] += F.softmax(p_log, dim=-1).cpu().numpy() / N_FOLDS
                offset += bs
        log(f"  fold {fold+1} LSTM 完成")

    log(f"\n2) LGB fold 重訓並推論（使用真實 CV split 做 early stopping）...")
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

    # ---- Per-task ensemble + plug-in bias ----
    log(f"\n3) 組合 + plug-in 校準...")
    ens_a = w_a * lstm_test_a + (1 - w_a) * lgb_test_a
    ens_p = w_p * lstm_test_p + (1 - w_p) * lgb_test_p

    # 加性 log-bias
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
    sub.to_csv("submissions/submission_v6.csv", index=False)

    log(f"\n{'='*60}")
    log(f"提交檔案: submission_v6.csv ({len(sub)} 筆)")
    log(f"  actionId: {dict(sorted(Counter(pred_action).items()))}")
    log(f"  pointId:  {dict(sorted(Counter(pred_point).items()))}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
