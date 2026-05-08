#!/usr/bin/env python3
"""
V7 P3: LSTM seed bagging.
跑兩個額外 seed (1337, 2024)，CV split 與 V5 一致 (StratifiedKFold random_state=42)。
每個 fold 訓練前重新設定 seed，讓模型初始化與 dataloader shuffle 不同。
輸出：model_v7_seed{seed}_fold{i}.pt
"""
import os
import sys
import pathlib
_HERE = pathlib.Path(__file__).resolve().parent
os.chdir(_HERE.parent.parent)
sys.path.insert(0, str(_HERE))

import time
import argparse
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from train_v5 import (
    SEED, N_FOLDS, WEIGHT_ALPHA, SERVE_ACTIONS, N_ACTION, N_POINT,
    train_df, prepare_samples, compute_class_weights,
    train_lstm_fold, log,
)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--folds", type=str, default="1,2,3,4,5",
                    help="逗號分隔的 fold indices (1-based)")
    args = ap.parse_args()

    seed = args.seed
    folds_to_run = [int(x) for x in args.folds.split(",")]

    log("=" * 60)
    log(f"V7 Seed Bagging · seed={seed} · folds={folds_to_run}")
    log(f"CV split 仍用 SEED=42 (與 V5 一致)")
    log("=" * 60)

    log("準備 samples + class weights...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    act_cnt = Counter(s["target_action"] for s in train_samples)
    pt_cnt = Counter(s["target_point"] for s in train_samples)
    action_w = compute_class_weights(act_cnt, N_ACTION, alpha=WEIGHT_ALPHA,
                                     exclude=set(SERVE_ACTIONS))
    point_w = compute_class_weights(pt_cnt, N_POINT, alpha=WEIGHT_ALPHA)
    log(f"  {len(train_samples)} samples ({time.time()-t0:.1f}s)")

    # CV setup (same as V5)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)  # SEED=42

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        fold_1based = fold + 1
        if fold_1based not in folds_to_run:
            continue

        out_path = f"models/v7/model_v7_seed{seed}_fold{fold_1based}.pt"
        if os.path.exists(out_path):
            log(f"\n[fold {fold_1based}] 已存在 {out_path}，跳過")
            continue

        log(f"\n{'─'*50}")
        log(f"[seed={seed}] Fold {fold_1based}/{N_FOLDS}")

        # 設定 seed 讓 dataloader shuffle + model init 不同
        set_seed(seed + fold)

        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        tr_s = [train_samples[i] for u in tr_uids for i in uid2idx.get(u, [])]
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        log(f"  Train: {len(tr_s)}, Val: {len(va_s)}")

        t_fold = time.time()
        model, best_score = train_lstm_fold(tr_s, va_s, fold, action_w, point_w)
        log(f"  fold {fold_1based} done in {(time.time()-t_fold)/60:.1f} min, best={best_score:.4f}")

        torch.save(model.state_dict(), out_path)
        log(f"  saved → {out_path}")

    log(f"\n{'='*60}")
    log(f"seed={seed} all requested folds complete")


if __name__ == "__main__":
    main()
