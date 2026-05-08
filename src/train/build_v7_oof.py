#!/usr/bin/env python3
"""
V7 OOF: bagged 3-seed LSTM OOF (seeds 42, 1337, 2024) + keep V5 LGB OOF.
輸出 v7_oof.npz: lstm_a, lstm_p (3-seed avg), lgb_a, lgb_p, la, lp
"""
import os
import sys
import pathlib
_HERE = pathlib.Path(__file__).resolve().parent
os.chdir(_HERE.parent.parent)
sys.path.insert(0, str(_HERE))

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from train_v5 import (
    SEED, N_FOLDS, BATCH_SIZE, N_ACTION, N_POINT, DEVICE,
    train_df, prepare_samples,
    RallyDataset, PingPongModel, log,
)

SEEDS = [42, 1337, 2024]
MODEL_PATHS = {
    42:   "models/v5/model_v5_lstm_fold{fold}.pt",
    1337: "models/v7/model_v7_seed1337_fold{fold}.pt",
    2024: "models/v7/model_v7_seed2024_fold{fold}.pt",
}


def predict_fold(model_path, samples):
    model = PingPongModel(player_drop_p=0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    dl = DataLoader(RallyDataset(samples, is_train=False), BATCH_SIZE,
                    shuffle=False, num_workers=0, pin_memory=True)
    probs_a = np.zeros((len(samples), N_ACTION), dtype=np.float64)
    probs_p = np.zeros((len(samples), N_POINT), dtype=np.float64)
    offset = 0
    with torch.no_grad():
        for batch in dl:
            sc = batch["seq_cat"].to(DEVICE)
            sn = batch["seq_num"].to(DEVICE)
            ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
            st = batch["static"].to(DEVICE)
            lens = batch["length"]
            nsns = batch["next_sn"]
            a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                 next_sns=nsns, apply_mask=True)
            bs = sc.size(0)
            probs_a[offset:offset+bs] = F.softmax(a_log, dim=-1).cpu().numpy()
            probs_p[offset:offset+bs] = F.softmax(p_log, dim=-1).cpu().numpy()
            offset += bs
    return probs_a, probs_p


def main():
    log("=" * 70)
    log("V7 Build OOF — 3-seed bagged LSTM")
    log("=" * 70)

    log("Loading V5 LGB OOF + labels from v5_oof.npz ...")
    v5 = np.load("artifacts/v5_oof.npz")
    lgb_a, lgb_p = v5["lgb_a"], v5["lgb_p"]
    la, lp = v5["la"].astype(np.int64), v5["lp"].astype(np.int64)

    log("Preparing samples ...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    log(f"  {len(train_samples)} samples ({time.time()-t0:.1f}s)")

    # Build fold val ordering matching v5_oof (concatenated in fold order)
    lstm_a = np.zeros((len(la), N_ACTION), dtype=np.float64)
    lstm_p = np.zeros((len(lp), N_POINT), dtype=np.float64)
    cursor = 0

    for fold, (_, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        fold_1b = fold + 1
        va_uids = set(np.array(uid_list)[va_uidx])
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        n_va = len(va_s)
        log(f"\n[fold {fold_1b}] val={n_va}, cursor={cursor}")

        seed_probs_a = np.zeros((n_va, N_ACTION), dtype=np.float64)
        seed_probs_p = np.zeros((n_va, N_POINT), dtype=np.float64)
        for seed in SEEDS:
            path = MODEL_PATHS[seed].format(fold=fold_1b)
            if not os.path.exists(path):
                log(f"  ⚠ missing {path}, skipping seed")
                continue
            t = time.time()
            pa, pp = predict_fold(path, va_s)
            seed_probs_a += pa / len(SEEDS)
            seed_probs_p += pp / len(SEEDS)
            log(f"  seed={seed} {path} predicted ({time.time()-t:.1f}s)")

        lstm_a[cursor:cursor+n_va] = seed_probs_a
        lstm_p[cursor:cursor+n_va] = seed_probs_p
        cursor += n_va

    assert cursor == len(la), f"cursor {cursor} != OOF len {len(la)}"

    # Sanity: bagged OOF macro-F1
    from sklearn.metrics import f1_score
    fa = f1_score(la, lstm_a.argmax(1), average="macro", zero_division=0)
    fp = f1_score(lp, lstm_p.argmax(1), average="macro", zero_division=0)
    log(f"\nBagged LSTM raw OOF: F1_a={fa:.4f} F1_p={fp:.4f}")

    fa5 = f1_score(la, v5["lstm_a"].argmax(1), average="macro", zero_division=0)
    fp5 = f1_score(lp, v5["lstm_p"].argmax(1), average="macro", zero_division=0)
    log(f"V5  LSTM raw OOF:     F1_a={fa5:.4f} F1_p={fp5:.4f}")

    np.savez("artifacts/v7_oof.npz",
             lstm_a=lstm_a, lstm_p=lstm_p,
             lgb_a=lgb_a, lgb_p=lgb_p,
             la=la, lp=lp)
    log(f"\n→ saved v7_oof.npz")


if __name__ == "__main__":
    main()
