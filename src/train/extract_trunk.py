#!/usr/bin/env python3
"""
Extract pooled trunk_out (128-dim final hidden representation) from trained
LSTM weights per train OOF sample and per test rally.

OOF integrity: per (seed, fold), only run inference on samples whose rally is
in that fold's val. Match-disjoint by construction.

Usage:
    V5Z_TEST_CSV=data/test_new.csv V5Z_PID_TEST_CSV=data/test.csv \
    V5Z_INFER_TEST_ONLY=1 V5Z_LSTM_WEIGHTS_TAG=v5z_md_full_s42 \
    V5Z_TAG=v5z_md_full_s42 V5Z_SEED=42 V5Z_MATCH_DISJOINT=1 \
    python3 src/train/extract_trunk.py

Outputs (written for whichever seed is provided):
    artifacts/trunk_train_oof_s{SEED}.npz  (sample_idx-aligned, 128-dim per sample)
    artifacts/trunk_test_NEWTEST_s{SEED}.npz  (rally-aligned, 128-dim per rally)
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, "src/train")
import train_v5z as M
from train_v5z import (
    PingPongModel, RallyDataset, prepare_samples,
    train_df, test_df, PLAYER_DROP_P, BATCH_SIZE, DEVICE, N_FOLDS, SEED, TAG, LSTM_WEIGHTS_TAG,
)


def extract_trunk_for_samples(model, samples):
    """Run model inference on samples, return trunk_out (n_samples, 128)."""
    if len(samples) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    dl = DataLoader(RallyDataset(samples, is_train=False), BATCH_SIZE,
                    shuffle=False, num_workers=0, pin_memory=True)
    out_arr = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            sc = batch["seq_cat"].to(DEVICE)
            sn = batch["seq_num"].to(DEVICE)
            ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
            st = batch["static"].to(DEVICE)
            lens = batch["length"]
            nsns = batch["next_sn"]
            out = model(sc, sn, ps, pr, pn, st, lens,
                        next_sns=nsns, apply_mask=True, return_trunk=True)
            trunk = out[-1]  # last element when return_trunk=True
            out_arr.append(trunk.cpu().numpy())
    return np.concatenate(out_arr, axis=0)


def main():
    print("=" * 70)
    print(f"Trunk Extraction — seed {SEED}")
    print("=" * 70)

    weights_tag = LSTM_WEIGHTS_TAG if LSTM_WEIGHTS_TAG else TAG
    print(f"Weights tag: {weights_tag}")
    print(f"Train rallies: {train_df['rally_uid'].nunique()}, test rallies: {test_df['rally_uid'].nunique()}")

    # Build train samples (same logic as train_v5z) — augmented k=1..N-1 prefixes
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    print(f"Train samples (augmented): {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Build fold split (match-disjoint) — must mirror train_v5z exactly
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()

    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    kf_splits = list(sgkf.split(uid_list, uid_arr, groups=match_arr))

    # Trunk OOF: per fold, only val sample_idx get this fold's trunk
    trunk_oof = np.zeros((len(train_samples), 128), dtype=np.float32)
    trunk_test_accum = np.zeros((len(test_samples), 128), dtype=np.float32)

    for fold, (tr_uidx, va_uidx) in enumerate(kf_splits):
        weight_path = f"models/v5z/model_{weights_tag}_lstm_fold{fold+1}.pt"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weight: {weight_path}")

        model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True), strict=False)
        model.eval()
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---  loaded {weight_path}")

        # Val samples for OOF trunk
        va_uids = set(np.array(uid_list)[va_uidx])
        va_sample_idx = [i for u in va_uids for i in uid2idx.get(u, [])]
        va_samples = [train_samples[i] for i in va_sample_idx]
        print(f"  Val samples: {len(va_samples)}  (over {len(va_uids)} rallies)")
        trunk_va = extract_trunk_for_samples(model, va_samples)
        for arr_idx, sample_idx in enumerate(va_sample_idx):
            trunk_oof[sample_idx] = trunk_va[arr_idx]

        # Test trunk: this fold's contribution (averaged later)
        trunk_test_fold = extract_trunk_for_samples(model, test_samples)
        trunk_test_accum += trunk_test_fold / N_FOLDS

    # Save
    out_oof = f"artifacts/trunk_train_oof_s{SEED}.npz"
    out_test = f"artifacts/trunk_test_NEWTEST_s{SEED}.npz"

    np.savez(out_oof,
             trunk=trunk_oof,
             sample_uids=np.array([s["uid"] for s in train_samples]),
             sample_lens=np.array([s["length"] for s in train_samples]),
             sample_sgp=np.array([s["sgp_label"] for s in train_samples]),
             seed=SEED)
    print(f"\n→ Saved {out_oof}  shape={trunk_oof.shape}")

    np.savez(out_test,
             trunk=trunk_test_accum,
             rally_uids=np.array([s["uid"] for s in test_samples]),
             seed=SEED)
    print(f"→ Saved {out_test}  shape={trunk_test_accum.shape}")


if __name__ == "__main__":
    main()
