#!/usr/bin/env python3
"""
Multi-prefix test extraction for H (sgp aux LSTM).

For each test rally with N_visible strokes:
  For k = 1..N_visible:
    For each fold's weights:
      Get sgp prediction at prefix length k
    Average across 5 folds
  Average all k predictions → rally's final prediction

This matches the OOF per-rally aggregation protocol (which gave AUC 0.7706),
expected to lift test AUC from single-prefix ~0.56 to multi-prefix ~0.65-0.70.

Usage:
    V5Z_TEST_CSV=data/test_new.csv V5Z_PID_TEST_CSV=data/test_new.csv \
    V5Z_INFER_TEST_ONLY=1 V5Z_LSTM_WEIGHTS_TAG=v5z_md_sgpaux_s42 \
    V5Z_TAG=v5z_md_sgpaux_s42 V5Z_SEED=42 V5Z_MATCH_DISJOINT=1 \
    python3 src/train/extract_sgp_aux_multiprefix.py
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "src/train")
from train_v5z import (
    PingPongModel, RallyDataset, prepare_samples,
    train_df, test_df, PLAYER_DROP_P, BATCH_SIZE, DEVICE, N_FOLDS, SEED,
    TAG, LSTM_WEIGHTS_TAG, MAX_SEQ_LEN, N_SEQ_CATS, N_SEQ_NUMS, PID2IDX, N_STATIC,
)


def build_multiprefix_test_samples(test_df_):
    """For each test rally, build samples for k=1..N_visible (all visible prefixes)."""
    import pandas as pd
    samples = []
    for uid, grp in test_df_.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        first = rows[0]
        for k in range(1, N + 1):
            ctx_rows = rows[:k]
            last_ctx = ctx_rows[-1]
            seq_cat = np.zeros((k, N_SEQ_CATS), dtype=np.int64)
            from train_v5z import SEQ_CAT_NAMES
            for i, r in enumerate(ctx_rows):
                for j, feat in enumerate(SEQ_CAT_NAMES):
                    seq_cat[i, j] = int(r[feat])
            seq_num = np.zeros((k, N_SEQ_NUMS), dtype=np.float32)
            for i, r in enumerate(ctx_rows):
                seq_num[i, 0] = r["scoreSelf"] / 15.0
                seq_num[i, 1] = r["scoreOther"] / 15.0
                seq_num[i, 2] = r["strikeNumber"] / 50.0
            pid_server = PID2IDX.get(first["gamePlayerId"], 0)
            pid_receiver = PID2IDX.get(first["gamePlayerOtherId"], 0)
            next_sn = k + 1
            pid_next = pid_server if next_sn % 2 == 1 else pid_receiver
            static = np.array([
                first["sex"], first["numberGame"] / 7.0, k / 50.0,
                next_sn % 2,
                (last_ctx["scoreSelf"] - last_ctx["scoreOther"]) / 15.0,
                (last_ctx["scoreSelf"] + last_ctx["scoreOther"]) / 30.0,
            ], dtype=np.float32)
            samples.append({
                "uid": uid, "seq_cat": seq_cat, "seq_num": seq_num,
                "pid_server": pid_server, "pid_receiver": pid_receiver, "pid_next": pid_next,
                "static": static, "length": k, "next_strike_num": next_sn,
                "sgp_label": 0,  # unused for test
            })
    return samples


def extract_sgp_for_samples(model, samples):
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)
    dl = DataLoader(RallyDataset(samples, is_train=False), BATCH_SIZE,
                    shuffle=False, num_workers=0, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            sc = batch["seq_cat"].to(DEVICE)
            sn = batch["seq_num"].to(DEVICE)
            ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
            st = batch["static"].to(DEVICE)
            lens = batch["length"]
            nsns = batch["next_sn"]
            res = model(sc, sn, ps, pr, pn, st, lens,
                        next_sns=nsns, apply_mask=True, return_sgp=True)
            sgp_logit = res[-1]
            out.append(torch.sigmoid(sgp_logit).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    print("=" * 70)
    print(f"Multi-prefix test sgp extraction — seed {SEED}")
    print("=" * 70)

    weights_tag = LSTM_WEIGHTS_TAG if LSTM_WEIGHTS_TAG else TAG
    print(f"Weights tag: {weights_tag}")

    test_samples_multi = build_multiprefix_test_samples(test_df)
    print(f"Multi-prefix test samples: {len(test_samples_multi)}")
    test_uids = np.array([s["uid"] for s in test_samples_multi])
    test_lens = np.array([s["length"] for s in test_samples_multi])

    sgp_test_accum = np.zeros(len(test_samples_multi), dtype=np.float32)

    for fold in range(N_FOLDS):
        weight_path = f"models/v5z/model_{weights_tag}_lstm_fold{fold+1}.pt"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing {weight_path}")
        model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True), strict=False)
        model.eval()
        print(f"Fold {fold+1}/{N_FOLDS}: loaded {weight_path}")
        sgp_test_fold = extract_sgp_for_samples(model, test_samples_multi)
        sgp_test_accum += sgp_test_fold / N_FOLDS

    # Aggregate per-rally: mean of all k predictions for each rally
    from collections import defaultdict
    rally_to_idx = defaultdict(list)
    for i, u in enumerate(test_uids):
        rally_to_idx[u].append(i)
    unique_rallies = sorted(rally_to_idx.keys())
    sgp_per_rally = np.array([sgp_test_accum[rally_to_idx[u]].mean() for u in unique_rallies], dtype=np.float32)

    print(f"\nMulti-prefix per-rally test pred stats:")
    print(f"  Rallies: {len(unique_rallies)}")
    print(f"  mean {sgp_per_rally.mean():.4f}, std {sgp_per_rally.std():.4f}, range [{sgp_per_rally.min():.4f}, {sgp_per_rally.max():.4f}]")

    out_path = f"artifacts/sgp_aux_test_multiprefix_NEWTEST_s{SEED}.npz"
    np.savez(out_path,
             rally_uids=np.array(unique_rallies),
             sgp_pred=sgp_per_rally,
             method=f"lstm_sgpaux_s{SEED}_multiprefix_avg",
             seed=SEED)
    print(f"\n→ Saved {out_path}")

    # Compare with single-prefix test pred
    single = np.load(f"artifacts/sgp_aux_test_NEWTEST_s{SEED}.npz", allow_pickle=True)
    single_uids = single["rally_uids"]
    single_pred = single["sgp_pred"]
    # Align order
    single_uid_to_pred = dict(zip(single_uids.tolist(), single_pred.tolist()))
    single_aligned = np.array([single_uid_to_pred[u] for u in unique_rallies])
    pearson = np.corrcoef(sgp_per_rally, single_aligned)[0, 1]
    print(f"\nPearson(multi-prefix, single-prefix test): {pearson:.4f}")
    print(f"  single mean={single_aligned.mean():.4f}, multi mean={sgp_per_rally.mean():.4f}")


if __name__ == "__main__":
    main()
