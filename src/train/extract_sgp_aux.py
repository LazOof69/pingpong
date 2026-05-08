#!/usr/bin/env python3
"""
Extract sgp predictions from LSTM weights trained with V5Z_SGP_AUX_W>0.

Per (rally, k) train sample: use that rally's val fold weights → sgp_logit → sigmoid.
Per test rally: average sgp_logit across all 5 folds → sigmoid.

Usage:
    V5Z_TEST_CSV=data/test_new.csv V5Z_PID_TEST_CSV=data/test_new.csv \
    V5Z_INFER_TEST_ONLY=1 V5Z_LSTM_WEIGHTS_TAG=v5z_md_sgpaux_s42 \
    V5Z_TAG=v5z_md_sgpaux_s42 V5Z_SEED=42 V5Z_MATCH_DISJOINT=1 \
    python3 src/train/extract_sgp_aux.py

Outputs:
    artifacts/sgp_aux_train_oof_s{SEED}.npz   (per-prepare_samples-sample sgp)
    artifacts/sgp_aux_test_NEWTEST_s{SEED}.npz (per-rally sgp, fold-averaged)
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
from train_v5z import (
    PingPongModel, RallyDataset, prepare_samples,
    train_df, test_df, PLAYER_DROP_P, BATCH_SIZE, DEVICE, N_FOLDS, SEED, TAG, LSTM_WEIGHTS_TAG,
)


def extract_sgp(model, samples):
    """Run inference, return sigmoid(sgp_logit) per sample."""
    if len(samples) == 0:
        return np.zeros((0,), dtype=np.float32)
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
            sgp_logit = res[-1]  # (B,)
            out.append(torch.sigmoid(sgp_logit).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    print("=" * 70)
    print(f"Extract sgp_aux predictions — seed {SEED}")
    print("=" * 70)

    weights_tag = LSTM_WEIGHTS_TAG if LSTM_WEIGHTS_TAG else TAG
    print(f"Weights tag: {weights_tag}")

    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    print(f"Train samples: {len(train_samples)}, test samples: {len(test_samples)}")

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

    sgp_oof = np.zeros(len(train_samples), dtype=np.float32)
    sgp_test_accum = np.zeros(len(test_samples), dtype=np.float32)

    for fold, (tr_uidx, va_uidx) in enumerate(kf_splits):
        weight_path = f"models/v5z/model_{weights_tag}_lstm_fold{fold+1}.pt"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing weight: {weight_path}")

        model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True), strict=False)
        model.eval()
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---  loaded {weight_path}")

        va_uids = set(np.array(uid_list)[va_uidx])
        va_sample_idx = [i for u in va_uids for i in uid2idx.get(u, [])]
        va_samples = [train_samples[i] for i in va_sample_idx]
        sgp_va = extract_sgp(model, va_samples)
        for arr_idx, sample_idx in enumerate(va_sample_idx):
            sgp_oof[sample_idx] = sgp_va[arr_idx]

        sgp_test_fold = extract_sgp(model, test_samples)
        sgp_test_accum += sgp_test_fold / N_FOLDS

    # Aggregate per-rally for both train OOF and test
    # Train: each rally has multiple (rally, k) samples → take MEAN per rally
    rally_to_oof_idx = {}
    for i, s in enumerate(train_samples):
        rally_to_oof_idx.setdefault(s["uid"], []).append(i)
    rally_uids_train = sorted(rally_to_oof_idx.keys())
    sgp_train_per_rally = np.array([sgp_oof[rally_to_oof_idx[u]].mean() for u in rally_uids_train], dtype=np.float32)

    # Get train sgp labels per rally
    rally_meta = train_df.groupby("rally_uid").first()[["serverGetPoint"]]
    sgp_label_per_rally = np.array([int(rally_meta.loc[u, "serverGetPoint"]) for u in rally_uids_train], dtype=np.float32)

    # Eval holdout AUC
    from sklearn.metrics import roc_auc_score
    auc_per_sample = roc_auc_score(
        np.array([sgp_label_per_rally[rally_uids_train.index(s["uid"])] if s["uid"] in rally_uids_train else 0
                  for s in train_samples]),
        sgp_oof,
    )
    auc_per_rally = roc_auc_score(sgp_label_per_rally, sgp_train_per_rally)
    print(f"\nOOF sgp AUC: per-sample {auc_per_sample:.4f}, per-rally aggregated {auc_per_rally:.4f}")

    out_oof = f"artifacts/sgp_aux_train_oof_s{SEED}.npz"
    out_test = f"artifacts/sgp_aux_test_NEWTEST_s{SEED}.npz"

    np.savez(out_oof,
             sgp_per_sample=sgp_oof,
             sample_uids=np.array([s["uid"] for s in train_samples]),
             sample_lens=np.array([s["length"] for s in train_samples]),
             sgp_per_rally=sgp_train_per_rally,
             rally_uids_train=np.array(rally_uids_train),
             sgp_label_per_rally=sgp_label_per_rally,
             oof_auc_per_sample=auc_per_sample,
             oof_auc_per_rally=auc_per_rally,
             seed=SEED)
    print(f"\n→ Saved {out_oof}")

    np.savez(out_test,
             rally_uids=np.array([s["uid"] for s in test_samples]),
             sgp_pred=sgp_test_accum,
             method=f"lstm_sgpaux_s{SEED}_5fold_avg",
             holdout_auc=auc_per_rally,
             seed=SEED)
    print(f"→ Saved {out_test}")


if __name__ == "__main__":
    main()
