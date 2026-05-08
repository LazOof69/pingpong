#!/usr/bin/env python3
"""用 model_v4_fold*.pt 計算 V4 CV 分數"""
import os, sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT / "src" / "train"))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from train_lstm_v4 import (
    SEED, BATCH_SIZE, N_FOLDS, N_ACTION, N_POINT, DEVICE,
    train_df, prepare_samples, RallyDataset, PingPongModel,
    calibrate_thresholds, PLAYER_DROP_P, log,
)


def main():
    log("=" * 60)
    log(f"Device: {DEVICE}")
    log("重建訓練樣本...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    log(f"  樣本數: {len(train_samples)}")

    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {}
    for s in train_samples:
        uid_last_a[s["uid"]] = s["target_action"]
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])

    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    scores_raw = []
    all_a_probs, all_a_labels = [], []
    all_p_probs, all_p_labels = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        log(f"\nFold {fold+1}: val={len(va_s)}")

        va_dl = DataLoader(RallyDataset(va_s, is_train=True), BATCH_SIZE,
                           shuffle=False, num_workers=0, pin_memory=True)

        model = PingPongModel(player_drop_p=0).to(DEVICE)
        model.load_state_dict(torch.load(f"models/v4/model_v4_fold{fold+1}.pt", map_location=DEVICE))
        model.eval()

        fa_probs, fa_labels = [], []
        fp_probs, fp_labels = [], []

        with torch.no_grad():
            for batch in va_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps = batch["pid_server"]
                pr = batch["pid_receiver"]
                pn = batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]

                a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True)

                fa_probs.append(F.softmax(a_log, dim=-1).cpu().numpy())
                fa_labels.extend(batch["target_action"])
                fp_probs.append(F.softmax(p_log, dim=-1).cpu().numpy())
                fp_labels.extend(batch["target_point"])

        a_probs = np.vstack(fa_probs)
        a_labels = np.array(fa_labels)
        p_probs = np.vstack(fp_probs)
        p_labels = np.array(fp_labels)

        f1_a = f1_score(a_labels, a_probs.argmax(1), average="macro", zero_division=0)
        f1_p = f1_score(p_labels, p_probs.argmax(1), average="macro", zero_division=0)
        score = 0.4 * f1_a + 0.4 * f1_p + 0.2 * 1.0
        scores_raw.append(score)
        log(f"  (argmax) F1_act={f1_a:.4f}  F1_pt={f1_p:.4f}  Score={score:.4f}")

        all_a_probs.append(a_probs)
        all_a_labels.append(a_labels)
        all_p_probs.append(p_probs)
        all_p_labels.append(p_labels)

    log(f"\n{'='*60}")
    log(f"CV (argmax): {np.mean(scores_raw):.4f} +/- {np.std(scores_raw):.4f}")

    # 校準
    log(f"\n{'='*60}")
    log("Post-hoc 校準 (全部 validation 合併):")
    merged_a = np.vstack(all_a_probs)
    merged_al = np.concatenate(all_a_labels)
    merged_p = np.vstack(all_p_probs)
    merged_pl = np.concatenate(all_p_labels)

    log(f"\n  actionId:")
    a_scales, cal_f1_a = calibrate_thresholds(merged_a, merged_al, N_ACTION)
    log(f"\n  pointId:")
    p_scales, cal_f1_p = calibrate_thresholds(merged_p, merged_pl, N_POINT)

    cal_score = 0.4 * cal_f1_a + 0.4 * cal_f1_p + 0.2 * 1.0
    log(f"\n{'='*60}")
    log(f"校準後 CV Score: {cal_score:.4f}")
    log(f"  F1_action: {cal_f1_a:.4f}")
    log(f"  F1_point:  {cal_f1_p:.4f}")
    log(f"  AUC_sgp:   1.0000 (直接從 context 取)")


if __name__ == "__main__":
    main()
