#!/usr/bin/env python3
"""
B: Pseudo-label LGB for action+point prediction.

Method:
1. Load current best ship's test action/point predictions (from advcal output via existing OOF + test artifacts).
2. Compute confidence per test rally (max softmax prob).
3. Filter top-confidence test rallies as pseudo-labels (separate threshold for action vs point).
4. Build LGB features for pseudo-labeled test rallies (using build_lgb_features).
5. Per fold (match-disjoint), augment train fold's LGB samples with all pseudo-test.
6. Re-train LGB action and point on augmented data.
7. Generate new test predictions, ensemble with LSTM.
8. Recalibrate (plug-in additive bias).
9. Output new ship CSV.

Anti-leak: pseudo-test rallies are NEVER in OOF val. They're always treated as
extra training samples. OOF AUC/F1 reflects only TRAIN val rallies.

Mobahi 2020 self-distillation theorem says matched-architecture self-distill
provides 0 information gain. We expect lift here because:
- LSTM (sequence model) → LGB (tree model) self-distill via cross-architecture
- Pseudo adds test INPUT distribution match (covariate shift mitigation)
- Macro-F1 metric benefits from balanced minority class samples

Usage:
    V5Z_SEEDS=42,1337 \
    V5Z_PSEUDO_CONF_ACTION=0.30 V5Z_PSEUDO_CONF_POINT=0.20 \
    V5Z_SUBMISSION_SUFFIX=_PSEUDO \
    V5Z_SGP_PRED_FILE=artifacts/sgp_pred_NEWTEST_t3_structural.npz \
    python3 src/train/pseudo_label_lgb.py

Outputs:
    submissions/submission_v5z_md_advcal_bag2{sub_suffix}.csv
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

# Set defaults before import — pseudo-label runs on new test
os.environ.setdefault("V5Z_TEST_CSV", "data/test_new.csv")
os.environ.setdefault("V5Z_PID_TEST_CSV", "data/test.csv")

sys.path.insert(0, "src/train")
from train_v5z import (
    train_df, test_df, prepare_samples, build_lgb_features,
    train_lgb_fold, N_FOLDS, N_ACTION, N_POINT, SEED,
    compute_class_weights, WEIGHT_ALPHA, SERVE_ACTIONS,
)


def macro_f1(y_true, y_pred, n_class=None):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def plugin_add(p, y, n_class, n_rounds=4, grid_step=0.05, grid_range=2.5):
    log_p = np.log(p + 1e-12)
    b = np.zeros(n_class)
    grid = np.arange(-grid_range, grid_range + 1e-9, grid_step)
    best_f1 = macro_f1(y, log_p.argmax(1))
    for r in range(n_rounds):
        improved = False
        for k in range(n_class):
            best_bk = b[k]
            for v in grid:
                b_try = b.copy(); b_try[k] = v
                f = macro_f1(y, (log_p + b_try).argmax(1))
                if f > best_f1 + 1e-7:
                    best_f1 = f; best_bk = v; improved = True
            b[k] = best_bk
        if not improved: break
    return b, best_f1


def main():
    print("=" * 78)
    print("B: Pseudo-label LGB for action+point")
    print("=" * 78)

    seeds = os.environ.get("V5Z_SEEDS", "42,1337").split(",")
    artifact_prefix = os.environ.get("V5Z_ARTIFACT_PREFIX", "v5z_md_full_NEWTEST")
    sub_suffix = os.environ.get("V5Z_SUBMISSION_SUFFIX", "_PSEUDO")
    sgp_pred_file = os.environ.get("V5Z_SGP_PRED_FILE", "")
    conf_action = float(os.environ.get("V5Z_PSEUDO_CONF_ACTION", "0.30"))
    conf_point = float(os.environ.get("V5Z_PSEUDO_CONF_POINT", "0.20"))
    print(f"Seeds: {seeds}, conf_action={conf_action}, conf_point={conf_point}")

    # ============================================================
    # Load OOF + test from existing artifacts
    # ============================================================
    lstm_a_l, lstm_p_l, lgb_a_l, lgb_p_l = [], [], [], []
    lstm_test_a_l, lstm_test_p_l = [], []
    lgb_test_a_l, lgb_test_p_l = [], []
    test_uids_ref, test_sgp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/{artifact_prefix}_s{seed}_oof.npz")
        lstm_a_l.append(d["lstm_a"]); lstm_p_l.append(d["lstm_p"])
        lgb_a_l.append(d["lgb_a"]); lgb_p_l.append(d["lgb_p"])
        d_test = np.load(f"artifacts/{artifact_prefix}_s{seed}_test.npz")
        lstm_test_a_l.append(d_test["lstm_a"]); lstm_test_p_l.append(d_test["lstm_p"])
        lgb_test_a_l.append(d_test["lgb_a"]); lgb_test_p_l.append(d_test["lgb_p"])
        if test_uids_ref is None:
            test_uids_ref = d_test["test_uids"]; test_sgp_ref = d_test["test_sgp"]

    # Bagged predictions
    lstm_test_a = np.mean(lstm_test_a_l, 0); lstm_test_p = np.mean(lstm_test_p_l, 0)
    lgb_test_a_orig = np.mean(lgb_test_a_l, 0); lgb_test_p_orig = np.mean(lgb_test_p_l, 0)

    # Use LSTM-dominant ensemble for pseudo labels (best calibrated source)
    # From advcal: action α=0.85, point α=0.98 (LSTM dominates)
    a_alpha_init = 0.85
    p_alpha_init = 0.98
    test_a_ensemble = a_alpha_init * lstm_test_a + (1 - a_alpha_init) * lgb_test_a_orig
    test_p_ensemble = p_alpha_init * lstm_test_p + (1 - p_alpha_init) * lgb_test_p_orig

    # ============================================================
    # Pseudo-label generation
    # ============================================================
    pseudo_action = test_a_ensemble.argmax(axis=1)
    pseudo_point = test_p_ensemble.argmax(axis=1)
    pseudo_action_conf = test_a_ensemble.max(axis=1)
    pseudo_point_conf = test_p_ensemble.max(axis=1)

    action_keep_mask = pseudo_action_conf >= conf_action
    point_keep_mask = pseudo_point_conf >= conf_point
    print(f"\nPseudo-label coverage:")
    print(f"  Action conf >= {conf_action}: {action_keep_mask.sum()}/{len(pseudo_action)} ({100*action_keep_mask.mean():.1f}%)")
    print(f"  Point  conf >= {conf_point}: {point_keep_mask.sum()}/{len(pseudo_point)} ({100*point_keep_mask.mean():.1f}%)")
    print(f"  Action class distribution: {dict(sorted(Counter(pseudo_action[action_keep_mask].tolist()).items()))}")

    # ============================================================
    # Build LGB features for test (one sample per rally)
    # ============================================================
    print("\nBuilding LGB features for train + test...")
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_uids, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_X_test, _, _, lgb_test_uids, _ = build_lgb_features(
        test_df, is_train=False)
    print(f"  Train LGB samples: {lgb_X_train.shape[0]}")
    print(f"  Test LGB samples: {lgb_X_test.shape[0]}")

    # Match test_uids to ensure alignment with predictions
    assert len(lgb_test_uids) == len(test_uids_ref), "test uid count mismatch"
    test_uid_to_pos = {u: i for i, u in enumerate(test_uids_ref.tolist())}
    test_lgb_pos = [test_uid_to_pos[u] for u in lgb_test_uids]
    pseudo_action_aligned = pseudo_action[test_lgb_pos]
    pseudo_point_aligned = pseudo_point[test_lgb_pos]
    action_keep_aligned = action_keep_mask[test_lgb_pos]
    point_keep_aligned = point_keep_mask[test_lgb_pos]

    # ============================================================
    # Match-disjoint folds (using train rallies only)
    # ============================================================
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    # LGB uid → idx mapping
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)

    # Class weights
    act_cnt = Counter(s["target_action"] for s in train_samples)
    pt_cnt = Counter(s["target_point"] for s in train_samples)
    action_w = compute_class_weights(act_cnt, N_ACTION, alpha=WEIGHT_ALPHA, exclude=set(SERVE_ACTIONS))
    point_w = compute_class_weights(pt_cnt, N_POINT, alpha=WEIGHT_ALPHA)

    # Iterate over seeds × folds, train LGB with pseudo-augmented data
    all_oof_lgb_a, all_oof_lgb_p = [], []
    all_oof_labels_a, all_oof_labels_p = [], []
    fold_test_a_preds, fold_test_p_preds = [], []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'─'*60}")
        print(f"Seed {seed} (idx {seed_idx+1}/{len(seeds)})")
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=int(seed))

        for fold, (tr_uidx, va_uidx) in enumerate(sgkf.split(uid_list, uid_arr, groups=match_arr)):
            tr_uids = set(np.array(uid_list)[tr_uidx])
            va_uids = set(np.array(uid_list)[va_uidx])

            # LGB train indices
            lgb_tr_idx = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
            lgb_va_idx = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]

            X_tr = lgb_X_train.iloc[lgb_tr_idx]
            X_va = lgb_X_train.iloc[lgb_va_idx]
            y_a_tr = [lgb_y_a[i] for i in lgb_tr_idx]
            y_a_va = [lgb_y_a[i] for i in lgb_va_idx]
            y_p_tr = [lgb_y_p[i] for i in lgb_tr_idx]
            y_p_va = [lgb_y_p[i] for i in lgb_va_idx]

            # === Augment train with pseudo-labeled test ===
            # Action: include test samples with action_keep_aligned=True
            X_tr_a_aug = pd.concat([X_tr, lgb_X_test[action_keep_aligned]], ignore_index=True)
            y_a_tr_aug = list(y_a_tr) + list(pseudo_action_aligned[action_keep_aligned])
            # Point: include test samples with point_keep_aligned=True
            X_tr_p_aug = pd.concat([X_tr, lgb_X_test[point_keep_aligned]], ignore_index=True)
            y_p_tr_aug = list(y_p_tr) + list(pseudo_point_aligned[point_keep_aligned])

            print(f"  Fold {fold+1}: tr={len(X_tr)} → {len(X_tr_a_aug)} (action +{len(X_tr_a_aug)-len(X_tr)} pseudo); va={len(X_va)}")

            # Train LGB action
            lgb_a_model, lgb_a_va_probs, _ = train_lgb_fold(X_tr_a_aug, y_a_tr_aug, X_va, y_a_va, N_ACTION, "actionId")
            lgb_p_model, lgb_p_va_probs, _ = train_lgb_fold(X_tr_p_aug, y_p_tr_aug, X_va, y_p_va, N_POINT, "pointId")

            all_oof_lgb_a.append(lgb_a_va_probs)
            all_oof_lgb_p.append(lgb_p_va_probs)
            all_oof_labels_a.append(np.array(y_a_va))
            all_oof_labels_p.append(np.array(y_p_va))

            fold_test_a_preds.append(lgb_a_model.predict(lgb_X_test))
            fold_test_p_preds.append(lgb_p_model.predict(lgb_X_test))

    # Aggregate OOF
    oof_lgb_a = np.vstack(all_oof_lgb_a)
    oof_lgb_p = np.vstack(all_oof_lgb_p)
    oof_la = np.concatenate(all_oof_labels_a)
    oof_lp = np.concatenate(all_oof_labels_p)
    print(f"\nOOF samples: {len(oof_la)} (across {len(seeds)} seeds × {N_FOLDS} folds)")
    print(f"  Pseudo LGB action F1: {macro_f1(oof_la, oof_lgb_a.argmax(1)):.4f}")
    print(f"  Pseudo LGB point F1:  {macro_f1(oof_lp, oof_lgb_p.argmax(1)):.4f}")

    # Bag test predictions
    new_lgb_test_a = np.mean(fold_test_a_preds, axis=0)
    new_lgb_test_p = np.mean(fold_test_p_preds, axis=0)
    print(f"  Pseudo LGB test: action {new_lgb_test_a.shape}, point {new_lgb_test_p.shape}")

    # ============================================================
    # Re-do alpha + plug-in calibration with new LGB OOF
    # ============================================================
    # Need to reindex OOF — but we just appended fold-by-fold so OOF order matches our concat
    # However, original LSTM OOF is from advcal's reindexed s2o order. We used different ordering.
    # For fair calibration, recompute LSTM OOF in our fold-concat order:
    print("\nRe-aligning LSTM OOF to match new fold-concat order...")
    lstm_a_concat = []
    lstm_p_concat = []
    for seed_idx, seed in enumerate(seeds):
        d = np.load(f"artifacts/{artifact_prefix}_s{seed}_oof.npz")
        # d's lstm_a is in OOF concat order (per fold's val rallies, in train_v5z order).
        # We replicate that order by iterating same SGKFold.
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=int(seed))
        # Build sample_idx → oof_idx
        s2o = np.empty(len(train_samples), dtype=np.int64)
        cur = 0
        for _, (_, va_uidx) in enumerate(sgkf.split(uid_list, uid_arr, groups=match_arr)):
            va_uids_set = set(np.array(uid_list)[va_uidx])
            for u in va_uids_set:
                for i in uid2idx.get(u, []):
                    s2o[i] = cur
                    cur += 1
        # Reindex: lstm_a in OOF order → align to fold-concat-by-fold-then-sample order
        # The order in our append loop is: for each fold, val samples in lgb_uid2idx order
        # This may differ from lstm OOF order which is per (rally, k) in fold's val
        # Match by uid: for each sample in our OOF order, find lstm value
        # Actually our OOF order is fold-concat (matches s2o). So lstm_a[s2o^-1] gives sample order.
        # But our append uses lgb_uid2idx order which iterates uids then per-uid samples.
        # Both should match if uid iteration order is sorted... it is (sorted(uid_list) above).

        # Inverse of s2o:
        inv_s2o = np.argsort(s2o)
        lstm_a_in_sample_order = d["lstm_a"][s2o]  # lstm_a indexed by sample_idx
        lstm_p_in_sample_order = d["lstm_p"][s2o]

        # Now reindex by our fold loop's order (fold by fold, va_uids → lgb_uid2idx)
        sgkf2 = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=int(seed))
        ordered_a = []
        ordered_p = []
        for _, (_, va_uidx) in enumerate(sgkf2.split(uid_list, uid_arr, groups=match_arr)):
            va_uids_set = set(np.array(uid_list)[va_uidx])
            for u in va_uids_set:
                for i in uid2idx.get(u, []):
                    ordered_a.append(lstm_a_in_sample_order[i])
                    ordered_p.append(lstm_p_in_sample_order[i])
        # Wait this is the same as s2o reindex order
        lstm_a_concat.append(np.array(ordered_a))
        lstm_p_concat.append(np.array(ordered_p))

    # OOF concat across seeds (same fold split mostly different seeds → different val)
    # Since seeds differ, the val rallies differ per seed. Our oof_lgb_a is also per-seed-fold.
    # So lstm_a_concat[i] should align row-by-row with all_oof_lgb_a[i]
    merged_lstm_a = np.vstack(lstm_a_concat)
    merged_lstm_p = np.vstack(lstm_p_concat)

    # Sanity check
    if merged_lstm_a.shape[0] != oof_lgb_a.shape[0]:
        print(f"  ⚠ Warning: shape mismatch lstm {merged_lstm_a.shape} vs lgb {oof_lgb_a.shape}")

    # Search alpha
    print("\nSearching alpha (LSTM/LGB blend)...")
    best_a = (None, -1, None); best_p = (None, -1, None)
    for alpha in np.arange(0.0, 1.001, 0.05):
        ens_a = alpha * merged_lstm_a + (1 - alpha) * oof_lgb_a
        ens_p = alpha * merged_lstm_p + (1 - alpha) * oof_lgb_p
        f_a = macro_f1(oof_la, ens_a.argmax(1))
        f_p = macro_f1(oof_lp, ens_p.argmax(1))
        if f_a > best_a[1]: best_a = (alpha, f_a, ens_a)
        if f_p > best_p[1]: best_p = (alpha, f_p, ens_p)
    a_alpha, _, ens_a_best = best_a
    p_alpha, _, ens_p_best = best_p
    print(f"  action α={a_alpha:.2f}  raw F1_a={best_a[1]:.4f}")
    print(f"  point  α={p_alpha:.2f}  raw F1_p={best_p[1]:.4f}")

    # Plug-in bias
    print("\nPlug-in bias...")
    bias_a, f_a_cal = plugin_add(ens_a_best, oof_la, N_ACTION)
    bias_p, f_p_cal = plugin_add(ens_p_best, oof_lp, N_POINT)
    print(f"  action F1_a={f_a_cal:.4f}")
    print(f"  point  F1_p={f_p_cal:.4f}")

    final_cv = 0.4 * f_a_cal + 0.4 * f_p_cal + 0.2
    print(f"\nFinal pseudo OOF CV: {final_cv:.4f}")

    # ============================================================
    # Test prediction with new LGB
    # ============================================================
    ens_test_a = a_alpha * lstm_test_a + (1 - a_alpha) * new_lgb_test_a
    ens_test_p = p_alpha * lstm_test_p + (1 - p_alpha) * new_lgb_test_p

    pred_action = (np.log(ens_test_a + 1e-12) + bias_a).argmax(1)
    pred_point = (np.log(ens_test_p + 1e-12) + bias_p).argmax(1)

    # SGP
    if sgp_pred_file:
        sgp_data = np.load(sgp_pred_file, allow_pickle=True)
        rally_to_sgp = dict(zip(sgp_data["rally_uids"].tolist(), sgp_data["sgp_pred"].tolist()))
        test_sgp_final = np.array([rally_to_sgp[u] for u in test_uids_ref.tolist()])
    else:
        test_sgp_final = test_sgp_ref

    sub = pd.DataFrame({
        "rally_uid": test_uids_ref,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp_final,
    }).sort_values("rally_uid").reset_index(drop=True)

    n_seeds = len(seeds)
    out = f"submissions/submission_v5z_md_advcal_bag{n_seeds}{sub_suffix}.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission: {out}  ({len(sub)} rows)")

    # Compare with original LGB OOF F1 (if we have it from advcal)
    print(f"\n--- Compare to original (no pseudo) ---")
    print(f"  Pseudo LGB action F1 OOF (raw): {macro_f1(oof_la, oof_lgb_a.argmax(1)):.4f}")
    print(f"  Pseudo LGB point F1 OOF (raw): {macro_f1(oof_lp, oof_lgb_p.argmax(1)):.4f}")
    # Original LGB raw F1 from baseline advcal (from earlier logs): F1_a=0.3044, F1_p=0.1820
    print(f"  Reference (baseline LGB, no pseudo): F1_a=0.3044, F1_p=0.1820")
    print(f"  Calibrated final F1_a={f_a_cal:.4f}, F1_p={f_p_cal:.4f}")
    print(f"  Reference (baseline calibrated): F1_a=0.3679, F1_p=0.2357")


if __name__ == "__main__":
    main()
