#!/usr/bin/env python3
"""
Length-stratified plug-in additive bias calibration for action/point predictions.

For each L bucket {1, 2, 3, 4, 5+}, separately tune:
- Per-class plug-in bias to maximize macro-F1 within that bucket
- Optionally: per-bucket alpha (LSTM/LGB blend weight)

For test rallies: apply bucket-specific bias based on visible L = N_visible.

Hypothesis: test class distribution differs from OOF for short rallies (L=1, 2 cover
54% of test). Globally tuned bias is a compromise; per-L bias should improve F1 on
test where L distribution matches per-bucket calibration.

Usage:
    V5Z_SEEDS=42,1337 V5Z_SUBMISSION_SUFFIX=_LSTRAT \
    V5Z_SGP_PRED_FILE=artifacts/sgp_pred_NEWTEST_t3_structural.npz \
    python3 src/train/advcal_length_stratified.py

Outputs:
    submissions/submission_v5z_md_advcal_bag2{sub_suffix}.csv
    artifacts/v5z_md_advcal_lstrat_params.npz
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src/train")
from train_v5z import prepare_samples, train_df, test_df, N_FOLDS


# Buckets matching test L distribution
def get_bucket(k):
    if k == 1: return 0
    if k == 2: return 1
    if k == 3: return 2
    if k == 4: return 3
    return 4  # 5+


def build_md_oof_reindex(seed):
    """sample_idx → oof_idx for StratifiedGroupKFold."""
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    match_arr = np.array([uid_to_match[u] for u in uid_list])
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    sample_idx_to_oof_idx = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for _, (_, va_uidx) in enumerate(sgkf.split(uid_list, uid_arr, groups=match_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_idx_to_oof_idx[i] = cur
                cur += 1
    return sample_idx_to_oof_idx, train_samples


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def plugin_add(p, y, n_class, n_rounds=4, grid_step=0.05, grid_range=2.5, init_b=None):
    log_p = np.log(p + 1e-12)
    b = np.zeros(n_class) if init_b is None else init_b.copy()
    grid = np.arange(-grid_range, grid_range + 1e-9, grid_step)
    best_f1 = macro_f1(y, log_p.argmax(1))
    if init_b is not None:
        best_f1 = macro_f1(y, (log_p + b).argmax(1))
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
    print("Length-stratified plug-in bias calibration")
    print("=" * 78)

    seeds = os.environ.get("V5Z_SEEDS", "42,1337").split(",")
    artifact_prefix = os.environ.get("V5Z_ARTIFACT_PREFIX", "v5z_md_full_NEWTEST")
    test_suffix = os.environ.get("V5Z_TEST_ARTIFACT_SUFFIX", "")
    sub_suffix = os.environ.get("V5Z_SUBMISSION_SUFFIX", "_LSTRAT")
    sgp_pred_file = os.environ.get("V5Z_SGP_PRED_FILE", "")
    print(f"Seeds: {seeds}, prefix: {artifact_prefix}, sub_suffix: {sub_suffix}")
    if sgp_pred_file:
        print(f"SGP override file: {sgp_pred_file}")

    # ============================================================
    # Load OOF arrays + labels with reindex
    # ============================================================
    lstm_a_l, lstm_p_l, lgb_a_l, lgb_p_l = [], [], [], []
    la_ref, lp_ref = None, None
    train_samples_ref = None
    s2o_ref = None
    for seed in seeds:
        d = np.load(f"artifacts/{artifact_prefix}_s{seed}_oof.npz")
        s2o, train_samples = build_md_oof_reindex(int(seed))
        if train_samples_ref is None:
            train_samples_ref = train_samples
            s2o_ref = s2o
        lstm_a_l.append(d["lstm_a"][s2o])
        lstm_p_l.append(d["lstm_p"][s2o])
        lgb_a_l.append(d["lgb_a"][s2o])
        lgb_p_l.append(d["lgb_p"][s2o])
        la_can = d["la"][s2o]; lp_can = d["lp"][s2o]
        if la_ref is None: la_ref, lp_ref = la_can, lp_can

    lstm_a = np.mean(lstm_a_l, 0); lstm_p = np.mean(lstm_p_l, 0)
    lgb_a = np.mean(lgb_a_l, 0);   lgb_p = np.mean(lgb_p_l, 0)
    n_action, n_point = lstm_a.shape[1], lstm_p.shape[1]
    print(f"OOF samples: {len(la_ref)}, action={n_action}, point={n_point}")

    # Sample lengths (k value per OOF sample)
    sample_lengths = np.array([s["length"] for s in train_samples_ref])
    oof_lengths = np.empty_like(sample_lengths)
    oof_lengths[s2o_ref] = sample_lengths
    print(f"OOF k distribution: {Counter(oof_lengths.tolist())}")

    # Test L distribution (for weighted aggregate F1)
    test_samples = prepare_samples(test_df, is_train=False)
    test_lengths = np.array([s["length"] for s in test_samples])
    test_buckets = np.array([get_bucket(l) for l in test_lengths])
    bucket_weights = np.bincount(test_buckets, minlength=5).astype(float)
    bucket_weights /= bucket_weights.sum()
    print(f"Test L distribution: 1={Counter(test_lengths.tolist()).get(1,0)}, 2={Counter(test_lengths.tolist()).get(2,0)}, 3={Counter(test_lengths.tolist()).get(3,0)}, ...")
    print(f"Test bucket weights: {bucket_weights}")

    # ============================================================
    # Global alpha search (baseline)
    # ============================================================
    print("\n--- Global alpha search (baseline) ---")
    best_a, best_p = (None, -1, None), (None, -1, None)
    for alpha in np.arange(0.0, 1.001, 0.01):
        ens_a = alpha * lstm_a + (1 - alpha) * lgb_a
        ens_p = alpha * lstm_p + (1 - alpha) * lgb_p
        f_a = macro_f1(la_ref, ens_a.argmax(1))
        f_p = macro_f1(lp_ref, ens_p.argmax(1))
        if f_a > best_a[1]: best_a = (alpha, f_a, ens_a)
        if f_p > best_p[1]: best_p = (alpha, f_p, ens_p)
    a_alpha, _, ens_a_global = best_a
    p_alpha, _, ens_p_global = best_p
    print(f"  action α={a_alpha:.2f}  raw F1_a={best_a[1]:.4f}")
    print(f"  point  α={p_alpha:.2f}  raw F1_p={best_p[1]:.4f}")

    # Global plug-in bias
    bias_a_global, f_a_global_cal = plugin_add(ens_a_global, la_ref, n_action)
    bias_p_global, f_p_global_cal = plugin_add(ens_p_global, lp_ref, n_point)
    print(f"  Global plug-in: F1_a={f_a_global_cal:.4f}, F1_p={f_p_global_cal:.4f}")

    # ============================================================
    # Length-stratified plug-in bias
    # ============================================================
    print("\n--- Per-bucket plug-in bias (alpha global, bias per L bucket) ---")
    bias_a_per_bucket = {}
    bias_p_per_bucket = {}
    bucket_F1_a = {}
    bucket_F1_p = {}
    for bucket_id in range(5):
        mask = np.array([get_bucket(l) == bucket_id for l in oof_lengths])
        if mask.sum() < 100:
            print(f"  Bucket {bucket_id}: too few samples ({mask.sum()}), use global bias")
            bias_a_per_bucket[bucket_id] = bias_a_global
            bias_p_per_bucket[bucket_id] = bias_p_global
            continue
        # Use global alpha but search per-bucket bias
        ens_a_b = a_alpha * lstm_a[mask] + (1 - a_alpha) * lgb_a[mask]
        ens_p_b = p_alpha * lstm_p[mask] + (1 - p_alpha) * lgb_p[mask]
        bias_a_b, f_a_b = plugin_add(ens_a_b, la_ref[mask], n_action, init_b=bias_a_global)
        bias_p_b, f_p_b = plugin_add(ens_p_b, lp_ref[mask], n_point, init_b=bias_p_global)
        bias_a_per_bucket[bucket_id] = bias_a_b
        bias_p_per_bucket[bucket_id] = bias_p_b
        bucket_F1_a[bucket_id] = f_a_b
        bucket_F1_p[bucket_id] = f_p_b
        # Also compute with global bias for comparison
        ens_a_global_in_bucket = a_alpha * lstm_a[mask] + (1 - a_alpha) * lgb_a[mask]
        f_a_global_in_b = macro_f1(la_ref[mask], (np.log(ens_a_global_in_bucket + 1e-12) + bias_a_global).argmax(1))
        f_p_global_in_b = macro_f1(lp_ref[mask], (np.log(p_alpha * lstm_p[mask] + (1 - p_alpha) * lgb_p[mask] + 1e-12) + bias_p_global).argmax(1))
        n_in_bucket = int(mask.sum())
        print(f"  Bucket L={bucket_id+1 if bucket_id<4 else '5+'}: n={n_in_bucket}, "
              f"F1_a global={f_a_global_in_b:.4f} → strat={f_a_b:.4f} (Δ={f_a_b - f_a_global_in_b:+.4f}), "
              f"F1_p global={f_p_global_in_b:.4f} → strat={f_p_b:.4f} (Δ={f_p_b - f_p_global_in_b:+.4f})")

    # ============================================================
    # Aggregate F1: test-distribution-weighted
    # ============================================================
    print("\n--- Test-distribution-weighted F1 estimates ---")
    # Apply per-bucket bias to OOF samples and compute overall F1 weighted by test L distribution
    pred_a_strat = np.zeros(len(la_ref), dtype=int)
    pred_p_strat = np.zeros(len(lp_ref), dtype=int)
    pred_a_global = np.zeros(len(la_ref), dtype=int)
    pred_p_global = np.zeros(len(lp_ref), dtype=int)
    for bucket_id in range(5):
        mask = np.array([get_bucket(l) == bucket_id for l in oof_lengths])
        if mask.sum() == 0:
            continue
        ens_a_b = a_alpha * lstm_a[mask] + (1 - a_alpha) * lgb_a[mask]
        ens_p_b = p_alpha * lstm_p[mask] + (1 - p_alpha) * lgb_p[mask]
        pred_a_strat[mask] = (np.log(ens_a_b + 1e-12) + bias_a_per_bucket[bucket_id]).argmax(1)
        pred_p_strat[mask] = (np.log(ens_p_b + 1e-12) + bias_p_per_bucket[bucket_id]).argmax(1)
        pred_a_global[mask] = (np.log(ens_a_b + 1e-12) + bias_a_global).argmax(1)
        pred_p_global[mask] = (np.log(ens_p_b + 1e-12) + bias_p_global).argmax(1)

    # Compute F1 per bucket and aggregate weighted by test bucket distribution
    f_a_strat_test = 0.0; f_a_global_test = 0.0
    f_p_strat_test = 0.0; f_p_global_test = 0.0
    for bucket_id in range(5):
        mask = np.array([get_bucket(l) == bucket_id for l in oof_lengths])
        if mask.sum() == 0:
            continue
        f_a_strat_test += bucket_weights[bucket_id] * macro_f1(la_ref[mask], pred_a_strat[mask])
        f_p_strat_test += bucket_weights[bucket_id] * macro_f1(lp_ref[mask], pred_p_strat[mask])
        f_a_global_test += bucket_weights[bucket_id] * macro_f1(la_ref[mask], pred_a_global[mask])
        f_p_global_test += bucket_weights[bucket_id] * macro_f1(lp_ref[mask], pred_p_global[mask])
    print(f"  Test-weighted F1_a: global {f_a_global_test:.4f} → stratified {f_a_strat_test:.4f} (Δ={f_a_strat_test - f_a_global_test:+.4f})")
    print(f"  Test-weighted F1_p: global {f_p_global_test:.4f} → stratified {f_p_strat_test:.4f} (Δ={f_p_strat_test - f_p_global_test:+.4f})")
    print(f"  Test-weighted Score: global {0.4*f_a_global_test + 0.4*f_p_global_test + 0.2:.4f}, stratified {0.4*f_a_strat_test + 0.4*f_p_strat_test + 0.2:.4f}")
    print(f"  → Δscore from F1 only: {0.4*(f_a_strat_test + f_p_strat_test) - 0.4*(f_a_global_test + f_p_global_test):+.4f}")

    # Standard OOF F1 (unweighted by test buckets, just global)
    f_a_strat_oof_total = macro_f1(la_ref, pred_a_strat)
    f_p_strat_oof_total = macro_f1(lp_ref, pred_p_strat)
    print(f"  OOF total (unweighted): F1_a={f_a_strat_oof_total:.4f} (vs global {f_a_global_cal:.4f}), F1_p={f_p_strat_oof_total:.4f} (vs global {f_p_global_cal:.4f})")

    final_cv_strat = 0.4 * f_a_strat_oof_total + 0.4 * f_p_strat_oof_total + 0.2
    print(f"\nFinal stratified OOF CV (assuming AUC=1.0): {final_cv_strat:.4f}")

    # ============================================================
    # Test prediction
    # ============================================================
    lstm_test_a_l, lstm_test_p_l = [], []
    lgb_test_a_l, lgb_test_p_l = [], []
    test_uids_ref, test_sgp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/{artifact_prefix}_s{seed}_test{test_suffix}.npz")
        lstm_test_a_l.append(d["lstm_a"]); lstm_test_p_l.append(d["lstm_p"])
        lgb_test_a_l.append(d["lgb_a"]);   lgb_test_p_l.append(d["lgb_p"])
        if test_uids_ref is None:
            test_uids_ref = d["test_uids"]; test_sgp_ref = d["test_sgp"]

    lstm_test_a = np.mean(lstm_test_a_l, 0); lstm_test_p = np.mean(lstm_test_p_l, 0)
    lgb_test_a = np.mean(lgb_test_a_l, 0);   lgb_test_p = np.mean(lgb_test_p_l, 0)
    ens_test_a = a_alpha * lstm_test_a + (1 - a_alpha) * lgb_test_a
    ens_test_p = p_alpha * lstm_test_p + (1 - p_alpha) * lgb_test_p

    # Apply per-rally bucket bias
    pred_action_strat = np.zeros(len(test_uids_ref), dtype=int)
    pred_point_strat = np.zeros(len(test_uids_ref), dtype=int)
    for i, l in enumerate(test_lengths):
        b = get_bucket(l)
        pred_action_strat[i] = (np.log(ens_test_a[i:i+1] + 1e-12) + bias_a_per_bucket[b]).argmax()
        pred_point_strat[i] = (np.log(ens_test_p[i:i+1] + 1e-12) + bias_p_per_bucket[b]).argmax()

    # SGP override
    if sgp_pred_file:
        if not os.path.exists(sgp_pred_file):
            raise FileNotFoundError(sgp_pred_file)
        sgp_data = np.load(sgp_pred_file, allow_pickle=True)
        rally_to_sgp = dict(zip(sgp_data["rally_uids"].tolist(), sgp_data["sgp_pred"].tolist()))
        test_sgp_final = np.array([rally_to_sgp[u] for u in test_uids_ref.tolist()])
        method = sgp_data["method"].item() if "method" in sgp_data.files else "unknown"
        print(f"Replaced test_sgp via {sgp_pred_file} (method={method})")
    else:
        test_sgp_final = test_sgp_ref

    sub = pd.DataFrame({
        "rally_uid": test_uids_ref,
        "actionId": pred_action_strat,
        "pointId": pred_point_strat,
        "serverGetPoint": test_sgp_final,
    }).sort_values("rally_uid").reset_index(drop=True)

    n_seeds = len(seeds)
    out = f"submissions/submission_v5z_md_advcal_bag{n_seeds}{sub_suffix}.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission: {out}  ({len(sub)} rows)")

    np.savez(
        f"artifacts/v5z_md_advcal_lstrat{sub_suffix}_params.npz",
        seeds=np.array(seeds),
        a_alpha=a_alpha, p_alpha=p_alpha,
        bias_a_per_bucket=bias_a_per_bucket, bias_p_per_bucket=bias_p_per_bucket,
        bias_a_global=bias_a_global, bias_p_global=bias_p_global,
        cv_strat=final_cv_strat,
        test_weighted_f1_a_strat=f_a_strat_test, test_weighted_f1_a_global=f_a_global_test,
        test_weighted_f1_p_strat=f_p_strat_test, test_weighted_f1_p_global=f_p_global_test,
    )

    print(f"\nF1 lift via length-stratified calibration:")
    print(f"  F1_a: +{f_a_strat_test - f_a_global_test:.4f}")
    print(f"  F1_p: +{f_p_strat_test - f_p_global_test:.4f}")
    print(f"  Total score lift (F1 only, ignoring AUC change): +{0.4*(f_a_strat_test + f_p_strat_test) - 0.4*(f_a_global_test + f_p_global_test):.4f}")


if __name__ == "__main__":
    main()
