#!/usr/bin/env python3
"""
V5-Z post-hoc calibration with bagging + plug-in additive log-bias.

Reads all artifacts/v5z_s*_oof.npz and v5z_s*_test.npz, bag-averages
LSTM and LGB probs across seeds, searches scalar ensemble weight α,
then applies per-class additive log-bias calibration to maximize
macro-F1 directly. Generates submissions/submission_v5z_advcal.csv.

Usage:
    python src/train/advcal_v5z.py
    # Picks up whichever seeds are present.
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import glob
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src/train")
from train_v5z import prepare_samples, train_df, N_FOLDS


def build_oof_reindex(seed):
    """Return sample_idx_to_oof_idx mapping for a given fold seed.
    Replays V5Z's StratifiedKFold(seed) split to compute the OOF
    position of each train sample.
    """
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    sample_idx_to_oof_idx = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for _, (_, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                sample_idx_to_oof_idx[i] = cur
                cur += 1
    return sample_idx_to_oof_idx


def macro_f1(y_true, y_pred, n_class):
    # Match baseline/V5 metric: default sklearn macro-F1 averages over
    # classes present in union(y_true, y_pred). Forcing all N classes
    # would penalize empty classes (e.g. action 17/18 never appear as
    # next-stroke targets) and disagrees with the competition metric.
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def plugin_add(p, y, n_class, n_rounds=4, grid_step=0.05, grid_range=2.5):
    """
    Plug-in additive log-bias for macro-F1.
    argmax_k (log p_k + b_k); coordinate descent on b_k.
    """
    log_p = np.log(p + 1e-12)
    b = np.zeros(n_class)
    grid = np.arange(-grid_range, grid_range + 1e-9, grid_step)
    best_f1 = macro_f1(y, log_p.argmax(1), n_class)
    for r in range(n_rounds):
        improved = False
        for k in range(n_class):
            best_bk = b[k]
            for v in grid:
                b_try = b.copy()
                b_try[k] = v
                f = macro_f1(y, (log_p + b_try).argmax(1), n_class)
                if f > best_f1 + 1e-7:
                    best_f1 = f
                    best_bk = v
                    improved = True
            b[k] = best_bk
        if not improved:
            break
    return b, best_f1


def main():
    print("=" * 70)
    print("V5-Z Advanced Calibration: bagging + plug-in additive log-bias")
    print("=" * 70)

    seed_filter = os.environ.get("V5Z_SEEDS", "").strip()
    tag_prefix = os.environ.get("V5Z_TAG_PREFIX", "v5z").strip()
    if seed_filter:
        wanted = set(seed_filter.split(","))
        oof_paths = [f"artifacts/{tag_prefix}_s{s}_oof.npz" for s in sorted(wanted, key=int)]
        test_paths = [f"artifacts/{tag_prefix}_s{s}_test.npz" for s in sorted(wanted, key=int)]
    else:
        oof_paths = sorted(glob.glob(f"artifacts/{tag_prefix}_s*_oof.npz"))
        test_paths = sorted(glob.glob(f"artifacts/{tag_prefix}_s*_test.npz"))
    assert len(oof_paths) == len(test_paths), "OOF / test seed count mismatch"
    seeds = [p.split(f"{tag_prefix}_s")[1].split("_oof")[0] for p in oof_paths]
    print(f"Seeds: {seeds}")

    # --- Reindex each seed's OOF to canonical sample-idx order ---
    print("Reindexing OOFs to sample-idx canonical order...")
    lstm_a_list, lstm_p_list, lgb_a_list, lgb_p_list = [], [], [], []
    la_ref, lp_ref = None, None
    for path, seed in zip(oof_paths, seeds):
        d = np.load(path)
        s2o = build_oof_reindex(int(seed))
        # canonical[i] = oof[s2o[i]]
        lstm_a_list.append(d["lstm_a"][s2o])
        lstm_p_list.append(d["lstm_p"][s2o])
        lgb_a_list.append(d["lgb_a"][s2o])
        lgb_p_list.append(d["lgb_p"][s2o])
        la_can = d["la"][s2o]
        lp_can = d["lp"][s2o]
        if la_ref is None:
            la_ref, lp_ref = la_can, lp_can
        else:
            assert np.array_equal(la_ref, la_can), f"sample-idx labels mismatch for seed {seed}"
            assert np.array_equal(lp_ref, lp_can), f"sample-idx labels mismatch for seed {seed}"

    lstm_a = np.mean(lstm_a_list, axis=0)
    lstm_p = np.mean(lstm_p_list, axis=0)
    lgb_a = np.mean(lgb_a_list, axis=0)
    lgb_p = np.mean(lgb_p_list, axis=0)

    n_action, n_point = lstm_a.shape[1], lstm_p.shape[1]
    print(f"Action classes: {n_action}, Point classes: {n_point}")
    print(f"OOF samples: {len(la_ref)}")

    # --- Raw component F1 ---
    print("\nRaw component (bagged across seeds):")
    print(f"  LSTM  action F1={macro_f1(la_ref, lstm_a.argmax(1), n_action):.4f}  "
          f"point F1={macro_f1(lp_ref, lstm_p.argmax(1), n_point):.4f}")
    print(f"  LGB   action F1={macro_f1(la_ref, lgb_a.argmax(1), n_action):.4f}  "
          f"point F1={macro_f1(lp_ref, lgb_p.argmax(1), n_point):.4f}")

    # --- Search ensemble weight α (LSTM heavy in [0,1]), fine grid ---
    print("\nSearch scalar α (ens = α*LSTM + (1-α)*LGB), step=0.01:")
    best_a, best_p = (None, -1.0), (None, -1.0)
    for alpha in np.arange(0.0, 1.001, 0.01):
        ens_a = alpha * lstm_a + (1 - alpha) * lgb_a
        ens_p = alpha * lstm_p + (1 - alpha) * lgb_p
        f_a = macro_f1(la_ref, ens_a.argmax(1), n_action)
        f_p = macro_f1(lp_ref, ens_p.argmax(1), n_point)
        if f_a > best_a[1]:
            best_a = (alpha, f_a, ens_a)
        if f_p > best_p[1]:
            best_p = (alpha, f_p, ens_p)
    a_alpha, _, ens_a_best = best_a
    p_alpha, _, ens_p_best = best_p
    print(f"  action α={a_alpha:.2f}  raw F1_a={best_a[1]:.4f}")
    print(f"  point  α={p_alpha:.2f}  raw F1_p={best_p[1]:.4f}")

    # --- Plug-in additive bias on best ensembles ---
    print("\nPlug-in additive log-bias (per-class):")
    bias_a, f_a_cal = plugin_add(ens_a_best, la_ref, n_action, n_rounds=4)
    bias_p, f_p_cal = plugin_add(ens_p_best, lp_ref, n_point, n_rounds=4)
    print(f"  action F1_a={f_a_cal:.4f}  bias_a={np.round(bias_a, 2)}")
    print(f"  point  F1_p={f_p_cal:.4f}  bias_p={np.round(bias_p, 2)}")

    final_cv = 0.4 * f_a_cal + 0.4 * f_p_cal + 0.2
    print(f"\nFinal OOF CV: {final_cv:.4f}")

    # --- Bag test preds ---
    lstm_test_a_list, lstm_test_p_list = [], []
    lgb_test_a_list, lgb_test_p_list = [], []
    test_uids_ref, test_sgp_ref = None, None
    for p in test_paths:
        d = np.load(p)
        lstm_test_a_list.append(d["lstm_a"])
        lstm_test_p_list.append(d["lstm_p"])
        lgb_test_a_list.append(d["lgb_a"])
        lgb_test_p_list.append(d["lgb_p"])
        if test_uids_ref is None:
            test_uids_ref = d["test_uids"]
            test_sgp_ref = d["test_sgp"]
        else:
            assert np.array_equal(test_uids_ref, d["test_uids"]), "test order changed"

    lstm_test_a = np.mean(lstm_test_a_list, axis=0)
    lstm_test_p = np.mean(lstm_test_p_list, axis=0)
    lgb_test_a = np.mean(lgb_test_a_list, axis=0)
    lgb_test_p = np.mean(lgb_test_p_list, axis=0)

    ens_test_a = a_alpha * lstm_test_a + (1 - a_alpha) * lgb_test_a
    ens_test_p = p_alpha * lstm_test_p + (1 - p_alpha) * lgb_test_p

    log_test_a = np.log(ens_test_a + 1e-12) + bias_a
    log_test_p = np.log(ens_test_p + 1e-12) + bias_p
    pred_action = log_test_a.argmax(1)
    pred_point = log_test_p.argmax(1)

    sub = pd.DataFrame({
        "rally_uid": test_uids_ref,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp_ref,
    }).sort_values("rally_uid").reset_index(drop=True)

    n_seeds = len(seeds)
    sub_tag = "v5z_advcal" if tag_prefix == "v5z" else f"{tag_prefix}_advcal"
    out = f"submissions/submission_{sub_tag}_bag{n_seeds}.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission: {out}  ({len(sub)} rows)")

    np.savez(
        f"artifacts/{sub_tag}_bag{n_seeds}_params.npz",
        seeds=np.array(seeds), a_alpha=a_alpha, p_alpha=p_alpha,
        bias_a=bias_a, bias_p=bias_p, cv=final_cv,
    )

    print(f"\n→ Compare:")
    print(f"  baseline floor (epoch5): 0.4143")
    if "v5z_s42" in seeds:
        print(f"  V5Z s42 multiplicative cal: 0.4788")
    print(f"  V5Z bag{n_seeds} additive cal: {final_cv:.4f}")


if __name__ == "__main__":
    main()
