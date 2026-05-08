#!/usr/bin/env python3
"""
Calibration for match-disjoint trained models. Mirrors advcal_v5z but with
StratifiedGroupKFold reindex (not StratifiedKFold).

Usage:
    V5Z_SEEDS=42,1337 python3 src/train/advcal_match_disjoint.py

Optional env vars:
    V5Z_ARTIFACT_PREFIX (default v5z_md_full)
    V5Z_TEST_ARTIFACT_SUFFIX (default empty; e.g. _NEWTEST → reads ..._s42_test_NEWTEST.npz)
    V5Z_SUBMISSION_SUFFIX (default empty; e.g. _NEWTEST → submission_v5z_md_advcal_bag2_NEWTEST.csv)
    V5Z_SGP_PRED_FILE (optional .npz with rally_uids + sgp_pred → override test_sgp from artifact)

Reads:
    artifacts/{prefix}_s{SEED}_oof.npz
    artifacts/{prefix}_s{SEED}_test{suffix}.npz

Outputs:
    submissions/submission_v5z_md_advcal_bag{N}{sub_suffix}.csv
    artifacts/v5z_md_advcal_bag{N}{sub_suffix}_params.npz
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src/train")
from train_v5z import prepare_samples, train_df, N_FOLDS


def build_md_oof_reindex(seed):
    """Match-disjoint reindex: sample_idx → oof_idx for StratifiedGroupKFold(groups=match_id)."""
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
    return sample_idx_to_oof_idx


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
    print("V5-Z Match-Disjoint Advanced Calibration")
    print("=" * 78)

    seeds = os.environ.get("V5Z_SEEDS", "42,1337").split(",")
    artifact_prefix = os.environ.get("V5Z_ARTIFACT_PREFIX", "v5z_md_full")
    test_suffix = os.environ.get("V5Z_TEST_ARTIFACT_SUFFIX", "")
    sub_suffix = os.environ.get("V5Z_SUBMISSION_SUFFIX", "")
    sgp_pred_file = os.environ.get("V5Z_SGP_PRED_FILE", "")
    print(f"Seeds: {seeds}")
    print(f"Artifact prefix: {artifact_prefix}")
    print(f"Test artifact suffix: '{test_suffix}'  Submission suffix: '{sub_suffix}'")
    if sgp_pred_file:
        print(f"SGP override file: {sgp_pred_file}")

    lstm_a_l, lstm_p_l, lgb_a_l, lgb_p_l = [], [], [], []
    la_ref, lp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/{artifact_prefix}_s{seed}_oof.npz")
        s2o = build_md_oof_reindex(int(seed))
        lstm_a_l.append(d["lstm_a"][s2o])
        lstm_p_l.append(d["lstm_p"][s2o])
        lgb_a_l.append(d["lgb_a"][s2o])
        lgb_p_l.append(d["lgb_p"][s2o])
        la_can = d["la"][s2o]; lp_can = d["lp"][s2o]
        if la_ref is None: la_ref, lp_ref = la_can, lp_can
        else:
            assert np.array_equal(la_ref, la_can), f"label mismatch seed {seed}"

    lstm_a = np.mean(lstm_a_l, 0); lstm_p = np.mean(lstm_p_l, 0)
    lgb_a = np.mean(lgb_a_l, 0);   lgb_p = np.mean(lgb_p_l, 0)
    n_action, n_point = lstm_a.shape[1], lstm_p.shape[1]
    print(f"OOF samples: {len(la_ref)}, action={n_action}, point={n_point}")

    print("\nRaw component (bagged):")
    print(f"  LSTM  action F1={macro_f1(la_ref, lstm_a.argmax(1)):.4f}  point F1={macro_f1(lp_ref, lstm_p.argmax(1)):.4f}")
    print(f"  LGB   action F1={macro_f1(la_ref, lgb_a.argmax(1)):.4f}  point F1={macro_f1(lp_ref, lgb_p.argmax(1)):.4f}")

    print("\nSearch alpha (step 0.01):")
    best_a, best_p = (None, -1, None), (None, -1, None)
    for alpha in np.arange(0.0, 1.001, 0.01):
        ens_a = alpha * lstm_a + (1 - alpha) * lgb_a
        ens_p = alpha * lstm_p + (1 - alpha) * lgb_p
        f_a = macro_f1(la_ref, ens_a.argmax(1))
        f_p = macro_f1(lp_ref, ens_p.argmax(1))
        if f_a > best_a[1]: best_a = (alpha, f_a, ens_a)
        if f_p > best_p[1]: best_p = (alpha, f_p, ens_p)
    a_alpha, _, ens_a_best = best_a
    p_alpha, _, ens_p_best = best_p
    print(f"  action α={a_alpha:.2f}  raw F1_a={best_a[1]:.4f}")
    print(f"  point  α={p_alpha:.2f}  raw F1_p={best_p[1]:.4f}")

    print("\nPlug-in additive bias:")
    bias_a, f_a_cal = plugin_add(ens_a_best, la_ref, n_action)
    bias_p, f_p_cal = plugin_add(ens_p_best, lp_ref, n_point)
    print(f"  action F1_a={f_a_cal:.4f}")
    print(f"  point  F1_p={f_p_cal:.4f}")

    final_cv = 0.4 * f_a_cal + 0.4 * f_p_cal + 0.2
    print(f"\nFinal match-disjoint OOF CV: {final_cv:.4f}")

    # Bag test predictions
    lstm_test_a_l, lstm_test_p_l = [], []
    lgb_test_a_l, lgb_test_p_l = [], []
    test_uids_ref, test_sgp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/{artifact_prefix}_s{seed}_test{test_suffix}.npz")
        lstm_test_a_l.append(d["lstm_a"]); lstm_test_p_l.append(d["lstm_p"])
        lgb_test_a_l.append(d["lgb_a"]);   lgb_test_p_l.append(d["lgb_p"])
        if test_uids_ref is None:
            test_uids_ref = d["test_uids"]; test_sgp_ref = d["test_sgp"]
        else:
            assert np.array_equal(test_uids_ref, d["test_uids"]), "test order mismatch"

    lstm_test_a = np.mean(lstm_test_a_l, 0); lstm_test_p = np.mean(lstm_test_p_l, 0)
    lgb_test_a = np.mean(lgb_test_a_l, 0);   lgb_test_p = np.mean(lgb_test_p_l, 0)

    ens_test_a = a_alpha * lstm_test_a + (1 - a_alpha) * lgb_test_a
    ens_test_p = p_alpha * lstm_test_p + (1 - p_alpha) * lgb_test_p
    pred_action = (np.log(ens_test_a + 1e-12) + bias_a).argmax(1)
    pred_point = (np.log(ens_test_p + 1e-12) + bias_p).argmax(1)

    if sgp_pred_file:
        if not os.path.exists(sgp_pred_file):
            raise FileNotFoundError(f"V5Z_SGP_PRED_FILE not found: {sgp_pred_file}")
        sgp_data = np.load(sgp_pred_file, allow_pickle=True)
        rally_to_sgp = dict(zip(sgp_data["rally_uids"].tolist(), sgp_data["sgp_pred"].tolist()))
        missing = [u for u in test_uids_ref.tolist() if u not in rally_to_sgp]
        if missing:
            raise ValueError(f"SGP pred missing for {len(missing)} rally_uids; first few: {missing[:5]}")
        test_sgp_final = np.array([rally_to_sgp[u] for u in test_uids_ref.tolist()])
        method = sgp_data["method"].item() if "method" in sgp_data.files else "unknown"
        print(f"Replaced test_sgp via {sgp_pred_file} (method={method})")
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

    np.savez(
        f"artifacts/v5z_md_advcal_bag{n_seeds}{sub_suffix}_params.npz",
        seeds=np.array(seeds), a_alpha=a_alpha, p_alpha=p_alpha,
        bias_a=bias_a, bias_p=bias_p, cv=final_cv,
    )

    print(f"\nCompare:")
    print(f"  baseline V5Z bag2 calibrated CV (rally-KFold): 0.4816  → LB 0.4285")
    print(f"  match-disjoint bag{n_seeds} calibrated CV:    {final_cv:.4f}  → expected LB ~{final_cv-0.008:.4f}")


if __name__ == "__main__":
    main()
