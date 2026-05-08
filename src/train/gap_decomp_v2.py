#!/usr/bin/env python3
"""
Gap Decomposition v2 — find the REAL source of OOF/LB gap = -0.053.

Length distribution shift was ruled out by fair_comparison.py
(T@(F-fit)=0.4814 vs F@(F-fit)=0.4816, Δ=-0.0002).

This script tests remaining hypotheses:
  H_class17    : LB scoring forces labels=range(N), penalizing absent class 17/18
  H_player     : test players underrepresented or absent in train
  H_class_dist : test action/point class distribution differs from train
  H_match      : test rallies come from different matches → different style
  H_residual   : after controlling above, what's left = fundamental noise / unknown
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import Counter

sys.path.insert(0, "src/train")
from train_v5z import prepare_samples, train_df, N_FOLDS

import advcal_v5z


def load_oof_bag2():
    """Reproduce advcal bag2 ensemble probs with α=0.64/0.50."""
    seeds = ["42", "1337"]
    lstm_a_list, lstm_p_list, lgb_a_list, lgb_p_list = [], [], [], []
    la_ref, lp_ref = None, None
    for seed in seeds:
        d = np.load(f"artifacts/v5z_s{seed}_oof.npz")
        s2o = advcal_v5z.build_oof_reindex(int(seed))
        lstm_a_list.append(d["lstm_a"][s2o])
        lstm_p_list.append(d["lstm_p"][s2o])
        lgb_a_list.append(d["lgb_a"][s2o])
        lgb_p_list.append(d["lgb_p"][s2o])
        la_can = d["la"][s2o]; lp_can = d["lp"][s2o]
        if la_ref is None: la_ref, lp_ref = la_can, lp_can

    lstm_a = np.mean(lstm_a_list, axis=0)
    lstm_p = np.mean(lstm_p_list, axis=0)
    lgb_a = np.mean(lgb_a_list, axis=0)
    lgb_p = np.mean(lgb_p_list, axis=0)
    return lstm_a, lstm_p, lgb_a, lgb_p, la_ref, lp_ref


def main():
    print("=" * 78)
    print("GAP DECOMPOSITION v2 — find REAL source of OOF/LB gap = -0.053")
    print("=" * 78)

    # Load OOF
    print("\nLoading OOF artifacts (bag2)...")
    lstm_a, lstm_p, lgb_a, lgb_p, la_ref, lp_ref = load_oof_bag2()

    # Load original calibration params
    p = np.load("artifacts/v5z_advcal_bag2_params.npz")
    a_alpha, p_alpha = float(p["a_alpha"]), float(p["p_alpha"])
    bias_a, bias_p = p["bias_a"], p["bias_p"]
    print(f"  α_action={a_alpha:.2f}  α_point={p_alpha:.2f}")

    ens_a = a_alpha * lstm_a + (1 - a_alpha) * lgb_a
    ens_p = p_alpha * lstm_p + (1 - p_alpha) * lgb_p
    pred_a = (np.log(ens_a + 1e-12) + bias_a).argmax(1)
    pred_p = (np.log(ens_p + 1e-12) + bias_p).argmax(1)

    n_action, n_point = ens_a.shape[1], ens_p.shape[1]

    # ======================================================================
    # H_class17 : labels=range(N) vs default (present-only)
    # ======================================================================
    print("\n" + "=" * 78)
    print("H_class17: macro-F1 with labels=range(N) vs default")
    print("=" * 78)

    f_a_default = f1_score(la_ref, pred_a, average="macro", zero_division=0)
    f_a_forced = f1_score(la_ref, pred_a, average="macro", zero_division=0,
                          labels=list(range(n_action)))
    f_p_default = f1_score(lp_ref, pred_p, average="macro", zero_division=0)
    f_p_forced = f1_score(lp_ref, pred_p, average="macro", zero_division=0,
                          labels=list(range(n_point)))

    cv_default = 0.4 * f_a_default + 0.4 * f_p_default + 0.2
    cv_forced = 0.4 * f_a_forced + 0.4 * f_p_forced + 0.2

    label_action = sorted(set(la_ref))
    label_point = sorted(set(lp_ref))
    print(f"\n  action labels in OOF: {label_action}")
    print(f"  point  labels in OOF: {label_point}")
    print(f"\n  F1_action  default={f_a_default:.4f}  forced={f_a_forced:.4f}  Δ={f_a_forced-f_a_default:+.4f}")
    print(f"  F1_point   default={f_p_default:.4f}  forced={f_p_forced:.4f}  Δ={f_p_forced-f_p_default:+.4f}")
    print(f"  CV         default={cv_default:.4f}  forced={cv_forced:.4f}  Δ={cv_forced-cv_default:+.4f}")
    print(f"  → If LB uses labels=range(N), gap contribution: {cv_forced-cv_default:+.4f}")

    # ======================================================================
    # H_player : player overlap between train and test
    # ======================================================================
    print("\n" + "=" * 78)
    print("H_player: player overlap train vs test")
    print("=" * 78)

    test_df = pd.read_csv("data/test.csv")
    tr_p1 = set(train_df["gamePlayerId"].unique())
    tr_p2 = set(train_df["gamePlayerOtherId"].unique())
    tr_players = tr_p1 | tr_p2
    te_p1 = set(test_df["gamePlayerId"].unique())
    te_p2 = set(test_df["gamePlayerOtherId"].unique())
    te_players = te_p1 | te_p2

    print(f"\n  train unique players: {len(tr_players)}")
    print(f"  test  unique players: {len(te_players)}")
    print(f"  test players ALSO in train: {len(te_players & tr_players)} ({100*len(te_players & tr_players)/len(te_players):.1f}%)")
    print(f"  test players NEVER in train: {len(te_players - tr_players)}")
    if te_players - tr_players:
        unseen = list(te_players - tr_players)[:10]
        print(f"    sample unseen ids: {unseen}")

    # Per-test-rally: are both players seen in train?
    test_rallies = test_df.groupby("rally_uid").first()
    test_rallies["p1_seen"] = test_rallies["gamePlayerId"].isin(tr_players)
    test_rallies["p2_seen"] = test_rallies["gamePlayerOtherId"].isin(tr_players)
    test_rallies["both_seen"] = test_rallies["p1_seen"] & test_rallies["p2_seen"]

    print(f"\n  test rallies both players seen: {test_rallies['both_seen'].sum()} / {len(test_rallies)} ({100*test_rallies['both_seen'].mean():.1f}%)")
    print(f"  test rallies one player unseen:  {(~test_rallies['both_seen']).sum()}")

    # Player frequency comparison
    tr_pcount = train_df["gamePlayerId"].value_counts()
    te_pcount = test_df["gamePlayerId"].value_counts()

    # For overlap players: rank by train freq
    overlap_players = list(te_players & tr_players)
    print(f"\n  Top 10 most frequent test players' train frequency:")
    top_test = te_pcount.head(15)
    for pid, te_cnt in top_test.items():
        tr_cnt = tr_pcount.get(pid, 0)
        ratio = te_cnt / max(tr_cnt, 1)
        print(f"    pid={pid:>5}  test={te_cnt:>4}  train={tr_cnt:>5}  test/train={ratio:.4f}")

    # ======================================================================
    # H_match : test matches in train?
    # ======================================================================
    print("\n" + "=" * 78)
    print("H_match: test match overlap with train")
    print("=" * 78)

    tr_matches = set(train_df["match"].unique())
    te_matches = set(test_df["match"].unique())
    print(f"\n  train matches: {len(tr_matches)}")
    print(f"  test  matches: {len(te_matches)}")
    print(f"  overlap: {len(te_matches & tr_matches)} ({100*len(te_matches & tr_matches)/len(te_matches):.1f}%)")
    print(f"  test matches NEVER in train: {len(te_matches - tr_matches)}")

    # ======================================================================
    # H_class_dist : action/point distribution drift
    # ======================================================================
    print("\n" + "=" * 78)
    print("H_class_dist: train rally last-stroke vs test 'last visible' stroke distribution")
    print("=" * 78)

    # Train last-stroke action/point per rally
    tr_last = train_df.groupby("rally_uid").last()
    tr_action_dist = tr_last["actionId"].value_counts(normalize=True).sort_index()
    tr_point_dist = tr_last["pointId"].value_counts(normalize=True).sort_index()

    # Test last visible stroke per rally (the LAST stroke we can see)
    te_last_visible = test_df.groupby("rally_uid").last()
    te_action_dist = te_last_visible["actionId"].value_counts(normalize=True).sort_index()
    te_point_dist = te_last_visible["pointId"].value_counts(normalize=True).sort_index()

    print("\n  Action distribution (train last-stroke vs test last-VISIBLE-stroke):")
    print(f"  {'class':>6}  {'train':>8}  {'test':>8}  {'diff':>8}")
    all_actions = sorted(set(tr_action_dist.index) | set(te_action_dist.index))
    for c in all_actions:
        tr_v = tr_action_dist.get(c, 0)
        te_v = te_action_dist.get(c, 0)
        print(f"  {c:>6}  {tr_v:>8.4f}  {te_v:>8.4f}  {te_v-tr_v:>+8.4f}")

    print("\n  Point distribution (train last-stroke vs test last-VISIBLE-stroke):")
    print(f"  {'class':>6}  {'train':>8}  {'test':>8}  {'diff':>8}")
    all_points = sorted(set(tr_point_dist.index) | set(te_point_dist.index))
    for c in all_points:
        tr_v = tr_point_dist.get(c, 0)
        te_v = te_point_dist.get(c, 0)
        print(f"  {c:>6}  {tr_v:>8.4f}  {te_v:>8.4f}  {te_v-tr_v:>+8.4f}")

    # ======================================================================
    # Decomposition Summary
    # ======================================================================
    print("\n" + "=" * 78)
    print("DECOMPOSITION SUMMARY")
    print("=" * 78)
    print(f"\n  Observed OOF→LB gap: 0.4816 - 0.4285 = -0.0531")
    print(f"\n  Hypotheses ranked by likely contribution:")
    print(f"    H_class17  (forced labels):   Δ_CV = {cv_forced-cv_default:+.4f}")
    print(f"    H_player   (overlap):         see above stats")
    print(f"    H_match    (match overlap):   {100*len(te_matches & tr_matches)/len(te_matches):.0f}% test matches in train")
    print(f"    H_class_dist:                 see distribution table")
    print(f"    H_residual (length already ruled out, ~+0.000)")


if __name__ == "__main__":
    main()
