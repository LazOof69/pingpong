#!/usr/bin/env python3
"""
Conditional Ensemble (C) — route test predictions by player seen/unseen status.

Rule:
  Both server AND receiver in train → use A v2 prediction (player features help)
  At least one unseen        → use baseline V5Z prediction (no player_stats reliance)

Rationale: A v2's LB regression -0.024 suggests player_stats overfit on OOF and
fail catastrophically on the 46% test rallies with unseen players. Routing those
to baseline V5Z (which uses LSTM player_embedding fallback only) bounds the loss.

Output: submissions/submission_v5z_conditional_bag2.csv
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd


def main():
    print("=" * 78)
    print("Conditional Ensemble — route by player seen/unseen")
    print("=" * 78)

    # Load both ship CSVs (already calibrated, ready predictions)
    av2 = pd.read_csv("submissions/submission_v5z_av2_advcal_bag2.csv")
    base = pd.read_csv("submissions/submission_v5z_advcal_bag2.csv")

    # Sanity: same row count and same rally_uid set
    assert len(av2) == len(base) == 1236, f"got {len(av2)} / {len(base)}"
    av2 = av2.sort_values("rally_uid").reset_index(drop=True)
    base = base.sort_values("rally_uid").reset_index(drop=True)
    assert (av2["rally_uid"] == base["rally_uid"]).all(), "uid mismatch"

    # Determine seen status per test rally
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_players = set(train_df["gamePlayerId"].unique()) | set(train_df["gamePlayerOtherId"].unique())

    test_meta = test_df.groupby("rally_uid").first()[["gamePlayerId", "gamePlayerOtherId"]].reset_index()
    test_meta["srv_seen"] = test_meta["gamePlayerId"].isin(train_players)
    test_meta["rcv_seen"] = test_meta["gamePlayerOtherId"].isin(train_players)
    test_meta["both_seen"] = test_meta["srv_seen"] & test_meta["rcv_seen"]

    # Stats
    n_both = test_meta["both_seen"].sum()
    n_partial = ((test_meta["srv_seen"] | test_meta["rcv_seen"]) & ~test_meta["both_seen"]).sum()
    n_none = (~test_meta["srv_seen"] & ~test_meta["rcv_seen"]).sum()
    print(f"\n  test rallies (1236):")
    print(f"    both seen      : {n_both:4d} ({100*n_both/1236:.1f}%) → use AV2")
    print(f"    one-side unseen: {n_partial:4d} ({100*n_partial/1236:.1f}%) → use baseline")
    print(f"    both unseen    : {n_none:4d} ({100*n_none/1236:.1f}%) → use baseline")
    print(f"    total to AV2   : {n_both:4d} ({100*n_both/1236:.1f}%)")
    print(f"    total to base  : {n_partial+n_none:4d} ({100*(n_partial+n_none)/1236:.1f}%)")

    # Sort to align with av2/base order
    test_meta = test_meta.sort_values("rally_uid").reset_index(drop=True)
    assert (test_meta["rally_uid"] == av2["rally_uid"]).all(), "uid order mismatch"

    # Route predictions
    use_av2 = test_meta["both_seen"].values
    out = av2.copy()
    cols_to_route = ["actionId", "pointId"]
    for c in cols_to_route:
        out.loc[~use_av2, c] = base.loc[~use_av2, c].values
    # serverGetPoint same in both (rally-level constant), keep as is

    # Sanity check
    diff_count = (out["actionId"] != av2["actionId"]).sum() + (out["pointId"] != av2["pointId"]).sum()
    expected_changes = (~use_av2).sum() * 2  # action + point
    actual_changes = ((out["actionId"] != av2["actionId"]) | (out["pointId"] != av2["pointId"])).sum()
    print(f"\n  sanity: {actual_changes} rallies changed (expected ≤ {(~use_av2).sum()})")

    # Print first few diff for spot check
    mask_changed = ~use_av2
    sample = test_meta[mask_changed].head(5)
    print(f"\n  sample routing (first 5 unseen):")
    for _, r in sample.iterrows():
        uid = r["rally_uid"]
        a_av2, p_av2 = av2[av2["rally_uid"] == uid].iloc[0][["actionId", "pointId"]]
        a_b, p_b = base[base["rally_uid"] == uid].iloc[0][["actionId", "pointId"]]
        print(f"    uid={uid}  srv={r['gamePlayerId']}({'S' if r['srv_seen'] else 'U'}) rcv={r['gamePlayerOtherId']}({'S' if r['rcv_seen'] else 'U'})  AV2=({a_av2},{p_av2})→base=({a_b},{p_b})")

    out_path = "submissions/submission_v5z_conditional_bag2.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(out)} rows)")
    print("\nExpected LB:")
    print("  Best case (AV2 helps seen, baseline OK on unseen): ~0.4290+")
    print("  Most likely (AV2 ≈ baseline on seen): ~0.4285 (back to baseline)")
    print("  Worst case (AV2 hurts seen too): < 0.4285")


if __name__ == "__main__":
    main()
