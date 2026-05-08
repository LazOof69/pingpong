#!/usr/bin/env python3
"""
Phase 2: serverGetPoint predictor for test_new.csv.

Bayesian-smoothed per-player win rate, averaged over server-side and receiver-side
estimators (P(server wins) ≈ avg(winrate_as_srv, 1 - winrate_as_rcv)).

Holdout: match-disjoint K-fold over train; per-fold winrates computed from
non-holdout pool only (avoids §2.10 A v2-style player-overlap leakage).

Usage:
    V5Z_KAPPA_GRID="5,10,20,50" python3 src/train/predict_sgp.py
    V5Z_TEST_CSV=data/test_new.csv  (default data/test_new.csv if absent)

Outputs:
    artifacts/sgp_pred_NEWTEST.npz  (rally_uids, sgp_pred, method)
    artifacts/sgp_holdout_curve.npz (kappa_grid, auc_grid)
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def per_player_stats(rally_meta, kappa, prior_sex):
    """Compute Bayesian-smoothed winrate_as_srv and winrate_as_rcv per player."""
    srv = rally_meta.groupby("gamePlayerId").agg(
        n_srv=("serverGetPoint", "size"),
        w_srv=("serverGetPoint", "sum"),
        sex=("sex", "first"),
    )
    rcv = rally_meta.groupby("gamePlayerOtherId").agg(
        n_rcv=("serverGetPoint", "size"),
        # As receiver, "loss" = serverGetPoint=1 (server won = receiver lost)
        l_rcv=("serverGetPoint", "sum"),
        sex=("sex", "first"),
    )

    srv["winrate_srv"] = (srv["w_srv"] + kappa * srv["sex"].map(prior_sex)) / (srv["n_srv"] + kappa)
    rcv["loserate_rcv"] = (rcv["l_rcv"] + kappa * rcv["sex"].map(prior_sex)) / (rcv["n_rcv"] + kappa)
    return srv["winrate_srv"].to_dict(), rcv["loserate_rcv"].to_dict()


def predict_rally(rally_meta_test, srv_dict, rcv_dict, prior_sex, prior_global):
    """
    For each test rally, predict P(serverGetPoint=1) =
        avg(server's winrate_srv, 1 - receiver's loserate_rcv).
    Cold-start: sex-marginal fallback for missing player.
    """
    preds = []
    for _, row in rally_meta_test.iterrows():
        srv_pid = row["gamePlayerId"]
        rcv_pid = row["gamePlayerOtherId"]
        sex = row["sex"]
        sex_p = prior_sex.get(sex, prior_global)

        srv_p = srv_dict.get(srv_pid, sex_p)
        # Receiver loserate p means P(receiver loses) = P(server wins). Symmetric estimator.
        rcv_loserate = rcv_dict.get(rcv_pid, sex_p)
        preds.append(0.5 * srv_p + 0.5 * rcv_loserate)
    return np.array(preds, dtype=np.float64)


def evaluate_kappa(train_df, kappa, n_folds=5):
    """Match-disjoint holdout AUC with proper non-leak winrate computation."""
    rally_meta_full = train_df.groupby("rally_uid").first()[
        ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex", "match"]
    ].reset_index()

    prior_global = float(rally_meta_full["serverGetPoint"].mean())
    prior_sex = {
        s: float(rally_meta_full[rally_meta_full["sex"] == s]["serverGetPoint"].mean())
        for s in rally_meta_full["sex"].unique()
    }

    uid_arr = rally_meta_full["serverGetPoint"].values
    match_arr = rally_meta_full["match"].values

    sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(rally_meta_full))
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(rally_meta_full, uid_arr, groups=match_arr)):
        tr_meta = rally_meta_full.iloc[tr_idx]
        va_meta = rally_meta_full.iloc[va_idx]
        # Compute winrates ONLY from training fold (no holdout leakage)
        srv_dict, rcv_dict = per_player_stats(tr_meta, kappa, prior_sex)
        oof_pred[va_idx] = predict_rally(va_meta, srv_dict, rcv_dict, prior_sex, prior_global)

    auc = roc_auc_score(rally_meta_full["serverGetPoint"].values, oof_pred)
    return auc, oof_pred, prior_sex, prior_global


def main():
    print("=" * 70)
    print("Phase 2: serverGetPoint Bayesian Predictor")
    print("=" * 70)

    test_csv = os.environ.get("V5Z_TEST_CSV", "data/test_new.csv")
    kappa_grid_str = os.environ.get("V5Z_KAPPA_GRID", "5,10,20,50")
    kappa_grid = [int(k) for k in kappa_grid_str.split(",")]

    print(f"Test CSV: {test_csv}")
    print(f"κ grid: {kappa_grid}")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv(test_csv)
    print(f"Train: {train_df.shape[0]} rows, {train_df['rally_uid'].nunique()} rallies")
    print(f"Test:  {test_df.shape[0]} rows, {test_df['rally_uid'].nunique()} rallies")

    print("\n--- κ grid search (match-disjoint holdout AUC) ---")
    aucs = []
    best_kappa, best_auc = None, -1
    for k in kappa_grid:
        auc, _, _, _ = evaluate_kappa(train_df, k)
        aucs.append(auc)
        marker = "  ★" if auc > best_auc else ""
        print(f"  κ={k:>3}: holdout AUC = {auc:.4f}{marker}")
        if auc > best_auc:
            best_auc, best_kappa = auc, k

    print(f"\nBest κ = {best_kappa} → AUC = {best_auc:.4f}")

    np.savez("artifacts/sgp_holdout_curve.npz",
             kappa_grid=np.array(kappa_grid),
             auc_grid=np.array(aucs))

    if best_auc < 0.55:
        print(f"\n⚠ Best AUC {best_auc:.4f} < 0.55 ship gate. Falling back to 0.5 placeholder.")
        method = "placeholder_0.5"
        train_rally_meta = train_df.groupby("rally_uid").first()[["gamePlayerId", "gamePlayerOtherId", "sex"]]
        prior_global = float(train_df.groupby("rally_uid").first()["serverGetPoint"].mean())
        # still output a real array for downstream consumption
        test_rally_meta = test_df.groupby("rally_uid").first().reset_index()
        sgp_pred = np.full(len(test_rally_meta), 0.5, dtype=np.float64)
        rally_uids = test_rally_meta["rally_uid"].values
    else:
        print(f"\n--- Production: train winrates with best κ={best_kappa} on FULL train, predict test ---")
        rally_meta_full = train_df.groupby("rally_uid").first()[
            ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex"]
        ].reset_index()
        prior_global = float(rally_meta_full["serverGetPoint"].mean())
        prior_sex = {
            s: float(rally_meta_full[rally_meta_full["sex"] == s]["serverGetPoint"].mean())
            for s in rally_meta_full["sex"].unique()
        }
        srv_dict, rcv_dict = per_player_stats(rally_meta_full, best_kappa, prior_sex)

        # Coverage check on test
        test_rally_meta = test_df.groupby("rally_uid").first().reset_index()
        srv_seen = test_rally_meta["gamePlayerId"].isin(srv_dict).sum()
        rcv_seen = test_rally_meta["gamePlayerOtherId"].isin(rcv_dict).sum()
        n = len(test_rally_meta)
        print(f"  Test rally coverage: server seen {srv_seen}/{n} ({srv_seen/n*100:.1f}%); receiver seen {rcv_seen}/{n} ({rcv_seen/n*100:.1f}%)")

        sgp_pred = predict_rally(test_rally_meta, srv_dict, rcv_dict, prior_sex, prior_global)
        rally_uids = test_rally_meta["rally_uid"].values
        method = f"bayesian_kappa{best_kappa}_avg_srv_rcv"

        print(f"  Pred mean: {sgp_pred.mean():.4f}  std: {sgp_pred.std():.4f}  min: {sgp_pred.min():.4f}  max: {sgp_pred.max():.4f}")

    out_path = "artifacts/sgp_pred_NEWTEST.npz"
    np.savez(out_path,
             rally_uids=rally_uids,
             sgp_pred=sgp_pred,
             method=method,
             holdout_auc=best_auc,
             best_kappa=best_kappa if best_auc >= 0.55 else -1)
    print(f"\n→ Saved {out_path}")
    print(f"   {len(sgp_pred)} predictions, method={method}")

    print("\n--- Acceptance gate check ---")
    if best_auc >= 0.65:
        print(f"  AUC {best_auc:.4f} ≥ 0.65 → consider LGB upgrade (separate phase)")
    elif best_auc >= 0.55:
        print(f"  AUC {best_auc:.4f} in [0.55, 0.65) → ship Bayesian alone")
    else:
        print(f"  AUC {best_auc:.4f} < 0.55 → use 0.5 placeholder")


if __name__ == "__main__":
    main()
