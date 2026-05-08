"""
Stage A probe: AV2 cross-rally aggregate player stats under MATCH-DISJOINT protocol.

A v2 (rally-KFold) was killed by player-overlap leakage (LB -0.024 vs OOF +0.004).
Plan E proposed AE-compressed stats. Before investing 14-20h, test if RAW stats
even help under match-disjoint.

Decision criteria:
- LGB-only OOF action F1 (with stats) vs base (without stats)
- If Δ < +0.002 → KILL Plan E (raw stats don't help → AE compression can't help)
- If Δ > +0.005 → consider Plan E with AE/PCA comparison
- Middle: ambiguous

Cost: ~30 min (LGB only, match-disjoint, reuse existing LSTM OOF for ensemble F1).
"""
import os, sys, time
import numpy as np
import pandas as pd

# Force AV2 + match-disjoint before importing train_v5z
os.environ["V5Z_AV2"] = "1"
os.environ["V5Z_MATCH_DISJOINT"] = "1"
SEED = int(os.environ.get("V5Z_SEED", "42"))

sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

from train_v5z import (
    prepare_samples, build_lgb_features, compute_player_stats,
    train_df, test_df, N_ACTION, N_POINT, N_FOLDS, log
)


def train_lgb(X_tr, y_tr, X_va, y_va, n_class, name, num_leaves=127):
    params = {
        "objective": "multiclass",
        "num_class": n_class,
        "metric": "multi_logloss",
        "num_leaves": num_leaves,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "class_weight": "balanced",
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=500,
                  valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = m.predict(X_va)
    f1 = f1_score(y_va, pred.argmax(axis=1), average="macro", zero_division=0)
    return m, pred, f1


def main():
    log("="*78)
    log(f"PROBE AV2 — MATCH-DISJOINT (seed={SEED})")
    log("Goal: do cross-rally aggregate player_stats add LGB OOF F1 under match-disjoint?")
    log("="*78)

    # Prepare samples (V5Z_AV2 already set in env)
    log("\nBuilding training samples + LGB features...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_arr = np.array([uid_to_match[u] for u in uid_list])

    kf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(kf.split(uid_list, uid_arr, groups=match_arr))
    log(f"  Match-disjoint folds: " + " ".join([f"f{i+1}={len(set(match_arr[va]))}" for i,(_,va) in enumerate(fold_splits)]))

    # Per-fold training with per-fold player_stats
    log("\n" + "="*78)
    log(f"Full {N_FOLDS}-fold LGB-only training (action+point) with V2 features")
    log("="*78)

    f1_a_folds, f1_p_folds = [], []
    f1_a_folds_no_v2, f1_p_folds_no_v2 = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(fold_splits):
        log(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        # Per-fold player_stats from train fold uids
        t0 = time.time()
        pstats, cold, seen = compute_player_stats(train_df, restrict_uids=tr_uids)
        log(f"  player_stats: {len(pstats)} players, {time.time()-t0:.1f}s")

        # Build features WITH AV2
        Xall, ya_all, yp_all, uids_all, _ = build_lgb_features(
            train_df, is_train=True, augment=True,
            player_stats=pstats, cold_marginals=cold, seen_players=seen)

        # uid -> indices
        uid2idx = {}
        for i, u in enumerate(uids_all):
            uid2idx.setdefault(u, []).append(i)
        tr_idx = [i for u in tr_uids for i in uid2idx.get(u, [])]
        va_idx = [i for u in va_uids for i in uid2idx.get(u, [])]
        X_tr, X_va = Xall.iloc[tr_idx], Xall.iloc[va_idx]
        ya_tr = [ya_all[i] for i in tr_idx]; ya_va = [ya_all[i] for i in va_idx]
        yp_tr = [yp_all[i] for i in tr_idx]; yp_va = [yp_all[i] for i in va_idx]

        log(f"  Training LGB action (with V2 features, {X_tr.shape[1]} cols)...")
        _, _, fa = train_lgb(X_tr, ya_tr, X_va, ya_va, N_ACTION, "action")
        log(f"  Training LGB point (with V2 features)...")
        _, _, fp = train_lgb(X_tr, yp_tr, X_va, yp_va, N_POINT, "point")
        f1_a_folds.append(fa); f1_p_folds.append(fp)
        log(f"  Fold {fold+1}: action F1={fa:.4f}  point F1={fp:.4f}  (V2 features)")

        # Repeat WITHOUT V2 features (drop AV2 cols)
        v2_cols = [c for c in Xall.columns if c.startswith("srv_") or c.startswith("rcv_")]
        X_tr_nov2 = X_tr.drop(columns=v2_cols)
        X_va_nov2 = X_va.drop(columns=v2_cols)
        log(f"  Training LGB action (without V2, {X_tr_nov2.shape[1]} cols, dropped {len(v2_cols)})...")
        _, _, fa0 = train_lgb(X_tr_nov2, ya_tr, X_va_nov2, ya_va, N_ACTION, "action")
        _, _, fp0 = train_lgb(X_tr_nov2, yp_tr, X_va_nov2, yp_va, N_POINT, "point")
        f1_a_folds_no_v2.append(fa0); f1_p_folds_no_v2.append(fp0)
        log(f"  Fold {fold+1}: action F1={fa0:.4f}  point F1={fp0:.4f}  (NO V2)")
        log(f"  Δ        action +{fa-fa0:+.4f}  point +{fp-fp0:+.4f}")

    log("\n" + "="*78)
    log(f"5-FOLD AVG: action(V2)={np.mean(f1_a_folds):.4f}  point(V2)={np.mean(f1_p_folds):.4f}")
    log(f"5-FOLD AVG: action(no V2)={np.mean(f1_a_folds_no_v2):.4f}  point(no V2)={np.mean(f1_p_folds_no_v2):.4f}")
    delta_a = np.mean(f1_a_folds) - np.mean(f1_a_folds_no_v2)
    delta_p = np.mean(f1_p_folds) - np.mean(f1_p_folds_no_v2)
    log(f"5-FOLD AVG Δ: action {delta_a:+.4f}  point {delta_p:+.4f}")
    log(f"5-FOLD AVG Δ blended (0.5+0.5): {(delta_a + delta_p)/2:+.4f}")

    log("\n" + "="*78)
    log("DECISION CRITERIA")
    log("="*78)
    blend = (delta_a + delta_p) / 2
    if blend < 0.002:
        log(f"  Blended Δ = {blend:+.4f} < +0.002 → **RAW stats don't help on LGB**")
        log(f"  → AE compression of these stats will NOT help (Attack 7 confirmed)")
        log(f"  → **KILL Plan E**")
    elif blend < 0.005:
        log(f"  Blended Δ = {blend:+.4f} (0.002-0.005) → marginal raw stats lift")
        log(f"  → AE 可能 marginal lift，但 +0.010 ship gate 機率仍低")
        log(f"  → 不直接 KILL，但建議降版 Plan E（更小範圍試）")
    else:
        log(f"  Blended Δ = {blend:+.4f} > +0.005 → raw stats DO help")
        log(f"  → AE compression has space to compete (Plan E worth pursuing)")
        log(f"  → 繼續 stage A probe phase 2 (PCA-5 vs AE-16 直接比較)")


if __name__ == "__main__":
    main()
