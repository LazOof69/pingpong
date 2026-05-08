#!/usr/bin/env python3
"""
Phase 4-D: LGB-quick + player ELO + cross features.

Adds to LGB-quick (39 features):
- server_elo, receiver_elo (per-fold computed from train rallies in chronological order)
- elo_diff, elo_avg
- srv_winrate × score_diff cross
- srv_winrate × ctx_len cross
- score_diff × ctx_len cross

Match-disjoint: ELO computed per fold from non-holdout pool (no leak).

Usage:
    python3 src/train/predict_sgp_lgb_elo.py

Outputs:
    artifacts/sgp_pred_NEWTEST_lgb_elo.npz
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ACTION_GROUPS = {}
for a in [1, 2, 3, 4, 5, 6, 7]:
    ACTION_GROUPS[a] = 1
for a in [8, 9, 10, 11]:
    ACTION_GROUPS[a] = 2
for a in [12, 13, 14]:
    ACTION_GROUPS[a] = 3
for a in [15, 16, 17, 18]:
    ACTION_GROUPS[a] = 4
ACTION_GROUPS[0] = 0

POINT_DEPTH = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
POINT_SIDE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}


def compute_elo(rally_meta, k_factor=24, init_rating=1500):
    """ELO from train fold rallies in rally_uid order."""
    ratings = {}
    sorted_meta = rally_meta.sort_values("rally_uid")
    for _, row in sorted_meta.iterrows():
        srv = row["gamePlayerId"]
        rcv = row["gamePlayerOtherId"]
        sgp = row["serverGetPoint"]
        r_s = ratings.get(srv, init_rating)
        r_r = ratings.get(rcv, init_rating)
        e_s = 1.0 / (1.0 + 10 ** ((r_r - r_s) / 400.0))
        a_s = float(sgp)
        ratings[srv] = r_s + k_factor * (a_s - e_s)
        ratings[rcv] = r_r + k_factor * ((1 - a_s) - (1 - e_s))
    return ratings


def bayesian_player_stats(rally_meta, kappa=50):
    prior_global = float(rally_meta["serverGetPoint"].mean())
    prior_sex = {
        s: float(rally_meta[rally_meta["sex"] == s]["serverGetPoint"].mean())
        for s in rally_meta["sex"].unique()
    }
    srv = rally_meta.groupby("gamePlayerId").agg(
        n=("serverGetPoint", "size"), w=("serverGetPoint", "sum"), sex=("sex", "first")
    )
    rcv = rally_meta.groupby("gamePlayerOtherId").agg(
        n=("serverGetPoint", "size"), w=("serverGetPoint", "sum"), sex=("sex", "first")
    )
    srv["winrate"] = (srv["w"] + kappa * srv["sex"].map(prior_sex)) / (srv["n"] + kappa)
    rcv["loserate"] = (rcv["w"] + kappa * rcv["sex"].map(prior_sex)) / (rcv["n"] + kappa)
    return srv["winrate"].to_dict(), srv["n"].to_dict(), \
           rcv["loserate"].to_dict(), rcv["n"].to_dict(), \
           prior_sex, prior_global


def make_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, elo, init_elo, known_matches=None):
    first = rows[0]
    visible = rows[:k]
    last_visible = visible[-1]
    sex = first["sex"]
    sex_p = prior_sex.get(sex, prior_global)
    srv_pid = first["gamePlayerId"]
    rcv_pid = first["gamePlayerOtherId"]
    srv_winrate = srv_w.get(srv_pid, sex_p)
    rcv_loserate = rcv_l.get(rcv_pid, sex_p)
    srv_elo = elo.get(srv_pid, init_elo)
    rcv_elo = elo.get(rcv_pid, init_elo)
    score_diff_start = first["scoreSelf"] - first["scoreOther"]

    f = {
        "ctx_len": k,
        "sex": sex,
        "numberGame": first["numberGame"],
        "score_self_start": first["scoreSelf"],
        "score_other_start": first["scoreOther"],
        "score_diff_start": score_diff_start,
        "score_sum_start": first["scoreSelf"] + first["scoreOther"],
        "is_deuce_start": int(first["scoreSelf"] >= 10 and first["scoreOther"] >= 10),
        "srv_winrate": srv_winrate,
        "srv_n_obs": srv_n.get(srv_pid, 0),
        "rcv_loserate": rcv_loserate,
        "rcv_n_obs": rcv_n.get(rcv_pid, 0),
        "srv_seen": int(srv_pid in srv_w),
        "rcv_seen": int(rcv_pid in rcv_l),
        "match_seen": int(known_matches is not None and first["match"] in known_matches),
        # ELO features (NEW)
        "srv_elo": srv_elo,
        "rcv_elo": rcv_elo,
        "elo_diff": srv_elo - rcv_elo,
        "elo_avg": (srv_elo + rcv_elo) / 2,
        # Last visible stroke
        "last_strikeId": last_visible["strikeId"],
        "last_handId": last_visible["handId"],
        "last_strengthId": last_visible["strengthId"],
        "last_spinId": last_visible["spinId"],
        "last_pointId": last_visible["pointId"],
        "last_actionId": last_visible["actionId"],
        "last_positionId": last_visible["positionId"],
        "last_action_group": ACTION_GROUPS.get(last_visible["actionId"], 0),
        "last_point_depth": POINT_DEPTH.get(last_visible["pointId"], 0),
        "last_point_side": POINT_SIDE.get(last_visible["pointId"], 0),
        "last_score_diff": last_visible["scoreSelf"] - last_visible["scoreOther"],
        # Cross features (NEW)
        "winrate_x_score_diff": srv_winrate * score_diff_start,
        "winrate_x_ctx_len": srv_winrate * k,
        "score_diff_x_ctx_len": score_diff_start * k,
        "elo_diff_x_ctx_len": (srv_elo - rcv_elo) * k,
    }

    groups = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible]
    f["n_attacks"] = sum(1 for g in groups if g == 1)
    f["n_controls"] = sum(1 for g in groups if g == 2)
    f["n_defenses"] = sum(1 for g in groups if g == 3)
    f["n_serves"] = sum(1 for g in groups if g == 4)
    srv_groups = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible if r["strikeNumber"] % 2 == 1]
    rcv_groups = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible if r["strikeNumber"] % 2 == 0]
    f["n_srv_attacks"] = sum(1 for g in srv_groups if g == 1)
    f["n_rcv_attacks"] = sum(1 for g in rcv_groups if g == 1)
    f["n_srv_strokes"] = len(srv_groups)
    f["n_rcv_strokes"] = len(rcv_groups)

    if k >= 2:
        prev = visible[-2]
        f["prev_actionId"] = prev["actionId"]
        f["prev_pointId"] = prev["pointId"]
        f["prev_action_group"] = ACTION_GROUPS.get(prev["actionId"], 0)
        f["prev_handId"] = prev["handId"]
        f["prev_spinId"] = prev["spinId"]
    else:
        f["prev_actionId"] = -1
        f["prev_pointId"] = -1
        f["prev_action_group"] = -1
        f["prev_handId"] = -1
        f["prev_spinId"] = -1

    return f


def build_train_features(df, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, elo, init_elo, known_matches=None, max_prefixes=5, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    feats, uids, sgp = [], [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        if N < 2:
            ks = [1]
        else:
            all_ks = list(range(1, N))
            ks = sorted(rng.choice(all_ks, size=min(max_prefixes, len(all_ks)), replace=False).tolist())
        for k in ks:
            f = make_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, elo, init_elo, known_matches)
            feats.append(f)
            uids.append(uid)
            sgp.append(int(rows[0]["serverGetPoint"]))
    return pd.DataFrame(feats), np.array(uids), np.array(sgp)


def build_test_features(df, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, elo, init_elo, known_matches=None):
    feats, uids = [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        f = make_feats(rows, N, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, elo, init_elo, known_matches)
        feats.append(f)
        uids.append(uid)
    return pd.DataFrame(feats), np.array(uids)


def main():
    print("=" * 70)
    print("Phase 4-D LGB + player ELO + cross features")
    print("=" * 70)

    test_csv = os.environ.get("V5Z_TEST_CSV", "data/test_new.csv")
    kappa = int(os.environ.get("V5Z_KAPPA", "50"))
    elo_k = float(os.environ.get("V5Z_ELO_K", "24"))
    init_elo = float(os.environ.get("V5Z_INIT_ELO", "1500"))
    n_folds = int(os.environ.get("V5Z_N_FOLDS", "5"))
    max_prefix = int(os.environ.get("V5Z_MAX_PREFIX", "5"))
    print(f"κ={kappa}, ELO K={elo_k}, init={init_elo}, folds={n_folds}, max_prefix={max_prefix}")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv(test_csv)
    print(f"Train: {len(train_df)} rows / {train_df['rally_uid'].nunique()} rallies")
    print(f"Test:  {len(test_df)} rows / {test_df['rally_uid'].nunique()} rallies")

    rally_meta_full = train_df.groupby("rally_uid").first()[
        ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex", "match"]
    ].reset_index()
    train_matches = set(rally_meta_full["match"].unique())

    uid_to_idx = {u: i for i, u in enumerate(rally_meta_full["rally_uid"].values)}
    uid_arr = rally_meta_full["serverGetPoint"].values
    match_arr = rally_meta_full["match"].values
    sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=42)

    oof_pred_per_rally = np.zeros(len(rally_meta_full))
    oof_count_per_rally = np.zeros(len(rally_meta_full))
    oof_label = rally_meta_full["serverGetPoint"].values.astype(np.float64)
    fold_models = []
    feat_cols_ref = None

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(rally_meta_full, uid_arr, groups=match_arr)):
        tr_meta = rally_meta_full.iloc[tr_idx]
        va_meta = rally_meta_full.iloc[va_idx]
        tr_uids = set(tr_meta["rally_uid"])
        va_uids = set(va_meta["rally_uid"])

        srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global = bayesian_player_stats(tr_meta, kappa)
        elo = compute_elo(tr_meta, k_factor=elo_k, init_rating=init_elo)

        tr_rally_df = train_df[train_df["rally_uid"].isin(tr_uids)]
        va_rally_df = train_df[train_df["rally_uid"].isin(va_uids)]

        rng_tr = np.random.RandomState(42 + fold)
        rng_va = np.random.RandomState(1000 + fold)
        X_tr, uid_tr, y_tr = build_train_features(tr_rally_df, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global, elo, init_elo, max_prefixes=max_prefix, rng=rng_tr)
        X_va, uid_va, y_va = build_train_features(va_rally_df, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global, elo, init_elo, max_prefixes=max_prefix, rng=rng_va)

        if feat_cols_ref is None:
            feat_cols_ref = list(X_tr.columns)

        print(f"\nFold {fold+1}/{n_folds}: Tr {len(X_tr)} samples, Va {len(X_va)}, features={X_tr.shape[1]}")

        model = lgb.train(
            {
                "objective": "binary", "metric": "auc",
                "learning_rate": 0.03, "num_leaves": 31,
                "min_data_in_leaf": 50, "lambda_l2": 1.0,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "seed": 42, "verbose": -1,
            },
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=600,
            valid_sets=[lgb.Dataset(X_va, label=y_va)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )

        va_pred = model.predict(X_va)
        prefix_auc = roc_auc_score(y_va, va_pred)
        for u, p in zip(uid_va, va_pred):
            i = uid_to_idx[u]
            oof_pred_per_rally[i] += p
            oof_count_per_rally[i] += 1
        fi = model.feature_importance(importance_type="gain")
        cols = list(X_tr.columns)
        top10 = sorted(zip(cols, fi), key=lambda x: -x[1])[:10]
        elo_in_top = sum(1 for n, _ in top10 if "elo" in n.lower())
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, ELO/cross-in-top10: {elo_in_top}")
        fold_models.append((model, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global, elo))

    oof_pred_rally = oof_pred_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label[rally_mask], oof_pred_rally[rally_mask])
    print(f"\n{'='*70}")
    print(f"OOF AUC: {overall_auc:.4f}")

    fi_last = pd.DataFrame({
        "feature": feat_cols_ref,
        "importance": fold_models[-1][0].feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\nTop 20 features by gain (last fold):")
    print(fi_last.head(20).to_string(index=False))

    # Test prediction
    print("\n--- Test prediction ---")
    srv_w_full, srv_n_full, rcv_l_full, rcv_n_full, prior_sex_full, prior_global_full = bayesian_player_stats(rally_meta_full, kappa)
    elo_full = compute_elo(rally_meta_full, k_factor=elo_k, init_rating=init_elo)

    test_pred = np.zeros(test_df["rally_uid"].nunique())
    test_uids_ref = None
    for model, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f, elo_f in fold_models:
        X_test, uid_test = build_test_features(test_df, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f, elo_f, init_elo, known_matches=train_matches)
        if test_uids_ref is None:
            test_uids_ref = uid_test
        test_pred += model.predict(X_test) / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_lgb_elo.npz"
    np.savez(out_path,
             rally_uids=test_uids_ref,
             sgp_pred=test_pred,
             method=f"lgb_elo_kappa{kappa}_elok{elo_k}",
             holdout_auc=overall_auc,
             oof_pred=oof_pred_rally,
             oof_label=oof_label,
             feature_names=np.array(feat_cols_ref))
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
