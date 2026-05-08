#!/usr/bin/env python3
"""
T2: T1 (LGB multi-class on R) + extended trajectory / dominance / reachability features.

New features over T1:
- Continuous xy coordinates (from POINT_DEPTH × POINT_SIDE)
- Trajectory: last_dx, last_dy, last_distance, last_angle, total_path_length
- Court coverage: unique_zones, corner_ratio, diagonal_ratio
- Dominance: server_attack_imbalance, max_attack_chain, last3_aggressor_score
- Reachability: position_x_pointId crosses
- Recent drift: x-side change in last 3 strokes
- Score pressure × dominance interaction

Same prefix-augmented match-disjoint protocol as T1.

Usage:
    python3 src/train/predict_sgp_t2_features.py
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ACTION_GROUPS = {a: g for a, g in [
    *[(i, 1) for i in [1,2,3,4,5,6,7]],
    *[(i, 2) for i in [8,9,10,11]],
    *[(i, 3) for i in [12,13,14]],
    *[(i, 4) for i in [15,16,17,18]],
    (0, 0)
]}
POINT_DEPTH = {0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3}
POINT_SIDE  = {0:0,1:1,2:2,3:3,4:1,5:2,6:3,7:1,8:2,9:3}
SIDE_X = {0:0, 1:-1, 2:0, 3:1}    # -1 left / 0 mid / 1 right
DEPTH_Y = {0:0, 1:1, 2:2, 3:3}    # 1 close / 2 mid / 3 far
CORNER_ZONES = {1, 3, 7, 9}        # close-fore, close-back, far-fore, far-back


def get_xy(point_id):
    return SIDE_X[POINT_SIDE.get(point_id, 0)], DEPTH_Y[POINT_DEPTH.get(point_id, 0)]


def bayesian_player_stats(rally_meta, kappa=50):
    prior_global = float(rally_meta["serverGetPoint"].mean())
    prior_sex = {s: float(rally_meta[rally_meta["sex"]==s]["serverGetPoint"].mean())
                 for s in rally_meta["sex"].unique()}
    srv = rally_meta.groupby("gamePlayerId").agg(n=("serverGetPoint","size"),
                                                  w=("serverGetPoint","sum"),
                                                  sex=("sex","first"))
    rcv = rally_meta.groupby("gamePlayerOtherId").agg(n=("serverGetPoint","size"),
                                                      w=("serverGetPoint","sum"),
                                                      sex=("sex","first"))
    srv["winrate"] = (srv["w"] + kappa*srv["sex"].map(prior_sex)) / (srv["n"]+kappa)
    rcv["loserate"] = (rcv["w"] + kappa*rcv["sex"].map(prior_sex)) / (rcv["n"]+kappa)
    return srv["winrate"].to_dict(), srv["n"].to_dict(), \
           rcv["loserate"].to_dict(), rcv["n"].to_dict(), \
           prior_sex, prior_global


def parity_bayes_uniform(L_max, P_N_train):
    pred = {}
    for l in range(1, L_max + 5):
        cands = [n for n in P_N_train if n > l]
        if not cands: pred[l] = 0.5; continue
        w = np.array([float(P_N_train[n])/(n-1) for n in cands])
        w = w / w.sum()
        s = np.array([1 - (n%2) for n in cands], dtype=float)
        pred[l] = float((w*s).sum())
    return pred


def make_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches=None):
    first = rows[0]; visible = rows[:k]; last = visible[-1]
    sex = first["sex"]; sex_p = prior_sex.get(sex, prior_global)

    # --- T1 base features (39) ---
    f = {
        "ctx_len": k,
        "sex": sex,
        "numberGame": first["numberGame"],
        "score_self_start": first["scoreSelf"],
        "score_other_start": first["scoreOther"],
        "score_diff_start": first["scoreSelf"] - first["scoreOther"],
        "score_sum_start": first["scoreSelf"] + first["scoreOther"],
        "is_deuce_start": int(first["scoreSelf"]>=10 and first["scoreOther"]>=10),
        "srv_winrate": srv_w.get(first["gamePlayerId"], sex_p),
        "srv_n_obs": srv_n.get(first["gamePlayerId"], 0),
        "rcv_loserate": rcv_l.get(first["gamePlayerOtherId"], sex_p),
        "rcv_n_obs": rcv_n.get(first["gamePlayerOtherId"], 0),
        "srv_seen": int(first["gamePlayerId"] in srv_w),
        "rcv_seen": int(first["gamePlayerOtherId"] in rcv_l),
        "match_seen": int(known_matches is not None and first["match"] in known_matches),
        "last_strikeId": last["strikeId"],
        "last_handId": last["handId"],
        "last_strengthId": last["strengthId"],
        "last_spinId": last["spinId"],
        "last_pointId": last["pointId"],
        "last_actionId": last["actionId"],
        "last_positionId": last["positionId"],
        "last_action_group": ACTION_GROUPS.get(last["actionId"], 0),
        "last_point_depth": POINT_DEPTH.get(last["pointId"], 0),
        "last_point_side": POINT_SIDE.get(last["pointId"], 0),
        "last_score_diff": last["scoreSelf"] - last["scoreOther"],
        "parity_bayes_unif": parity_bayes.get(k, 0.5),
    }
    g = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible]
    f["n_attacks"] = sum(1 for x in g if x==1)
    f["n_controls"] = sum(1 for x in g if x==2)
    f["n_defenses"] = sum(1 for x in g if x==3)
    f["n_serves"] = sum(1 for x in g if x==4)
    sg = [ACTION_GROUPS.get(r["actionId"],0) for r in visible if r["strikeNumber"]%2==1]
    rg = [ACTION_GROUPS.get(r["actionId"],0) for r in visible if r["strikeNumber"]%2==0]
    f["n_srv_attacks"] = sum(1 for x in sg if x==1)
    f["n_rcv_attacks"] = sum(1 for x in rg if x==1)
    f["n_srv_strokes"] = len(sg)
    f["n_rcv_strokes"] = len(rg)
    if k >= 2:
        prev = visible[-2]
        f["prev_actionId"] = prev["actionId"]
        f["prev_pointId"] = prev["pointId"]
        f["prev_action_group"] = ACTION_GROUPS.get(prev["actionId"], 0)
        f["prev_handId"] = prev["handId"]
        f["prev_spinId"] = prev["spinId"]
    else:
        f["prev_actionId"] = -1; f["prev_pointId"] = -1
        f["prev_action_group"] = -1; f["prev_handId"] = -1; f["prev_spinId"] = -1

    # --- T2 NEW features ---

    # XY coordinates
    xs, ys = [], []
    for r in visible:
        x, y = get_xy(r["pointId"])
        xs.append(x); ys.append(y)
    last_x, last_y = xs[-1], ys[-1]
    f["last_x"] = last_x
    f["last_y"] = last_y

    # Trajectory: last stroke vs previous
    if k >= 2:
        last_dx = xs[-1] - xs[-2]
        last_dy = ys[-1] - ys[-2]
        f["last_dx"] = last_dx
        f["last_dy"] = last_dy
        f["last_distance"] = float(np.sqrt(last_dx**2 + last_dy**2))
        f["last_angle"] = float(np.arctan2(last_dy, last_dx))  # radians
        f["is_diagonal_last"] = int(abs(last_dx) > 0 and abs(last_dy) > 0)
        f["is_cross_court_last"] = int(last_dx != 0 and xs[-2] * xs[-1] <= 0)  # crossed centerline
    else:
        f["last_dx"] = 0; f["last_dy"] = 0; f["last_distance"] = 0
        f["last_angle"] = 0; f["is_diagonal_last"] = 0; f["is_cross_court_last"] = 0

    # Multi-stroke trajectory aggregates
    if len(xs) >= 2:
        seg_distances = [float(np.sqrt((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)) for i in range(len(xs)-1)]
        f["total_path_length"] = sum(seg_distances)
        f["mean_seg_distance"] = float(np.mean(seg_distances))
        f["max_seg_distance"] = float(max(seg_distances))
        diagonal_count = sum(1 for i in range(len(xs)-1)
                              if abs(xs[i+1]-xs[i]) > 0 and abs(ys[i+1]-ys[i]) > 0)
        f["diagonal_count"] = diagonal_count
        f["diagonal_ratio"] = diagonal_count / len(seg_distances)
        # Drift (recent ball direction in x-axis, last 3 segments)
        recent_xs = xs[-min(3, len(xs)):]
        f["recent_x_drift"] = recent_xs[-1] - recent_xs[0]
        f["recent_y_drift"] = ys[-1] - ys[-min(3, len(ys))]
    else:
        f["total_path_length"] = 0; f["mean_seg_distance"] = 0; f["max_seg_distance"] = 0
        f["diagonal_count"] = 0; f["diagonal_ratio"] = 0
        f["recent_x_drift"] = 0; f["recent_y_drift"] = 0

    # Court coverage
    unique_xy = len(set(zip(xs, ys)))
    f["unique_zones"] = unique_xy
    f["unique_zones_ratio"] = unique_xy / k

    # Corner pressure
    n_corner = sum(1 for r in visible if r["pointId"] in CORNER_ZONES)
    f["n_corner_strokes"] = n_corner
    f["corner_ratio"] = n_corner / k
    f["last_is_corner"] = int(last["pointId"] in CORNER_ZONES)

    # Dominance index (positive = server dominating)
    server_attacks = sum(1 for r in visible
                         if r["strikeNumber"] % 2 == 1 and ACTION_GROUPS.get(r["actionId"], 0) == 1)
    receiver_attacks = sum(1 for r in visible
                           if r["strikeNumber"] % 2 == 0 and ACTION_GROUPS.get(r["actionId"], 0) == 1)
    server_defs = sum(1 for r in visible
                      if r["strikeNumber"] % 2 == 1 and ACTION_GROUPS.get(r["actionId"], 0) == 3)
    receiver_defs = sum(1 for r in visible
                        if r["strikeNumber"] % 2 == 0 and ACTION_GROUPS.get(r["actionId"], 0) == 3)
    f["dominance_index"] = (server_attacks - receiver_attacks) / max(k, 1)
    f["server_aggression_balance"] = (server_attacks - server_defs) / max(len(sg), 1)
    f["receiver_aggression_balance"] = (receiver_attacks - receiver_defs) / max(len(rg), 1)

    # Max attack chain (consecutive attack strokes)
    chains = []; current = 0
    for x in g:
        if x == 1:
            current += 1; chains.append(current)
        else:
            current = 0
    f["max_attack_chain"] = max(chains) if chains else 0

    # Last-3 strokes aggressor signal (server attack = +1, server defense = -1, else 0)
    last3 = visible[-min(3, k):]
    last3_score = 0
    for r in last3:
        is_srv = (r["strikeNumber"] % 2 == 1)
        ag = ACTION_GROUPS.get(r["actionId"], 0)
        if is_srv and ag == 1: last3_score += 1
        elif is_srv and ag == 3: last3_score -= 1
        elif not is_srv and ag == 1: last3_score -= 1
        elif not is_srv and ag == 3: last3_score += 1
    f["last3_aggressor_score"] = last3_score

    # Score pressure × dominance
    score_pressure = (first["scoreSelf"] - first["scoreOther"]) / 11.0
    f["pressure_x_dominance"] = score_pressure * f["dominance_index"]

    # Position × point interaction (raw cross — let LGB learn semantic mapping)
    f["pos_x_depth"] = last["positionId"] * POINT_DEPTH.get(last["pointId"], 0)
    f["pos_x_side"] = last["positionId"] * POINT_SIDE.get(last["pointId"], 0)

    # Strength escalation (last vs mean)
    strengths = [r["strengthId"] for r in visible]
    f["strength_last"] = strengths[-1]
    f["strength_mean"] = float(np.mean(strengths))
    f["strength_max"] = float(max(strengths))
    if len(strengths) >= 2:
        f["strength_last_minus_prev"] = strengths[-1] - strengths[-2]
    else:
        f["strength_last_minus_prev"] = 0

    return f


def build_train_features(df, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes,
                          known_matches=None, max_prefixes=5, rng=None, R_max=15):
    if rng is None: rng = np.random.RandomState(42)
    feats, uids, R_targets, sgp_targets, ks = [], [], [], [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        sgp = int(rows[0]["serverGetPoint"]) if "serverGetPoint" in rows[0] else 0
        if N < 2:
            cand_ks = [1]
        else:
            all_ks = list(range(1, N))
            cand_ks = sorted(rng.choice(all_ks, size=min(max_prefixes, len(all_ks)), replace=False).tolist())
        for k in cand_ks:
            f = make_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches)
            R = min(N - k, R_max)
            feats.append(f); uids.append(uid); R_targets.append(R); sgp_targets.append(sgp); ks.append(k)
    return pd.DataFrame(feats), np.array(uids), np.array(R_targets), np.array(sgp_targets), np.array(ks)


def build_test_features(df, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches=None):
    feats, uids, ks = [], [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        f = make_feats(rows, N, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches)
        feats.append(f); uids.append(uid); ks.append(N)
    return pd.DataFrame(feats), np.array(uids), np.array(ks)


def derive_sgp_from_R(R_probs, ks, R_max):
    sgp = np.zeros(len(ks))
    for i, k in enumerate(ks):
        for r in range(R_max + 1):
            if (k + r) % 2 == 0:  # sgp = 1 (parity 0)
                sgp[i] += R_probs[i, r]
    return sgp


def main():
    print("=" * 72)
    print("T2: T1 + xy/trajectory/dominance/reachability features")
    print("=" * 72)

    test_csv = os.environ.get("V5Z_TEST_CSV", "data/test_new.csv")
    kappa = int(os.environ.get("V5Z_KAPPA", "50"))
    n_folds = int(os.environ.get("V5Z_N_FOLDS", "5"))
    max_prefix = int(os.environ.get("V5Z_MAX_PREFIX", "5"))
    R_max = int(os.environ.get("V5Z_RMAX", "15"))

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv(test_csv)
    print(f"Train: {len(train_df)} rows / {train_df['rally_uid'].nunique()} rallies")
    print(f"Test:  {len(test_df)} rows / {test_df['rally_uid'].nunique()} rallies")

    rally_meta = train_df.groupby("rally_uid").first()[
        ["gamePlayerId","gamePlayerOtherId","serverGetPoint","sex","match"]
    ].reset_index()
    train_matches = set(rally_meta["match"].unique())

    train_N = train_df.groupby("rally_uid").size().values
    P_N_train = Counter(train_N.tolist())
    L_max = max(test_df.groupby("rally_uid").size().max(), train_N.max())
    parity_bayes = parity_bayes_uniform(L_max, P_N_train)

    uid_to_idx = {u: i for i, u in enumerate(rally_meta["rally_uid"].values)}
    uid_arr = rally_meta["serverGetPoint"].values
    match_arr = rally_meta["match"].values
    sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=42)

    oof_sgp_per_rally = np.zeros(len(rally_meta))
    oof_count_per_rally = np.zeros(len(rally_meta))
    oof_label = uid_arr.astype(np.float64)
    fold_models = []
    feat_cols_ref = None

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(rally_meta, uid_arr, groups=match_arr)):
        tr_meta = rally_meta.iloc[tr_idx]; va_meta = rally_meta.iloc[va_idx]
        tr_uids = set(tr_meta["rally_uid"]); va_uids = set(va_meta["rally_uid"])
        srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global = bayesian_player_stats(tr_meta, kappa)

        tr_rdf = train_df[train_df["rally_uid"].isin(tr_uids)]
        va_rdf = train_df[train_df["rally_uid"].isin(va_uids)]
        rng_tr = np.random.RandomState(42 + fold); rng_va = np.random.RandomState(1000 + fold)
        X_tr, uid_tr, R_tr, sgp_tr, k_tr = build_train_features(tr_rdf, srv_w, srv_n_, rcv_l, rcv_n_,
                                                                prior_sex, prior_global, parity_bayes,
                                                                max_prefixes=max_prefix, rng=rng_tr, R_max=R_max)
        X_va, uid_va, R_va, sgp_va, k_va = build_train_features(va_rdf, srv_w, srv_n_, rcv_l, rcv_n_,
                                                                prior_sex, prior_global, parity_bayes,
                                                                max_prefixes=max_prefix, rng=rng_va, R_max=R_max)

        if feat_cols_ref is None:
            feat_cols_ref = list(X_tr.columns)

        print(f"\nFold {fold+1}/{n_folds}: Tr {len(X_tr)} samples, Va {len(X_va)}, features={X_tr.shape[1]} (T1=40 + new={X_tr.shape[1]-40})")

        model = lgb.train(
            {
                "objective": "multiclass", "num_class": R_max + 1, "metric": "multi_logloss",
                "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
                "lambda_l2": 1.0, "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
                "seed": 42, "verbose": -1,
            },
            lgb.Dataset(X_tr, label=R_tr),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(X_va, label=R_va)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )

        R_probs_va = model.predict(X_va)
        sgp_pred_va = derive_sgp_from_R(R_probs_va, k_va, R_max)
        prefix_auc = roc_auc_score(sgp_va, sgp_pred_va)
        for u, p in zip(uid_va, sgp_pred_va):
            i = uid_to_idx[u]
            oof_sgp_per_rally[i] += p
            oof_count_per_rally[i] += 1

        # Top-15 features in this fold
        fi = model.feature_importance(importance_type="gain")
        cols = list(X_tr.columns)
        top15 = sorted(zip(cols, fi), key=lambda x: -x[1])[:15]
        new_in_top15 = sum(1 for n, _ in top15 if n in [
            "last_x","last_y","last_dx","last_dy","last_distance","last_angle","is_diagonal_last","is_cross_court_last",
            "total_path_length","mean_seg_distance","max_seg_distance","diagonal_count","diagonal_ratio",
            "recent_x_drift","recent_y_drift","unique_zones","unique_zones_ratio","n_corner_strokes","corner_ratio",
            "last_is_corner","dominance_index","server_aggression_balance","receiver_aggression_balance",
            "max_attack_chain","last3_aggressor_score","pressure_x_dominance","pos_x_depth","pos_x_side",
            "strength_last","strength_mean","strength_max","strength_last_minus_prev"
        ])
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, new-feats-in-top15: {new_in_top15}/15")
        fold_models.append((model, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global))

    oof_sgp_rally = oof_sgp_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label[rally_mask], oof_sgp_rally[rally_mask])
    print(f"\n{'='*72}")
    print(f"T2 OOF AUC (per-rally aggregated): {overall_auc:.4f}")

    # Top-20 across all folds (using last fold's importance)
    fi_last = pd.DataFrame({
        "feature": feat_cols_ref,
        "importance": fold_models[-1][0].feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\nTop 20 features by gain (last fold):")
    print(fi_last.head(20).to_string(index=False))

    # Test prediction
    print("\n--- Test prediction ---")
    srv_w_full, srv_n_full, rcv_l_full, rcv_n_full, ps_full, pg_full = bayesian_player_stats(rally_meta, kappa)
    test_pred = np.zeros(test_df["rally_uid"].nunique())
    test_uids_ref = None
    for model, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f in fold_models:
        X_te, uid_te, k_te = build_test_features(test_df, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f,
                                                  ps_f, pg_f, parity_bayes, known_matches=train_matches)
        if test_uids_ref is None: test_uids_ref = uid_te
        R_probs_te = model.predict(X_te)
        sgp_te = derive_sgp_from_R(R_probs_te, k_te, R_max)
        test_pred += sgp_te / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_t2_features.npz"
    np.savez(out_path,
             rally_uids=test_uids_ref,
             sgp_pred=test_pred,
             method=f"t2_lgb_R_features_kappa{kappa}_rmax{R_max}",
             holdout_auc=overall_auc,
             oof_pred=oof_sgp_rally,
             oof_label=oof_label,
             feature_names=np.array(feat_cols_ref))
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
