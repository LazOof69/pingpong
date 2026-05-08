#!/usr/bin/env python3
"""
T6: T5 + NMF rank-6 player style embeddings + k-NN borrowed winrate (Day 4).

Method:
- Per fold: build training-fold P × C behavior matrix V (action/point/strength/spin distributions, L1-normalized)
- KL-NMF rank-6: V ≈ W H, W = player embeddings, H = profile basis
- For unseen players: NNLS fold-in with H fixed (build their behavior vec from available rallies)
- k-NN borrow: for unseen player, find top-5 train neighbors via Hellinger distance, weighted-avg their EB winrate

New features (~16):
- nmf_srv_1..6, nmf_rcv_1..6 (12 dims)
- knn_borrowed_winrate_srv, knn_top1_sim_srv (2 dims for server)
- knn_borrowed_loserate_rcv, knn_top1_sim_rcv (2 dims for receiver)

Usage:
    python3 src/train/predict_sgp_t6_nmf.py
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import NMF
from scipy.optimize import nnls

import sys
sys.path.insert(0, "src/train")
from predict_sgp_t4_eb import (
    ACTION_GROUPS, POINT_DEPTH, POINT_SIDE, SIDE_X, DEPTH_Y, CORNER_ZONES,
    get_xy, classify_player_style, compute_player_aggregates,
    compute_multilevel_eb, get_eb_features, parity_bayes_uniform,
    compute_morris_importance, service_block_pos, build_score_sequence_features,
    derive_sgp_from_R,
)
from predict_sgp_t5_markov import compute_bigram_tables, compute_bigram_features

N_ACTION = 19
N_POINT = 10
N_STRENGTH = 5
N_SPIN = 7
N_BEHAVIOR_DIMS = N_ACTION + N_POINT + N_STRENGTH + N_SPIN  # 41

NMF_RANK = 6


def build_behavior_matrix(df, players=None):
    """Build per-player behavior matrix (counts then L1-normalized)."""
    if players is None:
        players = sorted(df["gamePlayerId"].unique())
    pid_to_idx = {p: i for i, p in enumerate(players)}
    V = np.zeros((len(players), N_BEHAVIOR_DIMS), dtype=np.float64)

    for pid, grp in df.groupby("gamePlayerId"):
        if pid not in pid_to_idx:
            continue
        i = pid_to_idx[pid]
        a_counts = np.bincount(grp["actionId"].clip(0, N_ACTION - 1).astype(int), minlength=N_ACTION)
        p_counts = np.bincount(grp["pointId"].clip(0, N_POINT - 1).astype(int), minlength=N_POINT)
        s_counts = np.bincount(grp["strengthId"].clip(0, N_STRENGTH - 1).astype(int), minlength=N_STRENGTH)
        sp_counts = np.bincount(grp["spinId"].clip(0, N_SPIN - 1).astype(int), minlength=N_SPIN)
        offset = 0
        V[i, offset:offset + N_ACTION] = a_counts; offset += N_ACTION
        V[i, offset:offset + N_POINT] = p_counts; offset += N_POINT
        V[i, offset:offset + N_STRENGTH] = s_counts; offset += N_STRENGTH
        V[i, offset:offset + N_SPIN] = sp_counts

    # L1 normalize each row (probability distribution per player)
    row_sums = V.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    V_norm = V / row_sums
    return V_norm, V, players, pid_to_idx


def fit_nmf(V_norm, rank=NMF_RANK, max_iter=300, seed=42):
    """Fit KL-NMF rank-r. Returns (W, H, fitted NMF model)."""
    nmf = NMF(n_components=rank, init="nndsvd", solver="mu",
              beta_loss="kullback-leibler", max_iter=max_iter,
              random_state=seed, tol=1e-4)
    W = nmf.fit_transform(V_norm + 1e-9)  # add tiny epsilon to avoid zero issues
    H = nmf.components_
    return W, H, nmf


def fold_in_nmf(v_norm, H, max_iter=50):
    """Solve min ||v - w H||^2 s.t. w >= 0 via NNLS. Returns w (rank,)."""
    # H is (rank, C), we want w (rank,) such that w @ H ≈ v
    # NNLS solves min ||A x - b||^2 with A = H.T (C, rank), x = w (rank,), b = v (C,)
    w, _ = nnls(H.T, v_norm)
    return w


def hellinger(p, q):
    """Hellinger distance between two probability vectors (in [0, 1])."""
    return np.sqrt(0.5 * np.sum((np.sqrt(np.maximum(p, 0)) - np.sqrt(np.maximum(q, 0)))**2))


def compute_knn_borrowed(target_pid, target_v, train_pids, train_V, eb_dict, side, k=5):
    """For target player (possibly unseen), find k-nearest train neighbors via Hellinger,
    return weighted-avg eb winrate/loserate + max similarity."""
    if len(train_pids) == 0:
        return {f"knn_borrow_{side}": 0.5, f"knn_top1_sim_{side}": 0.0, f"knn_avg_sim_{side}": 0.0}
    distances = np.array([hellinger(target_v, train_V[i]) for i in range(len(train_pids))])
    sims = 1.0 - distances  # in [0, 1]
    top_k_idx = np.argsort(-sims)[:k]
    top_pids = [train_pids[i] for i in top_k_idx]
    top_sims = sims[top_k_idx]
    # Weighted avg of borrowed value
    eb_key = f"eb_{side}_p"
    eb_values = []
    for pid in top_pids:
        if pid in eb_dict:
            eb_values.append(eb_dict[pid][eb_key])
        else:
            eb_values.append(0.5)
    eb_values = np.array(eb_values)
    weights = top_sims / (top_sims.sum() + 1e-9)
    borrowed = float((eb_values * weights).sum())
    return {
        f"knn_borrow_{side}": borrowed,
        f"knn_top1_sim_{side}": float(top_sims[0]) if len(top_sims) > 0 else 0.0,
        f"knn_avg_sim_{side}": float(top_sims.mean()) if len(top_sims) > 0 else 0.0,
    }


def make_feats(rows, k, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
               player_groups, prior_sex, prior_global, parity_bayes, morris_table,
               score_seq_feats, t1_pred_per_rally,
               per_player_logp, per_player_n, global_logp,
               player_nmf, train_player_pids, train_V_norm, eb_srv_dict, eb_rcv_dict,
               known_matches=None):
    first = rows[0]; visible = rows[:k]; last = visible[-1]
    sex = first["sex"]
    uid = first["rally_uid"]
    srv_pid = first["gamePlayerId"]
    rcv_pid = first["gamePlayerOtherId"]

    f = {
        "ctx_len": k, "sex": sex, "numberGame": first["numberGame"],
        "score_self_start": first["scoreSelf"], "score_other_start": first["scoreOther"],
        "score_diff_start": first["scoreSelf"] - first["scoreOther"],
        "score_sum_start": first["scoreSelf"] + first["scoreOther"],
        "is_deuce_start": int(first["scoreSelf"]>=10 and first["scoreOther"]>=10),
        "match_seen": int(known_matches is not None and first["match"] in known_matches),
        "last_strikeId": last["strikeId"], "last_handId": last["handId"],
        "last_strengthId": last["strengthId"], "last_spinId": last["spinId"],
        "last_pointId": last["pointId"], "last_actionId": last["actionId"],
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
    sgs = [ACTION_GROUPS.get(r["actionId"],0) for r in visible if r["strikeNumber"]%2==1]
    rgs = [ACTION_GROUPS.get(r["actionId"],0) for r in visible if r["strikeNumber"]%2==0]
    f["n_srv_attacks"] = sum(1 for x in sgs if x==1)
    f["n_rcv_attacks"] = sum(1 for x in rgs if x==1)
    f["n_srv_strokes"] = len(sgs); f["n_rcv_strokes"] = len(rgs)
    if k >= 2:
        prev = visible[-2]
        f["prev_actionId"] = prev["actionId"]; f["prev_pointId"] = prev["pointId"]
        f["prev_action_group"] = ACTION_GROUPS.get(prev["actionId"], 0)
        f["prev_handId"] = prev["handId"]; f["prev_spinId"] = prev["spinId"]
    else:
        f["prev_actionId"]=-1; f["prev_pointId"]=-1; f["prev_action_group"]=-1
        f["prev_handId"]=-1; f["prev_spinId"]=-1

    # T2 trajectory
    xs, ys = [], []
    for r in visible:
        x, y = get_xy(r["pointId"]); xs.append(x); ys.append(y)
    f["last_x"] = xs[-1]; f["last_y"] = ys[-1]
    if k >= 2:
        last_dx = xs[-1] - xs[-2]; last_dy = ys[-1] - ys[-2]
        f["last_dx"] = last_dx; f["last_dy"] = last_dy
        f["last_distance"] = float(np.sqrt(last_dx**2 + last_dy**2))
        f["last_angle"] = float(np.arctan2(last_dy, last_dx))
        f["is_diagonal_last"] = int(abs(last_dx) > 0 and abs(last_dy) > 0)
        f["is_cross_court_last"] = int(last_dx != 0 and xs[-2] * xs[-1] <= 0)
    else:
        f["last_dx"]=0; f["last_dy"]=0; f["last_distance"]=0
        f["last_angle"]=0; f["is_diagonal_last"]=0; f["is_cross_court_last"]=0
    if len(xs) >= 2:
        seg = [float(np.sqrt((xs[i+1]-xs[i])**2+(ys[i+1]-ys[i])**2)) for i in range(len(xs)-1)]
        f["total_path_length"] = sum(seg); f["mean_seg_distance"] = float(np.mean(seg))
        f["max_seg_distance"] = float(max(seg))
        f["diagonal_count"] = sum(1 for i in range(len(xs)-1)
                                   if abs(xs[i+1]-xs[i])>0 and abs(ys[i+1]-ys[i])>0)
        f["diagonal_ratio"] = f["diagonal_count"] / len(seg)
        recent_xs = xs[-min(3, len(xs)):]
        f["recent_x_drift"] = recent_xs[-1] - recent_xs[0]
        f["recent_y_drift"] = ys[-1] - ys[-min(3, len(ys))]
    else:
        f["total_path_length"]=0; f["mean_seg_distance"]=0; f["max_seg_distance"]=0
        f["diagonal_count"]=0; f["diagonal_ratio"]=0
        f["recent_x_drift"]=0; f["recent_y_drift"]=0
    unique_xy = len(set(zip(xs, ys)))
    f["unique_zones"] = unique_xy; f["unique_zones_ratio"] = unique_xy / k
    n_corner = sum(1 for r in visible if r["pointId"] in CORNER_ZONES)
    f["n_corner_strokes"] = n_corner; f["corner_ratio"] = n_corner / k
    f["last_is_corner"] = int(last["pointId"] in CORNER_ZONES)
    sa = sum(1 for r in visible if r["strikeNumber"] % 2 == 1 and ACTION_GROUPS.get(r["actionId"], 0) == 1)
    ra = sum(1 for r in visible if r["strikeNumber"] % 2 == 0 and ACTION_GROUPS.get(r["actionId"], 0) == 1)
    sd = sum(1 for r in visible if r["strikeNumber"] % 2 == 1 and ACTION_GROUPS.get(r["actionId"], 0) == 3)
    rd = sum(1 for r in visible if r["strikeNumber"] % 2 == 0 and ACTION_GROUPS.get(r["actionId"], 0) == 3)
    f["dominance_index"] = (sa - ra) / max(k, 1)
    f["server_aggression_balance"] = (sa - sd) / max(len(sgs), 1)
    f["receiver_aggression_balance"] = (ra - rd) / max(len(rgs), 1)
    chains = []; cur_c = 0
    for x in g:
        if x == 1: cur_c += 1; chains.append(cur_c)
        else: cur_c = 0
    f["max_attack_chain"] = max(chains) if chains else 0
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
    score_pressure = (first["scoreSelf"] - first["scoreOther"]) / 11.0
    f["pressure_x_dominance"] = score_pressure * f["dominance_index"]
    f["pos_x_depth"] = last["positionId"] * POINT_DEPTH.get(last["pointId"], 0)
    f["pos_x_side"] = last["positionId"] * POINT_SIDE.get(last["pointId"], 0)
    strengths = [r["strengthId"] for r in visible]
    f["strength_last"] = strengths[-1]; f["strength_mean"] = float(np.mean(strengths))
    f["strength_max"] = float(max(strengths))
    f["strength_last_minus_prev"] = strengths[-1] - strengths[-2] if len(strengths) >= 2 else 0

    # T3 structural priors
    seq = score_seq_feats.get(uid, {"prev_won": -1, "run_len": 0, "cum_strokes": 0})
    f["prev_point_won_by_server"] = seq["prev_won"]
    f["server_run_length"] = seq["run_len"]
    f["cum_strokes_in_match"] = seq["cum_strokes"]
    f["service_block_position"] = service_block_pos(first["scoreSelf"], first["scoreOther"])
    f["morris_importance"] = morris_table.get((first["scoreSelf"], first["scoreOther"]),
                                               morris_table.get((10, 10), 0.5))
    f["is_set_point_for_server"] = int(first["scoreSelf"] >= 10 and first["scoreSelf"] >= first["scoreOther"] + 1)
    f["is_set_point_against"] = int(first["scoreOther"] >= 10 and first["scoreOther"] >= first["scoreSelf"] + 1)
    f["t1_stacking_pred"] = float(t1_pred_per_rally.get(uid, 0.5))

    # T4 EB features
    srv_group_key = player_groups.get(srv_pid, "1_1_1")
    rcv_group_key = player_groups.get(rcv_pid, "1_1_1")
    eb_srv_feats = get_eb_features(srv_pid, "srv", eb_srv, srv_groups, params_srv, srv_group_key)
    eb_rcv_feats = get_eb_features(rcv_pid, "rcv", eb_rcv, rcv_groups, params_rcv, rcv_group_key)
    f.update(eb_srv_feats); f.update(eb_rcv_feats)
    f["eb_skill_diff"] = eb_srv_feats["eb_srv_logit"] - eb_rcv_feats["eb_rcv_logit"]
    f["eb_combined_p"] = 0.5 * (eb_srv_feats["eb_srv_p"] + (1 - eb_rcv_feats["eb_rcv_p"]))
    f["eb_uncertainty"] = eb_srv_feats["eb_srv_post_se"] + eb_rcv_feats["eb_rcv_post_se"]

    # T5 Markov bigram features
    bigram_feats = compute_bigram_features(visible, srv_pid, per_player_logp, per_player_n, global_logp)
    f.update(bigram_feats)

    # === T6 NEW: NMF embeddings + k-NN borrowed ===
    srv_emb = player_nmf.get(srv_pid, np.zeros(NMF_RANK))
    rcv_emb = player_nmf.get(rcv_pid, np.zeros(NMF_RANK))
    for i in range(NMF_RANK):
        f[f"nmf_srv_{i}"] = float(srv_emb[i])
        f[f"nmf_rcv_{i}"] = float(rcv_emb[i])
    f["nmf_skill_diff_l2"] = float(np.linalg.norm(srv_emb - rcv_emb))

    # k-NN borrowed: only meaningful if target is unseen by training EB
    srv_v = train_V_norm[train_player_pids.index(srv_pid)] if srv_pid in train_player_pids else None
    rcv_v = train_V_norm[train_player_pids.index(rcv_pid)] if rcv_pid in train_player_pids else None

    return f


def build_features(df, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                    player_groups, prior_sex, prior_global, parity_bayes, morris_table,
                    score_seq_feats, t1_pred_per_rally,
                    per_player_logp, per_player_n, global_logp,
                    player_nmf, train_player_pids, train_V_norm, eb_srv_dict, eb_rcv_dict,
                    known_matches=None, max_prefixes=5, rng=None, R_max=15, is_train=True):
    if rng is None: rng = np.random.RandomState(42)
    feats, uids, R_targets, sgp_targets, ks = [], [], [], [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        sgp = int(rows[0]["serverGetPoint"]) if "serverGetPoint" in rows[0] else 0
        if is_train:
            if N < 2: cand_ks = [1]
            else:
                all_ks = list(range(1, N))
                cand_ks = sorted(rng.choice(all_ks, size=min(max_prefixes, len(all_ks)), replace=False).tolist())
        else:
            cand_ks = [N]
        for k in cand_ks:
            f = make_feats(rows, k, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                           player_groups, prior_sex, prior_global, parity_bayes, morris_table,
                           score_seq_feats, t1_pred_per_rally,
                           per_player_logp, per_player_n, global_logp,
                           player_nmf, train_player_pids, train_V_norm, eb_srv_dict, eb_rcv_dict,
                           known_matches)
            R = min(N - k, R_max)
            feats.append(f); uids.append(uid); R_targets.append(R); sgp_targets.append(sgp); ks.append(k)
    return pd.DataFrame(feats), np.array(uids), np.array(R_targets), np.array(sgp_targets), np.array(ks)


def main():
    print("=" * 72)
    print("T6: T5 + NMF rank-6 player embeddings + k-NN borrow (Day 4)")
    print("=" * 72)

    test_csv = os.environ.get("V5Z_TEST_CSV", "data/test_new.csv")
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
    morris_table = compute_morris_importance(target=11, p=0.55)
    score_seq_train = build_score_sequence_features(train_df)
    score_seq_test = build_score_sequence_features(test_df)

    t1 = np.load("artifacts/sgp_pred_NEWTEST_t1_rcount.npz", allow_pickle=True)
    t1_oof_dict = {u: float(p) for u, p in zip(rally_meta["rally_uid"].values, t1["oof_pred"])}
    t1_test_dict = {u: float(p) for u, p in zip(t1["rally_uids"].tolist(), t1["sgp_pred"].tolist())}

    uid_to_idx = {u: i for i, u in enumerate(rally_meta["rally_uid"].values)}
    uid_arr = rally_meta["serverGetPoint"].values
    match_arr = rally_meta["match"].values
    sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=42)

    oof_sgp_per_rally = np.zeros(len(rally_meta))
    oof_count_per_rally = np.zeros(len(rally_meta))
    oof_label = uid_arr.astype(np.float64)
    fold_models = []
    fold_artifacts = []
    feat_cols_ref = None

    # Test-side player set (for fold-in)
    test_player_set = set(test_df["gamePlayerId"].unique()) | set(test_df["gamePlayerOtherId"].unique())

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(rally_meta, uid_arr, groups=match_arr)):
        tr_meta = rally_meta.iloc[tr_idx]; va_meta = rally_meta.iloc[va_idx]
        tr_uids = set(tr_meta["rally_uid"]); va_uids = set(va_meta["rally_uid"])

        player_aggs = compute_player_aggregates(train_df, restrict_uids=tr_uids)
        eb_srv, srv_groups, params_srv = compute_multilevel_eb(player_aggs, side="srv")
        eb_rcv, rcv_groups, params_rcv = compute_multilevel_eb(player_aggs, side="rcv")
        player_groups = {pid: f"{int(row['sex'])}_{int(row['hand_majority'])}_{int(row['style'])}"
                          for pid, row in player_aggs.iterrows()}
        prior_global = float(tr_meta["serverGetPoint"].mean())
        prior_sex = {s: float(tr_meta[tr_meta["sex"]==s]["serverGetPoint"].mean()) for s in tr_meta["sex"].unique()}

        tr_rdf = train_df[train_df["rally_uid"].isin(tr_uids)]
        va_rdf = train_df[train_df["rally_uid"].isin(va_uids)]
        per_player_logp, per_player_n, global_logp = compute_bigram_tables(tr_rdf, alpha=0.5)

        # === T6 NMF: per-fold ===
        train_V_norm, train_V_raw, train_player_pids, _ = build_behavior_matrix(tr_rdf)
        W_train, H_train, _ = fit_nmf(train_V_norm, rank=NMF_RANK, max_iter=200, seed=42 + fold)

        # Build player NMF dict for training-fold players
        player_nmf = {pid: W_train[i] for i, pid in enumerate(train_player_pids)}

        # Fold-in for val-fold players that are NOT in training fold (within-train cold-start)
        val_pids = set()
        for u in va_uids:
            row = rally_meta[rally_meta["rally_uid"] == u].iloc[0]
            val_pids.add(row["gamePlayerId"]); val_pids.add(row["gamePlayerOtherId"])
        unseen_in_fold = val_pids - set(train_player_pids)
        # Build val behavior matrix for these
        if unseen_in_fold:
            val_rdf_subset = train_df[train_df["gamePlayerId"].isin(unseen_in_fold) | train_df["gamePlayerOtherId"].isin(unseen_in_fold)]
            val_rdf_subset = val_rdf_subset[val_rdf_subset["rally_uid"].isin(va_uids)]
            for upid in unseen_in_fold:
                strokes_for_p = val_rdf_subset[val_rdf_subset["gamePlayerId"] == upid]
                if len(strokes_for_p) == 0:
                    player_nmf[upid] = np.zeros(NMF_RANK)
                    continue
                # Build behavior vector for this player
                a_counts = np.bincount(strokes_for_p["actionId"].clip(0, N_ACTION - 1).astype(int), minlength=N_ACTION)
                p_counts = np.bincount(strokes_for_p["pointId"].clip(0, N_POINT - 1).astype(int), minlength=N_POINT)
                s_counts = np.bincount(strokes_for_p["strengthId"].clip(0, N_STRENGTH - 1).astype(int), minlength=N_STRENGTH)
                sp_counts = np.bincount(strokes_for_p["spinId"].clip(0, N_SPIN - 1).astype(int), minlength=N_SPIN)
                v_p = np.concatenate([a_counts, p_counts, s_counts, sp_counts]).astype(np.float64)
                v_norm = v_p / max(v_p.sum(), 1)
                w_p = fold_in_nmf(v_norm, H_train)
                player_nmf[upid] = w_p

        # Fold-in for test players (cold-start vs train)
        test_unseen = test_player_set - set(train_player_pids)
        if test_unseen:
            test_rdf_subset = test_df[test_df["gamePlayerId"].isin(test_unseen) | test_df["gamePlayerOtherId"].isin(test_unseen)]
            for upid in test_unseen:
                strokes_for_p = test_rdf_subset[test_rdf_subset["gamePlayerId"] == upid]
                if len(strokes_for_p) == 0:
                    player_nmf[upid] = np.zeros(NMF_RANK)
                    continue
                a_counts = np.bincount(strokes_for_p["actionId"].clip(0, N_ACTION - 1).astype(int), minlength=N_ACTION)
                p_counts = np.bincount(strokes_for_p["pointId"].clip(0, N_POINT - 1).astype(int), minlength=N_POINT)
                s_counts = np.bincount(strokes_for_p["strengthId"].clip(0, N_STRENGTH - 1).astype(int), minlength=N_STRENGTH)
                sp_counts = np.bincount(strokes_for_p["spinId"].clip(0, N_SPIN - 1).astype(int), minlength=N_SPIN)
                v_p = np.concatenate([a_counts, p_counts, s_counts, sp_counts]).astype(np.float64)
                v_norm = v_p / max(v_p.sum(), 1)
                w_p = fold_in_nmf(v_norm, H_train)
                player_nmf[upid] = w_p

        rng_tr = np.random.RandomState(42 + fold); rng_va = np.random.RandomState(1000 + fold)
        X_tr, uid_tr, R_tr, sgp_tr, k_tr = build_features(tr_rdf, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                          params_srv, params_rcv, player_groups,
                                                          prior_sex, prior_global, parity_bayes, morris_table,
                                                          score_seq_train, t1_oof_dict,
                                                          per_player_logp, per_player_n, global_logp,
                                                          player_nmf, train_player_pids, train_V_norm,
                                                          eb_srv, eb_rcv,
                                                          max_prefixes=max_prefix, rng=rng_tr, R_max=R_max, is_train=True)
        X_va, uid_va, R_va, sgp_va, k_va = build_features(va_rdf, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                          params_srv, params_rcv, player_groups,
                                                          prior_sex, prior_global, parity_bayes, morris_table,
                                                          score_seq_train, t1_oof_dict,
                                                          per_player_logp, per_player_n, global_logp,
                                                          player_nmf, train_player_pids, train_V_norm,
                                                          eb_srv, eb_rcv,
                                                          max_prefixes=max_prefix, rng=rng_va, R_max=R_max, is_train=True)

        if feat_cols_ref is None:
            feat_cols_ref = list(X_tr.columns)

        new_t6 = [c for c in feat_cols_ref if c.startswith("nmf_")]
        if fold == 0:
            print(f"\nFold 1: total feats={X_tr.shape[1]}, T6 NMF new={len(new_t6)}: {new_t6}")
            print(f"  NMF embeddings cover {len(player_nmf)} players (train: {len(train_player_pids)}, fold-in: {len(player_nmf) - len(train_player_pids)})")

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
        fi = model.feature_importance(importance_type="gain")
        cols = list(X_tr.columns)
        top10 = sorted(zip(cols, fi), key=lambda x: -x[1])[:10]
        new_in_top10 = sum(1 for n, _ in top10 if n.startswith("nmf_"))
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, NMF-in-top10: {new_in_top10}/10")
        fold_models.append(model)
        fold_artifacts.append((eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                                player_groups, prior_sex, prior_global,
                                per_player_logp, per_player_n, global_logp,
                                player_nmf, train_player_pids, train_V_norm))

    oof_sgp_rally = oof_sgp_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label[rally_mask], oof_sgp_rally[rally_mask])
    print(f"\n{'='*72}")
    print(f"T6 OOF AUC (per-rally aggregated): {overall_auc:.4f}")

    fi_last = pd.DataFrame({
        "feature": feat_cols_ref,
        "importance": fold_models[-1].feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\nTop 25 features by gain (last fold):")
    print(fi_last.head(25).to_string(index=False))

    print("\n--- Test prediction ---")
    test_pred = np.zeros(test_df["rally_uid"].nunique())
    test_uids_ref = None
    for model, (eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                player_groups, prior_sex, prior_global,
                per_player_logp, per_player_n, global_logp,
                player_nmf, train_player_pids, train_V_norm) in zip(fold_models, fold_artifacts):
        X_te, uid_te, _, _, k_te = build_features(test_df, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                   params_srv, params_rcv, player_groups,
                                                   prior_sex, prior_global, parity_bayes, morris_table,
                                                   score_seq_test, t1_test_dict,
                                                   per_player_logp, per_player_n, global_logp,
                                                   player_nmf, train_player_pids, train_V_norm,
                                                   eb_srv, eb_rcv,
                                                   known_matches=train_matches, is_train=False)
        if test_uids_ref is None: test_uids_ref = uid_te
        R_probs_te = model.predict(X_te)
        sgp_te = derive_sgp_from_R(R_probs_te, k_te, R_max)
        test_pred += sgp_te / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_t6_nmf.npz"
    np.savez(out_path, rally_uids=test_uids_ref, sgp_pred=test_pred,
             method="t6_lgb_R_eb_markov_nmf",
             holdout_auc=overall_auc, oof_pred=oof_sgp_rally, oof_label=oof_label,
             feature_names=np.array(feat_cols_ref))
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
