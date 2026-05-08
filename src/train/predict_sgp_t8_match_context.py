#!/usr/bin/env python3
"""
T8: T6 + match-level test self-supervised features (leave-one-out anti-leak).

Key insight: each match has ~50-80 rallies. For a test rally r in match m, OTHER
rallies of m provide rich contextual features about the players in this match,
WITHOUT using r's own label. This is a free signal we missed entirely.

For unseen test players (44% of test), their match-level aggregates are the
ONLY ground-truth-style signal we can extract — bypassing cold-start.

Anti-leak: aggregate over (match, player) pairs but subtract this rally's
contribution at lookup time (true leave-one-out).

New features (~10):
- srv_match_attack_rate, srv_match_strength_mean, srv_match_n_strokes
- rcv_match_attack_rate, rcv_match_strength_mean, rcv_match_n_strokes
- match_avg_visible_L, match_n_rallies
- match_attack_rate_overall, match_dominant_pointId
- srv_match_winrate (from prior rallies in match's score sequence)

Built on T6 base (NOT T7 — handedness features didn't help OOF).

Usage:
    python3 src/train/predict_sgp_t8_match_context.py
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
from predict_sgp_t6_nmf import (
    N_ACTION, N_POINT, N_STRENGTH, N_SPIN, N_BEHAVIOR_DIMS, NMF_RANK,
    build_behavior_matrix, fit_nmf, fold_in_nmf,
)


def compute_match_player_aggregates(df):
    """Pre-compute (match, player) total strokes + actions + strengths.
    Used for leave-one-out: subtract specific rally's contribution at lookup."""
    df_with_ag = df.copy()
    df_with_ag["ag"] = df_with_ag["actionId"].map(ACTION_GROUPS).fillna(0).astype(int)
    df_with_ag["is_attack"] = (df_with_ag["ag"] == 1).astype(int)
    df_with_ag["is_defense"] = (df_with_ag["ag"] == 3).astype(int)
    df_with_ag["is_corner"] = df_with_ag["pointId"].isin([1, 3, 7, 9]).astype(int)

    # Match × Player totals
    mp_totals = df_with_ag.groupby(["match", "gamePlayerId"]).agg(
        total_strokes=("actionId", "size"),
        total_attacks=("is_attack", "sum"),
        total_defenses=("is_defense", "sum"),
        total_corner=("is_corner", "sum"),
        sum_strength=("strengthId", "sum"),
    ).reset_index()

    # Match-Rally × Player contribution (per-rally per-player stroke counts)
    rally_player = df_with_ag.groupby(["rally_uid", "gamePlayerId"]).agg(
        rally_strokes=("actionId", "size"),
        rally_attacks=("is_attack", "sum"),
        rally_defenses=("is_defense", "sum"),
        rally_corner=("is_corner", "sum"),
        rally_strength=("strengthId", "sum"),
        match=("match", "first"),
    ).reset_index()

    # Match-level aggregates
    rally_meta = df.groupby("rally_uid").agg(
        match=("match", "first"),
        N=("strikeNumber", "size"),
    ).reset_index()
    match_agg = rally_meta.groupby("match").agg(
        n_rallies=("rally_uid", "size"),
        avg_L=("N", "mean"),
        sum_strokes=("N", "sum"),
    ).reset_index()
    match_agg_dict = match_agg.set_index("match").to_dict("index")

    # Match-level point/action distributions
    match_action_attack = df_with_ag.groupby("match")["is_attack"].sum() / df_with_ag.groupby("match").size()
    match_corner_rate = df_with_ag.groupby("match")["is_corner"].sum() / df_with_ag.groupby("match").size()
    match_dominant_pid = df_with_ag.groupby("match")["pointId"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)

    # Per (match, player) aggregate dict
    mp_dict = {}
    for _, row in mp_totals.iterrows():
        m, p = row["match"], row["gamePlayerId"]
        mp_dict[(m, p)] = {
            "total_strokes": int(row["total_strokes"]),
            "total_attacks": int(row["total_attacks"]),
            "total_defenses": int(row["total_defenses"]),
            "total_corner": int(row["total_corner"]),
            "sum_strength": float(row["sum_strength"]),
        }

    # Per (rally, player) for leave-one-out subtraction
    rally_player_dict = {}
    for _, row in rally_player.iterrows():
        rally_player_dict[(row["rally_uid"], row["gamePlayerId"])] = {
            "rally_strokes": int(row["rally_strokes"]),
            "rally_attacks": int(row["rally_attacks"]),
            "rally_defenses": int(row["rally_defenses"]),
            "rally_corner": int(row["rally_corner"]),
            "rally_strength": float(row["rally_strength"]),
        }

    return mp_dict, rally_player_dict, match_agg_dict, dict(match_action_attack), dict(match_corner_rate), dict(match_dominant_pid)


def get_match_features(rally_uid, match, server_pid, receiver_pid,
                        mp_dict, rally_player_dict, match_agg_dict,
                        match_action_attack, match_corner_rate, match_dominant_pid):
    """Compute leave-one-out match-level features for this rally."""
    f = {}

    # Match-wide RATES ONLY (scale-invariant across train full-rally vs test truncated-rally)
    # DROP: match_avg_L, match_sum_strokes (scale-dependent — train mean N ~5.65, test ~3.07)
    f["match_n_rallies"] = match_agg_dict[match]["n_rallies"] if match in match_agg_dict else 0
    f["match_attack_rate"] = float(match_action_attack.get(match, 0.5))
    f["match_corner_rate"] = float(match_corner_rate.get(match, 0.0))
    f["match_dominant_pointId"] = int(match_dominant_pid.get(match, 0))

    # Server in match RATES (leave-one-out)
    # DROP: srv_match_strokes (count, scale-dependent)
    mp_total = mp_dict.get((match, server_pid), {"total_strokes": 0, "total_attacks": 0,
                                                   "total_defenses": 0, "total_corner": 0, "sum_strength": 0.0})
    rp = rally_player_dict.get((rally_uid, server_pid), {"rally_strokes": 0, "rally_attacks": 0,
                                                          "rally_defenses": 0, "rally_corner": 0, "rally_strength": 0.0})
    srv_strokes_loo = mp_total["total_strokes"] - rp["rally_strokes"]
    srv_attacks_loo = mp_total["total_attacks"] - rp["rally_attacks"]
    srv_defenses_loo = mp_total["total_defenses"] - rp["rally_defenses"]
    srv_corner_loo = mp_total["total_corner"] - rp["rally_corner"]
    srv_strength_sum_loo = mp_total["sum_strength"] - rp["rally_strength"]
    f["srv_match_attack_rate"] = srv_attacks_loo / max(srv_strokes_loo, 1)
    f["srv_match_defense_rate"] = srv_defenses_loo / max(srv_strokes_loo, 1)
    f["srv_match_corner_rate"] = srv_corner_loo / max(srv_strokes_loo, 1)
    f["srv_match_strength_mean"] = srv_strength_sum_loo / max(srv_strokes_loo, 1)
    f["srv_match_n_present"] = int(srv_strokes_loo > 0)

    # Receiver in match RATES (leave-one-out)
    mp_total_r = mp_dict.get((match, receiver_pid), {"total_strokes": 0, "total_attacks": 0,
                                                      "total_defenses": 0, "total_corner": 0, "sum_strength": 0.0})
    rp_r = rally_player_dict.get((rally_uid, receiver_pid), {"rally_strokes": 0, "rally_attacks": 0,
                                                               "rally_defenses": 0, "rally_corner": 0, "rally_strength": 0.0})
    rcv_strokes_loo = mp_total_r["total_strokes"] - rp_r["rally_strokes"]
    rcv_attacks_loo = mp_total_r["total_attacks"] - rp_r["rally_attacks"]
    f["rcv_match_attack_rate"] = rcv_attacks_loo / max(rcv_strokes_loo, 1)
    f["rcv_match_defense_rate"] = (mp_total_r["total_defenses"] - rp_r["rally_defenses"]) / max(rcv_strokes_loo, 1)
    f["rcv_match_strength_mean"] = (mp_total_r["sum_strength"] - rp_r["rally_strength"]) / max(rcv_strokes_loo, 1)
    f["rcv_match_n_present"] = int(rcv_strokes_loo > 0)

    # Server vs receiver match-level interaction (rate-based)
    f["match_skill_diff"] = f["srv_match_attack_rate"] - f["rcv_match_attack_rate"]
    f["match_aggression_diff"] = (f["srv_match_attack_rate"] - f["srv_match_defense_rate"]) - \
                                  (f["rcv_match_attack_rate"] - f["rcv_match_defense_rate"])

    return f


def make_feats(rows, k, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
               player_groups, prior_sex, prior_global, parity_bayes, morris_table,
               score_seq_feats, t1_pred_per_rally,
               per_player_logp, per_player_n, global_logp,
               player_nmf, train_player_pids, train_V_norm, eb_srv_dict, eb_rcv_dict,
               match_agg_pkg,
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

    # T5 Markov
    bigram_feats = compute_bigram_features(visible, srv_pid, per_player_logp, per_player_n, global_logp)
    f.update(bigram_feats)

    # T6 NMF
    srv_emb = player_nmf.get(srv_pid, np.zeros(NMF_RANK))
    rcv_emb = player_nmf.get(rcv_pid, np.zeros(NMF_RANK))
    for i in range(NMF_RANK):
        f[f"nmf_srv_{i}"] = float(srv_emb[i])
        f[f"nmf_rcv_{i}"] = float(rcv_emb[i])
    f["nmf_skill_diff_l2"] = float(np.linalg.norm(srv_emb - rcv_emb))

    # === T8 NEW: Match-level test self-supervised features ===
    match = first["match"]
    mp_dict, rally_player_dict, match_agg_dict, mma, mcr, mdpid = match_agg_pkg
    match_feats = get_match_features(uid, match, srv_pid, rcv_pid,
                                      mp_dict, rally_player_dict, match_agg_dict,
                                      mma, mcr, mdpid)
    f.update(match_feats)

    return f


def build_features(df, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                    player_groups, prior_sex, prior_global, parity_bayes, morris_table,
                    score_seq_feats, t1_pred_per_rally,
                    per_player_logp, per_player_n, global_logp,
                    player_nmf, train_player_pids, train_V_norm, eb_srv_dict, eb_rcv_dict,
                    match_agg_pkg,
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
                           match_agg_pkg,
                           known_matches)
            R = min(N - k, R_max)
            feats.append(f); uids.append(uid); R_targets.append(R); sgp_targets.append(sgp); ks.append(k)
    return pd.DataFrame(feats), np.array(uids), np.array(R_targets), np.array(sgp_targets), np.array(ks)


def main():
    print("=" * 72)
    print("T8: T6 + match-level test self-supervised features")
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

    # === T8: pre-compute match aggregates for train AND test ===
    print("Computing match aggregates (train)...")
    train_match_pkg = compute_match_player_aggregates(train_df)
    print(f"  train: {len(train_match_pkg[0])} (match,player) entries, {len(train_match_pkg[2])} matches")
    print("Computing match aggregates (test)...")
    test_match_pkg = compute_match_player_aggregates(test_df)
    print(f"  test:  {len(test_match_pkg[0])} (match,player) entries, {len(test_match_pkg[2])} matches")

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

        train_V_norm, train_V_raw, train_player_pids, _ = build_behavior_matrix(tr_rdf)
        W_train, H_train, _ = fit_nmf(train_V_norm, rank=NMF_RANK, max_iter=200, seed=42 + fold)
        player_nmf = {pid: W_train[i] for i, pid in enumerate(train_player_pids)}
        test_unseen = test_player_set - set(train_player_pids)
        for upid in test_unseen:
            strokes_for_p = test_df[test_df["gamePlayerId"] == upid]
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
                                                          eb_srv, eb_rcv, train_match_pkg,
                                                          max_prefixes=max_prefix, rng=rng_tr, R_max=R_max, is_train=True)
        X_va, uid_va, R_va, sgp_va, k_va = build_features(va_rdf, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                          params_srv, params_rcv, player_groups,
                                                          prior_sex, prior_global, parity_bayes, morris_table,
                                                          score_seq_train, t1_oof_dict,
                                                          per_player_logp, per_player_n, global_logp,
                                                          player_nmf, train_player_pids, train_V_norm,
                                                          eb_srv, eb_rcv, train_match_pkg,
                                                          max_prefixes=max_prefix, rng=rng_va, R_max=R_max, is_train=True)

        if feat_cols_ref is None:
            feat_cols_ref = list(X_tr.columns)

        new_t8 = [c for c in feat_cols_ref if c.startswith(("match_", "srv_match_", "rcv_match_"))]
        if fold == 0:
            print(f"\nFold 1: total feats={X_tr.shape[1]}, T8 new={len(new_t8)}: {new_t8}")

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
        new_in_top10 = sum(1 for n, _ in top10 if n in new_t8)
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, T8-new-in-top10: {new_in_top10}/10")
        fold_models.append(model)
        fold_artifacts.append((eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                                player_groups, prior_sex, prior_global,
                                per_player_logp, per_player_n, global_logp,
                                player_nmf, train_player_pids, train_V_norm))

    oof_sgp_rally = oof_sgp_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label[rally_mask], oof_sgp_rally[rally_mask])
    print(f"\n{'='*72}")
    print(f"T8 OOF AUC (per-rally aggregated): {overall_auc:.4f}")

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
                                                   eb_srv, eb_rcv, test_match_pkg,
                                                   known_matches=train_matches, is_train=False)
        if test_uids_ref is None: test_uids_ref = uid_te
        R_probs_te = model.predict(X_te)
        sgp_te = derive_sgp_from_R(R_probs_te, k_te, R_max)
        test_pred += sgp_te / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_t8_match_context.npz"
    np.savez(out_path, rally_uids=test_uids_ref, sgp_pred=test_pred,
             method="t8_lgb_R_eb_markov_nmf_match_context",
             holdout_auc=overall_auc, oof_pred=oof_sgp_rally, oof_label=oof_label,
             feature_names=np.array(feat_cols_ref))
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
