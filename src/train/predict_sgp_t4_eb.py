#!/usr/bin/env python3
"""
T4: T3 + Multi-level Empirical Bayes partial pooling (replaces κ=50 fixed shrinkage).

Key differences from T3:
- Per-player EB shrinkage uses method-of-moments κ̂ (data-driven)
- Group-level partial pooling on logit scale (sex × handedness × style)
- Exposes uncertainty as feature: eb_post_se, eb_shrinkage_λ
- Cold-start fallback: group mean (better than sex-marginal)

Usage:
    python3 src/train/predict_sgp_t4_eb.py
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


ACTION_GROUPS = {a: g for a, g in [
    *[(i, 1) for i in [1,2,3,4,5,6,7]],
    *[(i, 2) for i in [8,9,10,11]],
    *[(i, 3) for i in [12,13,14]],
    *[(i, 4) for i in [15,16,17,18]],
    (0, 0)
]}
POINT_DEPTH = {0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3}
POINT_SIDE  = {0:0,1:1,2:2,3:3,4:1,5:2,6:3,7:1,8:2,9:3}
SIDE_X = {0:0, 1:-1, 2:0, 3:1}
DEPTH_Y = {0:0, 1:1, 2:2, 3:3}
CORNER_ZONES = {1, 3, 7, 9}


def get_xy(point_id):
    return SIDE_X[POINT_SIDE.get(point_id, 0)], DEPTH_Y[POINT_DEPTH.get(point_id, 0)]


def classify_player_style(stats_row):
    """Bin player into {attacker=0, controller=1, defender=2}."""
    total = max(1, stats_row["n_strokes"])
    a = stats_row["n_attacks"] / total
    d = stats_row["n_defenses"] / total
    if a >= 0.35: return 0  # attacker
    if d >= 0.20: return 2  # defender
    return 1  # controller


def compute_player_aggregates(train_df, restrict_uids=None):
    """Per-player rally + stroke aggregates from train_df (optionally restricted)."""
    if restrict_uids is not None:
        df = train_df[train_df["rally_uid"].isin(restrict_uids)]
    else:
        df = train_df

    # Per-stroke counts (action group, hand)
    per_stroke = df.groupby("gamePlayerId").agg(
        n_strokes=("actionId", "size"),
    ).reset_index()
    # Action group counts
    df_grp = df.copy()
    df_grp["ag"] = df_grp["actionId"].map(ACTION_GROUPS).fillna(0).astype(int)
    ag_counts = df_grp.groupby(["gamePlayerId", "ag"]).size().unstack(fill_value=0)
    ag_counts.columns = [f"n_ag_{c}" for c in ag_counts.columns]
    # Hand majority
    hand_counts = df.groupby(["gamePlayerId", "handId"]).size().unstack(fill_value=0)
    hand_majority = hand_counts.idxmax(axis=1)
    # Sex (constant per player)
    sex_per_player = df.groupby("gamePlayerId")["sex"].first()

    # Per-rally (when this player is server) — use rally_meta
    rally_meta = df.groupby("rally_uid").first()[
        ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex"]
    ]
    srv_stats = rally_meta.groupby("gamePlayerId").agg(
        n_srv=("serverGetPoint", "size"),
        w_srv=("serverGetPoint", "sum"),
    )
    rcv_stats = rally_meta.groupby("gamePlayerOtherId").agg(
        n_rcv=("serverGetPoint", "size"),
        w_rcv=("serverGetPoint", "sum"),  # NOTE: this is from server's perspective
    )

    out = per_stroke.set_index("gamePlayerId").join(ag_counts, how="left").fillna(0)
    out["hand_majority"] = hand_majority
    out["sex"] = sex_per_player
    out = out.join(srv_stats, how="left").fillna(0)
    out = out.join(rcv_stats, how="left").fillna(0)
    out["n_attacks"] = out.get("n_ag_1", 0)
    out["n_defenses"] = out.get("n_ag_3", 0)

    # Style classification
    out["style"] = out.apply(classify_player_style, axis=1)
    return out  # index = gamePlayerId


def compute_multilevel_eb(player_aggs, side="srv"):
    """
    Multi-level EB partial pooling for player skill on logit scale.
    side: 'srv' (server-side winrate) or 'rcv' (receiver-side loserate from sgp=1 means receiver lost).
    Returns dict keyed by player_id with EB stats.
    """
    n_col = f"n_{side}"
    w_col = f"w_{side}"
    if side == "srv":
        # P(server wins) → keep as is
        df = player_aggs[[n_col, w_col, "sex", "hand_majority", "style"]].copy()
        df["p_hat"] = df[w_col] / df[n_col].replace(0, 1)
    else:
        # For receiver: P(receiver loses) = sgp=1 from receiver perspective
        # rcv_stats: w_rcv = sum of sgp when this player was receiver = how many times they lost
        # P(loses as receiver) = w_rcv / n_rcv
        df = player_aggs[[n_col, w_col, "sex", "hand_majority", "style"]].copy()
        df["p_hat"] = df[w_col] / df[n_col].replace(0, 1)

    # Filter players with enough observations
    df_ok = df[df[n_col] > 0].copy()
    if len(df_ok) == 0:
        return {}, {}, {}

    # Method of moments — global level
    p_bar = float(df_ok[w_col].sum() / df_ok[n_col].sum())
    var_p = float(df_ok["p_hat"].var())
    expected_binom_var = float((p_bar * (1 - p_bar) * (1 / df_ok[n_col])).mean())
    sigma2_btw_global = max(1e-6, var_p - expected_binom_var)
    kappa_global = p_bar * (1 - p_bar) / sigma2_btw_global - 1
    kappa_global = max(0.5, min(kappa_global, 1000))  # clamp

    # Single-level EB
    df_ok["p_eb1"] = (kappa_global * p_bar + df_ok[w_col]) / (kappa_global + df_ok[n_col])
    df_ok["p_eb1_clamped"] = df_ok["p_eb1"].clip(0.01, 0.99)
    df_ok["theta_eb1"] = np.log(df_ok["p_eb1_clamped"] / (1 - df_ok["p_eb1_clamped"]))
    # Fisher var on logit scale: v_i = 1 / (n_i * p̂(1-p̂))
    p_hat_clamped = df_ok["p_hat"].clip(0.01, 0.99)
    df_ok["v_logit"] = 1.0 / (df_ok[n_col] * p_hat_clamped * (1 - p_hat_clamped))
    df_ok["v_logit"] = df_ok["v_logit"].clip(1e-3, 100)

    # Group key: sex × hand × style
    df_ok["group"] = (df_ok["sex"].astype(int).astype(str) + "_" +
                      df_ok["hand_majority"].astype(int).astype(str) + "_" +
                      df_ok["style"].astype(int).astype(str))

    # Compute group means on logit scale (precision-weighted)
    group_means = {}
    for g, sub in df_ok.groupby("group"):
        weights = 1.0 / sub["v_logit"].values
        if weights.sum() <= 0:
            group_means[g] = float(np.log(p_bar / (1 - p_bar)))
        else:
            group_means[g] = float(np.sum(sub["theta_eb1"].values * weights) / weights.sum())

    # ANOVA-style σ²_w (within-group spread of θ̂ minus precision contribution)
    df_ok["mu_g"] = df_ok["group"].map(group_means)
    residuals_sq = (df_ok["theta_eb1"] - df_ok["mu_g"]) ** 2
    sigma2_w = float(max(1e-3, residuals_sq.mean() - df_ok["v_logit"].mean()))
    # Between-group σ²_b
    group_mean_array = np.array(list(group_means.values()))
    global_logit = float(np.log(p_bar / (1 - p_bar)))
    sigma2_b = float(max(1e-3, ((group_mean_array - global_logit) ** 2).mean()))

    # Two-level partial pooling: shrink θ̂_i toward μ̂_g[i] with weight from σ²_w
    df_ok["theta_pooled"] = ((df_ok["mu_g"] / sigma2_w + df_ok["theta_eb1"] / df_ok["v_logit"]) /
                              (1.0 / sigma2_w + 1.0 / df_ok["v_logit"]))
    df_ok["p_pooled"] = 1.0 / (1.0 + np.exp(-df_ok["theta_pooled"]))
    df_ok["shrinkage_lambda"] = df_ok[n_col] / (df_ok[n_col] + kappa_global)
    # Posterior SE on logit scale
    df_ok["post_se_logit"] = np.sqrt(1.0 / (1.0 / sigma2_w + 1.0 / df_ok["v_logit"]))

    eb_per_player = df_ok[[
        "p_pooled", "theta_pooled", "shrinkage_lambda", "post_se_logit", "mu_g", "n_srv" if side == "srv" else "n_rcv"
    ]].rename(columns={
        "p_pooled": f"eb_{side}_p",
        "theta_pooled": f"eb_{side}_logit",
        "shrinkage_lambda": f"eb_{side}_lambda",
        "post_se_logit": f"eb_{side}_post_se",
        "mu_g": f"eb_{side}_group_mean",
        "n_srv": f"eb_{side}_n", "n_rcv": f"eb_{side}_n",
    }).to_dict("index")

    return eb_per_player, group_means, {
        "p_bar": p_bar, "kappa": kappa_global, "sigma2_w": sigma2_w, "sigma2_b": sigma2_b,
        "global_logit": global_logit
    }


def get_eb_features(pid, side, eb_dict, group_means, params, fallback_group_key):
    """Get EB features for a player; fallback to group mean if unseen."""
    if pid in eb_dict:
        e = eb_dict[pid]
        return {
            f"eb_{side}_p": e[f"eb_{side}_p"],
            f"eb_{side}_logit": e[f"eb_{side}_logit"],
            f"eb_{side}_lambda": e[f"eb_{side}_lambda"],
            f"eb_{side}_post_se": e[f"eb_{side}_post_se"],
            f"eb_{side}_group_mean": e[f"eb_{side}_group_mean"],
            f"eb_{side}_n": e[f"eb_{side}_n"],
            f"eb_{side}_seen": 1,
        }
    # Cold-start fallback
    mu_g = group_means.get(fallback_group_key, params.get("global_logit", 0))
    p_g = 1.0 / (1.0 + np.exp(-mu_g))
    return {
        f"eb_{side}_p": p_g,
        f"eb_{side}_logit": mu_g,
        f"eb_{side}_lambda": 0.0,
        f"eb_{side}_post_se": np.sqrt(params.get("sigma2_w", 1.0) + params.get("sigma2_b", 1.0)),
        f"eb_{side}_group_mean": mu_g,
        f"eb_{side}_n": 0,
        f"eb_{side}_seen": 0,
    }


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


def compute_morris_importance(target=11, p=0.55):
    memo = {}
    def P_game(s_s, s_r):
        if s_s >= target and s_s >= s_r + 2: return 1.0
        if s_r >= target and s_r >= s_s + 2: return 0.0
        if (s_s, s_r) in memo: return memo[(s_s, s_r)]
        if s_s + s_r > 50: return 0.5
        result = p * P_game(s_s + 1, s_r) + (1 - p) * P_game(s_s, s_r + 1)
        memo[(s_s, s_r)] = result
        return result
    importance = {}
    for s_s in range(target + 10):
        for s_r in range(target + 10):
            importance[(s_s, s_r)] = abs(P_game(s_s + 1, s_r) - P_game(s_s, s_r + 1))
    return importance


def service_block_pos(score_self, score_other):
    if score_self >= 10 and score_other >= 10: return 1
    return ((score_self + score_other) % 2) + 1


def build_score_sequence_features(df):
    out = {}
    rally_meta = df.groupby("rally_uid").agg(
        match=("match", "first"), numberGame=("numberGame", "first"),
        scoreSelf=("scoreSelf", "first"), scoreOther=("scoreOther", "first"),
        gamePlayerId=("gamePlayerId", "first"), gamePlayerOtherId=("gamePlayerOtherId", "first"),
        N_strokes=("strikeNumber", "size"),
    ).reset_index()
    for (m, g), grp in rally_meta.groupby(["match", "numberGame"]):
        grp_sorted = grp.copy()
        grp_sorted["total"] = grp_sorted["scoreSelf"] + grp_sorted["scoreOther"]
        grp_sorted = grp_sorted.sort_values(by=["total", "scoreSelf"]).reset_index(drop=True)
        prev_winner = None
        run_len = 0
        cum_strokes = defaultdict(int)
        for i, row in grp_sorted.iterrows():
            uid = row["rally_uid"]; cur_server = row["gamePlayerId"]
            prev_won = int(prev_winner == cur_server) if prev_winner is not None else -1
            if prev_winner == cur_server: run_len += 1
            else: run_len = 0
            out[uid] = {
                "prev_won": prev_won, "run_len": run_len,
                "cum_strokes": cum_strokes[cur_server],
            }
            # Determine winner of this rally for next iteration
            if i + 1 < len(grp_sorted):
                nxt = grp_sorted.iloc[i + 1]
                if nxt["scoreSelf"] > row["scoreSelf"] and nxt["gamePlayerId"] == cur_server:
                    prev_winner = cur_server
                elif nxt["scoreOther"] > row["scoreOther"] and nxt["gamePlayerOtherId"] == row["gamePlayerOtherId"]:
                    prev_winner = row["gamePlayerOtherId"]
                elif nxt["gamePlayerId"] == row["gamePlayerOtherId"]:
                    if nxt["scoreSelf"] > row["scoreOther"]:
                        prev_winner = row["gamePlayerOtherId"]
                    else:
                        prev_winner = cur_server
                else:
                    prev_winner = None
            cum_strokes[cur_server] += row["N_strokes"]
    return out


def make_feats(rows, k, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
               player_groups, prior_sex, prior_global, parity_bayes, morris_table,
               score_seq_feats, t1_pred_per_rally, known_matches=None):
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

    # T2 features
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

    # === T4 NEW: Multi-level EB features for server and receiver ===
    srv_group_key = player_groups.get(srv_pid, "1_1_1")
    rcv_group_key = player_groups.get(rcv_pid, "1_1_1")
    eb_srv_feats = get_eb_features(srv_pid, "srv", eb_srv, srv_groups, params_srv, srv_group_key)
    eb_rcv_feats = get_eb_features(rcv_pid, "rcv", eb_rcv, rcv_groups, params_rcv, rcv_group_key)
    f.update(eb_srv_feats); f.update(eb_rcv_feats)

    # Useful EB derived
    f["eb_skill_diff"] = eb_srv_feats["eb_srv_logit"] - eb_rcv_feats["eb_rcv_logit"]
    f["eb_combined_p"] = 0.5 * (eb_srv_feats["eb_srv_p"] + (1 - eb_rcv_feats["eb_rcv_p"]))
    f["eb_uncertainty"] = eb_srv_feats["eb_srv_post_se"] + eb_rcv_feats["eb_rcv_post_se"]

    return f


def build_features(df, eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv,
                    player_groups, prior_sex, prior_global, parity_bayes, morris_table,
                    score_seq_feats, t1_pred_per_rally,
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
                           score_seq_feats, t1_pred_per_rally, known_matches)
            R = min(N - k, R_max)
            feats.append(f); uids.append(uid); R_targets.append(R); sgp_targets.append(sgp); ks.append(k)
    return pd.DataFrame(feats), np.array(uids), np.array(R_targets), np.array(sgp_targets), np.array(ks)


def derive_sgp_from_R(R_probs, ks, R_max):
    sgp = np.zeros(len(ks))
    for i, k in enumerate(ks):
        for r in range(R_max + 1):
            if (k + r) % 2 == 0:
                sgp[i] += R_probs[i, r]
    return sgp


def main():
    print("=" * 72)
    print("T4: T3 + Multi-level Empirical Bayes partial pooling")
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

    # T1 stacking
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
    fold_artifacts = []  # store EB stuff per fold for test inference
    feat_cols_ref = None

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(rally_meta, uid_arr, groups=match_arr)):
        tr_meta = rally_meta.iloc[tr_idx]; va_meta = rally_meta.iloc[va_idx]
        tr_uids = set(tr_meta["rally_uid"]); va_uids = set(va_meta["rally_uid"])

        # Per-fold player aggregates from training-fold rallies only
        player_aggs = compute_player_aggregates(train_df, restrict_uids=tr_uids)
        # Multi-level EB for server and receiver
        eb_srv, srv_groups, params_srv = compute_multilevel_eb(player_aggs, side="srv")
        eb_rcv, rcv_groups, params_rcv = compute_multilevel_eb(player_aggs, side="rcv")
        # Player → group key
        player_groups = {pid: f"{int(row['sex'])}_{int(row['hand_majority'])}_{int(row['style'])}"
                          for pid, row in player_aggs.iterrows()}

        prior_global = float(tr_meta["serverGetPoint"].mean())
        prior_sex = {s: float(tr_meta[tr_meta["sex"]==s]["serverGetPoint"].mean()) for s in tr_meta["sex"].unique()}

        tr_rdf = train_df[train_df["rally_uid"].isin(tr_uids)]
        va_rdf = train_df[train_df["rally_uid"].isin(va_uids)]
        rng_tr = np.random.RandomState(42 + fold); rng_va = np.random.RandomState(1000 + fold)
        X_tr, uid_tr, R_tr, sgp_tr, k_tr = build_features(tr_rdf, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                          params_srv, params_rcv, player_groups,
                                                          prior_sex, prior_global, parity_bayes, morris_table,
                                                          score_seq_train, t1_oof_dict,
                                                          max_prefixes=max_prefix, rng=rng_tr, R_max=R_max, is_train=True)
        X_va, uid_va, R_va, sgp_va, k_va = build_features(va_rdf, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                          params_srv, params_rcv, player_groups,
                                                          prior_sex, prior_global, parity_bayes, morris_table,
                                                          score_seq_train, t1_oof_dict,
                                                          max_prefixes=max_prefix, rng=rng_va, R_max=R_max, is_train=True)

        if feat_cols_ref is None:
            feat_cols_ref = list(X_tr.columns)

        T3_FEAT = set([
            # T1+T2+T3 base
            "ctx_len","sex","numberGame","score_self_start","score_other_start","score_diff_start",
            "score_sum_start","is_deuce_start","match_seen","last_strikeId","last_handId",
            "last_strengthId","last_spinId","last_pointId","last_actionId","last_positionId",
            "last_action_group","last_point_depth","last_point_side","last_score_diff",
            "parity_bayes_unif","n_attacks","n_controls","n_defenses","n_serves",
            "n_srv_attacks","n_rcv_attacks","n_srv_strokes","n_rcv_strokes",
            "prev_actionId","prev_pointId","prev_action_group","prev_handId","prev_spinId",
            "last_x","last_y","last_dx","last_dy","last_distance","last_angle",
            "is_diagonal_last","is_cross_court_last","total_path_length","mean_seg_distance",
            "max_seg_distance","diagonal_count","diagonal_ratio","recent_x_drift","recent_y_drift",
            "unique_zones","unique_zones_ratio","n_corner_strokes","corner_ratio","last_is_corner",
            "dominance_index","server_aggression_balance","receiver_aggression_balance",
            "max_attack_chain","last3_aggressor_score","pressure_x_dominance","pos_x_depth","pos_x_side",
            "strength_last","strength_mean","strength_max","strength_last_minus_prev",
            "prev_point_won_by_server","server_run_length","cum_strokes_in_match",
            "service_block_position","morris_importance","is_set_point_for_server",
            "is_set_point_against","t1_stacking_pred",
        ])
        new_t4 = [c for c in feat_cols_ref if c not in T3_FEAT]
        if fold == 0:
            print(f"\nFold {fold+1}/{n_folds}: Tr {len(X_tr)} samples, Va {len(X_va)}, total feats={X_tr.shape[1]}")
            print(f"  T4 new ({len(new_t4)}): {new_t4}")

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
        new_in_top10 = sum(1 for n, _ in top10 if n in new_t4)
        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}, T4-new-in-top10: {new_in_top10}/10")
        fold_models.append(model)
        fold_artifacts.append((eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv, player_groups,
                                prior_sex, prior_global))

    oof_sgp_rally = oof_sgp_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label[rally_mask], oof_sgp_rally[rally_mask])
    print(f"\n{'='*72}")
    print(f"T4 OOF AUC (per-rally aggregated): {overall_auc:.4f}")

    # Top-25 features last fold
    fi_last = pd.DataFrame({
        "feature": feat_cols_ref,
        "importance": fold_models[-1].feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\nTop 25 features by gain (last fold):")
    print(fi_last.head(25).to_string(index=False))

    # Test prediction
    print("\n--- Test prediction ---")
    test_pred = np.zeros(test_df["rally_uid"].nunique())
    test_uids_ref = None
    for model, (eb_srv, eb_rcv, srv_groups, rcv_groups, params_srv, params_rcv, player_groups,
                prior_sex, prior_global) in zip(fold_models, fold_artifacts):
        X_te, uid_te, _, _, k_te = build_features(test_df, eb_srv, eb_rcv, srv_groups, rcv_groups,
                                                   params_srv, params_rcv, player_groups,
                                                   prior_sex, prior_global, parity_bayes, morris_table,
                                                   score_seq_test, t1_test_dict,
                                                   known_matches=train_matches, is_train=False)
        if test_uids_ref is None: test_uids_ref = uid_te
        R_probs_te = model.predict(X_te)
        sgp_te = derive_sgp_from_R(R_probs_te, k_te, R_max)
        test_pred += sgp_te / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_t4_eb.npz"
    np.savez(out_path,
             rally_uids=test_uids_ref, sgp_pred=test_pred,
             method="t4_lgb_R_eb_multilevel",
             holdout_auc=overall_auc, oof_pred=oof_sgp_rally, oof_label=oof_label,
             feature_names=np.array(feat_cols_ref))
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
