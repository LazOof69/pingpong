#!/usr/bin/env python3
"""
V5-Z (Plan Z): V5 base + forward-only aux next-token head.

Diff vs V5:
  - LGB feature 'sgp' removed (official: leakage via per-rally outcome).
  - BiLSTM forward states drive an auxiliary next-token head that
    supervises every intra-prefix position. No leakage (causal).
  - Trained on new train.csv (14,995 rallies, mean 5.65 strokes).
  - Aux loss weight: AUX_LOSS_W. Main loss unchanged.
  - EPOCHS=40 PATIENCE=10 (more data, faster convergence).
"""

import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter
import lightgbm as lgb
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 超參數
# ============================================================
SEED = int(os.environ.get("V5Z_SEED", 42))
MAX_SEQ_LEN = 50
EMBED_DIM = 20
PLAYER_EMBED_DIM = 16
HIDDEN_DIM = int(os.environ.get("V5Z_HIDDEN", 192))
N_LSTM_LAYERS = int(os.environ.get("V5Z_LAYERS", 2))
N_ATTN_HEADS = 4
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = int(os.environ.get("V5Z_EPOCHS", 40))
PATIENCE = int(os.environ.get("V5Z_PATIENCE", 10))
N_FOLDS = 5
LABEL_SMOOTH = 0.05
WEIGHT_ALPHA = 0.5
PLAYER_DROP_P = 0.3
TAG = os.environ.get("V5Z_TAG", "v5z")
AV2_S0 = bool(int(os.environ.get("V5Z_AV2_S0", "0")))  # in-prefix per-side observed stats
AV2 = bool(int(os.environ.get("V5Z_AV2", "0")))         # full historical player stats (per-fold safe)
EB = bool(int(os.environ.get("V5Z_EB", "0")))           # Empirical-Bayes from prefix-only + global Beta prior shrinkage
EB_KAPPA = float(os.environ.get("V5Z_EB_KAPPA", "30"))  # shrinkage strength
PHASE = bool(int(os.environ.get("V5Z_PHASE", "0")))     # Wu Huanqun three-phase + EFOS + AOEL features
MULTITASK = bool(int(os.environ.get("V5Z_MULTITASK", "0")))  # MuLMINet-style aux heads (spin/hand/strength)
MT_W = float(os.environ.get("V5Z_MT_W", "0.15"))             # weight per aux head
MATCH_DISJOINT = bool(int(os.environ.get("V5Z_MATCH_DISJOINT", "0")))  # use StratifiedGroupKFold(groups=match_id) instead of StratifiedKFold
ARCH = os.environ.get("V5Z_ARCH", "lstm").lower()
# Aux next-token loss is causal-only — not safe with bidirectional encoders.
AUX_LOSS_W = 0.0 if ARCH in ("transformer", "mamba") else 0.1
FOCAL_GAMMA = float(os.environ.get("V5Z_FOCAL", "0"))
SOFTF1_W = float(os.environ.get("V5Z_SOFTF1", "0"))  # 0..1 weight on soft macro-F1
SGP_AUX_W = float(os.environ.get("V5Z_SGP_AUX_W", "0"))  # BCE loss on rally sgp; 0 = disabled (Candidate H)
SGP_ONLY = bool(int(os.environ.get("V5Z_SGP_ONLY", "0")))  # Pure-sgp training: skip action/point/aux losses
TEST_CSV = os.environ.get("V5Z_TEST_CSV", "data/test.csv")
INFER_TEST_ONLY = bool(int(os.environ.get("V5Z_INFER_TEST_ONLY", "0")))
# When loading existing weights, PID2IDX must match the test CSV used at training time.
# Default to "data/test.csv" in INFER_TEST_ONLY mode (training-time test) to keep player_embed shape stable.
PID_TEST_CSV = os.environ.get("V5Z_PID_TEST_CSV",
                              "data/test.csv" if INFER_TEST_ONLY else TEST_CSV)
LSTM_WEIGHTS_TAG = os.environ.get("V5Z_LSTM_WEIGHTS_TAG", "")  # empty → use TAG

N_ACTION = 19
N_POINT = 10
SERVE_ACTIONS = [15, 16, 17, 18]

# Ensemble 權重: LSTM vs LGB
W_LSTM = 0.4
W_LGB = 0.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. 載入資料
# ============================================================
log("=" * 60)
log(f"Device: {DEVICE}")
log("載入資料...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv(TEST_CSV)
TEST_HAS_SGP = "serverGetPoint" in test_df.columns
log(f"  Train: {train_df.shape[0]} rows, {train_df['rally_uid'].nunique()} rallies")
log(f"  Test:  {test_df.shape[0]} rows, {test_df['rally_uid'].nunique()} rallies (csv={TEST_CSV}, has_sgp={TEST_HAS_SGP})")

# 選手映射 — 必須與訓練時 PID2IDX 完全一致，否則 embedding shape mismatch 導致 load_state_dict 失敗
if PID_TEST_CSV != TEST_CSV:
    pid_test_df = pd.read_csv(PID_TEST_CSV)
    log(f"  PID test csv (different from inference test): {PID_TEST_CSV}")
else:
    pid_test_df = test_df
all_pids = sorted(
    set(train_df["gamePlayerId"].unique())
    | set(train_df["gamePlayerOtherId"].unique())
    | set(pid_test_df["gamePlayerId"].unique())
    | set(pid_test_df["gamePlayerOtherId"].unique())
)
PID2IDX = {pid: i + 1 for i, pid in enumerate(all_pids)}
N_PLAYERS = len(PID2IDX) + 1
log(f"  選手數: {len(PID2IDX)} (PID2IDX from train + {PID_TEST_CSV})")

# ============================================================
# 2. 序列特徵定義
# ============================================================
SEQ_CAT_SPEC = {
    "strikeId": 20, "handId": 4, "strengthId": 5, "spinId": 7,
    "pointId": 11, "actionId": 20, "positionId": 5,
}
SEQ_CAT_NAMES = list(SEQ_CAT_SPEC.keys())
N_SEQ_CATS = len(SEQ_CAT_NAMES)
N_SEQ_NUMS = 3
N_STATIC = 6

# 球種分組
ACTION_GROUPS = {}
for a in [1, 2, 3, 4, 5, 6, 7]:
    ACTION_GROUPS[a] = 1  # attack
for a in [8, 9, 10, 11]:
    ACTION_GROUPS[a] = 2  # control
for a in [12, 13, 14]:
    ACTION_GROUPS[a] = 3  # defense
for a in [15, 16, 17, 18]:
    ACTION_GROUPS[a] = 4  # serve
ACTION_GROUPS[0] = 0  # unknown

# 落點分組
POINT_DEPTH = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
POINT_SIDE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}


# ============================================================
# 3. LSTM 資料準備
# ============================================================
def prepare_samples(df, is_train=True, augment=True):
    samples = []
    grouped = df.groupby("rally_uid")
    total = len(grouped)

    for count, (uid, grp) in enumerate(grouped):
        if (count + 1) % 3000 == 0:
            log(f"  {count+1}/{total}...")

        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        first = rows[0]

        def _make(ctx_rows, target_row=None):
            k = len(ctx_rows)
            last_ctx = ctx_rows[-1]

            seq_cat = np.zeros((k, N_SEQ_CATS), dtype=np.int64)
            for i, r in enumerate(ctx_rows):
                for j, feat in enumerate(SEQ_CAT_NAMES):
                    seq_cat[i, j] = int(r[feat])

            seq_num = np.zeros((k, N_SEQ_NUMS), dtype=np.float32)
            for i, r in enumerate(ctx_rows):
                seq_num[i, 0] = r["scoreSelf"] / 15.0
                seq_num[i, 1] = r["scoreOther"] / 15.0
                seq_num[i, 2] = r["strikeNumber"] / 50.0

            pid_server = PID2IDX.get(first["gamePlayerId"], 0)
            pid_receiver = PID2IDX.get(first["gamePlayerOtherId"], 0)
            next_sn = k + 1
            pid_next = pid_server if next_sn % 2 == 1 else pid_receiver

            static = np.array([
                first["sex"],
                first["numberGame"] / 7.0,
                k / 50.0,
                (next_sn) % 2,
                (last_ctx["scoreSelf"] - last_ctx["scoreOther"]) / 15.0,
                (last_ctx["scoreSelf"] + last_ctx["scoreOther"]) / 30.0,
            ], dtype=np.float32)

            sample = {
                "uid": uid,
                "seq_cat": seq_cat,
                "seq_num": seq_num,
                "pid_server": pid_server,
                "pid_receiver": pid_receiver,
                "pid_next": pid_next,
                "static": static,
                "length": k,
                "next_strike_num": next_sn,
                "sgp_label": int(first["serverGetPoint"]) if "serverGetPoint" in first else 0,
            }

            if target_row is not None:
                sample["target_action"] = int(target_row["actionId"])
                sample["target_point"] = int(target_row["pointId"])
                # MuLMINet-style aux targets (next stroke's physical attributes)
                sample["target_spin"] = int(target_row["spinId"])
                sample["target_hand"] = int(target_row["handId"])
                sample["target_strength"] = int(target_row["strengthId"])

            return sample

        if is_train:
            start_k = 1 if augment else max(1, N - 1)
            for k in range(start_k, N):
                samples.append(_make(rows[:k], rows[k]))
        else:
            samples.append(_make(rows))

    return samples


# ============================================================
# 4. LightGBM 特徵工程
# ============================================================
def compute_player_stats(df, restrict_uids=None):
    """
    Per-player historical style stats. For per-fold leakage safety,
    pass restrict_uids = train fold's rally_uids (to exclude val).

    Returns:
      stats: dict[player_id] -> dict of 12 scalar features
      cold: dict[sex (1 or 2)] -> dict of same features (sex-marginal fallback)
      seen: set of player_ids present in df subset
    """
    if restrict_uids is not None:
        sub = df[df["rally_uid"].isin(restrict_uids)]
    else:
        sub = df

    rally_meta = sub.groupby("rally_uid").first()[
        ["gamePlayerId", "gamePlayerOtherId", "serverGetPoint", "sex"]
    ]
    rally_len = sub.groupby("rally_uid").size()

    feats = ["attack_ratio", "control_ratio", "defense_ratio",
             "forehand_ratio", "topspin_ratio",
             "short_ratio", "long_ratio",
             "fore_side_ratio", "back_side_ratio",
             "avg_rally_len", "winrate_as_srv", "winrate_as_rcv"]

    def stats_from_rows(rows):
        if len(rows) == 0:
            return {f: 0.0 for f in feats}
        n = len(rows)
        actions = rows["actionId"].values
        groups = np.array([ACTION_GROUPS.get(int(a), 0) for a in actions])
        points = rows["pointId"].values
        depths = np.array([POINT_DEPTH.get(int(p), 0) for p in points])
        sides = np.array([POINT_SIDE.get(int(p), 0) for p in points])
        return {
            "attack_ratio":   float((groups == 1).mean()),
            "control_ratio":  float((groups == 2).mean()),
            "defense_ratio":  float((groups == 3).mean()),
            "forehand_ratio": float((rows["handId"] == 1).mean()),
            "topspin_ratio":  float((rows["spinId"] == 1).mean()),
            "short_ratio":    float((depths == 1).mean()),
            "long_ratio":     float((depths == 3).mean()),
            "fore_side_ratio": float((sides == 1).mean()),
            "back_side_ratio": float((sides == 3).mean()),
        }

    # Per-player aggregations (over all strokes that player hit)
    stats = {}
    seen_players = set()
    for pid, grp in sub.groupby("gamePlayerId"):
        seen_players.add(int(pid))
        s = stats_from_rows(grp)
        # Rally-level stats
        as_srv_rallies = rally_meta[rally_meta["gamePlayerId"] == pid]
        as_rcv_rallies = rally_meta[rally_meta["gamePlayerOtherId"] == pid]
        avg_len = 0.0
        n_part = 0
        for uid in set(as_srv_rallies.index) | set(as_rcv_rallies.index):
            if uid in rally_len.index:
                avg_len += rally_len[uid]
                n_part += 1
        s["avg_rally_len"] = avg_len / max(n_part, 1)
        s["winrate_as_srv"] = float(as_srv_rallies["serverGetPoint"].mean()) if len(as_srv_rallies) else 0.0
        s["winrate_as_rcv"] = float(1 - as_rcv_rallies["serverGetPoint"].mean()) if len(as_rcv_rallies) else 0.0
        stats[int(pid)] = s

    # Sex-conditional cold-start marginals
    cold = {}
    for sex_v in [1, 2]:
        sex_uids = rally_meta[rally_meta["sex"] == sex_v].index
        sex_sub = sub[sub["rally_uid"].isin(sex_uids)]
        if len(sex_sub) == 0:
            cold[sex_v] = {f: 0.0 for f in feats}
            continue
        s = stats_from_rows(sex_sub)
        sex_meta = rally_meta[rally_meta["sex"] == sex_v]
        s["avg_rally_len"] = float(rally_len[sex_meta.index].mean()) if len(sex_meta) else 0.0
        s["winrate_as_srv"] = float(sex_meta["serverGetPoint"].mean()) if len(sex_meta) else 0.0
        s["winrate_as_rcv"] = 1.0 - s["winrate_as_srv"]
        cold[sex_v] = s

    return stats, cold, seen_players


def build_lgb_features(df, is_train=True, augment=True,
                       player_stats=None, cold_marginals=None, seen_players=None,
                       global_priors=None):
    """把每個 sample 轉成一列 flat features"""
    features = []
    labels_a, labels_p = [], []
    uids = []
    sgp_labels = []

    grouped = df.groupby("rally_uid")
    total = len(grouped)

    for count, (uid, grp) in enumerate(grouped):
        if (count + 1) % 3000 == 0:
            log(f"  [LGB] {count+1}/{total}...")

        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        first = rows[0]

        def _feats(ctx_rows, target_row=None):
            k = len(ctx_rows)
            last = ctx_rows[-1]
            next_sn = k + 1

            f = {}
            # 基本
            f["ctx_len"] = k
            f["next_sn"] = next_sn
            f["next_parity"] = next_sn % 2
            f["sex"] = first["sex"]
            f["numberGame"] = first["numberGame"]
            f["score_self"] = last["scoreSelf"]
            f["score_other"] = last["scoreOther"]
            f["score_diff"] = last["scoreSelf"] - last["scoreOther"]
            f["score_sum"] = last["scoreSelf"] + last["scoreOther"]
            f["player_self"] = first["gamePlayerId"]
            f["player_other"] = first["gamePlayerOtherId"]

            # 最後一拍的所有特徵
            f["last_strikeId"] = last["strikeId"]
            f["last_handId"] = last["handId"]
            f["last_strengthId"] = last["strengthId"]
            f["last_spinId"] = last["spinId"]
            f["last_pointId"] = last["pointId"]
            f["last_actionId"] = last["actionId"]
            f["last_positionId"] = last["positionId"]
            f["last_action_group"] = ACTION_GROUPS.get(last["actionId"], 0)
            f["last_point_depth"] = POINT_DEPTH.get(last["pointId"], 0)
            f["last_point_side"] = POINT_SIDE.get(last["pointId"], 0)

            # 倒數第二拍
            if k >= 2:
                prev = ctx_rows[-2]
                f["prev_actionId"] = prev["actionId"]
                f["prev_pointId"] = prev["pointId"]
                f["prev_handId"] = prev["handId"]
                f["prev_spinId"] = prev["spinId"]
                f["prev_strengthId"] = prev["strengthId"]
                f["prev_action_group"] = ACTION_GROUPS.get(prev["actionId"], 0)
            else:
                f["prev_actionId"] = -1
                f["prev_pointId"] = -1
                f["prev_handId"] = -1
                f["prev_spinId"] = -1
                f["prev_strengthId"] = -1
                f["prev_action_group"] = -1

            # 發球拍特徵 (永遠存在)
            serve = ctx_rows[0]
            f["serve_actionId"] = serve["actionId"]
            f["serve_pointId"] = serve["pointId"]
            f["serve_spinId"] = serve["spinId"]
            f["serve_handId"] = serve["handId"]
            f["serve_positionId"] = serve["positionId"]

            # 交互特徵
            f["last_action_x_point"] = last["actionId"] * 100 + last["pointId"]
            f["last_action_x_spin"] = last["actionId"] * 10 + last["spinId"]
            f["last_hand_x_point"] = last["handId"] * 100 + last["pointId"]
            f["serve_action_x_point"] = serve["actionId"] * 100 + serve["pointId"]

            # 統計: context 中各球種/落點出現次數
            action_counts = Counter(r["actionId"] for r in ctx_rows)
            for ag_name, ag_val in [("attack", 1), ("control", 2), ("defense", 3)]:
                f[f"ctx_{ag_name}_count"] = sum(
                    action_counts.get(a, 0) for a in range(19) if ACTION_GROUPS.get(a, 0) == ag_val
                )

            point_counts = Counter(r["pointId"] for r in ctx_rows)
            for d_name, d_val in [("short", 1), ("half", 2), ("long", 3)]:
                f[f"ctx_depth_{d_name}"] = sum(
                    point_counts.get(p, 0) for p in range(10) if POINT_DEPTH.get(p, 0) == d_val
                )
            for s_name, s_val in [("fore", 1), ("mid", 2), ("back", 3)]:
                f[f"ctx_side_{s_name}"] = sum(
                    point_counts.get(p, 0) for p in range(10) if POINT_SIDE.get(p, 0) == s_val
                )

            # 分數壓力
            f["is_deuce"] = int(last["scoreSelf"] >= 10 and last["scoreOther"] >= 10)
            f["server_game_point"] = int(last["scoreSelf"] >= 10 and last["scoreSelf"] > last["scoreOther"])
            f["receiver_game_point"] = int(last["scoreOther"] >= 10 and last["scoreOther"] > last["scoreSelf"])

            if AV2_S0:
                server_id = first["gamePlayerId"]
                srv_rows = [r for r in ctx_rows if r["gamePlayerId"] == server_id]
                rcv_rows = [r for r in ctx_rows if r["gamePlayerId"] != server_id]
                f["srv_strokes_in_prefix"] = len(srv_rows)
                f["rcv_strokes_in_prefix"] = len(rcv_rows)
                f["srv_attack_in_prefix"] = sum(1 for r in srv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 1)
                f["rcv_attack_in_prefix"] = sum(1 for r in rcv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 1)
                f["srv_forehand_in_prefix"] = sum(1 for r in srv_rows if r["handId"] == 1)
                f["rcv_forehand_in_prefix"] = sum(1 for r in rcv_rows if r["handId"] == 1)

            if PHASE:
                # Wu Huanqun three-phase + Wu/Lames EFOS + Wu 2022 AOEL features.
                # All computed from prefix only (leakage-safe).
                next_sn_local = k + 1
                # Phase: STAS=odd next strikes (server attacks 1,3,5...), RTAS=even (receiver attacks 2,4),
                # Stalemate=stroke 6+
                if next_sn_local >= 6:
                    next_phase = 3  # stalemate
                elif next_sn_local % 2 == 1:
                    next_phase = 1  # STAS (server tactical attack)
                else:
                    next_phase = 2  # RTAS (receiver tactical attack)
                f["next_phase"] = next_phase
                f["next_in_initial_offensive"] = int(next_sn_local <= 5)  # Wu 2022: IO phase contains 70.6% strokes

                # First Offensive Stroke (FOS): first non-serve attack class (1-7) in prefix
                fos_idx = -1
                for i, r in enumerate(ctx_rows):
                    aid = int(r["actionId"])
                    if 1 <= aid <= 7:  # attack class (excludes serves 15-18)
                        fos_idx = i + 1  # 1-indexed
                        break
                f["fos_idx"] = fos_idx  # -1 if not yet seen
                f["strokes_since_fos"] = (k - fos_idx + 1) if fos_idx > 0 else -1
                f["prefix_has_fos"] = int(fos_idx > 0)

                # AOEL flag (Wu 2022): "half-long" ball = pointId in middle row {4,5,6}
                # Scoring rate jumps from 60% to 97% after half-long. Strong rally-winner predictor.
                last_pid = int(last["pointId"])
                f["last_was_half_long"] = int(4 <= last_pid <= 6)
                # Count half-longs in prefix
                f["prefix_half_long_count"] = sum(1 for r in ctx_rows if 4 <= int(r["pointId"]) <= 6)

                # Stroke-side phase: which side hits next?
                # Server hits odd strokes (1,3,5...), receiver hits even (2,4,6...)
                f["next_side_is_server"] = int(next_sn_local % 2 == 1)
                # Side count in prefix
                f["server_strokes_count"] = sum(1 for i in range(k) if (i+1) % 2 == 1)
                f["receiver_strokes_count"] = sum(1 for i in range(k) if (i+1) % 2 == 0)

            if EB:
                # Empirical-Bayes per-rally features from VISIBLE PREFIX of THIS rally only
                # + global prior shrinkage. NO train-derived player history (handles unseen players).
                # global_priors must be provided: dict with attack_ratio, forehand_ratio, topspin_ratio
                assert global_priors is not None, "V5Z_EB=1 requires global_priors arg"
                kappa = EB_KAPPA
                server_id = first["gamePlayerId"]
                srv_rows = [r for r in ctx_rows if r["gamePlayerId"] == server_id]
                rcv_rows = [r for r in ctx_rows if r["gamePlayerId"] != server_id]

                def shrink(observed_count, total, prior_p):
                    return (observed_count + kappa * prior_p) / (total + kappa)

                # Server EB stats
                n_srv = len(srv_rows)
                if n_srv > 0:
                    srv_attack = sum(1 for r in srv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 1)
                    srv_control = sum(1 for r in srv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 2)
                    srv_defense = sum(1 for r in srv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 3)
                    srv_fh = sum(1 for r in srv_rows if r["handId"] == 1)
                    srv_ts = sum(1 for r in srv_rows if r["spinId"] == 1)
                else:
                    srv_attack = srv_control = srv_defense = srv_fh = srv_ts = 0
                f["srv_eb_attack"]   = shrink(srv_attack, n_srv, global_priors["attack_ratio"])
                f["srv_eb_control"]  = shrink(srv_control, n_srv, global_priors["control_ratio"])
                f["srv_eb_defense"]  = shrink(srv_defense, n_srv, global_priors["defense_ratio"])
                f["srv_eb_forehand"] = shrink(srv_fh, n_srv, global_priors["forehand_ratio"])
                f["srv_eb_topspin"]  = shrink(srv_ts, n_srv, global_priors["topspin_ratio"])

                # Receiver EB stats
                n_rcv = len(rcv_rows)
                if n_rcv > 0:
                    rcv_attack = sum(1 for r in rcv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 1)
                    rcv_control = sum(1 for r in rcv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 2)
                    rcv_defense = sum(1 for r in rcv_rows if ACTION_GROUPS.get(r["actionId"], 0) == 3)
                    rcv_fh = sum(1 for r in rcv_rows if r["handId"] == 1)
                    rcv_ts = sum(1 for r in rcv_rows if r["spinId"] == 1)
                else:
                    rcv_attack = rcv_control = rcv_defense = rcv_fh = rcv_ts = 0
                f["rcv_eb_attack"]   = shrink(rcv_attack, n_rcv, global_priors["attack_ratio"])
                f["rcv_eb_control"]  = shrink(rcv_control, n_rcv, global_priors["control_ratio"])
                f["rcv_eb_defense"]  = shrink(rcv_defense, n_rcv, global_priors["defense_ratio"])
                f["rcv_eb_forehand"] = shrink(rcv_fh, n_rcv, global_priors["forehand_ratio"])
                f["rcv_eb_topspin"]  = shrink(rcv_ts, n_rcv, global_priors["topspin_ratio"])

                # Diff features
                f["eb_attack_diff"] = f["srv_eb_attack"] - f["rcv_eb_attack"]
                f["eb_forehand_diff"] = f["srv_eb_forehand"] - f["rcv_eb_forehand"]
                f["eb_topspin_diff"] = f["srv_eb_topspin"] - f["rcv_eb_topspin"]

                # Sample size flags (encodes confidence)
                f["srv_eb_n"] = n_srv
                f["rcv_eb_n"] = n_rcv

            if AV2:
                assert player_stats is not None and cold_marginals is not None, \
                    "V5Z_AV2=1 requires player_stats and cold_marginals"
                srv_id = int(first["gamePlayerId"])
                rcv_id = int(first["gamePlayerOtherId"])
                sex_v = int(first["sex"])
                cold_default = cold_marginals.get(sex_v, cold_marginals[1])
                srv_st = player_stats.get(srv_id, cold_default)
                rcv_st = player_stats.get(rcv_id, cold_default)
                srv_seen = int(srv_id in (seen_players or {}))
                rcv_seen = int(rcv_id in (seen_players or {}))
                _AV2_FEATS = ["attack_ratio", "control_ratio", "defense_ratio",
                              "forehand_ratio", "topspin_ratio",
                              "short_ratio", "long_ratio",
                              "fore_side_ratio", "back_side_ratio",
                              "avg_rally_len", "winrate_as_srv", "winrate_as_rcv"]
                for fname in _AV2_FEATS:
                    f[f"srv_{fname}"] = srv_st[fname]
                    f[f"rcv_{fname}"] = rcv_st[fname]
                # Diff features
                f["srv_rcv_attack_diff"] = srv_st["attack_ratio"] - rcv_st["attack_ratio"]
                f["srv_rcv_winrate_diff"] = srv_st["winrate_as_srv"] - rcv_st["winrate_as_rcv"]
                f["srv_rcv_forehand_diff"] = srv_st["forehand_ratio"] - rcv_st["forehand_ratio"]
                f["srv_rcv_avglen_diff"] = srv_st["avg_rally_len"] - rcv_st["avg_rally_len"]
                f["srv_rcv_topspin_diff"] = srv_st["topspin_ratio"] - rcv_st["topspin_ratio"]
                # Cold-start flags + interaction
                f["srv_seen"] = srv_seen
                f["rcv_seen"] = rcv_seen
                f["srv_rcv_seen_combo"] = (1 - srv_seen) + (1 - rcv_seen)  # 0=both seen, 1=one unseen, 2=both unseen

            return f

        sgp_default = int(first["serverGetPoint"]) if "serverGetPoint" in first else 0
        if is_train:
            start_k = 1 if augment else max(1, N - 1)
            for k in range(start_k, N):
                features.append(_feats(rows[:k], rows[k]))
                labels_a.append(int(rows[k]["actionId"]))
                labels_p.append(int(rows[k]["pointId"]))
                uids.append(uid)
                sgp_labels.append(sgp_default)
        else:
            features.append(_feats(rows))
            uids.append(uid)
            sgp_labels.append(sgp_default)

    feat_df = pd.DataFrame(features)
    return feat_df, labels_a, labels_p, uids, sgp_labels


# ============================================================
# 5. LSTM Dataset & Model
# ============================================================
class RallyDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        L = s["length"]
        sc = s["seq_cat"]
        sn = s["seq_num"]

        if L < MAX_SEQ_LEN:
            sc = np.vstack([sc, np.zeros((MAX_SEQ_LEN - L, N_SEQ_CATS), dtype=np.int64)])
            sn = np.vstack([sn, np.zeros((MAX_SEQ_LEN - L, N_SEQ_NUMS), dtype=np.float32)])
        elif L > MAX_SEQ_LEN:
            sc = sc[-MAX_SEQ_LEN:]
            sn = sn[-MAX_SEQ_LEN:]
            L = MAX_SEQ_LEN

        out = {
            "seq_cat": torch.LongTensor(sc),
            "seq_num": torch.FloatTensor(sn),
            "pid_server": s["pid_server"],
            "pid_receiver": s["pid_receiver"],
            "pid_next": s["pid_next"],
            "static": torch.FloatTensor(s["static"]),
            "length": L,
            "next_sn": s["next_strike_num"],
        }
        if self.is_train:
            out["target_action"] = s["target_action"]
            out["target_point"] = s["target_point"]
            out["target_spin"] = s.get("target_spin", 0)
            out["target_hand"] = s.get("target_hand", 0)
            out["target_strength"] = s.get("target_strength", 0)
        out["sgp_label"] = s["sgp_label"]
        return out


def build_action_mask(next_strike_nums, n_action=N_ACTION, device="cpu"):
    B = len(next_strike_nums)
    mask = torch.zeros(B, n_action, device=device)
    for i in range(B):
        nsn = next_strike_nums[i]
        if isinstance(nsn, torch.Tensor):
            nsn = nsn.item()
        if nsn == 1:
            for a in range(n_action):
                if a not in [0] + SERVE_ACTIONS:
                    mask[i, a] = -1e9
        else:
            for a in SERVE_ACTIONS:
                if a < n_action:
                    mask[i, a] = -1e9
    return mask


class MultiHeadAttn(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, mask):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=~mask)
        return out.squeeze(1)


class PingPongModel(nn.Module):
    def __init__(self, player_drop_p=0.0):
        super().__init__()
        self.player_drop_p = player_drop_p

        self.cat_embeds = nn.ModuleDict()
        total_emb = 0
        for name, vocab in SEQ_CAT_SPEC.items():
            self.cat_embeds[name] = nn.Embedding(vocab + 1, EMBED_DIM, padding_idx=0)
            total_emb += EMBED_DIM

        self.player_embed = nn.Embedding(N_PLAYERS, PLAYER_EMBED_DIM, padding_idx=0)

        lstm_in = total_emb + N_SEQ_NUMS
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_in, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN + 1, HIDDEN_DIM)

        if ARCH == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM, nhead=N_ATTN_HEADS,
                dim_feedforward=HIDDEN_DIM * 2, dropout=DROPOUT,
                batch_first=True, activation="gelu",
            )
            self.lstm = nn.TransformerEncoder(enc_layer, num_layers=N_LSTM_LAYERS)
        elif ARCH == "mamba":
            from mamba_block import BidirectionalMamba
            self.lstm = BidirectionalMamba(
                d_model=HIDDEN_DIM, num_layers=N_LSTM_LAYERS,
                d_state=16, d_conv=4, expand=2, dropout=DROPOUT,
            )
        else:
            rnn_cls = nn.GRU if ARCH == "gru" else nn.LSTM
            self.lstm = rnn_cls(
                HIDDEN_DIM, HIDDEN_DIM // 2, N_LSTM_LAYERS,
                batch_first=True, dropout=DROPOUT if N_LSTM_LAYERS > 1 else 0,
                bidirectional=True,
            )
        self.attention = MultiHeadAttn(HIDDEN_DIM, N_ATTN_HEADS)

        static_in = N_STATIC + PLAYER_EMBED_DIM * 3
        self.static_proj = nn.Sequential(
            nn.Linear(static_in, 128), nn.ReLU(), nn.Dropout(DROPOUT),
        )

        fusion_dim = HIDDEN_DIM * 2 + 128
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROPOUT),
        )

        # Action head
        self.head_action = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_ACTION),
        )
        # Point head: 接收 trunk + action probs (串聯設計)
        self.head_point = nn.Sequential(
            nn.Linear(128 + N_ACTION, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_POINT),
        )

        fwd_dim = HIDDEN_DIM // 2
        self.head_aux_a = nn.Sequential(
            nn.Linear(fwd_dim, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_ACTION),
        )
        self.head_aux_p = nn.Sequential(
            nn.Linear(fwd_dim, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_POINT),
        )

        # Sgp aux head: predict rally outcome (binary) from trunk_out.
        # Active when V5Z_SGP_AUX_W > 0; trains trunk to encode rally-outcome signal.
        self.head_sgp = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

        # MuLMINet-style aux heads: predict next stroke's spin/hand/strength
        # Trained jointly with main heads. Event-invariant tasks regularize the trunk.
        if MULTITASK:
            self.head_mt_spin = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(32, SEQ_CAT_SPEC["spinId"] + 1),  # +1 for unknown=0
            )
            self.head_mt_hand = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(32, SEQ_CAT_SPEC["handId"] + 1),
            )
            self.head_mt_strength = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(32, SEQ_CAT_SPEC["strengthId"] + 1),
            )

    def forward(self, seq_cat, seq_num, pid_s, pid_r, pid_n, static, lengths,
                next_sns=None, apply_mask=False, true_action=None, teacher_forcing_p=0.0,
                return_trunk=False, return_sgp=False):
        B, L, _ = seq_cat.shape

        embs = []
        for i, name in enumerate(SEQ_CAT_NAMES):
            x = seq_cat[:, :, i] + 1
            x = x.clamp(0, SEQ_CAT_SPEC[name])
            embs.append(self.cat_embeds[name](x))
        x = torch.cat(embs + [seq_num], dim=-1)
        x = self.input_proj(x)

        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        if isinstance(lengths, torch.Tensor):
            lens = lengths.tolist()
        else:
            lens = list(lengths)
        lens = [max(1, min(l, L)) for l in lens]

        mask = torch.arange(L, device=x.device).unsqueeze(0) < \
               torch.tensor(lens, device=x.device).unsqueeze(1)
        if ARCH == "transformer":
            # Bidirectional self-attention over prefix; pad positions masked.
            lstm_out = self.lstm(x, src_key_padding_mask=~mask)
        elif ARCH == "mamba":
            # Mamba does not use packing; pad positions are zeroed via mask in attention layer.
            lstm_out = self.lstm(x, mask=mask)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=L)

        attn_out = self.attention(lstm_out, mask)

        idx = torch.tensor(lens, device=x.device) - 1
        last_h = lstm_out[torch.arange(B, device=x.device), idx.clamp(min=0)]

        def _pid(pid):
            return pid.to(x.device) if isinstance(pid, torch.Tensor) else torch.tensor(pid, device=x.device)
        pid_s_t, pid_r_t, pid_n_t = _pid(pid_s), _pid(pid_r), _pid(pid_n)

        if self.training and self.player_drop_p > 0:
            drop = torch.rand(B, device=x.device) < self.player_drop_p
            pid_s_t = pid_s_t.masked_fill(drop, 0)
            pid_r_t = pid_r_t.masked_fill(drop, 0)
            pid_n_t = pid_n_t.masked_fill(drop, 0)

        player_feats = torch.cat([
            self.player_embed(pid_s_t),
            self.player_embed(pid_r_t),
            self.player_embed(pid_n_t),
        ], dim=-1)

        static_out = self.static_proj(torch.cat([static, player_feats], dim=-1))
        fused = torch.cat([attn_out, last_h, static_out], dim=-1)
        trunk_out = self.trunk(fused)

        # Action prediction
        action_logits = self.head_action(trunk_out)

        if apply_mask and next_sns is not None:
            action_logits = action_logits + build_action_mask(next_sns, device=x.device)

        # Action → Point 串聯
        # Teacher forcing: 訓練時用一定機率用真實 action one-hot
        if self.training and true_action is not None and teacher_forcing_p > 0:
            use_tf = torch.rand(B, device=x.device) < teacher_forcing_p
            true_oh = F.one_hot(true_action, N_ACTION).float()
            pred_probs = F.softmax(action_logits.detach(), dim=-1)
            action_info = torch.where(use_tf.unsqueeze(1), true_oh, pred_probs)
        else:
            action_info = F.softmax(action_logits.detach(), dim=-1)

        point_input = torch.cat([trunk_out, action_info], dim=-1)
        point_logits = self.head_point(point_input)

        fwd_states = lstm_out[:, :, : HIDDEN_DIM // 2]
        aux_a_logits = self.head_aux_a(fwd_states)
        aux_p_logits = self.head_aux_p(fwd_states)

        # MuLMINet aux heads
        base_returns = (action_logits, point_logits, aux_a_logits, aux_p_logits)
        if MULTITASK:
            mt_spin_logits = self.head_mt_spin(trunk_out)
            mt_hand_logits = self.head_mt_hand(trunk_out)
            mt_strength_logits = self.head_mt_strength(trunk_out)
            base_returns = base_returns + (mt_spin_logits, mt_hand_logits, mt_strength_logits)

        extras = ()
        if return_sgp:
            extras = extras + (self.head_sgp(trunk_out).squeeze(-1),)
        if return_trunk:
            extras = extras + (trunk_out,)
        return base_returns + extras if extras else base_returns


# ============================================================
# 6. Class Weights
# ============================================================
def compute_class_weights(cnt, n_classes, alpha=WEIGHT_ALPHA, exclude=None):
    exclude = exclude or set()
    total = sum(n for c, n in cnt.items() if c not in exclude)
    n_valid = n_classes - len(exclude)
    w = np.ones(n_classes, dtype=np.float32)
    for c in range(n_classes):
        if c in exclude:
            w[c] = 0.0
        elif c in cnt and cnt[c] > 0:
            w[c] = (total / (n_valid * cnt[c])) ** alpha
        else:
            w[c] = 0.0
    valid_max = max(w[c] for c in range(n_classes) if c not in exclude) if n_valid > 0 else 1.0
    if valid_max > 0:
        for c in range(n_classes):
            if c not in exclude:
                w[c] /= valid_max
    return w


# ============================================================
# 7. Calibration
# ============================================================
def calibrate_thresholds(probs, labels, n_classes, n_steps=60, n_rounds=5):
    labels = np.array(labels)
    best_scales = np.ones(n_classes, dtype=np.float64)
    best_f1 = f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)

    for r in range(n_rounds):
        improved = False
        for c in range(n_classes):
            best_c = best_scales[c]
            for s in np.linspace(0.1, 5.0, n_steps):
                sc = best_scales.copy()
                sc[c] = s
                f1 = f1_score(labels, (probs * sc).argmax(axis=1), average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_c = s
                    improved = True
            best_scales[c] = best_c
        if not improved:
            break

    return best_scales, best_f1


# ============================================================
# 8. LSTM 訓練
# ============================================================
def train_lstm_fold(tr_samples, va_samples, fold_idx, action_w, point_w):
    tr_ds = RallyDataset(tr_samples, is_train=True)
    va_ds = RallyDataset(va_samples, is_train=True)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    if FOCAL_GAMMA > 0:
        # Class-weighted focal CE: -w_y * (1 - p_y)^gamma * log p_y
        # Label smoothing dropped (incompatible with focal's sharp pt term).
        a_w = torch.FloatTensor(action_w).to(DEVICE)
        p_w = torch.FloatTensor(point_w).to(DEVICE)
        def _focal(logits, target, w):
            log_p = F.log_softmax(logits, dim=-1)
            log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            focal = -((1 - pt) ** FOCAL_GAMMA) * log_pt
            return (focal * w[target]).mean()
        loss_action = lambda lg, tgt: _focal(lg, tgt, a_w)
        loss_point = lambda lg, tgt: _focal(lg, tgt, p_w)
        log(f"  Using focal CE with gamma={FOCAL_GAMMA}")
    else:
        loss_action = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(action_w).to(DEVICE), label_smoothing=LABEL_SMOOTH,
        )
        loss_point = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(point_w).to(DEVICE), label_smoothing=LABEL_SMOOTH,
        )

    best_score, best_state, wait = -1, None, 0

    for epoch in range(EPOCHS):
        model.train()
        t_loss, nb = 0, 0
        # Teacher forcing: 從 0.5 衰減到 0
        tf_p = max(0.0, 0.5 * (1.0 - epoch / 40.0))

        for batch in tr_dl:
            sc = batch["seq_cat"].to(DEVICE)
            sn = batch["seq_num"].to(DEVICE)
            ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
            st = batch["static"].to(DEVICE)
            lens = batch["length"]
            ta = torch.LongTensor(batch["target_action"]).to(DEVICE)
            tp = torch.LongTensor(batch["target_point"]).to(DEVICE)

            mt_targets = None
            if MULTITASK:
                t_spin = torch.LongTensor(batch["target_spin"]).to(DEVICE)
                t_hand = torch.LongTensor(batch["target_hand"]).to(DEVICE)
                t_strength = torch.LongTensor(batch["target_strength"]).to(DEVICE)
                mt_targets = (t_spin, t_hand, t_strength)

            need_sgp = SGP_AUX_W > 0 or SGP_ONLY
            model_out = model(sc, sn, ps, pr, pn, st, lens,
                              apply_mask=False, true_action=ta, teacher_forcing_p=tf_p,
                              return_sgp=need_sgp)
            sgp_logit_train = None
            if MULTITASK and need_sgp:
                a_log, p_log, aux_a, aux_p, mt_spin_log, mt_hand_log, mt_strength_log, sgp_logit_train = model_out
            elif MULTITASK:
                a_log, p_log, aux_a, aux_p, mt_spin_log, mt_hand_log, mt_strength_log = model_out
            elif need_sgp:
                a_log, p_log, aux_a, aux_p, sgp_logit_train = model_out
            else:
                a_log, p_log, aux_a, aux_p = model_out

            L_ce_a = loss_action(a_log, ta)
            L_ce_p = loss_point(p_log, tp)
            if SOFTF1_W > 0:
                # Per-batch soft macro-F1 (directly optimizes the eval metric).
                # Restricted to classes that actually appear in batch + targets.
                def _soft_f1(logits, target, n_class):
                    p = F.softmax(logits, dim=-1)
                    t = F.one_hot(target, num_classes=n_class).float()
                    tp = (p * t).sum(dim=0)
                    fp = (p * (1 - t)).sum(dim=0)
                    fn = ((1 - p) * t).sum(dim=0)
                    sf1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
                    has = t.sum(dim=0) > 0
                    return 1.0 - sf1[has].mean() if has.any() else torch.zeros(1, device=logits.device).squeeze()
                L_sf1_a = _soft_f1(a_log, ta, N_ACTION)
                L_sf1_p = _soft_f1(p_log, tp, N_POINT)
                L_main = (1 - SOFTF1_W) * (0.5 * L_ce_a + 0.5 * L_ce_p) + SOFTF1_W * (0.5 * L_sf1_a + 0.5 * L_sf1_p)
            else:
                L_main = 0.5 * L_ce_a + 0.5 * L_ce_p

            # Forward-only aux next-token loss: at position t < length-1,
            # forward LSTM state predicts seq[t+1]'s action/point.
            len_t = batch["length"] if isinstance(batch["length"], torch.Tensor) else torch.tensor(batch["length"])
            len_t = len_t.to(DEVICE)
            L_aux_max = aux_a.size(1)  # = MAX_SEQ_LEN
            tgt_a_seq = sc[:, 1:, 5]  # SEQ_CAT_NAMES idx 5 = actionId; [B, L-1]
            tgt_p_seq = sc[:, 1:, 4]  # idx 4 = pointId
            pos_idx = torch.arange(L_aux_max - 1, device=DEVICE).unsqueeze(0)  # [1, L-1]
            valid = pos_idx < (len_t - 1).unsqueeze(1)  # [B, L-1]
            tgt_a_seq = torch.where(valid, tgt_a_seq, torch.full_like(tgt_a_seq, -1))
            tgt_p_seq = torch.where(valid, tgt_p_seq, torch.full_like(tgt_p_seq, -1))
            L_aux_a = F.cross_entropy(aux_a[:, :-1, :].reshape(-1, N_ACTION), tgt_a_seq.reshape(-1), ignore_index=-1)
            L_aux_p = F.cross_entropy(aux_p[:, :-1, :].reshape(-1, N_POINT), tgt_p_seq.reshape(-1), ignore_index=-1)
            L_aux = 0.5 * L_aux_a + 0.5 * L_aux_p

            L = L_main + AUX_LOSS_W * L_aux

            # MuLMINet aux losses (regularize trunk via event-invariant tasks)
            if MULTITASK:
                t_spin, t_hand, t_strength = mt_targets
                # Targets need +1 because embedding uses +1 shift
                L_mt_spin = F.cross_entropy(mt_spin_log, t_spin)
                L_mt_hand = F.cross_entropy(mt_hand_log, t_hand)
                L_mt_strength = F.cross_entropy(mt_strength_log, t_strength)
                L_mt = MT_W * (L_mt_spin + L_mt_hand + L_mt_strength)
                L = L + L_mt

            # Sgp aux loss: BCE on rally outcome predicted from trunk_out
            if SGP_AUX_W > 0 or SGP_ONLY:
                sgp_target = batch["sgp_label"].float().to(DEVICE)
                L_sgp = F.binary_cross_entropy_with_logits(sgp_logit_train, sgp_target)
                if SGP_ONLY:
                    L = L_sgp  # discard CE losses, BCE only
                else:
                    L = L + SGP_AUX_W * L_sgp

            if torch.isnan(L) or torch.isinf(L):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            t_loss += L.item()
            nb += 1

        scheduler.step()

        # Validate
        model.eval()
        all_ap, all_at, all_pp, all_pt = [], [], [], []
        sgp_preds, sgp_labels_va = [], []
        with torch.no_grad():
            for batch in va_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]

                need_sgp_val = SGP_AUX_W > 0 or SGP_ONLY
                out = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True, return_sgp=need_sgp_val)
                a_log, p_log = out[0], out[1]

                all_ap.extend(a_log.argmax(1).cpu().numpy())
                all_at.extend(batch["target_action"])
                all_pp.extend(p_log.argmax(1).cpu().numpy())
                all_pt.extend(batch["target_point"])

                if need_sgp_val:
                    sgp_preds.extend(torch.sigmoid(out[-1]).cpu().numpy().tolist())
                    sgp_labels_va.extend(batch["sgp_label"].cpu().numpy().tolist() if isinstance(batch["sgp_label"], torch.Tensor) else list(batch["sgp_label"]))

        f1_a = f1_score(all_at, all_ap, average="macro", zero_division=0)
        f1_p = f1_score(all_pt, all_pp, average="macro", zero_division=0)
        if SGP_ONLY and sgp_preds:
            from sklearn.metrics import roc_auc_score as _auc
            try:
                sgp_auc = _auc(sgp_labels_va, sgp_preds)
            except ValueError:
                sgp_auc = 0.5
            score = sgp_auc  # SGP_ONLY: pick best by sgp AUC
        else:
            score = 0.4 * f1_a + 0.4 * f1_p + 0.2 * 1.0
            sgp_auc = None

        if (epoch + 1) % 5 == 0 or epoch == 0:
            sgp_str = f" sgp_AUC={sgp_auc:.4f}" if sgp_auc is not None else ""
            log(f"  E{epoch+1:3d}: loss={t_loss/max(nb,1):.4f} "
                f"F1_act={f1_a:.4f} F1_pt={f1_p:.4f} Score={score:.4f}{sgp_str} tf_p={tf_p:.2f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            log(f"  Early stop at epoch {epoch+1}")
            break

    log(f"  Fold {fold_idx+1} best: {best_score:.4f}")
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    return model, best_score


# ============================================================
# 9. LightGBM 訓練
# ============================================================
def train_lgb_fold(X_tr, y_tr, X_va, y_va, n_classes, task_name):
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
        "seed": SEED,
        "n_jobs": -1,
    }

    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_va, y_va, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    va_probs = model.predict(X_va)
    va_preds = va_probs.argmax(axis=1)
    f1 = f1_score(y_va, va_preds, average="macro", zero_division=0)
    log(f"  LGB {task_name}: F1={f1:.4f} (best_iter={model.best_iteration})")

    return model, va_probs, f1


# ============================================================
# 10. 主程式
# ============================================================
if __name__ == "__main__":
    log(f"\n{'='*60}")
    log("準備資料...")
    t0 = time.time()

    # LSTM samples
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"  LSTM 訓練樣本: {len(train_samples)}")
    log(f"  LSTM 測試樣本: {len(test_samples)}")

    # LGB features
    log("建立 LGB 特徵...")
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_uids, lgb_sgp = build_lgb_features(
        train_df, is_train=True, augment=True)
    lgb_X_test, _, _, lgb_test_uids, lgb_test_sgp = build_lgb_features(
        test_df, is_train=False)
    log(f"  LGB 特徵: {lgb_X_train.shape[1]} 個")
    log(f"  準備完成 ({time.time()-t0:.1f}s)")

    # Class weights
    act_cnt = Counter(s["target_action"] for s in train_samples)
    pt_cnt = Counter(s["target_point"] for s in train_samples)
    action_w = compute_class_weights(act_cnt, N_ACTION, alpha=WEIGHT_ALPHA,
                                     exclude=set(SERVE_ACTIONS))
    point_w = compute_class_weights(pt_cnt, N_POINT, alpha=WEIGHT_ALPHA)

    log(f"\n  actionId weights: { {i: round(action_w[i],3) for i in range(15)} }")
    log(f"  pointId weights:  { {i: round(point_w[i],3) for i in range(N_POINT)} }")

    # --- CV setup ---
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)

    # LGB uid → index mapping
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)

    uid_last_a = {}
    for s in train_samples:
        uid_last_a[s["uid"]] = s["target_action"]
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])

    log(f"\n{'='*60}")
    log(f"開始 {N_FOLDS}-fold CV (V5: LSTM+LGB Ensemble)")

    if MATCH_DISJOINT:
        from sklearn.model_selection import StratifiedGroupKFold
        uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
        match_arr = np.array([uid_to_match[u] for u in uid_list])
        kf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
        kf_splits = list(kf.split(uid_list, uid_arr, groups=match_arr))
        log(f"  CV: StratifiedGroupKFold by match_id (match-disjoint)")
        for fi, (tr, va) in enumerate(kf_splits):
            tr_m = set(match_arr[tr]); va_m = set(match_arr[va])
            log(f"    fold {fi+1}: {len(tr_m)} train matches | {len(va_m)} val matches")
    else:
        kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        kf_splits = list(kf.split(uid_list, uid_arr))
        log(f"  CV: StratifiedKFold by class (rally-level)")

    lstm_models = []
    lgb_a_models, lgb_p_models = [], []
    all_oof_lstm_a, all_oof_lstm_p = [], []  # (probs, labels)
    all_oof_lgb_a, all_oof_lgb_p = [], []
    all_oof_labels_a, all_oof_labels_p = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(kf_splits):
        log(f"\n{'─'*50}")
        log(f"Fold {fold+1}/{N_FOLDS}")

        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        # --- LSTM ---
        tr_s = [train_samples[i] for u in tr_uids for i in uid2idx.get(u, [])]
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        log(f"  LSTM Train: {len(tr_s)}, Val: {len(va_s)}")

        if INFER_TEST_ONLY:
            weights_tag = LSTM_WEIGHTS_TAG if LSTM_WEIGHTS_TAG else TAG
            weight_path = f"models/v5z/model_{weights_tag}_lstm_fold{fold+1}.pt"
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"V5Z_INFER_TEST_ONLY=1 but weight not found: {weight_path}. "
                    f"Set V5Z_LSTM_WEIGHTS_TAG to a tag whose weights exist."
                )
            lstm_model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
            # strict=False so old weights (no head_sgp) still load; head_sgp stays randomly init.
            lstm_model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True), strict=False)
            lstm_model.eval()
            log(f"  [INFER_ONLY] Loaded LSTM weights from {weight_path}")
        else:
            lstm_model, lstm_sc = train_lstm_fold(tr_s, va_s, fold, action_w, point_w)
            torch.save(lstm_model.state_dict(), f"models/v5z/model_{TAG}_lstm_fold{fold+1}.pt")
        lstm_models.append(lstm_model)

        # LSTM validation probs
        va_dl = DataLoader(RallyDataset(va_s, is_train=True), BATCH_SIZE,
                           shuffle=False, num_workers=0, pin_memory=True)
        lstm_a_p, lstm_p_p, lstm_la, lstm_lp = [], [], [], []
        with torch.no_grad():
            for batch in va_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]
                out = lstm_model(sc, sn, ps, pr, pn, st, lens,
                                          next_sns=nsns, apply_mask=True)
                a_log, p_log = out[0], out[1]
                lstm_a_p.append(F.softmax(a_log, dim=-1).cpu().numpy())
                lstm_p_p.append(F.softmax(p_log, dim=-1).cpu().numpy())
                lstm_la.extend(batch["target_action"])
                lstm_lp.extend(batch["target_point"])

        all_oof_lstm_a.append(np.vstack(lstm_a_p))
        all_oof_lstm_p.append(np.vstack(lstm_p_p))

        # --- LGB ---
        lgb_tr_idx = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va_idx = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        X_tr = lgb_X_train.iloc[lgb_tr_idx]
        X_va = lgb_X_train.iloc[lgb_va_idx]
        y_a_tr = [lgb_y_a[i] for i in lgb_tr_idx]
        y_a_va = [lgb_y_a[i] for i in lgb_va_idx]
        y_p_tr = [lgb_y_p[i] for i in lgb_tr_idx]
        y_p_va = [lgb_y_p[i] for i in lgb_va_idx]

        log(f"  LGB Train: {len(X_tr)}, Val: {len(X_va)}")

        lgb_a_model, lgb_a_va_probs, _ = train_lgb_fold(X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "actionId")
        lgb_p_model, lgb_p_va_probs, _ = train_lgb_fold(X_tr, y_p_tr, X_va, y_p_va, N_POINT, "pointId")
        lgb_a_models.append(lgb_a_model)
        lgb_p_models.append(lgb_p_model)

        all_oof_lgb_a.append(lgb_a_va_probs)
        all_oof_lgb_p.append(lgb_p_va_probs)
        all_oof_labels_a.append(np.array(lstm_la))
        all_oof_labels_p.append(np.array(lstm_lp))

        # Fold ensemble score
        ens_a = W_LSTM * np.vstack(lstm_a_p) + W_LGB * lgb_a_va_probs
        ens_p = W_LSTM * np.vstack(lstm_p_p) + W_LGB * lgb_p_va_probs
        f1_a = f1_score(lstm_la, ens_a.argmax(1), average="macro", zero_division=0)
        f1_p = f1_score(lstm_lp, ens_p.argmax(1), average="macro", zero_division=0)
        log(f"  Ensemble F1_act={f1_a:.4f} F1_pt={f1_p:.4f} Score={0.4*f1_a+0.4*f1_p+0.2:.4f}")

    # ============================================================
    # 11. OOF Ensemble 校準
    # ============================================================
    log(f"\n{'='*60}")
    log("OOF Ensemble 校準...")

    merged_lstm_a = np.vstack(all_oof_lstm_a)
    merged_lstm_p = np.vstack(all_oof_lstm_p)
    merged_lgb_a = np.vstack(all_oof_lgb_a)
    merged_lgb_p = np.vstack(all_oof_lgb_p)
    merged_la = np.concatenate(all_oof_labels_a)
    merged_lp = np.concatenate(all_oof_labels_p)

    # 搜索最佳 ensemble 權重
    log("\n  搜尋最佳 ensemble 權重...")
    best_w, best_ens_score = 0.4, -1
    for w_lstm in np.arange(0.0, 1.05, 0.05):
        ens_a = w_lstm * merged_lstm_a + (1 - w_lstm) * merged_lgb_a
        ens_p = w_lstm * merged_lstm_p + (1 - w_lstm) * merged_lgb_p
        f1_a = f1_score(merged_la, ens_a.argmax(1), average="macro", zero_division=0)
        f1_p = f1_score(merged_lp, ens_p.argmax(1), average="macro", zero_division=0)
        sc = 0.4 * f1_a + 0.4 * f1_p + 0.2
        if sc > best_ens_score:
            best_ens_score = sc
            best_w = w_lstm
    log(f"  最佳 LSTM 權重: {best_w:.2f}, Score: {best_ens_score:.4f}")

    # 用最佳權重做校準
    ens_a_merged = best_w * merged_lstm_a + (1 - best_w) * merged_lgb_a
    ens_p_merged = best_w * merged_lstm_p + (1 - best_w) * merged_lgb_p

    log(f"\n  actionId 校準:")
    a_scales, cal_f1_a = calibrate_thresholds(ens_a_merged, merged_la, N_ACTION)
    log(f"\n  pointId 校準:")
    p_scales, cal_f1_p = calibrate_thresholds(ens_p_merged, merged_lp, N_POINT)

    final_cv = 0.4 * cal_f1_a + 0.4 * cal_f1_p + 0.2
    log(f"\n  最終校準 CV: F1_act={cal_f1_a:.4f} F1_pt={cal_f1_p:.4f} Score={final_cv:.4f}")

    # ============================================================
    # 12. Test 預測
    # ============================================================
    log(f"\n{'='*60}")
    log("生成測試預測...")

    # LSTM test probs
    test_dl = DataLoader(RallyDataset(test_samples, is_train=False), BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)
    n_test = len(test_samples)
    lstm_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lstm_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)

    for model in lstm_models:
        model.to(DEVICE).eval()
        offset = 0
        with torch.no_grad():
            for batch in test_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]
                out = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True)
                a_log, p_log = out[0], out[1]
                bs = sc.size(0)
                lstm_test_a[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / N_FOLDS
                lstm_test_p[offset:offset+bs] += F.softmax(p_log, dim=-1).cpu().numpy() / N_FOLDS
                offset += bs

    # LGB test probs
    lgb_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lgb_test_p = np.zeros((n_test, N_POINT), dtype=np.float64)
    for m in lgb_a_models:
        lgb_test_a += m.predict(lgb_X_test) / N_FOLDS
    for m in lgb_p_models:
        lgb_test_p += m.predict(lgb_X_test) / N_FOLDS

    # Ensemble + calibration
    ens_test_a = best_w * lstm_test_a + (1 - best_w) * lgb_test_a
    ens_test_p = best_w * lstm_test_p + (1 - best_w) * lgb_test_p

    pred_action = (ens_test_a * a_scales).argmax(axis=1)
    pred_point = (ens_test_p * p_scales).argmax(axis=1)
    test_uids = [s["uid"] for s in test_samples]
    if TEST_HAS_SGP:
        test_sgp = [s["sgp_label"] for s in test_samples]
    else:
        # Placeholder; downstream advcal/Phase-2 step overrides with predicted values.
        test_sgp = [0.5] * len(test_samples)
        log(f"  [WARN] Test CSV has no serverGetPoint column → submission uses 0.5 placeholder. Replace via predict_sgp before LB.")

    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    })
    sub = sub.sort_values("rally_uid").reset_index(drop=True)
    sub.to_csv(f"submissions/submission_{TAG}.csv", index=False)

    log(f"\n提交檔案: submission_{TAG}.csv")
    log(f"  共 {len(sub)} 筆")
    log(f"  actionId: {dict(sorted(Counter(pred_action).items()))}")
    log(f"  pointId:  {dict(sorted(Counter(pred_point).items()))}")

    # Save OOF + test preds in OOF order for downstream calibration
    np.savez(
        f"artifacts/{TAG}_oof.npz",
        lstm_a=merged_lstm_a, lstm_p=merged_lstm_p,
        lgb_a=merged_lgb_a, lgb_p=merged_lgb_p,
        la=merged_la, lp=merged_lp,
        a_scales=a_scales, p_scales=p_scales,
        best_w=best_w, cv_score=final_cv,
    )
    np.savez(
        f"artifacts/{TAG}_test.npz",
        lstm_a=lstm_test_a, lstm_p=lstm_test_p,
        lgb_a=lgb_test_a, lgb_p=lgb_test_p,
        test_uids=np.array(test_uids), test_sgp=np.array(test_sgp),
    )
    np.save(f"artifacts/pred_{TAG}_ens_action.npy", ens_test_a)
    np.save(f"artifacts/pred_{TAG}_ens_point.npy", ens_test_p)

    log("\n完成!")
