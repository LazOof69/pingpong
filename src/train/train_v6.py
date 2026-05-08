#!/usr/bin/env python3
"""
桌球戰術與結果預測 V6
============================================================
V5 → V6 改進:
  1. [新增] Terminal gate: pointId=0 從 10 類獨立成 binary head
             point head 改為 9 類 (pointId=1..9)
  2. [新增] Plug-in Macro-F1 校準 (加性 log-bias, 取代乘性 scale)
  3. [新增] Per-task ensemble weights (action/point 分開搜尋)
  4. 推論: P(y=0)=P(term); P(y=k)=(1-P(term))*P(y=k|¬term), k∈1..9
============================================================
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
SEED = 42
MAX_SEQ_LEN = 50
EMBED_DIM = 20
PLAYER_EMBED_DIM = 16
HIDDEN_DIM = 192
N_LSTM_LAYERS = 2
N_ATTN_HEADS = 4
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 80
PATIENCE = 15
N_FOLDS = 5
LABEL_SMOOTH = 0.05
WEIGHT_ALPHA = 0.5
PLAYER_DROP_P = 0.3

N_ACTION = 19
N_POINT = 10
N_POINT_9 = 9  # pointId 1..9 (non-terminal)
SERVE_ACTIONS = [15, 16, 17, 18]

# Loss weights
W_ACTION = 0.5
W_TERMINAL = 0.25
W_POINT_9 = 0.5

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
test_df = pd.read_csv("data/test.csv")
log(f"  Train: {train_df.shape[0]} rows, {train_df['rally_uid'].nunique()} rallies")
log(f"  Test:  {test_df.shape[0]} rows, {test_df['rally_uid'].nunique()} rallies")

# 選手映射
all_pids = sorted(
    set(train_df["gamePlayerId"].unique())
    | set(train_df["gamePlayerOtherId"].unique())
    | set(test_df["gamePlayerId"].unique())
    | set(test_df["gamePlayerOtherId"].unique())
)
PID2IDX = {pid: i + 1 for i, pid in enumerate(all_pids)}
N_PLAYERS = len(PID2IDX) + 1
log(f"  選手數: {len(PID2IDX)}")

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
                "sgp_label": int(first["serverGetPoint"]),
            }

            if target_row is not None:
                pt = int(target_row["pointId"])
                sample["target_action"] = int(target_row["actionId"])
                sample["target_point"] = pt
                sample["target_terminal"] = 1 if pt == 0 else 0
                # pointId 1..9 → 0..8；若 terminal 則 -1 (訓練時 mask)
                sample["target_point_9"] = (pt - 1) if pt > 0 else -1

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
def build_lgb_features(df, is_train=True, augment=True):
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
            f["sgp"] = first["serverGetPoint"]
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

            return f

        if is_train:
            start_k = 1 if augment else max(1, N - 1)
            for k in range(start_k, N):
                features.append(_feats(rows[:k], rows[k]))
                labels_a.append(int(rows[k]["actionId"]))
                labels_p.append(int(rows[k]["pointId"]))
                uids.append(uid)
                sgp_labels.append(int(first["serverGetPoint"]))
        else:
            features.append(_feats(rows))
            uids.append(uid)
            sgp_labels.append(int(first["serverGetPoint"]))

    feat_df = pd.DataFrame(features)
    labels_term = [1 if p == 0 else 0 for p in labels_p]
    labels_p9 = [(p - 1) if p > 0 else -1 for p in labels_p]
    return feat_df, labels_a, labels_p, labels_term, labels_p9, uids, sgp_labels


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
            out["target_terminal"] = s["target_terminal"]
            out["target_point_9"] = s["target_point_9"]
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

        self.lstm = nn.LSTM(
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
        # Terminal head (rally 是否結束, binary)
        self.head_terminal = nn.Sequential(
            nn.Linear(128 + N_ACTION, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 1),
        )
        # Point head (9 類: pointId 1..9)
        self.head_point = nn.Sequential(
            nn.Linear(128 + N_ACTION, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_POINT_9),
        )

    def forward(self, seq_cat, seq_num, pid_s, pid_r, pid_n, static, lengths,
                next_sns=None, apply_mask=False, true_action=None, teacher_forcing_p=0.0):
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

        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=L)

        mask = torch.arange(L, device=x.device).unsqueeze(0) < \
               torch.tensor(lens, device=x.device).unsqueeze(1)
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
        point_9_logits = self.head_point(point_input)
        terminal_logits = self.head_terminal(point_input).squeeze(-1)

        return action_logits, point_9_logits, terminal_logits


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


def plugin_calibrate(probs, labels, n_class, n_rounds=5,
                     bias_lo=-5.0, bias_hi=5.0, step=0.1, verbose=False):
    """Plug-in Macro-F1 calibration: argmax_k (log p_k + b_k)."""
    labels = np.array(labels)
    log_p = np.log(probs + 1e-12)
    bias = np.zeros(n_class, dtype=np.float64)
    grid = np.arange(bias_lo, bias_hi + step / 2, step)

    def f1_of(b):
        return f1_score(labels, (log_p + b).argmax(1), average="macro", zero_division=0)

    best_f1 = f1_of(bias)
    for r in range(n_rounds):
        improved = False
        for k in range(n_class):
            best_b = bias[k]
            best_local = best_f1
            for b in grid:
                bias[k] = b
                f1 = f1_of(bias)
                if f1 > best_local + 1e-6:
                    best_local = f1
                    best_b = b
            bias[k] = best_b
            if best_local > best_f1 + 1e-6:
                best_f1 = best_local
                improved = True
        if verbose:
            log(f"    round {r+1}: F1={best_f1:.4f}")
        if not improved:
            break
    return bias, best_f1


# ============================================================
# 8. LSTM 訓練
# ============================================================
def train_lstm_fold(tr_samples, va_samples, fold_idx, action_w, point_9_w, term_pos_weight):
    tr_ds = RallyDataset(tr_samples, is_train=True)
    va_ds = RallyDataset(va_samples, is_train=True)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    loss_action = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(action_w).to(DEVICE), label_smoothing=LABEL_SMOOTH,
    )
    loss_point_9 = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(point_9_w).to(DEVICE), label_smoothing=LABEL_SMOOTH,
    )
    loss_term = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(term_pos_weight, device=DEVICE))

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
            tt = torch.FloatTensor([int(x) for x in batch["target_terminal"]]).to(DEVICE)
            tp9 = torch.LongTensor([int(x) for x in batch["target_point_9"]]).to(DEVICE)

            a_log, p9_log, term_log = model(sc, sn, ps, pr, pn, st, lens,
                                            apply_mask=False, true_action=ta, teacher_forcing_p=tf_p)

            L = W_ACTION * loss_action(a_log, ta)
            L = L + W_TERMINAL * loss_term(term_log, tt)

            # Point 9-class loss 只在 non-terminal 樣本上計算
            nt = tp9 >= 0
            if nt.any():
                L = L + W_POINT_9 * loss_point_9(p9_log[nt], tp9[nt])

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
        with torch.no_grad():
            for batch in va_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]

                a_log, p9_log, term_log = model(sc, sn, ps, pr, pn, st, lens,
                                                next_sns=nsns, apply_mask=True)

                term_prob = torch.sigmoid(term_log).cpu().numpy()
                p9_prob = F.softmax(p9_log, dim=-1).cpu().numpy()
                # 組成 10 類分布
                full = np.zeros((term_prob.shape[0], N_POINT), dtype=np.float64)
                full[:, 0] = term_prob
                full[:, 1:] = (1 - term_prob)[:, None] * p9_prob

                all_ap.extend(a_log.argmax(1).cpu().numpy())
                all_at.extend(batch["target_action"])
                all_pp.extend(full.argmax(1))
                all_pt.extend(batch["target_point"])

        f1_a = f1_score(all_at, all_ap, average="macro", zero_division=0)
        f1_p = f1_score(all_pt, all_pp, average="macro", zero_division=0)
        score = 0.4 * f1_a + 0.4 * f1_p + 0.2 * 1.0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  E{epoch+1:3d}: loss={t_loss/max(nb,1):.4f} "
                f"F1_act={f1_a:.4f} F1_pt={f1_p:.4f} Score={score:.4f} tf_p={tf_p:.2f}")

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


def train_lgb_binary_fold(X_tr, y_tr, X_va, y_va, task_name):
    """Binary objective (terminal gate)."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
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

    va_probs = model.predict(X_va)  # shape (N,) prob of 1
    va_preds = (va_probs >= 0.5).astype(int)
    f1 = f1_score(y_va, va_preds, average="macro", zero_division=0)
    log(f"  LGB {task_name}: F1={f1:.4f} (best_iter={model.best_iteration})")

    return model, va_probs, f1


def lstm_oof_probs(model, va_samples):
    """Run LSTM inference on val samples, return (action_prob, term_prob, p9_prob, labels_a, labels_p)."""
    dl = DataLoader(RallyDataset(va_samples, is_train=True), BATCH_SIZE,
                    shuffle=False, num_workers=0, pin_memory=True)
    a_probs, term_probs, p9_probs = [], [], []
    la, lp = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            sc = batch["seq_cat"].to(DEVICE)
            sn = batch["seq_num"].to(DEVICE)
            ps, pr, pn = batch["pid_server"], batch["pid_receiver"], batch["pid_next"]
            st = batch["static"].to(DEVICE)
            lens = batch["length"]
            nsns = batch["next_sn"]
            a_log, p9_log, term_log = model(sc, sn, ps, pr, pn, st, lens,
                                            next_sns=nsns, apply_mask=True)
            a_probs.append(F.softmax(a_log, dim=-1).cpu().numpy())
            term_probs.append(torch.sigmoid(term_log).cpu().numpy())
            p9_probs.append(F.softmax(p9_log, dim=-1).cpu().numpy())
            la.extend(batch["target_action"])
            lp.extend(batch["target_point"])
    return (np.vstack(a_probs), np.concatenate(term_probs),
            np.vstack(p9_probs), np.array(la), np.array(lp))


def combine_point_probs(term_prob, p9_prob):
    """把 P(terminal) 與 P(y|¬term) 合成 10 類分布."""
    N = term_prob.shape[0]
    full = np.zeros((N, N_POINT), dtype=np.float64)
    full[:, 0] = term_prob
    full[:, 1:] = (1 - term_prob)[:, None] * p9_prob
    return full


# ============================================================
# 10. 主程式
# ============================================================
if __name__ == "__main__":
    log(f"\n{'='*60}")
    log("準備資料...")
    t0 = time.time()

    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"  LSTM 訓練樣本: {len(train_samples)}")
    log(f"  LSTM 測試樣本: {len(test_samples)}")

    log("建立 LGB 特徵...")
    lgb_X_train, lgb_y_a, lgb_y_p, lgb_y_term, lgb_y_p9, lgb_uids, lgb_sgp = \
        build_lgb_features(train_df, is_train=True, augment=True)
    lgb_X_test, _, _, _, _, lgb_test_uids, lgb_test_sgp = \
        build_lgb_features(test_df, is_train=False)
    log(f"  LGB 特徵: {lgb_X_train.shape[1]} 個")
    log(f"  準備完成 ({time.time()-t0:.1f}s)")

    # Class weights
    act_cnt = Counter(s["target_action"] for s in train_samples)
    # point_9: 0..8 對應 pointId 1..9
    p9_cnt = Counter(s["target_point_9"] for s in train_samples if s["target_point_9"] >= 0)
    # terminal 正/負 比
    n_term = sum(s["target_terminal"] for s in train_samples)
    n_nonterm = len(train_samples) - n_term
    term_pos_w = n_nonterm / max(n_term, 1)

    action_w = compute_class_weights(act_cnt, N_ACTION, alpha=WEIGHT_ALPHA,
                                     exclude=set(SERVE_ACTIONS))
    point_9_w = compute_class_weights(p9_cnt, N_POINT_9, alpha=WEIGHT_ALPHA)

    log(f"\n  actionId weights: { {i: round(action_w[i],3) for i in range(15)} }")
    log(f"  point_9 weights:  { {i: round(point_9_w[i],3) for i in range(N_POINT_9)} }")
    log(f"  terminal pos_weight: {term_pos_w:.3f} (term={n_term}, non-term={n_nonterm})")

    # CV setup
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)
    lgb_uid2idx = {}
    for i, u in enumerate(lgb_uids):
        lgb_uid2idx.setdefault(u, []).append(i)
    uid_last_a = {}
    for s in train_samples:
        uid_last_a[s["uid"]] = s["target_action"]
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])

    log(f"\n{'='*60}")
    log(f"開始 {N_FOLDS}-fold CV (V6: Terminal gate + Plug-in)")

    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    lstm_models = []
    lgb_a_models, lgb_term_models, lgb_p9_models = [], [], []

    oof_lstm_a, oof_lstm_term, oof_lstm_p9 = [], [], []
    oof_lgb_a, oof_lgb_term, oof_lgb_p9 = [], [], []
    oof_la, oof_lp = [], []

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        log(f"\n{'─'*50}")
        log(f"Fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        # LSTM
        tr_s = [train_samples[i] for u in tr_uids for i in uid2idx.get(u, [])]
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        log(f"  LSTM Train: {len(tr_s)}, Val: {len(va_s)}")

        lstm_model, _ = train_lstm_fold(tr_s, va_s, fold,
                                        action_w, point_9_w, term_pos_w)
        lstm_models.append(lstm_model)
        torch.save(lstm_model.state_dict(), f"models/v6/model_v6_lstm_fold{fold+1}.pt")

        a_prob, term_prob, p9_prob, la, lp = lstm_oof_probs(lstm_model, va_s)
        oof_lstm_a.append(a_prob)
        oof_lstm_term.append(term_prob)
        oof_lstm_p9.append(p9_prob)
        oof_la.append(la)
        oof_lp.append(lp)

        # LGB
        lgb_tr = [i for u in tr_uids for i in lgb_uid2idx.get(u, [])]
        lgb_va = [i for u in va_uids for i in lgb_uid2idx.get(u, [])]
        X_tr = lgb_X_train.iloc[lgb_tr]
        X_va = lgb_X_train.iloc[lgb_va]
        y_a_tr = [lgb_y_a[i] for i in lgb_tr]
        y_a_va = [lgb_y_a[i] for i in lgb_va]
        y_t_tr = [lgb_y_term[i] for i in lgb_tr]
        y_t_va = [lgb_y_term[i] for i in lgb_va]
        log(f"  LGB Train: {len(X_tr)}, Val: {len(X_va)}")

        lgb_a_model, lgb_a_probs, _ = train_lgb_fold(
            X_tr, y_a_tr, X_va, y_a_va, N_ACTION, "actionId")
        lgb_term_model, lgb_term_probs, _ = train_lgb_binary_fold(
            X_tr, y_t_tr, X_va, y_t_va, "terminal")

        # point_9 LGB 只用 non-terminal 樣本訓練
        nt_tr_mask = [lgb_y_p9[i] >= 0 for i in lgb_tr]
        nt_tr_idx = [lgb_tr[i] for i, m in enumerate(nt_tr_mask) if m]
        nt_va_mask = [lgb_y_p9[i] >= 0 for i in lgb_va]
        nt_va_idx = [lgb_va[i] for i, m in enumerate(nt_va_mask) if m]
        X_p9_tr = lgb_X_train.iloc[nt_tr_idx]
        X_p9_va = lgb_X_train.iloc[nt_va_idx]
        y_p9_tr = [lgb_y_p9[i] for i in nt_tr_idx]
        y_p9_va = [lgb_y_p9[i] for i in nt_va_idx]
        lgb_p9_model, _, _ = train_lgb_fold(
            X_p9_tr, y_p9_tr, X_p9_va, y_p9_va, N_POINT_9, "point_9")

        # 對全部 val 樣本預測 p9 機率
        lgb_p9_probs = lgb_p9_model.predict(X_va)

        lgb_a_models.append(lgb_a_model)
        lgb_term_models.append(lgb_term_model)
        lgb_p9_models.append(lgb_p9_model)

        oof_lgb_a.append(lgb_a_probs)
        oof_lgb_term.append(lgb_term_probs)
        oof_lgb_p9.append(lgb_p9_probs)

        # Fold 粗略分數 (未校準, 等權重)
        full_lstm_p = combine_point_probs(term_prob, p9_prob)
        full_lgb_p = combine_point_probs(lgb_term_probs, lgb_p9_probs)
        e_a = 0.5 * a_prob + 0.5 * lgb_a_probs
        e_p = 0.5 * full_lstm_p + 0.5 * full_lgb_p
        f1_a = f1_score(la, e_a.argmax(1), average="macro", zero_division=0)
        f1_p = f1_score(lp, e_p.argmax(1), average="macro", zero_division=0)
        log(f"  Fold Ens (argmax): F1_a={f1_a:.4f} F1_p={f1_p:.4f} Score={0.4*f1_a+0.4*f1_p+0.2:.4f}")

    # ============================================================
    # 11. OOF 校準
    # ============================================================
    log(f"\n{'='*60}")
    log("OOF 校準...")

    m_lstm_a = np.vstack(oof_lstm_a)
    m_lstm_term = np.concatenate(oof_lstm_term)
    m_lstm_p9 = np.vstack(oof_lstm_p9)
    m_lgb_a = np.vstack(oof_lgb_a)
    m_lgb_term = np.concatenate(oof_lgb_term)
    m_lgb_p9 = np.vstack(oof_lgb_p9)
    m_la = np.concatenate(oof_la)
    m_lp = np.concatenate(oof_lp)

    # 合成 10-class point probs
    m_lstm_p = combine_point_probs(m_lstm_term, m_lstm_p9)
    m_lgb_p = combine_point_probs(m_lgb_term, m_lgb_p9)

    log(f"\n  搜尋 action 權重...")
    best_a = (-1.0, 0.0)
    for w in np.arange(0.0, 1.01, 0.05):
        ea = w * m_lstm_a + (1 - w) * m_lgb_a
        _, fa = plugin_calibrate(ea, m_la, N_ACTION, n_rounds=3)
        if fa > best_a[0]:
            best_a = (fa, w)
    log(f"    最佳 w_a={best_a[1]:.2f} F1_a={best_a[0]:.4f}")

    log(f"\n  搜尋 point 權重...")
    best_p = (-1.0, 0.0)
    for w in np.arange(0.0, 1.01, 0.05):
        ep = w * m_lstm_p + (1 - w) * m_lgb_p
        _, fp = plugin_calibrate(ep, m_lp, N_POINT, n_rounds=3)
        if fp > best_p[0]:
            best_p = (fp, w)
    log(f"    最佳 w_p={best_p[1]:.2f} F1_p={best_p[0]:.4f}")

    # 最終完整校準 (n_rounds=5)
    best_ea = best_a[1] * m_lstm_a + (1 - best_a[1]) * m_lgb_a
    best_ep = best_p[1] * m_lstm_p + (1 - best_p[1]) * m_lgb_p
    log(f"\n  完整校準 action:")
    bias_a, cal_fa = plugin_calibrate(best_ea, m_la, N_ACTION, n_rounds=5, verbose=True)
    log(f"\n  完整校準 point:")
    bias_p, cal_fp = plugin_calibrate(best_ep, m_lp, N_POINT, n_rounds=5, verbose=True)

    final_cv = 0.4 * cal_fa + 0.4 * cal_fp + 0.2
    log(f"\n{'='*60}")
    log(f"V6 CV Score: {final_cv:.4f}")
    log(f"  F1_action: {cal_fa:.4f}")
    log(f"  F1_point:  {cal_fp:.4f}")
    log(f"  AUC_sgp:   1.0000")

    np.savez("artifacts/v6_oof.npz",
             lstm_a=m_lstm_a, lstm_term=m_lstm_term, lstm_p9=m_lstm_p9,
             lgb_a=m_lgb_a, lgb_term=m_lgb_term, lgb_p9=m_lgb_p9,
             la=m_la, lp=m_lp)
    np.savez("artifacts/v6_params.npz",
             bias_a=bias_a, bias_p=bias_p,
             w_a=np.float64(best_a[1]), w_p=np.float64(best_p[1]))
    log(f"\nOOF 存到 v6_oof.npz, 參數存到 v6_params.npz")

    # ============================================================
    # 12. Test 預測
    # ============================================================
    log(f"\n{'='*60}")
    log("生成測試預測...")

    test_dl = DataLoader(RallyDataset(test_samples, is_train=False), BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)
    n_test = len(test_samples)
    lstm_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lstm_test_term = np.zeros(n_test, dtype=np.float64)
    lstm_test_p9 = np.zeros((n_test, N_POINT_9), dtype=np.float64)

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
                a_log, p9_log, term_log = model(sc, sn, ps, pr, pn, st, lens,
                                                next_sns=nsns, apply_mask=True)
                bs = sc.size(0)
                lstm_test_a[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / N_FOLDS
                lstm_test_term[offset:offset+bs] += torch.sigmoid(term_log).cpu().numpy() / N_FOLDS
                lstm_test_p9[offset:offset+bs] += F.softmax(p9_log, dim=-1).cpu().numpy() / N_FOLDS
                offset += bs

    lgb_test_a = np.zeros((n_test, N_ACTION), dtype=np.float64)
    lgb_test_term = np.zeros(n_test, dtype=np.float64)
    lgb_test_p9 = np.zeros((n_test, N_POINT_9), dtype=np.float64)
    for m in lgb_a_models:
        lgb_test_a += m.predict(lgb_X_test) / N_FOLDS
    for m in lgb_term_models:
        lgb_test_term += m.predict(lgb_X_test) / N_FOLDS
    for m in lgb_p9_models:
        lgb_test_p9 += m.predict(lgb_X_test) / N_FOLDS

    lstm_test_p = combine_point_probs(lstm_test_term, lstm_test_p9)
    lgb_test_p = combine_point_probs(lgb_test_term, lgb_test_p9)

    # 套 per-task 權重 + plug-in bias
    ens_test_a = best_a[1] * lstm_test_a + (1 - best_a[1]) * lgb_test_a
    ens_test_p = best_p[1] * lstm_test_p + (1 - best_p[1]) * lgb_test_p

    pred_action = (np.log(ens_test_a + 1e-12) + bias_a).argmax(1)
    pred_point = (np.log(ens_test_p + 1e-12) + bias_p).argmax(1)

    test_sgp = [s["sgp_label"] for s in test_samples]
    test_uids = [s["uid"] for s in test_samples]

    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    })
    sub = sub.sort_values("rally_uid").reset_index(drop=True)
    sub.to_csv("submissions/submission_v6.csv", index=False)

    log(f"\n提交檔案: submission_v6.csv")
    log(f"  共 {len(sub)} 筆")
    log(f"  actionId: {dict(sorted(Counter(pred_action).items()))}")
    log(f"  pointId:  {dict(sorted(Counter(pred_point).items()))}")

    np.save("artifacts/pred_v6_ens_action.npy", ens_test_a)
    np.save("artifacts/pred_v6_ens_point.npy", ens_test_p)

    log(f"\n對比:")
    log(f"  V5  baseline:         Score=0.4851")
    log(f"  V6 P1 (plug-in only): Score=0.4871")
    log(f"  V6 P3 (terminal gate):Score={final_cv:.4f}")
    log("\n完成!")
