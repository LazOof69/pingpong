#!/usr/bin/env python3
"""
桌球戰術與結果預測 V4
============================================================
V3 → V4 改進:
  1. [BUG FIX] actionId class weights: 排除發球類(15-18)，
     V3 中 2 筆髒資料導致 actionId=15/16 各只有 1 筆 target，
     反頻率權重爆炸到 3669，normalize 後所有真正類別權重被壓到 ~0.001
  2. [改進] class weights 用 inv_freq^alpha (alpha=0.5)，
     而非純 inv_freq，避免過度矯正稀有類
  3. [改進] 移除 SGP head 訓練，SGP 已從 context 直接可得 (AUC=1.0)，
     loss 全部分配給 actionId/pointId (0.5/0.5)
  4. [新增] Post-hoc 門檻校準: 在 validation 上搜尋每類最佳 scale，
     最大化 Macro F1
  5. [新增] Player embedding dropout: 訓練時 30% 機率 mask player ID，
     增強對冷啟動選手的魯棒性
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter
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
WEIGHT_ALPHA = 0.5       # class weight 指數 (0=均勻, 1=全反頻率)
PLAYER_DROP_P = 0.3      # 訓練時隨機 mask player ID 的機率

N_ACTION = 19
N_POINT = 10
SERVE_ACTIONS = [15, 16, 17, 18]
N_REAL_ACTION = 15        # 實際有效的 actionId (0-14)

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

# ============================================================
# 2. 選手 ID 映射
# ============================================================
log("建立選手映射...")
all_pids = sorted(
    set(train_df["gamePlayerId"].unique())
    | set(train_df["gamePlayerOtherId"].unique())
    | set(test_df["gamePlayerId"].unique())
    | set(test_df["gamePlayerOtherId"].unique())
)
PID2IDX = {pid: i + 1 for i, pid in enumerate(all_pids)}  # 0=unknown
N_PLAYERS = len(PID2IDX) + 1
log(f"  選手數: {len(PID2IDX)}")

# ============================================================
# 3. 序列特徵定義
# ============================================================
SEQ_CAT_SPEC = {
    "strikeId":    20,
    "handId":       4,
    "strengthId":   5,
    "spinId":       7,
    "pointId":     11,
    "actionId":    20,
    "positionId":   5,
}
SEQ_CAT_NAMES = list(SEQ_CAT_SPEC.keys())
N_SEQ_CATS = len(SEQ_CAT_NAMES)
N_SEQ_NUMS = 3

# 靜態特徵: sex, numberGame, n_context, next_parity, score_diff, total_score
# (移除了 sgp_ctx，因為不再訓練 SGP head)
N_STATIC = 6


# ============================================================
# 4. 資料準備
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
                sample["target_action"] = int(target_row["actionId"])
                sample["target_point"] = int(target_row["pointId"])

            return sample

        if is_train:
            start_k = 1 if augment else max(1, N - 1)
            for k in range(start_k, N):
                samples.append(_make(rows[:k], rows[k]))
        else:
            samples.append(_make(rows))

    return samples


# ============================================================
# 5. Dataset
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

        return out


# ============================================================
# 6. Action Masking
# ============================================================
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


# ============================================================
# 7. 模型
# ============================================================
class MultiHeadAttn(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, mask):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        key_padding_mask = ~mask
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class PingPongModel(nn.Module):
    def __init__(self, player_drop_p=0.0):
        super().__init__()
        self.player_drop_p = player_drop_p

        # 類別特徵嵌入
        self.cat_embeds = nn.ModuleDict()
        total_emb = 0
        for name, vocab in SEQ_CAT_SPEC.items():
            self.cat_embeds[name] = nn.Embedding(vocab + 1, EMBED_DIM, padding_idx=0)
            total_emb += EMBED_DIM

        # 選手嵌入
        self.player_embed = nn.Embedding(N_PLAYERS, PLAYER_EMBED_DIM, padding_idx=0)

        # LSTM 輸入投影
        lstm_in = total_emb + N_SEQ_NUMS
        self.input_proj = nn.Sequential(
            nn.Linear(lstm_in, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # Positional Encoding
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN + 1, HIDDEN_DIM)

        # BiLSTM
        self.lstm = nn.LSTM(
            HIDDEN_DIM, HIDDEN_DIM // 2, N_LSTM_LAYERS,
            batch_first=True, dropout=DROPOUT if N_LSTM_LAYERS > 1 else 0,
            bidirectional=True,
        )

        # Attention
        self.attention = MultiHeadAttn(HIDDEN_DIM, N_ATTN_HEADS)

        # 靜態特徵 + 選手嵌入
        static_in = N_STATIC + PLAYER_EMBED_DIM * 3
        self.static_proj = nn.Sequential(
            nn.Linear(static_in, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # 融合
        fusion_dim = HIDDEN_DIM * 2 + 128

        # 共享 Trunk
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # 只有 action 和 point 兩個 head（移除 SGP head）
        self.head_action = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_ACTION),
        )
        self.head_point = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, N_POINT),
        )

    def forward(self, seq_cat, seq_num, pid_s, pid_r, pid_n, static, lengths,
                next_sns=None, apply_mask=False):
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

        # 選手嵌入 + dropout (訓練時隨機 mask)
        def _get_pid_tensor(pid):
            if isinstance(pid, torch.Tensor):
                return pid.to(x.device)
            return torch.tensor(pid, device=x.device)

        pid_s_t = _get_pid_tensor(pid_s)
        pid_r_t = _get_pid_tensor(pid_r)
        pid_n_t = _get_pid_tensor(pid_n)

        if self.training and self.player_drop_p > 0:
            drop_mask = torch.rand(B, device=x.device) < self.player_drop_p
            pid_s_t = pid_s_t.masked_fill(drop_mask, 0)
            pid_r_t = pid_r_t.masked_fill(drop_mask, 0)
            pid_n_t = pid_n_t.masked_fill(drop_mask, 0)

        pe_s = self.player_embed(pid_s_t)
        pe_r = self.player_embed(pid_r_t)
        pe_n = self.player_embed(pid_n_t)
        player_feats = torch.cat([pe_s, pe_r, pe_n], dim=-1)

        static_in = torch.cat([static, player_feats], dim=-1)
        static_out = self.static_proj(static_in)

        fused = torch.cat([attn_out, last_h, static_out], dim=-1)
        trunk_out = self.trunk(fused)

        action_logits = self.head_action(trunk_out)
        point_logits = self.head_point(trunk_out)

        if apply_mask and next_sns is not None:
            action_mask = build_action_mask(next_sns, device=x.device)
            action_logits = action_logits + action_mask

        return action_logits, point_logits


# ============================================================
# 8. Post-hoc 門檻校準 (最大化 Macro F1)
# ============================================================
def calibrate_thresholds(probs, labels, n_classes, n_steps=50):
    """
    搜索每個類別的 scale factor，使 argmax(prob * scale) 的 Macro F1 最大化。
    probs: (N, C) softmax 機率
    labels: (N,) 真實標籤
    """
    labels = np.array(labels)
    best_scales = np.ones(n_classes, dtype=np.float64)
    best_f1 = f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)
    log(f"    校準前 Macro F1: {best_f1:.4f}")

    # 對每個類別獨立搜索 scale
    for round_i in range(3):  # 多輪迭代
        improved = False
        for c in range(n_classes):
            best_c_scale = best_scales[c]
            for s in np.linspace(0.3, 3.0, n_steps):
                scales_try = best_scales.copy()
                scales_try[c] = s
                preds = (probs * scales_try).argmax(axis=1)
                f1 = f1_score(labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_c_scale = s
                    improved = True
            best_scales[c] = best_c_scale
        if not improved:
            break

    log(f"    校準後 Macro F1: {best_f1:.4f}")
    log(f"    scales: {dict(zip(range(n_classes), best_scales.round(3)))}")
    return best_scales, best_f1


# ============================================================
# 9. 訓練一折
# ============================================================
def train_one_fold(tr_samples, va_samples, fold_idx, action_w, point_w):
    tr_ds = RallyDataset(tr_samples, is_train=True)
    va_ds = RallyDataset(va_samples, is_train=True)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = PingPongModel(player_drop_p=PLAYER_DROP_P).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    loss_action = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(action_w).to(DEVICE),
        label_smoothing=LABEL_SMOOTH,
    )
    loss_point = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(point_w).to(DEVICE),
        label_smoothing=LABEL_SMOOTH,
    )

    best_score = -1
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        t_loss = 0
        nb = 0
        for batch in tr_dl:
            sc = batch["seq_cat"].to(DEVICE)
            sn = batch["seq_num"].to(DEVICE)
            ps = batch["pid_server"]
            pr = batch["pid_receiver"]
            pn = batch["pid_next"]
            st = batch["static"].to(DEVICE)
            lens = batch["length"]
            ta = torch.LongTensor(batch["target_action"]).to(DEVICE)
            tp = torch.LongTensor(batch["target_point"]).to(DEVICE)

            a_log, p_log = model(sc, sn, ps, pr, pn, st, lens, apply_mask=False)

            # 100% loss 分給 action + point (各 50%)
            L = 0.5 * loss_action(a_log, ta) + 0.5 * loss_point(p_log, tp)

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

        # ---- Validate ----
        model.eval()
        all_ap, all_at = [], []
        all_pp, all_pt = [], []

        with torch.no_grad():
            for batch in va_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps = batch["pid_server"]
                pr = batch["pid_receiver"]
                pn = batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]

                a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True)

                all_ap.extend(a_log.argmax(1).cpu().numpy())
                all_at.extend(batch["target_action"])
                all_pp.extend(p_log.argmax(1).cpu().numpy())
                all_pt.extend(batch["target_point"])

        f1_a = f1_score(all_at, all_ap, average="macro", zero_division=0)
        f1_p = f1_score(all_pt, all_pp, average="macro", zero_division=0)
        # SGP 不再訓練，固定算作 1.0
        score = 0.4 * f1_a + 0.4 * f1_p + 0.2 * 1.0

        avg_loss = t_loss / max(nb, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  E{epoch+1:3d}: loss={avg_loss:.4f} "
                f"F1_act={f1_a:.4f} F1_pt={f1_p:.4f} "
                f"Score={score:.4f}")

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
    model = model.to(DEVICE)
    model.eval()
    return model, best_score


# ============================================================
# 10. Class Weight 計算 (修正版)
# ============================================================
def compute_class_weights(cnt, n_classes, alpha=WEIGHT_ALPHA, exclude=None):
    """
    inv_freq^alpha 權重。exclude 集合中的類別權重設為 0。
    alpha=0 → 均勻, alpha=1 → 全反頻率, alpha=0.5 → 平方根阻尼
    """
    exclude = exclude or set()
    total = sum(n for c, n in cnt.items() if c not in exclude)
    n_valid = n_classes - len(exclude)

    w = np.ones(n_classes, dtype=np.float32)
    for c in range(n_classes):
        if c in exclude:
            w[c] = 0.0
        elif c in cnt and cnt[c] > 0:
            inv_freq = total / (n_valid * cnt[c])
            w[c] = inv_freq ** alpha
        else:
            w[c] = 0.0

    # normalize: 有效類別中最大值 = 1
    valid_max = max(w[c] for c in range(n_classes) if c not in exclude) if n_valid > 0 else 1.0
    if valid_max > 0:
        for c in range(n_classes):
            if c not in exclude:
                w[c] /= valid_max

    return w


# ============================================================
# 11. 主程式
# ============================================================
if __name__ == "__main__":
    log(f"\n{'='*60}")
    log("準備增強訓練資料...")
    t0 = time.time()
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    log(f"  訓練樣本: {len(train_samples)} ({time.time()-t0:.1f}s)")

    log("準備測試資料...")
    t0 = time.time()
    test_samples = prepare_samples(test_df, is_train=False)
    log(f"  測試樣本: {len(test_samples)} ({time.time()-t0:.1f}s)")

    # --- 類別權重 ---
    act_cnt = Counter(s["target_action"] for s in train_samples)
    pt_cnt = Counter(s["target_point"] for s in train_samples)

    # [FIX] actionId: 排除發球類 15-18 (永遠不該被預測)
    action_w = compute_class_weights(act_cnt, N_ACTION, alpha=WEIGHT_ALPHA,
                                     exclude=set(SERVE_ACTIONS))
    point_w = compute_class_weights(pt_cnt, N_POINT, alpha=WEIGHT_ALPHA)

    log(f"\n  actionId 類別權重 (alpha={WEIGHT_ALPHA}, 排除15-18):")
    for i in range(N_ACTION):
        tag = " [excluded]" if i in SERVE_ACTIONS else ""
        log(f"    {i:2d}: {action_w[i]:.4f}{tag}")
    log(f"\n  pointId 類別權重 (alpha={WEIGHT_ALPHA}):")
    for i in range(N_POINT):
        log(f"    {i:2d}: {point_w[i]:.4f}")

    log(f"\n  增強資料 actionId 分布: {dict(sorted(act_cnt.items()))}")
    log(f"  增強資料 pointId 分布:  {dict(sorted(pt_cnt.items()))}")

    # --- CV ---
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)

    uid_last_a = {}
    for s in train_samples:
        uid_last_a[s["uid"]] = s["target_action"]

    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])

    log(f"\n{'='*60}")
    log(f"開始 {N_FOLDS}-fold CV (V4)")
    log(f"  改進: 修正 weight bug, alpha={WEIGHT_ALPHA}, 無 SGP head, player_drop={PLAYER_DROP_P}")
    log(f"  BiLSTM hidden={HIDDEN_DIM}, layers={N_LSTM_LAYERS}, attn_heads={N_ATTN_HEADS}")
    log(f"  batch={BATCH_SIZE}, lr={LR}, label_smooth={LABEL_SMOOTH}")
    log(f"  Loss: 0.5*action + 0.5*point (SGP 直接從 context 取)")

    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    models = []
    scores = []
    # 收集每個 fold 的 validation 機率，用於全局門檻校準
    all_va_action_probs = []
    all_va_action_labels = []
    all_va_point_probs = []
    all_va_point_labels = []

    for fold, (tr_uidx, va_uidx) in enumerate(kf.split(uid_list, uid_arr)):
        log(f"\n{'─'*40}")
        log(f"Fold {fold+1}/{N_FOLDS}")

        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])

        tr_s = [train_samples[i] for u in tr_uids for i in uid2idx.get(u, [])]
        va_s = [train_samples[i] for u in va_uids for i in uid2idx.get(u, [])]
        log(f"  Train: {len(tr_s)}, Val: {len(va_s)}")

        model, sc = train_one_fold(tr_s, va_s, fold, action_w, point_w)
        models.append(model)
        scores.append(sc)

        torch.save(model.state_dict(), f"models/v4/model_v4_fold{fold+1}.pt")
        log(f"  模型已儲存: model_v4_fold{fold+1}.pt")

        # 收集 validation 機率 (用於 post-hoc 校準)
        va_ds = RallyDataset(va_s, is_train=True)
        va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        fold_a_probs, fold_a_labels = [], []
        fold_p_probs, fold_p_labels = [], []

        with torch.no_grad():
            for batch in va_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps = batch["pid_server"]
                pr = batch["pid_receiver"]
                pn = batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]

                a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True)

                fold_a_probs.append(F.softmax(a_log, dim=-1).cpu().numpy())
                fold_a_labels.extend(batch["target_action"])
                fold_p_probs.append(F.softmax(p_log, dim=-1).cpu().numpy())
                fold_p_labels.extend(batch["target_point"])

        all_va_action_probs.append(np.vstack(fold_a_probs))
        all_va_action_labels.append(np.array(fold_a_labels))
        all_va_point_probs.append(np.vstack(fold_p_probs))
        all_va_point_labels.append(np.array(fold_p_labels))

    log(f"\n{'='*60}")
    log(f"CV Score (argmax): {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    for i, s in enumerate(scores):
        log(f"  Fold {i+1}: {s:.4f}")

    # ============================================================
    # 12. Post-hoc 門檻校準
    # ============================================================
    log(f"\n{'='*60}")
    log("Post-hoc 門檻校準...")

    # 合併所有 fold 的 validation 資料
    all_a_probs = np.vstack(all_va_action_probs)
    all_a_labels = np.concatenate(all_va_action_labels)
    all_p_probs = np.vstack(all_va_point_probs)
    all_p_labels = np.concatenate(all_va_point_labels)

    log(f"\n  actionId 校準 ({len(all_a_labels)} samples):")
    action_scales, cal_f1_a = calibrate_thresholds(all_a_probs, all_a_labels, N_ACTION)

    log(f"\n  pointId 校準 ({len(all_p_labels)} samples):")
    point_scales, cal_f1_p = calibrate_thresholds(all_p_probs, all_p_labels, N_POINT)

    cal_score = 0.4 * cal_f1_a + 0.4 * cal_f1_p + 0.2 * 1.0
    log(f"\n  校準後 CV Score: {cal_score:.4f}")

    # ============================================================
    # 13. 測試預測
    # ============================================================
    log(f"\n{'='*60}")
    log("生成測試預測...")

    test_ds = RallyDataset(test_samples, is_train=False)
    test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    n_test = len(test_samples)
    ens_action = np.zeros((n_test, N_ACTION), dtype=np.float64)
    ens_point = np.zeros((n_test, N_POINT), dtype=np.float64)

    for model in models:
        model = model.to(DEVICE)
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in test_dl:
                sc = batch["seq_cat"].to(DEVICE)
                sn = batch["seq_num"].to(DEVICE)
                ps = batch["pid_server"]
                pr = batch["pid_receiver"]
                pn = batch["pid_next"]
                st = batch["static"].to(DEVICE)
                lens = batch["length"]
                nsns = batch["next_sn"]

                a_log, p_log = model(sc, sn, ps, pr, pn, st, lens,
                                     next_sns=nsns, apply_mask=True)

                bs = sc.size(0)
                ens_action[offset:offset+bs] += F.softmax(a_log, dim=-1).cpu().numpy() / N_FOLDS
                ens_point[offset:offset+bs] += F.softmax(p_log, dim=-1).cpu().numpy() / N_FOLDS
                offset += bs

    # 套用校準 scale
    pred_action = (ens_action * action_scales).argmax(axis=1)
    pred_point = (ens_point * point_scales).argmax(axis=1)

    # SGP: 直接從 test context 複製
    test_sgp = [s["sgp_label"] for s in test_samples]
    test_uids = [s["uid"] for s in test_samples]

    # ============================================================
    # 14. 儲存提交
    # ============================================================
    sub = pd.DataFrame({
        "rally_uid": test_uids,
        "actionId": pred_action,
        "pointId": pred_point,
        "serverGetPoint": test_sgp,
    })
    sub = sub.sort_values("rally_uid").reset_index(drop=True)
    sub.to_csv("submissions/submission_v4.csv", index=False)

    log(f"\n提交檔案: submission_v4.csv")
    log(f"  共 {len(sub)} 筆預測")
    log(f"  actionId: {dict(sorted(Counter(pred_action).items()))}")
    log(f"  pointId:  {dict(sorted(Counter(pred_point).items()))}")
    log(f"  sgp: {dict(sorted(Counter(test_sgp).items()))}")

    # 保存機率
    np.save("artifacts/pred_v4_action_probs.npy", ens_action)
    np.save("artifacts/pred_v4_point_probs.npy", ens_point)
    np.save("artifacts/action_scales.npy", action_scales)
    np.save("artifacts/point_scales.npy", point_scales)
    log("  機率與校準參數已儲存")

    log("\n完成!")
