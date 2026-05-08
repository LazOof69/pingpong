# 桌球戰術預測系統 — 新手導讀

> 寫給第一次接觸這個 repo 的人。看完應該能理解我們在做什麼、系統怎麼運作、為什麼這樣設計。

---

## 1. 這個比賽在做什麼？

### 1.1 任務

一場桌球比賽由很多 **rally**（一個發球到失分為止的回合）組成。
每個 rally 由多個 **stroke**（揮拍）組成。

我們拿到的資料是這樣的：

```
rally_uid=15188:
  stroke 1: 發球, 正手, 強旋, ..., positionId=1
  stroke 2: 反手抽擊, 中間, ...
  stroke 3: 殺球, 正手, ...
  stroke 4: ???                 ← 這個我們要預測
```

要預測：
- **actionId**（球種，19 類）：殺球？削球？切球？...
- **pointId**（落點，10 類）：左半台短？右半台底線？...
- **serverGetPoint**（這個 rally 最後是發球方贏還是接球方贏，二分類）

### 1.2 評分公式

```
Score = 0.4 × Macro-F1(actionId) + 0.4 × Macro-F1(pointId) + 0.2 × AUC(serverGetPoint)
```

**Macro-F1** = 對每個類別算 F1 然後平均。**重點**：稀有類別跟常見類別等權，所以「對少數類別的辨識能力」很重要，不能只猜大類別。

### 1.3 訊號 / 雜訊比

- 14,995 個訓練 rally / 1,236 個測試 rally
- 訓練 rally 平均 5.65 拍、測試 rally 平均 **2.90 拍**（test 顯著比 train 短）
- `serverGetPoint` 在每個 rally 裡是常數（rally 結果），test 也看得到，**直接複製就 AUC ≈ 1.0**，所以這 0.2 的分數幾乎是白送

真正要拼的是 **Macro-F1(action)** 和 **Macro-F1(point)**。

---

## 2. 資料長什麼樣？

### 2.1 三個檔案

```
data/
├── train.csv         — 訓練資料（14,995 rallies, 84,707 strokes）
├── test.csv          — 測試資料（1,236 rallies, 3,589 strokes，最後一拍未給）
└── sample_submission.csv  — 提交檔範例
```

### 2.2 欄位說明

| 欄位 | 意義 |
|---|---|
| `rally_uid` | 一場 rally 的識別碼 |
| `strikeNumber` | 這拍是第幾拍（1, 2, 3, ...） |
| `actionId` | 球種（0-18），預測目標之一 |
| `pointId` | 落點（0-9），預測目標之一 |
| `serverGetPoint` | rally 最終發球方是否得分（0/1），rally 級常數，預測目標之一 |
| `gamePlayerId` / `gamePlayerOtherId` | 雙方選手 ID |
| `handId`, `strengthId`, `spinId`, `positionId` | 其他擊球屬性 |
| `scoreSelf`, `scoreOther` | 當前比分 |
| `sex`, `match`, `numberGame` | 比賽 metadata |

### 2.3 提交格式

每個 test rally 一筆，4 欄：

```csv
rally_uid,actionId,pointId,serverGetPoint
15188,4,9,1
15191,10,9,1
...
```

---

## 3. 解法概念（一句話總結）

> **兩個互補模型 ensemble + 直接優化 macro-F1 的後處理校準。**

### 3.1 為什麼用兩個模型？

- **LSTM**：擅長序列模式（這拍接什麼球種有時間關聯）
- **LightGBM (LGB)**：擅長手工特徵互動（比分壓力 × 球種 × 落點 × 選手）

兩個模型的「擅長領域」不同，**錯誤模式不同**，平均後互補。這比單一模型強。

### 3.2 為什麼還要做後處理校準？

模型輸出的是「機率」，但 macro-F1 的最佳決策邊界 **不一定是 argmax**。

比如某個稀有類別 cls 8（只佔 0.5%），模型給它的機率永遠不是最高，所以從不被預測，F1=0。

我們在後處理加 **per-class log-bias**（推薦理論：Koyejo et al. 2014）：

```
prediction = argmax_k ( log p_k + b_k )
```

`b_k` 用 OOF 資料 grid search 直接優化 macro-F1。
這個 bias 等同於「人為提高稀有類別的機率」，讓它們有機會被選中。

---

## 4. 系統架構

### 4.1 完整 Pipeline

```
data/train.csv ────┐
                   ├── prepare_samples ──┬── LSTM (BiLSTM + Attention)
                   ├── build_lgb_features┴── LGB (49 特徵)
data/test.csv  ────┘
                                │
                       OOF probs (5-fold CV)
                                │
                    α-blend (LSTM × α + LGB × 1-α)
                                │
                    Plug-in additive bias 校準
                                │
                    Test predictions ──> submission.csv
```

### 4.2 LSTM 子模型 (V5-Z)

```
Input: stroke 序列（每 stroke 7 個類別特徵 + 3 個數值特徵）
        ↓
[Embedding × 7] + [Numeric] + [Position embedding]
        ↓
[BiLSTM 192d / 2 layers]
        ↓
[Multi-head Attention 4 heads] + [Last hidden]
        ↓
[Player embedding × 3 (server/receiver/next)] ─── concat
        ↓
[Fusion MLP] ──> Action head (19 classes)
                       ↓ softmax
              ──> Point head (10 classes)  ← 串聯設計：吃 action 機率
              ──> Aux head: 每位置預測下一 stroke（only forward LSTM states，因果安全）
```

**關鍵設計**：

- **BiLSTM 雙向編碼**：對 prefix 內的所有位置看前後文，但 target 在 prefix 外（不洩漏）
- **Attention pooling**：序列彙整成一個向量，比單純取最後一拍更好
- **Action → Point 串聯**：球種影響落點，所以 action 機率餵給 point head
- **Aux head 只用 forward LSTM states**：這樣每個中間位置可以預測下一拍，不洩漏未來資訊
- **Player embedding**：選手身份很重要（不同選手戰術不同）

### 4.3 LGB 子模型

49 個手工特徵：

```
Context-level:        ctx_len, next_strike_num, sex, numberGame
Score:                scoreSelf, scoreOther, score_diff, score_sum, is_deuce, game_point
Last/Prev/Serve 拍:   actionId, pointId, handId, spinId, strengthId, positionId, action_group
Group counts:         attack/control/defense 出現次數
Point counts:         short/half/long、fore/mid/back 出現次數
Cross features:       last_action × point, last_hand × point, ...
Player IDs:           player_self, player_other
```

**注意**：移除了 `serverGetPoint` 欄位（per-rally 常數，當作 stroke-level 特徵會洩漏結果）。

### 4.4 Cross-Validation + OOF 機制

```
14,995 rallies → StratifiedKFold(5)
                    ↓
Fold 1: 訓練 80% / 驗證 20%  ─> OOF probs (這 20%)
Fold 2: 訓練 80% / 驗證 20%  ─> OOF probs (這 20%)
...
Fold 5: ...                  ─> OOF probs (這 20%)
                                    ↓
              拼接 5 折 = 全部 14,995 rallies 的 OOF probs
                                    ↓
                         用來做 α 搜尋 + bias 校準
```

OOF (Out-of-Fold) 的關鍵：每個樣本的預測都是 **沒看過這個樣本的模型** 算出來的，所以可以模擬「真正的測試表現」。

### 4.5 Augmentation：對每個 rally 產生多個訓練樣本

對一個有 N 拍的 rally，我們不只產生 1 個訓練樣本，而是 **N-1 個**：

```
Rally [s1, s2, s3, s4, s5]:
  Sample 1: input=[s1],         target=s2
  Sample 2: input=[s1,s2],      target=s3
  Sample 3: input=[s1,s2,s3],   target=s4
  Sample 4: input=[s1,s2,s3,s4],target=s5
```

這樣同一個 rally 可以變成多個（context, next-stroke）pair，**訓練訊號量增加 5 倍**。
而且自然涵蓋不同 context 長度，對應 test 中各種長度的 rally。

### 4.6 Bagging：seed-level diversity

不同隨機種子訓練的 LSTM 學到稍微不同的東西。把 2-3 個 seed 的 OOF probs 平均，可以降低 variance。

我們最終用 `{seed=42, seed=1337}` bag 2，丟掉 `seed=2024`（單獨表現比另兩個差）。

### 4.7 Calibration：用 plug-in additive log-bias

```python
# 對 action 19 類，每類找一個 b_k
ens_probs = α × LSTM_probs + (1-α) × LGB_probs
log_probs = log(ens_probs)
prediction = argmax_k (log_probs + b)  # b 是 19 維的 bias 向量

# 用 coordinate descent 在 OOF 上 grid search:
for round in 1..4:
    for k in 0..18:
        for v in [-2.5, -2.45, ..., 2.5]:
            try b[k] = v, 算 macro-F1
        keep 最好的 v
```

**為什麼這個方法？** 因為 macro-F1 是 non-decomposable 的（每個類別 F1 互相影響），standard CE 訓練的最佳 argmax 不等於最佳 macro-F1 決策。
plug-in bias 可以直接補償這個 gap，**theoretically optimal under 條件**（Koyejo 2014）。

---

## 5. 為什麼這樣設計？關鍵 Trade-off

### 5.1 為什麼選 BiLSTM 不是 Transformer？

實驗結果：Transformer 在這個任務上 **CV 0.4755 < BiLSTM 0.4788**。

原因：
1. **70k 樣本太少**：Transformer 通常需要 1M+ 樣本
2. **序列太短（mean 5.65）**：attention 的 long-range 優勢沒發揮空間
3. **RNN structural bias 適合短序列**：天然有「順序處理」的偏置

### 5.2 為什麼有了 class weights 還要 bias 校準？

- **class weights** 在訓練時調整 loss 對稀有類別的重視程度
- **bias 校準** 在推論時調整決策邊界

兩者解決不同問題：
- class weights 讓模型 **學到** 稀有類別的特徵
- bias 校準讓模型 **預測** 稀有類別（即使機率沒最高）

### 5.3 為什麼試了 Focal Loss 卻退步？

Focal Loss 的設計是「降低簡單樣本的 loss 權重」。但我們已經用 class weights 平衡 imbalance，再加 focal 等於 **雙重抑制大類**，反而拖累整體 F1。

**啟示**：technique 之間有相剋，不是疊越多越好。

### 5.4 為什麼 GRU、Larger LSTM、Soft-F1 都退步？

- **GRU**：與 LSTM 學到太相似的東西，ensemble 沒有多樣性貢獻
- **Larger LSTM**：70k 樣本上 192d 已經 saturate，再大就 overfit
- **Soft-F1 loss**：與 plug-in bias 有重疊功能，loss-level 與 post-hoc 二擇一即可

### 5.5 為什麼 OOF 0.4816 但 LB 0.4285？

OOF 混了「短 prefix（容易）」和「長 prefix（難）」的預測，但 test 主要是短 prefix（mean 2.9）。

**修補**：對 OOF 樣本根據 test 的 length 分布加權，重新校準 α 和 bias。新版校準的 α 從 0.64/0.50 → 0.76/0.63（LSTM 比重提高），對應「短 context 下 LSTM 比 LGB 更可靠」的事實。

---

## 6. 怎麼跑這個系統？

### 6.1 環境需求

```
Python 3.10+
PyTorch (with CUDA)
LightGBM
scikit-learn
pandas, numpy
```

GPU 建議 ≥ 4GB VRAM（RTX 3060 6GB 跑得動）。

### 6.2 訓練單個模型（seed=42）

```bash
cd /mnt/c/Users/butte/OneDrive/桌面/code\ projects/競賽/pingpong
V5Z_SEED=42 V5Z_TAG=v5z_s42 python3 src/train/train_v5z.py
```

輸出：
- `models/v5z/model_v5z_s42_lstm_fold{1..5}.pt` — LSTM 權重
- `artifacts/v5z_s42_oof.npz` — OOF 預測 + 標籤
- `artifacts/v5z_s42_test.npz` — Test 預測
- `submissions/submission_v5z_s42.csv` — 單 seed 提交檔

時間：RTX 3060 約 30-40 分鐘。

### 6.3 訓練多 seed 做 bagging

```bash
V5Z_SEED=42   V5Z_TAG=v5z_s42   python3 src/train/train_v5z.py
V5Z_SEED=1337 V5Z_TAG=v5z_s1337 python3 src/train/train_v5z.py
```

### 6.4 後處理 calibration（生成最終提交）

```bash
V5Z_SEEDS="42,1337" python3 src/train/advcal_v5z.py
```

輸出：`submissions/submission_v5z_advcal_bag2.csv`

### 6.5 切換架構（Env vars）

```bash
V5Z_ARCH=gru          # 用 GRU 代替 LSTM
V5Z_ARCH=transformer  # 用 BiTransformer
V5Z_FOCAL=2.0         # 啟用 focal loss γ=2
V5Z_SOFTF1=0.3        # 加入 soft macro-F1 loss
V5Z_HIDDEN=256        # 更大的 hidden size
V5Z_LAYERS=3          # 更多層
V5Z_EPOCHS=50         # 更多 epoch
```

（注意：實驗證明這些大多會退步，預設配置已是最佳）

---

## 7. 概念詞彙表

| 詞 | 意義 |
|---|---|
| **Rally** | 一個發球到結束的完整回合 |
| **Stroke** | 一拍揮拍動作 |
| **OOF (Out-of-Fold)** | 在 cross-validation 中，每筆樣本的預測都來自「沒看過這筆樣本」的模型 |
| **Macro-F1** | 對每個類別算 F1 後平均，稀有類等權 |
| **Bagging** | 多個獨立模型的預測平均，降低 variance |
| **Ensemble** | 多個不同類型模型的輸出組合 |
| **Plug-in additive bias** | 在 log 機率上加 per-class 常數 b_k 來優化 macro-F1 |
| **Class weight** | 訓練時稀有類別 loss 放大，避免被忽略 |
| **Augmentation** | 從一筆原始資料產生多筆訓練樣本 |
| **Calibration** | 後處理校準模型輸出，使其更貼近真實分布或目標指標 |
| **Teacher forcing** | 訓練時部分機率用真實 label 取代模型預測作為下游輸入 |
| **Player embedding** | 把選手 ID 學成低維向量，捕捉選手風格 |
| **Causal mask** | 在 attention/RNN 中只能看過去不能看未來 |

---

## 8. 看程式碼從哪裡開始？

| 想了解 | 看哪個檔 |
|---|---|
| 整個訓練流程 | `src/train/train_v5z.py` 從 `if __name__ == "__main__"` 看起 |
| LSTM 模型結構 | `src/train/train_v5z.py` 的 `class PingPongModel` |
| 樣本怎麼準備 | `src/train/train_v5z.py` 的 `prepare_samples()` 和 `build_lgb_features()` |
| 後處理校準 | `src/train/advcal_v5z.py` 的 `plugin_add()` |
| baseline 對照 | `baseline code/baseline code.py`（單檔簡單實作） |
| 為什麼這樣做 | `docs/V5Z_FINAL_REPORT.md`（完整實驗 retrospective） |

---

## 9. 結語

這個系統不是一次到位寫出來的。經過 14 個失敗實驗（Transformer、GRU、Focal Loss、Soft-F1、更大模型、不同 calibration 等），才確認 V5-Z 配置是當前資料量與序列特性下的最佳組合。

**核心經驗**：
1. **資料量決定模型容量上限** — 70k 樣本撐不起 Transformer
2. **Technique 會相剋** — class weights + focal 是雙重抑制
3. **OOF ≠ test** — 校準時要意識到 distribution shift（特別是 length 分布）
4. **後處理 calibration 比 in-loss 重要** — plug-in bias 是 macro-F1 任務的金標
5. **多樣性比強度重要** — bagging 兩個獨立 LSTM 比訓練一個更大的 LSTM 好

如果你要改這個系統，先讀 `V5Z_FINAL_REPORT.md` 知道哪些路已經走過。
