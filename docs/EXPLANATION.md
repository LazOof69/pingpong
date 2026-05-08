# 桌球戰術與結果預測競賽 - 完整說明文件

---

## 一、競賽目標

根據一場桌球比賽中「已發生的前 n-1 拍」時序資料，預測三件事：

| 任務 | 預測目標 | 評估指標 | 權重 |
|------|----------|----------|------|
| 任務一 | **actionId** — 下一拍（第 n 拍）的球種 | Macro F1-Score | 0.4 |
| 任務二 | **pointId** — 下一拍（第 n 拍）的落點 | Macro F1-Score | 0.4 |
| 任務三 | **serverGetPoint** — 這個回合發球者是否得分 | AUC-ROC | 0.2 |

**最終分數 = 0.4 × S1 + 0.4 × S2 + 0.2 × S3**

> Macro F1 = 所有類別的 F1 取平均，**不論該類別樣本多少都一樣重要**（稀有球種也要預測好）。  
> AUC-ROC = 雖以 AUC-ROC 評分，但**提交格式要求 serverGetPoint 必須為整數 0 或 1**（現實中一分要嘛贏、要嘛輸，不存在 0.5 分）。此時 ROC 曲線退化為兩個點，S3 上限被壓縮。

---

## 二、資料集總覽

### 2.1 檔案

| 檔案 | 用途 | 行數 |
|------|------|------|
| `train.csv` | 訓練資料（含完整標註） | 84,707 rows、14,995 個 rally |
| `test.csv` | 測試資料（不含預測目標） | 3,589 rows、1,236 個 rally |
| `sample_submission.csv` | 提交格式範例（只有標頭） | 空 |

### 2.2 資料結構

每一筆 row = **一場比賽中某一小分（rally）中的某一次揮拍**，按時間順序排列。

**一個 rally 的範例**（rally_uid=1，5 拍）：

```
rally_uid=1, strikeNumber=1: 選手1發球 → actionId=15(傳統), pointId=9(反手長球)
rally_uid=1, strikeNumber=2: 選手2接發球 → actionId=12(削球), pointId=5(中路半出台)
rally_uid=1, strikeNumber=3: 選手1回擊   → actionId=10(搓球), pointId=6(反手半出台)
rally_uid=1, strikeNumber=4: 選手2回擊   → actionId=10(搓球), pointId=5(中路半出台)
rally_uid=1, strikeNumber=5: 選手1回擊   → actionId=1(拉球),  pointId=0(出界/掛網)
→ serverGetPoint=0（發球者失分）
```

### 2.3 欄位說明

#### 識別欄位

| 欄位 | 意義 | 備註 |
|------|------|------|
| `rally_uid` | 小分的唯一 ID | train: 1~15187，test: 15188~19404（不重疊） |
| `match` | 場次編號 | 同一場比賽可能有多局 |
| `numberGame` | 局數 | 這場比賽的第幾局 |
| `rally_id` | 局內的小分編號 | 第幾局的第幾分 |
| `strikeNumber` | 拍次 | 該小分內的第幾拍（1 開始） |

#### 選手與比分

| 欄位 | 意義 | 備註 |
|------|------|------|
| `sex` | 性別 | 1=男，2=女 |
| `gamePlayerId` | 本拍擊球者 ID | 奇數拍=發球者，偶數拍=接球者 |
| `gamePlayerOtherId` | 本拍接球者 ID | 和上面相反 |
| `scoreSelf` | 擊球者目前得分 | 本局累計 |
| `scoreOther` | 對手目前得分 | 本局累計 |
| `serverGetPoint` | 發球者是否得分 | 1=是，0=否。**整個 rally 每拍此值相同**。是預測目標之一 |

#### 技術動作特徵

| 欄位 | 意義 | 值域 |
|------|------|------|
| `strikeId` | 揮拍狀態 | 1=發球, 2=接發球, 4=第三板之後, 8=無(未錄影), 16=暫停 |
| `handId` | 正反手 | 0=無, 1=正手, 2=反手 |
| `strengthId` | 擊球力道 | 0=無, 1=強, 2=中, 3=弱 |
| `spinId` | 旋轉 | 0=無, 1=上旋, 2=下旋, 3=不旋, 4=側上旋, 5=側下旋 |
| `positionId` | 球員站位 | 0=無, 1=左, 2=中, 3=右（僅前兩拍有意義） |

#### 預測目標

| 欄位 | 意義 | 值域 |
|------|------|------|
| `actionId` | **球種** | 0~18，共 19 類（見下表） |
| `pointId` | **落點** | 0~9，共 10 類（見下表） |
| `serverGetPoint` | **發球者得分** | 0 或 1 |

---

## 三、類別值域詳解

### 3.1 actionId（球種，19 類）

| ID | 名稱 | 英文 | 類型 |
|----|------|------|------|
| 0 | 無 | zero | — |
| 1 | 拉球 | drive | 進攻 |
| 2 | 反拉 | counter drive | 進攻 |
| 3 | 殺球 | smash | 進攻 |
| 4 | 擰球 | backhand twist | 進攻 |
| 5 | 快帶 | fast drive | 進攻 |
| 6 | 推擠 | fast push | 進攻 |
| 7 | 挑撥 | flip | 進攻 |
| 8 | 拱球 | pimple's long push | 控制 |
| 9 | 磕球 | pimple's fast push | 控制 |
| 10 | 搓球 | long push | 控制 |
| 11 | 擺短 | drop shot | 控制 |
| 12 | 削球 | chop | 防守 |
| 13 | 擋球 | block | 防守 |
| 14 | 放高球 | lob | 防守 |
| **15** | **傳統發球** | **traditional** | **發球** |
| **16** | **勾手發球** | **hook** | **發球** |
| **17** | **逆旋轉發球** | **reverse** | **發球** |
| **18** | **下蹲式發球** | **squat** | **發球** |

> **重要規則**：actionId 15~18 是「發球」，只能出現在 strikeNumber=1。  
> 因為 test 永遠是要預測 strikeNumber ≥ 2 的拍，**推論時必須遮罩掉 15~18**。

**訓練集 actionId 分布**：

```
actionId=0  (無):      5,597  ← 不常見
actionId=1  (拉球):   15,435  ← 最多的非發球球種
actionId=10 (搓球):   11,208  ← 第二多
actionId=15 (傳統發球): 8,562 ← 最多的發球類型
actionId=13 (擋球):    7,848
actionId=2  (反拉):    6,339
actionId=6  (推擠):    6,635
actionId=12 (削球):    4,522
actionId=5  (快帶):    4,192
actionId=11 (擺短):    3,522
actionId=4  (擰球):    2,638
actionId=3  (殺球):    2,129
actionId=16 (勾手):    1,748
actionId=7  (挑撥):    1,413
actionId=9  (磕球):      794
actionId=17 (逆旋轉):    696
actionId=14 (放高球):     613
actionId=18 (下蹲式):     444
actionId=8  (拱球):       372  ← 最稀有
```

**類別嚴重不均衡**：拉球(15,435) vs 拱球(372) 差 41 倍。Macro F1 要求稀有類也要準。

### 3.2 pointId（落點，10 類，九宮格）

把球桌分成 3×3 的九宮格，加上 0=出界/掛網：

```
              ┌──────────┬──────────┬──────────┐
    近網      │ 1 正手短  │  2 中間短 │ 3 反手短  │
              ├──────────┼──────────┼──────────┤
    半出台    │ 4 正手半  │  5 中路半 │ 6 反手半  │
              ├──────────┼──────────┼──────────┤
    底線      │ 7 正手長  │  8 中間長 │ 9 反手長  │
              └──────────┴──────────┴──────────┘
    0 = 掛網/出界（未落在九宮格內）
```

> 注意：「正手」和「反手」是**相對於接球者的持拍手**。左手和右手持拍者的正手方向相反。

**訓練集分布**：

```
pointId=9 (反手長): 17,871  ← 最多
pointId=0 (出界):   15,263  ← rally 最後一拍幾乎都是 0
pointId=8 (中間長): 13,415
pointId=7 (正手長):  9,509
pointId=5 (中路半):  9,370
pointId=6 (反手半):  5,905
pointId=4 (正手半):  5,205
pointId=2 (中間短):  4,941
pointId=1 (正手短):  2,464
pointId=3 (反手短):    764  ← 最稀有
```

> **pointId=0 的特殊性**：訓練集中 rally 最後一拍有 99.99% 的 pointId=0，因為最後一拍通常是出界或掛網。

### 3.3 serverGetPoint（二元分類）

| 值 | 意義 | 訓練集數量 |
|----|------|-----------|
| 0 | 發球者**失分** | 39,893 |
| 1 | 發球者**得分** | 44,814 |

大致平衡（53% vs 47%）。

---

## 四、Train vs Test 差異

| 特性 | Train | Test |
|------|-------|------|
| 總行數 | 84,707 | 3,589 |
| Rally 數 | 14,995 | 1,236 |
| rally_uid 範圍 | 1 ~ 15,187 | 15,188 ~ 19,404（不重疊） |
| Rally 長度 | 2~52 拍，中位數 5 | context 1~24 拍，中位數 2 |
| 性別 | 男 39,750 / 女 44,957 | 男 1,877 / 女 1,712 |
| 選手數 | 166 人 | 63 人（其中 40 人有出現在 train） |

> **關鍵**：test 中有 23 位選手**完全沒出現在 train 中**（冷啟動問題）。模型不能太依賴 player ID。

> **test 的任務**：每個 rally 提供前 1~24 拍的 context，要求預測「第 n 拍」的 actionId、pointId 以及整個 rally 的 serverGetPoint。每個 rally 只需提交一筆預測。

---

## 五、模型架構 — `train_lstm.py`

### 5.1 整體流程

```
train.csv → 資料增強 → 5-fold CV 訓練 → 5 個 BiLSTM 模型 → Ensemble 預測 → submission.csv
```

### 5.2 資料增強（核心設計）

一般做法：每個 rally 只取「用前 N-1 拍預測第 N 拍」→ 14,995 個樣本。

我們的做法：**每個 rally 長度 N，產生 N-1 個訓練樣本**：

```
Rally = [拍1, 拍2, 拍3, 拍4, 拍5]

樣本 1: context=[拍1]         → 預測 拍2 的 actionId, pointId, sgp
樣本 2: context=[拍1, 拍2]    → 預測 拍3 的 actionId, pointId, sgp
樣本 3: context=[拍1, 拍2, 拍3] → 預測 拍4 的 actionId, pointId, sgp
樣本 4: context=[拍1, 拍2, 拍3, 拍4] → 預測 拍5 的 actionId, pointId, sgp
```

> 效果：訓練樣本從 ~14,995 增加到 ~69,712，而且模型學會處理各種不同 context 長度（和 test 的分布更匹配，test 中位數 context 長度僅 2）。

### 5.3 特徵工程

#### 序列特徵（每一拍一組）

**類別特徵**（各自有獨立的 Embedding 層，維度=20）：
- `strikeId` — 發球/接發球/Rally/...
- `handId` — 正手/反手
- `strengthId` — 力道強中弱
- `spinId` — 旋轉類型
- `pointId` — 前面拍的落點
- `actionId` — 前面拍的球種
- `positionId` — 站位

**數值特徵**（3 個，標準化）：
- `scoreSelf / 15` — 自己得分
- `scoreOther / 15` — 對手得分
- `strikeNumber / 50` — 拍次

#### 靜態特徵（每個樣本一組，共 7 個）

| 特徵 | 意義 | 處理 |
|------|------|------|
| `sex` | 性別 | 原始值 |
| `numberGame / 7` | 第幾局 | 標準化 |
| `serverGetPoint` | 發球者得分（context 中已知） | 原始值 |
| `context_length / 50` | 已知拍數 | 標準化 |
| `next_strike_parity` | 下一拍是奇數拍(=1)還是偶數拍(=0) | 辨識擊球者 |
| `score_diff / 15` | 比分差 | 標準化 |
| `total_score / 30` | 比分和 | 標準化 |

#### 選手嵌入（Player Embedding，維度=16）

- `pid_server` — 發球者 ID
- `pid_receiver` — 接發球者 ID
- `pid_next` — **下一拍的擊球者 ID**（奇數拍=發球者，偶數拍=接球者）

> 捕捉選手個人風格（某些選手偏好拉球、某些偏好搓球等）。  
> 未知選手使用 padding_idx=0 的零向量。

### 5.4 神經網路架構

```
                 序列類別特徵          序列數值特徵
                     │                     │
              ┌──────┴──────┐              │
              │  7個 Embedding │              │
              │  各20維       │              │
              └──────┬──────┘              │
                     │                     │
                     └────────┬────────────┘
                              │ concat → (B, L, 143)
                    ┌─────────┴─────────┐
                    │   Input Projection  │
                    │   Linear → LayerNorm │
                    │   → ReLU → Dropout  │
                    └─────────┬─────────┘
                              │ (B, L, 192)
                    ┌─────────┴─────────┐
                    │  + Position Embed   │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │     BiLSTM          │
                    │  2層, hidden=96×2   │
                    │  → (B, L, 192)      │
                    └────┬─────────┬─────┘
                         │         │
              ┌──────────┴──┐  ┌──┴──────────┐
              │ Multi-Head   │  │ Last Hidden  │
              │ Attention    │  │ (最後有效時步) │
              │ (4 heads)    │  │              │
              │ → (B, 192)   │  │ → (B, 192)   │
              └──────┬──────┘  └──────┬──────┘
                     │                │
                     │     ┌──────────┘
                     │     │     ┌─────────────────────┐
                     │     │     │  靜態特徵(7)          │
                     │     │     │  + 選手嵌入(16×3=48)  │
                     │     │     │  → Linear(55→128)     │
                     │     │     │  → ReLU → Dropout     │
                     │     │     │  → (B, 128)           │
                     │     │     └──────────┬────────────┘
                     │     │                │
                     └─────┴────────────────┘
                              │ concat → (B, 512)
                    ┌─────────┴─────────┐
                    │    Shared Trunk     │
                    │  512 → 256 → 128    │
                    │  BN + ReLU + Drop   │
                    └─────────┬─────────┘
                              │ (B, 128)
                 ┌────────────┼────────────┐
                 │            │            │
          ┌──────┴──────┐ ┌──┴──────┐ ┌──┴──────┐
          │ Action Head  │ │Point Head│ │SGP Head  │
          │ 128→64→19   │ │128→64→10│ │128→32→1  │
          │ (分類)       │ │(分類)    │ │(二元)     │
          └──────┬──────┘ └──┬──────┘ └──┬──────┘
                 │            │            │
            actionId      pointId    serverGetPoint
           (19類 logits) (10類 logits) (1個 logit → sigmoid → 機率)
```

### 5.5 Action Masking（動作遮罩）

由於 actionId 15~18 是發球動作，只能出現在 strikeNumber=1，而 test 中預測的都是 strikeNumber ≥ 2：

```python
# 推論時：將發球動作的 logit 設為 -1e9（而非 -inf，避免 NaN 梯度）
if next_strike == 1:
    mask 掉所有非發球動作
else:
    mask 掉 actionId 15, 16, 17, 18
```

> **只在驗證和推論時套用**，訓練時不遮罩（因為 label 本身已正確，遮罩反而造成梯度不穩定）。

### 5.6 損失函數

```
總損失 = 0.4 × CrossEntropy(actionId) + 0.4 × CrossEntropy(pointId) + 0.2 × BCEWithLogits(sgp)
```

- **CrossEntropyLoss** + **class weights**（反頻率加權）+ **label smoothing=0.05**
  - Class weights：稀有類別得到更高權重，緩解不均衡
  - Label smoothing：防止模型過於自信，增加泛化能力
  - 用 CrossEntropy 而非 Focal Loss — **更穩定，不會因 class weights + masking 產生 NaN**
- **BCEWithLogitsLoss** 用於 serverGetPoint（二元分類，輸出機率）
- 損失權重 0.4:0.4:0.2 和競賽評分權重一致

### 5.7 訓練策略

| 項目 | 設定 | 說明 |
|------|------|------|
| 優化器 | AdamW (lr=3e-4, wd=1e-4) | 學習率刻意壓低以穩定訓練 |
| 排程 | CosineAnnealingWarmRestarts (T0=20, T_mult=2) | 週期性重啟 |
| 梯度裁剪 | max_norm=0.5 | 防止梯度爆炸 |
| Epochs | 最多 80 | — |
| Early Stopping | patience=15 | 以 Overall Score 為指標 |
| CV | StratifiedKFold 5-fold | 按 rally_uid 分，保證同一 rally 的所有樣本在同一 fold |
| NaN 偵測 | 跳過 NaN/Inf loss 的 batch | 防禦性措施 |

### 5.8 推論與 Ensemble

1. 5 個 fold 的模型各自對 test 做推論
2. 對 softmax 機率取平均（而非 logits 取平均）
3. actionId / pointId：取 argmax
4. serverGetPoint：平均機率 > 0.5 → 1，否則 → 0（**提交要求整數**）
5. 推論時**套用 Action Masking**

---

## 六、曾遇到的問題與修復

### 6.1 V1 — GBM 方案

- 用 LightGBM + XGBoost，actionId Macro F1 只有 ~49.6%
- 原因：只用「預測最後一拍」的樣本（14,995 筆），無資料增強

### 6.2 V2 — 第一版 LSTM（NaN 爆炸）

訓練時所有 831,788 個模型參數變成 NaN。根本原因：

1. **`-inf` 遮罩**：action masking 使用 `-inf`，softmax 後產生 0，log(0)→NaN，梯度爆炸
2. **Focal Loss 不穩定**：和 class weights + masking 組合後數值不穩定
3. **學習率太高**：8e-4，在不穩定環境下加速 NaN 擴散
4. **訓練時也套遮罩**：造成梯度在被遮罩的類別上不穩定

### 6.3 V3 — 當前修復版（6 項修復）

| 修復 | 從 | 改為 |
|------|----|------|
| 遮罩值 | `-inf` | `-1e9` |
| 損失函數 | Focal Loss | `CrossEntropyLoss` + label_smoothing=0.05 |
| 訓練遮罩 | 訓練+推論都遮罩 | **只有推論時遮罩** |
| 學習率 | 8e-4 | 3e-4 |
| 梯度裁剪 | 1.0 | 0.5 |
| NaN 保護 | 無 | 偵測 NaN/Inf loss → 跳過 batch |

---

## 七、輸出檔案

| 檔案 | 內容 |
|------|------|
| `submission_int.csv` | **正式提交用**。serverGetPoint 為 0/1 整數（符合格式要求） |
| `submission.csv` | 參考用。serverGetPoint 為機率值（僅供檢視模型信心分布） |
| `model_fold{1-5}.pt` | 5 個 fold 的模型權重 |
| `pred_action_probs.npy` | actionId 的 softmax 機率 (1236×19) |
| `pred_point_probs.npy` | pointId 的 softmax 機率 (1236×10) |
| `pred_sgp_probs.npy` | serverGetPoint 的 sigmoid 機率 (1236,) |

---

## 八、執行方式

```powershell
cd "C:\Users\butte\OneDrive\桌面\code projects\競賽\pingpong"
python -u train_lstm.py
```

預計訓練時間：RTX 3060 約 20~40 分鐘（取決於 early stopping）。

---

## 九、潛在改進方向

1. **Transformer 替代 LSTM** — Self-attention 可能更好捕捉長距離依賴
2. **Ensemble with GBM** — 結合 LSTM 的機率和 LightGBM 的機率（加權平均或 stacking）
3. **特徵交互** — 加入「前一拍的 actionId × 當前 handId」等交叉特徵
4. **更細的 player embedding** — 用選手的歷史統計（常用球種、勝率）作為額外特徵
5. **對抗訓練** — Mixup / CutMix 做資料增強
6. **Post-processing** — 根據桌球規則做後處理（如某些球種後接某些球種的機率極低）
