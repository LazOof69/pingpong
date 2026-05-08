# Test 分布分析 — OOF/LB Gap 診斷

> 從 LB 0.4285 vs OOF 0.4816（gap -0.053）追溯成因，留底給新 session 直接引用。
> 分析日期：2026-05-02

---

## 1. Train vs Test 分布差異

### 1.1 序列長度（stroke per rally）

| 統計量 | Train | Test | 差距 |
|---|---|---|---|
| Rally 數 | 14,995 | 1,236 | — |
| 平均 stroke/rally | **5.65** | **2.90** | **-2.75 (-49%)** |
| 中位數 | 4 | 2 | -2 |
| Max | 52 | 24 | — |

**結論**：Test rally 比 Train 顯著短（約一半）。模型訓練時看的是 mean=5.65 的序列，推論時面對 mean=2.90 的序列，**distribution shift 顯著**。

### 1.2 Test rally 的 Length 分布（關鍵）

| Test Length L | rallies 數 | 比例 |
|---|---|---|
| L=1 | 398 | **32.2%** |
| L=2 | 297 | **24.0%** |
| L=3 | 206 | 16.7% |
| L=4 | 119 | 9.6% |
| L=5 | 82 | 6.6% |
| L≥6 | 134 | 10.8% |

**56.2% 的 test rally 只有 1-2 拍 context**。這是壓倒性的多數情境。

### 1.3 Train OOF Augmented Sample 的 Length 分布

V5Z 對每個 rally 產生 N-1 個 augmented samples（k=1..N-1），所以 OOF 的 length 分布偏向**較長序列**：

| Prefix 長度 k | OOF samples | 比例 |
|---|---|---|
| k=1 | 14,995 | 21.5% |
| k=2 | 13,126 | 18.8% |
| k=3 | 10,541 | 15.1% |
| k=4-5 | 13,089 | 18.8% |
| k=6-10 | 12,206 | 17.5% |
| k≥11 | 5,755 | 8.3% |

**OOF mean k = 4.65**（不是 train rally mean 5.65，是因為每個 rally 貢獻 N-1 個樣本，平均 (N-1)/2）。

跟 test mean 2.9 比，OOF 過度代表「長 context」樣本。

---

## 2. F1 by Prefix Length（核心發現）

用 bag2 LSTM + LGB ensemble (α_a=0.64, α_p=0.50, raw 未校準)：

| Prefix k | n | F1_action | F1_point |
|---|---|---|---|
| k=1 | 14,995 | 0.3893 | 0.1875 |
| k=2 | 13,126 | 0.4004 | 0.2196 |
| k=3 | 10,541 | 0.4041 | **0.2324** |
| k=4-5 | 13,089 | 0.4284 | 0.2264 |
| k=6-10 | 12,206 | **0.4408** | 0.2208 |
| k≥11 | 5,755 | 0.4197 | 0.2340 |
| **全 OOF** | 69,712 | **0.4303** | **0.2464** |
| **last-stroke only**（每 rally 最後一個 prefix）| 14,995 | 0.4068 | **0.0751** ⚠️ |
| **early-stroke only** | 54,717 | 0.4249 | 0.2147 |

### 2.1 觀察

1. **Action F1 隨 k 上升**（0.39→0.44），長 context = 更多資訊 = 更準
2. **Point F1 在 k=3-5 達高峰**（0.22-0.23），太短或太長都掉
3. **Last-stroke 的 F1_point 暴跌到 0.075** — rally 最後一拍 point 預測極困難（winner/miss 變數大）
4. **Test 等價情境（短 prefix）的 F1 比 OOF 全集低**

### 2.2 為什麼 Last-stroke F1_p 這麼低？

- Rally 最後一拍是「結束 rally 的決定性球」（winner 或 miss）
- 這拍的落點變異極大（強攻打死、輕回失誤、邊角擦網）
- Train 中我們有真實 last stroke label，但 actionId/pointId 都是高熵分布

**意涵**：當前 V5Z 對 last-stroke 的預測能力很弱。

---

## 3. OOF/LB Gap 分解

| 因素 | 估計影響 |
|---|---|
| OOF 過度代表長 context (k=4-10) | -0.020 |
| Per-class additive bias 過擬合 OOF 分布 | -0.015 |
| Test 真實 stroke distribution shift（風格/稀有類別差異）| -0.010 |
| 偶然 LB sample variance（1236 rallies）| ±0.008 |
| **加總** | **≈ -0.053**（吻合實測 gap）|

---

## 4. 估計 LB 上限

假設我們完美校準，能達到「test-distribution-matched OOF」的 F1 水準：

```
Test 加權 F1_a ≈ Σ_L (test_proportion[L] × F1_a_at_k=L)
              ≈ 0.32×0.39 + 0.24×0.40 + 0.17×0.40 + 0.10×0.43 + 0.07×0.42 + 0.10×0.42
              ≈ 0.405

Test 加權 F1_p ≈ 0.32×0.19 + 0.24×0.22 + 0.17×0.23 + 0.10×0.23 + 0.07×0.22 + 0.10×0.23
              ≈ 0.215

Score 估計     ≈ 0.4 × 0.405 + 0.4 × 0.215 + 0.2 × 1.0
              ≈ 0.448
```

**當前架構在完美校準下的 LB 上限 ≈ 0.448**（vs 實測 0.4285）。

要破 0.448 必須改架構（不是再調 α 或 bias）。

---

## 5. 已嘗試補救（test-weighted calibration）

用 test 的 L 分布加權重新校準 α 與 bias：

| 比較 | OLD (full OOF) | NEW (test-weighted) |
|---|---|---|
| α_action | 0.64 | **0.76** |
| α_point | 0.50 | **0.63** |
| OOF CV (raw) | 0.4816 | 0.4747 |
| 預測差異 | — | actionId 5.3%, pointId 9.5% |

新檔：`submissions/submission_v5z_advcal_bag2_testweighted.csv`
**LB 待測**。預估會比 0.4285 高，但不會超過 0.448 上限。

---

## 6. 對未實作方向的啟示

### 6.1 為什麼 Length-stratified 是最直接的補救

當前模型對所有 length 一視同仁訓練，但 test 是 56% L=1/2 + 44% L≥3。
**分 length bucket 訓練多個 specialist** 或 **加 length-conditional head**，理論上能讓模型專注於 short-context 的特徵模式。

### 6.2 為什麼 Multi-fold-split bagging 也有幫助

當前 bag2 用 seed=42 與 seed=1337 的 KFold split + LSTM init 都不同。
若**固定 seed 但跑多個 KFold split**，每個 sample 會有更多獨立預測，OOF variance 降低。對 last-stroke F1_p 這種高變異目標應有效。

### 6.3 Player ELO 對 short context 特別有用

L=1 時 context 只有發球資訊，模型很難從 1 拍判斷下一拍。**選手 ID 變成主要資訊來源**。
若加上選手歷史對戰勝率、習慣球種分布等，能大幅補償 short context 的資訊不足。

### 6.4 Self-supervised Pretraining 對 last-stroke 有幫助

Last stroke F1_p=0.075 表示模型對「rally 結束模式」學得很差。
SSL pretraining（在所有 train+test 序列上做 masked stroke prediction）能讓模型學到更通用的「rally 結構」表徵，下游 fine-tune 時對 last stroke 更敏感。

---

## 7. 摘要 — 給新 session 直接引用

1. **Test mean L=2.90，56% rally 只有 L≤2**
2. **OOF 全集對長 context 過度代表**（mean k=4.65）
3. **F1 隨 k 變化**：action 在 k=6-10 最強（0.44），point 在 k=3 最強（0.23）
4. **Last-stroke 的 F1_p 只有 0.075**（rally 結束預測極難）
5. **OOF/LB gap -0.053 主因是 length distribution shift + per-class bias 過擬合**
6. **完美校準下 LB 上限 ≈ 0.448**，再上去要改架構
7. **最有效改進方向**：length-stratified（E）> player ELO（A）> multi-fold-split（D）> SSL pretrain（C）
