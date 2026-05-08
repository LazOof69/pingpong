# V7 計畫（opus agent 辯論後修訂）

> 狀態：**執行中**
> 基線：V6 P1 CV = **0.4871** (F1_a=0.4358, F1_p=0.2820)
> **現實 ceiling（agent 分析）**：0.495-0.508（不是 0.515）
> 修訂目標：**0.492-0.498**

---

## 0. 辯論結論（保留原始論點以供追溯）

**原 V7 計畫最大缺陷**：0.500 目標數學上不合理：
- AUC=1.0 鎖 0.2
- oracle pointId=0.272 → OOF 頂 ~0.29 → F1_p 上限 ~0.30
- action 稀有類 (15-18) 樣本太少，Macro-F1 頂 ~0.47
- Score 頂 = 0.4·0.47 + 0.4·0.30 + 0.2·1.0 = **0.508**
- 實務 ceiling 約 **0.495-0.500**

**修訂策略**：先診斷、再動模型。seed bagging 3 小時 GPU 在 0.002 基線上是壞 ROI。

---

## 1. 修訂後的執行順序

### P0 · Diagnostic Triage（30 min，純 CPU）

**目的**：決定後續 Phase 是否值得跑。

1. **LSTM vs LGB 相關性**
   - 對每個類別算 `corr(LSTM_probs[:, k], LGB_probs[:, k])`
   - 若多數類 > 0.9 → ensemble trick 無效
2. **Per-class support histogram**
   - Train action class distribution
   - 若 class 17 < 10 樣本 → Macro-F1 永遠壓低
3. **LSTM fold-to-fold F1 variance**
   - 每 fold 算 LSTM F1_a / F1_p
   - 若 σ < 0.01 → seed bagging 無用
4. **額外**：Ensemble flatness 驗證
   - 重現 `w_a = w_p = 0.35` 是因 LGB 主導還是兩者互抵

**退出條件**：
- σ_LSTM > 0.015 且 corr < 0.85 → 繼續 P3（seed bagging 有意義）
- σ_LSTM < 0.01 或 corr > 0.9 → 跳過 P3，只做 P1+P2

### P1 · Rank Averaging + Per-Class Thresholds（1-2 hr CPU）

在**現有 V6 P1 機率**上：
1. **Rank averaging**：probs → rank → 加權平均 → 校準
2. **Per-class cost-sensitive thresholds**（不只是 bias）
   - 對 action 稀有類（15-18），直接搜 threshold 代替 bias
3. 預期增益：**+0.003 ~ +0.006**

### P2 · LGB Feature Expansion（2 hr）

**只做能讓 LGB 與 LSTM 去相關的特徵**：
- Out-of-fold target encoding + Bayesian smoothing
- 玩家的 career-level action/point 傾向（跨 rally 統計）
- **跳過 `rally_prefix_action_ngram`**（agent 指出是 seq_cat 的 re-encoding）
- 預期增益：**+0.002 ~ +0.005**

### P3 · Seed Bagging（僅當 P0 診斷通過 + P1+P2 清出 +0.007 才執行）

- 2 個 seed（非 3 個），省 ~2 小時
- 預期增益：**+0.003 ~ +0.006**

---

## 2. 回退策略

| 情境 | 行動 |
|---|---|
| P0 診斷顯示低 σ、高 corr | 跳 P3，只做 P1+P2，ship V7 at ~0.492 |
| P1 rank avg 無增益 | 保留 prob averaging，只做 per-class threshold |
| P2 LGB feat 過擬合 | 回退到 V5 feature set |
| 全部失敗 | 維持 V6 P1 submission |

---

## 3. 最差情況與最佳情況

- 最差：仍為 V6 P1 = **0.4871**
- 預期：**0.490-0.495**
- 最佳（所有 P 都命中上緣）：**0.498**

---

## 4. 當前狀態

- [ ] P0 診斷執行中
- [ ] 基於 P0 結果決定後續
