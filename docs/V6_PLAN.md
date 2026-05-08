# V6 改進計畫

> 狀態：**待執行**（等待使用者批准）
> 基線：V5 CV = **0.4851** (F1_a=0.4354, F1_p=0.2774, AUC=1.0)
> 目標：CV = **0.505 ~ 0.520**（+0.020 ~ +0.035）

---

## 1. 核心診斷（辯論 agent 結論）

**關鍵觀察**：V5 F1_point = 0.2774 已經 **逼近 oracle 上限 0.272**（in-sample 記憶化的 F1 天花板）。
這意味著：

> **訊號已經榨乾，不是架構問題，是「決策理論」問題。**
> 我們不需要更大的模型，需要更好的 Macro-F1 優化層。

### 1.1 V5 三個確認的 hard bug

| # | Bug | 證據 | 影響 |
|---|---|---|---|
| **B1** | 校準搜尋範圍 `[0.1, 5.0]` 過窄 | actionId=8 需要 scale ≈ 20 才能拉進預測 | 稀有類永遠被壓制 |
| **B2** | pointId=0 汙染 | 22% 的 augmented targets 是 rally-終點（point=0）；最後一拍 99.99% 是 pt=0 | pointId 頭學的是「是否終點」而非「落點」 |
| **B3** | Ensemble 權重跨任務共用 | LSTM 贏 action，LGB 贏 point，但強制用同一個 `w_lstm` | 兩邊都 sub-optimal |

### 1.2 為什麼這三點是「決策層」而非架構

- **B1**：模型的 **機率** 是好的，但我們用 argmax 做決策時，稀有類的機率永遠輸給多數類 → 改用 **加性 bias** 就好，免訓練。
- **B2**：模型其實學到了「此刻是否終點」這個強訊號，但我們把它混進 pointId=0 → **拆出 terminal gate** 就能讓 pointId=1~9 學真正的落點分佈。
- **B3**：兩個模型的 OOF 機率已經存在，只需要 **分別搜尋權重**，零訓練成本。

---

## 2. Golden Combo（按優先度排序）

### P0 · Plug-in Threshold Calibration（+0.010 ~ +0.025）

**理論**：Ye et al. (2012), Koyejo et al. (2014) 證明 Macro-F1 的最優決策規則是：
$$\hat{y} = \arg\max_k \left( \log p_k + b_k \right)$$
其中 $b_k$ 是 per-class 的加性 bias，可在 OOF 上用座標下降搜尋。

**與 V5 的差別**：
- V5 用 **乘性 scale** `p_k *= s_k`，`s_k ∈ [0.1, 5.0]`（等價 log 空間 `[-2.3, 1.6]`）
- V6 改用 **加性 bias** `log p_k + b_k`，`b_k ∈ [-5, 5]`（更寬且更穩定）

**實作**：
1. 替換 `calibrate_thresholds` 的搜尋邏輯
2. 對 action / point 分別搜尋（action 19 類、point 10 類）
3. 座標下降：逐類搜尋，直到 Macro-F1 收斂（通常 2~3 round）

**驗證條件**：先在 **現有的 V5 OOF 機率** 上跑，不需重訓模型。

**風險**：低。純決策層優化，模型不變。

---

### P1 · Terminal Gate for pointId（+0.008 ~ +0.020）

**問題**：pointId=0 意為「rally 結束（無下一拍）」。V5 把它當成 10 類中的一類，但它本質是 **二分類問題**（終點 vs 非終點），混進來會：
- 稀釋 pointId=1~9 的學習訊號
- 使得 pointId=0 在最後一拍的 99.99% 機率壓過其他類

**設計**：
- 新增 head: `terminal_logit`（二分類）
- pointId head 改為 **9 類**（只預測 1~9）
- 推論規則：
  ```
  if terminal_prob > 0.5:
      point_pred = 0
  else:
      point_pred = 1 + argmax(point_9_probs)  # 加上 P0 的 plug-in bias
  ```
- Loss：BCE(terminal) + CE(point_9 | not terminal)

**對 LGB 的影響**：
- LGB 的 pointId 目標也改為 9 類（只在 non-terminal 樣本上訓練）
- 加上 LGB terminal gate（同 features）

**風險**：中。需要重訓 LSTM + LGB，但邏輯清晰。

---

### P2 · Per-Task Ensemble Weights + Rank Averaging（+0.003 ~ +0.008）

**V5 的做法**：
```python
e_a = w_lstm * la_p + (1-w_lstm) * ga_p
e_p = w_lstm * lp_p + (1-w_lstm) * gp_p  # 同一個 w
```

**V6 改進**：
1. **分任務搜尋**：`w_a` 和 `w_p` 獨立
2. **Rank averaging**（選配）：先把每個模型的機率轉 rank，再加權平均
   - 優點：對校準差異不敏感
   - 缺點：丟失絕對機率資訊（與 plug-in 的 log-prob 不相容）
   - **決定：先用機率平均，若 P0 + P1 後還有差距再試 rank**

**實作**：
```python
for w_a in np.arange(0, 1.05, 0.05):
    for w_p in np.arange(0, 1.05, 0.05):
        # 分別校準 + 算 F1
```

**風險**：極低。純後處理。

---

## 3. 執行順序（由低風險到高風險）

### Phase 1：Plug-in 快速驗證（1~2 小時）

**不重訓，只改決策**：

1. 寫 `plugin_calibrate.py`：
   - 載入 V5 OOF 機率（`pred_v5_ens_action.npy` 等）
   - 用加性 bias + 座標下降搜尋
   - 輸出：per-class bias、校準後 F1
2. 若 F1_action/F1_point 提升 ≥ +0.01 → 繼續
3. 若提升 < +0.005 → 回頭檢查實作（不該失敗）

**退出條件**：F1_a + F1_p 增益 ≥ +0.015

---

### Phase 2：Per-Task Ensemble（30 分鐘）

仍不重訓：

1. 在現有 OOF 上：`for w_a, w_p in grid:`
2. 分別套 P0 的 plug-in bias
3. 記錄最佳 `(w_a*, w_p*)`

**退出條件**：Phase 1 + 2 總增益 ≥ +0.020

---

### Phase 3：Terminal Gate 重訓（5~8 小時 GPU）

**需要改 `train_v5.py` → `train_v6.py`**：

1. `PingPongModel` 加 `terminal_head`（Linear → 1）
2. pointId head 改為 9 類
3. Dataset 產生 `target_terminal`（= 1 if point==0 else 0）、`target_point_9`（= point - 1 if point > 0 else ignore）
4. Loss：`BCE + CE_9 + CE_action + terminal_bce`
5. LGB 同步改：加 `train_lgb_fold` 的 terminal 任務；pointId 只在 non-terminal 上訓練
6. 推論端：先 terminal gate 二分，再套 P0 plug-in

**退出條件**：CV ≥ 0.505

---

### Phase 4（選配）· Rank Averaging / Soft Macro-F1 Loss

僅在 Phase 3 後仍未達 0.515 時考慮：
- Rank averaging：簡單
- Soft Macro-F1 loss：需改訓練迴圈，風險較高

---

## 4. 風險與回退策略

| Phase | 風險 | 回退 |
|---|---|---|
| P1 (Plug-in) | 過擬合 OOF | 5-fold 內再切 inner CV 驗證 bias 穩定性 |
| P2 (Per-task) | `(w_a, w_p)` 過擬合 | 用 Phase 1 的 bias 固定後再搜尋 |
| P3 (Terminal) | 重訓爆掉 / 不收斂 | 保留 V5 模型、對比 loss 曲線 |

**若 Phase 3 失敗**：至少 Phase 1 + 2 已經鎖定 +0.020，不回退到 V5。

---

## 5. 產出檔案清單

| 檔案 | 用途 | Phase |
|---|---|---|
| `plugin_calibrate.py` | 加性 bias 校準（獨立腳本） | P1 |
| `eval_cv_v6.py` | 完整 V6 評測（含 per-task + plug-in + terminal） | P1~P3 |
| `train_v6.py` | V6 訓練（terminal gate） | P3 |
| `V6_PLAN.md` | 本文件 | 已 |
| `EXPLANATION.md` | 更新 V6 章節 | P3 後 |
| `submission_v6.csv` | 最終提交 | P3 後 |

---

## 6. 關鍵引言（辯論 agent）

> **"You're treating a decision-theoretic problem as an architecture problem.**
> **The V5 architecture is good enough. Fix the decision layer first."**

> **"Plug-in calibration is a free lunch that V5's multiplicative scaling left on the table.**
> **Terminal gate is Bayesian disentanglement that the pointId head can't learn on its own."**

---

## 7. 決定點

請使用者批准以下任一：

- **(A)** 執行完整 Phase 1~3
- **(B)** 先跑 Phase 1（快速驗證 plug-in 效果）再決定
- **(C)** 修改計畫（指出要改的地方）
