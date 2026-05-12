# 桌球戰術預測賽 — V5-Z

> Next-stroke prediction for competitive table tennis rallies.
> 給定 rally 前 n-1 拍，預測第 n 拍的球種、落點與整個 rally 結果。

---

## 1. 任務

一場桌球比賽由多個 **rally**（一個發球到失分為止的回合）組成，每個 rally 由多個 **stroke**（揮拍）組成。比賽資料逐 stroke 標註動作、力道、旋轉、落點等屬性。

給定一個 rally 的前 n-1 拍，要同時預測：

| 目標 | 類別數 | 指標 | 權重 |
|---|---|---|---|
| `actionId` — 第 n 拍球種 | 19 | Macro-F1 | 0.4 |
| `pointId` — 第 n 拍落點 | 10 | Macro-F1 | 0.4 |
| `serverGetPoint` — rally 最終發球方是否得分 | 2 | AUC | 0.2 |

```
Score = 0.4 × Macro-F1(actionId) + 0.4 × Macro-F1(pointId) + 0.2 × AUC(serverGetPoint)
```

`serverGetPoint` 是 rally-level 常數、test 全可見，直接複製即可拿到 AUC ≈ 1.0；真正戰場是兩個 Macro-F1。

---

## 2. 資料

```
data/
├── train.csv         — 14,995 rallies / 84,707 strokes（含完整標註）
├── test.csv          — 1,236 rallies / 3,589 strokes（最後一拍未給）
└── sample_submission.csv
```

關鍵分布差異：
- 訓練 rally 平均 **5.65 拍**，測試 rally 平均 **2.90 拍**
- **56% 的測試 rally 只有 L ≤ 2**
- 校準時若不對 OOF 做 length-weighted 加權，會嚴重高估 LB

完整欄位定義見 [`docs/rule.md`](docs/rule.md)（含主辦 ID 對照表：strikeId、handId、strengthId、spinId、actionId 19 類、pointId 10 類、positionId）。

---

## 3. 當前最佳成績

| 項目 | 值 |
|---|---|
| **Ship 檔** | `submissions/submission_v5z_md_advcal_bag2_PSEUDO.csv` |
| **Public LB**（May 4 sklearn env，frozen） | **0.3578** |
| OOF CV（PSEUDO_v1） | 0.4366 |
| OOF CV（V5-Z baseline，無 PSEUDO） | 0.4816 |
| Baseline floor（主辦 baseline code） | 0.4143 |
| 目前榜首 LB | 0.4459 |

OOF/LB 落差 ≈ 0.09，主要肇因於 test 短 rally 比例顯著高於 train（distribution shift on context length）。完整 retrospective 見 [`docs/V5Z_FINAL_REPORT.md`](docs/V5Z_FINAL_REPORT.md)，分布診斷見 [`docs/TEST_DISTRIBUTION_ANALYSIS.md`](docs/TEST_DISTRIBUTION_ANALYSIS.md)。

---

## 4. 解法概念

> **兩個互補模型 ensemble + 直接優化 macro-F1 的後處理校準**

```
data/train.csv ──┐
                 ├─ prepare_samples ─┬─ BiLSTM (V5-Z, 192d / 2L)
                 ├─ build_lgb       ─┴─ LightGBM (49 hand-crafted features)
data/test.csv  ──┘
                          ↓
                  OOF probs (5-fold StratifiedKFold)
                          ↓
                  α-blend：α·LSTM + (1-α)·LGB
                          ↓
                  Plug-in additive log-bias 校準（Koyejo 2014）
                          ↓
                  Test predictions → submission.csv
```

### 4.1 LSTM 子模型 — V5-Z

| 元件 | 設定 |
|---|---|
| 輸入 | 7 個 categorical（strike/hand/strength/spin/point/action/position）+ 3 個 numeric（scoreSelf/Other/strikeNumber） |
| Embedding | 每 cat 維度 20 + position embedding |
| Player embedding | 16 維 × 3 路（server / receiver / next），dropout 0.3 |
| Encoder | **BiLSTM 192d / 2 layers / dropout 0.3** |
| Pooling | Multi-head attention（4 heads）+ last hidden |
| Action head | MLP(128 → 64 → 19) |
| Point head | MLP(128+19 → 64 → 10)，串聯吃 softmax(action) |
| Aux head | Forward-only LSTM states → 每位置預測下一 stroke（因果安全） |
| Loss | 0.5·CE(action) + 0.5·CE(point) + 0.1·Aux，class-weighted + label smoothing 0.05 |
| Optimizer | AdamW lr=3e-4 wd=1e-4 + CosineAnnealingWarmRestarts |
| Schedule | 40 epoch / patience 10，OOF score early stop |
| Augmentation | 每個 N-stroke rally → N-1 個 (prefix, next) 訓練樣本 |

### 4.2 LightGBM 子模型

49 個手工特徵：context 長度、比分壓力（is_deuce、game_point）、last/prev/serve 三拍的 7 個 categorical、attack/control/defense group counts、point depth/side counts、交互特徵（last_action × pointId 等）、player IDs。

注意：`serverGetPoint` **已被移除**為輸入特徵 — rally-level 常數做 stroke-level feature 等於直接洩漏結果。

### 4.3 Bagging

最終 bag：`seed=42 + seed=1337`（捨棄 `seed=2024`，單獨 CV 偏弱 0.002+ 會稀釋訊號）。每個 seed 跑獨立 StratifiedKFold + 獨立 LSTM init + 獨立 LGB seed。

### 4.4 Plug-in additive log-bias 校準

對 macro-F1 有理論最優保證（Koyejo et al., 2014）。

```python
prediction = argmax_k ( log p_k + b_k )
# coordinate descent, 4 rounds
# α 步長 0.01 掃 [0, 1]，b_k 步長 0.05 掃 [-2.5, 2.5]
```

校準結果（V5-Z baseline）：
- α_action = 0.64，α_point = 0.50
- F1_action 0.4303 → 0.4389（+0.0086）
- F1_point 0.2464 → 0.2650（+0.0186）

---

## 5. 環境與執行

### 5.1 環境

- Python 3.10+
- PyTorch（含 CUDA）
- LightGBM
- scikit-learn（**注意：sklearn 1.8.0 與 May 4 env 不可互換**，詳見 §6.2）
- pandas、numpy
- GPU ≥ 4GB VRAM（RTX 3060 6GB 可跑完整 pipeline）

### 5.2 訓練單個 seed

```bash
V5Z_SEED=42 V5Z_TAG=v5z_s42 python3 src/train/train_v5z.py
```

輸出：
- `models/v5z/model_v5z_s42_lstm_fold{1..5}.pt`
- `artifacts/v5z_s42_oof.npz` / `artifacts/v5z_s42_test.npz`
- `submissions/submission_v5z_s42.csv`

時間：RTX 3060 約 30-40 分鐘。

### 5.3 多 seed bagging

```bash
V5Z_SEED=42   V5Z_TAG=v5z_s42   python3 src/train/train_v5z.py
V5Z_SEED=1337 V5Z_TAG=v5z_s1337 python3 src/train/train_v5z.py
```

### 5.4 後處理校準 → 產生提交檔

```bash
V5Z_SEEDS="42,1337" python3 src/train/advcal_v5z.py
```

輸出：`submissions/submission_v5z_advcal_bag2.csv`

### 5.5 切換架構（多為退步驗證用）

```bash
V5Z_ARCH=gru | transformer   # 切換 encoder
V5Z_HIDDEN=256               # 更大 hidden（會 overfit）
V5Z_LAYERS=3
V5Z_EPOCHS=50
V5Z_FOCAL=2.0                # 啟用 focal loss（與 class weights 重疊抑制）
V5Z_SOFTF1=0.3               # 加 soft macro-F1 loss
```

實驗證明上述大多會退步，預設配置已是 70k 樣本量下的最佳。完整 env vars 清單見 `docs/V5Z_FINAL_REPORT.md` §5.2。

---

## 6. 關鍵發現與雷區

### 6.1 22 個已驗證失敗變體（KILL list）

任何新方向動工前必須答：「這跟下列已驗證失敗變體有什麼根本不同？」

| 失敗變體 | 為什麼失敗 |
|---|---|
| `serverGetPoint` 當 stroke-level feature | 直接洩漏 rally outcome |
| seed=2024 加進 bag | 該 seed 偏弱 0.002+，稀釋訊號 |
| GRU / BiTransformer | 70k 樣本太少；序列太短（mean 5.65）|
| Larger LSTM (256d / 3L) | 70k 樣本上 saturate，再大就 overfit |
| Focal Loss | 與 class weights 雙重抑制大類 |
| Soft-F1 Loss | 與 plug-in calibration 重疊功能 |
| LGB serve mask | LSTM 已 mask serves，邊際效益 noise 內 |
| 強制 `labels=range(N)` 校準 | class 17/18 永不出現，會被誤拉低 |
| 粗 grid 校準 | 細 grid (α=0.01, b=0.05) 才是金標 |
| Length-stratified training | 對抗 distribution shift 路徑驗證 saturated |
| Truncation augmentation | Plan W reincarnation — 改變 first-moment bucket weight |
| ShuttleSet external data | G2 schema verify：真實 overlap 44%，遠低於 plan 假設的 85% |
| Per-player features | tail-class density median 4/cell，SE 比 lift target 大 22× |
| Embedding mixup | uniform-λ 不改 first-moment + LSTM gate 破壞 mixup linearity |
| MC Dropout TTA | flip 與真標相關性 0，F1_action/F1_point 同時退步 |
| Cross-family stack (CatBoost/XGB) | nested OOF blend 反低於 G4 |
| Hierarchical action prediction | coarse→fine 邏輯邊際 |
| Decision Transformer (D4) | Pearson 0.91/0.94 與 V5Z 高度相關，G4 marginal fail |

20-KILL base rate：**0/20 = 0% LB +0.005 lift** in tested mechanism classes。完整列表見 `docs/V5Z_FINAL_REPORT.md` §2。

### 6.2 sklearn env regression（2026-05-12 active）

**現象**：sklearn 1.8.0 下 `StratifiedGroupKFold` 的 fold splits 與 May 4 env 不同 → α/bias 校準偏移 → test predictions 跑掉，同 OOF 下 LB 退 0.010。

**狀態**：搶救計畫進行中（`docs/RESCUE_PLAN_2026-05-12.md`）。Ship 檔不換，但目前環境無法重現 0.3578 LB。

**啟示**：env shift 在這個資料量下 dominate within-env tuning by 10×。任何長期計畫須先固定 sklearn 版本 + freeze fold splits。

### 6.3 三項決策準則（每個 phase 必看）

1. **Full OOF CV** — α-blended、post-bias 的傳統 macro-F1
2. **Test-weighted OOF** — 用 test L 分布加權的 OOF，**主決策指標**
3. **F1 by length bucket** — k=1, 2, 3, 4-5, 6-10, 11+ 各自的 F1_action / F1_point

決策規則：
- Test-weighted OOF 退步 > 0.002 → abandon
- Full OOF 升但 test-weighted OOF 沒升 → long-context overfit
- LB 是 1236 rallies，sample variance ±0.008 是常態，每方向 ≤ 2 次提交

### 6.4 核心瓶頸

完美校準 LB 上限 ≈ 0.448（接近 #1 的 0.4459）。我們 0.3578，gap 0.088 ≈ 完整 OOF→LB gap。

要追 #1 須結構性突破：
- ~~Capacity extension~~（saturated）
- ~~Inference-time TTA~~（pilot KILL）
- ~~Generative augmentation~~（Debate-KILL）
- ~~Cross-family ensemble~~（D2 KILL）
- ~~Decision Transformer~~（D4 KILL）
- **Manual error analysis + targeted features**（剩餘長期路徑）
- **PSEUDO + 3-feature LGB ablation**（Phase 7-D 結合，剩餘短期路徑）

---

## 7. Repo 結構

```
.
├── CLAUDE.md                       — 給 Claude session 的 working agreement（動工前必讀）
├── README.md                       — 本檔
├── data/                           — train.csv / test.csv / sample_submission.csv
├── baseline code/                  — 主辦提供 baseline（floor CV 0.4143）
├── src/
│   ├── train/
│   │   ├── train_v5z.py            — 主訓練 pipeline（LSTM + LGB OOF）
│   │   ├── advcal_v5z.py           — 校準 + ship submission
│   │   ├── pseudo_label_lgb*.py    — Phase 7-D PSEUDO 機制
│   │   ├── d1_hierarchical_*.py    — Phase D1 hierarchical 嘗試
│   │   ├── d4_phase_d_*.py         — Phase D4 Decision Transformer
│   │   ├── cf_blend.py / cross_family_stack.py — D2 cross-family GBDT
│   │   └── ...
│   ├── predict/                    — 老版 predict / diagnose 腳本
│   └── analysis/
├── artifacts/                      — OOF / test npz（每 seed 一份）
│   └── _old_v7_stale/              — 前版資料訓練的 artifacts，已歸檔
├── models/v5z/                     — LSTM 權重
├── submissions/                    — 所有 ship 過的 CSV
├── logs/                           — 所有訓練 / 校準 log
├── scripts/                        — utility scripts
└── docs/                           — 完整文檔（見下節）
```

---

## 8. 文檔導覽

| 檔案 | 用途 |
|---|---|
| [`docs/SYSTEM_GUIDE.md`](docs/SYSTEM_GUIDE.md) | 新手導讀，含完整 pipeline 圖、概念詞彙表、Q&A |
| [`docs/V5Z_FINAL_REPORT.md`](docs/V5Z_FINAL_REPORT.md) | 權威 retrospective，所有 phase 結束後寫這裡 |
| [`docs/TEST_DISTRIBUTION_ANALYSIS.md`](docs/TEST_DISTRIBUTION_ANALYSIS.md) | OOF/LB gap 診斷與分布分析 |
| [`docs/RESCUE_PLAN_2026-05-12.md`](docs/RESCUE_PLAN_2026-05-12.md) | sklearn env regression 搶救計畫 |
| [`docs/rule.md`](docs/rule.md) | 主辦官方任務說明 + ID 對照表 |
| [`CLAUDE.md`](CLAUDE.md) | 工作流程鐵律（Plan → Debate#1 → Implement → Code Review → Debate#2 → 三指標驗證 → 寫 retrospective）|
| `docs/PHASE_*` / `docs/D*_*` | 各 phase 的 plan、briefing、retrospective |

---

## 9. 核心經驗（5 條）

1. **資料量決定模型容量上限** — 70k 樣本撐不起 Transformer / GRU / 更大 LSTM
2. **Technique 會相剋** — class weights + Focal Loss 是雙重抑制；Soft-F1 + plug-in bias 是功能重疊
3. **OOF ≠ test** — 校準時要意識到 distribution shift（特別是 length 分布）
4. **後處理 calibration 比 in-loss 重要** — plug-in additive bias 是 macro-F1 任務的金標
5. **多樣性比強度重要** — bagging 兩個獨立 LSTM 比訓練一個更大的 LSTM 好；但 ensemble 內部必須有 inductive bias 差異，否則 ensemble noise 內

---

## 10. 授權與引用

本 repo 為個人競賽用途。主辦資料、ID 對照與評分公式詳見 [`docs/rule.md`](docs/rule.md)。

外部理論引用：
- Koyejo, O. O., Natarajan, N., Ravikumar, P., & Dhillon, I. S. (2014). *Consistent Binary Classification with Generalized Performance Metrics.* NeurIPS. — Plug-in additive bias 的最優性依據
