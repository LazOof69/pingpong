# CLAUDE.md — 桌球戰術預測賽

> 給未來 Claude session 的 working agreement。動工前必讀。

---

## 1. 專案速覽

桌球 next-stroke 預測競賽。給 rally 前 n-1 拍，預測第 n 拍的 `actionId` (19 類)、`pointId` (10 類)、與整個 rally 結果 `serverGetPoint` (二分類)。

**評分**：`0.4 × Macro-F1(action) + 0.4 × Macro-F1(point) + 0.2 × AUC(serverGetPoint)`

`serverGetPoint` 是 rally-level 常數，test 可見直接複製即可 (~0.998 AUC)。真正戰場是兩個 Macro-F1。

---

## 2. 當前狀態（frozen at 2026-05-02）

- **Ship**：`submissions/submission_v5z_advcal_bag2.csv`
- **OOF CV**：0.4816 ｜ **LB**：0.4285 ｜ **Gap**：-0.053
- **V5Z 框架已 saturate**，14 個架構/loss/calibration 變體全 regression（詳見 `docs/V5Z_FINAL_REPORT.md` §2）
- **唯一未測 ship**：`submissions/submission_v5z_advcal_bag2_testweighted.csv`（test L 分布加權校準後）

**核心瓶頸（已診斷）**：train mean L=5.65 vs test mean L=2.90，56% test rally 只有 L≤2。OOF 過度代表長 context (mean k=4.65)。完美校準 LB 上限 ≈ 0.448 — 要破必改 model itself，不能再調 α/bias。

---

## 3. 核心鐵律 — 每個 Phase 必須走完的流程

**Phase 定義**：任何「實作 + 評估」的工作單元。例：「加 player ELO 特徵」、「實作 length-stratified training」、「跑 SSL pretraining」皆是一個 phase。Bug fix / 路徑修正 / 重跑既有 pipeline 不算 phase。

每個 phase **必須** 依序走完以下，缺一不可：

```
1. Plan          — 方案、mechanism、評估準則、回退策略
2. Debate #1     — 跑 `adversarial-debate` skill 攻 PLAN（≥2 視角）
3. Plan 修正     — 根據 debate 結論調整
4. Implement     — 寫程式
5. Code Review   — 跑 `code-reviewer` skill（見 §3.2）
6. Debate #2     — 跑 `adversarial-debate` skill 攻 IMPLEMENTATION + 結果（找 leakage / OOF overfit / distribution shift 漏洞）
7. 驗證決策      — 用三指標判斷（見 §6）
8. 記錄          — 寫進 `docs/V5Z_FINAL_REPORT.md` §2 (失敗) 或 §1 (成功 ship)，並更新 memory
```

**鐵律不可違背**：
- 失敗的實驗也要走完，並寫 retrospective
- 不准合併 Debate #1 與 #2（兩次的攻擊面不同：plan 階段攻假設、result 階段攻證據）
- 不准跳 code-review，即使「只是改幾行」
- 不准用「結果是 marginal 所以省略 debate」當理由

### 3.1 Debate 怎麼做

每次 debate 至少 2 個獨立視角，**必須真的對抗**（不是禮貌性 confirm）。

**Token 規範（強制執行）**：呼叫 `adversarial-debate` skill 時：
- **Skill prompt（傳給 skill 的 ARGUMENTS）一律用英文**
- **Skill 內部產出 audit / attacks / verdict 一律用英文**
- **❗ Skill 跑完後，回給 user 的訊息（包含 verdict 摘要、表格、action items）必須翻譯回中文**
- **不准把英文 debate output 直接貼給 user**。即使 verdict 內容很完整，也要重寫成中文 summary
- 同樣規則套用 `time-series-analyst` 等 token-heavy skill：內部英文、回 user 中文

違反例：跑完 debate 後直接 paste 英文 attacks/verdict — 等於沒翻譯，user 體驗破碎。

**首選方式：跑 `adversarial-debate` skill**（專為此設計，會 spawn 對抗 agent 攻擊既有結論）。Debate #1 在 Plan 寫完後跑、Debate #2 在 implementation + 結果出來後跑。

備援形式（`adversarial-debate` 不適用時才用）：
- **Spawn 兩個 Agent 並行**：用 `Agent` tool 開兩個，一個持「方案會成功」，一個持「方案會失敗 / 有隱藏 leakage / 是 14 個失敗變體之一的偽裝」。各自獨立分析後我做仲裁。
- **內部雙人格**（僅限小 phase）：明確標註 `[Pro]` 與 `[Con]` 兩段論述各 ≥3 點，不能 hand-wave。

每次 debate 結束必須產出：
- 最強的 con 論點是什麼
- 是否仍要進行（若是，如何 mitigate con）
- 若否，回退到什麼

### 3.2 Code Review 怎麼做

**首選方式：跑 `code-reviewer` skill**（Python 等多語支援、含 best practice / security scan / review checklist）。對本專案要特別 prompt 專注於：
- Leakage（rally-level 常數 vs stroke-level feature 的邊界，特別是 `serverGetPoint`）
- OOF / test 分布對齊（length bucket、class 17/18 處理）
- env var 規範（`V5Z_*` 系列）
- artifact 命名一致性（`v5z_<tag>_s<seed>_*`）

備援：若 `code-reviewer` 不適用（例如非程式碼類改動），可用 **`simplify` skill** 或 spawn `general-purpose` agent 做 adversarial review。

`/security-review` 與 `/review` 不適用本專案（不是 GitHub PR 場景）。

---

## 4. time-series-analyst 何時主動使用

`/time-series-analyst` 是**主動觸發**工具，下列場景**必須**使用：

| 場景 | 為什麼 |
|---|---|
| 評估新方向 ROI / 排序候選方案 | TS 框架可分辨「對症 vs 繞過 distribution shift」|
| 設計新的 CV split / fold 策略 | Time-series CV 與 standard KFold 的差異 |
| 校準策略選擇（importance weighting / length stratification / domain adaptation）| 屬於 TS 的 covariate shift 處理 |
| 實驗結果判斷是 signal 還 noise | OOF Δ < 0.002 可能是 fold variance |
| 對 last-stroke entropy ceiling 做理論判斷 | 高熵 target 的 forecasting 上限 |

**不需要**用於：純架構選擇、loss function 變體、hyperparameter sweep。

---

## 5. 不要重蹈的覆轍（從 V5Z_FINAL_REPORT §2 萃取）

動工前先確認新提案 **不是** 下列 14 個已驗證失敗變體之一或其輕微變化：

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

**任何新方向動工前必須答**：「這跟上述 14 個有什麼根本不同？」答不出來 → 不做。

---

## 6. 評估準則（所有 phase 通用）

**三個指標必須同時看**：

1. **Full OOF CV** — 傳統 macro-F1（α-blended，post-bias）
2. **Test-weighted OOF** — 用 test L 分布 (32% L=1, 24% L=2, ...) 加權的 OOF — **主決策指標**，最接近 LB
3. **F1 by length bucket** — k=1, 2, 3, 4-5, 6-10, 11+ 各自的 F1_action / F1_point

**決策規則**：
- Test-weighted OOF 退步 > 0.002 → abandon，不要 hyperparam 救
- Full OOF 升但 test-weighted OOF 沒升 → 是 long-context overfit，不 ship
- Length bucket 必須拆，整體分數可能掩蓋特定 bucket 的退步

**LB 提交預算**：每個方向 ≤ 2 次。LB 是 1236 rallies，sample variance ±0.008 是常態。

---

## 7. 系統運作 (詳見 `docs/SYSTEM_GUIDE.md`)

### 7.1 訓練單 seed
```bash
V5Z_SEED=42 V5Z_TAG=v5z_s42 python3 src/train/train_v5z.py
```

### 7.2 校準 + ship
```bash
V5Z_SEEDS="42,1337" python3 src/train/advcal_v5z.py
```

### 7.3 切換架構（多為退步驗證用）
```bash
V5Z_ARCH=gru|transformer
V5Z_FOCAL=2.0
V5Z_SOFTF1=0.3
V5Z_HIDDEN=256
V5Z_LAYERS=3
V5Z_EPOCHS=50
```

完整 env vars 清單見 `docs/V5Z_FINAL_REPORT.md` §5.2。

---

## 8. 檔案地圖

```
data/                           — train.csv / test.csv / sample_submission.csv (2026-04-30 新版)
baseline code/baseline code.py  — 主辦提供 baseline，floor CV 0.4143
src/train/train_v5z.py          — 主訓練（LSTM + LGB OOF）
src/train/advcal_v5z.py         — 校準 + ship submission
src/predict/                    — 老版 predict / diagnose 腳本
artifacts/v5z_*_oof.npz / _test.npz  — 核心數據
artifacts/_old_v7_stale/        — 前版資料訓練的 artifacts，已歸檔
models/v5z/                     — LSTM 權重
submissions/                    — 所有 ship 過的 CSV
logs/                           — 所有訓練 log
docs/SYSTEM_GUIDE.md            — 新手導讀
docs/V5Z_FINAL_REPORT.md        — 最終 ship + 失敗清單 + 未實作方向
docs/TEST_DISTRIBUTION_ANALYSIS.md — OOF/LB gap 診斷
docs/rule.md                    — 主辦官方任務說明
docs/V6_PLAN.md / V7_PLAN.md    — stale，已被 V5Z 取代
```

---

## 9. Memory 與 docs 的分工

- **`docs/V5Z_FINAL_REPORT.md`**：權威 retrospective，所有 phase 結束後寫這裡
- **`docs/TEST_DISTRIBUTION_ANALYSIS.md`**：診斷與分布分析的 living doc
- **Memory (`MEMORY.md`)**：跨 session 的 ship 狀態、user style、資料更新事件
- **新 phase 開場**：先讀 memory + V5Z_FINAL_REPORT + TEST_DISTRIBUTION_ANALYSIS 三件，不要重新從頭探索

---

## 10. 寫程式的小規範

- 不在 train/test 共用 player feature 計算前做 leave-one-out / fold-wise 隔離 → 直接洩漏
- artifact 命名：`v5z_<tag>_s<seed>_{oof,test}.npz` / `pred_v5z_<tag>_s<seed>_ens_{action,point}.npy`
- log 命名：`logs/<train_or_advcal>_v5z_<tag>_s<seed>.log`
- Class 17/18 在 next-stroke 永不出現，校準時不要 `labels=range(N)`
- 任何用到 `serverGetPoint` 為輸入的 LGB feature 必須 abort
- 改 `train_v5z.py` 前先看是不是 env var 能解的（90% 的變體都是）

---

## 11. 何時 escalate 給 user

- 要消耗 LB 提交配額前
- 要刪 artifacts/ 或 models/ 任何檔案前
- 要改 ship CSV 前
- 任何 phase 走到 Debate #2 後仍有未解 con 論點
- 發現 docs 之間有矛盾
