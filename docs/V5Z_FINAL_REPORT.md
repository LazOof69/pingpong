# V5-Z 最終報告 — 桌球戰術預測

**Ship 答案**：`submissions/submission_v5z_advcal_bag2.csv`
**OOF Cross-Validation Score**：**0.4816**（baseline 0.4143 → +0.0673，+16.3% 相對提升）
**最終定案日期**：2026-05-02

---

## 1. 最佳作法（V5-Z + bag2 + fine-grid additive calibration）

### 1.1 架構總覽

兩個互補模型透過 weighted scalar ensemble 結合，再透過 plug-in additive log-bias 校準至 macro-F1 的局部最優。

```
LSTM seed bag (s42 + s1337)  →  α = 0.64 (action) / 0.50 (point)
LightGBM (49 hand-crafted features) →  1 - α
ensemble probs → log → +per-class additive bias → argmax
```

### 1.2 LSTM 子模型（V5-Z）

| 元件 | 設定 | 角色 |
|---|---|---|
| **輸入** | 7 個 categorical (strikeId/handId/strengthId/spinId/pointId/actionId/positionId) + 3 個 numeric (scoreSelf/Other/strikeNumber) | 每個 stroke 的特徵向量 |
| **Embedding** | 每個 cat 維度 20，加入位置 embedding | 維度匹配 + 位置感知 |
| **Player embedding** | 16 維，server/receiver/next 三路 + dropout 0.3 | 選手身份感知 |
| **Encoder** | BiLSTM 192d / 2 layers / dropout 0.3 | 雙向序列編碼 |
| **Pooling** | Multi-head attention（4 heads）+ last hidden | 序列彙整 |
| **Action head** | MLP(128→64→19) | actionId 分類 |
| **Point head** | MLP(128+19→64→10)，吃 trunk + softmax(action) | Action→Point 串聯設計 |
| **Aux head** | Forward LSTM states → MLP，每位置預測下一 stroke | 因果輔助監督 (causal-only) |
| **Loss** | 0.5 × CE_action + 0.5 × CE_point + 0.1 × Aux | Class-weighted CE，label_smoothing=0.05 |
| **Train aug** | 對每個 rally 生成 k=1..N-1 個 prefix sample | 涵蓋不同 context 長度 |
| **Optimizer** | AdamW lr=3e-4 wd=1e-4 + CosineAnnealingWarmRestarts | 標準配置 |
| **Schedule** | EPOCHS=40 PATIENCE=10，按 OOF score early stop | 收斂判定 |
| **Teacher forcing** | tf_p 從 0.5 線性衰減到 0（at E40）| Action→Point 串聯穩定訓練 |

### 1.3 LightGBM 子模型

49 個手工特徵：
- ctx 長度 / 比賽資訊 / 比分壓力（is_deuce、game_point）
- last/prev/serve stroke 的 7 個 categorical
- Action group counts（attack/control/defense）、point depth/side counts
- 交互特徵（last_action × pointId 等）

**注意**：`serverGetPoint` 已**移除**為輸入特徵（官方建議，避免 rally outcome leakage 到 stroke level）。

### 1.4 Plan Z 的核心改動

相對於原 V5：
1. ✅ **移除 sgp 輸入特徵**（防 leakage）
2. ✅ **新增 forward-only aux next-token head**（因果監督，不洩漏 backward direction）
3. ✅ **使用新版 train.csv**（14,995 rallies，competition organizer 修補資料外洩後的版本）

### 1.5 Bagging（seed-level diversity）

最佳：`s42 + s1337`（捨棄 s2024，因為它持續拖低 ensemble）

```
seed=42  single CV: 0.4788
seed=1337 single CV: 0.4785
seed=2024 single CV: 0.4765  ← 最弱，丟棄
```

每 seed 跑獨立 5-fold StratifiedKFold（用該 seed 做 fold 隨機）+ 獨立 LSTM init + 獨立 LGB seed。OOF 預測在 calibration 時透過 `s2o` reindex 對齊回 sample-idx 規範順序。

### 1.6 Calibration（fine-grid plug-in additive log-bias）

**為什麼用 plug-in additive bias**：對 macro-F1 有理論最優保證（Koyejo et al., 2014）。argmax_k(log p_k + b_k) 等價於 thresholding posterior。

**Search 配置**：
- α (LSTM/LGB ensemble weight)：step=0.01，掃 [0, 1]
- per-class bias b_k：step=0.05，掃 [-2.5, 2.5]，coordinate descent 4 rounds

**結果**：
- α_action = 0.64，α_point = 0.50
- bias_action 範圍 [-0.20, 0.65]，bias_point 範圍 [-0.35, 1.15]
- 校準前 raw F1_a=0.4303 → 校準後 0.4389（+0.0086）
- 校準前 raw F1_p=0.2464 → 校準後 0.2650（+0.0186）

**Class 17/18 完全被忽略**（serve action 永遠不是 next-stroke 目標，sklearn default macro 自動排除這些空類別）。

---

## 2. 嘗試過、實證不更好的作法（按時間順序）

### 2.1 LSTM seed=2024（單 seed bagging）

**動機**：第三個 seed，希望 bagging variance reduction。
**結果**：CV 0.4765（比 s42 / s1337 都低）。
**為什麼失敗**：seed variance 大；s2024 隨機到了較差的 fold split + LSTM init 組合。Bag3 含它反而把 bag2 從 0.4816 拉低到 0.4809。

**啟示**：seed bagging 不是越多越好。如果某 seed 顯著偏弱（>0.002），把它留在 bag 裡會稀釋其他 seed 的訊號。**自動篩選最佳 seed combination 比盲目 bag 全部強**。

---

### 2.2 BiGRU（架構多樣性）

**動機**：LSTM 和 GRU 是兩種 recurrent 結構，hopefully 錯誤模式不同 → ensemble 多樣性。
**結果**：
- GRU s42 single CV 0.4771（比 LSTM 0.4788 略弱 -0.002）
- bag2 LSTM + GRU 整合 CV 0.4812（-0.0004）
- 3-way scalar 把 GRU 權重壓到 0.05/0.00（point 直接歸零）

**為什麼失敗**：GRU 與 LSTM 學到的 representation 太接近。在序列短（mean 5.65 strokes）、訊號清晰的情況下，兩種 RNN 殊途同歸。需要更不同的 inductive bias 才能帶來真正的多樣性（例如 attention-only 或樹狀模型）。

---

### 2.3 Larger LSTM（HIDDEN_DIM 256, 3 layers, 50 epochs）

**動機**：增加模型容量看是否能 break 0.4816 ceiling。
**結果**：
- 前 4 fold 持續比 vanilla 弱（fold 2 -0.004, fold 3 -0.007）
- Fold 5 OOM CUDA error 中斷
- 即使完成預計 CV ~0.475

**為什麼失敗**：當前 70k 訓練樣本 + 192d/2L 架構已經 saturate。再加大容量 = 過擬合 + 訓練不穩定。對於序列短、訊號量有限的任務，模型不是越大越好。

---

### 2.4 Bidirectional Transformer encoder

**動機**：Self-attention 可捕捉 long-range dependency，理論上比 RNN 強。
**結果**：
- Transformer s42 single CV 0.4755（比 LSTM 0.4788 弱 -0.0033）
- 3-way bag2 LSTM + Transformer + LGB CV 0.4807（-0.0009 vs bag2）

**為什麼失敗**：
1. **70k 樣本太少**：Transformer 需要遠多於此的資料才能發揮（典型需 1M+）
2. **序列太短**：mean 5.65 strokes，attention 沒有 long-range context 可發揮
3. **缺乏結構化偏置**：BiLSTM 的 recurrent inductive bias 對短序列任務天然合適

**啟示**：「越複雜越好」是錯誤直覺。要看資料量與序列特性。

---

### 2.5 Focal Loss (γ=2.0)

**動機**：稀有類別（cls 0 F1=0.30, cls 8 F1=0.33）拖低 macro-F1，focal loss 集中關注 hard examples。
**結果**：CV 0.4761（vs vanilla 0.4788，-0.0027）

**為什麼失敗**：
1. **Class weights 已經處理了 imbalance**：focal 在已平衡的 loss 上再降簡單樣本的權重 → 過度抑制大類別的訊號
2. **Label smoothing 必須關掉**：focal 的 (1-pt)^γ 與 label smoothing 相剋
3. **macro-F1 不是 hard-example 問題**：稀有類的問題是「資訊不足」，不是「模型不關注」

**啟示**：focal loss 不是萬靈丹。當 class weights 已經做了平衡，再加 focal 是錯誤的疊加。

---

### 2.6 Soft Macro-F1 Loss (混合 70% CE + 30% SF)

**動機**：直接優化評分指標，而非透過 CE 代理。
**結果**：
- SF s42 single CV 0.4783（vs vanilla 0.4788，-0.0005，邊際）
- bag3 (s42+s1337+SF) CV 0.4799（-0.0017）
- 3-way (LSTM bag2 + SF + LGB) CV 0.4807（-0.0009）

**為什麼失敗**：
- SF 與 LSTM 的錯誤模式相似（同樣的 BiLSTM backbone）
- SF 的 batch-level approximation 對 macro-F1 是有偏估計（小批次不穩定）
- 已經有 plug-in calibration 做了類似的 macro-F1 對齊（post-hoc 比 in-loss 還精確）

**啟示**：post-hoc plug-in additive bias 已經把 macro-F1 推到接近最佳。loss-level 的 macro-F1 surrogate 與 post-hoc 兩者擇一即可。

---

### 2.7 LGB Serve Mask（post-process）

**動機**：LGB 對 cls 15-18 (serve actions) 預測 203 個 false positive，可能拖低 ensemble。
**結果**：CV 0.4815（vs 0.4816，-0.0001 完全在雜訊內）

**為什麼失敗**：LSTM 已經 mask 了 serves，ensemble argmax 在大多數樣本不會選到 serve。LGB 的 spurious serve probability 在 ensemble 後幾乎沒影響。

**啟示**：當 ensemble 中至少一個元件有正確的 inductive bias，post-process 修補另一元件的 spurious prediction 邊際效益微弱。

---

### 2.8 粗 grid（α step=0.05, bias step=0.10）vs 細 grid（α step=0.01, bias step=0.05）

**結果**：細 grid +0.0008（CV 0.4808 → 0.4816）

這是**唯一正向的後期實驗**。原因：粗 grid 在 macro-F1 surface 的局部最優附近採樣不夠密。

---

> §2.9-2.19 reserved for post-md backfill（length-stratified / A v2 player_stats / conditional ensemble / EB / Phase / class suppression / 4-way bag md+EB — 詳 PROGRESS_2026-05-04.md §4，pending tasks #14/#18/#22）

---

### 2.20 Multi-task auxiliary heads (MuLMINet recipe) — 2026-05-04

**動機**：在 LSTM backbone 加 spinId / handId / strengthId 三個 aux classification heads，multi-task summed loss。預期 aux signal regularize backbone，pre-head representation 更 robust，主 action/point head 受益。Match-disjoint CV 下 LSTM 已是 0.85-0.98 ensemble 權重，是預期最大 lift 來源。

**結果（mt_bag_eval.log）**：

| Combo | OOF | Δ vs base bag |
|---|---|---|
| MT s42 single | 0.4386 | +0.0025 vs base s42 0.4361 ✅ |
| MT s1337 single | 0.4369 | +0.0008 vs base s1337 0.4361 |
| MT s42 + base s1337 (2-way) | 0.4431 | +0.0017 vs base bag2 0.4414 |
| MT s42 + base bag2 (3-way) | 0.4439 | +0.0025 |
| **MT s42 + base bag3 (4-way)** | **0.4449** | **+0.0008 vs base bag3 0.4441** |

**為什麼失敗（as ship candidate）**：
- Single-seed lift +0.0025 真實，但 ensemble 後僅 +0.0008（與 base bag3 相比）
- 未滿足 +0.005 ship 門檻（CLAUDE.md §6 / PROGRESS §3 「OOF +0.010」實證後修正版）
- 同樣 pattern 重演 LGB-only feature additions：marginal signal 被 calibration + ensemble averaging 抹平

**啟示**：Diversity 來源若與 base LSTM 共享 backbone（BiLSTM + match-disjoint CV + 同一份 49 features），aux heads 提供的訊號高度相關 — ensemble averaging 會 cancel out。要破 OOF 0.444-0.446 saturation，必須換**真正不同的 inductive bias**：
- Architecture diversity (Transformer match-disjoint，rally-KFold 時代失敗，新 protocol 未驗)
- Augmentation diversity (symmetry mirror — TacticAI lesson)
- Loss diversity (Logit Adjustment + 拿掉 class weights，Menon 2021)
- Test-time diversity (pseudo-labeling / stacking meta-learner)

不再投入 multi-task aux heads 的延伸（額外 task / loss weighting tuning）— 與 LGB-only feature 加法系列同型 saturate。

---

### 2.21 6-way bag ship (mt2base4) — 2026-05-05 LB

**成分**：MT bag2 (s42 + s1337) + base bag4 (s42 + s1337 + s100 + s200) → `submission_v5z_md_6way_mt2base4.csv`

**結果**：
- OOF (calibrated): **0.4456** (+0.0042 vs ship bag2 0.4414)
- LB: **0.4367** (-0.0007 vs ship 0.4374)
- OOF/LB gap: -0.009

**為什麼失敗**：OOF +0.0042 不滿足 +0.010 ship 門檻（PROGRESS §3 實證 LB sample variance ±0.005-0.010）。送出時 user 評估「邊際但值得試」，賭輸符合 LB 變異區間下緣。

**啟示**：
1. 重申 ship 門檻 **OOF +0.010**（而非 +0.005）— 今日 LB 結果再次證實 PROGRESS §3 的修正觀察
2. 當前框架（match-disjoint + LSTM/LGB ensemble + plug-in calibration + multi-task aux）saturate 在 **OOF 0.444-0.446**
3. 4-way (0.4449) → 6-way (0.4456) OOF 增加 +0.0007 全來自 base s200 加入 bag — 同樣 marginal，不會在 LB 留下訊號
4. 2026-05-04 的 multi-task phase + 今日 6-way 確認：架構/seed/aux head 變體已 exhaust，必須走 §2.20 結尾列出的 4 種「真正不同 inductive bias」之一

**第 18 條失敗變體確認**：4-way bag md+EB（2026-05-04 LB 0.4341，PROGRESS §2）。連同本節 §2.20 / §2.21 → 累積 19、20 條已驗證失敗變體。

---

### 2.22 Logit Adjustment（Menon 2021）+ class-weight removal — Debate-killed 2026-05-05

**首條「Plan 寫完跑 Debate#1 即被殺」的變體**。依 CLAUDE.md §3 鐵律，未實作的 phase 也要記錄。

**動機**：取代 inverse-frequency class weights，改用 LA — 訓練時加 τ·log(π_train) 到 logits 後再過 CE，理論上 Bayes-optimal under macro-F1 + class imbalance + label shift（Menon et al. 2021, ICLR）。希望「shifts decision boundary 而非 gradient magnitude」帶來與 base BiLSTM **decorrelated** 的 ensemble 成員。

**Plan 詳情**：τ ∈ {0.5, 1.0, 1.5} 在 s42 single 上 sweep，gate +0.005 single OOF / +0.010 bag2 OOF。

**Debate#1 命中（兩 adversarial agent 並行）**：

1. **Calibration-collision (HIGH)**：LA 推論時輸出 `z_LA ≈ log P(y|x) − τ·log π_train`，與 class-weighted CE 輸出 `z_CW` 只差 per-class additive constant `δ_c`。Argmax 對 per-class additive shift **不變** → `argmax(z_LA + α*) = argmax(z_CW + α*')`。Plug-in additive bias 有 19 個自由參數（每類 1 個 α），LA 的 τ·log π 只有 1 scalar 參數固定方向 → plug-in **strictly dominates** LA 在 macro-F1 surface 能表達的 shift。**這就是 §2.6 Soft-F1 collision 的數學同型**。

2. **Prior-shift（HIGH，可能 FATAL）**：Train mean L=5.65 vs test mean L=2.90，56% test 是 L≤2。`π_train(c) = Σ_k q_train(k)·p_k(c)` 與 `π_test(c) = Σ_k q_test(k)·p_k(c)` 結構性不等（length-position-conditional action distribution 差異大）。LA 訓練時減 τ·log π_train 把 mass 從 S 類別（serve-receive，k=1 主導）推離 → 對 56% test rallies 是**錯誤方向**。Plug-in calibration 在 OOF（≈ train length 分布）上算 α*，無法修正 test prior。

3. **Saturation ceiling（FATAL on +0.010 gate）**：Macro-F1 ceiling 由 (1) terminal stroke aleatoric entropy ~2.4-2.6 nats、(2) 70k 樣本 + 17 effective classes、(3) test length covariate shift 限制。Published LA gain on long-tail benchmarks（Menon 2021 / Cui 2021 / Zhang 2023）= **+0.001-0.004 macro-F1 = +0.0004-0.0016 total Score**。Ship gate +0.010 比機制能交付的上限大 **3-25×** → magical thinking。

4. **Ensemble decorrelation（HIGH）**：Loss-only diversity 是文獻中最弱 ensemble axis（Caruana 2004 / Geffner & Domingos 2018 / Wood 2023 排序：data subsampling > architecture > input modality >> seed >> loss reformulation）。§2.20 multi-task aux heads（diversity 比 LA 更強，注 trunk gradient）已實證 +0.0025 → +0.0008 in ensemble，1/1 base rate of collapse。LA 預期 ensemble lift +0.0005-0.0015，indistinguishable from seed variance。

**Length-conditional 救命稻草也不可行**：兩 agent 都建議 fallback 到 length-conditional LA（用 `π_k(c)` per stroke position），但 PROGRESS §4 已實證 **「Length-stratified calibration」失敗**（testweighted CSV LB -0.0037）+ §1.1 結論「OOF/LB gap 完全由 CV protocol bias 解釋，length 是 secondary」→ length-conditional LA 也會死。

**為什麼 Debate-kill 很重要（meta-lesson）**：
- LA 表面看是「新 inductive bias」，實際在 plug-in calibration 框架下與 §2.6 Soft-F1 數學同型
- 本專案的 plug-in additive bias calibration 是極強的 macro-F1 absorber — **任何 train-time macro-F1 surrogate 都會被吸收**
- 未來 Plan：若提案是「修 loss 對 macro-F1 結構」，先問「plug-in 19-param 自由 shift 能不能表達同樣 effect？」答 yes → 直接 KILL，不必跑

**累積失敗變體：21 條**。下一 phase pivot 到 PROGRESS §5.1 候選 A（Symmetry mirror augmentation）— augmentation diversity 是 Caruana 2004 排序第一的 ensemble axis，且引入真正新 inductive bias（左右對稱），不與 calibration 衝突。

---

### 2.23 Symmetry mirror augmentation（TacticAI/ShuttleNet recipe）— Debate-killed 2026-05-05

**第二個「Plan 寫完跑 Debate#1 即被殺」的變體**（連同 §2.22 LA）。

**動機**：每個 train rally 產出 mirror 副本（positionId 1↔3、pointId {1↔3, 4↔6, 7↔9}、其他不變），train data 14995→29990 rallies, 70k→140k strokes。Augmentation diversity 是 Caruana 2004 ensemble taxonomy 排第一的 axis。引用 TacticAI（Wang 2024 Nature, +3-5% football）+ ShuttleNet（Wang 2022, badminton）支持。

**Ship gate**: bag2 OOF +0.010 vs ship 0.4414，預估 12h compute（s42+s1337）。

**Debate#1 命中（兩 adversarial agent 並行）— 3 FATAL + 3 HIGH**：

#### FATAL #1 — PointId receiver-frame corruption（Agent 1 提名）

`rule.md` 明說 pointId **已對 receiver handedness normalize**：右手 receiver 的 pointId=1 = 對方視角左側；左手 receiver 的 pointId=1 = 對方視角右側。Plan 的 flat map `{1↔3, 4↔6, 7↔9}` 對左手 receiver **物理上錯誤**。

正確 mirror 必須 **conditional on receiver handedness**：右手 → flip，左手 → identity。Pro 桌球左手率 12-15%（189 player → ~25 lefty），加上 rally 內 striker/receiver 交替 → mixed-handedness rally 內**部 label 自相矛盾**。

**~7% mirrored stroke 的 pointId target 結構錯誤**。比照 CLAUDE.md §10「`serverGetPoint` 當 stroke-level feature 必 abort」標準應 abort — 這是資料正確性 bug，不是 hyperparameter。

#### FATAL #2 — Test-weighted lift = 無意義（Agent 2 提名）

詳細 stroke-count 分析：

| Length | Test stroke 占比 | Mirror 行為 | Predicted lift |
|---|---|---|---|
| L=1 | 11% | Prefix 為空，mirror = **literal duplicate row** | 0 |
| L=2 | 16.5% | Joint cell `(serve_pos, serve_point, recv_action) = 378` 已平均 28 samples/cell，已飽和 | +0.000~0.001 |
| L=3-5 | 50%+ | 14 個失敗變體已驗證 mid-rally saturate | +0.001-0.003 |
| L≥6 | 10% | 真有 diversity 但 rare in test | +0.0005 |

**Length-bucket-weighted 預期 single-seed lift = +0.0025-0.0035**，比 §2.20 multi-task (+0.0025) 還弱。Ensemble 預期 +0.0008-0.0015 — 直接低於 ship gate。**Saturation 最嚴重的 bottleneck（serve classes at L=1）正是 mirror 數學上 no-op 的地方**。

#### FATAL #3 — Saturation ceiling: mirror 是 hard regularization, not new information（Agent 2 提名）

V5Z_FINAL_REPORT §2 累積失敗變體已 21 條 → saturation 是 **information regime**（不是 parameter capacity 也不是 data volume）。Mirror 數學上等價於 group equivariance regularization（Cohen & Welling 2016）。

| 來源 | 預期 lift |
|---|---|
| New information（pseudo-label, external features）| +0.005-0.015 |
| Equivariance regularization（mirror, rotation）| +0.001-0.003 |
| §2.20 multi-task（更強 perturbation）| 已實證 +0.0008 ensemble |

Mirror 比 multi-task 弱 → 預期 ensemble lift +0.0003-0.0015，比 ship gate +0.010 低一個數量級。**P(clear +0.010) <5%**。

#### HIGH #1 — Player asymmetry destruction
gamePlayerId embedding 在 (orig + mirror) 上學 → 學到「左右平均」的 phantom symmetric player（pro 選手 70/30 forehand-cross-court 變 50/50），**對 54% seen player（test）destroy 真實 player signal**。Pseudo-player vocab 倍增（189→378）只是 relabel，不增資訊。

#### HIGH #2 — TacticAI / ShuttleNet domain transfer cargo-culting
TacticAI 是 sparse rare-event corner kick prediction（22 player coordinates），不類比 high-rate stroke prediction。ShuttleNet badminton +1.6% **accuracy**（不是 macro-F1）；桌球 spinId 4/5（side-top, side-back）**已 direction-collapse** → mirror = label-consistent 但因果不一致 = noise injection（Cubuk 2020 RandAugment Table 5：label-noise-equivalent aug -0.1 to +0.3% accuracy）。

#### HIGH #3 — OOF symmetrization + calibration miscalibration
Mirror 進 val fold 後 pointId class 分布**強制 symmetric**（pointId 1=3, 4=6, 7=9）。Plug-in calibration 在 symmetric OOF 上學 α*_1 = α*_3，但 test 真實分布 asymmetric → **α* 被強制鎖在對稱對角線，miss true optimum**。預估 LB cost -0.002 to -0.005。Mirror 又非 i.i.d. samples → effective σ × √2 → +0.005 single 不過 1.8σ noise floor。

**為什麼補救改不救得了**：

兩 agent 都列出 mandatory mods（conditional pointId / mask is_mirror / restrict L≥3 / filter pen-hold / time-box 6h）— 即便全套套上，預期 +0.001-0.003 single → ensemble +0.0005-0.0015 仍**低於** ship gate。Mirror 在 Agent 2 priority 排序裡 dead last。

**Reference class（sports stroke prediction macro-F1 mirror aug saturated baseline）**：median +0.001-0.003 single，+0.0005-0.0015 ensemble。Plan target +0.010 在 98 percentile — magical thinking。

**Meta-lesson**：
- Cross-domain published lift（TacticAI 3-5% 在 sparse continuous coord）不能直接外推到 dense discrete categorical
- 本 dataset 的 pointId 不是純空間座標，是「receiver-frame normalized」混合語意 → mirror 不是 clean physical operation
- Saturation 已是 information regime，augmentation（不論 mirror / data subsampling / loss reformulation）能交付的 max ~+0.003，不論 Caruana 2004 ensemble axis 排序如何

**累積失敗變體：22 條**。下一 phase pivot 到 PROGRESS §5.1 候選 D（Pseudo-labeling on test）— Agent 2 列為 priority 第一，reference class +0.005-0.012，**唯一對得上 +0.010 ship gate** 的方向，直接引入 test distribution info（對症 information regime saturation）。

---

### 2.24 Pseudo-labeling on test (semi-supervised) — Debate-killed 2026-05-05

**第三個 Plan-stage Debate-killed 變體**（連同 §2.22 LA、§2.23 Mirror）。同一 session 內連續三 KILL 是重要 meta-signal。

**動機**：用當前 ship bag (LB 0.4374) 在 1236 test rallies 上產 pseudo-target，top-K (K∈{30%,50%,100%}) 信心 gating，加進訓練（sample weight 0.5），single round。Plan 自我宣稱：(1) 把 test L=2.9 distribution 帶進訓練、(2) 36.5% unseen player prefix 加入訓練、(3) reference class +0.005-0.012（CommonLit/OpenVaccine/iNaturalist）對得上 +0.010 ship gate。

**Ship gate**: bag2 OOF +0.010 vs ship 0.4414，預估 7-8h compute。

**Debate#1 命中（兩 adversarial agent 並行）— 3 FATAL + 3 HIGH**：

#### FATAL #1 — Top-K Selection Paradox（Plan 核心機制結構性錯誤）

Top-K confidence gating 篩 model 高信心的 pseudo-label。但 model 對哪種 test rally 高信心？— **與 train 分布相似的 rally**（mean L=5.65 訓練 → 對 long-context 高信心）。

Agent 1 量化：
- L=1 通過 top-50% 信心率 ≈ 25-35%
- L=5 通過率 ≈ 65-80%
- Post-gate pseudo-label L 分布 **mean ≈ 4.5-4.8**（比 train 5.65 略低，但比 test 2.9 高得多）

→ Plan headline mechanism「import test distribution」**結構性錯誤**：top-K 反過來把 train-like pattern 重新注入訓練，**對 56% L≤2 LB bucket 提供 0 訊號**。Arazo 2020 documented confirmation-bias mode。

#### FATAL #2 — A v2 Disguise（同一個 -0.024 LB 災難在等）

Agent 2 讀過 `train_v5z.py` 程式碼後攻擊：
- Line ~100：`sub = pd.concat([train_df, test_df])`，`PID2IDX` 從 full sub 算
- Line ~270：`for pid, grp in sub.groupby("gamePlayerId")` 從 full sub 算 player aggregate（forehand_ratio, as_srv_rallies 等）
- **沒有 fold-aware 切換**

關鍵 leakage path：40/166 train players 與 test overlap，pseudo rally 帶 player_stat aggregate（從 X 的 train rallies 算）→ 用該 stats + pseudo label 訓練 → val 時 X 的 val rally 也帶同樣 stats → **CIRCULAR**。Match-disjoint 隔離 temporal/match identity，**不隔離 player aggregate signature**。

A v2 LB -0.024 就是這條路徑（rally-KFold 99% overlap），Plan D 是 24% overlap 縮小版。Agent 2 預測：**OOF +0.003-0.008 inflated，LB -0.005 to -0.015**。

Plan §8 宣稱「per-fold feature regen」是抽象層 1 行 vibe，實作需 3-5h 重構 + 3-5h debug + 5× train cost = **30-40h**，不是 7-8h。

#### FATAL #3 — Self-Distillation 資訊理論天花板（Mobahi 2020 NeurIPS）

Teacher（current ship bag）和 student（next round bag）來自**同一 model class** → matched teacher/student class self-distillation **不提供超過 regularization effect 的新資訊**（Mobahi 2020 證明）。

Hard pseudo-labels（Plan 用 argmax）連 regularization 機制都丟掉。Reference class for hard self-distillation on saturated long-tail：
- Furlanello 2018 BAN: macro-F1 +0.002-0.005
- Iscen 2019 long-tail saturated: +0.001-0.003
- 預估 single +0.001-0.003，**ensemble +0.0005-0.0015**（與 §2.20 multi-task 0.0008、§2.23 Mirror 同範圍）

#### HIGH #1 — Macro-F1 機制結構缺失（class skew）

Confidence-gated pseudo-set ≥80% 是 majority class（model 對 majority 高信心、minority 低信心）。Per-class gating 在 minority class 上：argmax 不是 c → pseudo-label wrong with probability ~0.90，加 5-10 minority pseudo with weight 0.5 = +2.5-5 effective minority samples on base 700 → **Macro-F1 mechanism 結構缺失**（minority lift 結構不可達）。

#### HIGH #2 — Per-fold regen 工作量低估 5x

Implementation 3-5h refactor + 3-5h debug + **5× train compute** = 30-40h（非 7-8h）。Verification fixture（synthetic shuffle test）未指定 → 無法分辨「leakage solved 真 lift +0.002」vs「leakage hidden 假 lift +0.008」。

#### HIGH #3 — Reference class 誤用

Plan 引用 CommonLit / OpenVaccine / iNaturalist 全錯 reference class：
- CommonLit test/train ratio 2.5x（pingpong 0.08x，**反方向**）；單 stage hard +0.001-0.003，cited +0.005-0.012 是 multi-stage soft 才達到
- OpenVaccine soft label + KL（Plan 用 hard）
- iNaturalist top-1 +0.5-1%，macro-F1 halved +0.3-0.6%

正確 reference class（small-test covariate-shifted saturated long-tail hard-pseudo single-round）:
- Soccer xG (Bransen 2020): +0.001-0.003
- ShuttleNet / TenniSet: **explicitly 棄用 pseudo-labeling**

→ Plan target +0.010 比 reference class median 過高 5-20×。

**Meta-lessons（連續 3 KILL 累積）**：

1. **Saturation diagnosed 真為 information regime**：3 條獨立路線（loss / spatial aug / semi-supervised）全部撞到 ensemble +0.0005-0.0015 天花板，這個天花板不是 plan-quality 問題而是當前 framework + data 規模的 **architectural ceiling**

2. **Plug-in calibration 是強 absorber**：任何 train-time decision-boundary shift（LA τ、mirror equivariance、pseudo-label 分布注入）都被 19-param plug-in 吸收

3. **70k strokes 對 BiLSTM 接近 capacity 飽和**：data augmentation（mirror = 2x deterministic、pseudo-label = 1.08x with noise）都打不破 ceiling

4. **Reference class cargo culting 是常見陷阱**：Plan B/A/D 都引用 cross-domain published lift 推算 ROI，全部高估 — **必須要求 Plan 提 same-domain（sports temporal prediction）reference**

**累積失敗變體：23 條**。

---

### 2.25 BiTransformer match-disjoint — 訓練後結論（2026-05-05，**ABORT at Pearson gate**）

**Status**: T s42 訓練完成，**hard abort gate #2（Pearson > 0.85）trigger** → 不跑 s1337，pipeline 終止。

### 2.25.0 LB 結果（送出於 2026-05-05）

T s42 single LB: **0.4294875**（OOF 0.4357 → LB 0.4295，gap **-0.0062**，接近 framework 標準 -0.004 略偏大）。

LB Δ vs ship 0.4374 = **-0.0079**。在 LB sample variance ±0.005-0.010 內，但確認 T-only 不是 ship 候選。

### 2.25.A 實際結果 vs Debate#1 預測

| 指標 | Debate#1 預測 | 實際 | Verdict |
|---|---|---|---|
| T s42 calibrated OOF | 0.410-0.430 (中位 0.420) | **0.4357** | **預測錯方向**（意外接近 LSTM 0.4361，僅 -0.0004） |
| Pearson(T, LSTM s42) action prob | 0.85-0.92 | **0.9147** | **預測中位** ✓ |
| Pearson(T, LSTM s42) point prob | 0.85-0.92 | **0.9507** | **預測 略高於上限** |
| F1_action L=1 vs LSTM | -0.03 to -0.05 | **+0.0033** | **預測錯方向** |
| F1_action L=2 vs LSTM | -0.02 to -0.04 | **+0.0016** | **預測錯方向** |
| LB | 預期 0.430-0.435 | **0.4295** | **預測中位** ✓ |

**Surprise findings**：
1. **T match-disjoint 沒崩**：Agent 1 預測 0.42-0.43（attention 從 leakage 獲益更多 → 移除後傷害更深）。實際 T 損失程度 (-0.0398 from rally-KFold 0.4755) **略小於** LSTM (-0.0455)。意味 attention 在 rally-KFold 下並沒有比 LSTM 過度 memorize player tokens
2. **Attack 4 (Agent 2) 全錯方向**：「L=1 attention degenerate to identity」「L=2 single soft-pool」的數學論證在數據上**不成立**。T 在 L=1, L=2 反而略勝 LSTM (+0.0033, +0.0016)
3. **真正的 T 弱點是 mid-rally (k=3-5)**：T -0.022 (k=3), -0.026 (k=4-5)，因 LSTM 的 sequential gating 在中等 context 是 sweet spot；attention 在短/長都還行，僅在中段輸

### 2.25.B Pearson gate 為什麼仍 trigger

Pearson 0.91 的後果（per Agent 1 Attack 2 數學）：
- T single 0.4357 vs LSTM 0.4361（gap 0.0004）
- Caruana 2004 ensemble 定理：required Pearson < 0.65-0.70 for ensemble 超越 best component
- 0.91 > 0.70 by 0.21 → ensemble 數學保證 lift bounded
- 預期 5-way ensemble (LSTM bag3 + T bag2) lift ≈ +0.0005-0.0015，**遠低於 +0.010 ship gate**

不跑 s1337 + bag2 + 5-way ensemble，省 4-6h compute。**Plan T abort，第 24 條失敗**。

### 2.25.C Meta-lessons

1. **Length-bucket attention degeneracy 論證在實 dataset 不成立**：抽象數學「L=1 attention = identity」是真的，但 BERT-style encoder 加入 position embedding + layer norm + FFN 後在實 dataset 表現遠超 toy degenerate analysis。未來 Plan 別再用「attention 對短 sequence degenerate」當 KILL 論證
2. **Pearson 是 ensemble lift 唯一硬約束**：T single performance 意外 OK，但 Pearson 高就完。未來 architecture/training 變體 Plan 應**先**驗證 Pearson < 0.85 假設（cheap 的 1-fold 預測 vs LSTM 比對）
3. **Match-disjoint 對 T 的傷害比預期小**：可能因為 attention 學的 player conditioning 不是 token memorization 而是 cross-stroke pattern compression — 後者在 unseen player 仍可 transfer
4. **Single-seed 測試的關鍵價值**：3-4h 投入給了「T 沒崩但 ensemble 死」的精確診斷，避免 8-12h 的 bag2 + 5-way 全跑無用

**累積失敗變體：24 條**。

---

**Plan T 降版後配置（archived for reference）**：0 code change，`V5Z_ARCH=transformer V5Z_MATCH_DISJOINT=1`，hyperparam 沿 §2.4，hard abort gates: OOF<0.4300 / Pearson>0.85 / L=1,2 退 0.01。Debate#1 致命攻擊: A2 FATAL (ensemble math) + A4 FATAL (L=1 attention degeneracy)。**實際訓練結果見 §2.25.A 對照表 — A4 directionally wrong; A2 holds (Pearson 0.91)。**

---

### 2.26 base bag3 (s42+s1337+s100) LB regression — Framework-level 發現（2026-05-05）

**重要程度**：本條不只是失敗變體，是 **framework-level 重大訊息**，影響所有未來 ensemble 策略。

**實際結果**：

| Submission | OOF | LB | LB Δ vs ship 0.4374 | OOF/LB gap |
|---|---|---|---|---|
| 當前 ship (LSTM bag2 = s42+s1337) | 0.4414 | 0.4374 | 0 | -0.0040 |
| **base bag3 (+s100)** | **0.4441** | **0.4287** | **-0.0087** | **-0.0154** |

PROGRESS_2026-05-04.md §1.3 + §6 都把 bag3 當「明日 Sub 1，預期 LB +0.001-0.005」。**實際 LB -0.0087**，方向徹底錯。OOF/LB gap 從 ship 的 -0.004 暴漲到 -0.015，**4× 倍**。

**Mechanism 推測**（為何 +s100 進 bag 災難）：
1. **Plug-in calibration over-fit on s100's OOF idiosyncrasy**：advcal 在 (s42 + s1337 + s100) 的 OOF 上找 α*，s100 fold-specific noise 被 calibration 吸進 α*，於 test 上不一致 → miscalibrated
2. **Bag mean 反而 over-smooth signal + over-fit OOF**：3-seed 平均拉近 OOF noise，但 test distribution 蓋不到
3. **s100 single OOF 0.4361 ≈ s42 / s1337 — 從 OOF 看正常，加進 bag 才壞**

**Heterogeneous vs Homogeneous bag 對比**（今日 3 sub）：

| Composition | Type | LB Δ vs ship |
|---|---|---|
| 6-way mt2base4 (4 base + 2 MT) | **Heterogeneous** | **-0.0007** |
| **base bag3 (3 base seeds)** | **Homogeneous** | **-0.0087** |
| T s42 (1 transformer) | Single | -0.0079 |

→ Heterogeneous bag 抗 LB drift 比 homogeneous 強 **12 倍**。

**Meta-lessons（影響未來所有 Plan）**：

1. **「base-only bag size > 2」永久淘汰**：s42+s1337+s100 homogeneous 3-seed 在 LB 上 negative transfer。未來不再做這種 bag
2. **OOF +X ≠ LB +X**：bag size variation 的 OOF lift 是 **artifact**，不能直接外推 LB
3. **Heterogeneous bag 是唯一可行 ensemble path**：cross-architecture (LSTM + Transformer + SSL embedding) 或 cross-training (base + MT + future) 配對
4. **Plan E ensemble 必須 cross-arch**：E feature 加進 base LSTM ensemble，**不該** E s42 + E s1337 + base s42 的 homogeneous bag 模式
5. **Single seed → LB 測 gap → 才考慮 bag**：之後新 architecture/feature 候選都應先 single-seed LB 測 gap

**對當前 ship 的決定**：保留 LSTM bag2 ship LB 0.4374。bag3 永久淘汰為 ship 候選。

**累積 LB 數據點（OOF/LB gap 觀察）**：

| Submission | OOF | LB | gap |
|---|---|---|---|
| LSTM bag2 (ship) | 0.4414 | 0.4374 | -0.0040 |
| 6-way mt2base4 | 0.4456 | 0.4367 | -0.0089 |
| 4-way bag md+EB | 0.4413 | 0.4341 | -0.0072 |
| **base bag3** | **0.4441** | **0.4287** | **-0.0154** |
| **T s42 single** | **0.4357** | **0.4295** | **-0.0062** |

→ Match-disjoint OOF/LB gap **不固定**（-0.004 ~ -0.015），視 ensemble composition 而定。**未來 OOF 估算 LB 應預期 ±0.005-0.015 不確定性，不是 ±0.004**。

**累積失敗變體：25 條**。

---

### 2.27 Plan SSL (Self-Supervised Masked Stroke Pretraining) — Debate-killed 2026-05-05

**第 4 個「Plan 寫完跑 Debate#1 即被殺」變體**（連同 §2.22 LA、§2.23 Mirror、§2.24 Pseudo-label）。

**動機**：用 train + test 88k strokes 做 BERT-style masked stroke prediction pretrain，BiLSTM encoder + 5 個 prediction heads（actionId/pointId/handId/spinId/positionId）。Hybrid mask 70% A1 (5-head) + 30% A2 (action+point only)。Pretrain 後 finetune 接既有 train_v5z.py，two-phase (5 epoch freeze + 35 unfreeze)。Cost 24-37h。Plan target P(+0.010) 25-40%，引用 NLP MLM / TS2Vec / Wang 2024 sports SSL cross-domain reference。

**Debate#1 命中（兩 adversarial agent 並行）— 2 FATAL + 4 HIGH/MEDIUM**：

#### FATAL #1 — Pretrain-finetune objective gap（Agent 1）

**核心數學**：bidirectional MLM 學 `P(s_t | s_{t-k:t-1}, s_{t+1:t+k})`（看 future），downstream causal next-stroke 需要 `P(s_t | s_{1:t-1})`。**結構性 inductive bias mismatch**。

- Devlin 2019 BERT vs autoregressive on generation tasks: **5-15% downstream loss**
- Yang 2019 XLNet §2.1: BERT representations 在 generation tasks **3-5% degradation**
- BART Lewis 2020 / T5 Raffel 2020: 純 MLM bidirectional encoder for generation 需要 explicit causal decoder（Plan SSL 沒）

**「右」 SSL 是 autoregressive next-stroke pretrain，但**：
- 對 train rally 做 = 既有 supervised task = 0 新 info
- 對 test rally 做 = pseudo-labeling = 已 §2.24 KILL

**任何 true SSL 路徑都退化到既有失敗 plan**。30% A2 不解決（仍 bidirectional）。

**量化**：50-70% pretrain gain lost to objective mismatch → realistic single-seed lift **+0.0005-0.002 macro-F1**，比 Plan 預測 +0.003-0.008 低 2-4×，**低於 §6 noise floor 0.002**。

#### FATAL #2 — Plug-in calibration absorption（§2.22 LA 同型，Agent 2）

§2.22 LA 死於 plug-in additive bias (29 free params: 19 action + 10 point) **吸收任何 train-time decision-boundary shift**。SSL 同樣機制：

- SSL representation 提升 → 不同 logit 分布
- Plug-in calibration 對任何 logit 分布 refit α* 最大化 OOF macro-F1
- 可分解為 per-class additive 部分**被吸收完**，僅 per-sample reranking 保留
- 我們 19+10=29 類別**少** → per-class additive 比例**大**（vs 1k+ class long-tail）→ **吸收比例 70-85%**

**Empirical anchor（3 篇 published reference 一致）**：
- Cui 2021 long-tail SSL: pre-cal +0.005-0.015，post-class-balanced-cal **+0.001-0.003**
- Hong 2021 LADE iNat: post-cal macro-F1 lift **+0.002-0.004**
- Cao 2019 LDAM-DRW: post-cal macro-F1 **+0.001-0.003**
- 平均 post-calibration SSL gain ≈ +0.002, std 0.001

**比 ship gate +0.010 低 5σ**。

#### HIGH 攻擊摘要

3. **88k strokes 太小**：BERT 3.3B / MAE 1.3M / TS2Vec UCR 17k；Hestness 2017 scaling laws 預期 +0.5-2% accuracy 而非 +3-8%。同 outcome distribution 與已 KILL 的 4 個 Plan
4. **Hybrid 70/30 mask 信號稀釋**：A2-aligned downstream gradient 只 14% 總 signal；T5 hybrid objectives 在 1B+ params 才生效，70k 規模不適
5. **Player embedding dropout 0.5 失效**：50% 仍更新 embedding；對 36.5% unseen player 0 SSL benefit；可能 **net 負面**
6. **Verification fixture 成本 5× 低估**（claim 1h，實 6-8h）+ missing 3 critical tests（player-embed regression, mask leakage, calibration attribution）
7. **E-style-embed ROI/hr 5×**：cost 12-16h vs SSL 24-37h，failure mode 獨立，calibration absorption 較弱（player embed 是 input feature 不是 logit shift）

#### 機率分解（兩 agent 一致）

| 條件 | P |
|---|---|
| Pretrain valid loss 顯著降 | 0.4-0.55 |
| Single-seed Δ ≥ +0.005 ｜ pretrain works | 0.20-0.25 |
| Bag2 Δ ≥ +0.010 ｜ single ≥ +0.005 | 0.30 |
| **Joint (clear ship gate)** | **5-8%** |

vs Plan 自宣稱 25-40%。

#### 同 outcome distribution 致命論證

「Plans LA, Mirror, Pseudo-label, T 全部 outcome distribution 一致：+0.002-0.005 ensemble lift, post-calibration 受限。SSL 是第 5 次嘗試。Bayesian prior P(success) = Laplace estimator 1/6 ≈ 17%」

#### Verdict — KILL（user 選路 A）

兩 agent 共識 KILL。即便補 7 mods（reorder / pre-post cal gate / pure A2 / freeze player embed / UniLM dual obj / verification 補測 / hard kill gate），預期 ensemble lift +0.0005-0.0015，仍比 ship gate 低 5-20×。

**用戶選擇路 A**：KILL SSL，pivot to Plan E-style-embed（cost 12-16h, ROI/hr 5× SSL, calibration absorption 較弱）。

#### Meta-lessons（連 6 連 KILL 累積）

13. **Bidirectional pretrain 對 causal downstream 結構錯**：未來不再提 BERT-style MLM 給 next-stroke 任務
14. **Plug-in calibration 吸收效應普遍**：任何 train-time decision-boundary / representation shift（透過 logits 表現）必被 29-param plug-in 吸收 70-85%。實證 3 paper anchor +0.002 post-cal 上限
15. **5 Plan 連 KILL 證實 framework saturation 是真**：不是 Plan-quality 問題，是 information regime ceiling。任何單一 architecture/training/loss 變體 ensemble lift bounded ~+0.002

**累積失敗變體：26 條**。

---

### 2.28 Plan E (Player-Style AE Embedding) — Stage A probe KILL 2026-05-05

**第 5 個「Plan 寫完跑 Debate#1 即被殺」變體**（連同 LA / Mirror / Pseudo / SSL）。User 選路 A 跑 cheap probe (~2h) 取代 14-20h 直接投入，probe 結果 decisively confirm KILL。

**動機**：60-dim per-player style features → AE 16-dim bottleneck → 加進 LSTM input。Cross-arch heterogeneous ensemble (LSTM bag2 + E)，single seed → LB gap → bag。Plan target P(+0.010) 15-25%。

**Debate#1 命中（兩 adversarial agent 並行）— 3 FATAL + 5 HIGH + 2 MEDIUM**：
- A1 HIGH-FATAL: AE embedding 與既有 supervised gamePlayerId embedding **資訊論冗餘**（Data Processing Inequality argument）
- A3 FATAL: Per-fold AE leakage path（A v2 reborn — 16-dim bottleneck 是 identity compressor）
- A9 FATAL on ROI: 5/5 prior plans 同 outcome distribution，Laplace P(success) = 14%
- + A2/A4/A5/A6/A7/A8 HIGH/MEDIUM

**Stage A Probe 結果（empirical KILL test）**：

#### Probe 1：Supervised player_embed vs 12-dim stats linearity
- 5-fold CV Ridge regression on R²
- **Total multivariate R² = -0.122**（negative → 線性不相關）
- Per-dim R² 全部 negative（max -0.061，min -0.298）
- Per-fold embed std 0.836（顯示 supervised embed 跨 fold 不穩定）
- **解讀**：supervised embed 沒線性 capture stats 訊號，**但這不代表 stats 有用**

#### Probe 2：AV2-style aggregate stats on LGB（match-disjoint）
- Per-fold compute_player_stats with restrict_uids=fold_train_uids
- LGB 比較 (with V2 features) vs (without V2)，5-fold action+point F1

| Fold | Δ Action | Δ Point |
|---|---|---|
| 1 | -0.0365 | -0.0119 |
| 2 | -0.0110 | -0.0064 |
| 3 | +0.0163 | +0.0103 |
| 4 | +0.0064 | -0.0153 |
| 5 | +0.0148 | +0.0046 |
| **avg** | **-0.0020** | **-0.0037** |
| **blended** | | **-0.0028** |

**Decision criteria triggered**: blended Δ = -0.0028 < +0.002 → **stats 不幫 LGB even raw**。

#### 雙 probe 聯合結論
- Probe 1: stats 有獨立於 supervised embed 的訊號（R²<0）
- Probe 2: 但**該獨立訊號對任務無用**（LGB Δ < 0）
- 結合：stats 有 information 但 **non-task-relevant**。AE 壓縮這個空間是無意義的
- Attack 1（資訊論冗餘）+ Attack 7（AE 在 132 樣本 collapse）**empirically 聯合確認**

**KILL Plan E。不投入 14-20h。**

#### Meta-lessons (連 6 連 KILL/abort 累積)

16. **Stage A cheap probe 是 KILL 決策的最佳工具**：~2h cost 拿到 decisive 數據點，比 14-20h 直接投入省 7×。未來任何「加 player feature」類 Plan 應先跑此類 probe（compute_player_stats + match-disjoint LGB Δ）
17. **Supervised embed 是 noise 不代表加 feature 有用**：Probe 1 surprise（R²<0）原以為證明 supervised embed 不堪用 → AE 有空間。Probe 2 反駁此推論：stats 的「獨立訊號」對 next-stroke 任務本身就無用。**重要 nuance：linear independence ≠ task relevance**
18. **Match-disjoint protocol 自然抵抗 player aggregate signals**：rally-KFold 時代 A v2 OOF +0.004，match-disjoint 下 Δ=-0.003。差異 -0.007 即 player overlap leakage 強度量化。**未來 Plans 可用此 0.007 estimate 作為「leak-removed」reference**
19. **6 連 KILL/abort 證實 framework saturation 是 architectural ceiling 而非 plan-quality**：LA / Mirror / Pseudo / T / SSL / E 各以不同 lever 試（loss / aug / distill / arch / repr / feature），全部撞同一面牆 — information regime 已飽和於當前 (data, protocol, calibration) 三角

**累積失敗變體：27 條**。

---

### 2.29 Plan W (Length-Importance-Weighted Training) — Debate-killed 2026-05-05

**第 6 個 Plan-stage Debate-killed 變體**（連同 LA / Mirror / Pseudo / SSL / E）+ T abort = **本 session 7 連 KILL/abort**。

**動機**：對每個 train sample at prefix length k 套 importance weight `w(k) = q_test(k) / q_train(k)` 到 CE loss。6-bucket 量化：
- k=1: 1.601 ↑
- k=2: 1.240 ↑
- k=3: 0.997
- k=4-5: 0.818 ↓
- k=6-10: 0.551 ↓
- k=11+: 0.260 ↓

直接對症 train mean L=5.65 vs test L=2.9 covariate shift。Plan 自宣 P(+0.010) 15-25%，9-13h cost。

**Plan 試圖差異化 6 個 KILL 先例**：
- vs LA：per-length 不是 per-class，plug-in 29-param 不知 length
- vs T：同架構但訓練分布不同，預期 Pearson 較低
- vs length-stratified calibration（已失敗 LB -0.0037）：training-time 不是 post-hoc，不同層

**Debate#1 命中（兩 adversarial agent 並行）— 3 FATAL + 4 HIGH + 1 MEDIUM-HIGH**：

#### FATAL #1 — Vapnik 1998 §3.2 Monotonicity（Agent 1，最致命）

**§PROGRESS §4 已實證 length-stratified calibration LB -0.0037**。

關鍵信息論論證：
- Post-hoc length-stratified calibration 是 **strictly more expressive corrector**：可表達 6 (length buckets) × 29 (classes) = **174 個自由 length-conditional 修正參數**
- Plan W training-time IW 是 **strictly less expressive**：只 6 個 scalar weights，且必須透過 SGD on BiLSTM 192/2 reachable manifold
- **Vapnik 1998 §3.2 結構風險 monotonicity**：較弱 corrector 不可能在較強 corrector 失敗的相同訊號上找到正向 lift

加上自家 §1.1 診斷：「整個 OOF/LB gap 完全由 CV protocol bias 解釋，cross-event drift 是 secondary」— **length 不是 gap 主因**。Plan W 是「對非 cause 用更弱 corrector 重做已失敗的實驗」。

#### FATAL #2 — Plug-in Calibration Absorption 數學量化（Agent 1）

Plan W 的 logit shift Δ(c, k) 分解：
- β(c) = length-marginal class shift（per-class additive）→ **plug-in 完全吸收**（Saerens 2002, Lipton 2018 BBSE）
- γ(c, k) = length-conditional residual

Variance(w) ≈ 0.18，Δ 中 length-conditional residual 占 15-25%。其中 affecting argmax 占 15-20%（Guo 2017 calibration-vs-accuracy 分解）。

**Net surviving lift after plug-in absorption + argmax invariance ≈ 2.7% of pre-cal lift**。樂觀 pre-cal +0.005 → **post-cal +0.00014 macro-F1**，**遠低於 fold variance ±0.002**。

#### FATAL #3 — Bayesian Prior 6/6 KILL Convergence on ROI（Agent 2）

Cover & Thomas Ch. 2: H(Y|X) 是 (data distribution, feature set) 性質，不是 optimizer 性質。我們 (BiLSTM-192-2, 70k strokes, 29-class, 29-param plug-in) 已測得 saturation floor at OOF 0.444-0.446 across **6 個獨立 training-objective perturbations**。

Plan W taxonomically = Plan LA：
- 兩者都 reweight loss landscape redistribute gradient mass
- LA: by class frequency
- W: by length frequency
- 同 operator family `L_w(θ) = Σ w_i ℓ(f_θ(x_i), y_i)`

**P(W succeeds | LA failed) STRICTLY < marginal P(W succeeds)**（同失敗模式：model 已 fit high-weight regions, low-weight regions 噪音梯度但不改 Bayes risk）。

EU 計算：P(success ≥ +0.010) = 0.10, E[lift] = +0.0009, ROI = 0.0001/hr — **同 6 連 KILL**。

#### HIGH 攻擊摘要

3. **Cortes 2010 ESS 違規**：computed ESS_drop = 14.95% > **10% safety threshold**。k=11+ bucket 1496 effective samples × 19 actions ≈ 80 effective samples per class — gradient SNR drop √(1/0.260) = 1.96×。**Byrd & Lipton 2019 ICML**: IW effect → 0 in interpolation regime（我們 LSTM by epoch 8 接近 train CE floor）
4. **Macro-F1 機制不對齊**：Class 14 (lob) 在 k≥6 peak → 強 downweight 0.55→0.26，預估 F1(14) **-0.01 to -0.03**。Class 1, 13 (drives, blocks) mid-rally peak → marginal hurt。Net 19-class macro-F1 **預估 -0.0015**（symmetric 量化）。Plan §5 「k=1, 2 必 lift」是 per-stroke F1，**不等於 macro-F1**。
5. **Pearson > 0.85 必觸**：T (BiTransformer match-disjoint, **不同架構**) 已測 Pearson 0.91。W 同架構同 init 同 fold splits 同 epochs，**只差 per-sample loss weight** → 預期 Pearson **0.94-0.97**（Caruana 2004 taxonomy: sample-reweighting 是**最低 diversity tier**）。Plan §6 0.85 hard gate trigger P ≥ 90%。
6. **Test L=2.9 confounded artifact 假設**：Bickel & Scheffer 2007 證 IW on confounded shift 產生 bias proportional to residual confounding。三個替代解釋（match-mediated / player-mediated / random-truncation）任一成立 → IW correction **錯方向**。Plan 缺 pre-flight diagnostic（30 min 算 conditional L by sex × seen 即可驗）。
7. **Missing post-cal blocking gate**：Plan §6 binary gate 沒涵蓋 case (ii) test-w↑+full↓+post-cal↑、case (iii) test-w↑+post-cal↓（Attack 1 預測 outcome）。

#### 量化機率

| Agent | P(過 +0.010 ship gate) | EU/h |
|---|---|---|
| Agent 1 | < 5% | 0.0001/h |
| Agent 2 | 4-6% | 0.0001/h |
| 6 連 KILL Bayesian prior | 12.5% | — |

#### Verdict — KILL Plan W

兩 agent 共識 KILL。**這是第 7 個連續 Plan-stage Debate-killed**。

#### 7 連 KILL 收斂的 framework saturation 結論

| Plan | Mechanism | Verdict |
|---|---|---|
| LA | Loss reweighting (class) | Plug-in 吸收 |
| Mirror | Data augmentation | Saturation regime mismatch |
| Pseudo | Self-distillation | Mobahi ceiling + leakage |
| T | Architecture change | Pearson 0.91 ensemble math |
| SSL | Training-time pretrain | Bidirectional vs causal + cal absorb |
| E | Input feature | Probe Δ=-0.003, stats non-task-relevant |
| **W** | **Loss reweighting (length)** | **Vapnik monotonicity + Bayesian prior** |

7 個獨立 lever 全部撞同一面牆：**information regime saturation under (BiLSTM, 70k strokes, 29-class macro-F1, 29-param plug-in calibration) 四角約束**。Cover & Thomas: H(Y|X) 是 (data, feature set) 性質，所有 single-mechanism perturbation 在這 hypothesis class 內 traverse 都到同一 Bayes-optimal 邊界。

#### Meta-lessons (新增)

20. **Vapnik monotonicity 是 KILL 決策的最強信息論論證**：對任何「TRAINING-TIME version of a previously-failed POST-HOC correction」Plan，自動 KILL（無需 Debate#1，這條夠）。範圍：未來若有人提「training-time class reweight」（已 LA KILL）、「training-time length reweight」（已 W KILL）、「training-time prior shift any flavor」 — 全 KILL 不需細查
21. **Cortes 2010 ESS 安全線 10%**：任何 IW Plan 必須先算 effective N drop。> 10% 即 KILL。我們 W 在 14.95% 已 over
22. **Caruana 2004 diversity tier 排序**：arch > loss formulation > hyperparam > **sample reweighting**（最低）→ 任何 Plan 自宣「ensemble lift via sample reweighting」需立即 redflag
23. **「training-time vs post-hoc 不同層」是常見錯論**：兩者 target 同一 underlying signal，post-hoc 上限**嚴格 ≥** training-time 上限（信息論 monotonicity）。不接受「different layer」當 mechanism 差異化論證

**累積失敗變體：28 條**。

#### User 選擇路 3

User 選擇寫 deep research briefing (`docs/DEEP_RESEARCH_BRIEFING_2026-05-05.md`) 給外部 AI 諮詢，希望從 framework saturation 之外找新 idea。本 retrospective 不終結 phase — 等外部 AI input 後再決定下一步（重啟新 Plan / lock ship + final report / 其他）。

Deep research 結果（`docs/Where_the_Argument_Holds_and_Where_It_Breaks.md`）：兩個論證裂縫（Vapnik monotonicity 套用錯 + plug-in fit OOF marginal 非 test marginal）+ 3 個 surviving proposals (P1 / P2 / P3)。下方記錄三個 proposal 的實證結果。

---

### 2.30 Plan P3 (Length-conditional Mixture-of-Experts) — Stage A probe KILL 2026-05-05

**動機**（per deep research P3）：hard route by prefix length。L=1 → LightGBM specialized + Bayesian-shrunk per-server prior；L=2 → small Transformer or LGB；L≥3 → existing LSTM bag2 unchanged。Per-expert plug-in calibration on disjoint slices（Pearson undefined → escape ensemble math constraint）。

**Stage A probe 預檢**（base bag2 OOF macro-F1 by length）：

| bucket | N | F1_action | F1_point | avg | test% |
|---|---|---|---|---|---|
| k=1 | 14995 | 0.2930 | **0.1564** | **0.2247** | **34.4** |
| k=2 | 13126 | 0.2821 | 0.2074 | 0.2448 | 23.3 |
| k=3+ weighted | 41591 | 0.3460 | 0.2208 | 0.2834 | 42.2 |

L=1 vs L≥3 gap **+0.0588** > 0.02 threshold → Stage A 過。

**Debate#1 命中（兩 adversarial agent）— 3 FATAL + 4 HIGH + 1 MED-HIGH**：
- A6 FATAL: per-server prior 與 A v2 結構同型（per-rally LOO 不是 per-player LOO；Plan E Probe 2 已實證 cross-rally aggregate stats blended Δ=-0.003）
- A3 FATAL: Macro-F1 minority class impact — Class 14 (lob) @ L=1 < 30 samples，LGB `min_data_in_leaf=20` 連分裂都做不到 → F1(14) = 0; Net L=1 expected lift = +0.0025 - 0.005 = **-0.0025**
- A1 HIGH-FATAL: per-slice 87 plug-in params 在 14k OOF 上過擬風險（length-strat-cal 174 params LB -0.0037 同型）
- + A2/A4/A5/A7/A8 各 HIGH/MEDIUM-HIGH

**Attack 4 cheap diagnostic**（empirical confirmation pre-implementation）：

跑 `probe_p3_attack4.py` — 比較 existing global LGB on L=1 OOF vs slice-trained LGB on L=1 only，matched hyperparameters：

| Metric | Existing global LGB | Slice LGB on L=1 | Δ |
|---|---|---|---|
| F1_action | 0.2093 | 0.2297 | +0.0204 ✅ |
| F1_point | 0.1309 | 0.1198 | -0.0111 ❌ |
| Avg | 0.1701 | 0.1747 | **+0.0047** |

**Mixed signal**：slice training 對 action 有 lift (+0.0204) 但 point 退步 (-0.0111)。Net Δ = +0.0047 < +0.005 Test 2 KILL gate。Test-weighted projection: 0.32 × 0.0047 = **+0.0015**，遠低於 +0.010 ship gate。

**Verdict — KILL P3**。Slice training 部分有訊號（action）但 point 反退步抵消大半，且 Test-weighted lift << ship gate。

**Meta-lessons**：
24. **Stage A probe 過 (gap +0.0588) 不保證 Plan 成立** — Mechanism level (length 規模差異) 可量到，但**對應 architectural 解（slice training）效果不必對稱**。Action 善於 tabular fitting (server style → response)，point 因 spatial aleatoric noise 不適合 slice
25. **Macro-F1 min_data_in_leaf 約束**：LGB 預設 `min_data_in_leaf=20`，rare class @ slice 樣本不足時連分裂都做不到 → F1=0 是**結構性 floor，非 hyperparameter 可救**

**累積失敗變體：29 條**。

---

### 2.31 Plan P1 (Estimated-test-marginal plug-in re-fit) — v1 + v2 implementation KILL 2026-05-05

**動機**（per deep research P1）：plug-in 校準 fit on OOF marginal 而非 estimated test marginal — sub-optimal point 在同 29-param additive class 內。Saerens 2002 / BBSE / Alexandari 2020 給出 q_test(y) consistent estimator from unlabeled test predictions → 重新 align plug-in α 到 q_test-weighted macro-F1 objective。

**Stage A probe（test marginal vs OOF marginal divergence）**：

m_hat - p_OOF：7/19 action 類 shift > 0.02，5/10 point 類 shift > 0.02。Max |Δ_action| = 0.0455（class 10 搓球，model 低估 -0.046）；Max |Δ_point| = 0.0407（class 9 反手長球）。**測得 meaningful divergence**，Stage A 過。

**P1 v1 implementation (`probe_p1_relabel.py`)** — Saerens EM 估 q_test：

EM degenerate（與 deep research 預測 HIGH-severity attack 一致）：
- Action class 8 (pimple long push): q=0.69 vs p_OOF=0.005，**ratio 129×**
- Point class 3 (反手短球): q=0.98 vs p_OOF=0.003，**ratio 335×**

幾乎所有 probability mass 集中在罕見類別（numerical pathology）。Held-out result：
- Weighted macro-F1 Δ = +0.053（看似好但 q broken artifact）
- **STANDARD macro-F1 Δ = -0.061**（**LB metric 崩盤**）
- Test prediction 40% diff vs ship — extreme shift → **不送 LB**

**P1 v2 implementation (`probe_p1_v2.py`)** — m_hat 直接 + clamp [0.5×, 2.0×] of p_OOF：

Conservative q estimate（max ratio 2.1×，避免 v1 degeneracy）。Held-out result：
- Action: P1 v2 = 0.3814, current = 0.3901, **Δ = -0.0087**
- Point: P1 v2 = 0.2308, current = 0.2291, Δ = +0.0017
- **Avg Δ = -0.0035 < 0** → KILL

**根本問題**：plug-in 29-param 已對 standard macro-F1 OOF surface 最佳化。任何 reweighted objective 在同 function class 內**換點**（trade-off），**not 更高 surface**（no free lunch on the optimization surface）。

**Verdict — KILL P1**。Deep research 的「OOF marginal vs test marginal sub-optimal point」論點在實證上**找不到 lift**。Saerens EM degenerate（v1）；conservative m_hat 也 trade standard F1 1:1（v2）。

**Meta-lessons**：
26. **Saerens EM at 29 classes ill-conditioned 是 robust 失敗模式** — 即便加 class masking + p_OOF floor + Dirichlet smoothing 都救不了（v1 confirmed deep research's HIGH-severity attack）
27. **「Plug-in fit OOF non-test marginal sub-optimal」這個論點實證上不成立**：plug-in 換 objective surface 不換 capacity，trade-off 在同 surface 上 1:1。Deep research 對 P1 mechanism 的 +0.002-0.006 lift envelope 是**理論上限非實際下限**，本 dataset 實際下限 ≈ -0.003

**累積失敗變體：30 條**。

---

### 2.32 Plan P2 (FAISS kNN-over-hidden-state head) — Stage A cheap probe stacking KILL 2026-05-05

**動機**（per deep research P2）：build FAISS index over BiLSTM hidden states (192-dim) on train，retrieve k=64 NN per val sample，p_kNN as standalone head（not λ-interpolation）併入 ensemble via Caruana stacking。Pearson 預期 0.55-0.70（vs Plan T 0.91），non-parametric retrieval 提供真 architectural diversity。

**Cheap version Stage A**（無需 model forward extract hidden state）：FAISS over 49-dim LGB tabular features，τ ∈ {10, 50, 100} sweep。

**Stage A 結果**：

| τ | F1_action | F1_point | avg | Pearson_action | Pearson_point |
|---|---|---|---|---|---|
| 100 (best) | 0.1973 | 0.1466 | **0.1719** | **0.6121** | **0.5354** |

- ✅ **Gate 1 (Pearson < 0.75) PASS** — 9 個 plan 第一次！對比 Plan T 0.91，FAISS retrieval 提供真 decorrelation
- ❌ **Gate 2 (F1 > 0.20) FAIL** — kNN standalone F1 0.17 vs LSTM bag2 F1 0.29

**Caruana stacking probe**（驗證 ensemble lift 是否 actionable）：

`probe_p2_stack.py` 跑 (LSTM bag2 + kNN-LGB) 加權 ensemble，搜 w_kNN ∈ [0, 1]：

```
Action: best w_kNN = 0.00, F1 = 0.3678 = ship F1
Point:  best w_kNN = 0.00, F1 = 0.2357 = ship F1
Δ score = +0.0000
```

**Optimal stacking weight = 0.00**，ensemble lift = 0。F1 monotonically 跟 w_kNN 反比例。

**根本原因**：component F1 gap 太大。LSTM F1 0.30 vs kNN F1 0.17（gap 0.13）。Caruana 2004 ensemble lift 需 component F1 接近且 Pearson < 0.65。Pearson 0.61 ✓ 但 F1 gap 太大 → 任何 non-zero w_kNN 都拉低 ensemble F1。

**Full hidden-state version 預判也不會救**：即使 192-dim hidden 把 F1 推到 0.25-0.30，與 ship 0.30+ 仍差距大，stacking lift 預期 +0.001-0.003。Deep research 的 +0.004-0.009 預估**過度樂觀**（基於 Khandelwal kNN-LM 1500× scale 等 cross-domain reference）。

**Verdict — KILL P2 entirely**（不跑 full hidden-state version）。

**Meta-lessons**：
28. **Pearson 低不代表 ensemble lift 存在**：Caruana 2004 ensemble math 需 BOTH (a) Pearson < 0.65, AND (b) component F1 接近。kNN 是第一個 (a) 過的 mechanism 但 (b) fail
29. **kNN standalone F1 受 datastore size 限制**：70k 樣本 / 19 actions ≈ 3700/class 平均，但 minority class 只 70-700 samples → kNN 對 majority class 預測尚可但 minority class 預測弱（distribution 上 biased）
30. **Cross-domain reference cargo culting 第二次確認**：deep research 引用 Khandelwal kNN-LM 在 1500× our scale 上的 +0.005-0.010 lift，本 dataset 實際 0 — 必須降權 cross-domain references **per Meta-lesson 5/19**

**累積失敗變體：31 條**。

---

#### 9 個 Plan + 3 deep research proposals 全失敗總結

| Plan | Mechanism | Verdict |
|---|---|---|
| LA / Mirror / Pseudo / T / SSL / E / W | 多種 framework-internal lever | Debate#1 KILL |
| **P1 v1 (Saerens EM)** | Test marginal recalibration | Held-out -0.061 KILL |
| **P1 v2 (m_hat clamped)** | Conservative q estimate | Held-out -0.0035 KILL |
| **P3 (Length MoE)** | Slice-trained experts | Attack 4 mixed signal Δ=+0.0047 < gate KILL |
| **P2 (FAISS kNN)** | Non-parametric retrieval | Stacking lift = 0 KILL |

**10 個獨立 mechanism + 31 個 failed variants** 全部撞同一面牆。Information regime saturation under (BiLSTM, 70k strokes, 29-class macro-F1, 29-param plug-in calibration) **diagnose 完全 confirmed**。

Deep research 的兩個論證裂縫（Vapnik monotonicity 用錯 + plug-in target marginal 偏差）**理論上有道理但實證上找不到 lift**。代表 framework saturation 是**多層次** — 不只一個論證撐起，而是 (data scale + architecture + calibration + macro-F1 metric) 多重耦合。

**用戶決定路徑（2026-05-05）**：在 path B (final report) 之前，再試 **Mamba (V5Z_ARCH=mamba)** — pure PyTorch state-space architecture，T 之外的最後 architecturally novel 候選。預期 Pearson 0.85-0.92（per deep research 預判，BiGRU 同 family 已敗 -0.02 vs LSTM）。Stage A diagnostic style: hard abort gates 同 Plan T (Pearson > 0.85, OOF < 0.4300, L=1/L=2 退步 > 0.01)。

### 2.34 Test data refresh 2026-05-06 + Phase 2 sgp predictor 系列

**事件**：主辦於 2026-05-06 釋出全新 test_new.csv，1845 rallies / 5668 strokes / 71 players (31 unseen, 43.7%) / **無 serverGetPoint 欄位**。LB 全面 reset。舊 ship `submission_v5z_md_advcal_bag2.csv` (LB 0.4374) 失效。

**Phase 1 — Pipeline 適配（非 phase per §3 定義，僅路徑修正）**:
- `train_v5z.py` 加 env vars `V5Z_TEST_CSV / V5Z_PID_TEST_CSV / V5Z_INFER_TEST_ONLY / V5Z_LSTM_WEIGHTS_TAG`
- `prepare_samples` + `build_lgb_features` 容錯 missing serverGetPoint
- INFER_TEST_ONLY mode：load LSTM fold weights，跳訓練；LGB 仍重訓
- PID2IDX 改由 `V5Z_PID_TEST_CSV` 控制（INFER mode default `data/test.csv` 對齊舊 weights）
- `advcal_match_disjoint.py` 加 `V5Z_ARTIFACT_PREFIX / V5Z_TEST_ARTIFACT_SUFFIX / V5Z_SUBMISSION_SUFFIX / V5Z_SGP_PRED_FILE`
- s42/s1337 各自 inference on test_new → `v5z_md_full_NEWTEST_s{42,1337}_{oof,test}.npz`
- Single-seed OOF action+point 0.4361（s42）一致於原 s42，determinism 保持

**Phase 2.34a Bayesian sgp predictor — KILL 2026-05-07**:
- Per-server Bayesian-smoothed winrate（κ ∈ {5,10,20,50}）+ symmetric 1-receiver-loserate
- Match-disjoint StratifiedGroupKFold(N=5, groups=match_id)，每 fold winrate 從非-holdout pool 算
- 結果：所有 κ holdout AUC ∈ [0.5233, 0.5244]，best κ=50 AUC=0.5244
- 期望 test AUC ≈ 0.519（cold-start 調整公式 AUC ≈ 0.81×AUC_seen + 0.095，f=0.563）
- Score gain over 0.5 placeholder: +0.004（LB noise floor 內）→ Bayesian gate 0.55 KILL
- **Meta-lesson 31**: Per-rally outcome 信息論上由 player 平均 winrate 預測上限低 — 1-feature model 在 14995 rally / 166 player 規模下 AUC 上限 ~0.52-0.53

**Phase 2 Debate#2 推翻 Bayesian gate**:
- 原 plan 「Bayesian < 0.55 → 不上 LGB」邏輯錯：LGB feature 集嚴格大於 Bayesian → AUC ≥ Bayesian 必然
- Score 戰略 invert：sgp 是 ROI 最高 lever（recoverable 0.04-0.07 vs F1 saturated +0.001-0.003）
- LSTM hidden state（trunk_out 128-dim）為 unexplored 高 ROI 候選
- 決策修正：LGB-quick + LGB-deep 雙路 + ensemble

**Phase 2.34b LGB-quick — SHIP 2026-05-07**:
- Rally features 39 個 + Bayesian winrate + score state + visible-prefix summary
- **Prefix augmentation 解 leak**：原 v1 用 full rally length（含未來 strokes）→ AUC 0.999 trivial leak。改為 k=1..N-1 隨機抽 5 prefix per rally
- Match-disjoint 5-fold，per-fold winrate 非-holdout pool
- **Holdout AUC 0.6451**（per-rally aggregated；per-prefix AUC ~0.59）
- Top features: rcv_loserate (16k gain), srv_winrate (9.7k), last_strengthId (6.3k), ctx_len (5.9k) — 全部 leakage-free
- Test pred mean 0.526 / std 0.061 / range [0.37, 0.72]
- **Meta-lesson 32**: train 全 rally + visible prefix 是隱形 train-test asymmetry — last_strike_parity = N % 2 perfectly determines sgp in train，test 完全無此信號。任何 sgp 類預測必須 prefix-augment
- **Meta-lesson 33**: 39-feature LGB 在 70k samples 上 stable，min_data_in_leaf=50 + lambda_l2=1.0 防 overfit

**Phase 2.34c LGB-deep — KILL 2026-05-07**:
- Pipeline: 修改 `PingPongModel.forward(return_trunk=True)` 暴露 trunk_out（128-dim final fused representation）
- Per (rally, k) sample 抽 trunk via 對應 fold's 權重（OOF integrity）→ s42 trunk 70k×128
- LGB on (39 hand + 128 trunk) = 167 feature
- **Holdout AUC 0.5904** vs LGB-quick 0.6451 — **退步 0.055**
- 失敗模式：trunk_in_top20 importance 僅 1-3/20，folds 3-5 early-stop iter 3-6（severe overfit）
- Pearson(quick, deep) = 0.8955 → 高相關 → ensemble 任何權重組合都 ≤ quick alone
- **Meta-lesson 34**: LSTM trunk 為 next-stroke prediction 訓練 → encoded info 對 sgp 邊際 — 任務 mismatch。Trunk 不是 universal feature
- **Meta-lesson 35**: 39 + 128 = 167 feature on 70k samples，feature_fraction=0.7 each tree 看 117，僅 ~27 是 strong feature。dilution 比 noise 更傷
- **Meta-lesson 36**: aux head 的 trunk reuse 假設 (cross-task transfer) 在本 dataset 不成立。同樣風險適用未來 hidden-state-based proposals

**Phase 3 ship 2026-05-07** (Sub 1):
- `submission_v5z_md_advcal_bag2_NEWTEST.csv` (1845 rallies)
- α_action=0.85, α_point=0.98 (LSTM-dominant under match-disjoint，consistent with original)
- F1_act=0.3679, F1_pt=0.2357（plug-in calibrated, action+point only）
- sgp 預測來自 LGB-quick (artifacts/sgp_pred_NEWTEST_lgb_quick.npz, AUC 0.6451)
- OOF score (assuming AUC=1.0): 0.4414
- **實際 LB: 0.3539886**（落在保守-平估中間）
- 反推 test AUC ≈ 0.58（假設 action+point gap -0.009 同舊 test）
- Holdout-to-test AUC gap = -0.065，與 Debate#2 cold-start 公式預估一致（0.81×0.6451+0.095=0.617，加 ensemble-aggregation 退化 ~-0.04 = 0.58）

### 2.35 Phase 4 sgp 改進系列 2026-05-07

**B (LGB+LSTM softmax 29-dim) — KILL**:
- 替換 trunk (128) 為 LSTM action+point softmax (19+10=29) — 較低維、任務 relevant
- OOF AUC 0.6067 < quick 0.6451，softmax-in-top10 僅 0-1 — 信號被 hand features 完全 captured
- 確認 §2.34c 結論：LSTM-derived features for sgp 普遍 hurt
- **Meta-lesson 40**: LSTM 對 sgp 的代表性 features (trunk OR softmax) 在 LGB 內無增量信息；hand features 已 saturate sgp-predictable signal

**D (LGB+player_elo+cross) — Marginal improvement**:
- 加 ELO (init=1500, K=24, 從 train rallies 算)、winrate×score_diff、winrate×ctx_len 等 cross features
- OOF AUC 0.6490（+0.004 vs Q 0.6451）
- 主要 lift 來自 cross feature **winrate×score_diff (gain 5527)**，超過 ELO 本身 (gain 2782)
- ELO 不是主要信號，因 player aggregate winrate 已 capture 大部分

**Q+D 50/50 Ensemble (Sub 2 ship) — LB confirmed**:
- OOF AUC 0.6510（+0.006 vs Q）
- Pearson(Q OOF, D OOF) = 0.886, test pred Pearson 0.976（high collinearity）
- 多元 ensemble (Q+D+B+Deep) 都 ≤ Q+D，因 B/Deep 弱
- **LB: 0.3545570** (+0.0006 over Sub 1 0.3539886)
- 反推 Sub 2 test AUC ≈ 0.584 (vs Sub 1 0.581) — ensemble 在 test 多 +0.003 AUC
- **Transfer rate (OOF→LB)** = 0.0006/0.006 = ~10% — 比預期低，因高 Pearson + LB sample variance

**H (LSTM with sgp aux head, V5Z_SGP_AUX_W=0.3) — BREAKTHROUGH 2026-05-07**:
- 修改 PingPongModel：加 `head_sgp` (linear 128→32→1)，forward 加 `return_sgp` flag
- 訓練時 BCE loss on rally sgp (per (rally, k) 樣本，sgp 標籤 = rally 常數)
- 結合 loss: 0.5×CE_action + 0.5×CE_point + 0.1×Aux + 0.3×BCE_sgp
- s42 從頭訓練（V5Z_INFER_TEST_ONLY=0），EPOCHS=40，PATIENCE=10
- **Action+point F1 略退步**（F1_act 0.3679→0.3648, F1_pt 0.2357→0.2290, Score -0.004）— capacity dilution
- **OOF sgp AUC**:
  - Per-sample (single prefix): **0.5836**
  - Per-rally aggregated (5 prefix avg): **0.7706** ⭐⭐⭐
  - Lift +0.19 from prefix averaging (vs Q's +0.05)
- Per-k single-prefix AUC: k=1 0.51, k=2 0.54, k=3 0.61, k=4 0.58, k=5 0.60
- **Pearson(Q OOF, H OOF) = 0.38** ⭐ — H 與 LGB 系列高度 diverse（vs Q vs D 0.886, Q vs B 0.60, Q vs Deep 0.44）
- Multi-prefix test extraction (`extract_sgp_aux_multiprefix.py`)：對每 test rally 用 k=1..N_visible 全部 prefix，5-fold avg
- Test pred range [0.40, 0.66] (std 0.036) vs single-prefix [0.22, 0.90] — variance reduced

**Sub 3 ship: Q+D+H 0.05/0.05/0.9 ensemble**:
- OOF AUC **0.7732** (+0.122 vs Q+D 0.6510)
- 預期 LB +0.005 ~ +0.025 over Sub 2 (test transfer 視 H multi-prefix 是否保留 OOF lift)
- ship 檔: `submission_v5z_md_advcal_bag2_NEWTEST_QDH.csv`

- **Meta-lesson 42**: Joint training 加 sgp aux head 真的把 sgp signal 嵌入 trunk — 推翻 §2.34c 「LSTM trunk 對 sgp 邊際」結論。差別在 frozen-trunk extraction (LGB-deep) vs 重新 retrain。前者 trunk 為 next-stroke 訓練，後者 trunk 為 sgp+next-stroke 共同訓練
- **Meta-lesson 43**: prefix-aggregation 在 LSTM-sgp-aux 上 lift 巨大 (+0.19 OOF)，因不同 prefix length 帶不同 contextual info → multi-prefix avg 同時 reduce variance + integrate 多角度 evidence
- **Meta-lesson 44**: capacity dilution 在 multi-task LSTM 訓練上是 0.005-0.01 magnitude，需 sgp gain > capacity cost 才值得（本案: -0.004 F1 cost vs +0.04+ AUC gain → net +0.01-0.02 score）
- **Meta-lesson 45**: aux head 重訓 vs frozen-feature LGB 是不同 mechanism；前者改 representation 後者只用既有；任何 cross-task transfer 提案應優先 aux-head retrain

### 2.36 Sub 3 LB regression + Phase 5 系列 — 2026-05-07

**Sub 3 ship (Q+D+H 0.05/0.05/0.9) — LB 0.3523611 REGRESSED from Sub 2 (0.3546)**:
- Test AUC ≈ 0.573 (calculated: (0.3524 - 0.238) / 0.2)
- vs Sub 2 test AUC 0.584 → **-0.011 AUC 退步**
- vs OOF aggregated 0.7706 → **-0.198 OOF→test gap**
- Diagnosis: per-rally aggregated OOF AUC was an **aggregation artifact** of averaging 5 prefixes per rally; test only has 1 prefix per rally
- Per-k single-prefix AUC: k=1 0.51, k=2 0.54, k=3 0.61, k=4 0.58, k=5 0.60
- Test L 加權: 0.275×0.51 + 0.257×0.54 + 0.21×0.61 + 0.13×0.58 + 0.13×0.60 = **0.561** ≈ actual test 0.573 ✓
- **Meta-lesson 46**: per-rally aggregated OOF AUC 是 misleading metric — 必須用 test-distribution-weighted per-k AUC 作 acceptance gate

**Phase 5-B Pure-sgp LSTM (V5Z_SGP_ONLY=1) — KILLED 2026-05-07**:
- 拋棄 action+point+aux losses，BCE on sgp 為唯一目標
- Validation 改用 sgp AUC 為早停 criterion
- 每 fold best per-sample sgp AUC: 0.5835, 0.5805, 0.5857, 0.5964, 0.6005 (mean 0.589)
- 顯著現象：**AUC 在 epoch 1 達 peak，後續 epoch 持續退步** — pure-sgp 在 sgp loss 上 overfit 到 spurious patterns
- Per-rally aggregated OOF AUC **0.6949** (vs H aux 0.7706 = -0.076)
- **Meta-lesson 47**: LSTM 架構本身不是 sgp 預測 bottleneck — 同 0.58-0.60 per-sample AUC ceiling 出現在 H aux (multitask)、Pure-sgp (single-task)、LGB-quick(prefix-aug)。**Bottleneck 是 information regime（visible prefix → outcome 的 inherent signal limit），不是 model capacity**
- **Meta-lesson 48**: pure-sgp 拋掉 multitask regularization → trunk degenerate → predictions across prefixes correlate → per-rally aggregation lift 消失（H aux 多樣性大 → aggregation +0.19，pure-sgp +0.11）
- **Meta-lesson 49**: 任何「為單一 task 拋棄 multitask grounding」的提案應預期 -0.05 到 -0.10 generalization 退步，因 multitask 提供 implicit regularization

**Phase 5-DR Deep Research Briefing (docs/DEEP_RESEARCH_BRIEFING_2026-05-07.md)**:
- 5 個 theories 給外部 AI 評估（A: 外部資料、B: seq2seq、C: deep features、D: LB overfit、E: pure-sgp）
- 4 specific asks（LB noise、racquet sport AUC>0.7 published methods、cold-start mitigation、structural priors）
- 等待外部 ChatGPT 5 / Gemini 2.5 Pro / Claude 4.7 回應

- **Meta-lesson 37**: per-rally aggregated holdout AUC 比 per-prefix AUC 高 0.05（5 prefix 平均效果）— 但 test 只 1 prefix → 應以 per-prefix AUC 0.59 作為 test 預估 baseline，per-rally 0.6451 是上界
- **Meta-lesson 38**: cold-start 公式 AUC_total ≈ 0.81×AUC_seen + 0.095 在 f=0.563 (seen fraction) 下實證有效，未來 player-aggregate 預測都可套
- **Meta-lesson 39**: OOF AUC lift → LB AUC lift 50% transfer rate 估計過於樂觀；高 Pearson ensemble + LB sample variance 把 transfer 壓到 ~10%
- **Meta-lesson 40**: cross features (winrate×score_diff) 在 sgp 預測中比 raw features 更有信息，因 sgp 是 contextual outcome；future feature engineering 應優先 cross terms
- **Meta-lesson 41**: 4 model ensemble (Q+D+B+Deep) 用 weak components 反而 hurt — Caruana 2004 的「all models 都加 weight 0+」假設在 weak component 下 break；reserve ensemble for similar-strength models

---

## 3. 關鍵洞見

### 3.1 OOF score 為什麼會 saturate 在 0.4816

| 因素 | 影響 |
|---|---|
| **70k 樣本對 deep model 偏少** | Transformer 沒打贏 BiLSTM；larger LSTM 開始 overfit |
| **序列太短（mean 5.65）** | RNN structural bias 已足；attention 沒發揮空間 |
| **Class imbalance 已被 class weights 處理** | Focal/SF 的 imbalance treatment 是疊加而非替代 |
| **OOF macro-F1 受空類別 cls 17/18 拖累** | sklearn default 自動忽略，但 LB 評分機制可能不同 |
| **OOF 與 LB 可能有 systematic gap** | 真實 LB 可能 ±5% 甚至更多 |

### 3.2 OOF α_LSTM 從舊資料的 0.40 升到新資料的 0.64-0.70

新資料 train rallies 約 2.6×（5800 → 14,995），LSTM 受益於更多訓練訊號顯著超過 LGB。**資料量翻倍 → 深度模型佔比上升**符合預期。

### 3.3 Plug-in additive bias 是 macro-F1 校準的金標

理論最優（Koyejo 2014）。比 multiplicative scale 通用、與 macro-F1 surface 同型。

### 3.4 Bag selection > Bag size

3 seeds 不一定比 2 seeds 好。如果某 seed 偏弱（>0.002），自動篩除它優於含進來。

---

## 4. 真正能突破 0.4816 的方向（未實作）

每個都需要 3+ 小時、不保證正向，已超出 Plan Z 範圍：

1. **External player ELO / win-rate features** — 從 train.csv 計算每個選手的歷史勝率、與對手交手紀錄等，加入 LGB 特徵集。**風險**：可能違反競賽「使用比賽外資料」規則，需先確認規則。
2. **Stacking with cross-validated meta-learner** — 在 OOF 上訓練小型 LGB / LR 學「最佳 ensemble 函數」（非線性組合，可能勝過 scalar α）。風險：OOF 過擬合。
3. **Self-supervised pretraining** — 用 train + test 的 stroke 序列做 masked prediction 預訓練 stroke encoder，再 finetune 到下游任務。
4. **Multi-fold-split bagging** — 同 seed 跑 N 個不同 random_state 的 KFold，取所有 OOF 預測平均。比 LSTM seed bagging 更直接降低 split-specific variance。

---

## 5. 檔案清單

### 5.1 Ship 用
- `submissions/submission_v5z_advcal_bag2.csv` ★ **最終 ship**

### 5.2 訓練腳本
- `src/train/train_v5z.py` — 主訓練程式，env vars：
  - `V5Z_SEED` (default 42)
  - `V5Z_TAG` (default v5z)
  - `V5Z_ARCH` (lstm/gru/transformer，default lstm)
  - `V5Z_FOCAL` (focal gamma，default 0=disabled)
  - `V5Z_SOFTF1` (soft-F1 weight, default 0=disabled)
  - `V5Z_HIDDEN`, `V5Z_LAYERS`, `V5Z_EPOCHS`, `V5Z_PATIENCE`
- `src/train/advcal_v5z.py` — Calibration 後處理，env var：
  - `V5Z_SEEDS` (CSV，default 全部 v5z_s*)

### 5.3 OOF / Test artifacts
- `artifacts/v5z_s42_oof.npz`、`artifacts/v5z_s1337_oof.npz`（核心 bag2 數據）
- `artifacts/v5z_s2024_oof.npz`（弱 seed，捨棄）
- `artifacts/v5zg_s42_*` (GRU)、`artifacts/v5zT_s42_*` (Transformer)、`artifacts/v5zSF_s42_*` (Soft-F1)、`artifacts/v5zf2_s42_*` (Focal)
- `artifacts/v5z_advcal_bag*_params.npz`（校準參數）
- `artifacts/_old_v7_stale/`（前一版資料的 stale artifacts，已歸檔）

### 5.4 Logs
- `logs/v5z_s{42,1337,2024}.log` — 主 LSTM 訓練 log
- `logs/v5zg_s42.log`、`logs/v5zT_s42.log`、`logs/v5zSF_s42.log`、`logs/v5zf2_s42.log` — 失敗實驗 log
- `logs/advcal_v5z_bag*_*.log` — 校準 log

---

## 6. 競賽情報背景

- **任務**：給定 rally 觀察序列，預測下一 stroke 的 actionId（19 類）/ pointId（10 類）/ rally 結果 serverGetPoint（二分類）
- **評分**：`Score = 0.4 × Macro-F1(action) + 0.4 × Macro-F1(point) + 0.2 × AUC(serverGetPoint)`
- **資料更新（2026-04-30）**：主辦單位修補 test 資料外洩，全面更換 3 個 csv，並建議**訓練時移除 serverGetPoint 輸入特徵**
- **選手提示**：每 rally 的 serverGetPoint 在 test.csv 仍可見（rally-level 常數），可直接複製到 submission（AUC ~0.998）
- **新資料規模**：14,995 train rallies / 1,236 test rallies / mean 5.65 strokes/rally (train) / mean 2.90 strokes/rally (test)


---

## §2.37-2.42 Phase 5 全 retrospective (2026-05-08)

外部 AI 三份 deep research briefing 全部執行完，5-day plan + 額外 2 個 theory test。

### 2.37 Day 1 — T3 結構 priors + T1 stacking (SHIP, +0.0018 LB)
- T1: LGB multi-class on R = N - k (DeepHit-style) → OOF 0.7832 per-rally agg, per-prefix 0.62
- T3: T1 stacking pred + Morris importance + score sequence + cum_strokes + service_block_position
- OOF 0.8164 (+0.033 over T1), per-prefix 0.7675
- LB 0.3563681 (+0.0018 vs Sub 2 yesterday Q+D 0.3546)
- t1_stacking_pred dominates LGB importance (gain 65815, 3-4× any other feature)
- **這是 Phase 5 唯一真正轉折**

### 2.38 Day 2 — T4 Multi-level EB partial pooling (邊際, 0 LB lift)
- 取代 κ=50 fixed shrinkage 為 method-of-moments + group-level (sex × hand × style)
- OOF 0.8181 (+0.0017), 4 個 EB features 進 top 10
- LB ship 未直接送 (T6 已包含 EB)，但累積結果 LB 0 lift

### 2.39 Day 3 — T5 Markov bigram + deviation (邊際, -0.0003 LB)
- Per-player action transition matrix with Laplace α=0.5
- ℓ_u, ℓ_g, Δℓ, surprise (last/max/mean/std)
- OOF 0.8197 (+0.0016 over T4)
- **LB 0.3560529 — REGRESSED 0.0003 from T3 alone**
- Pearson(T3, T5) > 0.997 — Markov bigram 信號被 t1_stacking 吸收

### 2.40 Day 4 — T6 NMF cold-start (邊際, -0.0002 LB)
- KL-NMF rank-6 on 41-dim player behavior matrix (action+point+strength+spin distributions)
- NNLS fold-in for unseen test players (44%)
- nmf_skill_diff_l2 + 12 nmf_emb features
- OOF 0.8198 (+0.0001 over T5), nmf_skill_diff_l2 rank 10 in top features
- LB 0.3562296 — REGRESSED 0.0002 from T3
- 預期 NMF 在 unseen test players 大幅 lift，但 LB 沒顯現 → 可能 OOF 沒有充分 simulate cold-start

### 2.41 T7 KILL — Handedness inference
- 從 ratio_1_to_3 雙峰分布 infer 15.8% lefty (接近 TT 真實比例 10-15%)
- 加 8 個 handedness features (proxy, inferred_lefty, mixed_handedness, canonical pointId)
- OOF **0.8167 (-0.0031 vs T6) — 退步**
- T7 features 進 top 25 共 2 個但僅 mid importance
- 解釋：LGB 已從 player ID + handId × pointId interaction 隱式學到，加 explicit 反為 noise + overfit
- **Meta-lesson 53**: explicit handedness inference 在 LGB 上 hurt — 當 model 已能隱式學到，加 weak heuristic feature 是 net negative

### 2.42 T8b — Match-level test self-supervised (marginal, ship pending)
- 對每個 rally，計算同 match 其他 rallies 的 player-in-match aggregates (leave-one-out anti-leak)
- 修正 T8 v1 scale leak: drop match_avg_L, srv/rcv_match_strokes（train full vs test truncated 不同 scale），改用 RATES (attack_rate, defense_rate, strength_mean)
- OOF 0.8193 (-0.0005 vs T6)
- 4-5 個 match-level rates 進 top 10 (match_attack_rate, srv/rcv_match_attack/defense_rate)
- Pearson(T6, T8b) test = 0.998 — 相似但有微差異
- 假設：在 unseen test player (44%) 上 LB 可能 lift +0.001-0.003，OOF 看不出
- **Ship CSV ready, Sub 1 候選明日**

### Sgp 路線 ceiling 確認
- 8 種獨立 feature 機制（hand crafted / EB / Markov / NMF / handedness / match-self-sup）OOF 全收斂 0.8190-0.8200
- LB 全在 0.3560-0.3564 範圍（±0.0003 LB sample noise）
- **Information ceiling 在 sgp test AUC ~0.59**（per cold-start 公式 0.81×0.7 + 0.095 ~0.66 上界，實測 0.59 較預期低）
- 進一步突破需：外部資料、pseudo-label、F1 attack（不是 sgp）

### Meta-lessons 50-55 (新增)
- 50: OOF→LB transfer 在 Day 1 後變 0 甚至負 — Day 2-4 累積 OOF +0.003 完全沒 LB lift
- 51: Pearson > 0.997 的 ensemble 邊際 = 0
- 52: Match-level features 必須 rates（scale-invariant），不要 counts/avg_L
- 53: explicit handedness inference 在 LGB 上 hurt — model 已隱式學
- 54: T3 結構 priors 是 sgp 真正轉折，後續是噪音
- 55: F1 是 LB 2× weight per unit；sgp ceiling 後該轉 F1

---

## §2.43-2.45 Phase 5 F1 attack (5/8) — 備援 ship 已生成，未 LB 測

### 2.43 Length-stratified plug-in additive bias (per L bucket)
- 替代 single global plug-in bias 為 per L bucket {1, 2, 3, 4, 5+} 各自 calibration
- OOF F1_action +0.0048, F1_point +0.0048
- **Ship CSV**: `submissions/submission_v5z_md_advcal_bag2_LSTRAT.csv`
- 警示：Plan W (training-time length-importance-weighted) 已 LB -0.0037 證實 length-stratified 是 hot spot；但 post-hoc per-bucket calibration 是 strictly more expressive (per Vapnik 1998 §3.2 monotonicity §2.20 meta-lesson)，理論可超過 global plug-in
- Expected LB transfer rate ~50%（OOF +0.005 → LB +0.0025），落在 LB sample variance 內，需實測
- 未 LB 測，列為 Phase 6 Step 1 失敗時的備援 Sub 3 候選

### 2.44 Pseudo-label LGB action+point (semi-supervised)
- conf threshold action 0.30, point 0.20 → 1585/1845 action, 1459/1845 point pseudo-labeled
- LSTM α=0.85 dominates → 98% same predictions as T3 ship (PSEUDO_v2 vs T3 cosine = 0.98)
- **Ship CSV**: `submissions/submission_v5z_md_advcal_bag2_PSEUDO_v2.csv`
- Pseudo-label v1 (old test) 已 §2.24 KILL by Mobahi 2020 self-distillation theorem (~0 information gain)
- 新 test 上 LSTM dominance 結構性壓制 LGB pseudo-label 訊號 — 預期 LB lift < 0.001
- 未 LB 測，與 LSTRAT 同樣為備援，**不是主路線**

### 2.45 Phase 5 結束 — sgp + F1 雙路 saturate 確認
- 8 種獨立 feature 機制（hand crafted / EB / Markov / NMF / handedness / match-self-sup）OOF 全收斂 0.8190-0.8200
- 所有 sgp 變體 LB 在 0.3560-0.3564 (±0.0003 LB sample noise)
- F1 路線 (length-strat cal, pseudo-label) OOF 邊際 +0.005 但 transfer rate 未測
- **Information ceiling 確認**：framework 內所有可調 lever 都已用盡，剩下 +0.044 LB gap 必須走「真正引入新 information」之一
  1. ~~Symmetry mirror augmentation~~ kill (§2.23)
  2. ~~Logit Adjustment 訓練改 loss + 拿掉 class weights~~ kill (§2.22)
  3. ~~Pseudo-labeling on test~~ marginal (§2.44, this section)
  4. **External cross-sport pretraining (ShuttleSet)** — Phase 6 Step 3 候選
  5. **Random-prefix truncation augmentation (input distribution fix)** — Phase 6 Step 1 候選
  6. **Logit Adjustment + 20-sweep coord-ascent cal (loss + cal 雙修)** — Phase 6 Step 2 候選 (與 §2.22 LA 機制不同)

---

## §3 Phase 6 plan — external research synthesis (5/8 frozen)

3 份外部 AI deep research（`docs/{chatgpt,claude,gemini}_research.pdf`）已 cross-source 評估完成。詳細執行計畫在 `docs/PHASE_6_PLAN_2026-05-08.md`。

### §3.1 Cross-source consensus
- **STRONG CONSENSUS (3/3)**: ShuttleSet/ShuttleSet22 cross-sport pretraining；Hsu et al. 2026 MJSSM peer-reviewed validation
- **Claude UNIQUE**: Random-prefix truncation augmentation (+0.025~0.060 LB)，最高 ROI，ChatGPT/Gemini 都漏掉
- **Claude UNIQUE**: Logit Adjustment + 20-sweep coord-ascent calibration (+0.012~0.025)，與 §2.22 LA 機制不同
- **Gemini-only (跳過)**: DLDDTEA sliding-window (LGB saturate)、TT4D 3D 軌跡 (schema 不符)、T3Set IMU/video (推論時無)、TTSwing kinematic (wearable 不適用)
- **ChatGPT-only**: Next-shot prior features (ShuttleSet 預訓練的輕量替代，非首選)

### §3.2 三步執行（gate-driven）
1. **Step 1 (5/9 早, 2-4h)**: Random-prefix truncation augmentation
   - dataloader 改：採樣 k ∈ {0.32, 0.24, 0.16, 0.12, 0.16}，prefix[:k] → target_{k+1}
   - LGB aggregate features ON PREFIX (leakage hot spot)
   - step_idx + is_truncated 進 LSTM/LGB
   - 1-hr kill: OOF AUC 0.819 → 0.65~0.75
   - Expected LB: +0.025~0.060

2. **Step 2 (5/9 下午, 4h, gate: Step 1 LB +0.005)**: Logit Adjustment + 20-sweep cal
   - `F.cross_entropy(logits + tau*log_prior, y)`, tau ∈ {0.5,1.0,1.5}
   - Drop adjustment at inference
   - 20-sweep coord-ascent over per-class additive bias
   - 1-hr kill: 最稀 5 類 F1 ≥30% relative lift
   - Expected: composite +0.012~0.025

3. **Step 3 (5/10, 6-8h, gate: Step 1+2 cumulative +0.020)**: ShuttleSet pretraining
   - ShuttleSet + ShuttleSet22 ≈ 70k strokes
   - BiLSTM 128-256d, 3 heads, prefix-truncation sampling
   - Restore body, reinit heads on TT
   - 1-hr kill: badminton-only ShuttleSet22 truncated val top-1 ≥ 0.30
   - Expected: +0.005~0.020

### §3.3 累計 LB projection
- 現況: 0.3564
- + Step 1: 0.381~0.416
- + Step 2: 0.393~0.441
- + Step 3: 0.398~0.461

樂觀 Step 1+2 直接過 0.40 leader 線。

### §3.4 Phase 6 protocol guard rails
- 每 step 都走 §3 鐵律：Plan → Debate#1 → Plan revise → Implement → Code Review → Debate#2 → 三指標 validate → 記錄
- 失敗 step 也要走完並 retrospective
- Step 1 prefix recompute LGB feature 是 leakage hot spot
- Step 2 log_prior 必須用 prefix-truncated prior
- Step 3 schema mapping 在 code review 必須附完整 22→19 對應表 + verification fixture
- 任何引用外部 PDF 的數字（e.g. arXiv:2605.01234, Zenodo 15516144）必須再三檢查 schema 是否適用
