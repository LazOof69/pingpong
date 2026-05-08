# Deep Research Briefing — Table Tennis Next-Stroke Prediction (2026-05-05)

**Purpose**: hand-off document for external AI consultation. We have hit a framework saturation ceiling after 7 consecutive Plan-stage Debate-killed attempts (28 failed variants total). Need fresh perspective on what direction could break through.

**Reader instructions**: read TL;DR + sections 1-3 first to load context. Sections 4-6 are exhaustive failure log + meta-lessons. Section 7 is the open question.

---

## TL;DR

- **Competition**: AI CUP table tennis next-stroke prediction. Score = 0.4×Macro-F1(action, 19 cls) + 0.4×Macro-F1(point, 10 cls) + 0.2×AUC(serverGetPoint).
- **Current ship**: LSTM bag2 (s42 + s1337) with plug-in additive bias calibration. **LB 0.4374, rank #3**.
- **Framework**: BiLSTM (hidden=192, 2 layers) + LightGBM ensemble, `StratifiedGroupKFold(N=5, groups=match_id)` (match-disjoint CV), plug-in additive bias calibration on macro-F1.
- **Saturation diagnosis**: OOF saturated at 0.444-0.446 across 7+ independent training-objective perturbations. **Framework saturation appears to be architectural (information regime ceiling)**, not plan-quality.
- **What's blocked**:
  - All 5 directions in our prior PROGRESS §5.1 candidate list: Mirror / Logit Adjustment / Pseudo-label / BiTransformer / SSL / Player-style autoencoder / Length-importance-weighting — ALL killed
  - Player feature directions blocked: competition de-identifies train+test, no external player ELO can be joined
  - Length-stratified post-hoc calibration already empirically failed (LB -0.0037)
  - Same-architecture variants converge to Pearson 0.91+ (T's actual measurement) → ensemble math caps lift at +0.001-0.002
- **Open question**: Given the constraints, what is a fundamentally NEW direction that escapes the (BiLSTM, 70k strokes, 29-class macro-F1, 29-param plug-in calibration) saturation pattern?

---

## 1. Competition Setup

### Task
Given a rally's first n-1 strokes (the "prefix"), predict:
- **actionId** (next stroke type, 19 classes — drives, smashes, serves, defensive shots, etc.)
- **pointId** (next stroke landing position, 10 classes — 9-grid + "out")
- **serverGetPoint** (rally outcome, binary, AUC scored)

### Scoring
`Score = 0.4 × Macro-F1(action) + 0.4 × Macro-F1(point) + 0.2 × AUC(serverGetPoint)`

`serverGetPoint` is rally-level constant, visible in test → directly copied (~0.998 AUC). The actual battle is the two Macro-F1 metrics.

### Rules (relevant constraints)
- 3 LB submissions per day, daily reset
- Custom data + open-source resources allowed (with disclosure)
- **Test set is de-identified** — match IDs and player IDs in test do NOT correspond to identifiable real-world entities. Reverse-matching to public records is forbidden.
- Final report required for top 25% / >baseline teams (judged for awards)

### Data
- **Train**: 14,995 rallies / 84,707 strokes / 166 unique players / 216 matches / mean L=5.65
- **Test**: 1,236 rallies / 3,589 strokes / 63 unique players (40 seen + 23 unseen, **36.5% unseen**) / 55 matches / mean L=2.90
- **Train/test match disjoint**: 0 match overlap

### Critical distribution shift
- Train mean L=5.65, **test mean L=2.90** (test rallies are systematically shorter)
- 56% of test rallies have L≤2 (32% L=1, 24% L=2)
- Test prefix length distribution shows up-weighting of short prefixes:

| Prefix length k | Train | Test | w(k) = q_test/q_train |
|---|---|---|---|
| k=1 | 21.5% | 34.4% | 1.601 |
| k=2 | 18.8% | 23.3% | 1.240 |
| k=3 | 15.1% | 15.1% | 0.997 |
| k=4-5 | 18.8% | 15.4% | 0.818 |
| k=6-10 | 17.5% | 9.6% | 0.551 |
| k=11+ | 8.3% | 2.1% | 0.260 |

---

## 2. Current Framework

### 2.1 Architecture (LSTM side)
```
Input per stroke:
  - Categorical embeddings: actionId (19 cls), pointId (10), handId (3),
    spinId (6), positionId (4), strikeId, strengthId
  - Player embedding (gamePlayerId, 16-dim, supervised by next-stroke CE)
  - Numerical features: stroke index, prev_action_xpoint, etc.

BiLSTM: hidden_dim=192, num_layers=2, dropout=0.3, MAX_SEQ_LEN=50
Static features (49-dim) → MLP → fusion → main heads
Heads: actionId classifier (19), pointId classifier (10) + aux heads
Loss: weighted CE with class weights (inverse frequency ** WEIGHT_ALPHA), label smoothing
```

### 2.2 Architecture (LGB side)
- 49 hand-engineered features per (rally, prefix_position) pair
- Includes: action/point/hand/spin frequencies in prefix, server identity, last-stroke features, EB-shrunk player observations, phase indicators
- Standard LightGBM multiclass training
- 5-fold per match-disjoint splits

### 2.3 Ensemble + calibration
1. LSTM probs and LGB probs averaged with α-blend (LSTM weight optimized on OOF, ~0.85 for action, ~0.98 for point)
2. **Plug-in additive bias calibration**: per-class scalar bias shifts logits to maximize OOF macro-F1
   - 19 free params for action + 10 for point = **29 free calibration params**
   - Theoretically optimal for macro-F1 under per-class label shift (Koyejo et al. 2014)
3. Final argmax → submission

### 2.4 OOF/LB gap (empirical)
| Submission | OOF | LB | gap |
|---|---|---|---|
| LSTM bag2 (current ship) | 0.4414 | 0.4374 | -0.004 |
| 6-way mt2base4 (heterogeneous bag) | 0.4456 | 0.4367 | -0.009 |
| 4-way bag md+EB | 0.4413 | 0.4341 | -0.007 |
| **base bag3 (homogeneous LSTM)** | **0.4441** | **0.4287** | **-0.015** |
| **T s42 single (BiTransformer)** | **0.4357** | **0.4295** | **-0.006** |

Match-disjoint OOF/LB gap is variable (-0.004 to -0.015) depending on ensemble composition. Base bag2 (s42+s1337) gives the smallest gap.

---

## 3. Saturation Diagnosis (Why We're Stuck)

### 3.1 Empirical: 7 consecutive Plan-stage KILL/abort

| # | Plan | Mechanism | Verdict | Pre-cal lift est | Post-cal lift est |
|---|---|---|---|---|---|
| 1 | LA (Logit Adjustment, Menon 2021) | Per-class logit shift via train prior | Debate-KILL | +0.005 | +0.001-0.003 |
| 2 | Mirror augmentation (TacticAI/ShuttleNet recipe) | Left-right symmetry data aug | Debate-KILL | +0.003 | +0.0005-0.0015 |
| 3 | Pseudo-label on test | Self-distillation via top-K confident test predictions | Debate-KILL | +0.003 | +0.0005-0.0015 |
| 4 | T (BiTransformer match-disjoint) | Architecture diversity | **Real-run abort** at Pearson 0.91 | — | bag lift +0.0005-0.0015 |
| 5 | SSL (Masked stroke pretraining) | BERT-style on train+test 88k strokes | Debate-KILL | +0.005 | +0.001-0.003 |
| 6 | E (Player-style autoencoder embedding) | 60-dim handcrafted stats → AE-16 → input feature | **Stage A probe KILL** | empirical Δ=-0.003 | — |
| 7 | W (Length-importance-weighted training) | Per-sample IW by `q_test(k)/q_train(k)` | Debate-KILL | <+0.001 | +0.0001 |

### 3.2 Information-theoretic argument (decisive for current direction class)

Two convergent arguments rule out further "framework-internal" perturbations:

**Argument A — Plug-in calibration absorption (Vapnik 1998 §3.2 monotonicity)**:
- Plug-in additive bias is a 29-param post-hoc corrector on macro-F1
- Any train-time logit/decision-boundary shift expressible as per-class additive is FULLY absorbed
- 6/6 prior plans had pre-cal lift compressed by 70-85% post-calibration
- Empirical anchor: SSL post-calibration gain across 3 references (Cui 2021, Hong 2021, Cao 2019) averages +0.002 ± 0.001

**Argument B — Length-stratified post-hoc calibration EMPIRICALLY failed**:
- We ran post-hoc length-stratified calibration with up to 6 (length buckets) × 29 (classes) = 174 free correction params
- Result: **LB -0.0037 vs ship**
- This is a STRICTLY MORE EXPRESSIVE corrector than any training-time IW (Plan W)
- Information-theoretic monotonicity: if more-expressive corrector failed, less-expressive cannot succeed

**Argument C — Self-diagnosis of OOF/LB gap source**:
> "OOF/LB gap completely explained by CV protocol bias (player overlap), cross-event drift secondary"
- Length is NOT the primary gap source per our own diagnostic
- Player-overlap fix (match-disjoint CV) already implemented → gap reduced from -0.053 to -0.004
- No further "low-hanging" CV protocol fix available

### 3.3 Bayesian prior on framework-internal mechanism class

- 6 independent training-objective perturbations (LA, Mirror, Pseudo, T, SSL, E) all converged on +0.001-0.003 post-cal lift
- Laplace successor rule: P(8th plan succeeds | 0/6 success) = 1/8 = **12.5%**
- Each plan claimed different mechanism but hit same architectural ceiling
- The convergence is no longer evidence "about specific plans" — it's measurement of framework saturation

### 3.4 Pearson constraint (Plan T empirical lesson)

Plan T (BiTransformer match-disjoint) actually trained:
- T s42 single OOF = 0.4357 (≈ LSTM s42 0.4361, surprise — not crashed)
- **Pearson(T, LSTM) = 0.9147 action / 0.9507 point**
- Same architecture variants converge to high Pearson 0.85+ (per Lakshminarayanan 2017, Wenzel 2020, Fort 2019)
- Caruana 2004 ensemble theorem: with high Pearson, ensemble lift bounded by (1-ρ)/2 × component error
- For our setting: ensemble lift cap ≈ +0.001-0.002, well below +0.010 ship gate

---

## 4. Comprehensive Failure Log (28 variants)

### 4.1 Rally-KFold era (V5Z §2.1-2.8, 8 variants — superseded by match-disjoint protocol)
1. LSTM seed=2024 (weak seed, dilutes bag)
2. BiGRU (worse than BiLSTM at 70k scale)
3. Larger LSTM (HIDDEN=256, 3 layers, 50 epochs — saturates earlier)
4. **BiTransformer encoder** (HIDDEN=192, 2 layers, 4 heads — Score 0.4755 vs LSTM 0.4816)
5. Focal Loss γ=2.0 (overlaps with class weights, double-suppresses majority)
6. Soft Macro-F1 Loss (overlaps with plug-in calibration, batch-level approximation noisy)
7. LGB serve mask (LSTM already masks, post-process noise)
8. Coarse calibration grid (fine grid +0.0008, only positive late-experiment)

### 4.2 Match-disjoint era — empirical failures
9. Length-stratified post-hoc calibration (LB **-0.0037**)
10. **A v2 historical player_stats** (rally-KFold OOF +0.004 → **LB -0.024**, per-rally LOO not per-player LOO)
11. In-prefix observed Step 0 (OOF Δ=-0.0001, redundant)
12. Conditional ensemble routing seen/unseen (LB unchanged — refuted "unseen is gap source" hypothesis)
13. EB shrinkage features on LGB (Δ=0, LGB only 2-15% ensemble weight)
14. Phase + EFOS LGB features (Δ=0)
15. Class suppression on calibration (Δ=0, calibration saturated)
16. 4-way bag md+EB (LB 0.4341, -0.003 from md bag2)

### 4.3 Match-disjoint era — Plan-stage Debate-killed
17. **§2.20 Multi-task aux heads (MuLMINet recipe)**: MT s42 single OOF +0.0025 → 4-way ensemble +0.0008. Backbone-shared aux heads collapse in ensemble.
18. **§2.21 6-way bag mt2base4 ship (2026-05-05)**: OOF 0.4456 (+0.0042) → **LB 0.4367 (-0.0007)**. Re-confirmed ship gate +0.010, not +0.005.
19. **§2.22 Logit Adjustment (Menon 2021)**: Plug-in 29-param strictly dominates τ·log(π) 1-param on macro-F1 surface.
20. **§2.23 Mirror augmentation**: 3 FATAL — pointId receiver-frame corruption + L=1 attention degenerate (later refuted by T data) + reference class +0.001-0.003 below gate.
21. **§2.24 Pseudo-labeling on test**: top-K confidence gating filters OUT covariate-shifted samples (re-injects train distribution); Mobahi 2020 self-distillation theorem caps lift; A v2 disguise leakage path.
22. **§2.25 BiTransformer match-disjoint (T)** — actually run: OOF 0.4357 ≈ LSTM but **Pearson 0.9147/0.9507** trigger hard abort. LB 0.4295.
23. **§2.26 base bag3 LB regression**: framework-level finding — homogeneous LSTM bag size > 2 has negative LB transfer (OOF +0.0027 → LB -0.0087).
24. **§2.27 SSL (Self-Supervised Masked Pretraining)**: bidirectional MLM mathematically wrong for causal next-stroke (Devlin 2019, BART, T5 all show 5-15% downstream loss); plug-in absorption + 88k too small.
25. **§2.28 Plan E (Player-Style AE Embedding)** — Stage A probe: supervised player_embed vs 12-dim stats R²=-0.122 (uncorrelated); LGB-only match-disjoint AV2-style aggregate stats blended Δ=-0.0028 < +0.002 gate.
26. **§2.29 (this debate) Plan W (Length-Importance-Weighted Training)**: Vapnik monotonicity (post-hoc length-strat-cal already failed) + plug-in absorption. Predicted post-cal +0.00014.

(Variants 1-8 + 9-16 + 17-26 = 26 unique entries; with 27th and 28th being the immediate prior pre-Plan-W; total currently logged in V5Z_FINAL_REPORT §2 is 28.)

### 4.4 Multi-task auxiliary heads detail (§2.20)
- Action / point / hand / spin / strength multi-head training (MuLMINet recipe, Wang 2022)
- s42 + s1337 trained: single +0.0025 vs base
- 3-way ensemble (MT s42 + base bag2): +0.0025 vs base bag2
- 4-way ensemble (MT s42 + base bag3): **+0.0008 vs base bag3**
- Saturated, not shipped

---

## 5. Meta-Lessons (19 accumulated, drive Plan filtering)

1. **rally-KFold OOF inflates by ~0.05 due to player overlap** — must use match-disjoint
2. Match-disjoint protocol was the largest single intervention (LB +0.0089 from 0.4285 → 0.4374)
3. Plug-in additive bias calibration is the gold-standard macro-F1 corrector (Koyejo 2014)
4. **Bag selection > bag size** for homogeneous bags
5. **Cross-domain published lift ≠ same-domain transfer** (TacticAI football +3-5% ≠ table tennis applicable)
6. **pointId is receiver-handedness normalized**, not absolute spatial — any spatial transform must check semantic layer
7. **Saturation is information regime, not capacity / data volume regime** (augmentation upper bound +0.003)
8. **Bidirectional MLM pretrain is structurally wrong for causal task** (BERT vs autoregressive 5-15% downstream loss on generation)
9. **Plug-in calibration absorbs 70-85% of train-time decision-boundary shifts** on 19+10 macro-F1 (3-paper empirical anchor)
10. **5-6 independent KILL plans converge on +0.001-0.003 post-cal lift** = direct measurement of architectural ceiling
11. **Pearson > 0.85 is hard ensemble math constraint** — same-arch variants typically Pearson 0.85-0.95 (Plan T measured 0.91-0.95)
12. **Match-disjoint doesn't equal player-disjoint** — 24% of train players appear in test, 63% of test players appear in train — but their MATCHES don't overlap
13. **Per-rally LOO ≠ per-player LOO** for feature engineering (A v2 LB -0.024 lesson)
14. **Single seed → LB gap test → bag** is mandatory new candidate workflow
15. **Homogeneous bag > 2 has negative LB transfer** (base bag3 LB -0.0087 vs ship)
16. **Heterogeneous bag (cross-arch / cross-training) is the only viable ensemble path**
17. **Stage A cheap probes (~2h) save 14-20h Plan E investments** — prefer empirical KILL tests
18. **Linear independence ≠ task relevance** (Plan E supervised embed R²=-0.12 vs stats was misleading; stats themselves don't help)
19. **Same-domain reference class is mandatory** for lift estimates (sport temporal prediction reference class is sparse — most papers don't use SSL / IW / AE-on-supervised)

---

## 6. What Has Been Ruled Out (and Why)

### 6.1 Ruled out: framework-internal training-time perturbations
- **Loss reformulation** (LA, focal, soft-F1): plug-in absorbs
- **Data augmentation** (Mirror): semantic mismatch + saturation regime mismatch
- **Self-distillation / pseudo-label**: confirmation bias + self-distillation ceiling
- **Architecture diversity within BiLSTM-equivalent**: T showed Pearson 0.91, ensemble math death
- **Pretraining** (SSL): bidirectional vs causal mismatch + small data
- **Input features** (player AE, EB, A v2): A v2 LB -0.024; per-fold-aware feature regen complex; raw stats Δ=0 on LGB (empirical probe)
- **Sample reweighting** (W): post-hoc length-strat-cal already failed → training-time variant blocked by Vapnik monotonicity

### 6.2 Ruled out: external information sources
- **Player ELO from public sources**: train and test are de-identified; player IDs don't match real-world entities; cannot join external rating
- **Match metadata from public sources**: same de-identification issue
- **Reverse-matching test rallies to real games**: explicitly forbidden by competition rules
- **Team collaboration**: explicitly forbidden by competition rules

### 6.3 Ruled out: post-hoc methods
- **Length-stratified calibration**: tested, LB -0.0037
- **Class-conditional richer calibration** (e.g., temperature × scale × bias = ~38 params): tested, calibration is saturated at fine-grid additive
- **Stacking meta-learner on bag OOF probs**: high overfit risk on small OOF

### 6.4 Ruled out: bag-size scaling
- **Homogeneous LSTM seed bag size > 2**: empirical -0.0087 LB transfer (Meta-lesson 15)

---

## 7. THE OPEN QUESTION (for deep research)

**Given the framework saturation diagnosis and the constraints above, what direction has any chance of breaking the ceiling?**

We're looking for ideas that satisfy ALL these criteria:

1. **Outside the saturated mechanism class**:
   - NOT a loss reformulation, sample reweighting, or per-class adjustment (plug-in absorbs)
   - NOT a same-architecture variant (Pearson ≥ 0.85 ensemble constraint)
   - NOT a pretraining objective on the same data (information-theoretic ceiling at 70k)
   - NOT a feature derived from train rallies' aggregated player statistics (Vapnik monotonicity vs already-failed length-strat-cal)

2. **Within the data and rule constraints**:
   - 70k train strokes / 14995 rallies (small for deep learning by modern standards)
   - 19+10 macro-F1 metric (severely class-imbalanced; rare classes matter)
   - Test is de-identified — no external player/match data can be joined
   - 3 LB submissions per day with daily reset
   - Final report disclosure required for any unusual technique

3. **Has reasonable expected lift**:
   - Must plausibly clear +0.005 single-seed OOF gate (current ship gate set at +0.010 due to bag size LB risk per Meta-lesson 15)
   - Must survive plug-in calibration's 70-85% absorption
   - Cross-arch ensemble Pearson < 0.85 (the only path beyond same-architecture saturation)

4. **Is implementable in finite time**:
   - User has ~weeks before competition deadline
   - 20-40h compute budget per direction acceptable
   - Code-level specification with verification fixture mandatory (per accumulated meta-lessons)

### Specific questions for the deep research AI

A. **Are we missing a theoretical angle?** Is the framework saturation truly architectural, or is there a known technique we haven't tried that operates outside the (BiLSTM, plug-in calibration) joint constraint?

B. **Is there a way to extract NEW information from the existing 70k strokes** that hasn't been tried? Not handcrafted features (failed); not autoencoder compression (failed); not test pseudo-labels (failed). Something that operates on the data structure itself.

C. **Is the macro-F1 metric exploitable in a way we haven't seen?** All our attempts targeted "improve representation/training" — is there a meta-learning or cost-sensitive approach specific to macro-F1 that escapes plug-in absorption?

D. **What about model architectures we haven't tried that might outperform BiLSTM at 70k scale?**
   - State-space models (Mamba/S4)?
   - Graph neural networks (treating rally as a graph)?
   - Kernel methods (with hand-engineered features as kernels)?
   - Anything else from recent literature?

E. **Could we leverage the rally-as-language structure?**
   - 19+10 = 29 classes is a small vocabulary
   - Rally sequences have grammar-like dependencies
   - Are there small-vocabulary sequence modeling techniques (e.g., from low-resource MT) that apply?

F. **Are there ensemble strategies beyond what we've tried?**
   - We've ruled out homogeneous bagging, same-arch ensemble, and shared-feature ensembles
   - Can a fundamentally heterogeneous ensemble (different model classes) escape the Pearson constraint?
   - What if we deliberately train on subsamples that emphasize rare classes (per-class IW), then ensemble?

G. **Is there a competition-specific exploit we're missing?**
   - 32% test rallies are L=1 (single visible stroke = serve)
   - Predicting next stroke from L=1 prefix is essentially predicting receive given serve
   - Is there a known pattern (serve type → receive distribution) we could exploit more directly?
   - Are there known table-tennis-specific priors (ITTF rule books, professional play patterns) that could be encoded as priors?

H. **Should we accept the ceiling?**
   - If framework saturation is truly architectural, is the right move to (a) lock current ship and write final report, or (b) keep trying with vanishing P(success)?
   - What's the threshold of accumulated evidence at which "exhaust possibilities" becomes "cargo-culting"?

---

## 8. Quick reference

### Files / artifacts
- Current ship: `submissions/submission_v5z_md_advcal_bag2.csv`
- Framework: `src/train/train_v5z.py` (main), `src/train/advcal_match_disjoint.py` (calibration)
- Failure log: `docs/V5Z_FINAL_REPORT.md` §2 (28 variants)
- Per-day progress: `docs/PROGRESS_2026-05-04.md` (last sync)
- Test distribution analysis: `docs/TEST_DISTRIBUTION_ANALYSIS.md`

### Key numbers cheat sheet
- Ship LB: 0.4374 (rank #3)
- OOF saturation: 0.444-0.446
- Ship gate: OOF +0.010 (+0.005 also tested, false positive ratio too high)
- LB sample variance: ±0.005-0.010 (1236 rallies)
- Match-disjoint OOF/LB gap: -0.004 (bag2 ship), variable -0.004 to -0.015 by composition
- Train/test overlap: 0 matches, 40/63 (63%) test players seen in train
- Compute envelope per direction: 4-40h
- Failed variants total: 28
- Plan-stage Debate-killed in current session: 7 consecutive

---

*End of briefing. Reader: please respond with concrete proposals that satisfy section 7's criteria, OR with arguments that section 3's saturation diagnosis is wrong (specifying which assumption is false).*
