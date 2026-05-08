# Table Tennis Next-Stroke Prediction — Deep Research Briefing

**Purpose**: This document is a self-contained briefing to share with another AI (Claude/GPT/Gemini deep research mode) to get external perspectives on improvement directions. We are currently rank #4 on the leaderboard with diminishing returns from local experimentation.

**Ask**: What approaches, techniques, or framings might top teams be using that we haven't considered?

---

## 1. Competition Task

Predict the next stroke in a table tennis rally given the previously observed strokes.

For each test rally, given strokes 1..n-1 (the visible prefix), predict:
- `actionId` — stroke type (19 classes: e.g. drive, smash, push, serve types)
- `pointId` — landing position (10 classes: 9 zones + 0=null/out)
- `serverGetPoint` — whether the server eventually wins this rally (binary, rally-level constant)

**Scoring**: `Score = 0.4 × Macro-F1(action) + 0.4 × Macro-F1(point) + 0.2 × AUC(serverGetPoint)`

**Critical scoring detail**: `serverGetPoint` is a rally-level constant and appears unchanged for every stroke of the same rally. It is visible in test data. Simply copying it gives AUC ≈ 0.998 (effectively a free 0.2 of the score). The real competition is on the two Macro-F1 metrics.

**Macro-F1 detail**: We use sklearn's default macro F1 (averages over classes present in `union(y_true, y_pred)`) — i.e., it does NOT force `labels=range(N)` to include classes that never appear. The provided baseline code does the same, so we believe the leaderboard scoring follows this convention.

---

## 2. Data Characteristics

### Sizes and shapes
- **Train**: 14,995 rallies, 84,707 strokes
- **Test**: 1,236 rallies, 3,589 strokes
- Mean strokes per rally: **train 5.65, test 2.90** (test rallies are ~half as long)
- **Test length distribution**: L=1 (32%), L=2 (24%), L=3 (17%), L=4 (10%), L=5 (7%), L≥6 (11%)
- Each train rally is augmented to N-1 prefix→target samples → ~70k OOF training samples

### Categorical features per stroke
- `strikeId` (1=serve, 2=receive, 4=3rd-stroke-onward, 8=missing, 16=timeout)
- `handId` (1=forehand, 2=backhand, 0=other)
- `strengthId` (1=strong, 2=medium, 3=slow)
- `spinId` (1=topspin, 2=backspin, 3=no-spin, 4=side-top, 5=side-back)
- `pointId` (1-9 = 3×3 grid on opponent's side, 0=missing/out)
- `actionId` (1-7 attack, 8-11 control, 12-14 defense, 15-18 serve types)
- `positionId` (1=left, 2=middle, 3=right; only meaningful for first 2 strokes)

### Per-rally context
- `gamePlayerId` — main view player who hit this stroke
- `gamePlayerOtherId` — opponent
- `scoreSelf`, `scoreOther`
- `sex` (1=male, 2=female)
- `match` (match identifier), `numberGame` (game number within match), `rally_id`

### Critical distribution shifts (the structural problem)
1. **Player overlap**: 166 train players, 63 test players, **23 test players (37%) NEVER in train**. These 23 unseen players appear in 570 test rallies (46%).
2. **Match disjointness**: 216 train matches, 55 test matches, **0% overlap**. Every test match is a different event.
3. **Length asymmetry**: train mean L=5.65 vs test mean L=2.90 (test much shorter; verified that calibration weighting by L doesn't help, see §4).
4. **Class 15-18 (serve actions) never appear as next-stroke targets** in train OOF, because serves are always stroke 1 (start of rally), never targets of "predict next stroke".

---

## 3. Our Current Best Solution (V5Z bag2)

**Leaderboard score: 0.4285 (rank #4)** as of submission. OOF cross-validation: 0.4816.

### Architecture

**LSTM submodel** (V5Z):
- Input: 7 categorical features per stroke (embedded 20d each) + 3 numeric (scoreSelf/Other/strikeNumber/50) + position embedding
- Player embedding: 16d × 3 (server, receiver, next-hitter), dropout 0.3
- Encoder: BiLSTM 192d × 2 layers, dropout 0.3
- Pooling: Multi-head attention (4 heads) + last hidden state
- Action head: MLP(128→64→19)
- Point head: MLP(128 + 19 → 64 → 10) — feeds in softmax(action) (sequential head)
- Aux head: per-position next-stroke prediction using FORWARD-only LSTM states (causal-safe)
- Loss: 0.5×CE_action + 0.5×CE_point + 0.1×Aux, class-weighted, label_smoothing=0.05
- Train: AdamW lr=3e-4 wd=1e-4, CosineAnnealingWarmRestarts, 40 epochs early-stop on OOF score
- Teacher forcing: tf_p decays 0.5 → 0 over 40 epochs

**LGB submodel**: 49 hand-crafted features
- Context length, next stroke number, sex, numberGame
- Score state: scoreSelf, scoreOther, score_diff, score_sum, is_deuce, game_point flags
- Last/prev/serve stroke: handId, strengthId, spinId, pointId, actionId, positionId, action_group
- Action group counts in prefix (attack/control/defense)
- Point depth/side counts in prefix
- Cross features: last_action × pointId, last_hand × pointId, etc.
- Player IDs (raw, both sides)
- Note: `serverGetPoint` is REMOVED from features (would directly leak rally outcome)

**Ensemble**: `ens = α × LSTM + (1-α) × LGB`
- Action: α = 0.64
- Point: α = 0.50

**Calibration**: Plug-in additive log-bias (Koyejo et al. 2014)
- `prediction = argmax_k (log p_k + b_k)`
- Coordinate descent on b_k, grid step 0.05, range [-2.5, 2.5], 4 rounds
- Done on OOF macro-F1 directly (no surrogate)

**Bagging**: Seeds {42, 1337}. Seed 2024 dropped (single CV 0.4765 vs 0.4788 for s42 and 0.4785 for s1337; including it hurt the bag).

---

## 4. What We've Tried and Why Each Failed (18 verified failures)

### Architecture/loss variants (14, all regressed in OOF)
| Variant | Result | Why |
|---|---|---|
| Add seed 2024 to bag | CV -0.0007 | Weak seed dilutes |
| BiGRU instead of LSTM | CV -0.0017 | Same inductive bias, no diversity |
| BiTransformer encoder | CV -0.0033 | Need 1M+ samples; sequences too short (mean 5.65) |
| Larger LSTM (256d/3L, 50 epochs) | CV ~-0.005 + OOM | 70k samples saturates 192d/2L already |
| Focal loss γ=2.0 | CV -0.0027 | Double-suppresses majority with existing class weights |
| Soft Macro-F1 loss (70% CE + 30% SF) | CV -0.0005 to -0.0017 | Redundant with plug-in calibration |
| LGB serve-class output mask | CV -0.0001 | LSTM already masks serves; ensemble argmax unaffected |
| Coarse calibration grid (0.05/0.10) | CV -0.0008 | Fine grid (0.01/0.05) is meaningful improvement |
| Forced labels=range(N) calibration | CV ~-0.05 | class 17/18 absent from OOF labels, gets F1=0 |

### Targeted attempts post-V5Z (4, all regressed in LB)

#### Attempt 1: Length-stratified calibration (testweighted)
Re-fit α and bias using sample weights matching test L distribution.
- Result: α went from (0.64, 0.50) to (0.76, 0.63). LB **-0.0037**.
- We then ran a "fair comparison": principled importance-weighted OOF showed α optimum is essentially unchanged (0.66, 0.50). Test-L-weighted OOF ≈ uniform OOF (Δ=0.0001).
- **Conclusion**: Length distribution shift is NOT a meaningful gap source despite the visible asymmetry.

#### Attempt 2: Step 0 cheap baseline (in-prefix observed player stats)
Added 6 LGB features: srv/rcv stroke count, attack count, forehand ratio (computed from prefix only, zero leakage).
- Result: testw OOF -0.0001, full OOF -0.0004. **Failed gate.**
- **Conclusion**: In-prefix observed player signal is redundant with existing 49 LGB features.

#### Attempt 3: A v2 Full (per-player historical stats with per-fold leakage handling)
~31 new LGB features per rally:
- Per-player aggregated stats (action/point/winrate/handedness/avg-rally-length distributions) for both server and receiver
- Sex-conditional cold-start fallback for 23 unseen test players
- `srv_seen`, `rcv_seen` binary flags + interaction
- Per-fold computation: `player_stats_f` computed only from train fold rallies (excluded val fold)
- num_leaves swept on fold 1, picked 127

OOF result: **CV 0.4858 (+0.0042)**, F1_action 0.4396 (+0.0007), F1_point 0.2748 (+0.0098), CV_testw 0.4860.
- k=1 length bucket action F1 +0.0352 (huge — appeared to validate the player axis hypothesis for L=1 rallies which are 32% of test)
- Calibration shifted α_action 0.64 → 0.53, α_point 0.50 → 0.50

**LB result: 0.4044 (-0.0241 vs baseline 0.4285)**. Catastrophic.

OOF/LB gap blew from -0.053 to -0.0814. The +0.0042 OOF gain was illusion: the per-fold-leakage-safe scheme **still leaked** because KFold splits rallies (not players), so train fold and val fold have ~99% player overlap. OOF measured "performance on mostly seen players", which doesn't reflect the 46% unseen-player test condition.

#### Attempt 4: Conditional Ensemble (route by seen/unseen)
Hypothesis: AV2's bad predictions are concentrated in 46% unseen-player test rallies. Route those to baseline V5Z, keep AV2 for 54% seen.
- Routed 570 of 1236 test rallies to baseline. 191 of those got different predictions (the rest had same argmax under both calibrations).
- **LB result: 0.4044 (identical to AV2)**.
- The change had ZERO measurable effect.

This was the diagnostic moment that broke the "unseen player is the main problem" theory.

---

## 5. Diagnosed Root Cause (Updated)

The bottleneck is **NOT**:
- Length distribution shift (verified by fair comparison)
- Class 17/18 forced labels (baseline code uses default macro)
- Unseen-player count alone (verified by Conditional Ensemble producing identical LB)
- Calibration choices (saturated)
- Architecture (14 variants all regressed)

The bottleneck **IS** (best current understanding):

**Cross-event style drift** — even for SEEN players, their behavioral statistics from train tournaments do not transfer cleanly to test tournaments. 100% match disjointness means every test match is a different event with different conditions (court, opponents, timing, possibly years apart). Player-style features are biased estimators across events.

This is why:
1. Adding player_stats helped OOF (per-fold validation has high player overlap → features look informative)
2. Hurt LB uniformly (test players, even seen ones, behave differently in new events)
3. Conditional routing by "seen status" didn't help (the issue isn't unseen-ness, it's per-event drift)

**Estimated LB ceiling**: We computed perfect-calibration ceiling at ~0.448 assuming length distribution was the main gap. After learning length isn't the main gap and cross-event drift is unbounded, this ceiling estimate is suspect. Top teams may be at 0.45-0.46.

---

## 6. Constraints

- **Compute**: Single RTX 3060 6GB. Each LSTM training (5-fold) takes 30-40 min per seed.
- **LB budget**: ~5 submissions per day (currently exhausted today).
- **External data**: Likely not allowed (competition rule).
- **Time budget**: Open-ended for the user but currently rank #4, want top 3.
- **Code base**: Python + PyTorch + LightGBM + sklearn.

---

## 7. Things We Have NOT Tried

- **Test-time adaptation (TTA)** — small gradient steps on visible test prefix to adapt LSTM
- **Self-supervised pretraining** — masked-stroke prediction on train+test sequences (corpus only 16k sequences though)
- **Pseudo-labeling** — use confident predictions on test as additional training signal
- **Match-style proxy features** — compute features only from observations within the same test match (no need for player history; works for unseen)
- **Player clustering with style inference** — cluster train players into K=5-8 style groups; for unseen test players, infer cluster from their visible-in-test stroke patterns
- **Adversarial validation re-weighted OOF** — train classifier P(test|x), use as importance weight for calibration (debate suggested it would degenerate due to match disjointness)
- **Group-aware cross-validation** — split folds by player or by match instead of stratified by class (would create OOF that's a more honest LB estimator)
- **Different problem framing** — predict full distribution rather than argmax; ordinal regression on point grid; multi-task with intermediate targets
- **Domain knowledge encoding** — table tennis tactical theory (e.g., serve-receive theory, attack-defense transitions, point-construction patterns)
- **Stacking with cross-validated meta-learner** — non-linear ensemble combiner instead of scalar α blend
- **Test-time augmentation** — predict multiple variants of each test rally (e.g., reverse stroke order? not applicable here probably)
- **More aggressive seed bagging within current architecture** (more than 2 seeds, but selected for diversity)

---

## 8. Specific Asks for Deep Research

1. **What techniques have winners of similar competitions used?** Specifically:
   - Sports next-event prediction (tennis, badminton, table tennis if any)
   - Small-imbalanced-multi-class macro-F1 maximization
   - Sequence prediction with severe distribution shift
   - Cross-event/cross-tournament generalization in sports analytics

2. **Are there known recipes for cross-event style drift?** This is a real research problem (athlete-style stationarity is a non-trivial assumption), and we haven't found a clean published solution.

3. **Is there a way to use the test set as unlabeled data**? Pseudo-labeling, semi-supervised, transductive learning — what works at this scale (16k sequences, 19+10 classes)?

4. **Are we missing a critical evaluation strategy?** Group-by-player or group-by-match KFold would give a more honest OOF estimator. Worth implementing?

5. **Domain knowledge encoding** — is there published table tennis tactical knowledge that maps to features we should include? (e.g., the "third ball attack" doctrine, serve-receive grids, etc.)

6. **For point prediction (the F1_point ceiling at ~0.27 in OOF)**: are there geometric or spatial priors specific to a 9-zone court that would help? We treat point as a flat 10-class problem; structurally it's 3×3 grid + null.

7. **Last-stroke F1 collapse** — at the rally-ending stroke, our F1_point drops to 0.075 (vs ~0.22 mid-rally). High entropy in winning shots. Anything beyond the obvious (more data) for this?

8. **Calibration**: We use plug-in additive log-bias. Are there better post-hoc methods for macro-F1 specifically that we should consider? (Decision-threshold optimization with learned thresholds?)

---

## 9. Summary One-Liner

We have a saturated BiLSTM+LGB+plug-in-calibration framework at LB 0.4285 (rank #4) on a small-data (15k rallies, 70k augmented samples), severe-distribution-shift (100% match-disjoint, 46% unseen players), short-sequence (mean 2.9 strokes test) next-stroke prediction task. 18 documented attempts have failed to improve LB. We need fresh ideas for cross-event generalization or alternative framings.

**Please suggest 3-5 concrete directions with brief mechanism explanations and expected ROI.**
