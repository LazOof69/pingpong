# External Resource Briefing — Closing 0.044 Score Gap

## Context (one-paragraph)

Table tennis next-stroke prediction competition. Score = 0.4 × Macro-F1(action_id, 19 classes) + 0.4 × Macro-F1(point_id, 10 classes) + 0.2 × AUC(serverGetPoint, binary). Train: 14,995 rallies / 84,707 strokes / 166 unique players. Test: 1,845 rallies / 5,668 strokes / 71 players (44% unseen by train), no `serverGetPoint` column. Each row is one stroke. **`pointId` is in receiver's body frame** (forehand/backhand-relative, mirrors for left-handed receivers).

Test rallies are TRUNCATED prefixes of full rallies — mean visible L=3.07 vs train mean N=5.65. **L=1: 28%, L=2: 26%, L≥3: 47%**. We've established `sgp = 1 - parity(N)` deterministically in train (TT rule: last hitter is the failing player) but N is hidden in test.

We currently sit at **LB 0.3564**. Leader at **0.40** (gap +0.044). Competition rules: **external data ALLOWED for training, generative AI tools ALLOWED, but reverse-mapping anonymous player IDs to real-world identities is BANNED**. Test set is de-identified.

## Our Pipeline (briefly)

**LSTM + LGB ensemble**, match-disjoint StratifiedGroupKFold(N=5, groups=match_id). Plug-in additive bias calibration on macro-F1.

After ~10 incremental feature engineering attempts (Bayesian player skills, multi-level EB, Markov bigram, NMF cold-start, handedness inference, match-level test self-supervised), our **OOF saturated at 0.819 per-rally aggregated AUC** but **LB stuck at 0.356 ± 0.0003 across 5 submissions**. Day 1 (T3 = structural priors + multi-class R prediction stacking) gave +0.0018 LB lift; Days 2-4 gave 0 LB.

**F1 components**: F1_action ≈ 0.371, F1_point ≈ 0.224 on test. **F1 has 2x weight per unit** vs AUC in the score formula. Length-stratified plug-in calibration just gave +0.005 OOF F1 lift.

## Score Math Decomposition

```
Current LB: 0.4 × 0.595 + 0.2 × 0.59 = 0.356
Leader 0.40 = 0.4 × F1_sum + 0.2 × AUC

Plausible leader scenarios:
  F1_sum 0.65, AUC 0.70 → 0.260 + 0.140 = 0.40
  F1_sum 0.70, AUC 0.60 → 0.280 + 0.120 = 0.40
  F1_sum 0.65, AUC 0.75 → 0.260 + 0.150 = 0.41
```

To bridge 0.044, we need **F1_sum +0.060 to +0.110 OR AUC +0.10 to +0.20**, OR balanced.

## Specific Questions for External Research (priority-ordered)

### Q1 (HIGH PRIORITY): Public TT stroke-level datasets we can use for pretraining

We need a TT dataset with stroke-level granularity matching our schema (action_type, landing_zone, receiver_handedness, score_state). Required:
- ≥50,000 stroke samples
- Stroke types and landing zones labeled
- Public download (competition allows external data)

**Specific asks**:
- Is there a published TT dataset like ShuttleSet (badminton, Wang et al. 2021)? ShuttleSet has 3,685 rallies with action labels — does TT have an equivalent?
- IIT Tokyo / Kasper Stage TT analytics: any open-source data?
- Chinese national / regional TT associations: do they release anonymized stroke data?
- Could we use **badminton ShuttleSet** (different sport but similar serve-rally structure) as **pretraining data** for our TT LSTM? What modifications?

### Q2 (HIGH PRIORITY): Sports outcome prediction with truncated rallies — 2024-2026 published methods

We hit AUC 0.62 ceiling under random truncation per information-theoretic analysis. Leader's 0.40 implies AUC 0.65-0.75. What methods recently achieved this?

**Specific asks**:
- Papers/Kaggle solutions 2024-2026 for **sports outcome prediction with truncated/partial-observation sequences**
- Specific architectures: Transformer with imputation, Mamba state-space, Diffusion-based completion?
- Self-supervised pretraining objectives that lift downstream sports tasks (masked stroke prediction, rally completion, etc.)
- ShuttleScorer (badminton, Wang+ 2021) achieved AUC 0.84 on RALLY-WINNER prediction with full rallies. Has anyone adapted for truncated input?

### Q3 (MEDIUM): Pseudo-label techniques that beat Mobahi self-distillation ceiling on small datasets (<100k samples)

§2.24 we killed pure pseudo-labeling on old test (Mobahi 2020 self-distillation theorem ~0 information gain). On NEW test we haven't tried. Specifically for SEMI-SUPERVISED setting with 70k labeled + 1.8k unlabeled (test) and macro-F1 metric.

**Specific asks**:
- FixMatch / FlexMatch / MixMatch variants for tabular + sequence data
- Confidence-thresholded pseudo-label with class-balanced sampling
- Co-training / multi-view pseudo-label
- Specific class-balanced SSL methods that lift macro-F1 (not just accuracy)

### Q4 (MEDIUM): Macro-F1 lift on imbalanced multi-class with rare classes

Action classes 17/18 (specific serve types) NEVER appear in next-stroke labels (rule constraint). Some rare attack/defense classes have <5% frequency. Macro-F1 is dragged by rare classes.

**Specific asks**:
- Class-balanced focal loss variants 2024-2026 for macro-F1
- Hierarchical multi-class (predict group then class) — concrete implementations
- Per-class threshold tuning beyond plug-in additive bias (what works in Kaggle competitions)
- Methods to handle "always-zero" classes properly

## What I'm Doing in Parallel (so don't suggest these)

- Length-stratified plug-in bias calibration (just done, +0.005 OOF F1 lift)
- Hierarchical action prediction (group → action) — 2-3h
- Pseudo-label LGB action+point — 2-3h
- Stacking meta-learner across our 8 sgp variants — 1-2h
- Test-time augmentation (multi-prefix avg) — 1h

## Constraints

- Implementation budget: 4-8h per direction
- Cannot reverse-map anonymous player IDs (rule violation)
- LB submissions: 3/day
- I'm comfortable with PyTorch, sklearn, LightGBM, transformers, sklearn pipelines
- No GPU memory issue (single GPU 24GB)

## Output Format Requested

For each top-3 ranked recommendation:
1. **Specific resource/method** (URL if dataset, citation if paper)
2. **Implementation steps** (concrete enough to start coding)
3. **Expected lift estimate** with reasoning
4. **Failure mode I can detect early** (1-hour kill experiment)
5. **Ranked by expected ROI** (lift / effort)

Aggressive concrete recommendations preferred over caveats.
