# Deep Research Briefing — Sgp AUC Ceiling Mystery (2026-05-07)

## TL;DR

We're stuck at test AUC ≈ 0.58 for predicting `serverGetPoint` (rally-level binary outcome) on a table tennis next-stroke competition. The leaderboard #1 has implied test AUC ≈ 0.80 — a +0.22 gap that we cannot bridge with our current methods. We need outside expertise on:

1. **Is AUC 0.80 actually achievable on this task** given the data structure, or is the leaderboard somehow artificial?
2. **What mechanisms can push from 0.58 to 0.80** that we're missing?

This briefing details task setup, what we've tried, our ceiling diagnosis, and specific theories needing your evaluation.

---

## Task Setup

**Competition**: Table tennis "next-stroke prediction" + rally outcome.

**Score formula**: `0.4 × Macro-F1(actionId) + 0.4 × Macro-F1(pointId) + 0.2 × AUC(serverGetPoint)`

- `actionId`: 19 classes (stroke type for next stroke)
- `pointId`: 10 classes (court zone for next stroke)
- `serverGetPoint`: rally-level binary (whether the rally's server won that point)

**Training data**: 14,995 rallies, 84,707 strokes, 166 unique players.
- Each rally has multiple strokes; mean rally length N=5.65 strokes.
- Each rally has a single `serverGetPoint` (constant across rally's strokes).

**Test data (released 2026-05-06, replacing earlier version)**: 1,845 rallies, 5,668 strokes, 71 players (31 unseen by train, 43.7%). Mean visible length L=3.07 strokes per rally. **No `serverGetPoint` column — must predict it.**

The earlier test had `serverGetPoint` directly visible — competitors got AUC ~0.998 free by copying it. New test removes that, so AUC must be modeled.

## Leaderboard Context

**Pre-data-change** (free sgp): top scores ~0.45 (action+point F1 ~0.6, AUC ~0.998).

**Post-data-change** (today, 2026-05-07):
- #1: 0.3985 → implied AUC ≈ **0.80** (assuming similar action+point F1)
- Us (#14): 0.3546 → implied AUC ≈ 0.58
- All competitors dropped, but #1 dropped less than us

The +0.22 AUC gap between us and #1 is the mystery.

## Our Sgp Predictor Pipeline

We tried 4 distinct approaches:

### 1. Bayesian per-server winrate (κ-smoothed)
- Per-train-player: smoothed winrate when serving
- Symmetric: average with (1 - receiver's loserate)
- κ grid {5, 10, 20, 50}, sex-marginal cold-start
- **Holdout AUC 0.524** (basically random + tiny signal)

### 2. LGB with hand-crafted rally features (Q)
- 39 features: Bayesian winrates, score state, sex/numberGame, last/prev stroke attributes, action-group counts within visible rally
- **Critical fix: prefix augmentation** — for each train rally with N strokes, generate samples for k=1..N-1 prefixes (otherwise `last_strike_parity = N % 2` perfectly leaks sgp via train rally completion)
- Match-disjoint StratifiedGroupKFold(N=5, groups=match_id), per-fold winrate from non-holdout pool
- **Holdout AUC 0.6451** (per-rally aggregated over 5 prefixes)
- **Test AUC ≈ 0.58 (LB confirmed)**

### 3. LGB + LSTM features (B and "deep" variants)
- B: + LSTM action+point softmax (29 dims). Holdout AUC 0.6067 (worse).
- Deep: + LSTM trunk_out (128 dims). Holdout AUC 0.5904 (worse).
- LSTM was originally trained for next-stroke prediction (action+point); its hidden states don't carry sgp signal as standalone features.

### 4. LGB + player ELO + cross features (D)
- + ELO ratings (init=1500, K=24, computed from train rallies in chronological order)
- + cross terms: winrate × score_diff, winrate × ctx_len, score_diff × ctx_len
- **Holdout AUC 0.6490** (+0.004 over Q)
- Q+D 50/50 ensemble: **Holdout AUC 0.6510** (best LGB-only)
- **Test AUC ≈ 0.584 (LB confirmed)**

### 5. LSTM with sgp aux head retrain (H)
- Modified LSTM: added `head_sgp` (linear: trunk → 32 → 1)
- Loss: 0.5 × CE_action + 0.5 × CE_point + 0.1 × Aux + 0.3 × BCE_sgp
- Joint training: trunk learns sgp signal directly (not frozen-feature extraction)
- **OOF per-rally aggregated AUC 0.7706** ⭐ (per-sample only 0.5836)
- **OOF per-k single-prefix AUC: k=1 0.51, k=2 0.54, k=3 0.61, k=4 0.58, k=5 0.60**
- Multi-prefix test ensemble (k=1..N_visible avg, 5-fold avg)
- Q+D+H 0.05/0.05/0.9 ensemble: holdout AUC 0.7732
- **Test AUC ≈ 0.573 (LB confirmed)** — REGRESSED from Q+D!

The OOF→test gap of 0.7706 → 0.573 (= -0.198) was massive. Diagnosis: per-rally aggregated OOF was an artifact of averaging 5 prefixes; test only has 1 visible prefix per rally.

## Score Math

Score = 0.4 × F1_a + 0.4 × F1_p + 0.2 × AUC

**Action+point F1** is saturated (32 failed variant attempts in our framework history; OOF F1_a=0.368, F1_p=0.236, sum 0.604). Test F1 sum likely 0.59-0.60 (small gap from match-disjoint OOF→test).

**AUC contribution at our current ship (LB 0.3546)**:
- Action+point ≈ 0.4 × 0.595 = 0.238
- AUC ≈ 0.20 × 0.58 = 0.116
- Total ≈ 0.354 ✓

**For #1 LB 0.3985 (assuming similar action+point)**:
- Action+point ≈ 0.238
- AUC contribution = 0.3985 - 0.238 = 0.160
- AUC = 0.160 / 0.20 = **0.80**

The arithmetic strongly suggests #1 has test AUC ≈ 0.80.

## Why We Think We're Capped at 0.58

### Cold-start / unseen-player drag
- 31 of 71 test players (43.7%) unseen by train → no winrate info
- Sex-marginal fallback (P(sgp=1 | sex) ≈ 0.55) is weak
- Cold-start formula: AUC_overall ≈ 0.81 × AUC_seen + 0.095 (where 0.095 is the unseen-pool's 0.5 contribution)
- Even with AUC_seen = 0.7, overall ≤ 0.66

### Information-theoretic ceiling on per-rally outcome
- Each rally is a single binary outcome of a stochastic match
- Even with perfect knowledge of player skills, intrinsic randomness limits AUC
- Our best per-prefix AUC across all variants peaked at 0.61 (single sample, not aggregated)

### Train-test asymmetry on visible context
- Train rallies: visible = full rally (we know how it ended)
- Test rallies: visible = truncated prefix (continuation hidden)
- Train's `last_strike_parity = N % 2` is a perfect sgp predictor; test's `last_visible_parity` is uninformative
- Prefix augmentation breaks this, but the underlying information asymmetry remains
- Maybe test rallies are cut at SYSTEMATIC points (e.g., always before decisive stroke) — if so, certain features could correlate with sgp post-cut, but we don't know the cut policy

## Theories Needing External Evaluation

### Theory A: AUC 0.80 requires external data
- Player rankings from public TT databases (ITTF rankings, etc.)
- If #1 augmented training with external player skill ratings, that bypasses cold-start
- **Question**: Is this kind of external data common in domain literature?

### Theory B: AUC 0.80 requires specific sequence-to-sequence model
- LSTM/Transformer trained to predict full continuation of rally (next 5-10 strokes)
- Derive sgp from predicted continuation (e.g., who hit the predicted last stroke)
- Different from our current "predict sgp directly from prefix"
- **Question**: Has this approach been used in racquet sports temporal prediction?

### Theory C: AUC 0.80 requires deep feature engineering we missed
- Specific features that capture rally outcome better than aggregate winrate
- Examples: head-to-head record, momentum (last-N-rallies winrate), set/game pressure
- **Question**: What are the canonical "rally outcome" features in sports analytics literature?

### Theory D: AUC 0.80 isn't real / artificial top
- Maybe leaderboard top is overfitting to public LB via repeated submission
- New test split has 1845 rallies → AUC SE ≈ 0.012, so 0.80 ± 0.024 95% CI
- Top could be at 0.78 rather than 0.80
- **Question**: Public LB overfitting in 1.5k-test competitions — typical?

### Theory E: AUC 0.80 requires LSTM specifically for sgp (no multitask dilution)
- Currently testing this (V5Z_SGP_ONLY=1 retraining, ~1.5h)
- Trade off action+point F1 (saturated anyway) for sgp capacity
- Conservative estimate: +0.03-0.08 AUC over our current 0.58

## Specific Asks

1. **Validate or refute Theory D**: For a 1845-rally test set, is AUC 0.80 self-consistent or possibly leaderboard noise?

2. **Beyond LSTM-with-sgp-head**: Are there published methods for racquet-sport rally-outcome prediction with AUC > 0.7 on similar-scale data (10k-100k rallies)? Architectures, features, training tricks?

3. **Cold-start mitigation**: Standard approaches when 44% of test entities are unseen (besides external data)? Domain adaptation? Meta-learning?

4. **Structural priors**: Is there a way to encode "table tennis rally physics" (stroke alternation, scoring rules, fatigue, etc.) as inductive bias to bypass our generic LSTM?

## Our Current Status

- **Best ship**: Sub 2 LB 0.3546 (Q+D 50/50 ensemble)
- **Sub 3 (Q+D+H)**: LB 0.3524 (regressed; OOF was misleading)
- **Daily LB submissions**: 3 per day, used today
- **Background**: pure-sgp LSTM (V5Z_SGP_ONLY=1) training s42 (~1.5h ETA)
- **Tomorrow's plan**: ship pure-sgp LSTM if it improves, then incorporate research findings

If you have specific approaches to evaluate (e.g., "try this transformer config" or "use this loss function"), please be concrete: cite refs, give expected AUC range with reasoning, identify failure modes.
