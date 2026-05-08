# Table Tennis Next-Stroke Prediction — Compressed Briefing

I'm rank #4 on a competition leaderboard with severe diminishing returns. Looking for fresh ideas.

## Task
Given the first n-1 strokes of a table tennis rally, predict stroke n's:
- `actionId` (stroke type, 19 classes)
- `pointId` (landing position on 9-zone grid + null, 10 classes)
- `serverGetPoint` (rally outcome, binary — but it's a rally-level constant visible in test, so trivially copy → AUC ~1.0)

**Score = 0.4·MacroF1(action) + 0.4·MacroF1(point) + 0.2·AUC(serverGetPoint)**

Macro-F1 uses sklearn default (averages over present classes, NOT forced range(N)).

## Data
- Train: 14,995 rallies / 84,707 strokes / mean 5.65 strokes/rally
- Test: 1,236 rallies / 3,589 strokes / **mean 2.90 strokes/rally** (much shorter)
- Test L distribution: 32% L=1, 24% L=2, 17% L=3, ... heavy short-prefix
- 166 train players, 63 test players, **23 NEVER in train (46% test rallies have ≥1 unseen player)**
- 216 train matches, 55 test matches, **0% overlap** (every test match is a new event)
- ~70k augmented OOF samples (each rally → N-1 prefix→target pairs)

## Best Solution (LB 0.4285, OOF 0.4816)
- **LSTM**: BiLSTM 192d/2L + multi-head attention + player embedding + sequential action→point heads + aux next-token head
- **LightGBM**: 49 hand-crafted features (last/prev/serve stroke attributes, score state, action group counts, cross features, raw player IDs)
- **Ensemble**: α·LSTM + (1-α)·LGB, α=0.64 action / 0.50 point
- **Calibration**: plug-in additive log-bias (Koyejo 2014), grid-search on OOF macro-F1
- **Bagging**: seeds {42, 1337}

## 18 Failed Attempts — Key Lessons
- **14 architecture/loss variants** (BiGRU, BiTransformer, larger LSTM, focal loss, soft macro-F1, more seeds, ...) all regressed OOF
- **Length-stratified calibration** (testweighted): LB -0.004. **Length shift is NOT the gap source** (verified by importance-weighted OOF: testw OOF ≈ uniform OOF Δ=0.0001)
- **A v2: per-player historical stats** (with proper per-fold leakage handling, sex-cold-start, num_leaves sweep): OOF +0.004 → **LB -0.024 (catastrophic)**. The per-fold OOF was still inflated because KFold splits rallies, so train/val folds share ~99% players — OOF measured "performance on seen players" which doesn't reflect test conditions.
- **Conditional ensemble** (route 46% unseen-player test rallies to baseline V5Z, keep AV2 for 54% seen): **LB unchanged at -0.024**. The diagnostic moment that broke the "unseen player is the problem" theory.

## Diagnosed Bottleneck: Cross-Event Style Drift
Even SEEN players play DIFFERENTLY in test tournaments vs train tournaments. 100% match disjointness means every test match is a different event — different conditions, opponents, time. Player-style features from train events are biased estimators for test events. This is NOT just an "unseen player" problem — it's a structural per-event drift problem.

## Constraints
- Single RTX 3060 6GB, ~30 min per LSTM training (5-fold)
- LB submissions limited (~5/day)
- External data likely not allowed
- Want to beat rank #4

## What I Haven't Tried
TTA, SSL pretrain (corpus 16k seqs is small), pseudo-labeling, match-style proxy features, player clustering with style inference, adversarial validation re-weighted OOF, group-by-match KFold, ordinal regression on point grid, table tennis tactical domain knowledge encoding, stacking meta-learner.

## Ask
Suggest 3-5 concrete directions with mechanism + expected ROI. Particularly interested in:
1. **Cross-event style drift** — how do top sports analytics teams handle this? Any published recipe?
2. **Using test set as unlabeled data** — pseudo-labeling / transductive at this scale?
3. **Domain knowledge encoding** — table tennis tactical theory features I should add?
4. **Group-aware CV** — would by-match KFold give honest OOF that prevents future A v2 disasters?
5. **What might rank #1-3 know that I don't?**

Be brutal — assume my framework engineering is competent and the obvious has been ruled out.
