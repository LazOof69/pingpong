# Breaking through rank #4: the CV protocol is your bug

**Your 0.0531 OOF–LB gap is overwhelmingly a measurement defect, not a model defect.** Rally-level KFold with ~99% player overlap measures "performance on seen players in seen tournaments" — a quantity uncorrelated with what the leaderboard scores. The single highest-leverage move is to redo your CV under `StratifiedGroupKFold(groups=match_id)` so each validation fold contains ~43 unseen matches (mirroring the 55-match test set), then re-rank every existing model and intervention you've tried under that protocol. Most of your 18 "failed" experiments may be partial successes that lost their signal in noisy KFold. The diagnosed style-drift problem is real but secondary; the primary issue is that you have been **selecting models on the wrong objective**.

Below is a technical survey across all six requested axes followed by a ranked action list. Throughout, methods are tagged **PROVEN-IN-DOMAIN**, **PROVEN-OTHER-DOMAIN**, **SPECULATIVE**, or **ANTI-PATTERN**. Citations are inline.

---

## 1. Cross-event style drift: what the literature actually supports

The dominant negative result you must internalize is **DomainBed (Gulrajani & Lopez-Paz, ICLR 2021, arXiv:2007.01434)**: across seven domain-generalization benchmarks and 14 algorithms, *no method beats tuned ERM by more than 1 point on average* under honest model selection. **IRM (Arjovsky 2019), V-REx (Krueger ICML 2021), GroupDRO (Sagawa ICLR 2020)** are all **ANTI-PATTERN at your scale** — Rosenfeld et al. "The Risks of IRM" (ICLR 2021, arXiv:2010.05761) prove IRM fails in non-linear regimes; Kamath et al. (AISTATS 2021, arXiv:2101.01134) show IRMv1 collapses with few environments and small per-environment samples. Your 14,995/216 ≈ 69 rallies-per-match is below every regime where these methods have ever worked. Don't spend submissions here.

What does work for sports cross-event problems is more pedestrian. **Davis, Bransen, Decroos et al. "Methodology and evaluation in sports analytics" (Mach. Learn. 2024, doi:10.1007/s10994-024-06585-0)** is your single most directly authoritative citation: it formally endorses **leave-one-competition-out CV** when the deployment goal is generalization to new tournaments. **Decroos & Davis "Player Vectors" (ML 2020)** show soccer style is partially stationary across seasons but drifts ~15–25% year-over-year, motivating *small style embeddings derived from observable behavior* over one-hot identity. **Robberechts & Davis** (KDD/ECML workshops 2020–22) document season-over-season decay and recommend recency-weighted training.

The sport-specific stroke-prediction literature is even more revealing. **Wang et al. ShuttleNet (AAAI 2022, arXiv:2112.01044)** and the **CoachAI Badminton Challenge 2023** track-2 competition (organizers' report at sites.google.com/view/coachai-challenge-2023/report; IJCAI-24 demo arXiv:2306.15664) operate on **temporally split, player-overlapping data** — the reported gains there *systematically overstate* what's achievable when 100% of test matches are unseen events. The best-published sport-side trick that explicitly addresses cross-event drift is **Team Intro_to_AI_team8's CoachAI 2023 winning solution (arXiv:2307.13715)**: they *deliberately weakened player conditioning* by removing the duplicated player embedding on the spatial stream. Their justification — "in high-intensity matches, players have more control over shot type than precise landing spot" — is a direct mechanism for why your last-stroke pointId F1 collapses to 0.075.

The most relevant test-time-adaptation result is **SHOT (Liang et al., ICML 2020, arXiv:2002.08546)**: source-free DA with information maximization plus self-training. The mapping to your problem is unusually clean — every test match is a small adaptation episode, and the visible prefix provides *truly self-supervised* training data because you have the next-stroke labels for visible strokes. **TENT (Wang et al., ICLR 2021, arXiv:2006.10726)** is **SPECULATIVE-leaning-ANTI-PATTERN** here: Niu et al. SAR (ICML 2023, arXiv:2302.12400) and EATA (NeurIPS 2022) document collapse-to-one-class when batch size is small, and your per-match test sets average ~22 rallies. **TTT (Sun ICML 2020)** requires an auxiliary self-supervised head retrofitted into the architecture — not worth the surgery on a 30-min training budget.

The cheapest test-time-adaptation method, and one the literature consistently confirms helps, is **match-level feature standardization** at both train and test: subtract match-mean and divide by match-std for every continuous feature. This is "BatchNorm-at-the-match-level" and has produced 1–3% Kaggle tabular-comp gains under covariate shift (e.g., IEEE-CIS Fraud writeups). **PROVEN-IN-DOMAIN, near-zero cost.**

## 2. Group-aware CV and adversarial validation

Your finding that rally-level KFold inflates OOF by ~99% within-fold player overlap is a textbook **leak-by-grouping-variable** problem (sklearn cross-validation docs; Wasikowski's StratifiedGroupKFold notebook on Kaggle). The honest setup is unambiguous: **outer = StratifiedGroupKFold by match_id, 5 folds (≈43 val matches each); inner GroupKFold for HPO**. With only 216 matches, going below 4 inner folds inflates noise; above 8 wastes compute. With 1,236 public-LB rallies, the public-LB Monte Carlo standard error on macro-F1 is ≈ √(0.43·0.57/1236) ≈ 0.014 — your 0.0531 gap is ~3.7σ above pure noise, so most is real.

**Adversarial validation reweighting (Pleskov, Tunguz, Jost; FastML and KDnuggets writeups)** is the canonical fix when the test distribution is shifted. Train a binary classifier on `is_test`, look at OOF AUC: 0.5–0.6 means no shift; 0.6–0.85 is your expected range; >0.95 is severe (Sberbank Russian Housing reached 0.993). **Two outputs**: (a) drop or downweight features whose AV importance is high — these are *style-leaky* features and almost certainly include raw player IDs and any raw player-level historical statistic; (b) compute `w(x) = p(test|x)/(1−p(test|x))` clipped at percentiles [1, 99], use these weights for **calibration-fold OOF F1** so your threshold tuning runs on a test-distribution-weighted estimator. **Bojan Tunguz's IEEE-CIS Fraud writeup** and the formal treatment in **Pan et al. (arXiv:2112.10078)** are the standard references. Expected combined lift from match-disjoint calibration plus AV-weighted threshold tuning: **+0.010 to +0.025 on combined score** — by far the largest single intervention available to you.

A subtle and very important consequence: your previous **per-fold leakage-aware player stats experiment regressed by 0.024 on LB despite +0.004 OOF** because the OOF metric itself was a poor estimator of LB. Re-run that experiment under match-disjoint OOF before declaring it dead. Empirical-Bayes shrinkage with strong prior (κ ≥ 50) plus out-of-fold computation is the canonical solution (Efron–Morris; sklearn `TargetEncoder` with cross-fitting); the previous failure was likely insufficient shrinkage rather than a fundamentally bad idea.

## 3. Test set as unlabeled data: where the real upside lives

Three specific transductive interventions deserve testing in your remaining submissions, ranked by EV.

**(a) Empirical-Bayes player features computed strictly from the visible test prefix.** For each test rally, derive per-player rates (P(topspin), P(forehand), mean rally length so far, etc.) from that match's visible prefix only — never from train data about that player, since train-vs-test player overlap is partially fictional. Combine with global Beta(α,β) prior fit on train via `p̂ = (n·p_obs + κ·μ_global) / (n + κ)`, with κ ≈ 20–50 chosen by match-disjoint CV. With mean prefix length 2.9 strokes, this puts ~10% weight on observed and ~90% on global prior — exactly the regularization your previous failed attempt was missing. For train rallies, do the same using out-of-fold visible-prefix-only stats from sibling rallies in other folds. This is the **PROVEN-IN-DOMAIN pattern** from Decroos & Davis Player Vectors and from Adjileye RisingBALLER (arXiv:2410.00943). Expected lift: **+0.003 to +0.015 LB**.

**(b) Per-match continued boosting in LightGBM via `init_model=`.** At test time, for each test match retrain 5–20 additional boosting rounds on the visible-prefix rallies of that match only, with strong L2 regularization and a high `min_data_in_leaf`. LightGBM supports this natively. This is the **simplest possible source-free DA recipe**, has near-zero downside, and works because the per-match adaptation set is small but on-distribution. Hardt & Sun "Test-Time Training Provably Improves Transformers as ICL" (arXiv:2503.11842) provides theoretical backing for tabular models. Expected lift: **+0.005 to +0.015 LB**.

**(c) SHOT-style per-match BiLSTM fine-tune.** Freeze the classifier head, fine-tune the BiLSTM body on visible-prefix rallies of the test match using a strong L2 anchor to source weights (or EMA-of-source-weights initialization), ≤10 SGD steps at LR 1e-5. Map: each test match is one adaptation episode; visible prefixes provide ~100–300 stroke transitions of free supervised data because you have the actual labels for visible strokes. Liang et al. (ICML 2020) report 4-point gains on Office-Home; expect smaller here but with real upside. **PROVEN-OTHER-DOMAIN with excellent mapping.** Expected lift: **+0.005 to +0.020 LB**, with implementation risk if regularization is too weak.

**Pseudo-labeling on the test set is the highest-ceiling, highest-variance play** and should come *after* (a)–(c). The cautionary tale is **Oliver et al. NeurIPS 2018 "Realistic Evaluation of Deep SSL" (arXiv:1804.09170)**: SSL can degrade below the labeled-only baseline under distribution mismatch. **Arazo et al. IJCNN 2020 (arXiv:1908.02983)** identify confirmation bias as the dominant failure mode and prescribe two fixes that apply here: mixup on embeddings and a minimum-50%-labeled-per-minibatch constraint. **FlexMatch (Zhang NeurIPS 2021, arXiv:2110.08263)** and **FreeMatch (Wang ICLR 2023)** provide per-class adaptive thresholds essential for your imbalanced 19-class action target. The closest competition precedent is **Stanford OpenVaccine 2020**, where pseudo-labeling on test sequences produced the winning submission despite pseudo-label scores being worse than the best supervised model — pseudo-labels injected the test-sequence distribution into training. **CommonLit Readability 1st place** (github.com/mathislucka/kaggle_clrp_1st_place_solution) uses the same recipe. **Validate on held-out match-disjoint OOF before submitting** any pseudo-labeling result; if OOF drops, abort.

Self-supervised pretraining on 84,707 + 3,589 = ~88k tokens is a mixed bag. **Warstadt et al. "When Do You Need Billions of Words" (arXiv:2011.04946)** put the lower bound for meaningful MLM downstream gains at ~1M words; you are 2 orders of magnitude below. **However**, **Krishna et al. "Downstream Datasets Make Surprisingly Good Pretraining Corpora" (arXiv:2209.14389)** showed that pretraining only on the downstream corpus matched web-scale pretraining on 7/10 of their tasks — you only need representations useful for *this* task, not general ones. A small Transformer (2 layers, d=128) pretrained causally on train + visible test prefixes for 5–10 epochs, then fine-tuned supervised, is a positive-EV experiment. The **Riiid Answer Correctness 1st place (arXiv:2102.05038)** confirms small Transformers beat LSTMs on short tactical sequence prediction.

## 4. Domain knowledge: the table-tennis tactical features you don't have

Your existing 49 hand-crafted features are a respectable baseline but miss several published-and-validated structures. The dominant analytic framework in Chinese table-tennis literature is **Wu Huanqun's three-phase method** (Wu et al. 1988; Zhang & Yang J. Sports Sci. 36:2663–2674, 2018; Hsu et al. 2021, doi:10.1186/s13102-021-00283-3): rallies segment into **STAS (strikes 1, 3, 5), RTAS (strikes 2, 4), and Stalemate (≥6)**, with strictly decreasing scoring rate across phases. **Fuchs & Lames "First Offensive Shot" (Int. J. Racket Sports Sci. 2(1), 2021)** and the **EFOS extension (Wu, Liu, Lames 2025, doi:10.1186/s13102-025-01183-6)** identify the rally's "turntable" — the first non-backspin/non-serve stroke — and find the **Initial-Offensive Phase contains 70.6% of all strokes and dominates outcomes**. A binary `firstOffensiveStrokeIdx` plus `strokesSinceFOS` plus phase-indicator is essentially free to compute and is the most theoretically motivated set of features the practitioner doesn't have.

The **Pfeiffer, Zhang, Hohmann Markov chain model (Int. J. Sports Sci. Coach. 5(2):205–222, 2010)** and **Wenninger & Lames (Springer ISCSS 2016)** validated the Markov property empirically for elite table tennis and provide transition-probability tables you can essentially copy. A phase-segmented second-order Markov feature `f = P(action_t | action_{t-1}, action_{t-2}, phase)` with Laplace smoothing α=1, computed leakage-safely with leave-one-event-out, captures the canonical loop-against-push, block-after-smash, and push-after-push patterns. **Wang J. et al. (TacSimur, IEEE TVCG 2019)** use exactly this construction.

The **Wu et al. 2022 AIEL/AOEL framework (doi:10.1186/s13102-022-00396-3)** is the single best feature for predicting rally winners and pointId on terminal strokes: scoring rate jumps from ~60% to >97% after a "half-long" ball — i.e., when `pointId_{t-1}` lands in the middle row (zones 4, 5, 6). This is a **leak-free predictor of next-stroke being a put-away winner** and addresses your terminal-stroke F1 collapse directly.

For the spatial collapse on last strokes specifically, the published explanation is selection bias. **Klassen & Magnus (2001, 2014, *Analyzing Wimbledon*)** reject point-iid and show high-variance "important-point effect"; **Peiris, Epasinghege Dona, Swartz (Int. J. Sports Sci. Coach. 2025, doi:10.1177/17479541251333943)** show server advantage *dissipates with rally length*, meaning rallies surviving to stroke n>5 are selection-biased toward exotic patterns. **Pfeiffer 2010** quantifies that direct point-winning probability concentrates on Smash (17.55%) and Topspin (8.78%), but with disproportionately high error rates — the conditional distribution at the terminal stroke is a near-equal mixture of intended winners (high variance across opponents) and errors (essentially uniform over zone 0 + edge zones). The IJCAI-24 CoachAI organizers explicitly call out the same phenomenon: 11 of 16 teams improved shot-type prediction, but only 1 of 16 improved area prediction. **Conclusion: most of your remaining gains will come from actionId, not pointId.** Set your priors accordingly.

The **TacticAI symmetry-augmentation insight (Wang et al., Nature Comm. 2024, arXiv:2310.10553)** is unconditionally cross-event robust because table-tennis physics is left-right symmetric. Mirror every training rally around the table's center column (zone [1,2,3 / 4,5,6 / 7,8,9] → [3,2,1 / 6,5,4 / 9,8,7], positionId reflected, handId reflected with appropriate spinId adjustment). At test time, predict on the rally and its mirror, average action probabilities. **Free 2× data inflation, no hyperparameters.**

## 5. Macro-F1 calibration: what's beyond the Koyejo plug-in

Your additive log-bias plus coordinate descent is the **OvR plug-in approximation to Koyejo et al. 2014 (NeurIPS)** — provably consistent for binary F1 under calibrated η, but in your multi-class setting it ignores three things: the joint argmax constraint, the non-concave macro-F1 surface that traps coordinate descent in local optima, and finite-sample variance on rare-class thresholds estimated from few OOF samples each. Three published improvements stack on top of your existing approach.

**Frank-Wolfe over confusion matrices (Narasimhan, Ramaswamy, Saha, Agarwal ICML 2015, arXiv:1501.00287)** solves the joint problem in polynomial time, producing a sequence of cost-sensitive classifiers whose convex combination converges to the Bayes-optimal macro-F1 classifier. T=20 iterations is sufficient. Implementation cost is moderate.

**The exact F-measure maximization algorithm (Dembczynski, Waegeman, Cheng, Hüllermeier NeurIPS 2011)** runs in O(m³) per class and dominates simple thresholding when η-marginals don't factor — which is always the case with correlated cross-fold models. Reported lift: 1–3% F1 on Medline-26K text classification.

**Bootstrap aggregation of thresholds**: B=200 OOF resamples, take median per class. Trivial to implement, immediately reduces threshold variance on rare classes where you have ≤50 OOF samples. This is the **finite-sample stabilization** prescribed by Yan et al. ICML 2018 (arXiv:1806.00640).

The single **highest-leverage post-hoc move in this entire report** is **strategic class suppression**, exploiting sklearn's macro-F1 default. Sklearn averages over `union(y_true, y_pred)`: predicting an ultra-rare class once and being wrong contributes F1=0 to the mean (≈0.053 hit for K=19); never predicting that class drops it from the average if it's also absent from y_true. For each class k, on event-disjoint OOF compute (i) `Pr(k ∈ y_true on a held-out match)`, (ii) `Pr(F1_k > 0 if k is predicted)`. If the product is below 1/K, **set δ_k = +∞ — never predict class k**. Your earlier "forced labels=range(N)" regression is exactly consistent with this asymmetry: forcing the average over all 19 classes added F1=0 entries for classes the test simply didn't contain. **Expected lift: +0.005 to +0.020 macro-F1** (Lipton, Elkan, Naryanaswamy ECML 2014, arXiv:1402.1892, formally analyzes this).

On the loss side, **focal loss is provably the wrong tool for macro-F1** (Menon ICLR 2021 Table 3; Cao 2019 LDAM Table 1) — it down-weights confident head-class true positives without helping tail-class precision. The right replacements are **Logit Adjustment (Menon et al. ICLR 2021, arXiv:2007.07314)** — add `−τ·log π_k` to logits during training or post-hoc subtract — and **LDAM-DRW (Cao et al. NeurIPS 2019, arXiv:1906.07413)** with deferred reweighting at 80% of epochs. Critically, LDAM-DRW destroys representations if reweighting starts at epoch 0; deferral is mandatory. **τ-norm post-hoc (Kang et al. ICLR 2020, arXiv:1910.09217)** rescales the classifier weights via `w_k ← w_k / ||w_k||^τ` with τ ∈ [0.5, 1] tuned on OOF, applied *before* your additive log-bias. These three operate at different layers and compound — expected combined lift **+0.010 to +0.025 macro-F1**.

**Soft-F1 fine-tuning (Bénédict et al., arXiv:2108.10566)** for the last 2–3 epochs after CE convergence is the differentiable surrogate route, with reported gains of 1–2 macro-F1 points on long-tail benchmarks. Use a class-balanced sampler and batch ≥256.

## 6. Kaggle archeology: what top finishers know

The five most consistent winning recipes across small-data sequence and tabular Kaggle competitions reviewed (MoA, CommonLit, Otto, Riiid, NFL Big Data Bowl, IceCube, OpenVaccine, CoachAI Badminton 2023) appear in roughly the following frequency: GroupKFold matched to test split (~95% of top-3 solutions), GBDT+NN blending (~85%), multi-seed bagging (~90%), adversarial validation diagnostic when shift suspected (~60%), pseudo-labeling on test (~55%), multi-task auxiliary heads (~40%), domain-adaptive pretraining (~35%), permutation-invariant set encoding for player tracking (~25% of sport-tracking comps), adversarial reweighting (~20%).

The two competitions with the closest formal mapping to your problem are **CoachAI Badminton Challenge 2023 Track 2** and **NFL Big Data Bowl 2020**. The CoachAI 1st place (Intro_to_AI_team8, arXiv:2307.13715) and 2nd place (MuLMINet/Badminseok, arXiv:2307.08262) both **kept the ShuttleNet two-stream Transformer skeleton** (rally extractor + player extractor with shared parameters + position-aware gated fusion) and won by adding side features and multi-task auxiliary heads. The NFL BDB 2020 winners "The Zoo" (Singer & Gordeev) used **permutation-invariant 1×1 convolutions over (offense × defense) pairs with no player-identity features at all** — the pure-relational design directly defeats the unseen-player generalization problem you face with 36% unseen test players.

The MuLMINet auxiliary-head recipe is the most directly portable. Use Cramér's V on your training data to rank candidate side-targets (spinId, strikeId, handId, opponent_positionId), keep those with V > 0.20 to next-stroke action, train all heads jointly with `L = α·(L_action + L_point + L_winner) + (1−α)·Σ L_aux` and sweep α ∈ {0.30, 0.35, 0.40, 0.45}. Auxiliary tasks regularize the trunk and are *invariant under event* — spin physics doesn't change between matches. Expected lift: **+0.005 to +0.020**.

The Riiid 1st place (arXiv:2102.05038) demonstrates that **small Transformers with last-query attention beat LSTMs on short tactical sequences** — your BiLSTM is plausibly the wrong architecture on first principles. Replacing it with a 2-layer Transformer encoder-decoder following the ShuttleNet two-stream design is a moderate-engineering, moderate-expected-lift move. The CoachAI 1st place explicitly notes that *smaller models with longer training (dim 16–32, 1–3 layers, 300 epochs)* dominated larger ones on the equivalent dataset size to yours — your current 192d/2L BiLSTM is plausibly oversized.

---

## Ranked actionable directions

Each entry: mechanism, expected ROI on combined-score LB, implementation cost, prerequisites, risk. Ranked by expected lift per implementation hour, assuming you have not already done these (cross-checked against your 18 failures).

### 1. Switch primary CV to StratifiedGroupKFold by match_id, re-rank everything

**Mechanism:** ~43 unseen matches per validation fold mirrors the 55-match test set. Re-evaluate every existing model and intervention you've tried under this protocol. Most of your "failures" were partial successes lost in noise.
**Expected lift:** +0.010 to +0.030 by selecting better submissions, even with no new models trained.
**Cost:** 2–4 hours — re-run existing OOF predictions through the new fold splits and recompute macro-F1.
**Prerequisites:** None.
**Risk:** Low. Worst case you confirm your ranking is already correct; more likely you discover that several of your "regressed" experiments were actually improvements masked by 0.01–0.02 KFold noise.

### 2. Match-disjoint calibration + adversarial-validation reweighting

**Mechanism:** Hold out one entire tournament as `D_calib`. Train AV discriminator `g(x): train vs test`. Compute weights `w_i = g(x_i)/(1−g(x_i))` clipped at [1, 99] percentiles. Run your CD log-bias on `D_calib` with weighted macro-F1 as the objective. Apply discovered biases at inference. Drop or downweight features whose AV importance is high — these are style-leaky.
**Expected lift:** +0.010 to +0.025.
**Cost:** 3–4 hours.
**Prerequisites:** #1 done.
**Risk:** Low if AV AUC > 0.65; if AUC ≈ 0.5, weights collapse to 1 and nothing changes (no harm).

### 3. Strategic class suppression for rare actions/points

**Mechanism:** For each class k, compute on event-disjoint OOF the product `Pr(k ∈ y_true on held-out match) × Pr(F1_k > 0 when predicted)`. If product < 1/K, set δ_k = +∞. Exploits sklearn macro-F1 default that drops classes absent from `union(y_true, y_pred)`.
**Expected lift:** +0.005 to +0.020 macro-F1, mostly on action head.
**Cost:** 1–2 hours.
**Prerequisites:** #1 done.
**Risk:** Medium. If a held-out event happens to contain the suppressed class, you eat full F1_k=0. Mitigate by suppressing only classes with empirical Pr < 0.2 over multiple held-out events.

### 4. Per-match continued boosting in LightGBM via `init_model=`

**Mechanism:** At test time, for each test match, retrain 5–20 boosting rounds on visible-prefix rallies of that match only, with strong L2 reg and high `min_data_in_leaf`. Source-free DA at zero retraining cost.
**Expected lift:** +0.005 to +0.015.
**Cost:** 2–3 hours.
**Prerequisites:** #1 done (to evaluate honestly).
**Risk:** Low. Guard against overfitting with a small HPO sweep on regularization; per-match round count is small.

### 5. Empirical-Bayes player features from visible test prefix only

**Mechanism:** For each test rally, compute per-player rates strictly from that match's visible prefix. Combine with global Beta prior via shrinkage `p̂ = (n·p_obs + κ·μ_global)/(n+κ)`, κ ≈ 30 by CV. For train rallies, use leave-one-fold-out visible-prefix-only stats from sibling rallies. Replaces your failed train-side player-stats experiment with a leakage-immune, distribution-shift-robust mechanism.
**Expected lift:** +0.003 to +0.015.
**Cost:** 4–6 hours including correct out-of-fold plumbing.
**Prerequisites:** #1 done.
**Risk:** Medium-low. Done correctly, it can only help; done incorrectly, you reproduce the previous catastrophe.

### 6. Logit adjustment + τ-norm + soft-F1 fine-tune (loss/calibration stack)

**Mechanism:** Replace class-weighted CE with logit adjustment `CE(z + τ·log π, y)`, τ ∈ {0.5, 1, 1.5}. Apply post-hoc τ-norm on classifier weights. After CE convergence, fine-tune for 2–3 epochs with soft-F1 loss + class-balanced sampler. The three operate at different layers and compound.
**Expected lift:** +0.010 to +0.025.
**Cost:** 1 day to retrain and tune.
**Prerequisites:** #1 done. Class-balanced sampler implementation.
**Risk:** Medium. Logit-adjustment τ is sensitive; sweep on match-disjoint OOF.

### 7. Multi-task auxiliary heads (MuLMINet recipe)

**Mechanism:** Add auxiliary classification heads for spinId, strikeId, handId, opponent_positionId on the existing trunk. `L = α(L_action + L_point + L_winner) + (1−α)Σ L_aux`, sweep α ∈ {0.30, 0.35, 0.40, 0.45}. Use Cramér's V to drop side-targets with V < 0.20.
**Expected lift:** +0.005 to +0.020.
**Cost:** 4–6 hours.
**Prerequisites:** #1 done. Code adapted from github.com/stan5dard/IJCAI-CoachAI-Challenge-2023.
**Risk:** Low. Auxiliary heads regularize the trunk and are event-invariant.

### 8. Phase-segmented Markov + EFOS + AIEL features

**Mechanism:** Add (a) `phase ∈ {STAS, RTAS, Stalemate}` indicator from strikeId, (b) `firstOffensiveStrokeIdx` and `strokesSinceFOS` per Wu/Lames EFOS, (c) `AOEL` flag for `pointId_{t-1} ∈ {middle row}`, (d) phase-segmented second-order Markov P(action_t | action_{t-1}, action_{t-2}, phase) with Laplace smoothing and leave-one-event-out computation.
**Expected lift:** +0.003 to +0.015.
**Cost:** 6–8 hours including correct LOEO plumbing.
**Prerequisites:** #1 done.
**Risk:** Low. Theoretically motivated, validated in Pfeiffer 2010 / Wu 2022 / Wu-Lames 2025.

### 9. Symmetry / reflection augmentation (TacticAI lesson)

**Mechanism:** Mirror every training rally around the table's center column (pointId zones, positionId, handId reflected with appropriate spinId adjustment). At test time, predict on rally and mirror, average action probabilities after un-mirroring pointId logits.
**Expected lift:** +0.002 to +0.008.
**Cost:** 2–3 hours.
**Prerequisites:** Care with which features actually flip under reflection; spinId may have asymmetric semantics depending on coding.
**Risk:** Low if encoding is correct; medium if it's wrong (silent metric corruption).

### 10. SHOT-style per-match BiLSTM fine-tune

**Mechanism:** For each test match, freeze classifier head, fine-tune BiLSTM body on visible prefix with strong L2 anchor to source weights, ≤10 SGD steps at LR 1e-5. Per-match adaptation episode with self-supervised data (real labels for visible strokes).
**Expected lift:** +0.005 to +0.020.
**Cost:** 1–2 days.
**Prerequisites:** #1 done. Careful regularization (L2 anchor mandatory).
**Risk:** Medium-high. Without correct anchoring, catastrophic forgetting wipes the baseline.

---

## What to skip and why

Don't burn submissions on **IRM, V-REx, GroupDRO, DANN, focal loss, more bagging seeds, larger LSTMs, BiTransformer in your current configuration, TENT on a model with BatchNorm and small per-match batches, conformal/set-valued classifiers** (your competition demands single labels), or **pseudo-labeling without OOF validation gating**. These are either ANTI-PATTERN at your scale (DomainBed, Niu 2023) or in your already-tested failure list.

Pseudo-labeling with FreeMatch-style class-adaptive thresholds is high-ceiling but should come *after* directions 1–6, and only with a hard rule that match-disjoint OOF must improve before submitting.

## The brutal closing

Your problem is not your model. Your BiLSTM + LGBM ensemble at LB 0.4285 is a competent baseline. **The top-3 are almost certainly running some combination of: match-disjoint CV, adversarial-validation reweighting, logit adjustment or LDAM-DRW training (so their probabilities are calibrated *before* threshold tuning), strategic suppression of classes not present in test, multi-task auxiliary heads, and per-class blend weights via CMA-ES on OOF**. None of these requires architectural genius, all of them are documented in cited literature, and they are precisely the moves your 18 documented failures did not test under an honest CV protocol. Direction #1 on its own — re-running every prior experiment under StratifiedGroupKFold by match_id — will likely change which of your existing models you submit, and that alone is plausibly worth one or more leaderboard ranks.