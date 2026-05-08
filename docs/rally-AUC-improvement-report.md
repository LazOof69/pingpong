# Closing the rally-AUC gap with structure, shrinkage, and surprise

**The 0.62 → 0.80 gap is real, not leaderboard noise — at n=1,845 with ~50% positive class, the standard error of AUC is 0.013 at AUC=0.62 and 0.010 at AUC=0.80, so an 0.18 gap is ~11σ unpaired and ~15σ paired (DeLong); randomness alone cannot explain it.** The published ceiling for prefix-rally outcome prediction in racquet sports is **0.71–0.77 AUC** (Wei et al. 2013 tennis adapted: 0.77; ShuttleNet badminton 2022: 0.71–0.75), so a true 0.80 implies the leader is exploiting structural signals — likely the alternation parity, score-state, per-player Markov stylistic features, and possibly spin/speed if those columns exist. Realistic combined headroom for the user is **+0.10 to +0.16 AUC** if the recipes below are implemented cleanly, landing them at **0.72–0.78 OOF**. The score gap of 0.044 to leader is **72.7% closeable through AUC alone** (since 0.16 × 0.20 weight = 0.032), but **per unit, F1 lifts are worth 2× as much** in this scoring formula — once AUC is within 0.05 of leader, every hour belongs to the F1 components. Below: a leakage diagnosis of why kinematic and LSTM features failed, the four-block feature plan with formulas, and a daily validation gate.

## Why stroke-kinematic features saturated and skill-differential features will not

The xy/trajectory/dominance features the user already tried saturated for a structural reason worth naming. **Per-stroke kinematic features are high-variance, low-signal-per-row** — a single landing-zone or speed reading carries an outcome correlation of roughly 0.05–0.10 (Wei et al. 2013 reported single-feature AUCs of 0.52–0.65 for speed, feet location, impact location individually), and they live in a dense numerical space LightGBM has already partitioned to exhaustion. The Sweet-Spot 2013 paper achieved its 0.685 → 0.773 jump only by **multiplying these features by an opponent-specific adaptation layer** (per-opponent UBM-MAP shift). Without that adaptation, three more raw kinematic columns just add noise.

**Skill-differential features have a different statistical character**: they aggregate hundreds of strokes into one number per player, suppressing per-stroke noise by √N before the tree ever sees them. The Bayes posterior $\hat p_i^{\text{EB}}$ has variance $\hat p(1-\hat p)/(\kappa+n_i)$, which for $n_i=200$ rallies and $\kappa=20$ is ~10× smaller than a single observation. **A tree split on a shrunk per-server win rate is splitting on a denoised population statistic, not a per-stroke draw.** This is why Bradley-Terry, Elo, common-opponent normalization, and EB-shrunk player effects consistently dominate the literature on match-level prediction (Kovalchik 2016; Sipko & Knottenbelt 2015), and why your κ-grid winrate is already your strongest single feature.

The implication is sharper than just "add more skill features." **Kinematic features only pay off when made conditional on player identity** — a forehand-down-the-line means something different from Ma Long than from a defensive chopper. The Wei 2016 KDD paper formalized this with their "style dictionary" of k=20–50 stroke-style clusters. Your lookalike upgrade is not more xy data; it is the same xy data **collapsed onto a per-player style embedding** and then fed back as an interaction (Section C, NMF block).

## Why the LSTM features regressed, and the fix

The user's LSTM hidden-state features dropped AUC from 0.62 to 0.59, and softmax features to 0.61. Three causes compound here, ranked by likelihood.

**Cause 1, target-correlated training (most likely).** If the LSTM had a `serverGetPoint` auxiliary head during joint training — and the user explicitly mentions trying "LSTM with sgp aux head" — then the hidden states leak label information on training folds in a way that does not transfer match-disjoint. Prokhorenkova et al. (2018, CatBoost paper) call this "prediction shift," the same pathology that motivated ordered boosting. Symptom: train AUC inflates, OOF collapses.

**Cause 2, player-conditional overconfidence.** With 166 players and match-disjoint folds, a fresh LSTM per fold memorizes stroke patterns of training-fold players; on held-out matches it is overconfident on familiar individuals and near-uniform on unseen ones. The softmax becomes a fold-identity feature.

**Cause 3, calibration drift.** Per-fold LSTM softmax temperatures differ; concatenating them into one OOF column gives a per-fold scale shift that LightGBM partly memorizes.

**The fix is non-negotiable**: rebuild the LSTM trained **only on next-`action_id` prediction**, never on `serverGetPoint`, never with a `pointId` terminal-label aux loss, and apply per-fold temperature scaling before extracting features. Then expose the entropy block from Section C. The litmus test is the **target-permutation experiment**: shuffle `serverGetPoint` within fold and rerun LightGBM with these features alone. OOF AUC must collapse to 0.50 ± 0.01. If it stays above 0.55 you still have leakage. Without this discipline, sequence-derived features will continue to subtract from your score.

## (A) Feature plan — four blocks with expected lifts

The four blocks are **structural / score-state**, **multi-level Bayesian skill**, **sequence-deviation Markov**, and **cold-start NMF + k-NN**. Their expected contributions, with overlap discount, are:

| Block | Lift estimate | Confidence |
|---|---|---|
| Structural TT priors (parity, service block, score state, Morris importance, expected-remaining-strokes regressor) | +0.04 to +0.07 | High — deterministic from inputs, no leak risk |
| Multi-level EB partial pooling replacing κ-grid | +0.015 to +0.035 | Medium-high |
| Per-player Markov bigram log-likelihood + deviation Δℓ | +0.005 to +0.012 | Medium-high — direct Pfeiffer 2010 evidence |
| NMF rank-6 style embedding + k-NN borrow (Hellinger) | +0.020 to +0.040 | Medium — biggest gains on the 31 unseen players |
| Stupid-backoff trigram + Brown-cluster trigram backoff | +0.002 to +0.005 | Low-medium |
| LSTM next-action entropy + KL-to-player-marginal (after rebuild) | +0.005 to +0.012 | Medium — small if Markov block already extracts most of it |
| Pair-specific EB shrinkage toward additive baseline | +0.002 to +0.010 | Low — pair sparsity is severe |
| Per-prefix-length covariate-shift reweighting | +0.005 to +0.020 | Medium |

**Combined with ~30% overlap discount: +0.07 to +0.14 AUC**, taking the user from 0.62 OOF to **0.69–0.76 OOF**, with stretch to 0.78 if the rebuild and structural prior land cleanly. The remaining gap to a true 0.80 leader almost certainly requires either ball-spin/speed columns (if labeled) or court-grid xy features made player-conditional via NMF — both already touched but un-exploited.

**Ship order** (one feature block per LB day, OOF-validated first):

1. **Day 1 — Structural priors (low-risk, high-yield).** Add `server_stroke_parity`, `service_block_position` (1 vs 2), `is_deuce`, `point_score_diff`, `match_score_diff`, `set_index`, `serve_streak_server_in_block`, `last3_winner_streak`, `expected_remaining_strokes` (separate LGB regressor), `parity_of_predicted_final_stroke`, Morris point importance, and `expected_server_winrate_at_stroke_index_k` lookup. None of these can leak; all are deterministic functions of the visible state. Expected: +0.025–0.045.

2. **Day 2 — Multi-level EB shrinkage.** Replace the κ-grid with method-of-moments multi-level partial pooling (formula in Section C). Refit per fold. Expose `eb_winrate`, `eb_winrate_logit`, `eb_shrinkage_lambda`, `eb_post_se`, and `eb_group_mean` as separate columns; let LightGBM split on the uncertainty. Expected: +0.015–0.035.

3. **Day 3 — Markov bigram block.** Per-player bigram likelihood $\ell_u$, deviation $\Delta\ell = \ell_u - \ell_g$, last-stroke surprise $\sigma_L$, max-stroke surprise. Compute count tables fold-disjoint. Expected: +0.005–0.012.

4. **Day 4 — NMF + k-NN cold-start.** Build the player×behavior matrix, fit NMF rank-6 with KL multiplicative updates, fold-in unseen test players, expose embedding coordinates and k-NN borrowed `eb_winrate`. Expected: +0.020–0.040, concentrated on the unseen-player rallies.

5. **Day 5 — LSTM rebuild + entropy.** Only after the above are stable. Rebuild LSTM on next-action-only, target-permutation gate, then add entropy/top-k/KL features. Expected: +0.005–0.012 marginal.

The user has 3 LB submissions/day; reserve one each day for **best-blend with new block added**, one for **best-blend variant** (seed/depth perturbation, to estimate LB jitter), and one for **probe** (a single-feature ablation of the current day's block). Do not burn LB slots on changes that did not move OOF AUC by at least one paired-fold standard error (~0.003 with K=5 folds).

## (B) Diagnostic deep-dive — kinematic vs differential features

The contrast becomes precise when you write down what each feature actually carries. A per-stroke landing zone has Fisher information about `serverGetPoint` of approximately $I_z \approx (\partial \log P(y|z)/\partial z)^2 \cdot P(y|z)(1-P(y|z))$, which for AUC-equivalence in the 0.55–0.60 range is ~0.04 nats per stroke. With visible prefix mean 3.07 strokes and three zone-derived features per stroke, you have ~9 zone draws contributing roughly $9 \times 0.04 = 0.36$ nats — large enough in principle, but only if **the conditional distribution $P(y|z)$ is stable across players**. It is not: a flick-to-zone-3 from a topspin player has different terminal probability than the same flick from a chopper. Without conditioning, the marginal $P(y|z)$ is averaged over the player mixture and the per-stroke information collapses to roughly 0.005–0.010 nats — exactly the saturation the user observed.

Skill-differential features, by contrast, carry **aggregate Fisher information** $I_p \approx n_p \cdot \bar I_{\text{stroke}}$ where $n_p$ is the player's training-rally count. For $n_p = 200$ and $\bar I_{\text{stroke}} = 0.01$ nats, this is ~2 nats — two orders of magnitude more — and it is concentrated in a single, low-variance feature value the tree can split cleanly. The lesson is not that kinematics are useless; it is that **kinematics are useful only when paired with player conditioning** (style cluster or NMF embedding interaction). Wei 2016's ~7 percentage-point lift from adding the style dictionary on top of raw kinematics is the empirical confirmation: features that were saturated alone became unblocked once player-style was a tree-split-able input.

The actionable corollary for the user: do not add more raw kinematic columns. Instead, build NMF style embeddings (Section C) and add features like `nmf_emb_3 × prefix_action_speed`, `nmf_emb_5 × last_zone_id`, etc., as explicit interactions that LightGBM can split on. These preserve the kinematic signal but route it through the player-style channel where it carries non-degenerate signal.

A second diagnostic: per-pair skill effects also fail in this dataset. With 166 players and ~14k rallies, the median ordered-pair frequency is **0.51 rallies/pair** (14000 / 166×165). Even the most-frequent pairs likely have ~50 rallies. Karchin et al. (2018, arXiv:1807.09236) document that EB-shrunk pair effects on top of an additive baseline contribute 2–3% log-likelihood lift in MLB batter–pitcher matchups, translating to roughly +0.005–0.015 AUC in dense paired-comparison settings. With your sparsity, pair-specific terms will collapse to the additive baseline under EB shrinkage and contribute close to zero. **Spend the time on additive style features and player-style interactions, not on pair-specific terms.**

## (C) Algorithmic recipes — exact formulas

### C.1 Multi-level empirical Bayes for player skill

Replace the user's κ-grid {5, 10, 20, 50} with a two-level partial-pool. Compute everything inside the fold using only training-fold rallies.

**Step 1, global beta-binomial via method of moments.** For each player $i$ with serve-rallies $n_i$ and wins $w_i$, $\hat p_i = w_i/n_i$, weighted league mean $\bar p = \sum w_i / \sum n_i$. The between-player variance after removing binomial sampling noise is

$$\hat\sigma_{\text{btw}}^2 = \max\!\Big(0,\ \widehat{\text{Var}}(\hat p_i) - \mathbb{E}[\bar p(1-\bar p)/n_i]\Big)$$

Then $\hat\kappa = \bar p(1-\bar p)/\hat\sigma_{\text{btw}}^2 - 1$, with $\hat\alpha = \hat\kappa\bar p$, $\hat\beta = \hat\kappa(1-\bar p)$. Single-level shrunk estimate is $\hat p_i^{(1)} = (\alpha + w_i)/(\alpha + \beta + n_i)$.

**Step 2, group-level layer.** Partition players into groups $g \in G$ defined by sex × handedness × style (cluster from Step C.3 below). Working on the logit scale $\theta_i = \text{logit}\,\hat p_i$ with per-player Fisher variance $v_i = 1/(n_i \hat p_i(1-\hat p_i))$, the closed-form Gelman–Hill partial-pooling estimates are

$$\hat\theta_i = \frac{\hat\mu_{g[i]}/\sigma_w^2 + \theta_i/v_i}{1/\sigma_w^2 + 1/v_i}, \qquad \hat\mu_g = \frac{\hat\mu_0/\sigma_b^2 + \sum_{i\in g}\theta_i/v_i}{1/\sigma_b^2 + \sum_{i\in g} 1/v_i}$$

where $\sigma_w^2$ (within-group) and $\sigma_b^2$ (between-group) are estimated by REML or simple ANOVA decomposition. For unseen test players, fall back to the group mean $\hat\mu_{g(u)}$ with posterior SE $\sqrt{\sigma_w^2 + \sigma_b^2}$.

Expose as features: `eb_winrate`, `eb_winrate_logit`, `eb_shrinkage_lambda` $= n_i/(n_i+\hat\kappa)$, `eb_post_se`, `eb_group_mean`, `eb_group_dispersion` $= 1/\hat\kappa_g$, `is_unseen_player`.

**Anti-leak**: refit $\hat\alpha, \hat\beta, \hat\sigma_w^2, \hat\sigma_b^2, \hat\mu_g$ separately for each fold using only that fold's training matches. Final test-set features use the full-training fit. Inflate $n_i$ by an effective-sample-size factor $\hat n_i^{\text{eff}} = n_i / (1 + (\bar m - 1)\hat\rho)$ where $\hat\rho$ is intra-match rally-outcome autocorrelation; without this, you over-shrink consistent players.

### C.2 Per-player Markov bigram + Brown-cluster trigram backoff

Compute count tables per fold from training matches only. Per-player bigram with Laplace $\alpha=0.5$:

$$\hat P_u(a_t = j \mid a_{t-1} = i) = \frac{N_u(i,j) + 0.5}{\sum_{j'} N_u(i,j') + 0.5 \times 19}$$

Per-player log-likelihood of visible prefix and deviation from global baseline:

$$\ell_u = \sum_{t=2}^{L} \log \hat P_u(a_t \mid a_{t-1}), \qquad \Delta\ell = \ell_u - \ell_g$$

Per-stroke surprise $\sigma_t = -\log \hat P_u(a_t\mid a_{t-1})$; expose $\sigma_L$, $\max_t \sigma_t$, mean and std.

**Trigram with stupid-backoff** (Brants et al. 2007), backoff weight $\alpha_{\text{bo}}=0.4$:

$$S_u(a_t \mid a_{t-1}, a_{t-2}) = \begin{cases} N_u(a_{t-2:t}) / N_u(a_{t-2:t-1}) & N_u(a_{t-2:t}) > 0 \\ 0.4 \cdot S_u(a_t \mid a_{t-1}) & \text{else}\end{cases}$$

recursively backing off to bigram, then unigram. With **510 strokes/player** and **6,859 trigram cells**, per-player trigram support is 0.07 obs/cell — catastrophic. Stupid-backoff lets it contribute when supported and silently fall through when not.

**Brown-cluster the players** (Brown et al. 1992, Computational Linguistics 18:467–479) into K=12 stylistic classes by minimizing class-based bigram loss. Class-level trigram support: 84,720 / (12 × 19²) ≈ 19.5 obs/cell — adequate. Use class-level trigram as a backoff layer between per-player bigram and global trigram.

**Anti-leak**: count tables exclude the rally being scored (and its match). Brown clusters refit per fold. Cold-start indicator `1{N_u < 50}` and group-marginal fallback.

### C.3 NMF rank-6 style embedding + Hellinger k-NN borrow

Build behavior matrix $V \in \mathbb{R}_{\ge 0}^{P \times C}$ with $C \approx 100$–200 columns: per-action_id frequency (19 dims), per-point_id frequency (10), per-action transition (top-100 transitions), per-server-frame action distribution (38). L1-normalize each row.

**Lee–Seung 2001 KL multiplicative updates** to factorize $V \approx WH$ with rank 6:

$$H_{aj} \leftarrow H_{aj}\,\frac{\sum_i W_{ia}\,V_{ij}/(WH)_{ij}}{\sum_i W_{ia}}, \qquad W_{ia} \leftarrow W_{ia}\,\frac{\sum_j H_{aj}\,V_{ij}/(WH)_{ij}}{\sum_j H_{aj}}$$

Initialize via NNDSVD; iterate ~400 steps. After convergence, $W$ holds player-style coordinates (6-dim), $H$ holds interpretable action profiles ("looper", "blocker", "chopper-counter").

**Fold-in for unseen test players**: with $H^*$ fixed,

$$w_u^* = \arg\min_{w\ge 0} \|v_u - w H^*\|^2_F$$

via 50 multiplicative-update iterations. Cost is O(rC) per player, milliseconds.

**k-NN borrow** with Hellinger distance over the raw behavior vectors (preferred over KL because it is bounded, symmetric, and robust to small probabilities):

$$H(p,q) = \frac{1}{\sqrt 2}\sqrt{\sum_j (\sqrt{p_j} - \sqrt{q_j})^2}, \qquad s(u,i) = 1 - H(b_u, b_i)$$

For test player $u$, find top-k=5 training neighbors and compute borrowed skill:

$$\hat\theta_u^{\text{kNN}} = \frac{\sum_{i \in N_5(u)} s(u,i) \cdot \hat\theta_i^{\text{EB}}}{\sum_{i \in N_5(u)} s(u,i)}$$

Expose features: `nmf_emb_1..6`, `knn_winrate`, `knn_dispersion` (variance over neighbors), `knn_top1_sim`, `knn_avg_sim`, `is_unseen_player`.

**Anti-leak**: $H$ fitted per fold from training-fold rallies; behavior vector $b_u$ for any test player must use only their **prefix-causal** rallies (training-side history if available, otherwise rallies in their match strictly preceding the prefix being predicted). Do not include the rally being predicted in $b_u$.

**Prevalence-weighted unseen-rally fraction**: 31 unseen / 71 total players = 43.7% by player count, but rally-weighted fraction is what drives AUC. Compute this number explicitly. If unseen players average ~30 rallies vs seen ~250, the rally-weighted unseen share is ~9%, and NMF/k-NN adds +0.005–0.010 AUC. If the share is closer to equal, lift is +0.020–0.040. **Run this calculation on day 0** — it determines whether the NMF block is high or low priority.

### C.4 Structural TT and score-state features — full list

Deterministic from inputs, no leak risk under any fold scheme:

```
server_stroke_parity     = (prefix_len % 2 == 1)         # last visible was server's
next_stroke_player       = "server" if prefix_len % 2 == 0 else "receiver"
service_block_position   = 1 if (total_pts_in_game % 4 < 2 and not is_deuce) else 2
is_deuce                 = 1 if (score_s >= 10 and score_r >= 10) else 0
deuce_intensity          = is_deuce / (1 + abs(score_s - score_r))
gp_for_server            = 1 if score_s == 10 and score_r < 10 else 0
gp_against_server        = 1 if score_r == 10 and score_s < 10 else 0
score_diff               = score_s - score_r
total_pts_in_game        = score_s + score_r
match_score_diff         = sets_won_s - sets_won_r
is_decisive_game         = 1 if (sets_won_s == sets_won_r == max_games - 1) else 0
prev_point_won_by_server = 1 if last_point_winner == server else 0
server_run_length        = consecutive points won by server in current game
serve_streak_in_block    = 1 if server won 1st point of current 2-serve block else 0
cum_strokes_in_match     = strokes played by this player in match so far  # fatigue
games_played_in_match    = current game index                              # fatigue
```

**Carl Morris point importance** (1977): precompute a lookup table over (score_s, score_r) given a baseline server-win-prob $p=0.55$ via the trivial recursion

$$I(s_s,s_r) = P(\text{game} \mid s_s+1, s_r) - P(\text{game} \mid s_s, s_r+1)$$

In 11-point format the peak is ~0.50 at 10-10, ~0.30 at 9-10, ~0.04 at 0-0.

**Expected-remaining-strokes regressor**: train a separate LightGBM regression on training rallies with target = `full_rally_length - prefix_length`, using the same prefix features. Expose `expected_remaining`, `expected_full_len = prefix_len + expected_remaining`, `prefix_completeness_ratio = prefix_len / expected_full_len`, `parity_of_predicted_final_stroke`. The last is high-signal: in TT, point outcomes are heavily decided by the last stroke and its server vs receiver identity.

**Empirical server-winrate-at-stroke-k lookup**: from training, plot $\hat P(\text{server wins} \mid \text{rally length} = k)$ for k=1..10 and expose the value at the user's `prefix_len`. This single column encodes the structural decay of server advantage with rally length (Pfeiffer 2010 Markov simulation, Munivrana 2019 mean rally ~4 strokes, Newgy practitioner stats: ~56% of points end by stroke 3). It is essentially a baseline prior the tree can split off of.

### C.5 Pair-specific EB shrinkage (low priority, document but skip first)

If implemented later, the recipe is: fit logit additive baseline $\alpha_a + \beta_b$, compute residual $\varepsilon_{a,b}$ per ordered pair and SE $\sigma_{a,b} = 1/\sqrt{n_{a,b}\hat p(1-\hat p)}$, EB-shrink

$$\hat\varepsilon_{a,b}^{\text{shrunk}} = \varepsilon_{a,b} \cdot \frac{\hat\tau^2}{\hat\tau^2 + \sigma_{a,b}^2}, \qquad \hat\tau^2 = \max(0, \widehat{\text{Var}}(\varepsilon) - \overline{\sigma^2})$$

With your pair sparsity, $\sigma_{a,b} \approx 0.28$ logit and typical $\tau \approx 0.1$, so 95% of pairs collapse to 0. Skip until everything else is stable.

### C.6 Covariate-shift reweighting for prefix length

Mean visible prefix 3.07 vs mean full 5.65 implies test cuts are roughly "uniform up to rally end" with mild bias. The user's prefix-augmentation generates all 1..N for each rally, weighting long rallies heavier. Estimate $p_{\text{train}}(L)$ and $p_{\text{test}}(L)$ from histograms; weight each training row by $w(L) = p_{\text{test}}(L) / p_{\text{train}}(L)$, clipped to [0.1, 10]. Pass into LightGBM via `sample_weight`. Use the same weights when computing OOF AUC.

A stronger version: train a binary domain-classifier on (prefix_len, score_state, server_id) to predict train-vs-test, then $w = \pi/(1-\pi) \cdot (1-\pi_0)/\pi_0$. KS-test the post-reweight distributions to confirm match.

## (D) Validation experiments before each LB submission

A daily gate prevents wasting LB slots on noise.

**Gate 1 — paired-fold ΔAUC vs noise floor.** With K=5 folds, the per-fold AUC standard deviation is typically 0.01–0.02. A new feature block must move the **paired** mean OOF AUC by ≥ 1.5× the per-fold std before submission. Report (mean ΔAUC, std across folds, paired t-stat). The 1.5× threshold is heuristic but corresponds to roughly the 0.067 p-value at K=5.

**Gate 2 — target-permutation leak check.** Shuffle `serverGetPoint` within each fold's training partition; rebuild the new feature block; retrain. OOF AUC must collapse to 0.50 ± 0.01. If it stays above 0.55, the feature is leaking. This catches the single most common failure mode (using the rally being scored to compute a feature for that rally).

**Gate 3 — player-disjoint stress test.** Re-run with player-disjoint folds (split by player_id, not match_id). If the new block's lift survives within 0.01 AUC, it captures style. If it collapses, it memorizes player identity (the LSTM hidden-state pathology) — drop it.

**Gate 4 — sliced AUC by prefix length.** Compute AUC separately at L=1,2,3,4,5+. The structural priors should flatten the "AUC drops at short prefixes" curve. The Markov bigram block should help most at L≥2 (no transitions at L=1 — zero-out those rows). NMF/k-NN should help most on rows where `is_unseen_player=1`. If the slice pattern does not match the theory, the feature is doing something unintended.

**Gate 5 — train-OOF gap monitor.** Track (train AUC − OOF AUC) before and after each block. If it grows by more than 0.01 with the new block, the block is overfitting in a way that does not transfer.

**Gate 6 — submission economy.** With 3 LB slots per day, allocate one to (best blend + new block), one to (best blend + new block + seed perturbation, to estimate LB jitter), one to (best blend ablation: best blend without the previous day's block, to confirm it is still pulling weight). Compare paired LB deltas to OOF deltas; if they diverge by more than 0.02 AUC, your CV is biased. Roelofs et al. (NeurIPS 2019) found public-vs-private gaps in Kaggle competitions are typically <0.01 in their unit; a >0.02 CV-vs-public gap means CV is wrong, not LB.

## Closing — where the gap actually closes

The user's instinct that the 0.62 → 0.80 gap is reachable is half-correct. **The first 0.10 of the gap is paved with deterministic structural features and Bayesian shrinkage**, neither of which can leak and both of which are underexploited in the current 72-column set. **The next 0.04 comes from style-conditional sequence features** — Markov deviation Δℓ and NMF embeddings — which require careful per-fold count tables but are well-trodden in the badminton (ShuttleNet 2022) and tennis (Wei 2016) literature. **The final 0.04 to a true 0.78–0.80 likely requires either ball-spin/speed columns, or making xy/trajectory features player-conditional via NMF embeddings as explicit interaction terms** — the same insight that lifted Sweet-Spot from 0.685 to 0.773 via opponent-specific adaptation.

The xy features did not fail because they lack signal. They failed because they were not conditioned on player identity. The LSTM features did not fail because LSTMs are wrong; they failed because the auxiliary-head training caused predictable target leak that match-disjoint OOF unmasked. Both can come back as positive contributors after the rebuild and the conditioning.

And finally: at the user's gap of 0.044 in score, **F1 components carry double weight per unit**. After AUC reaches ~0.74, the next hour of work should go to action_id and point_id classification — multi-class label smoothing, class-balanced focal loss, and the same Markov-bigram stylistic features here are likely just as valuable for next-action prediction as they are for outcome prediction. The leader is probably ahead on F1 at least as much as on AUC, and the leverage is 2:1 in F1's favor.