# Feature Engineering Briefing — Sgp Prediction Hits Per-Prefix AUC Ceiling at 0.62

## Context

Table tennis next-stroke prediction competition. We're predicting `serverGetPoint` (rally-level binary outcome) from a truncated visible prefix of the rally. Train data: 14,995 rallies / 84,707 strokes / 166 unique players. Test: 1,845 rallies / 5,668 strokes / 71 players (43.7% unseen). We've established `sgp = 1 - parity(N)` (TT rule: last stroke is the failing stroke; player who hits last loses) with 99.91% determinism in train, but test rally true length N is hidden by truncation.

A previous deep research pass (briefing at `docs/DEEP_RESEARCH_BRIEFING_2026-05-07.md`) derived an information-theoretic ceiling of **AUC ≈ 0.62** under random truncation given current feature space. We've hit that ceiling. Leaderboard #1 is at AUC ≈ 0.80 — implying they break one of (a) random truncation, (b) closed-feature-space, (c) cold-start collapse-to-0.5 assumption. We need feature engineering ideas that could break (b).

## Current Best Pipeline (T2)

- **Architecture**: LGB multi-class on R = N - k (remaining strokes count, capped at R_max=15). At inference, sgp = sum_{r: parity(L+r) = 0} P(R = r | features). DeepHit-style (Lee et al. AAAI 2018) — multi-class supervision is richer than binary parity.
- **Holdout protocol**: match-disjoint StratifiedGroupKFold(N=5, groups=match_id), per-fold winrate computed from non-holdout pool only.
- **Sample structure**: prefix augmentation — for each train rally with N strokes, draw 5 random k ∈ [1, N-1] prefixes; predict R = N - k.
- **OOF performance**: per-prefix AUC 0.623, per-rally aggregated 0.7835.

## Feature Set (72 features)

### T1 base (40 features)
**Player aggregates**: srv_winrate (Bayesian κ=50), srv_n_obs, rcv_loserate, rcv_n_obs, srv_seen, rcv_seen, match_seen.
**Score state**: score_self_start, score_other_start, score_diff_start, score_sum_start, is_deuce_start.
**Visible prefix length**: ctx_len, parity_bayes_unif (closed-form Bayes prior under uniform truncation).
**Last stroke attributes**: last_strikeId, last_handId, last_strengthId, last_spinId, last_pointId, last_actionId, last_positionId, last_action_group, last_point_depth, last_point_side, last_score_diff.
**Prev stroke attributes**: prev_actionId, prev_pointId, prev_action_group, prev_handId, prev_spinId.
**Prefix counts**: n_attacks, n_controls, n_defenses, n_serves, n_srv_attacks, n_rcv_attacks, n_srv_strokes, n_rcv_strokes.
**Misc**: sex, numberGame.

### T2 additions (32 features) — all marginal contribution
**XY coordinates** (continuous from POINT_DEPTH × POINT_SIDE):
- last_x ∈ {-1, 0, 1}, last_y ∈ {1, 2, 3}
- last_dx, last_dy, last_distance, last_angle, is_diagonal_last, is_cross_court_last

**Multi-stroke trajectory**:
- total_path_length, mean_seg_distance, max_seg_distance
- diagonal_count, diagonal_ratio
- recent_x_drift (over last 3 strokes), recent_y_drift

**Court coverage**:
- unique_zones, unique_zones_ratio
- n_corner_strokes (zones {1, 3, 7, 9}), corner_ratio, last_is_corner

**Dominance**:
- dominance_index = (n_srv_attacks - n_rcv_attacks) / k
- server_aggression_balance, receiver_aggression_balance (attacks - defenses, normalized)
- max_attack_chain (longest consecutive attack streak)
- last3_aggressor_score (signed sum: srv attack +1, srv defense -1, rcv attack -1, rcv defense +1)

**Score-pressure interaction**:
- pressure_x_dominance = (score_diff / 11) × dominance_index

**Position × point cross**:
- pos_x_depth = positionId × POINT_DEPTH[last_pointId]
- pos_x_side = positionId × POINT_SIDE[last_pointId]

**Strength aggregates**:
- strength_last, strength_mean, strength_max, strength_last_minus_prev

## Top-20 Feature Importance (T2, by LGB gain on last fold)

```
1.  rcv_loserate         10801   ← player aggregate (Bayesian)
2.  srv_winrate           9943   ← player aggregate
3.  last_strengthId       9921   ← stroke-level (raw cat)
4.  n_attacks             8189   ← prefix count
5.  strength_last         7145   ← stroke-level (NEW, redundant with #3)
6.  srv_n_obs             6563   ← confidence
7.  rcv_n_obs             6548   ← confidence
8.  ctx_len               6119   ← prefix length
9.  last_pointId          4615   ← stroke-level
10. sex                   4348
11. last_actionId         4338
12. mean_seg_distance     4261   ← NEW trajectory
13. last_score_diff       4090
14. score_sum_start       3888
15. numberGame            3723
16. strength_mean         3637   ← NEW aggregate
17. total_path_length     3635   ← NEW trajectory
18. parity_bayes_unif     3432   ← prior
19. n_controls            3369
20. max_attack_chain      3113   ← NEW dominance
```

**Take**:
- Player aggregates (rcv_loserate, srv_winrate) dominate.
- Stroke-level raw categoricals (strengthId, pointId, actionId) capture much of what trajectory features encode redundantly.
- Among 32 new features, only ~5 (mean_seg_distance, total_path_length, max_attack_chain, strength_mean, dominance proxies via attack counts) reach top-20.
- Per-prefix AUC moved from T1's 0.62 to T2's 0.623 (+0.003) — within noise.

## Hypothesis: We've Saturated Current Feature Space

The visible prefix encoding (per-stroke {strikeId, handId, strengthId, spinId, pointId, actionId, positionId} + score state + player IDs) appears near-saturated for sgp prediction at per-prefix AUC ~0.62. Reasons:

1. **Redundancy with raw categoricals**: LGB already learns interactions implicitly. New continuous derivatives (xy, trajectory) add limited fresh signal.
2. **Rally-outcome information is sparse**: Even visibility-rich prefixes don't strongly predict outcome because rally continuation has high inherent randomness.
3. **Cold-start dominates the ceiling**: 43.7% of test rallies have unseen-by-train server players → per-pair Bayes-optimal AUC for those is ~0.5, dragging overall AUC down.

## What Could Break the Ceiling — Asks for External Thinking

We need feature engineering ideas that capture **fundamentally new structural information** the current set misses, not derivatives of what we have. Specifically:

### Dimension 1 — Within-Rally Sequential / Symbolic Patterns

We treat the visible strokes as a bag-of-features (with stroke-level extracts) plus aggregate counts. But rally dynamics are sequential. Things we haven't captured:

- **Stroke n-grams**: bigrams (action[i], action[i+1]), trigrams over strokes/zones/strengths. Specific patterns may correlate with outcome (e.g., "attack → defense → attack" by server = winning rhythm).
- **Motif detection**: repeated sequences (e.g., back-and-forth at same zones). Are recurrent motifs predictive?
- **Rally phase encoding**: serve / opening (k=1-3) / middle / late, with phase-conditional features.
- **Symmetry / periodicity flags**: is the rally rhythmic (alternating zones) or chaotic?
- **Style-mismatch indicators**: server is attacking at corners but receiver returns to center (server controlling) vs receiver returns to corners too (battle).

**Question**: Are there published TT/badminton rally-classification features we're missing? Specific n-gram or motif methods that worked elsewhere?

### Dimension 2 — Player Style and Matchup

We only use player-level aggregate winrate. Players have STYLES (attacker / counter-puncher / defender). Matchup outcomes may depend on style compatibility, not just average skill.

- **Per-player style fingerprint**: action-distribution histograms over their training rallies (forehand-attack ratio, defense ratio, preferred zones, preferred strengths).
- **Server vs receiver style mismatch**: cosine distance between style histograms. Maybe attacker-vs-attacker rallies are different from attacker-vs-defender.
- **Per-player effectiveness as server / receiver split**: not just overall winrate, but conditional winrate when serving vs when receiving.
- **Right-hand vs left-hand mismatch**: handId differences may affect zone preferences.

But: **player overlap leakage risk**. Our previous attempt §2.10 (A v2 historical player stats) failed badly (LB -0.024) because per-fold style features had information leakage via player overlap. We've abandoned generic per-player aggregate stats. **How can style features bypass this trap?** Per-fold computation, leave-one-rally-out, or careful match-disjoint aggregation?

### Dimension 3 — Score-State Importance

We use score_diff, score_sum at rally start. But specific score states matter differently:

- **Set point / match point flags** (10-10, 10-9, etc. with TT 11-point rules).
- **Game-leading vs trailing context**: in game-trailing situations, players may take more risks → outcome more random.
- **Score-state × player handedness × dominance**: may have interaction effects.
- **Post-game rest / fatigue proxies**: rally i in game j may correlate with rally i in game j' if same match.

What score-related features have been shown to lift AUC in tennis/TT outcome prediction at the rally level (not match level)?

### Dimension 4 — Test-Rally-Specific Features

Test rallies have full rally_id and match info. Even though we can't use sgp from test, we can use test rally STRUCTURE:

- **Match-level test priors**: for each test match, average visible L, attack rate, etc. — features that summarize "this match's character".
- **Rally-position-in-match**: is this rally early or late in its match? Could correlate with player fatigue or lead/trail dynamics.
- **Cross-rally consistency**: if a server has multiple test rallies, their visible-context features may be consistent (or chaotic).

Is there a principled way to use cross-rally test signals without leaking the test labels we're predicting? Self-supervised approaches?

### Dimension 5 — Latent / Learned Representations

Hand-crafted features may miss combinatorial structure. Latent representations may help:

- **Autoencoder embedding of full rally sequences**: train AE on train rallies (full N strokes), embed visible prefix as feature.
- **Word2Vec-style stroke embeddings**: treat stroke types as tokens, learn embeddings from rally co-occurrence.
- **Graph embeddings**: build player-stroke-pattern graph, embed via random walk or GCN.
- **Cluster-based features**: k-means on stroke-feature space, use cluster IDs as categorical features.

Has any sports prediction paper benefited from deep latent representations as inputs to gradient boosting? When does this help vs hurt?

### Dimension 6 — Truncation-Structure Probes

We've shown KS test rejects simple deterministic truncation rules (uniform, ⌈N/2⌉, N-2). But the truncation might still be **partially structured** in subtle ways:

- **Conditional cut-points**: maybe the organizer cuts at "before the next attack" or "before strength changes" — yielding a specific signal in last visible stroke type.
- **Per-match cut policies**: maybe each match has its own cut rule.
- **Probabilistic cut hint**: the visible last stroke type's distribution differs from all-strokes distribution, indicating truncation rule.

Specific features to engineer:
- **last_stroke_type_distribution_z-score**: how unusual is last visible stroke vs marginal stroke distribution?
- **per-match last-stroke pattern**: cluster matches by their visible last-stroke distribution.

What test-time structure-mining approaches have you seen? Probabilistic cut-point estimation?

## Specific Asks

1. **Rank these 6 dimensions** by expected AUC lift (+0.0X) for our setup. Defend with reasoning.
2. **Propose specific features within Dimension 1 (sequential patterns)** that could capture sgp signal we're missing. We're skeptical about pure n-grams (sample size, overfitting). Better candidates?
3. **Style-fingerprint features in Dimension 2**: how to engineer per-player style features that DON'T fall into the player-overlap leakage trap (§2.10 failure)?
4. **Latent representations (Dimension 5)**: when does AE/GCN-embedding help downstream LGB on small (~70k) tabular tasks? Cite papers if possible.
5. **Truncation probes (Dimension 6)**: realistic methods for inferring partial truncation structure from test data alone?

## Constraints

- LGB on tabular features (we want to keep this stack; only the feature engineering layer changes).
- Match-disjoint OOF protocol must be preserved. No per-fold leakage allowed.
- Implementation budget: 1-2 hour features have priority. We can spend 6-8 hours if expected lift is +0.05+ in AUC.
- Test-distribution-weighted per-prefix AUC (not per-rally aggregated) is the only valid OOF metric — we got burned earlier by aggregation-artifact OOF.

## Output Format

Per dimension: (a) ranked priority, (b) most promising specific feature spec, (c) implementation cost in hours, (d) expected per-prefix AUC lift with reasoning, (e) failure mode I can detect early.

Don't suggest dimensions we already covered (xy, trajectory, dominance) — those are saturated. We need NEW directions.
