# Hostile Audit of a Saturation Diagnosis: Where the Argument Holds and Where It Breaks

**Bottom line up front.** The team is right that ~80% of the documented variants are genuinely closed under their plug-in absorption argument and the Pearson-0.7 ensemble math. They are wrong on **two specific load-bearing claims**: (i) Argument B's Vapnik-monotonicity step from length-strat-cal failure to training-time IW failure is mathematically invalid because the function classes are not nested; (ii) the 29-param plug-in is fit against the **OOF label marginal**, not the **estimated test marginal** — this is a sub-optimal point inside their own additive function class, and label-shift adaptation (Saerens 2002 / BBSE / Alexandari 2020) corrects it without leaving the class. Three new directions survive their constraints: **estimated-test-marginal recalibration**, **kNN/retrieval over BiLSTM hidden state as a stand-alone second head (not λ-interpolated)**, and **length-conditional MoE with a non-sequence L=1 expert** that escapes the Pearson constraint by partitioning the evaluation set rather than ensembling on it. Expected aggregate ceiling lift is +0.003–0.010 macro-F1, mostly from #1 and #3. If none ships through Debate#1, accept rank #3 and write the final report — Argument D's Laplace prior is informative; the 8th plan's success probability is in the 10–25% range, not the 0% the team's framing implies.

---

## 1. Diagnosis audit

### Claims I agree with

The closures of **mirror augmentation** (M6 — pointId is receiver-handedness-normalized; spatial flip corrupts semantics — verified against TacticAI's D2 symmetry which is football-pitch-specific), **bidirectional MLM pretraining** (M8, structurally wrong for causal next-token; downstream gap is real), **homogeneous bag > 2** (M15, empirical -0.0087 LB), **A v2 historical player_stats** (per-rally LOO leaks player identity), **same-domain TacticAI transfer** (DeepMind's gain is on football corner-kick set-piece graph classification, not on sequential turn-based 19-class macro-F1; scale and task differ), **stacking meta-learner on bag OOF probs** (verified — at 5 base models × 14k OOF rallies the meta-learner can survive only with strong convex-combination + L2 constraints, which collapses to Caruana 2004 forward selection anyway), and **external player ELO / reverse-matching** (rule-prohibited).

The same-domain reference search confirms one uncomfortable fact: **no published racket-sport stroke prediction paper reports macro-F1**. ShuttleNet (43,191 strokes, CE-only), MuLMINet (33,612 strokes, CE+MAE), RallyTemPose (top-k accuracy with documented heavy class skew — 12% Clear acc), Hsu/Wu MJSSM 2026 (10,517 strokes, single focal player Lin Yun-Ju, top-1/top-5 accuracy only). Every architecture lift in this corpus is on aggregate metrics dominated by majority classes. The team's macro-F1 framing is more rigorous than any published baseline, which makes published lift numbers structurally over-optimistic when remapped to macro-F1.

### Where they prematurely closed the door

**Argument B is mathematically invalid.** The Vapnik-monotonicity step requires nested function classes. Post-hoc length-stratified calibration learns:

```
F_post = { x ↦ argmax( W_ERM·φ_ERM(x) + b(class, length(x)) ) }
```

Training-time importance weighting produces a different φ:

```
F_IW = { x ↦ argmax( W_IW·φ_IW(x) + b ) }
```

These classes are not nested. φ_IW spans different directions in feature space than φ_ERM. **Byrd & Lipton 2019** (ICML) and **Xu, Ye & Ruan 2021** show explicitly that with L2/BN/finite training, IW changes the converged classifier's margin distribution, not just its decision-boundary offset. **Shimodaira 2000 / Sugiyama 2007** prove that under model misspecification, IW gives a different (and better) minimizer than ERM — and a 2-layer BiLSTM at 84k strokes on a 29-way macro-F1 problem is almost certainly misspecified relative to the true conditional. The team's empirical 174-param failure is also suspicious: at 6 length-bins × 29 classes with mean L=5.65, the L=1 and L≥10 bins have very few OOF samples, and 174 params on macro-F1 OOF objective is a textbook overfit-on-OOF / regress-on-LB pattern that does **not** generalize to a statement about training-time IW. Argument B should be retracted; class-balanced CE and LDAM-DRW (Cao 2019) at training time remain unkilled.

**Argument C is asserted, not demonstrated.** Match-disjoint CV is not player-disjoint CV (M12 explicitly notes 24% train players in test, 63% test players in train, but matches don't overlap). The team has not run the diagnostic that distinguishes the two hypotheses: (a) per-rally OOF accuracy decomposed by `is_player_seen ∈ {0,1}` × `length_bucket`; (b) a player-stratified CV protocol whose OOF/LB gap is compared to the match-stratified gap. Without those numbers, "length is not the primary gap source" is a plausible prior, not a finding. The 36.5% unseen-player rate is mid-range relative to domain-generalization benchmarks (DomainNet/PACS run 25–75% unseen) — neither trivially small nor obviously dominant. **The probability that length-shift is materially contributing (≥30% of the gap) is high enough to invalidate Argument C as a load-bearing closure.**

**The plug-in absorption argument has a critical caveat the team has not exploited.** The plug-in is fit on **OOF logits with OOF label marginals as the macro-F1 target**. The optimal additive bias under per-class label shift is c_y = log(q_test(y) / p_OOF(y)) — formally inside the team's own 29-param additive class — but their current vector is the wrong point inside that class because they used p_OOF(y) as the implicit target marginal. **Saerens, Latinne & Decaestecker 2002 EM**, **BBSE (Lipton et al. 2018)**, and especially **Alexandari, Kundaje & Shrikumar 2020** ("Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation") provide consistent estimators of q_test(y) from unlabeled test predictions. Re-pointing the optimization to the estimated test marginal is technically still per-class additive but is **not the same vector** the team currently ships. Given the test rally-length distribution shift (mean L=2.90 vs train 5.65; 32% test L=1, 24% test L=2 vs train averaging ≈18% L=1, ≈14% L=2), it is structurally inevitable that q_test(y) ≠ p_OOF(y) for several action and point classes — the serve, receive, and short-rally-terminating classes are systematically over-represented in test.

### Per-class vs per-sample distinction (the absorber's blind spot)

Mathematically the plug-in is the function class **{ℓ(x) + b : b ∈ ℝ^29}**. Any correction g(x,y) reducible to b(y) is fully absorbed in the limit of OOF/test alignment. **Per-sample corrections** g(x,y) = h(x)·δ(y∈S(x)) + r — where the correction's *value at class y* depends on x — are provably outside this class and cannot be absorbed by any setting of b. This is what the team should be hunting. Importantly, "per-sample" must mean the correction's *cross-class ranking* depends on x, not just that the correction's magnitude varies with x. Sample-dependent temperature scaling (Joy 2023) is **rank-preserving within sample** and therefore useless for hard-label macro-F1 — it lifts ECE but not argmax accuracy. The team should not be fooled by the literature here.

---

## 2. New direction proposals (three; one defensive, two probative)

### Proposal P1 — Estimated-test-marginal plug-in re-fit (label-shift adaptation)

**Mechanism.** Run Saerens EM or BBSE-soft on the current BiLSTM bag2 logits over the test set:

1. Compute mean softmax over test → empirical predicted marginal m̂(y).
2. Estimate confusion matrix C[ŷ,y] from OOF (rows = predicted, cols = true).
3. BBSE: q̂(y) = C^{-1} m̂. Or Saerens EM: iterate q̂^{(t+1)}(y) ∝ Σ_x p(y|x) · q̂^{(t)}(y) / p_OOF(y), normalize, until convergence (~20 iters).
4. Re-optimize the 29-param additive bias **on OOF logits**, but with samples reweighted by w(y) = q̂(y) / p_OOF(y) when computing macro-F1. The optimizer (current Nelder-Mead or coordinate-descent) is unchanged; only the objective's per-sample weight changes.
5. Apply the new b_test at inference. Plug-in vector is the only artifact swapped.

**Why it escapes Section-7 saturation criteria.** (1.a) Yes, this *is* a per-class adjustment — but it is not the team's current vector. The team's claim "plug-in absorbs 70-85%" presupposes the plug-in is targeted at the test marginal, which it is not. This proposal makes that presupposition true. (1.b/c/d) trivially N/A — no new architecture, no pretraining, no aggregated-player-stat features.

**Same-domain reference.** No racket-sport paper. Closest reference: **Alexandari et al. 2020** report 5–15% relative classification-error reduction on label-shift benchmarks (CIFAR-10/100 with imposed class skew) at scales of 50k samples, comparable to the team's scale. **Discount**: Alexandari's setting has ground-truth shift control; the team's shift is observational. Discount factor ~0.5.

**Predicted lift.** Conditioning on (a) 84k stroke scale, (b) 19+10 class budget, (c) plug-in *already fitted* to OOF marginal: the reachable gain depends entirely on the magnitude of q_test(y) − p_OOF(y). The structural argument: test mean rally length 2.90 vs train 5.65 means each test rally contains a higher proportion of "early-rally" strokes. Among the 19 action classes, the serve, return, and short-rally-terminator classes are over-weighted in test by the length-distribution-induced sampling. A back-of-envelope: if 5–8 of the 19 action classes have q/p ratios in [0.7, 1.6], the macro-F1 gain from re-pointing the plug-in is in the **+0.002 to +0.006 single-seed OOF lift** range; ensemble lift is essentially additive (the plug-in is applied post-bag). The team should run the diagnostic `mean(softmax(test_logits)) per class − empirical p_OOF(y)` first; if every class shifts by < 1% in absolute, this proposal dies and they should report so.

**Cost.** 4–8 hours. Stage A probe is the diagnostic itself (30 minutes). Verification fixture: re-fit b on a held-out OOF fold simulating estimated marginal vs true OOF marginal; the gain on simulated test should be > +0.001 macro-F1 to merit a real submission.

**Failure modes.**
- *FATAL candidate*: "BBSE confusion matrix is ill-conditioned at 29 classes with imbalanced support; Saerens EM diverges or returns near-uniform q̂." Mitigation: shrinkage estimator on C, or use the soft-EM variant which is stable. Diagnostic: condition number of C; if > 100 fall back to the simpler estimator.
- *HIGH candidate*: "If test predictions are themselves miscalibrated (which they are — that's why plug-in exists), m̂ is biased and q̂ is biased. You're chasing a fixed point of a biased operator." Mitigation: Alexandari's bias-corrected temperature scaling step before the EM/BBSE; this is the precise reason that paper outperforms vanilla BBSE. Without it, this proposal is fragile.

### Proposal P2 — Stand-alone kNN-over-hidden-state head, NOT λ-interpolated, ensembled via stacking with convex constraints

**Mechanism.** Build a datastore from the BiLSTM bag2 hidden states:

1. For every (rally, t) in train, store key k = h_{rally,t} ∈ ℝ^192 (last-layer pre-MLP-fusion BiLSTM hidden, post-LayerNorm), value v = y_{t+1}. ~70k entries × 192 float32 ≈ 54 MB. FAISS `IndexFlatL2` is exact and sub-second on this size.
2. At inference, retrieve k=64 nearest neighbors. Compute p_kNN(y|x) ∝ Σ_{i∈N} 1[y=v_i] · exp(−d_i / τ), τ tuned on OOF.
3. **Critical departure from Khandelwal**: do NOT λ-interpolate p_kNN with p_LSTM. Instead use p_kNN as a **stand-alone second head**, then combine via Caruana-2004 forward selection on OOF macro-F1 with non-negative convex weights. λ-interpolation produces ρ ≈ 0.85–0.92 with the parametric LSTM (Drozdov et al. 2022 confirm this). The stand-alone p_kNN is empirically lower-correlated (ρ ≈ 0.55–0.70) because L2-NN in BiLSTM hidden space disagrees with parametric softmax precisely on the long-tail patterns that drive macro-F1.
4. Apply the team's plug-in calibration on the final ensembled logits (which is now a partially-x-conditional object).

**Why it escapes the absorber.** Per-sample reranking is mathematically forced: the retrieved neighbor set N(x) depends on x, not on y. Two test samples x₁, x₂ with identical p_LSTM logits but different N(x) have different p_kNN distributions over the 19 classes. The cross-class ranking induced by the kNN component varies with x. **Plug-in calibration applied after stacking will absorb the per-class component of the kNN distribution (essentially recovering the kNN's marginal class shift) but cannot absorb the per-sample reranking.** Expect plug-in to absorb 50–70% of the gross kNN macro-F1 contribution, leaving a residual 30–50% that is genuine x-conditional lift.

**Why it escapes Pearson.** Plan T (BiTransformer) gave Pearson 0.91 because it is gradient-trained on the same loss with similar inductive bias. kNN is **non-parametric, instance-based, gradient-free** — its error structure is driven by datastore coverage and metric pathology, not representation-learning idiosyncrasies. Khandelwal 2020 → Drozdov 2022 → Xu 2023 → Yamauchi 2025 line establishes that retrieval contributes information not explainable by parametric LMs at the same scale. Predicted Pearson with BiLSTM: **0.55–0.70** for stand-alone p_kNN; with LightGBM: **0.40–0.55** (different feature space).

**Section 7 criteria.** (1.a) Not a loss reformulation; the BiLSTM is unchanged. The combiner is not a per-class adjustment because p_kNN itself is per-sample. (1.b) Pearson well below 0.85, predicted ~0.6. (1.c) No pretraining; the encoder is the existing BiLSTM, frozen, used only to build the datastore. (1.d) Not aggregated player statistics; the datastore is per-stroke, not per-player.

**Same-domain reference.** No racket-sport paper. Closest references are **kNN-LM (Khandelwal 2020)** at WT103 scale (103M tokens, 250k vocab) and **RAC (Long et al. CVPR 2022, retrieval-augmented classification for long-tail)** at Places365-LT scale (1.8M images, 365 classes). **Discount factor**: kNN-LM's headline 18.65 → 16.12 ppl on WT103 is at scale ~1500× the team's. Yamauchi et al. 2025 ("Long-Tail Crisis in Nearest Neighbor LMs") show kNN-LM gains concentrate on **head tokens with high-similarity contexts**, not on the long tail. With 19 action classes (effectively all "head" by NLP standards), the gain mechanism shifts from "vocabulary expansion" (NLP) to "long-tail context retrieval" (this task) — a different mechanism entirely. Expected gain envelope: 25–40% of kNN-LM's NLP lift translated to macro-F1 ≈ 0.003–0.007 absolute.

**Predicted lift.** Single-seed OOF: **+0.002 to +0.006 macro-F1** on the action head, **+0.001 to +0.003** on the point head (smaller because point has only 10 classes and the class-conditional retrieval signal is weaker). Ensemble lift (kNN head added to current bag): **+0.004 to +0.009 macro-F1** because the kNN head is the first low-Pearson element in the team's ensemble — Caruana 2004 lift scales with diversity. The current LightGBM contribution is small (weight 0.02–0.15) precisely because LGB and BiLSTM share the 49-static-features substrate; kNN over BiLSTM hidden state introduces a genuinely different decision surface.

**Cost.** Stage A probe (4 hours): build flat FAISS index over OOF-fold-1 train hidden states, evaluate retrieval-only macro-F1 on OOF-fold-1 val, sweep k ∈ {16, 32, 64, 128} and τ. Gate: if stand-alone p_kNN macro-F1 > 0.30 (i.e., better than uniform-prior random) AND Pearson with bag2 < 0.75 on the validation fold, proceed. Plan investment (10–14 hours): full datastore build over all train, OOF predictions for stacking, Caruana forward selection with non-negative weights, single-seed LB submission to verify gap test (M14).

**Failure modes.**
- *FATAL candidate*: "At 70k tokens with 19-class vocabulary, the kNN distribution is a near-deterministic copy of BiLSTM's argmax — it just memorizes what BiLSTM already learned." This is the strongest critique. Defense: empirically test on Stage A probe — measure ρ(p_kNN, p_LSTM) on validation. If ρ > 0.8, the proposal dies cheaply (4-hour kill). If ρ ∈ [0.55, 0.75] as predicted, the proposal survives. Note that Drozdov 2022 §3.2 shows kNN agrees with parametric LM on confident predictions and disagrees on the long tail — exactly the macro-F1-relevant region. Yamauchi 2025 caveats apply but the mechanism for *this* problem is long-tail context retrieval, not softmax bottleneck breaking.
- *HIGH candidate*: "kNN-LM gain at WT103 scale (103M tokens) does not transfer to 70k tokens — Figure 2b of Khandelwal shows λ optimum collapsing to near-zero as datastore shrinks; at 70k entries the optimal interpolation weight is in the 0.05–0.15 range, capping lift below the +0.010 ship gate." Defense: this proposal does not λ-interpolate; it uses p_kNN as a stand-alone head whose contribution is determined by Caruana stacking weights, not λ. The Khandelwal scale-collapse argument applies to interpolation weight, not to stand-alone usage. Stage A probe directly measures the stand-alone OOF macro-F1, sidestepping the extrapolation.

### Proposal P3 — Length-conditional mixture-of-experts with non-sequence L=1 expert

**Mechanism.** Hard-route on prefix length:

```
L=1  → Expert_A: GBDT (LightGBM) on serve features only
                 (server_id, server_handedness, score_state, side, set, server_position)
                 + a Bayesian-shrinkage class-prior over receiver actions per server.
L=2  → Expert_B: small Transformer-encoder on (stroke_1, static_features), 1 layer, 64 hidden.
L≥3 → Expert_C: current BiLSTM bag2, unchanged.
```

Apply plug-in calibration **per-expert independently** (29-param vector per expert, total 87 params, but each fit only on the corresponding length-bucket OOF — sample size is sufficient: train has ≈ 4.7k length-1 prefixes if the team's full train has typical length distribution, ≈ 3.6k length-2 prefixes, the remainder for ≥3).

**Why this escapes the Pearson constraint entirely.** This is the cleanest mathematical point in the report. The Caruana 2004 forward-selection threshold and the Krogh-Vedelsby 1995 ambiguity decomposition compute Pearson over **the same evaluation set across models scoring the same examples**. With hard length-routing, each expert only ever scores its own slice — they share **zero examples** — so Pearson is undefined and the lift comes from per-slice optimization, not output decorrelation. Total error is `Σ p(L_bucket) × err_bucket`, with no covariance term.

**Why "conditional ensemble routing seen/unseen" failed but length-routing might not.** The team's prior failed routing experiment used `is_player_seen ∈ {0,1}`, which creates **distribution shift across the boundary** — the unseen-router has to generalize from a smaller, biased training set. Length is observable, deterministic, and creates **causal regime shift**: an L=1 prefix has no return-pattern history; an L=5 prefix does. This is not a routing convenience, it's a structural difference in the conditional p(y|x) generative process. Coaching literature (Malagoli Lanzoni 2014, Tamaki 2017, Frontiers Psychology 2023) confirms server-tactical-advantage is most significant in the first 3 strokes; serve-receive transitions are statistically distinct from mid-rally transitions.

**Why the L=1 expert specifically is non-trivial.** At L=1, the 32% of test rallies have only the serve as input. A BiLSTM trained on full rallies averages gradients across all lengths and produces logits at L=1 that are essentially `MLP(static_features ⊕ serve_embedding)` — i.e., a tabular classifier with extra steps. A *purpose-built* tabular GBDT or Bayesian-shrinkage hierarchical multinomial model on serve features should match or beat the BiLSTM on this slice, possibly by 2–5 percentage points of action-class macro-F1. Hsu/Wu 2026 MJSSM TT data and ShuttleNet's τ=2 vs τ=8 ablations together imply the L=1 / L≤2 regime has materially different predictive ceilings than mid-rally.

**Section 7 criteria.** (1.a) The L=1 expert is a different architecture, not a per-class adjustment. (1.b) GBDT and small Transformer are different families from BiLSTM; but more importantly, they evaluate on disjoint slices, so Pearson is N/A. (1.c) No pretraining. (1.d) Serve features per-stroke are not aggregated player statistics; player_id is used as a categorical feature inside the GBDT (which the team already does in their LightGBM ensemble side), not as an aggregated style score.

**Same-domain reference.** No published racket-sport paper does length-conditional routing on stroke prediction. The closest signal is ShuttleNet's τ-ablation (CE 1.9802 at τ=8, 2.0755 at τ=2 on the same model — a ~5% CE penalty as conditioning shrinks), which strongly suggests a separate L=1/2 expert would extract gain that a unified architecture leaves on the table. Soccer-event literature (Seq2Event KDD 2022, NMSTPP 2024) does not partition by length but operates at scales where short-prefix events are a small fraction of total. Discount factor: 0.5–0.6 because the empirical regime is not directly tested.

**Predicted lift.** Conditioning on (a) 32% of test rallies at L=1, 24% at L=2, (b) plausible 2–4 point macro-F1 improvement on the L=1 slice from a purpose-built model, (c) plug-in absorption of per-class shift within slice but not cross-slice routing: total expected gain is `0.32 × Δ_L1 + 0.24 × Δ_L2 + 0.44 × Δ_L≥3`. With Δ_L1 ≈ +0.02 to +0.04 macro-F1 (plausible for a tabular model purpose-built on a regime where the BiLSTM is misallocated), Δ_L2 ≈ +0.005 to +0.015, Δ_L≥3 ≈ 0 (no expert change), the aggregate is **+0.008 to +0.017 macro-F1** — comfortably above the +0.010 ship gate in the optimistic case. **However**: this prediction assumes the team has not already empirically validated that the BiLSTM's per-length F1 is near-optimal on each slice. They should run that diagnostic first.

**Cost.** Stage A probe (3 hours): take current BiLSTM bag2 OOF predictions, decompose macro-F1 by `length(prefix) ∈ {1, 2, ≥3}`, and report per-bucket per-class F1. If L=1 macro-F1 is within 0.005 of the L≥3 macro-F1, this proposal is dead — the BiLSTM is already extracting most of the L=1 signal. If L=1 macro-F1 is 0.02+ below L≥3, proceed. Plan investment (12–18 hours): build the GBDT L=1 expert and the small Transformer L=2 expert with proper match-disjoint StratifiedGroupKFold; per-expert plug-in calibration; LB gap test on single submission.

**Failure modes.**
- *FATAL candidate*: "The BiLSTM's L=1 macro-F1 is within 0.005 of its L≥3 macro-F1 — the static-features path already extracts the L=1 signal." This kills the proposal cheaply. Diagnostic is the Stage A probe itself.
- *HIGH candidate*: "Per-expert plug-in introduces 87 free params instead of 29; in the team's experience 174 length-strat-cal params overfit OOF and regressed on LB by 0.0037. 87 is between 29 and 174 — same overfit pathology." Defense: per-expert plug-in is fit on disjoint slices, so each 29-param vector has its full slice-OOF as fitting data. 174-param length-strat operated on the joint set with rare-bin overfitting. The two are structurally different. Still, careful regularization (early-stopping the optimizer, OOF-of-OOF protocol) is required.

### Why I am not proposing other directions

**Mamba/S4/S5**: empirical Pearson with BiLSTM at L=5.65 is predicted 0.80–0.92. Selective state-space's entire inductive-bias advantage is on long sequences (Gu & Dao 2023 ablations all at context ≥ 1024; DeciMamba arXiv:2406.14528 documents limited effective receptive field). At L=5.65 the selection mechanism has nothing to forget. Plan T's 0.91 Pearson on BiTransformer is the canonical evidence that "fancy sequence model on short sequences collapses to same effective hypothesis class". Skip.

**GNN over rally-as-path**: trivially equivalent to BiLSTM in expressivity. Pearson likely > 0.85. Only a *cross-rally* graph (player-history edges spanning rallies) would decorrelate, but this re-introduces the aggregated-player-statistics problem (1.d) the team has already killed with the A v2 result.

**TENT, MEMO**: TENT is BN-tied; BiLSTM has no BN. MEMO needs label-preserving augmentations; M6 blocks the natural ones. Both reduce to entropy minimization, which on confident logits with an imbalanced class distribution typically *worsens* macro-F1 (entropy-min reinforces head classes).

**Sample-aware F1 decision rules (Lipton/Elkan 2014, Dembczynski EFP)**: per-sample reranking that does escape plug-in, but documented gains over plug-in for these methods are < 1% relative. At 0.445 macro-F1 baseline that's +0.001–0.004 absolute. Mention to the team but not as a primary proposal — the cost (1–2 days) is high relative to the ceiling.

**Negative correlation learning of co-trained BiLSTMs**: forces low Pearson by construction (Liu & Yao 1999), but the resulting members live in the same hypothesis class as the base BiLSTM. The lift from NCL ensembles in published sequence classification work is +0.5–1% relative on aggregate metrics — likely below ship gate at 0.445 macro-F1.

**Joachims SVM-perf, SoftF1/Lovász surrogates**: in the limit converge to per-class threshold solutions equivalent to plug-in. Gain over CE+plug-in empirically < 2% in published settings, mostly recoverable post-hoc.

**Autoregressive TTT**: not foreclosed by M8 (TTT does not require bidirectional pretraining; AR-TTT exists at LLM scale per arXiv:2505.20633, 2505.19475). Mathematically produces per-sample reranking. But empirical lift is highly uncertain at 70k scale on a domain with mild shift, instability per Darestani 2022 is real, and the cost (3–5 days) is high. Lower priority than P1/P2/P3 but the only "research bet" worth flagging.

---

## 3. Anti-cargo-cult filter

**P1 (test-marginal recalibration).**
- Relabeling of a prior variant? **No**. The team's plug-in is fit against OOF marginal; this proposal targets the estimated test marginal. Distinct.
- Same-domain reference matches constraints? **Partial**. Alexandari 2020 is at comparable scale (50k samples, similar class count) but on imposed CIFAR shift, not observational sport-temporal shift. Discount factor 0.5 applied.
- Double-counting cross-domain gains? **No** — Alexandari's gains are within the same per-class additive function class the team uses; this is in-class re-pointing, not borrowed from large-scale NLP.
- Plug-in absorbed? **The objection is exactly inverted**. The current plug-in *is* the absorber, fit on the wrong target. P1 *is* the better plug-in. The 70-85% absorption claim presupposes target alignment.
- Known failure pattern? Vapnik-monotonicity does not apply (P1 is not a strict-superset corrector). Caruana lowest-diversity tier does not apply (no ensembling). Pearson > 0.85 same-arch convergence does not apply.

**P2 (stand-alone kNN head).**
- Relabeling? **Distinct from variant 3 (test pseudo-labeling)** — pseudo-labeling generates test labels from model confidence; kNN retrieves train labels via similarity. Distinct from variant 6 (autoencoder embedding) — that was a learned compression of player stats; this is non-parametric retrieval over per-stroke hidden states. Distinct from variant 1 (logit adjustment) — kNN distribution is x-conditional, not y-additive.
- Same-domain reference? **No racket-sport reference exists**. Cross-domain references (Khandelwal 2020 at 1500× scale, Long 2022 RAC at 22× scale) come with stated discount factors. The Yamauchi 2025 long-tail caveat is acknowledged.
- Double-counting? Khandelwal's 18.65 → 16.12 ppl on WT103 is **not** the lift estimate — that is at 1500× scale and on perplexity, a different metric. The +0.003–0.007 macro-F1 estimate is conditioned on this team's scale and metric, not extrapolated.
- Plug-in absorbed? **Argued mathematically not absorbed**. Per-sample reranking via x-dependent neighbor sets. Empirical absorption test is the Stage A probe (measure post-cal residual lift); cheap to falsify.
- Known failure pattern? Pearson predicted < 0.7 (the whole point); Vapnik N/A; Caruana directly applicable in the favorable direction.

**P3 (length-conditional MoE).**
- Relabeling? Adjacent to "conditional ensemble routing seen/unseen" (variant: routing on player-novelty), which the team killed because the routing variable creates train-test distribution shift. Length-routing is *not* this — length is observable on test without distribution shift, and the regime change is causal not statistical. Distinct.
- Same-domain reference? **No racket-sport paper does this**. ShuttleNet's τ-ablation provides indirect evidence that length matters for prediction quality. Discount factor 0.5–0.6.
- Double-counting? No cross-domain large-scale lift is claimed.
- Plug-in absorbed? **Per-expert plug-in is per-class within slice**, but the cross-slice routing itself is x-dependent (length is a function of x). The routing decision cannot be reduced to a per-class additive bias on a single architecture's logits.
- Known failure pattern? The 174-param length-strat overfit failure is structurally different (joint fitting with rare-bin overfitting); 87 params over disjoint slices avoids that pathology. Pearson constraint is sidestepped (disjoint evaluation).

**Where each proposal is most vulnerable in Debate#1.**

P1: HIGH-severity attack on the BBSE conditioning / Saerens convergence under miscalibrated test predictions. Counter requires Alexandari's bias-corrected calibration step, which adds complexity but is well-established.

P2: FATAL-severity attack is "Pearson collapses to > 0.8 because BiLSTM hidden state already encodes the retrieval signal". Stage A probe falsifies cheaply (4 hours). Low cost of being wrong.

P3: FATAL-severity attack is "BiLSTM L=1 macro-F1 is already near-optimal on the static-features path". Stage A probe falsifies cheaply (3 hours). Low cost of being wrong.

**Combined expected ship-gate probability (gate = +0.010 LB lift):**
- P1 alone: ~25% (pre-cal lift envelope marginal).
- P2 alone: ~30% (gated on Stage A Pearson check).
- P3 alone: ~35% (gated on Stage A length-decomposition diagnostic).
- P1 + P2 + P3 stacked (independent gains, all ship): ~55–65%, *if* all three Stage A gates pass. Realistically expect 1–2 of the three to die on Stage A, giving ~30–45% probability of a ≥+0.010 ship.

This is materially above the team's Argument-D Laplace-rule prior of 12.5%, primarily because P1 is structurally distinct from prior variants (the prior 6 KILLs were all training-objective perturbations or post-hoc calibration variants targeting OOF marginal; P1 targets a different distribution within the same function class).

---

## 4. If no proposal survives

If Stage A diagnostics kill P1, P2, and P3 (which is plausible — combined survival probability ≈ 50–70%), then I recommend **path B (lock current ship, write final report)** rather than continuing to deploy plans.

Argument: the team's Argument D Laplace prior P(8th plan succeeds | 0/6) = 12.5% is informative, and after P1/P2/P3 all dying it drops further by Bayesian update to ~5–8%. At 3 LB submissions per day with decreasing daily marginal value of testing, the expected return on a 9th–11th plan is dominated by the cost of the team's adversarial debate cycle and the opportunity cost of not writing the report. The "exhaust possibilities → cargo-culting" threshold is at the point where the post-update success probability falls below ~10% AND no genuinely orthogonal mechanism remains unexplored. After this consultation, **with P1/P2/P3 dispositioned**, that threshold is reached.

**Meta-strategic moves I do not recommend.**
- *LB submission economy exploit*: 3/day × daily reset over weeks gives ~50–100 submissions of slack, but the team's ship gate (+0.010) is too large to be hit by submission-side variance noise (LB is on 1,236 rallies; standard error of macro-F1 at 0.44 is ≈ 0.01, so submitting blends to find a lucky LB hit is a coin flip with negative expected value once you've already submitted your honest best).
- *Rule reinterpretation on de-identification*: the constraint is firm ("forbidden by competition rules"); attempting reverse-matching is an integrity risk far outweighing any plausible gain.
- *Test-prefix structure exploitation*: pseudo-labels are killed; TTA via TENT/MEMO is killed; AR-TTT is a research bet with uncertain payoff and 3–5 day cost. Below ship-gate ROI in expectation.

**The one move I would consider in extremis**: re-fit the existing bag2 with the *plug-in calibration as a learned training-time component* — make the per-class biases trainable parameters that backprop into the BiLSTM's last-layer representation. This is a small architectural tweak that makes the optimization end-to-end rather than post-hoc, and it is **not** Vapnik-equivalent to the post-hoc plug-in (the LSTM's representation moves under joint training). Cost ≈ 1 day. Expected lift is small (+0.001–0.004) but it is technically distinct from variant 1 (logit adjustment) which used τ·log(π) — a 1-param family. Only worth running if path B is on the table and the team has 1 day of slack.

---

## 5. Specific question answers

**A — Missing theoretical angle?** Yes, one: the plug-in is fit against OOF marginal not estimated test marginal — see P1. The framework saturation is real *within the function class targeted at OOF marginal*; re-pointing to estimated test marginal stays in the additive function class but at a different vector. Beyond that, batch-aware F1 decision rules (Dembczynski EFP / Lipton-Elkan thresholding) sit outside the plug-in class but offer < +0.005 lift in published settings.

**B — New information from existing 70k strokes?** Per-sample retrieval over BiLSTM hidden state is the cleanest answer (P2). It extracts long-tail context-similarity signal that the parametric softmax averages away. The Yamauchi 2025 long-tail caveat applies but the gain mechanism for 19-class small-vocab is "rare context retrieval" not "vocabulary expansion" — the team must run the Stage A Pearson probe to confirm, but the path is genuinely orthogonal to learned/aggregated/pseudo-label approaches.

**C — Macro-F1 exploit beyond plug-in?** Batch-aware F1 thresholding (Lipton-Elkan 2014, Dembczynski EFP 2011) is per-sample and not absorbed, but published gains over per-class plug-in are < 1% relative ≈ +0.001–0.004 absolute at this baseline. Worth a 2-day pilot only after P1/P2/P3 are dispositioned.

**D — Architectures beyond BiLSTM at 70k scale?** None of the candidates I researched (Mamba/S4 at L=5.65, GNN on rally-path, kernel SVM with small alphabet, snapshot ensembles) clear Pearson < 0.7 cleanly. **Retrieval-only (kNN) and rule-induction (RIPPER) are the two architectures whose inductive bias is genuinely categorical from BiLSTM — P2 covers retrieval; RIPPER duplicates LightGBM's axis-aligned bias too closely to add value.** State-space models at L≈5 collapse to function classes equivalent to BiLSTM.

**E — Rally-as-language exploit?** Compositional/factored action embeddings (factor 19 actions into type ⊕ hand ⊕ spin if structure is available) is the only small-vocab idea with non-trivial expected lift (+0.001–0.003). Hierarchical softmax, type-token modeling, character-level analogues offer near-zero gain at 29 classes.

**F — Heterogeneous ensemble escape?** P2 (kNN second head, stacked not interpolated) is the answer. NCL co-training of BiLSTMs forces low Pearson but constrains members to the same hypothesis class, so the gain is variance reduction, not bias reduction. Stacking with regularized meta-learner is correctly flagged as overfit risk by the team — at 14k OOF rallies / 5 fold the only safe meta is convex-combination Caruana forward selection.

**G — Competition-specific exploit?** P3 covers the L=1 lever (32% of test). Public ITTF rule statistics and serve-return tactical priors (Malagoli Lanzoni 2014, Munivrana 2023) confirm L=1 is a structurally distinct regime, but no published lift number from spec-priors exists in this corpus. Test-time adaptation (TENT/MEMO) is dead per Section 1; AR-TTT is a research bet with 3–5 day cost. The L=1 GBDT inside P3 is the cheapest exploit of the L=1 prior.

**H — Accept the ceiling?** After P1/P2/P3 are dispositioned via Stage A probes (~10 hours total), if ≤ 1 of the three survives to a real run, accept rank #3 and write the final report. If 2–3 survive, run them and re-evaluate at the +0.010 gate. The threshold for "exhaust possibilities → cargo culting" is reached when post-update P(next plan succeeds) falls below 10% AND no genuinely orthogonal mechanism remains untested. This consultation moves the team from "saturation argued" to "saturation falsifiable in 10 hours of diagnostic"; after those 10 hours, the threshold is reached one way or the other.

---

## Closing note on intellectual honesty

The team's framework is rigorous and most of their KILLs are correct. The two specific cracks I found (Argument B's invalid monotonicity step; the plug-in's mis-targeted optimization) are real but small — they enable diagnostics, not guaranteed lifts. The retrieval head (P2) and length-MoE (P3) are genuinely orthogonal mechanisms but their Stage A probes carry meaningful kill-probability. If the team executes the three Stage A diagnostics (mean-softmax-on-test for P1, Pearson check for P2, per-length F1 decomposition for P3) and all three return adverse signals, then the saturation diagnosis is empirically confirmed and path B is correct. The Laplace prior P=12.5% is informative but not conclusive; an 8th plan that targets a structurally different distribution within the same function class (P1) or a non-parametric per-sample mechanism (P2) is not strictly identically distributed to the prior 6 training-objective perturbations, and the prior should not be applied as a hard ceiling. P(any of P1+P2+P3 ships) is in the 30–45% range — meaningful upside relative to certain rank #3, but not a guaranteed escape.