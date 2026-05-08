# Red Flags

Specific warning signs in research reports. These aren't proof of low quality — every flag has exceptions — but each one shifts the burden of proof toward the report having problems. Multiple red flags in the same document is a strong signal.

This file is organized by category. Scan during Phase 4 and note any flags that fire.

---

## Quantitative red flags

**Suspiciously precise numbers without derivation.**
"$4.7 trillion market by 2030" with no source, or "73% of executives believe X" without describing the survey, often means the number was constructed to hit a desired headline. Even when such numbers happen to be approximately right, they were not produced by a process that distinguishes truth from convenience.

**Dramatic numbers compared to weak baselines.**
"3× improvement over baseline" — what's the baseline? An untrained model? An older method? A strawman? Strong improvements over weak baselines are nearly free.

**Cherry-picked time windows.**
The data starts in a year that happens to be a local minimum for the comparison group, or ends just before an event that would change the conclusion. "Performance from 2020 to 2024" when 2019 data is available and would change the answer — flag.

**Y-axes truncated to exaggerate effects.**
A bar chart showing 51% vs 49% as towering vs tiny bars. Inflates visual significance; common in marketing-style charts.

**Statistical significance without effect size.**
"Statistically significant at p < 0.05" tells you nothing about whether the effect is large enough to matter. With large enough samples, trivial effects are significant.

**P-values clustered just below significance thresholds.**
Across many results in a paper, if many p-values are between 0.04 and 0.05 and few are between 0.06 and 0.10, the data was likely massaged to clear the threshold (p-hacking).

**Reported numbers don't reconcile.**
Sub-totals don't add to totals; percentages don't sum to 100%; figures referenced in text don't match the figure values. Suggests the document wasn't carefully checked, which raises the prior on other errors.

---

## Methodological red flags

**Proprietary methodology / proprietary data with no validation.**
"Our proprietary scoring algorithm shows..." with no published methodology, no comparison to public alternatives, no track record. The proprietary framing converts the methodology from a checkable claim to a marketing asset.

**No out-of-sample test for an empirical / quantitative claim.**
Especially: a backtest with no out-of-sample period, a model with no held-out data, a survey-based finding with no follow-up wave to test stability.

**No limitations section, or a limitations section that lists only minor issues.**
A confident, polished report with a limitations section that says "more research is needed" is less honest than a report with explicit, substantive caveats. The strongest reports are often the most caveated.

**Missing comparison groups.**
"Companies that adopted X grew 30% over 3 years" — compared to what? Companies that didn't? Companies that adopted Y? The comparison group is where causal claims are made or broken.

**Data reconstruction of unfalsifiable scenarios.**
"If our policy had been implemented in 2010, GDP today would be 5% higher." This kind of counterfactual relies on a model whose structure cannot be tested, since the alternate world doesn't exist.

**Surveys of self-reported performance.**
"81% of executives say their company is data-driven." Self-reports are nearly worthless for measuring objective performance; treat as measures of belief or aspiration only.

---

## Logical red flags

**Conclusion broader than the evidence.**
The methodology investigated US large-cap stocks 2018–2024; the conclusion is about "equities." The methodology surveyed 30 senior managers in finance; the conclusion is about "leaders." Watch for the silent expansion.

**"This time is different" without justification.**
When a report acknowledges a historical pattern but argues this case is exempt, demand specifics. What's the structural change? Is the change documented elsewhere? Has it been load-bearing in past exemption arguments that turned out wrong?

**Asymmetric burden of proof.**
The author's preferred conclusion gets confirmed by suggestive evidence; the opposing conclusion would require ironclad proof. Flip the conclusion — would the same evidence be considered as supportive in the opposite direction?

**Composition / division errors.**
Properties of components assumed to apply to wholes ("each module is fast, so the system is fast"), or properties of wholes assumed to apply to components ("the company is profitable, so this product line is profitable").

**Equivocation on key terms.**
A term defined narrowly in the methodology section is used broadly in the conclusion. Common offenders: "risk", "efficiency", "engagement", "performance", "AI", "decentralized."

**Argument from authority used as load-bearing evidence.**
"As Nobel laureate X has said..." in a context where X is not a domain expert on the specific question, or where X's statement is being used to short-circuit an empirical argument.

---

## Framing red flags

**Loaded question.**
"Why is X the best approach?" assumes X is best. "How should we address the X crisis?" assumes there's a crisis. The question shapes the answer space; loaded questions narrow it artificially.

**False dichotomy.**
The report frames a choice as A vs B when reasonable third options exist. Especially common in policy and business strategy: "regulate or innovate," "build or buy," when "regulate carefully and selectively" or "buy then build" are usually viable.

**Stage-of-maturity frameworks (Crawl/Walk/Run, Bronze/Silver/Gold, Level 1–5).**
Often vibes-based and unfalsifiable. Useful as a sales tool, weak as analysis. Flag if the framework's stage definitions aren't operationalized.

**Quadrant charts with subjective axes.**
2×2 plots positioning vendors / strategies / countries on axes like "vision" vs "execution." The placement is rarely justified by quantitative criteria; it's usually editorial.

**Anecdotes as evidence.**
A vivid case study is presented as if it represents a population. "Company X tried Y and revenue grew 40%" tells you nothing about what happens to the median company that tries Y.

---

## Conflict-of-interest red flags

**Funding source aligns suspiciously with conclusion.**
A pharmaceutical-funded study finds the funder's drug works. An industry-funded report finds the industry needs less regulation. A consulting firm's analysis recommends services the firm sells. Alignment isn't proof of bias, but it's a flag.

**Hidden funding / about page.**
If the funder isn't disclosed prominently, find them. The funder's website "About" page often makes the alignment obvious.

**Authors with prior public commitment to the conclusion.**
Authors who have publicly argued for the conclusion before are not just analyzing it; they're defending a prior position. The analysis can still be sound, but it's not a fresh look.

**Selectively cited literature.**
The report cites supporting work but not contradicting work, or cites contradicting work only in a footnote that dismisses it briefly. Pull the missing citations and check if they actually contradict.

**"In partnership with" / "sponsored by" labels.**
Sponsored research is often legitimate but should be read as an extended marketing artifact unless the methodology is clearly independent.

---

## Presentation red flags

**Excessive jargon protecting weak content.**
Dense terminology that, when paraphrased, reveals the underlying claim is either trivial or unsupported. "Synergistic alignment of cross-functional capability stacks" might just mean "people working together." Translate, then evaluate.

**Slide-deck masquerading as analysis.**
A deck with strong visual production values, sparse text, and big claims often hides that the underlying analysis is thin. Visual polish is uncorrelated with analytical rigor.

**No methodology description at all.**
A report with conclusions and recommendations but no description of how the conclusions were reached. The reader is being asked to trust the brand, not the analysis.

**Tone disproportionate to evidence.**
Apocalyptic language about modest findings, or breakthrough language about incremental improvements, signals the rhetoric is doing work the evidence can't.

**No citations / unverifiable citations.**
Claims that aren't sourced, or sources that are paywalled / private / "data on file" with no path to verification.

---

## Domain-specific red flags

### Quant / backtest specific
- Sharpe > 3 with no live track record (almost always overfitting)
- "We optimized over a small parameter grid" with the grid undisclosed
- Strategy designed and tested on the same period, presented as if validated
- Benchmark is a deliberately weak choice (e.g., simple buy-and-hold against a strategy that uses leverage)
- Costs assumed at zero or "standard" without specifying the venue/instrument

### Crypto whitepapers
- Token burn schedule designed to look deflationary while protocol economics require inflationary subsidies
- Team page with stock-photo headshots
- Citations to unrelated cryptography papers used for legitimacy
- "Web3 / metaverse / AI" attached to an idea that doesn't need any of those

### Sell-side research
- Price targets clustered with the broker pack (no independent view)
- Recommendation contradicts the analyst's stated risk concerns
- "Buy on weakness" / "wait for entry" — emergency-exit framings that protect the recommendation if it fails

### Academic finance
- Sample period that omits the 2008 crisis (in a paper claiming a robust effect)
- Strategy that requires shorting hard-to-borrow names without modeling borrow costs
- Claim of an anomaly that decays sharply post-publication date — possibly the anomaly was always artifact, or possibly arbitraged away (both are informative)

### ML / AI papers
- Benchmark cherry-picking (paper X is good on the benchmarks the authors highlight, mediocre on others)
- Metrics that overlap with training data (data contamination)
- "State of the art" claim with no comparison to recent strong baselines
- Compute / inference costs not reported
- Failures highlighted only briefly, often in supplementary material

---

## How to use this list

**Pattern over instances.** A single red flag isn't damning. Three or four red flags clustering in the same report is a strong signal of low quality.

**Severity matters.** "Suspiciously precise numbers" is a mild flag. "No methodology section in a quantitative report" is a severe flag. Note severity, not just count.

**Don't treat absence of flags as endorsement.** A report can avoid every flag on this list and still be wrong. The flags are evidence of problems, not the only path to problems.

**Surface flags transparently.** When red flags fire, name them in the evaluation. The user benefits from the explicit label, not just a vague sense that "something seemed off."

## Translating red-flag patterns into a calibrated verdict

Red flags are inputs to the Phase 5 "Recommended posture" call. The mapping is not a formula — judgment still matters — but the rough translation below keeps Phase 5 honest. Apply this *after* you've also weighted severity and the report's strengths; the patterns below assume flags are real failures, not misreadings.

**0–2 mild flags fire.** Treat the report on its merits. Note flags in evaluation but they don't dominate the verdict. Posture: *take the claim seriously, with the noted caveats*.

**3–5 flags, mixed severity.** The report has real problems but isn't structurally broken. The headline claim may still be roughly right; the precision and confidence with which it's stated almost certainly isn't. Posture: *treat as one data point, discount confidence substantially, and look for independent corroboration before acting*.

**6+ flags, or any 2 severe flags (no methodology in a quantitative claim, undisclosed proprietary scoring system, conclusion entirely contained in the funder's commercial interest, no out-of-sample on a forward-looking quantitative claim).** The report is not informative *about its claim*; it may still be informative *about the publisher* (what they want the audience to believe, what's currently being marketed). Posture: *do not use as evidence for the claim itself; if a decision rests on this report alone, the decision is unsupported*.

**Most or all of the list fires.** This is genuinely rare and meaningful when it happens — it indicates the document was not produced by a process that distinguishes truth from convenience at any stage. Posture: *the document's information value about its stated claim is approximately zero; treat as content, not analysis. If a user is being asked to act on the basis of this document, the appropriate response is to ask for the underlying analysis, not to debate the document's content*.

**Caveat.** A document with many red flags can still describe a true conclusion — bad reasoning sometimes lands on right answers. The verdict above is about *the report's evidentiary value*, not about the underlying claim. If the underlying claim is independently important, it deserves its own investigation; the red-flag-laden report just isn't the way to investigate it.
