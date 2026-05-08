# Quality Checklist

A structured checklist for evaluating report quality. Run through these axes during Phase 4. Not every axis applies to every report — use the ones relevant to the report type identified in Phase 1.

The output of this phase is *not* a grade. It's a calibrated assessment of where the report is strong, where it's weak, and how much weight the user should put on its conclusions.

---

## Axis 1: Methodology

Is the method appropriate for the claim being made?

**Specific checks:**
- Does the method actually answer the question the report claims to answer? (E.g., a survey of intent does not measure behavior.)
- Is the analytical technique standard for the field, novel-and-justified, or novel-and-unexplained?
- Are assumptions of the method satisfied by the data? (e.g., independence, normality, stationarity assumptions for the relevant statistical tests)
- For empirical work: was the analysis pre-specified, or selected after seeing the data?
- For models: are model choices motivated, or arbitrary?
- For simulations: are parameter ranges realistic? Have authors swept over reasonable alternatives?

**Strong methodology looks like:** the method is well-suited to the question, assumptions are checked, alternative methods would have produced similar results, and the authors say so explicitly.

**Weak methodology looks like:** the method is one of many reasonable choices and others would have produced different answers, but the report doesn't say so.

---

## Axis 2: Data quality

Is the data appropriate, sufficient, and clean?

**Specific checks:**
- Source: is the data from a primary source or secondhand? Is the secondhand source itself reliable?
- Completeness: are there missing observations, and how were they handled?
- Sampling: how was the sample selected? Is selection correlated with the outcome variable?
- Time period: does the period cover enough variation to support the claim? (Single-regime data is a major issue for any claim that's supposed to generalize.)
- Categories: are the categorical variables defined the same way throughout? Have definitions changed over time?
- Granularity: is the data at the right unit of analysis? (Aggregating individuals to averages destroys variance information.)

**For backtests specifically:**
- Point-in-time correctness: would this data have been available, in this form, at the historical decision time?
- Survivorship: is the universe biased toward survivors?
- Restated data: have figures been revised since original publication? Are you using the original or revised values?

**Strong data quality looks like:** primary source, complete sample, point-in-time correct, with a transparent description of any cleaning steps.

---

## Axis 3: Sample size and statistical power

Is the evidence base large enough to support the claim?

**Specific checks:**
- For statistical claims: was the study powered to detect the effect it claims? (Underpowered + significant = the effect is probably either fake or much larger than reality. This is a known statistical phenomenon called "winner's curse.")
- For empirical generalizations: does the sample contain enough variation across the dimensions the claim is generalized over? (10 case studies of US tech companies don't support a global claim about all industries.)
- For backtests: how many independent observations does the data really contain? Daily data for 5 years is ~1250 observations, but for a strategy holding positions 10 days on average, the effective sample is closer to 125. Sharpe ratios on small effective samples have wide confidence intervals.
- For surveys: is the response rate disclosed? Is non-response correlated with the question being asked?

**Useful heuristic:** the number you should attend to is *effective* sample size, not raw N. A million observations from one source can be less informative than 100 observations from independent sources.

---

## Axis 4: Logical structure

Do the conclusions follow from the evidence?

**Specific checks:**
- Are the premises clearly stated?
- Does the conclusion follow even if the premises are true? (Validity)
- Are there hidden steps where the argument quietly substitutes one claim for a stronger one?
- Is correlation being treated as causation? If yes, is there a causal mechanism proposed and tested?
- Is the scope of the conclusion equal to, or larger than, the scope of the evidence? (Reports often establish "X is true in our sample" and conclude "X is true in general.")
- Is "absence of evidence" being treated as "evidence of absence"? (Sometimes correct, sometimes a fallacy — depends on whether evidence would have been visible if it existed.)

**Strong logical structure looks like:** premises stated, derivation steps explicit, conclusion calibrated to what the evidence actually supports, scope of generalization explicitly bounded.

---

## Axis 5: Scope inflation

Does the headline overstate what the methodology can support?

**Specific checks:**
- Compare the abstract / executive summary / press release to the actual results. Do the strong words ("proves", "demonstrates", "transforms") match the underlying evidence's strength?
- Compare the methodology's scope (sample, time, geography, demographic) to the conclusion's scope. Mismatches are scope inflation.
- For predictive claims: was the claim made *before* seeing the validation data, or after?
- Does the report carefully distinguish "we found" (in our sample) from "is true" (in general)? Or does it slide between them?

**Strong calibration looks like:** the abstract's strength matches the body's evidence; generalization is explicitly bounded; the report concedes where its findings might not apply.

**Weak calibration looks like:** the headline claim is much stronger than the body would support; generalization is implicit and broad; counter-cases aren't discussed.

---

## Axis 6: Conflicts of interest and incentives

Who benefits if this is believed?

**Specific checks:**
- Funding source: who paid for this? Is there any ambiguity in their interests?
- Author affiliation: where do the authors work, and does their employer have a stake in the conclusion?
- Prior public commitments: have the authors publicly endorsed this conclusion before? (Reversing a public position is costly; this can bias analysis toward defending the prior position.)
- Career incentives: is the conclusion of the type that's career-rewarded in the author's field? (Surprising-but-true findings are more publishable than null results, which biases the published literature.)
- Direct commercial alignment: does the report's recommended action funnel business to the publisher?

**Important nuance:** conflicts of interest don't automatically make a report wrong. Conflicted authors can produce excellent work. But conflicts shift the burden of proof — the reader should require *more* evidence than usual, not dismiss the work outright.

---

## Axis 7: Engagement with opposition

Does the report take counter-arguments seriously?

**Specific checks:**
- Are the strongest opposing arguments named and engaged?
- When opposition is mentioned, is it the version a smart, well-resourced opponent would actually defend? Or a strawman?
- Are alternative explanations for the data considered and ruled out, or assumed away?
- If the field is contested, does the report acknowledge the contestation?
- Are caveats and limitations integrated into the conclusion, or quarantined in a section the reader can skip?

**Strong engagement looks like:** the report identifies the strongest opposing position, makes the case for it as forcefully as the case against it, and explains specifically why the evidence still favors the report's conclusion.

**Weak engagement looks like:** opposition is dismissed in a sentence; alternative explanations aren't named; caveats are buried; the strongest counter-argument the report engages with is much weaker than the actual strongest one.

---

## Axis 8: Reproducibility

Could a reader independently verify the work?

**Specific checks:**
- Is the data available?
- Is the analysis code available?
- For backtests: are entry/exit rules specified to a level where someone else could implement them?
- For surveys: is the questionnaire available?
- For models: are parameter values, training procedures, and seeds specified?
- Has the work been independently replicated by someone unaffiliated with the authors?

**Why this matters:** unreproducible work can still be true, but can't be checked. The community can't catch errors. For high-stakes claims, demand reproducibility; for casual reads, note its absence and adjust confidence accordingly.

---

## Axis 9: Track record

Has reasoning of this shape worked before?

**Specific checks:**
- What's the reference class of reports like this one? What's their hit rate?
- For predictions: what's the author's / publisher's prior calibration?
- For methodologies: have similar methods produced replicable results in this domain?
- For institutions: does this institution have a culture of accuracy or of advocacy?

**Useful frame:** even a brilliantly-argued report has only the prior probability of its reference class. A 20-year-old prediction methodology with a 20% historical hit rate doesn't get to claim 80% confidence on this prediction just because the case "looks strong."

---

## Synthesizing the axes

Don't aggregate to a single number. Different axes are informative in different ways:

- A report can be *methodologically strong* and *commercially conflicted* — the methodology compensates partially for the conflict, but conflict still matters.
- A report can have *good data* and *weak logic* — the underlying numbers are real, but the conclusions don't follow from them.
- A report can have *strong reproducibility* and *small effective sample* — you can verify what they did, but the result is consistent with noise.

In Phase 5 synthesis, name the components separately. "The data is high-quality; the methodology is appropriate; the conclusion overstates by extending to a population not in the sample" is more useful than "moderately credible."

---

## How to handle high-quality reports

If a report passes most axes cleanly, *say so* — don't manufacture concerns to seem balanced. Confidence in a high-quality report is a real outcome of the evaluation, not a failure to find issues.

The skill isn't "always find problems." The skill is "calibrate accurately to what the evidence supports." Sometimes that's high confidence. Sometimes that's low confidence. The work is to figure out which.
