---
name: research-report-analysis
description: Use when the user provides a research report, paper, whitepaper, broker note, backtest report, industry analysis, policy paper, or similar document and asks Claude to read, analyze, evaluate, summarize, or critique it (中文觸發詞 — 研究報告、論文、白皮書、研報、報告、分析這篇、幫我讀、值不值得信、這份報告在說什麼). Trigger whenever the user uploads a PDF/document, pastes a long research-style text, or references a report by URL and wants Claude to do anything more substantive than acknowledge it. The skill produces a structured analysis covering what the report claims, how strong the evidence actually is, where it's likely to be wrong, and what it means for the user. Do NOT use for short news articles, factual lookups, technical documentation (API docs, tutorials), code reviews, or content where the user explicitly just wants a translation/summary with no evaluation.
---

# Research Report Analysis

A protocol for reading research reports the way a careful analyst reads them — extracting structure, evaluating quality, and producing a synthesis the user can actually act on.

## Why this exists

Research reports are not neutral information. They are arguments, written by someone, for someone, in service of some purpose — academic publication, business advocacy, market influence, regulatory compliance, internal decision support. Treating them as if they were Wikipedia entries (just extract and restate) loses the most important signal: how much should the reader actually update on what the report says?

The default LLM mode for handling reports is "summarize what's in it." That mode is fine when the user only wants the gist. But when a user gives you a research report, they usually have a real question behind it — should I trust this? Should I act on this? Is this consistent with what I already know? — and that question requires more than summary. It requires reading the report as something with a structure, an author, an audience, and a set of evidentiary commitments that may or may not hold up.

This skill produces that deeper read.

## When to engage

Engage when the user gives you a research-style document and wants more than a one-line gist:
- Academic paper (finance, ML, biomedicine, social science, etc.)
- Industry research (Gartner, McKinsey, sell-side broker notes, consulting reports)
- Company-issued material (earnings calls, investor decks, 10-Ks, white papers)
- Crypto / protocol whitepapers
- Backtest reports or quant strategy memos
- Policy papers (think tank, government, NGO)
- Conference proceedings or technical notes presenting empirical claims

Do not engage for: short news articles (under ~1000 words and not making a structured argument), pure technical documentation (API docs, README, tutorials), product marketing copy with no analytical content, or cases where the user has explicitly said "just give me the TL;DR, no evaluation."

If the document type is unclear, ask the user once: "What's your goal here — quick summary, full evaluation, or something specific?" Then proceed.

## The five-phase protocol

### Phase 1 — Triage

Before reading deeply, classify the report. This determines reading strategy.

Five questions, answered briefly:

1. **What type is this?** Academic paper, industry research, company material, whitepaper, backtest, policy paper, broker note, etc. See `references/report-types.md` for the typology and per-type reading strategy.
2. **Who wrote it?** Author, affiliation, funding source. The same words mean different things from different mouths.
3. **Who is it written for?** Peers? Customers? Regulators? Retail investors? Internal management? The audience determines what the author can and cannot say.
4. **When was it written?** Information half-life varies wildly by domain. A 2019 forex paper and a 2019 LLM paper have different shelf lives.
5. **What's the reference class?** This isn't one report; it's an instance of a category. What's the typical track record of reports in this category?
6. **Does the form match the substance?** A document can wear the clothes of one category while serving the purpose of another — a marketing whitepaper styled as a research paper, a content-marketing article structured as a backtest memo, a sponsored "study" formatted like academic research. When form and substance diverge, classify by **substance**, not form. The tells: (a) author/methodology opacity disproportionate to the document's pretense at rigor, (b) rhetoric calibrated for retention/sharing rather than peer scrutiny, (c) recommended actions that conveniently route value to the publisher or its allies, (d) production polish higher than analytical depth. When this divergence is present, name it explicitly in the Triage output — it changes how everything downstream should be read.

Triage takes 1–2 minutes and is the highest-leverage step. Skipping it means treating a sell-side note like a peer-reviewed paper, or a marketing whitepaper like a research paper. They are not the same.

### Phase 2 — Map

Build a structural map before deep reading. Research reports are not novels — do not read them linearly.

Recommended order:
1. **Abstract / executive summary** — what the report claims to be about.
2. **Conclusions / recommendations** — what the report wants the reader to believe or do.
3. **Section headings + figures/tables** — the report's skeleton. Skim for the shape.
4. **Methodology section** — for empirical reports, this is where the load-bearing decisions live.
5. **Limitations section** (if it exists) — what the author already knows is shaky.

After the map pass, you should be able to state: the report's central thesis, its three or four main supporting claims, the methodology in one sentence, and where the evidence lives. If you can't, the report is either poorly structured or you missed something — go back.

### Phase 3 — Extract

Now do the deep extraction using the right template from `references/extraction-templates.md`.

Core elements to extract from any report:

- **Central thesis.** The single load-bearing claim. State it in one sentence.
- **Key claims.** The 3–7 specific assertions that support the thesis. Number them.
- **Methodology.** How the research was done. For empirical work: data, time period, sample, model, validation approach. For theoretical work: framework, assumptions, derivation logic.
- **Evidence.** What's offered as support — datasets, citations, case studies, experimental results, simulations.
- **Conclusions / recommendations.** What the report tells the reader to do or believe.
- **Stated limitations.** What the author admits.
- **Unstated assumptions.** Things treated as given that aren't actually given. These are usually the most informative — what would have to be true for the argument to hold? Often it's something the author considered too obvious to mention, but isn't.

Be precise. Quotes are useful here for thesis and key claims (kept under 15 words each, with attribution), but everything else should be in your own words.

### Phase 4 — Evaluate

The hardest phase, and the one most often skipped. The question is: how much should a reader actually update on this?

Run the report against the quality checklist in `references/quality-checklist.md`. Then check `references/red-flags.md` for warning signs of bad-faith or low-quality work.

Major evaluation axes:

- **Methodological soundness.** Is the method appropriate for the claim? For quantitative claims especially: sample size, statistical power, look-ahead bias, survivorship bias, multiple comparisons.
- **Data quality.** Sources reliable? Time period sufficient? Categories appropriate? Missing data accounted for?
- **Logical structure.** Do the conclusions actually follow from the evidence? Is there a quiet leap from "we showed X in our sample" to "X is true in general"?
- **Scope inflation.** Does the report's headline overstate what its methods support?
- **Conflicts of interest.** Who paid for this? Who benefits if it's believed? Has the author committed publicly to this position before?
- **Engagement with opposition.** Does the report engage with the strongest counter-arguments, or only with weak versions?
- **Reproducibility.** Is there enough detail that someone else could replicate the work? Is data/code released? Have others replicated it?

Output for this phase is a calibrated assessment, not a verdict of "good" or "bad". Reports are mixtures — strong methodology with overreaching conclusions, weak data with sound logic, etc. Name the components.

### Phase 5 — Synthesize

Produce something the user can use.

Synthesis is not summary. Summary repeats the report. Synthesis fuses what the report says with how much it should be trusted, what's missing, and what it implies for the user's situation.

The synthesis includes:

1. **What the report claims, condensed.** 3–5 sentences. The reader's working understanding.
2. **Evidence quality assessment.** Where the report is strong, where it's weak, where you don't have enough information to tell.
3. **What's missing.** Questions the report should have addressed but didn't, alternative explanations not considered, comparison points not made.
4. **Reference-class context.** How this report compares to others making similar claims. Does it cite, contradict, or replicate prior work? Is it an outlier?
5. **Implications for the user** *(if user context warrants this)*. What this means for their situation, decision, or research. Be specific. "Interesting findings" is not implications.
6. **Recommended posture.** Take the claim at face value? Treat as one data point among many? Treat as marketing? Discard?

The output structure is in the format section below.

## Handoff to other skills

This skill produces an analysis. It does not, by default, attack the report's conclusions adversarially — that's a separate operation. If the report makes a claim that the user is considering acting on (e.g., "this paper says strategy X works; should I deploy?"), surface this in the synthesis and offer to run the conclusion through the `adversarial-debate` skill for stress testing. Do not auto-chain — let the user decide.

Conversely, if the report contains a quantitative result the user wants to verify computationally (e.g., re-run a backtest, replicate a chart), hand off to the relevant analysis skill rather than doing it inline within this protocol.

## Tone and stance

The analyst posture is **calibrated, specific, and not deferential to authority.** Specifically:

- **Authority isn't evidence.** A paper from a famous lab, a report from a top-tier consultancy, a whitepaper from a heavily-funded protocol — none of these get a free pass. The methodology is the methodology, regardless of letterhead.
- **Be specific about strength.** "The methodology has issues" is useless. "The sample period is 2018–2023, which contains exactly one liquidity regime; the headline result is conditional on that regime persisting" is useful.
- **Don't manufacture concerns.** If the report is solid, say it's solid. The skill is calibration, not contrarianism.
- **Distinguish report-quality from claim-truth.** A bad report can still have a true conclusion (someone got the right answer for the wrong reasons). A good report can still have a false conclusion (the field has moved on, or the result didn't replicate). Evaluate the report; flag if your view of the underlying claim differs from what the report's quality alone would imply.
- **Translate jargon when it carries content.** Don't translate when it doesn't. The user is usually domain-aware enough to know "Sharpe ratio"; they may not be aware that the report computed it on overlapping windows.

## Output format

Default structure:

```
## Triage
- Type: [report category]
- Author / affiliation: [...]
- Audience: [...]
- Date / vintage: [...]
- Reference class: [...]
- Form vs substance: [match / mismatch — and if mismatch, which dominates the read]

## What it claims
**Thesis:** [one sentence]
**Key claims:**
1. [...]
2. [...]
3. [...]
**Methodology:** [one paragraph]
**Conclusions / recommendations:** [...]

## Evaluation
**Strengths:** [specific, not generic]
**Weaknesses:** [specific, with mechanism]
**Unstated assumptions:** [the load-bearing ones the report didn't address]
**Red flags:** [if any — see red-flags.md]
**What's missing:** [what should be in the report but isn't]

## Reference-class context
[How this fits with other work in the same category. Does it agree? Contradict? Extend? Is it an outlier?]

## Synthesis
[3–6 sentences fusing claim + evidence quality + implications]

## Recommended posture
[Take seriously / treat as one data point / treat as marketing / discard, with reasoning]

## Open questions
[What you couldn't determine from the report alone — what the user might want to investigate further]
```

For shorter or lower-stakes reports, collapse sections and keep total output under a page. For long, high-stakes reports (deploying capital, making a hire, writing your own paper that builds on this one), do the full structure.

## Worked example (compressed)

> **User:** *(uploads a 30-page broker research note titled "Why Mid-Cap Tech Will Lead in 2026")*

**Triage:**
- Type: sell-side equity research note.
- Author: a major bank's equity strategy team.
- Audience: institutional clients (implicit: clients who pay for this research and execute trades through the bank).
- Date: Q2 2026.
- Reference class: bank macro/sector calls. Aggregate track record on 12-month sector calls is mediocre — roughly coin-flip across the industry, with calls often correlated with the bank's existing book.

**What it claims:**
- Thesis: mid-cap tech will outperform large-cap tech and the broader market over 12 months.
- Key claims: (1) AI capex cycle is broadening from hyperscalers to mid-caps; (2) mid-cap tech valuations are at a historic discount to large-cap; (3) short-rate easing will benefit duration-sensitive growth names.
- Methodology: scenario analysis, multiples comparison, and a proprietary "AI exposure score." No backtest of the score's predictive power is shown.

**Evaluation:**
- Strengths: claim 2 (valuation discount) is verifiable from public data and approximately correct.
- Weaknesses: the proprietary AI score is undefined; the scenario analysis assumes rates ease, which is a separate forecast not interrogated; no historical track record of the strategy team's prior 12-month calls is shown.
- Unstated assumptions: that capex broadening implies revenue broadening on the same timeline (typically lagged 4–8 quarters); that valuation discount converges (it has often persisted for years).
- Red flags: undefined proprietary score, no track record, recommendation aligns with the bank's prime brokerage book.

**Reference-class context:** sell-side sector calls of this shape have a documented hit rate of roughly 50% over 12 months, with selection bias in which calls are remembered. This call doesn't break that mold.

**Synthesis:** The valuation discount is real. The mechanism by which the discount closes (AI capex broadening + rate easing) is plausible but rests on two separate forecasts the report doesn't justify. The headline call is a directional bet dressed up as analysis; treat it as one practitioner's view, not a base rate.

**Recommended posture:** Treat as one data point. Take the valuation observation seriously; discount the directional call to roughly its 50/50 base rate.

**Open questions:** What's this team's actual track record on prior calls? What's the bank's positioning in this trade? Has anyone replicated the AI exposure score externally?

That's the skill. Structured reading, calibrated evaluation, actionable synthesis.
