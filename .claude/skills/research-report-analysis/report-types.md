# Report Types

Different kinds of reports demand different reading strategies. The same skill that works on an academic paper will misread a sell-side note, and vice versa. This file gives you the per-type reading strategy.

For each type: **what it is**, **what to read first**, and **what to be especially skeptical of**.

---

## 1. Academic paper (peer-reviewed)

**What it is:** A formal research output, typically following IMRaD structure (Introduction, Methods, Results, Discussion). Has been through peer review, though the bar varies enormously across journals and fields.

**What to read first:**
1. Abstract — claims and main result.
2. Conclusion / Discussion — implications and stated limitations.
3. Methods — most of the load-bearing decisions live here.
4. Results figures — the actual evidence, often clearer than the prose.
5. The specific things in the introduction the paper says it's contributing — these are the load-bearing claims.

**Be especially skeptical of:**
- *p-values without effect sizes.* "Statistically significant" tells you almost nothing about whether the effect is large enough to matter.
- *Underpowered studies.* Small N + a detected effect = the effect is probably either fake or much larger than reality.
- *Pre-registration status.* If the hypothesis wasn't pre-registered, the analysis was selected from a garden of forking paths.
- *Replication status.* Has this been replicated? In a different lab? Many headline findings haven't.
- *Citation circles.* Some subfields are dominated by groups that cite each other; "widely cited" can mean "cited by a small group repeatedly."
- *Journal tier ≠ truth.* Top-tier journals filter harder, but also chase novelty, which selects for surprising-and-likely-wrong over modest-and-likely-true.

**Useful base rate:** in social science and biomedicine, the replication rate for high-profile experimental findings is roughly 40–60%. In quant finance, headline backtest results commonly fail to replicate out-of-sample. In ML, papers often "replicate the headline number" but not "the claimed mechanism."

---

## 2. Industry research / consulting reports

**What it is:** Reports from consulting firms (McKinsey, BCG, Bain), research firms (Gartner, Forrester, IDC), or industry associations. Published either as marketing for paid services or as paid client work.

**What to read first:**
1. Executive summary.
2. The one-line "implication" or "recommendation" tucked at the end of each section — this is what the firm wants you to internalize.
3. The footnotes / methodology box — usually thin, and the thinness is informative.
4. Data sources — often surveys with self-reported numbers from the respondents being studied.

**Be especially skeptical of:**
- *"Proprietary methodology."* This is a tell. Real methodologies are published in enough detail to be replicated.
- *Surveys of executives self-reporting their own performance.* These are nearly worthless for measuring outcomes; treat as a measure of executive sentiment, not reality.
- *Numbers with high precision and no derivation.* "$4.7T market by 2030" with no source is invented. Even approximately-right numbers in this format were made up; they just got lucky.
- *Categories that exactly fit the firm's service offerings.* If the recommended actions all happen to require expensive engagements, the categories were drawn to support the engagements.
- *"Leaders pull away from laggards."* Self-fulfilling tautology — leaders are defined by performance, so of course they're pulling away.
- *Stage-of-maturity frameworks (Crawl/Walk/Run, etc.).* Often vibes-based and unfalsifiable; useful as a sales tool, not as analysis.

**Useful base rate:** large consulting / analyst reports tend to be directionally right and quantitatively unreliable. Use them for thesis generation, not for inputs to a model.

---

## 3. Sell-side broker research / equity research notes

**What it is:** Investment research published by brokerage firms for their institutional and (sometimes) retail clients. Common formats: company initiation, quarterly update, sector outlook, thematic call.

**What to read first:**
1. Rating + price target — the actionable claim.
2. Key thesis points (usually bulleted at the top).
3. Estimate revisions (vs prior, vs consensus) — the actual update.
4. Risks section — underrated; often the most informative part.
5. The disclosures page — long but informative on conflicts.

**Be especially skeptical of:**
- *Anchored price targets.* PTs cluster near current price plus modest deltas; analysts rarely make brave calls.
- *Track record opacity.* Most firms don't publish a clean track record; ratings have a documented bias toward Buy/Overweight.
- *Maintenance-of-coverage incentive.* Analysts can't downgrade aggressively without losing access to management; ratings may be stickier than the underlying view.
- *Banking conflicts.* Has the firm done equity/debt deals for the company? This is disclosed but easy to miss.
- *Estimate clustering.* "Above consensus" is meaningful only if the analyst was right last time. Most estimates herd.
- *"This time the stock will work" reasoning that uses the same setup the analyst pitched 6 months ago.*

**Useful base rate:** sell-side equity ratings have roughly market-rate predictive power on 12-month forward returns — i.e., not much. The information value is usually in the *changes* (upgrade/downgrade, estimate revision) more than the absolute level.

---

## 4. Company-issued material (earnings, decks, 10-Ks, white papers)

**What it is:** Material produced by the company itself. Earnings transcripts, investor presentations, regulatory filings (10-K, 10-Q, 8-K, S-1), product/technology white papers.

**What to read first:**
1. For earnings: the prepared remarks first (rehearsed), then Q&A (less rehearsed, more informative).
2. For 10-Ks: Risk Factors, MD&A, then financial statements.
3. For investor decks: the appendix — this is where awkward numbers live.
4. For white papers: the abstract, then the experiments / claims tables, then the limitations.

**Be especially skeptical of:**
- *Adjusted / non-GAAP metrics.* What got adjusted out, and is it actually non-recurring?
- *Year-over-year comparisons that skip awkward years.* Comparing 2024 to 2019 may be hiding 2020–2023.
- *KPIs that change definition.* "Active users" can be redefined over time; check the footnotes.
- *"In-line with our expectations"* or *"as expected"* about something that wasn't pre-disclosed. If you didn't know they expected it, the framing is post-hoc.
- *Forward-looking statements.* Legally protected from accountability; treat as aspiration, not commitment.
- *Engineering/research papers from product teams.* These have a different optimization than academic papers — they exist to support the product narrative. The methodology may be solid; the framing usually isn't disinterested.

**Useful base rate:** company-issued material is generally factually accurate (regulatory teeth) but selectively framed. The information is reliable; the emphasis is not.

---

## 5. Crypto / protocol whitepapers

**What it is:** Documents accompanying a token launch, protocol design, or crypto research result. Range from genuine technical contributions to marketing dressed in LaTeX.

**What to read first:**
1. The token economics / incentive design — most whitepapers fail here, and this is where the value (or its absence) lives.
2. The actual technical novelty — what's new vs what's recombination of existing primitives?
3. The team and funding — who built this and who paid?
4. The "use case" claims — and whether they require this protocol or could be done with a database.

**Be especially skeptical of:**
- *Solutions in search of problems.* "Decentralized X using a token" where X works fine without decentralization or a token.
- *Hand-waved security models.* "Trustless" used loosely; "Byzantine fault tolerant" without specifying threat model.
- *Token mechanics that imply unlimited demand.* Burn-and-mint where demand assumes adoption that hasn't happened.
- *Unverifiable team / fake credentials.* Common.
- *Roadmaps with milestones that all coincidentally land after the token sale.*
- *Citations to academic crypto papers used for legitimacy without actually building on the cited result.*

**Useful base rate:** the modal whitepaper does not deliver what it promises. Treat the technical claims as serious only when accompanied by working code, third-party audits, and a real implementation history.

---

## 6. Backtest reports / quant strategy memos

**What it is:** A document presenting a quantitative trading strategy's historical performance, often as a justification for deploying capital. Can be internal (to a PM, to a risk committee) or external (to investors).

**What to read first:**
1. Headline performance metrics — Sharpe, max drawdown, return, win rate.
2. **The data section** — date range, instrument universe, how data was obtained, how splits/dividends/borrow were handled.
3. The methodology — entry/exit rules, position sizing, parameters.
4. Out-of-sample / walk-forward sections (if any).
5. Costs and frictions — modeled how, calibrated to what?

**Be especially skeptical of:**
- *In-sample headline numbers.* Sharpe 2.5 on data the strategy was designed against tells you nothing.
- *Window selection.* "We tested 2018–2024." Why not 2008–2024? Often because the longer window includes regimes the strategy fails in.
- *Survivorship bias.* Universe = "current S&P 500 components" backtested historically is biased. Was the universe reconstituted point-in-time?
- *Look-ahead bias.* Any data used in the analysis that wouldn't have been available at the decision time.
- *Overlapping windows / improper Sharpe annualization.* Inflates Sharpe.
- *Unrealistic costs.* Crypto strategies that ignore exchange-specific costs and slippage; equity strategies using mid-prices instead of bid/ask.
- *Capacity unmentioned.* The strategy may work at $1M and not at $100M.
- *The strategy ran during one regime.* 2018–2024 contains roughly 2 distinct regimes; Sharpe 1.4 across 2 regimes is weak generalization signal.
- *Parameter count.* If 8 parameters were tuned, the equivalent random-search Sharpe is non-trivial.
- *No live track record.* A backtest is a hypothesis, not a result.

**Useful base rate:** the modal "Sharpe 1.5+" backtest does not survive live deployment. Realized Sharpe is commonly 30–60% of backtest Sharpe for retail-accessible strategies; closer to 70–90% for institutional with proper validation discipline.

---

## 7. Policy papers / think tank reports

**What it is:** Reports from think tanks, government agencies, NGOs, advocacy organizations, intended to influence policy or public opinion.

**What to read first:**
1. Executive summary.
2. Recommendations section — the actual ask.
3. Funding / about page — who paid, what's their priors.
4. Methodology — often soft (literature review, expert interviews) rather than hard empirical.
5. Citations — pull a few and see if they actually support the claim made.

**Be especially skeptical of:**
- *Quantified claims with thin sourcing.* "X costs the economy $Y billion" is often back-of-envelope plus a multiplier.
- *Cherry-picked international comparisons.* "Country X does it this way" without engaging why other countries don't.
- *Asymmetric burden of proof.* Costs of action quantified precisely; costs of inaction described as "incalculable."
- *Coalitional alignment.* If every recommendation perfectly matches the funder's pre-existing position, the report is advocacy in research clothing.
- *Expert-survey methodology.* "We surveyed 50 experts" — selected how? Funded by whom?
- *Bipartisan or apolitical framing for a partisan thesis.* Sometimes genuine, often cosmetic.

**Useful base rate:** policy papers are usually directionally informative about the funder's preferences and only sometimes informative about the empirical question. Read for the question being asked, not just the answer given.

---

## When the type doesn't fit

If the report doesn't cleanly fit a category, default to:
1. Identify the closest category.
2. Use that category's reading strategy as a starting point.
3. Adjust for the specific differences (audience, funding source, claim type).

The categories aren't exhaustive — they're starting points that beat starting from scratch.
