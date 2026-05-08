# Extraction Templates

Per-type extraction templates. Use the template that matches the report type identified in Phase 1. The templates are designed to surface what matters most for each type — academic papers care about pre-registration and effect sizes; backtest reports care about look-ahead and costs.

When extracting, use the user's own words for the **thesis** (one short quote allowed, under 15 words). Everything else should be in your own words — paraphrase, don't transcribe.

---

## Generic template (use when type is unclear)

```
THESIS: [one sentence, the load-bearing claim]

KEY CLAIMS (numbered):
1. [...]
2. [...]
3. [...]

METHODOLOGY: [one paragraph]

EVIDENCE TYPES: [data / citations / case studies / experiments / simulations / argument-only]

CONCLUSIONS: [what the report says the reader should believe or do]

STATED LIMITATIONS: [what the author admits]

UNSTATED ASSUMPTIONS: [the load-bearing things the report didn't surface]

SCOPE: [what universe/cases the claim is supposed to cover]
```

---

## Academic paper

```
THESIS: [one sentence]

CONTRIBUTION (the paper's claimed novelty):
- [specific contribution 1]
- [specific contribution 2]

HYPOTHESES: [what was tested]
PRE-REGISTERED? [yes / no / not specified]

DATA / SAMPLE:
- Source: [...]
- Size: N = [...]
- Time period: [...]
- Selection: [how was the sample chosen?]

METHOD:
- Design: [experimental / observational / simulation / theoretical]
- Statistical approach: [...]
- Multiple comparison correction: [yes / no / unclear]

KEY RESULTS:
- [Result 1]: effect size [...], statistical significance [...], confidence interval [...]
- [Result 2]: [...]

ROBUSTNESS CHECKS: [what additional analyses confirm the result]

STATED LIMITATIONS: [author's own caveats]

REPLICATION STATUS: [has anyone independently replicated this? note if unknown]

CONFLICTS / FUNDING: [as disclosed]
```

---

## Backtest report / quant strategy memo

```
THESIS: [one sentence — what edge does this strategy claim to capture?]

INSTRUMENT UNIVERSE: [what was traded]
DATA SOURCE: [...]
DATA TIME RANGE: [start — end]
NUMBER OF DISTINCT MARKET REGIMES IN RANGE: [estimate; e.g., 2 regimes for 2018–2024 in equities]

ENTRY / EXIT RULES: [...]
POSITION SIZING: [...]
NUMBER OF PARAMETERS: [count — and were any optimized?]

VALIDATION APPROACH:
- In-sample period: [...]
- Out-of-sample period: [...] (or "none stated" — flag this)
- Walk-forward? [yes / no]
- Cross-validation? [yes / no]

COSTS MODELED:
- Transaction costs: [bps or absolute]
- Slippage: [model used]
- Borrow / financing: [for shorts]
- Capacity: [stated? assumed?]

HEADLINE METRICS:
- Sharpe: [in-sample / out-of-sample / live]
- Max drawdown: [...]
- Return: [...]
- Win rate: [...]
- Time-to-recovery: [if reported]

LIVE TRACK RECORD: [exists? duration? performance vs backtest?]

KNOWN BIAS RISKS:
- Survivorship: [addressed / not addressed]
- Look-ahead: [addressed / not addressed]
- Selection on dependent variable: [...]
- Multiple hypothesis testing: [how many strategies were tried before this one was reported?]

CAPACITY: [stated AUM? plausible AUM?]

REGIME DEPENDENCY: [does the strategy rely on a specific regime — vol level, correlation regime, rate regime?]
```

---

## Sell-side equity research note

```
RATING: [Buy / Hold / Sell or equivalent]
PRICE TARGET: [...]
PRICE TARGET HORIZON: [12 months typical]
CURRENT PRICE: [at time of report]
IMPLIED RETURN: [PT vs current]

THESIS (in 1–3 sentences): [...]

KEY DRIVERS:
1. [...]
2. [...]
3. [...]

ESTIMATE REVISIONS: [vs prior, vs consensus]

RISKS (as listed): [...]

VALUATION METHOD: [DCF / multiples / SOTP / other]

KEY ASSUMPTIONS:
- [Revenue growth assumption]
- [Margin assumption]
- [Multiple assumption]
- [Terminal value assumption, if DCF]

DISCLOSED CONFLICTS:
- Banking relationship: [...]
- Holdings: [...]
- Market making: [...]

ANALYST TRACK RECORD ON THIS NAME: [if observable from prior notes]

COMPARABLE FIRMS' RATINGS: [are most analysts on the same side?]
```

---

## Industry / consulting report

```
HEADLINE CLAIM: [...]

KEY ARGUMENTS:
1. [...]
2. [...]

DATA SOURCES:
- [Source 1]: [primary survey / secondary research / proprietary database / vendor data]
- [Source 2]: [...]

SAMPLE:
- Survey N: [if applicable]
- Respondent profile: [job titles, regions, sectors]
- Selection method: [...]

QUANTIFIED CLAIMS: [list all major numbers and their sources]
- "$X market by year Y" — derived from: [...]
- "X% of leaders say..." — N = [...], asked when, by whom

RECOMMENDED ACTIONS: [what the report says you should do]

COMMERCIAL ALIGNMENT: [does the firm sell services that map to the recommended actions?]

FRAMEWORK USED: [if a proprietary framework — note name and whether definitions are operationalized]
```

---

## Policy paper / think tank report

```
PUBLISHER: [...]
PUBLISHER'S FUNDING: [as disclosed]
PUBLISHER'S STATED MISSION: [...]
PUBLISHER'S APPARENT POLITICAL VALENCE: [if discernible — note as inference]

CENTRAL CLAIM: [...]
RECOMMENDED POLICY: [the actual ask]

EVIDENCE TYPES USED:
- Statistical: [list claims with sources]
- Comparative (other countries / jurisdictions): [...]
- Expert opinion: [...]
- Case studies: [...]
- Modeled projections: [...]

KEY QUANTIFIED CLAIMS: [with each, note: where the number came from, and how strong that source is]

TREATMENT OF OPPOSITION:
- Are counter-arguments addressed? [...]
- Are they steelmanned or strawmanned?
- Are costs of recommendation quantified, or only benefits?

CITATION CHECK: [pull 2–3 citations and verify the cited source actually supports the claim made]
```

---

## Crypto / protocol whitepaper

```
PROTOCOL NAME: [...]
PROBLEM IT CLAIMS TO SOLVE: [in plain language]
WOULD A NON-CRYPTO SOLUTION WORK FOR THE SAME PROBLEM? [yes / no — and why]

TECHNICAL NOVELTY:
- [Component 1]: novel / standard / recombination of [...]
- [Component 2]: [...]

CONSENSUS / SECURITY MODEL: [what assumptions does the security rely on?]
THREAT MODEL (as stated): [if vague, flag]

TOKEN ECONOMICS:
- Supply schedule: [...]
- Inflation / emission: [...]
- Distribution: [team / investors / public / treasury — % each]
- Vesting: [schedule]
- Demand mechanism: [why does anyone need this token?]
- Burn / sink: [if any]

INCENTIVE ANALYSIS:
- What does the design incentivize each actor to do?
- What does the design rely on participants NOT doing?

TEAM:
- Identified individuals: [yes / partial / pseudonymous]
- Verifiable credentials: [yes / no]
- Prior projects: [...]

ROADMAP:
- Pre-token milestones: [...]
- Post-token milestones: [...]
- Capacity to deliver: [evidence?]

AUDITS / FORMAL VERIFICATION: [exist? from whom?]

WORKING CODE: [is there a functioning implementation? mainnet? testnet? proposal-only?]
```

---

## Company-issued material (10-K / earnings / investor deck)

```
COMPANY: [...]
DOCUMENT TYPE: [10-K / 10-Q / earnings transcript / investor deck / white paper]
PERIOD COVERED: [...]

HEADLINE FINANCIAL FIGURES:
- Revenue: [GAAP / non-GAAP — note both]
- Earnings / loss: [...]
- Cash: [...]
- Operating metrics: [as relevant — DAUs, ARR, etc.]

KEY NARRATIVES:
1. [What they want investors to focus on]
2. [...]

ADJUSTMENTS / NON-GAAP RECONCILIATION:
- What was adjusted out: [...]
- Recurrence assessment: [are these adjustments actually non-recurring, or have they recurred for multiple periods?]

SEGMENT BREAKDOWN: [if reported]

FORWARD-LOOKING STATEMENTS:
- Guidance: [revenue / EPS / operating metrics]
- Confidence framing: [hard guidance, "expect", or aspirational]

RISK FACTORS (10-K only):
- New risks added vs prior 10-K: [these are the most informative]
- Removed risks: [also informative]

Q&A HIGHLIGHTS (earnings only):
- Questions management deflected: [...]
- Questions answered with unusual specificity: [...]
- Tone shifts: [...]

KPI DEFINITIONS:
- Have any KPIs been redefined in this period? [...]
- Are operating metric definitions consistent with prior periods? [...]
```

---

## Notes on using templates

**Don't fill in fields with "N/A" if the report didn't address something — that's information.** Note explicitly: "Limitations section: not present" or "Out-of-sample period: not stated." The absence is part of the evaluation.

**If a template field is consistently empty across an entire report, the report may be of a different type than you classified it as.** Reconsider Phase 1 if many fields don't apply.

**If the user is in a domain not covered above** (medical study, legal brief, ML benchmark report, etc.), use the generic template and add domain-specific fields ad-hoc.
