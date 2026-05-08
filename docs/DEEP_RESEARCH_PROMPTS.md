# Prompt 模板 — 配合 DEEP_RESEARCH_BRIEFING 使用

每個模板貼在 briefing 內容之後當 follow-up question。

---

## 模板 1：Deep Research 模式（Gemini Deep Research / ChatGPT Pro Research / Perplexity Pro）

> Read the briefing above carefully. I'm rank #4 on this competition leaderboard and have exhausted my own ideas after 18 documented failed attempts.
>
> Please conduct deep research on the following:
>
> 1. **Past competitions in sports next-event prediction** (table tennis, badminton, tennis, baseball pitch prediction, soccer pass prediction). Find published winning solutions, especially those handling cross-event distribution shift. Cite specific Kaggle/AICUP/CodaLab competitions and their writeups if available.
>
> 2. **State-of-the-art techniques for cross-event style drift** in sports analytics — methods that handle "same player, different tournament" generalization. Include academic papers (last 5 years).
>
> 3. **Pseudo-labeling / semi-supervised learning at small scale** (15k labeled + 1k unlabeled sequences). What works when transformer-style SSL is too data-hungry? Specifically pseudo-labeling, FixMatch-style, or transductive approaches.
>
> 4. **Group-aware cross-validation strategies** for small competition data. Specifically, what's the trade-off between rally-stratified KFold (current) vs match-grouped KFold vs player-grouped KFold for the OOF metric to reliably predict leaderboard performance?
>
> 5. **Macro-F1 optimization beyond plug-in additive bias**. Are there post-hoc calibration methods that outperform Koyejo 2014's plug-in approach for small imbalanced multiclass settings?
>
> 6. **Table tennis tactical knowledge encoding**: what published features or representations from table tennis match analysis (academic or coaching literature) might capture patterns my hand-crafted 49 features miss?
>
> Output format:
> - For each direction, give: name, mechanism (1-2 sentences), evidence/citation, expected ROI for my situation, implementation complexity
> - End with a ranked top-3 recommendation with reasoning
> - Be skeptical of my own diagnosis — if I'm wrong about cross-event drift being the bottleneck, say so

---

## 模板 2：Chat 模式（GPT-4/5, Claude, Gemini regular chat）

> Read the briefing above. I'm currently rank #4. Be brutal and concrete.
>
> Tell me 5 things I should try next, ranked by expected leaderboard improvement per hour of work. For each:
> - **Mechanism**: how/why it might work (1 sentence)
> - **Concrete steps**: bullet list of implementation
> - **Expected LB impact**: with reasoning, not just optimism
> - **Failure mode**: how I'd know it didn't work
>
> Constraints I'm pushing back on:
> - Don't suggest things I've already tried (see §4 of briefing)
> - Don't suggest "more bagging" or "more seeds" (saturated)
> - Don't suggest pure architecture swaps (14 variants failed)
> - DO challenge my diagnosed bottleneck (cross-event drift) if you disagree
>
> If you think the right answer is "stop and ship baseline", say so honestly.

---

## 模板 3：Adversarial Second Opinion（找漏洞）

> Read the briefing above. Your job is NOT to suggest new directions — it's to attack my diagnosis and methodology.
>
> Specifically:
>
> 1. Is my "cross-event style drift" diagnosis actually correct? What evidence am I missing or misinterpreting? Could the LB regression have a totally different cause (data leakage in baseline, evaluation script difference, test set artifacts)?
>
> 2. Looking at my 18 failures — is there a pattern I'm not seeing? Are some of them actually correct experiments that I dismissed too quickly?
>
> 3. The Conditional Ensemble result (LB unchanged at 0.4044) — am I interpreting it correctly? What other explanations are there besides "unseen player isn't the cause"?
>
> 4. Is OOF 0.4816 → LB 0.4285 (gap -0.053) actually unusual for this kind of competition, or normal? What's the typical OOF/LB gap pattern in similar setups?
>
> 5. What questions have I NOT asked that I should have? What blind spots in my own analysis would a more experienced ML engineer immediately spot?
>
> Be extremely critical. Assume I'm overconfident in my framing.

---

## 模板 4：Domain Expert（攻 table tennis 知識）

> Read the briefing above. I'm trying to predict the next stroke in a table tennis rally but my purely statistical approach has plateaued at LB 0.4285.
>
> I suspect I'm missing **table tennis tactical knowledge** that human coaches use. Please:
>
> 1. List the most important tactical patterns in modern table tennis (third-ball attack, serve-receive variations, attack-defense transitions, rally tempo control, etc.)
>
> 2. For each pattern, suggest how I'd encode it as a feature given my available data (handId, strengthId, spinId, pointId, actionId, positionId, score state)
>
> 3. Are there well-known **player typology systems** in table tennis (e.g., "looper" vs "blocker" vs "chopper" vs "pushblocker")? How would I infer player type from observed strokes?
>
> 4. What does **service-receive theory** say about predicting stroke 2 (receive) given stroke 1 (serve)? My L=1 test rallies are 32% of the data, and current model F1 on them is poor.
>
> 5. For **rally-ending strokes** (winners or errors), is there published analysis on what tactical patterns predict them? My F1_point on last-stroke is 0.075 (terrible).
>
> 6. Would **stroke spin/spin-interaction** matter more than I'm currently modeling? Same for footwork/positioning context.
>
> Cite Chinese / English / Japanese / Korean tactical literature if relevant. Open to academic papers, coaching books, or pro-level analysis articles.

---

## 使用建議

**最高 ROI 流程**：
1. **模板 2** 給 Claude / GPT-5 / Gemini 各一份 → 拿 5×3=15 個方向想法
2. **模板 4** 給 Claude / GPT-5 → 拿桌球專業知識 features
3. **模板 3** 給 Gemini Deep Research → 攻擊我們的 diagnosis 找盲點
4. **模板 1** 給 Perplexity / ChatGPT Deep Research → 找 published 解法

把所有結果整理回來，我用 adversarial-debate 篩選出 top 2 值得實作的方向。

**注意**：Deep research 模式（模板 1）可能跑 5-10 分鐘並消耗 quota，先用模板 2/4 試水溫。
