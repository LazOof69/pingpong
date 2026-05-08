# Session Handoff — 2026-05-08 close

This doc is for the next Claude session (likely on a different machine). Read in order, then start Phase 6 Step 1.

---

## 1. Repo

- GitHub: https://github.com/LazOof69/pingpong (private)
- Last commit: `ce1be1f` Initial commit, Phase 5 close + Phase 6 plan
- Branch: `main`
- Clone:
  ```bash
  gh repo clone LazOof69/pingpong
  cd pingpong
  ```

---

## 2. Current state (one paragraph)

**LB stuck at 0.3564** (5 subs over 4 days converge ±0.0003); leader at 0.40, gap +0.044. **Phase 5 (sgp + F1 paths) saturated** — 8 independent feature mechanisms all OOF 0.819, all LB 0.356. Ship CSV is `submissions/submission_v5z_md_advcal_bag2_T3.csv`. **Phase 6 plan is set** (3 steps, gate-driven; details in `docs/PHASE_6_PLAN_2026-05-08.md`).

---

## 3. Read order for new session

1. **`CLAUDE.md`** — working agreement (auto-loaded by Claude Code). §3 鐵律 (Plan → Debate#1 → Implement → Code Review → Debate#2 → Validate → Record) is mandatory for every phase.
2. **`docs/PHASE_6_PLAN_2026-05-08.md`** — full 3-step plan with concrete dataloader code, kill criteria, gates, expected lifts.
3. **`docs/V5Z_FINAL_REPORT.md` §2.43-2.45 + §3** — Phase 5 close retrospective + Phase 6 outline.
4. **`docs/EXTERNAL_RESOURCE_BRIEFING_2026-05-08.md`** — original briefing sent to external AI (context for the 3 PDF reports).
5. (Optional) `docs/{chatgpt,claude,gemini}_research.pdf` — original research sources. Note: claude_research.pdf is image-only (no extractable text); render with `pypdfium2` to PNG and Read those.

---

## 4. Open tasks (TaskList — these will not transfer between Claude sessions; recreate if needed)

- **#19 [pending] Phase 6 Step 1**: Random-prefix truncation augmentation (next up)
- **#20 [pending, blocked-by #19] Phase 6 Step 2**: Logit Adjustment + 20-sweep coord-ascent calibration
- **#21 [pending, blocked-by #20] Phase 6 Step 3**: ShuttleSet cross-sport pretraining

---

## 5. Concrete next action

Start Phase 6 Step 1 by running §3 鐵律:

1. **Plan**: write a plan doc (e.g. `docs/PHASE_6_STEP_1_PLAN.md`) covering:
   - Mechanism: random-prefix truncation per test L distribution `P(k=1..5+) = (0.32, 0.24, 0.16, 0.12, 0.16)`
   - Modifications to `src/train/train_v5z.py` (dataloader + LSTM input + LGB feature recompute)
   - 1-hr kill criterion: OOF AUC must drop from 0.819 → 0.65~0.75
   - Rollback: if LB lift < +0.005, abandon and pivot to Step 2 LA-only

2. **Debate #1**: invoke `adversarial-debate` skill on the plan (English args, Chinese output to user). Strongest expected attacks:
   - Attack: prefix recompute on LGB aggregate features is the leakage hot spot — verify `match_avg_L`, `cum_strokes_in_match`, any `strokeNumber.max()`-style stat is computed ONLY on prefix
   - Attack: BiLSTM directionality leaks full-rally state via backward pass — may need causal-only LSTM
   - Attack: 1-hr kill threshold 0.65~0.75 may need adjustment based on length-matched validation set construction

3. **Implement**: edit `src/train/train_v5z.py`. Critical files to inspect first:
   - `src/train/train_v5z.py` line ~270 (player aggregate feature build path)
   - `src/train/train_v5z.py` prepare_samples + build_lgb_features (these need prefix-aware versions)

4. **Code Review**: invoke `code-reviewer` skill, focus on leakage / OOF-test alignment / `V5Z_*` env var convention / `v5z_<tag>_s<seed>_*` artifact naming.

5. **Debate #2**: post-implementation attack (different angles than Debate #1). Focus: result interpretation, OOF/LB transfer assumption, distribution-shift fix verification.

6. **Validate** with three metrics (per CLAUDE.md §6):
   - Full OOF CV (α-blended, post-bias)
   - Test-weighted OOF (using test L distribution as weights) ← main decision metric
   - F1 by length bucket (k=1, 2, 3, 4-5, 6-10, 11+)

7. **Record**: write retrospective to `docs/V5Z_FINAL_REPORT.md` §2.46 (or wherever next available section is).

---

## 6. Key environment vars to know

```bash
# Train one seed (match-disjoint CV is mandatory for OOF)
V5Z_SEED=42 V5Z_TAG=v5z_md_<descriptor>_s42 V5Z_MATCH_DISJOINT=1 \
  python3 src/train/train_v5z.py

# New test mode (NEWTEST data refresh, see §1.6 of state)
V5Z_TEST_CSV=data/test_new.csv V5Z_PID_TEST_CSV=data/test_new.csv \
  V5Z_INFER_TEST_ONLY=1 V5Z_LSTM_WEIGHTS_TAG=v5z_md_full_s42 \
  V5Z_TAG=v5z_md_full_NEWTEST_s42 V5Z_SEED=42 V5Z_MATCH_DISJOINT=1 \
  python3 src/train/train_v5z.py

# Calibrate + ship
V5Z_SEEDS="42,1337" V5Z_ARTIFACT_PREFIX=v5z_md_full_NEWTEST \
  V5Z_SUBMISSION_SUFFIX=_NEWTEST \
  python3 src/train/advcal_match_disjoint.py
```

Full env var list: `docs/V5Z_FINAL_REPORT.md` §5.2.

---

## 7. Caveats for new machine

### 7.1 Files NOT in git (regeneratable but heavy)

- `artifacts/` (971MB) — npz of OOF/test predictions per seed/tag. **Required to skip retraining**. Either:
  - rsync from old machine: `rsync -avz old:pingpong/artifacts/ pingpong/artifacts/`
  - OneDrive sync (if project dir is in OneDrive — it is on the original machine)
  - Regenerate by running training (~hours per seed × 5 folds)
- `models/` (394MB) — PyTorch LSTM weights `models/v5z/v5z_md_full_s{42,1337}.pt`. Required for INFER_TEST_ONLY mode. Same options as artifacts/.
- `archive/` (17MB) — old rally-KFold artifacts, mostly stale, optional.

### 7.2 Memory NOT in git

Claude Code's per-machine memory at `~/.claude/projects/-mnt-c-...-pingpong/memory/` will NOT transfer. The new session starts blank from this directory. **All critical context is duplicated in `CLAUDE.md` + `docs/V5Z_FINAL_REPORT.md` + this handoff doc**, so this is OK — just re-create relevant memory entries as new context emerges.

If you want, you can manually copy `memory/MEMORY.md` and `memory/*.md` to the new machine's same path.

### 7.3 Python deps

```bash
pip install --user --break-system-packages \
  torch lightgbm scikit-learn pandas numpy pdfplumber pypdfium2
```

### 7.4 GPU

V5Z LSTM training assumes CUDA. Check `python3 -c "import torch; print(torch.cuda.is_available())"`. Single GPU 24GB is sufficient.

---

## 8. Quick-paste prompt for new session

Paste this in the new Claude Code session in the project root:

```
I'm continuing work on the table tennis prediction competition. We just closed Phase 5 (LB stuck at 0.3564, gap +0.044 to leader 0.40) and have a Phase 6 3-step plan ready.

Read in order:
1. CLAUDE.md (working agreement)
2. docs/SESSION_HANDOFF_2026-05-08.md (this is the handoff doc)
3. docs/PHASE_6_PLAN_2026-05-08.md (the plan)
4. docs/V5Z_FINAL_REPORT.md §2.43-2.45 + §3 (recent retrospective)

Then start Phase 6 Step 1 (random-prefix truncation augmentation) by writing the plan doc and running adversarial-debate skill on it (Debate #1) per §3 of CLAUDE.md.

Caveats: artifacts/ (971MB) and models/ (394MB) are .gitignored — verify they exist locally or sync from previous machine before training.
```

---

## 9. Phase 6 expected outcome (recap)

| Step | Effort | Expected LB lift | Cumulative LB |
|---|---|---|---|
| 0 (current) | — | — | 0.3564 |
| 1 truncation aug | 2-4h | +0.025~0.060 | 0.381~0.416 |
| 2 LA + 20-sweep cal | 4h (gate +0.005) | +0.012~0.025 | 0.393~0.441 |
| 3 ShuttleSet pretraining | 6-8h (gate +0.020 cum) | +0.005~0.020 | 0.398~0.461 |

Target: pass leader 0.40 line. Realistic: Step 1+2 gets there; Step 3 is overshoot insurance.
