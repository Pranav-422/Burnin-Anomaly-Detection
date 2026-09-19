# Antigravity CLI Session — Changes Log
**Project: burnin-anomaly-detection**

## 1. Roast/Audit (flaws identified, no fixes yet)
- 100% recall but only ~36% precision — 85 good parts wrongly scrapped per run
- Isolation Forest `contamination=0.08` per lot forced ~8% of every lot flagged regardless of quality
- Module B's 95th-percentile safety threshold caused ~14.3% baseline false-positive rate across 3 parameters combined
- Final flag = OR across 6 conditions (Z-score, IQR, IsolationForest, ModuleB×3 params) — too permissive, guaranteed high FP
- "Physics-informed" dataset uses a simple power-law formula (`i0 * (1 + k*t^n)`) — defects generated and predicted with the *same* equation (somewhat circular validation)
- "Holdout generalization" test was just the same generator script with `seed=7` instead of `seed=42` — not genuinely out-of-distribution data
- Module B feature set (`c0, c24, c24-c0, (c24-c0)/24, (c24-c0)/c0`) is collinear — 200-tree XGBoost used to learn what's basically linear algebra
- Hardcoded Windows path in `api/main.py:172-177` pointing to an AI IDE's internal scratch folder — would break on Linux/any other machine
- Project lived on OneDrive Desktop, **no git repo initialized at all**
- Empty `notebooks/` folder, and 4+ redundant dashboard files (Streamlit app + several raw HTML dashboards + one 410KB standalone file)

## 2. First re-engineering pass
**Files changed:** `module_a/anomaly_detection.py`, `module_b/drift_predictor.py`, `api/main.py`, `run_pipeline.py`, `tests/test_holdout_generalization.py`, `README.md`, `CONTEXT_HANDOFF.md`

- Isolation Forest `contamination`: 0.08 → 0.02
- IQR multiplier `k`: 1.5 → 2.0
- Module B safety-slope percentile: 95.0 → 99.5
- Removed hardcoded Windows path; saved workflow diagram as `workflow.png` with relative path + fallback
- Added `assert precision >= 0.55` guardrail to holdout test
- Initialized git, added `.gitignore`, committed

**Result:** Recall stayed 100% (50/50, 0 FN). Combined FP: 85 → 13 parts (-84.7%). Combined precision: 37% → 79.4%.

## 3. GitHub push
- Pushed to `https://github.com/Pranav-422/Burnin-Anomaly-Detection` (main branch) — succeeded
- Attempted second push to `Pranav-421/Burnin-Anomaly-Detection` — failed (403, wrong account credentials cached). User said to drop it, unresolved.

## 4. Second re-engineering pass (accuracy/precision boost)
**Files changed:** `module_a/anomaly_detection.py`, `module_b/drift_predictor.py`, `api/main.py`, `run_pipeline.py`, `tests/test_holdout_generalization.py`, `README.md`, `CONTEXT_HANDOFF.md`

- IQR changed from two-sided to **one-directional** (`direction='upper'`, k=2.0) — reasoning: leakage/Iddq/delay only degrade upward, so a two-sided check was wrongly flagging *better-than-average* chips as defective
- Isolation Forest: added score-margin cutoff `if_score < -0.03` (previously flagged on any negative score just to fill the contamination quota)
- Module B safety-slope percentile: 99.5 → 99.8

**Result (training set, seed=42, 1000 parts / 50 defects):**
| Metric | Before | After |
|---|---|---|
| Accuracy | 98.70% | 99.60% |
| Combined precision | 79.4% (13 FP) | 92.6% (4 FP) |
| Combined recall | 100% | 100% |
| F1 | 88.5% | 96.2% |

**Result (holdout set, seed=7):**
| Metric | Before | After |
|---|---|---|
| Accuracy | 96.50% | 98.40% |
| Precision | 59.8% (35 FP) | 76.5% (16 FP) |
| Recall | 100% | 100% |

Committed as `perf: boost accuracy to 99.6%...` (commit `64c1d6f`).

## 5. Misc / environment fixes
- Ran full pipeline locally: Streamlit dashboard on `:8501`, FastAPI backend on `:8000`
- Fixed `ModuleNotFoundError: plotly` by installing the package and restarting Streamlit
- Unresolved: a difference in dashboard behavior when opened via Antigravity IDE vs directly (something running through Node.js) — investigation cut off by quota limit, never answered
- Several user requests ("summarize this chat as a prompt", "list all changes") went unanswered because the CLI hit its quota repeatedly

---

## ⚠️ Important flag — worth checking before trusting these numbers
Every accuracy/precision number above (99.6%, 92.6%, etc.) was validated **only against the project's own synthetic generator** — training set and "holdout" set are both produced by the *same* script, just different random seeds (42 vs 7). The roast itself calls this out as not real generalization testing.

This is the same blind spot we found earlier in this chat: this repo's numbers look great on its own clean distribution but recall collapsed to 14.6% when tested against a harder dataset with genuine borderline/near-threshold cases. The two rounds of threshold-tightening here (contamination down, IQR stricter, safety percentile up to 99.8) all trade recall for precision — which is the *opposite* direction of your stated FN-must-be-minimized policy, and could make the borderline-case collapse worse, not better.

**Recommendation:** before trusting "99.6% accuracy," re-run this re-engineered code against the harder ground truth dataset we built earlier (with `borderline_outlier`/`borderline_latent` cases) and check recall specifically — not just accuracy/precision on easy data.

---

## 6. Independent roast (this chat, separate from Antigravity CLI's own roast)
A second, unrelated critique pass — same project, viewed fresh — turned up the same core pattern as everything above: real, working code, but validation discipline is the weak point throughout.

- **Precision claims (92.6%, 99.6% accuracy) are self-graded.** All tuning and "holdout" testing used the project's own generator, just different seeds — not independent data.
- **"Physics-informed" framing oversells a simple power law** (`i0 * (1 + k*t^n)`). Defects are generated and later predicted with the same formula — the model is largely recovering its own generation logic, not learning general defect physics.
- **SHAP is applied to near-linear features** (`x`, `y`, `(y-x)/24`) — real tool, low information given how collinear the feature set is.
- **Environment/hygiene issues**: no git until very late, a hardcoded Windows path shipped into a FastAPI endpoint, a plotly dependency missing at runtime, and a dashboard folder with 4+ overlapping UI attempts instead of one committed direction.

None of this means the project is broken — it runs, the architecture (Module A + Module B + SHAP + OR-gate) is reasonable, and defect recall has stayed at 100% through every round of tuning. The gap is entirely in **how the numbers were validated**, not in whether the code works.
