# FIFA World Cup 2026 Predictor

**Independent ML forecasting project** — walk-forward Poisson/Dixon-Coles model: tune on **2018**, confirm on **2022**, forecast **2026**.

Author initiative (not a cohort or group assignment). Primary docs: `README.md`, this file, `Executive_Report.html`, `presentation.ipynb`.

---

## Two environments

| Task | Conda env | How to run |
|------|-----------|------------|
| **Presentation** (view CSVs only) | `base` | Jupyter → kernel **Python 3 (ipykernel)** |
| **Training** (backtest / production) | `deeplearning` | Anaconda Prompt, CMD, or wrapper scripts below |

`Anaconda Prompt` and `cmd` are the same for our purposes — both support `conda activate`.

---

## Presentation notebook (no PyTorch)

1. Open `presentation.ipynb`
2. Kernel → **Python 3 (ipykernel)** (`base`)
3. Cell 1:

```python
SKIP_TRAINING = True
SKIP_PRODUCTION = True
BEST_ITER = 4      # auto-updated from 2018 Brier if summary exists
ITER_2026 = 5      # manual choice for 2026 tables (independent of BEST_ITER)
```

4. **Kernel → Restart Kernel**, then run all cells
5. Dialog **File changed on disk** → **Load from disk** (keeps agent fixes; do not Overwrite)

---

## Training — pick ONE method

All run in `deeplearning`. Close any open `outputs/*.csv` first (avoids `Permission denied`).

### Method A — Anaconda Prompt or CMD (recommended)

```cmd
conda activate deeplearning
cd /d "C:\Users\sas_c\OneDrive\Documents\2026\AI Engineer\Course\MS AI Machine Learning\fifa_2026_predictor"
python run_backtest.py
python run_production.py --iteration 5
```

Quick test:

```cmd
python run_backtest.py --year 2018 --epochs 100 --simulations 2000
```

### Method B — PowerShell wrappers

```cmd
cd /d "C:\Users\sas_c\OneDrive\Documents\2026\AI Engineer\Course\MS AI Machine Learning\fifa_2026_predictor"
powershell -ExecutionPolicy Bypass -File run_backtest.ps1
powershell -ExecutionPolicy Bypass -File run_production.ps1 --iteration 5
```

### Method C — Without activating env

```cmd
conda run -n deeplearning python run_backtest.py
conda run -n deeplearning python run_production.py --iteration 5
```

### Method D — Double-click

`run_backtest_deeplearning.cmd`

---

## Group-stage predictions (12 groups → round of 32)

2026 has **12 groups** (A–L), not 8: top 2 from each group (24) plus **8 best third-place** teams = **32** knockout qualifiers.

```cmd
conda activate base
python run_group_predictions.py --iteration 5 --simulations 10000
```

Outputs:

| File | Content |
|------|---------|
| `group_prediction_sheet_2026_iter5.csv` | Sheet-style: predicted 1st, 2nd, best 3rd per group |
| `group_predictions_2026_iter5.csv` | Per-team P(1st/2nd/3rd/advance) |
| `group_summary_2026_iter5.csv` | One row per group |
| `knockout_qualifiers_2026_iter5.csv` | Top 32 teams by P(advance) |

Compare manually to `general/2026 FIFA World Cup Prediction Sheet.pdf`.

---

## Knockout stage probabilities (QF, SF, Final, Win)

Uses saved metrics only (no retraining):

```cmd
conda activate base
python run_stage_probabilities.py --iteration 5 --simulations 5000
```

Outputs `tournament_stage_probs_2026_iter{N}.csv` with columns:

| Column | Meaning |
|--------|---------|
| `p_reach_qf` | P(won Round of 16) — 8 teams per sim |
| `p_reach_sf` | P(won Quarter-final) — 4 teams per sim |
| `p_reach_final` | P(won Semi-final feeding the final) — 2 teams per sim |
| `model_win_pct` | P(won Final) |

`top4_probs_2026_iter{N}.csv` is kept for backward compatibility; `model_top4_pct` = `p_reach_sf` (SF reach, not Final reach).

---

## Executive report (HTML / PDF)

Cursor does not build PowerPoint natively. Use the HTML report instead:

```cmd
conda activate base
cd /d "C:\Users\sas_c\OneDrive\Documents\2026\AI Engineer\Course\MS AI Machine Learning\fifa_2026_predictor"
python generate_executive_report.py --iteration 5
```

Opens `Executive_Report.html` in your browser:

- Executive summary, KPIs, methodology
- **48-team flag collage** (flags via [flagcdn.com](https://flagcdn.com) — not official FIFA branding)
- Backtest Brier table, 2026 model vs OPTA

**Save as PDF:** Chrome/Edge → **Ctrl+P** → Save as PDF.

**Paste into PowerPoint:** Insert → Pictures (PDF), or copy sections from the browser.

Regenerate after any new backtest/production run.

---

## Model iterations

| Iter | Features |
|------|----------|
| 1 | Multiplicative Poisson, uniform weights |
| 2 | Log-linear, uniform weights |
| 3 | Tier weights **1.0 / 0.85 / 0.5** + 2-year time decay |
| 4 | Iter 3 + **Elo anchor** |
| 5 | Iter 4 + **Dixon-Coles tau** (match-level calibration) |

**Brier score** — mean squared error between predicted (home/draw/away) probabilities and the actual result; lower is better (~0.67 = naive one-third guess). Used to compare iterations on held-out World Cup matches.

**Dixon-Coles** — adjusts Poisson scorelines with a τ (ρ) factor on low scores (0–0, 1–1, etc.) to better calibrate draws; applied in iteration 5 training and match-level inference.

- Notebook **BEST_ITER** = lowest 2018 Brier (development year)
- **ITER_2026** = your choice for 2026 comparison / win-prob tables (often 5 if 2022 confirmation matters)

---

## Training data from `results.csv`

Each run uses **only the last 5 years before its cutoff** (`date < target_date`, `HORIZON_DAYS = 1825`). Not one pooled window from 2013→2026.

| Run | Cutoff | Matches | Date range |
|-----|--------|---------|------------|
| 2018 development | 2018-06-14 | 4,617 | 2013-06-15 → 2018-06-12 |
| 2022 confirmation | 2022-11-20 | 4,473 | 2017-11-22 → 2022-11-19 |
| 2026 production | 2026-06-11 | 5,239 | 2021-06-12 → 2026-06-05 |

Includes friendlies, qualifiers, Nations League, etc. (weighted by tier from iter 3+). World Cup matches on/after cutoff are **held out** for Brier scoring.

Refresh counts after updating `results.csv`:

```cmd
conda activate base
python report_training_counts.py
```

---

## Key output files

| File | Purpose |
|------|---------|
| `backtest_summary_all.csv` | Brier scores — all years × iterations |
| `comparison_2018_iter{N}.csv` | OPTA vs Model vs Actual (2018) |
| `comparison_2022_iter{N}.csv` | OPTA vs Model vs Actual (2022) |
| `comparison_2026_iter{N}.csv` | OPTA vs Model (2026) |
| `win_probs_2026_iter{N}.csv` | 2026 tournament win % by iteration |
| `tournament_stage_probs_2026_iter{N}.csv` | QF / SF / Final / Win reach probabilities |
| `top4_probs_2026_iter{N}.csv` | SF reach (`p_reach_sf`) — backward-compat alias |
| `final_2026_probabilities.csv` | Last production run (overwritten each time) |
| `Executive_Report.html` | Executive report (regenerate with script above) |

---

## Project deliverables (portfolio / sharing)

| Item | Notes |
|------|-------|
| `presentation.ipynb` | Walk-through notebook (`base` kernel, view-only) |
| `Executive_Report.html` | Executive summary → Print to PDF |
| `outputs/` | Backtest metrics, 2026 forecasts, group & stage probabilities |
| `src/` + `run_*.py` | Reproducible model pipeline |
| `group_prediction_sheet_2026_iter5.csv` | Per-group advancement picks |
| `tournament_stage_probs_2026_iter5.csv` | QF / SF / Final / Win reach % |
| `BASELINE_README.md` | This file |
| `requirements_baseline.txt` / `requirements_presentation.txt` | Dependencies |

**Stale / reference only:** `general/fifa_2026_predictor_ file structure.txt` (early scaffold — folder layout has changed).

---

## NumPy + PyTorch troubleshooting (`deeplearning` only)

If you see `Numpy is not available` or `_ARRAY_API not found`:

```cmd
powershell -ExecutionPolicy Bypass -File fix_deeplearning_env.ps1
```

Verify:

```cmd
conda activate deeplearning
python -c "import numpy; import torch; print(numpy.__version__); print(torch.tensor([1.0]))"
```

Expected: `1.26.4` and `tensor([1.])` with no error.

Running `python run_backtest.py` from `(base)` exits with a message to activate `deeplearning` — that is intentional.

---

## Not included (future work)

- Live in-tournament predictor (daily `results.csv` + rolling cutoff) — not automated yet
- Official FIFA images/logos — use flags or add media assets manually
- xG, FIFA rank, squad value as features (deferred; Elo used as anchor instead)
