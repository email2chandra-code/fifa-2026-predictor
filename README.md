# FIFA World Cup 2026 Predictor

Independent machine-learning project that forecasts FIFA World Cup 2026 outcomes using a walk-forward Poisson / Dixon-Coles model trained on 150+ years of international match results.

**Pipeline:** tune on **2018** (development) → confirm on **2022** (holdout) → forecast **2026** (production).

Not affiliated with FIFA or OPTA. Author-owned initiative.

**Two related projects in this repo:**

| Project | Description | Key paths |
|---------|-------------|-----------|
| **1. FIFA 2026 Predictor** | Walk-forward Poisson / Dixon-Coles ML pipeline (7 iterations) | `src/`, `run_*.py`, `presentation.ipynb`, `web/` |
| **2. DataCamp competition** | Professor-encouraged DataCamp *Predict the FIFA World Cup 2026* submission | `general/datacamp_*.ipynb`, `data/datacamp/`, `scripts/build_datacamp_fixtures.py` |

DataCamp deliverables (group + knockout submissions, fixture CSVs, notes) live under `general/` and `data/datacamp/`. Regenerate competition fixtures with `python scripts/build_datacamp_fixtures.py`.

---

## Highlights

- **7 model iterations** — multiplicative Poisson → log-linear → tier/time weights → Elo anchor → Dixon-Coles τ → opponent difficulty → competition confederation strength
- **Walk-forward validation** — no future leakage; 5-year rolling training window per cutoff
- **Monte Carlo bracket** — 48-team 2026 format (12 groups, 32 knockout qualifiers)
- **Deliverables** — tournament win %, group advancement picks, QF/SF/Final reach probabilities, executive HTML report

| Production model (iter 7) | Role |
|---------------------------|------|
| 2018 Brier tuning | 6a D_opp + 6b C_conf competition weights |
| 2022 confirmation | Best held-out Brier (0.7262) |
| 2026 forecast | `run_production.py --iteration 7` |

---

## Quick start

### View results (no training)

```cmd
conda activate base
jupyter notebook presentation.ipynb
```

In Cell 1 set `SKIP_TRAINING = True` and `SKIP_PRODUCTION = True`, then restart the kernel and run all cells.

Open `Executive_Report.html` in a browser (Ctrl+P → Save as PDF).

### Full pipeline (training)

Requires conda env `deeplearning` with PyTorch and `numpy==1.26.4`. Close any open `outputs/*.csv` before running.

```cmd
conda activate deeplearning
cd fifa_2026_predictor
python run_backtest.py
python run_production.py --iteration 7
python run_group_predictions.py --iteration 7
python run_stage_probabilities.py --iteration 7
python generate_executive_report.py --iteration 7
```

### Lightweight scripts (use saved metrics, no retrain)

```cmd
python run_group_predictions.py --iteration 7 --simulations 10000
python run_stage_probabilities.py --iteration 7 --simulations 10000
```

### Prediction cheat sheet (web UI)

Head-to-head Dixon-Coles calculator, fillable 12-group bracket, knockout slots, model pre-fill, and print layout. No Streamlit or `.pkl` files.

```cmd
python scripts/export_ui_data.py --iteration 7
cd web
python -m http.server 8080
```

Open http://localhost:8080/prediction_cheat_sheet.html (use a local server; `file://` may not load the JSON bundle).

**Share with friends (phones):**
- Quick guide: https://fifa2026-chandra.netlify.app/for_friends.html
- Cheat sheet: https://fifa2026-chandra.netlify.app/prediction_cheat_sheet.html
- Executive report: https://fifa2026-chandra.netlify.app/Executive_Report.html

Open in any mobile browser; no install, and bracket picks save on the device.

---

## Project structure

```
fifa_2026_predictor/
├── src/                    # Model, features, simulator, evaluation
├── run_backtest.py         # Walk-forward 2018 + 2022 backtest
├── run_production.py       # 2026 production forecast
├── run_group_predictions.py
├── run_stage_probabilities.py
├── generate_executive_report.py
├── presentation.ipynb      # Walk-through notebook (base env)
├── Executive_Report.html   # Executive summary (regenerate via script)
├── outputs/                # Metrics, probabilities, comparisons
├── saved_models/           # PyTorch weights
├── web/                    # Printable prediction cheat sheet (HTML/CSS/JS)
├── scripts/                # UI data export helpers
├── data/                   # FIFA schedule, OPTA, DataCamp fixture copies (data/datacamp/)
├── general/                # Reference data, DataCamp notebook & submission outputs
```

---

## Key outputs (`outputs/`)

| File | Description |
|------|-------------|
| `backtest_summary_all.csv` | Brier scores — all years × iterations |
| `win_probs_2026_iter7.csv` | Tournament win probability per team |
| `tournament_stage_probs_2026_iter7.csv` | P(reach QF / SF / Final / Win) |
| `group_prediction_sheet_2026_iter7.csv` | Model picks: 1st, 2nd, best 3rd per group |
| `knockout_qualifiers_2026_iter7.csv` | Top 32 teams by P(advance to R32) |
| `comparison_2026_iter7.csv` | Model vs OPTA win % |

---

## Model iterations

| Iter | Features |
|------|----------|
| 1 | Multiplicative Poisson, uniform weights |
| 2 | Log-linear, uniform weights |
| 3 | Tier weights (1.0 / 0.85 / 0.5) + 2-year time decay |
| 4 | Iter 3 + Elo anchor regularization |
| 5 | Iter 4 + Dixon-Coles τ (match-level calibration) |
| 6 | Iter 5 + opponent difficulty — `D_opp = clip(exp((Elo_opp − 1500) / 300), 0.35, 1.00)` |
| 7 | Iter 6 + competition confederation strength — `C_conf` from tournament context (UEFA 1.0, AFC 0.80, …), `final_weight = time × tier × D_opp × C_conf` |

**Brier score** — mean squared error on held-out World Cup match probabilities (lower is better; ~0.67 = naive one-third guess).

---

## Environments

| Task | Conda env | Notes |
|------|-----------|-------|
| Notebook / reports | `base` | No PyTorch required |
| Training / simulation | `deeplearning` | `numpy==1.26.4`, `torch==2.2.2` |

See `requirements_baseline.txt` and `requirements_presentation.txt`. For NumPy/PyTorch issues on Windows, run `fix_deeplearning_env.ps1`.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| **README.md** (this file) | Overview & quick start |
| **BASELINE_README.md** | Full operational guide (paths, troubleshooting, wrappers) |
| **SNAPSHOT_NOTES.txt** | Milestone snapshot notes for archiving |
| **Executive_Report.html** | Printable executive summary |
| **Methodology_Handout.html/.pdf** | Faculty brief on training data & methodology |

---

## Limitations

- No xG, FIFA rank, or squad-market features (Elo used as stability anchor)
- Neutral venues assumed for World Cup simulation (home advantage zeroed at inference)
- No live in-tournament automation yet
- Asian qualifier goal volume can inflate attack ratings (see `japan_attack_sensitivity_2026_iter5.csv`)

---

## License & disclaimer

Independent research project. Do not use FIFA copyrighted logos without permission. Team flags in the executive report are sourced from [flagcdn.com](https://flagcdn.com).
