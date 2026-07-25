"""Tune SQUAD_BLEND_BETA on 2018 Brier (walk-forward)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import run_backtest_year
from src.config import OUTPUTS, SQUAD_BLEND_BETA

BETAS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]


def main():
    rows = []
    for beta in BETAS:
        summary = run_backtest_year(
            2018, iterations=(8,), epochs=200, simulations=1000, squad_blend_beta=beta
        )
        brier = float(summary.iloc[0]["brier_score"])
        rows.append((beta, brier))
        print(f"beta={beta:.2f}  2018 Brier={brier:.4f}")

    best_beta, best_brier = min(rows, key=lambda x: x[1])
    print(f"\nBest beta={best_beta:.2f}  Brier={best_brier:.4f}  (config={SQUAD_BLEND_BETA})")

    out = OUTPUTS / "squad_blend_tune_2018.csv"
    pd.DataFrame(rows, columns=["beta", "brier_2018"]).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
