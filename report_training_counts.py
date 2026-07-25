"""Print how many matches from results.csv are used per walk-forward cutoff."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import BACKTEST_CONFIGS, HORIZON_DAYS, PRODUCTION_CONFIG, RESULTS_PATH
from src.features import load_and_weight_data

CUTS = [
    ("2018 development", BACKTEST_CONFIGS[2018]["target_date"]),
    ("2022 confirmation", BACKTEST_CONFIGS[2022]["target_date"]),
    ("2026 production", PRODUCTION_CONFIG["target_date"]),
]


def main():
    print(f"Source: {RESULTS_PATH}")
    print(f"Horizon: {HORIZON_DAYS} days (~5 years), iteration 5 weights\n")
    print(f"{'Run':<22} {'Cutoff':<12} {'Matches':>8}  {'From':<12} {'To':<12}")
    print("-" * 70)

    rows = []
    for label, target_date in CUTS:
        df, _, _, _, _, _ = load_and_weight_data(
            RESULTS_PATH, target_date, iteration=5
        )
        dmin = df["date"].min().date()
        dmax = df["date"].max().date()
        n = len(df)
        rows.append((label, target_date, n, dmin, dmax))
        print(f"{label:<22} {target_date:<12} {n:>8,}  {dmin}  {dmax}")

    print("\nRules:")
    print("  - Include: all international matches with date < cutoff")
    print("  - Window: last 5 years before cutoff only")
    print("  - Exclude: World Cup matches on/after cutoff (held out for Brier)")


if __name__ == "__main__":
    main()
