"""Entry point: python run_backtest.py [--year 2018] [--iterations 1,2,3,4,5]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MONTE_CARLO_SIMULATIONS


def main():
    parser = argparse.ArgumentParser(description="FIFA World Cup walk-forward backtest")
    parser.add_argument("--year", type=int, default=None, help="Single year (2018 or 2022)")
    parser.add_argument("--iterations", type=str, default="1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=MONTE_CARLO_SIMULATIONS)
    args = parser.parse_args()

    from src.env_check import require_torch

    require_torch()
    from src.backtest import run_backtest_year, run_full_backtest

    iterations = tuple(int(x) for x in args.iterations.split(","))

    if args.year:
        run_backtest_year(
            args.year,
            iterations=iterations,
            epochs=args.epochs,
            simulations=args.simulations,
        )
    else:
        run_full_backtest(
            iterations=iterations,
            epochs=args.epochs,
            simulations=args.simulations,
        )


if __name__ == "__main__":
    main()
