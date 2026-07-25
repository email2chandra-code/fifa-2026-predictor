"""
Group-stage advancement predictions for 2026 (12 groups → round of 32).

Outputs CSVs comparable to a printable prediction sheet:
  - group_predictions_2026_iter{N}.csv      (per team probabilities)
  - group_summary_2026_iter{N}.csv        (one row per group)
  - group_prediction_sheet_2026_iter{N}.csv (sheet-style columns)
  - knockout_qualifiers_2026_iter{N}.csv   (top 32 by P(advance))
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MONTE_CARLO_SIMULATIONS, OUTPUTS, PRODUCTION_CONFIG, SCHEDULE_2026, SHOOTOUTS_PATH
from src.io_utils import safe_read_csv, write_csv
from src.shootouts import build_shootout_rates, load_shootouts
from src.simulator import (
    build_prediction_sheet_table,
    prepare_schedule_2026,
    run_group_stage_monte_carlo,
)


def main():
    parser = argparse.ArgumentParser(description="2026 group-stage advancement Monte Carlo")
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=MONTE_CARLO_SIMULATIONS)
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to metrics CSV (default: outputs/metrics_2026_iter{N}.csv)",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Shootout history cutoff (default: PRODUCTION_CONFIG target_date)",
    )
    args = parser.parse_args()

    cutoff = args.cutoff or PRODUCTION_CONFIG["target_date"]

    metrics_path = Path(args.metrics) if args.metrics else OUTPUTS / f"metrics_2026_iter{args.iteration}.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics not found: {metrics_path}. Run: python run_production.py --iteration {args.iteration}"
        )

    schedule = prepare_schedule_2026(SCHEDULE_2026)
    df_metrics = safe_read_csv(metrics_path)

    print(f"=== 2026 group predictions | iter {args.iteration} | {args.simulations:,} sims ===")
    detail, summary, qualifiers = run_group_stage_monte_carlo(
        schedule, df_metrics, iteration=args.iteration, simulations=args.simulations
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    it = args.iteration
    paths = {
        "detail": OUTPUTS / f"group_predictions_2026_iter{it}.csv",
        "summary": OUTPUTS / f"group_summary_2026_iter{it}.csv",
        "sheet": OUTPUTS / f"group_prediction_sheet_2026_iter{it}.csv",
        "qualifiers": OUTPUTS / f"knockout_qualifiers_2026_iter{it}.csv",
    }

    write_csv(detail, paths["detail"])
    write_csv(summary, paths["summary"])
    write_csv(build_prediction_sheet_table(summary), paths["sheet"])
    write_csv(qualifiers, paths["qualifiers"])

    print("\n12 groups (A-L) -> 32 knockout qualifiers (24 auto + 8 best thirds)\n")
    print(build_prediction_sheet_table(summary).to_string(index=False))
    print(f"\nWrote:\n  {paths['sheet']}\n  {paths['detail']}\n  {paths['qualifiers']}")


if __name__ == "__main__":
    main()
