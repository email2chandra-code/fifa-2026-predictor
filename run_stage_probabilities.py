"""
Knockout stage reach probabilities for 2026 (QF, SF, Final, Win).

Uses existing metrics CSV without retraining (like run_group_predictions.py).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MONTE_CARLO_SIMULATIONS, OUTPUTS, PRODUCTION_CONFIG, SCHEDULE_2026, SHOOTOUTS_PATH
from src.io_utils import safe_read_csv, write_csv
from src.shootouts import build_shootout_rates, load_shootouts
from src.simulator import (
    build_stage_probs_dataframe,
    prepare_schedule_2026,
    run_monte_carlo_simulation,
)


def _print_top10(df, col, label):
    print(f"\nTop 10 by {label}:")
    sub = df.nlargest(10, col)[["team", col]]
    for _, row in sub.iterrows():
        print(f"  {row['team']:<20} {row[col]:.1%}")


def main():
    parser = argparse.ArgumentParser(description="2026 knockout stage reach probabilities")
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

    print(f"=== 2026 stage probabilities | iter {args.iteration} | {args.simulations:,} sims ===")
    shootouts_df = load_shootouts(SHOOTOUTS_PATH)
    shootout_rates, shootout_counts = build_shootout_rates(shootouts_df, cutoff)
    win_probs, top4, stage_probs = run_monte_carlo_simulation(
        schedule,
        df_metrics,
        iteration=args.iteration,
        simulations=args.simulations,
        format_32=False,
        shootout_rates=shootout_rates,
        shootout_counts=shootout_counts,
    )

    stage_df = build_stage_probs_dataframe(df_metrics, stage_probs, win_probs)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"tournament_stage_probs_2026_iter{args.iteration}.csv"
    write_csv(stage_df, out_path)

    _print_top10(stage_df, "p_reach_qf", "P(reach QF)")
    _print_top10(stage_df, "p_reach_sf", "P(reach SF)")
    _print_top10(stage_df, "p_reach_final", "P(reach Final)")
    _print_top10(stage_df, "model_win_pct", "P(Win)")

    print(f"\nWrote: {out_path}")
    print(f"(top4_probs alias: P(reach SF), {len(top4)} teams with nonzero SF reach)")


if __name__ == "__main__":
    main()
