"""Roll forward to 2026 production forecast."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MONTE_CARLO_SIMULATIONS

def run_production(iteration=5, epochs=300, simulations=MONTE_CARLO_SIMULATIONS, cutoff=None):
    from src.env_check import require_torch
    require_torch()
    import torch

    from src.config import (
        HORIZON_DAYS,
        MONTE_CARLO_SIMULATIONS,
        OUTPUTS,
        PRODUCTION_CONFIG,
        RESULTS_PATH,
        SAVED_MODELS,
        SCHEDULE_2026,
        SHOOTOUTS_PATH,
    )
    from src.evaluation import extract_team_metrics, save_comparison_table
    from src.features import load_and_weight_data
    from src.shootouts import build_shootout_rates, load_shootouts
    from src.simulator import (
        build_stage_probs_dataframe,
        prepare_schedule_2026,
        run_monte_carlo_simulation,
        win_probs_to_dataframe,
    )
    from src.training import execute_model_calibration
    
    
    target_date = cutoff or PRODUCTION_CONFIG["target_date"]
    print(f"=== 2026 Production | Iteration {iteration} | cutoff {target_date} ===")

    schedule = prepare_schedule_2026(SCHEDULE_2026)

    df_train, team_to_id, id_to_team, attack_prior, defense_prior, elo_ratings = (
        load_and_weight_data(
            RESULTS_PATH,
            target_date_str=target_date,
            iteration=iteration,
            schedule_path=SCHEDULE_2026,
            horizon_days=HORIZON_DAYS,
            squad_cutoff_label="2026" if iteration >= 8 else None,
        )
    )

    model = execute_model_calibration(
        df_train,
        num_teams=len(team_to_id),
        iteration=iteration,
        epochs=epochs,
        attack_prior=attack_prior,
        defense_prior=defense_prior,
    )

    SAVED_MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVED_MODELS / "weights_2026_production.pt")

    metrics = extract_team_metrics(model, id_to_team, elo_ratings=elo_ratings)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUTS / f"metrics_2026_iter{iteration}.csv", index=False)

    print(f"  Running {simulations:,} Monte Carlo simulations...")
    shootouts_df = load_shootouts(SHOOTOUTS_PATH)
    shootout_rates, shootout_counts = build_shootout_rates(shootouts_df, target_date)
    win_probs, top4, stage_probs = run_monte_carlo_simulation(
        schedule,
        metrics,
        iteration=iteration,
        simulations=simulations,
        format_32=False,
        shootout_rates=shootout_rates,
        shootout_counts=shootout_counts,
    )

    win_df = win_probs_to_dataframe(win_probs)
    win_df.to_csv(OUTPUTS / f"win_probs_2026_iter{iteration}.csv", index=False)
    win_df.to_csv(OUTPUTS / "final_2026_probabilities.csv", index=False)

    top4_df = win_probs_to_dataframe(top4, value_col="model_top4_pct")
    top4_df.to_csv(OUTPUTS / f"top4_probs_2026_iter{iteration}.csv", index=False)
    top4_df.to_csv(OUTPUTS / "final_2026_top4_probabilities.csv", index=False)

    stage_df = build_stage_probs_dataframe(metrics, stage_probs, win_probs)
    stage_df.to_csv(OUTPUTS / f"tournament_stage_probs_2026_iter{iteration}.csv", index=False)

    comparison_path = save_comparison_table(2026, iteration=iteration, model_win_probs=win_df)
    print(f"  Saved 2026 probabilities → {OUTPUTS / 'final_2026_probabilities.csv'}")
    print(f"  OPTA comparison → {comparison_path}")

    if not win_df.empty:
        top = win_df.iloc[0]
        print(f"  Model favorite: {top['team']} ({top['model_win_pct']:.1%})")

    return win_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=MONTE_CARLO_SIMULATIONS)
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Training/simulation cutoff date (default: PRODUCTION_CONFIG target_date)",
    )
    args = parser.parse_args()
    run_production(
        iteration=args.iteration,
        epochs=args.epochs,
        simulations=args.simulations,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()
