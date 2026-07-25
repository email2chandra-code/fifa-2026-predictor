"""
Two-tournament walk-forward backtest:
  2018 — development (tune iteration choice)
  2022 — confirmation (2018 WC included in training)
"""

import pandas as pd

from .config import (
    BACKTEST_CONFIGS,
    HORIZON_DAYS,
    MONTE_CARLO_SIMULATIONS,
    OUTPUTS,
    RESULTS_PATH,
    SAVED_MODELS,
    SCHEDULE_2026,
)
from .evaluation import (
    brier_score,
    extract_team_metrics,
    predict_world_cup_matches,
    save_backtest_outputs,
    save_comparison_table,
    summarize_backtest,
)
from .features import get_world_cup_matches, load_and_weight_data, load_results
from .simulator import (
    prepare_schedule_32_team,
    run_monte_carlo_simulation,
    win_probs_to_dataframe,
)
from .io_utils import safe_read_csv, write_csv
from .training import execute_model_calibration


def _merge_summary(new_df, path):
    """Merge new (year, iteration) rows into an existing summary CSV."""
    if path.exists():
        old = safe_read_csv(path)
        merged = pd.concat([old, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["year", "iteration"], keep="last")
        merged = merged.sort_values(["year", "iteration"]).reset_index(drop=True)
    else:
        merged = new_df
    write_csv(merged, path)
    return merged


def run_backtest_year(
    year,
    iterations=(1, 2, 3, 4, 5),
    epochs=300,
    schedule_path=SCHEDULE_2026,
    simulations=MONTE_CARLO_SIMULATIONS,
    squad_blend_beta=None,
):
    """
    Train and evaluate all iterations for one World Cup year.

    Returns
    -------
    summary_df : DataFrame
        One row per iteration with Brier score and metadata.
    """
    if year not in BACKTEST_CONFIGS:
        raise ValueError(f"No backtest config for {year}. Available: {list(BACKTEST_CONFIGS)}")

    cfg = BACKTEST_CONFIGS[year]
    target_date = cfg["target_date"]

    df_results = load_results(RESULTS_PATH)
    wc_matches = get_world_cup_matches(df_results, year)
    tournament_schedule, _ = prepare_schedule_32_team(wc_matches)

    summaries = []

    for iteration in iterations:
        print(f"\n=== Backtest {year} | Iteration {iteration} | cutoff {target_date} ===")

        df_train, team_to_id, id_to_team, attack_prior, defense_prior, elo_ratings = (
            load_and_weight_data(
                RESULTS_PATH,
                target_date_str=target_date,
                iteration=iteration,
                schedule_path=schedule_path,
                horizon_days=HORIZON_DAYS,
                evaluation_matches=wc_matches,
                squad_cutoff_label=str(year) if iteration >= 8 else None,
                squad_blend_beta=squad_blend_beta,
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

        import torch

        SAVED_MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            SAVED_MODELS / f"weights_{year}_iter{iteration}.pt",
        )

        metrics = extract_team_metrics(model, id_to_team, elo_ratings=elo_ratings)
        metrics_path = OUTPUTS / f"metrics_{year}_iter{iteration}.csv"
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        write_csv(metrics, metrics_path)

        predictions = predict_world_cup_matches(
            model, wc_matches, team_to_id, iteration=iteration
        )

        scored = predictions[~predictions["skipped"]]
        brier = brier_score(
            list(zip(scored["p_home"], scored["p_draw"], scored["p_away"])),
            list(zip(scored["team1_score"], scored["team2_score"])),
        )

        save_backtest_outputs(predictions, year, iteration)

        print(f"  Running {simulations:,} Monte Carlo bracket simulations...")
        win_probs, _, _ = run_monte_carlo_simulation(
            tournament_schedule,
            metrics,
            iteration=iteration,
            simulations=simulations,
            format_32=True,
        )
        model_win_df = win_probs_to_dataframe(win_probs)
        write_csv(model_win_df, OUTPUTS / f"win_probs_{year}_iter{iteration}.csv")

        comparison_path = save_comparison_table(
            year, iteration=iteration, model_win_probs=model_win_df
        )

        summary = summarize_backtest(predictions, year, iteration, brier)
        top_pick = model_win_df.iloc[0] if not model_win_df.empty else None
        summary["model_top_pick"] = top_pick["team"] if top_pick is not None else None
        summary["model_top_pick_pct"] = top_pick["model_win_pct"] if top_pick is not None else None
        summaries.append(summary)

        print(f"  Brier score: {brier:.4f}  ({summary['matches_scored']} matches)")
        if top_pick is not None:
            print(f"  Model favorite: {top_pick['team']} ({top_pick['model_win_pct']:.1%})")
        print(f"  OPTA comparison → {comparison_path}")

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUTS / f"backtest_summary_{year}.csv"
    _merge_summary(summary_df, summary_path)
    _merge_summary(summary_df, OUTPUTS / "backtest_summary_all.csv")
    print(f"\nSaved summary → {summary_path}")
    return summary_df


def run_full_backtest(iterations=(1, 2, 3, 4, 5), epochs=300, simulations=MONTE_CARLO_SIMULATIONS):
    """Run 2018 development backtest, then 2022 confirmation backtest."""
    dev = run_backtest_year(2018, iterations=iterations, epochs=epochs, simulations=simulations)
    conf = run_backtest_year(2022, iterations=iterations, epochs=epochs, simulations=simulations)

    combined = pd.concat([dev, conf], ignore_index=True)
    combined_path = OUTPUTS / "backtest_summary_all.csv"
    _merge_summary(combined, combined_path)

    best_dev = dev.loc[dev["brier_score"].idxmin()]
    print("\n--- Recommended iteration (lowest 2018 Brier) ---")
    print(f"  Iteration {int(best_dev['iteration'])}: Brier {best_dev['brier_score']:.4f}")
    print(f"  Model favorite: {best_dev['model_top_pick']} ({best_dev['model_top_pick_pct']:.1%})")

    best_conf = conf[conf["iteration"] == best_dev["iteration"]].iloc[0]
    print(f"  Confirmed on 2022: Brier {best_conf['brier_score']:.4f}")
    print(f"  Model favorite: {best_conf['model_top_pick']} ({best_conf['model_top_pick_pct']:.1%})")
    print(f"  Actual winner:   {best_conf['actual_winner']}")

    return combined


if __name__ == "__main__":
    run_full_backtest()
