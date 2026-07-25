import numpy as np
import pandas as pd

from .competition_weights import competition_weight_series
from .elo import build_elo_priors, elo_to_embedding_priors, opponent_difficulty_weights
from .squad_features import blend_elo_squad_priors, load_squad_values, squad_to_embedding_priors
from .io_utils import safe_read_csv
from .config import (
    HALF_LIFE_DAYS,
    HORIZON_DAYS,
    SQUAD_BLEND_BETA,
    TEAM_ALIASES,
    TOURNAMENT_TIER_MAP,
    TOURNAMENT_WEIGHTS,
)

WORLD_CUP_MATCH_COUNT = 64


def load_results(results_path):
    """Load and parse the historical results CSV."""
    df = safe_read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def standardize_team_name(name):
    if pd.isna(name):
        return name
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def standardize_teams_in_df(df, columns):
    """Apply team name aliases to one or more columns in place."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)


def get_world_cup_matches(df_results, year):
    """
    Extract FIFA World Cup finals tournament matches for a given year.

    Returns a schedule-compatible frame with team1/team2 and actual scores.
    """
    wc = df_results[
        (df_results["tournament"] == "FIFA World Cup")
        & (df_results["date"].dt.year == year)
    ].copy()

    if len(wc) != WORLD_CUP_MATCH_COUNT:
        raise ValueError(
            f"Expected {WORLD_CUP_MATCH_COUNT} World Cup matches for {year}, found {len(wc)}"
        )

    standardize_teams_in_df(wc, ["home_team", "away_team"])

    wc = wc.rename(
        columns={
            "home_team": "team1",
            "away_team": "team2",
            "home_score": "team1_score",
            "away_score": "team2_score",
        }
    )
    wc["year"] = year
    wc["match_number"] = range(1, len(wc) + 1)
    return wc.sort_values("date").reset_index(drop=True)


def _tournament_weight(tournament_name):
    tier = TOURNAMENT_TIER_MAP.get(tournament_name, "tier_c")
    return TOURNAMENT_WEIGHTS[tier]


def load_and_weight_data(
    history_path,
    target_date_str,
    iteration=4,
    schedule_path=None,
    horizon_days=HORIZON_DAYS,
    evaluation_matches=None,
    opp_difficulty_params=None,
    competition_strength_map=None,
    squad_blend_beta=None,
    squad_cutoff_label=None,
):
    """
    Load history, apply temporal isolation and optional weighting by iteration.

    Parameters
    ----------
    history_path : str or Path
        Path to results.csv.
    target_date_str : str
        Training cutoff; matches on or after this date are excluded.
    iteration : int
        Model iteration (1-8) controlling decay, weights, Elo anchor, tau,
        opponent difficulty (6a), competition confederation strength (6b),
        and squad-value prior blend (8).
    schedule_path : str or Path, optional
        Future schedule used only to extend the team ID vocabulary.
    horizon_days : int
        Rolling training window length (default 1825 = 5 years).
    evaluation_matches : DataFrame, optional
        Held-out matches (e.g. World Cup) whose teams must appear in mappings.
    opp_difficulty_params : dict, optional
        Iteration 6+ overrides for OPP_DIFFICULTY_CONFIG (clip bounds, scale).
    competition_strength_map : dict, optional
        Iteration 7 overrides for COMPETITION_CONFEDERATION_STRENGTH.
    """
    df_history = load_results(history_path)

    df_schedule = None
    if schedule_path is not None:
        df_schedule = safe_read_csv(schedule_path)
        standardize_teams_in_df(df_schedule, ["team1", "team2"])

    target_dt = pd.to_datetime(target_date_str)

    # Temporal isolation — no future leakage
    df_history = df_history[df_history["date"] < target_dt].copy()

    # Rolling horizon window
    df_history["days_ago"] = (target_dt - df_history["date"]).dt.days
    df_history = df_history[df_history["days_ago"] <= horizon_days].copy()

    standardize_teams_in_df(df_history, ["home_team", "away_team"])

    team_names = set(df_history["home_team"].tolist() + df_history["away_team"].tolist())
    if df_schedule is not None:
        team_names.update(df_schedule["team1"].tolist() + df_schedule["team2"].tolist())
    if evaluation_matches is not None:
        team_names.update(
            evaluation_matches["team1"].tolist() + evaluation_matches["team2"].tolist()
        )

    unique_teams = sorted(team_names)
    team_to_id = {team: idx for idx, team in enumerate(unique_teams)}
    id_to_team = {idx: team for team, idx in team_to_id.items()}

    df_history["home_id"] = df_history["home_team"].map(team_to_id)
    df_history["away_id"] = df_history["away_team"].map(team_to_id)
    df_cleaned = df_history.dropna(
        subset=["home_score", "away_score", "home_id", "away_id"]
    ).copy()

    if iteration in [1, 2]:
        df_cleaned["final_weight"] = 1.0
    else:
        df_cleaned["tournament_weight"] = df_cleaned["tournament"].apply(_tournament_weight)
        lam = np.log(2) / HALF_LIFE_DAYS
        df_cleaned["time_weight"] = np.exp(-lam * df_cleaned["days_ago"])
        df_cleaned["final_weight"] = (
            df_cleaned["tournament_weight"] * df_cleaned["time_weight"]
        )

    attack_prior, defense_prior, elo_ratings = None, None, None
    if iteration >= 6:
        # Iteration 6a+: discount each side by opponent difficulty (walk-forward Elo).
        if iteration >= 7:
            df_cleaned["competition_weight"] = competition_weight_series(
                df_cleaned["tournament"], strength_map=competition_strength_map
            )
            df_cleaned["final_weight"] = (
                df_cleaned["final_weight"] * df_cleaned["competition_weight"]
            )

        d_opp, elo_ratings = opponent_difficulty_weights(
            df_cleaned, team_to_id, params=opp_difficulty_params
        )
        base_weight = df_cleaned["final_weight"]
        df_cleaned["weight_home_side"] = base_weight * d_opp["d_for_home"]
        df_cleaned["weight_away_side"] = base_weight * d_opp["d_for_away"]
        df_cleaned["opp_difficulty"] = (d_opp["d_for_home"] + d_opp["d_for_away"]) / 2.0
        df_cleaned["final_weight"] = base_weight * df_cleaned["opp_difficulty"]
        attack_elo, defense_elo = elo_to_embedding_priors(team_to_id, elo_ratings)
        if iteration >= 8:
            label = squad_cutoff_label or str(pd.to_datetime(target_date_str).year)
            squad_vals = load_squad_values(label)
            beta = SQUAD_BLEND_BETA if squad_blend_beta is None else squad_blend_beta
            attack_sq, defense_sq = squad_to_embedding_priors(
                team_to_id, squad_vals, elo_ratings=elo_ratings
            )
            attack_prior, defense_prior = blend_elo_squad_priors(
                attack_elo, defense_elo, attack_sq, defense_sq, beta
            )
        else:
            attack_prior, defense_prior = attack_elo, defense_elo
    elif iteration >= 4:
        (attack_prior, defense_prior), elo_ratings = build_elo_priors(
            df_cleaned, team_to_id
        )

    return df_cleaned, team_to_id, id_to_team, attack_prior, defense_prior, elo_ratings
