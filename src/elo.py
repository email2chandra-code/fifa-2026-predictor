"""
Elo ratings from international match history.

Used as an anchor prior in iteration 4+ so attack/defense embeddings
regress toward a stable strength signal instead of overfitting shocks
(e.g. Saudi Arabia 2-1 Argentina) with no mean reversion.
"""

import numpy as np
import pandas as pd

from .config import (
    ELO_ATTACK_COEF,
    ELO_BASELINE,
    ELO_HOME_ADV,
    ELO_K_FACTOR,
    ELO_SCALE,
    OPP_DIFFICULTY_CONFIG,
)


def _match_score(home_score, away_score):
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def compute_elo_ratings(
    df_matches, team_to_id, k_factor=ELO_K_FACTOR, home_adv=ELO_HOME_ADV, return_prematch=False
):
    """
    Walk forward through chronological matches and return final Elo per team name.

    Home advantage is applied only when fitting Elo on historical qualifiers;
    World Cup evaluation uses neutral venues separately.

    With return_prematch=True, also return {row_index: (home_elo, away_elo)}
    captured *before* each match is applied (walk-forward, no leakage).
    """
    ratings = {team: float(ELO_BASELINE) for team in team_to_id}
    prematch = {}

    history = df_matches.sort_values("date").copy()
    for idx, row in history.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        if home not in ratings or away not in ratings:
            continue

        rh = ratings[home]
        ra = ratings[away]
        if return_prematch:
            prematch[idx] = (rh, ra)
        expected_home = 1.0 / (1.0 + 10.0 ** ((ra - rh + home_adv) / ELO_SCALE))
        actual_home, actual_away = _match_score(row["home_score"], row["away_score"])

        ratings[home] = rh + k_factor * (actual_home - expected_home)
        ratings[away] = ra + k_factor * (actual_away - (1.0 - expected_home))

    if return_prematch:
        return ratings, prematch
    return ratings


def opponent_difficulty_weights(df_matches, team_to_id, params=None):
    """
    Iteration 6a (Option A): per-side opponent difficulty multipliers D_opp.

    Each side's difficulty is D(elo_opp) = clip(exp((elo_opp - center) / scale),
    min_weight, max_weight) using the opponent's pre-match walk-forward Elo
    (no leakage). The home side's goals are weighted by the away team's Elo
    and vice versa, so beating weak opposition inflates attack ratings less.

    Returns (DataFrame[d_for_home, d_for_away] aligned to df_matches.index,
    final elo_ratings dict).
    """
    cfg = dict(OPP_DIFFICULTY_CONFIG)
    if params:
        cfg.update(params)

    ratings, prematch = compute_elo_ratings(df_matches, team_to_id, return_prematch=True)

    pre = pd.DataFrame.from_dict(
        prematch, orient="index", columns=["elo_home_pre", "elo_away_pre"]
    ).reindex(df_matches.index)

    def _difficulty(elo_opp):
        raw = np.exp((elo_opp - cfg["elo_center"]) / cfg["elo_scale"])
        return np.clip(raw, cfg["min_weight"], cfg["max_weight"])

    d_opp = pd.DataFrame(
        {
            "d_for_home": _difficulty(pre["elo_away_pre"]),  # home faces away's Elo
            "d_for_away": _difficulty(pre["elo_home_pre"]),
        },
        index=df_matches.index,
    ).fillna(1.0)
    return d_opp, ratings


def elo_to_embedding_priors(team_to_id, elo_ratings):
    """
    Map Elo ratings to attack/defense embedding priors for log-linear iterations.

    Stronger teams → higher attack prior, lower defensive-weakness prior.
    """
    n_teams = len(team_to_id)
    attack_prior = np.zeros(n_teams, dtype=np.float32)
    defense_prior = np.zeros(n_teams, dtype=np.float32)

    for team, team_id in team_to_id.items():
        elo = elo_ratings.get(team, ELO_BASELINE)
        z = (elo - ELO_BASELINE) / ELO_SCALE
        attack_prior[team_id] = z * ELO_ATTACK_COEF
        defense_prior[team_id] = -z * ELO_ATTACK_COEF

    return attack_prior, defense_prior


def build_elo_priors(df_matches, team_to_id):
    """Compute Elo ratings and return numpy prior arrays indexed by team id."""
    elo_ratings = compute_elo_ratings(df_matches, team_to_id)
    return elo_to_embedding_priors(team_to_id, elo_ratings), elo_ratings
