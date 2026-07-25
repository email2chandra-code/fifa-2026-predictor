"""Walk-forward shootout win rates from historical penalty data."""

import pandas as pd

from .features import standardize_teams_in_df
from .io_utils import safe_read_csv

ALPHA = 2
BETA = 2
SHOOTOUT_BLEND_WEIGHT = 0.7
MIN_SHOOTOUTS_FOR_BLEND = 2


def load_shootouts(shootouts_path):
    """Load and parse shootouts.csv with standardized team names."""
    df = safe_read_csv(shootouts_path)
    df["date"] = pd.to_datetime(df["date"])
    standardize_teams_in_df(df, ["home_team", "away_team", "winner"])
    return df


def build_shootout_rates(shootouts_df, cutoff_date, alpha=ALPHA, beta=BETA):
    """
    Build Beta-shrunk shootout win rates for each team using data before cutoff.

    p = (wins + alpha) / (n + alpha + beta)
    """
    cutoff = pd.to_datetime(cutoff_date)
    hist = shootouts_df[shootouts_df["date"] < cutoff].copy()

    wins = {}
    counts = {}
    for _, row in hist.iterrows():
        t1, t2 = row["home_team"], row["away_team"]
        winner = row["winner"]
        for team in (t1, t2):
            counts[team] = counts.get(team, 0) + 1
        if winner in (t1, t2):
            wins[winner] = wins.get(winner, 0) + 1

    rates = {}
    for team, n in counts.items():
        w = wins.get(team, 0)
        rates[team] = (w + alpha) / (n + alpha + beta)

    return rates, counts


def shootout_win_prob(team1, team2, shootout_rates):
    """Return P(team1 wins a penalty shootout vs team2) from individual rates."""
    p1 = shootout_rates.get(team1, 0.5)
    p2 = shootout_rates.get(team2, 0.5)
    total = p1 + p2
    if total <= 0:
        return 0.5
    return p1 / total


def blend_knockout_tie_prob(
    team1,
    team2,
    att_dict,
    shootout_rates,
    shootout_counts=None,
    w=SHOOTOUT_BLEND_WEIGHT,
):
    """
    Blend shootout history with attack-ratio fallback for knockout ties.

    Uses weight w when both teams have at least MIN_SHOOTOUTS_FOR_BLEND records;
    otherwise falls back to attack ratio only.
    """
    a1 = max(att_dict.get(team1, 1.0), 1e-6)
    a2 = max(att_dict.get(team2, 1.0), 1e-6)
    p_att = a1 / (a1 + a2)

    counts = shootout_counts or {}
    n1 = counts.get(team1, 0)
    n2 = counts.get(team2, 0)
    if min(n1, n2) >= MIN_SHOOTOUTS_FOR_BLEND:
        p_pen = shootout_win_prob(team1, team2, shootout_rates)
        return w * p_pen + (1.0 - w) * p_att
    return p_att
