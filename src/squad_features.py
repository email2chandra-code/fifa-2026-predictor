"""Squad market-value priors for iteration 8 (blend with Elo)."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ELO_ATTACK_COEF, ELO_BASELINE, ELO_SCALE, SQUAD_DIR, TEAM_ALIASES
from .io_utils import safe_read_csv


def _standardize_team(name):
    if pd.isna(name):
        return name
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())

# Transfermarkt national-team pages (slug, club_id).
TM_NATIONAL_TEAMS = {
    "Algeria": ("algeria", 3614),
    "Argentina": ("argentina", 3437),
    "Australia": ("australia", 3433),
    "Austria": ("austria", 3383),
    "Belgium": ("belgium", 3382),
    "Bolivia": ("bolivia", 3581),
    "Bosnia and Herzegovina": ("bosnien-und-herzegowina", 3446),
    "Brazil": ("brazil", 3439),
    "Canada": ("canada", 3510),
    "Cape Verde": ("cape-verde", 4311),
    "Colombia": ("colombia", 3816),
    "Croatia": ("croatia", 3556),
    "Czechia": ("czech-republic", 3445),
    "Czech Republic": ("czech-republic", 3445),
    "Curaçao": ("curacao", 21843),
    "DR Congo": ("dr-congo", 3854),
    "Ecuador": ("ecuador", 3440),
    "Egypt": ("egypt", 3672),
    "England": ("england", 3299),
    "France": ("france", 3377),
    "Germany": ("germany", 3262),
    "Ghana": ("ghana", 3444),
    "Haiti": ("haiti", 14167),
    "Iran": ("iran", 3583),
    "Italy": ("italy", 3376),
    "Ivory Coast": ("ivory-coast", 3591),
    "Côte d'Ivoire": ("ivory-coast", 3591),
    "Japan": ("japan", 3435),
    "Jordan": ("jordan", 3589),
    "Kosovo": ("kosovo", 42967),
    "Mexico": ("mexico", 3443),
    "Morocco": ("morocco", 3575),
    "Netherlands": ("netherlands", 3379),
    "New Zealand": ("new-zealand", 3434),
    "Norway": ("norway", 3440),
    "Panama": ("panama", 3579),
    "Paraguay": ("paraguay", 3589),
    "Portugal": ("portugal", 3300),
    "Qatar": ("qatar", 3582),
    "Saudi Arabia": ("saudi-arabia", 3804),
    "Scotland": ("scotland", 3380),
    "Senegal": ("senegal", 3499),
    "Serbia": ("serbia", 3438),
    "South Africa": ("south-africa", 3440),
    "South Korea": ("korea-republic", 3431),
    "Spain": ("spain", 3375),
    "Sweden": ("sweden", 3443),
    "Switzerland": ("switzerland", 3384),
    "Tunisia": ("tunisia", 3670),
    "USA": ("united-states", 3505),
    "United States": ("united-states", 3505),
    "Uruguay": ("uruguay", 3441),
    "Uzbekistan": ("uzbekistan", 14163),
    "Wales": ("wales", 3864),
}


def squad_values_path(cutoff_label):
    """Return CSV path for a cutoff label like 2018, 2022, 2026."""
    return SQUAD_DIR / f"squad_values_{cutoff_label}.csv"


def load_squad_values(cutoff_label="2026"):
    """Load squad value table; returns dict team_name -> value_eur."""
    path = squad_values_path(cutoff_label)
    if not path.exists():
        path = SQUAD_DIR / "squad_values.csv"
    if not path.exists():
        return {}

    df = safe_read_csv(path)
    df["team"] = df["team"].apply(_standardize_team)
    return dict(zip(df["team"], df["squad_value_eur"].astype(float)))


def squad_to_embedding_priors(team_to_id, squad_values, elo_ratings=None):
    """
    Map log squad value to attack/defense priors (z-scored across teams with data).

    Missing teams: impute from Elo z-score mapped to squad scale when available,
    else zero prior (neutral).
    """
    n_teams = len(team_to_id)
    attack_prior = np.zeros(n_teams, dtype=np.float32)
    defense_prior = np.zeros(n_teams, dtype=np.float32)

    log_vals = []
    team_z = {}
    for team, team_id in team_to_id.items():
        val = squad_values.get(team)
        if val and val > 0:
            lv = float(np.log(val))
            log_vals.append(lv)
            team_z[team] = (team_id, lv)

    if not log_vals:
        return attack_prior, defense_prior

    mu = float(np.mean(log_vals))
    sigma = float(np.std(log_vals)) or 1.0

    for team, team_id in team_to_id.items():
        if team in squad_values and squad_values[team] > 0:
            z = (np.log(squad_values[team]) - mu) / sigma
        elif elo_ratings and team in elo_ratings:
            z = (elo_ratings[team] - ELO_BASELINE) / ELO_SCALE
        else:
            z = 0.0
        attack_prior[team_id] = z * ELO_ATTACK_COEF
        defense_prior[team_id] = -z * ELO_ATTACK_COEF

    return attack_prior, defense_prior


def blend_elo_squad_priors(attack_elo, defense_elo, attack_squad, defense_squad, beta):
    """Convex blend: (1-beta)*Elo + beta*Squad."""
    b = float(beta)
    attack = (1.0 - b) * attack_elo + b * attack_squad
    defense = (1.0 - b) * defense_elo + b * defense_squad
    return attack.astype(np.float32), defense.astype(np.float32)
