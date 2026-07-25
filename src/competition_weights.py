"""
Iteration 6b (7): competition confederation strength (C_conf).

Discounts matches in weaker competitive contexts (e.g. AFC qualifiers,
Kirin Cup) without penalizing teams by continent. Inferred from the
tournament name, not the national team.

final_weight (iter 7) = time_weight * tournament_weight * D_opp * C_conf
"""

import pandas as pd

from .config import COMPETITION_CONFEDERATION_STRENGTH

# Explicit tournament -> confederation bucket (checked before keyword rules).
TOURNAMENT_CONFEDERATION = {
    "FIFA World Cup": "FIFA",
    "FIFA World Cup qualification": "FIFA",
    "Confederations Cup": "FIFA",
    "UEFA Euro": "UEFA",
    "UEFA Euro qualification": "UEFA",
    "UEFA Nations League": "UEFA",
    "Copa América": "CONMEBOL",
    "Copa América qualification": "CONMEBOL",
    "African Cup of Nations": "CAF",
    "African Cup of Nations qualification": "CAF",
    "AFC Asian Cup": "AFC",
    "AFC Asian Cup qualification": "AFC",
    "Oceania Nations Cup": "OFC",
    "Gold Cup": "CONCACAF",
    "Gold Cup qualification": "CONCACAF",
    "CONCACAF Nations League": "CONCACAF",
    "CONCACAF Series": "CONCACAF",
    "EAFF Championship": "AFC",
    "EAFF Championship qualification": "AFC",
    "Kirin Cup": "AFC",
    "Kirin Challenge Cup": "AFC",
    "Arab Cup": "CAF",
    "Arab Cup qualification": "CAF",
    "Gulf Cup": "AFC",
    "Friendly": "NEUTRAL",
    "FIFA Series": "NEUTRAL",
    "Merdeka Tournament": "AFC",
    "Asian Games": "AFC",
    "Dynasty Cup": "AFC",
    "King's Cup": "AFC",
    "Lunar New Year Cup": "AFC",
}


def infer_confederation(tournament_name):
    """Map a results.csv tournament string to a confederation strength bucket."""
    if pd.isna(tournament_name):
        return "NEUTRAL"
    name = str(tournament_name).strip()
    if name in TOURNAMENT_CONFEDERATION:
        return TOURNAMENT_CONFEDERATION[name]

    lower = name.lower()
    if "fifa world cup" in lower or "confederations cup" in lower:
        return "FIFA"
    if "uefa" in lower or ("euro" in lower and "caf" not in lower):
        return "UEFA"
    if "copa" in lower or "conmebol" in lower:
        return "CONMEBOL"
    if "african" in lower or "afcon" in lower or "arab cup" in lower:
        return "CAF"
    if (
        "afc" in lower
        or "asian" in lower
        or "kirin" in lower
        or "eaff" in lower
        or "gulf cup" in lower
        or "merdeka" in lower
        or "dynasty cup" in lower
        or "king's cup" in lower
    ):
        return "AFC"
    if "concacaf" in lower or "gold cup" in lower or "nafc" in lower:
        return "CONCACAF"
    if "oceania" in lower or "ofc" in lower:
        return "OFC"
    if "friendly" in lower or "exhibition" in lower:
        return "NEUTRAL"
    return "NEUTRAL"


def competition_multiplier(tournament_name, strength_map=None):
    """Return C_conf for one tournament (Opta-inspired confederation index)."""
    strength = strength_map or COMPETITION_CONFEDERATION_STRENGTH
    conf = infer_confederation(tournament_name)
    return float(strength.get(conf, strength["NEUTRAL"]))


def competition_weight_series(tournaments, strength_map=None):
    """Vectorized C_conf multipliers aligned to a tournament column."""
    return tournaments.map(lambda t: competition_multiplier(t, strength_map=strength_map))
