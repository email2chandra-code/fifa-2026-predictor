from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

_kaggle_results = (
    BASE
    / "general"
    / "Kaggle_martj42international-football-results-from-1872-to-2017"
    / "results.csv"
)
_data_results = BASE / "data" / "results.csv"
RESULTS_PATH = _data_results if _data_results.exists() else _kaggle_results

_kaggle_shootouts = (
    BASE
    / "general"
    / "Kaggle_martj42international-football-results-from-1872-to-2017"
    / "shootouts.csv"
)
_data_shootouts = BASE / "data" / "shootouts.csv"
SHOOTOUTS_PATH = _data_shootouts if _data_shootouts.exists() else _kaggle_shootouts

SCHEDULE_2026 = BASE / "data" / "2026_FIFA_World_Cup_Schedule.csv.csv"
if not SCHEDULE_2026.exists():
    SCHEDULE_2026 = BASE / "data" / "2026_FIFA_World_Cup_Schedule.csv"

OPTA_PATH = BASE / "data" / "OPTA Rankings & Prediction for 2022 2026.csv"

# Column pairs in the combined OPTA CSV (team name, win %)
OPTA_YEAR_COLUMNS = {
    2018: (5, 6),
    2022: (3, 4),
    2026: (1, 2),
}
FORMER_NAMES_PATH = (
    BASE
    / "general"
    / "Kaggle_martj42international-football-results-from-1872-to-2017"
    / "former_names.csv"
)

SAVED_MODELS = BASE / "saved_models"
OUTPUTS = BASE / "outputs"
SQUAD_DIR = BASE / "data" / "squad"

HORIZON_DAYS = 1825  # 5 years
HALF_LIFE_DAYS = 730  # 2-year half-life for time decay
MONTE_CARLO_SIMULATIONS = 10_000

# Walk-forward backtest: tune on 2018, confirm on 2022, then forecast 2026.
BACKTEST_CONFIGS = {
    2018: {
        "target_date": "2018-06-14",
        "role": "development",
        "actual_winner": "France",
        "opener": "Russia vs Saudi Arabia",
    },
    2022: {
        "target_date": "2022-11-20",
        "role": "confirmation",
        "actual_winner": "Argentina",
        "opener": "Qatar vs Ecuador",
    },
}

PRODUCTION_CONFIG = {
    "target_date": "2026-06-11",
    "actual_winner": None,
}

# Iteration 3+ tournament importance tiers (spec: 1.0 / 0.85 / 0.5)
TOURNAMENT_WEIGHTS = {
    "tier_a": 1.00,  # World Cup, continental finals, qualifiers
    "tier_b": 0.85,  # Nations League, Gold Cup
    "tier_c": 0.50,  # Friendlies, exhibitions
}

# Elo anchor (iteration 4+): shrink embeddings toward Elo-implied priors
ELO_BASELINE = 1500.0
ELO_SCALE = 400.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADV = 65.0  # home edge in Elo expected-score formula (training history only)
ELO_ATTACK_COEF = 0.45
ELO_ANCHOR_WEIGHT = 0.15

# Dixon-Coles rho (iteration 5): low-score correlation / draw polishing
DC_RHO_INIT = -0.03

# Iteration 6a (Option A): opponent difficulty weighting.
# Per side, D(elo_opp) = clip(exp((elo_opp - elo_center) / elo_scale), min, max)
# using the opponent's walk-forward pre-match Elo (no leakage); the match row
# weight multiplier is the mean of the two sides' difficulties, so
# final_weight = time_weight * tournament_weight * D_opp.
# Tuned on 2018 Brier only (walk-forward): max_weight 1.15 -> 1.00 because
# up-weighting strong-opponent matches hurt calibration (0.7512 vs ~0.7473);
# elo_scale 400 -> 300 is a 2018 tie (0.7473 vs 0.7474) chosen for a deeper
# weak-opponent discount; the 0.35 floor rarely binds (spec default kept).
OPP_DIFFICULTY_CONFIG = {
    "min_weight": 0.35,  # floor — weak-opponent matches keep some signal
    "max_weight": 1.00,  # cap — discount-only; never up-weight elite fixtures
    "elo_center": 1500.0,  # Elo at which D = 1.0
    "elo_scale": 300.0,  # divisor in the exponent
}

# Iteration 6b (7): competition confederation strength C_conf.
# Applied to the tournament context (not team continent). Tune AFC/CAF on 2018 Brier.
COMPETITION_CONFEDERATION_STRENGTH = {
    "FIFA": 1.05,  # World Cup / global finals
    "UEFA": 1.00,  # baseline
    "CONMEBOL": 0.95,
    "CAF": 0.88,
    "AFC": 0.80,
    "CONCACAF": 0.78,
    "OFC": 0.65,
    "NEUTRAL": 0.70,  # unmapped friendlies / exhibitions
}

# Model iteration roles (for docs / runners)
ITERATION_ROLES = {
    1: "multiplicative baseline, uniform weights",
    2: "log-linear, uniform weights",
    3: "log-linear + tier weights (1.0/0.85/0.5) + time decay",
    4: "iter 3 + Elo anchor regularization",
    5: "iter 4 + Dixon-Coles tau (match-level calibration)",
    6: "iter 5 + opponent difficulty weighting (D_opp, Option A)",
    7: "iter 6 + competition confederation strength (C_conf, Option B / 6b)",
    8: "iter 7 + squad market-value prior blend (Transfermarkt top-23, beta tuned on 2018)",
}

# Iteration 8: blend Elo priors with log squad market value priors.
# attack_prior = (1-beta)*elo_z + beta*squad_z  (same for defense, negated)
SQUAD_BLEND_BETA = 0.20

TOURNAMENT_TIER_MAP = {
    "FIFA World Cup": "tier_a",
    "FIFA World Cup qualification": "tier_a",
    "UEFA Euro": "tier_a",
    "UEFA Euro qualification": "tier_a",
    "Copa América": "tier_a",
    "Copa América qualification": "tier_a",
    "African Cup of Nations": "tier_a",
    "African Cup of Nations qualification": "tier_a",
    "AFC Asian Cup": "tier_a",
    "AFC Asian Cup qualification": "tier_a",
    "Confederations Cup": "tier_a",
    "Oceania Nations Cup": "tier_a",
    "Arab Cup qualification": "tier_a",
    "UEFA Nations League": "tier_b",
    "CONCACAF Nations League": "tier_b",
    "Gold Cup": "tier_b",
    "Gold Cup qualification": "tier_b",
    "Arab Cup": "tier_b",
    "Gulf Cup": "tier_b",
    "CONCACAF Series": "tier_b",
    "EAFF Championship": "tier_c",
    "EAFF Championship qualification": "tier_c",
    "Kirin Cup": "tier_c",
    "Kirin Challenge Cup": "tier_c",
    "Merdeka Tournament": "tier_c",
    "Asian Games": "tier_c",
    "Dynasty Cup": "tier_c",
    "King's Cup": "tier_c",
    "Lunar New Year Cup": "tier_c",
    "FIFA Series": "tier_c",
    "Friendly": "tier_c",
}

TEAM_ALIASES = {
    "United States": "USA",
    "Czech Republic": "Czechia",
    "Türkiye": "Turkey",
}

# Official playoff winners (June 2026) — DataCamp placeholders -> resolved teams.
PLAYOFF_RESOLUTION = {
    "UEFA Playoff A": "Bosnia and Herzegovina",
    "UEFA Playoff B": "Sweden",
    "UEFA Playoff C": "Turkey",
    "UEFA Playoff D": "Czechia",
    "FIFA Playoff 1": "DR Congo",
    "FIFA Playoff 2": "Iraq",
}
