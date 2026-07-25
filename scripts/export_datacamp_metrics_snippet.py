"""Print METRICS dict for pasting into DataCamp group_predictions cell."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import BASE

m = pd.read_csv(BASE / "outputs/metrics_2026_iter8.csv")
lookup = dict(zip(m["Team_Name"], zip(m["Attacking_Strength"], m["Defensive_Weakness"])))

DATACAMP_TO_MODEL = {
    "USA": "USA",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "UEFA Playoff A": "Bosnia and Herzegovina",
    "UEFA Playoff B": "Sweden",
    "UEFA Playoff C": "Turkey",
    "UEFA Playoff D": "Czechia",
    "FIFA Playoff 1": "DR Congo",
    "FIFA Playoff 2": "Iraq",
}

dc_teams = [
    "Mexico", "South Africa", "South Korea", "UEFA Playoff D", "Canada", "UEFA Playoff A",
    "USA", "Paraguay", "Australia", "UEFA Playoff C", "Qatar", "Switzerland", "Brazil",
    "Morocco", "Haiti", "Scotland", "Germany", "Curaçao", "Netherlands", "Japan",
    "Côte d'Ivoire", "Ecuador", "UEFA Playoff B", "Tunisia", "Spain", "Cabo Verde",
    "Belgium", "Egypt", "Saudi Arabia", "Uruguay", "Iran", "New Zealand", "Austria",
    "Jordan", "France", "FIFA Playoff 2", "Norway", "Senegal", "Algeria", "Argentina",
    "Portugal", "FIFA Playoff 1", "Uzbekistan", "Colombia", "England", "Croatia",
    "Ghana", "Panama",
]

print("METRICS = {")
for t in dc_teams:
    mn = DATACAMP_TO_MODEL.get(t, t)
    if mn not in lookup:
        print(f"    # MISSING: {t} -> {mn}")
        continue
    att, defe = lookup[mn]
    print(f"    {t!r}: {{'att': {att:.6f}, 'def': {defe:.6f}}},")
print("}")
