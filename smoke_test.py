"""Quick validation: group inference + 500 Monte Carlo sims."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RESULTS_PATH
from src.features import get_world_cup_matches, load_results
from src.simulator import prepare_schedule_32_team, run_monte_carlo_simulation

df = load_results(RESULTS_PATH)
wc2018 = get_world_cup_matches(df, 2018)
wc2022 = get_world_cup_matches(df, 2022)

for year, wc in [(2018, wc2018), (2022, wc2022)]:
    sched, t2g = prepare_schedule_32_team(wc)
    groups = sorted(set(t2g.values()))
    print(f"{year}: {len(groups)} groups, group stage rows={( sched['stage']=='group').sum()}")

import pandas as pd
fake_metrics = pd.DataFrame({
    "Team_Name": list(set(wc2022["team1"]) | set(wc2022["team2"])),
    "Attacking_Strength": 1.2,
    "Defensive_Weakness": 1.0,
})
sched, _ = prepare_schedule_32_team(wc2022)
win, top4, stage = run_monte_carlo_simulation(sched, fake_metrics, iteration=2, simulations=500)
print("2022 MC top 5:", sorted(win.items(), key=lambda x: -x[1])[:5])
print("stage keys:", sorted(stage.keys()))
print("OK")
