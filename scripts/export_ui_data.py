"""Export model outputs to a JSON bundle for the web cheat sheet.

Iteration-aware: --iteration N reads outputs/*_iter{N}.csv and writes
web/data/iter{N}_bundle.json (e.g. iter 6 = opponent difficulty weighting).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DC_RHO_INIT, OUTPUTS, PRODUCTION_CONFIG, SCHEDULE_2026
from src.io_utils import safe_read_csv
from src.simulator import groups_from_schedule, prepare_schedule_2026
from src.venue_timezones import parse_schedule_datetime


def export_bundle(iteration=5, rho=None, cutoff=None, out_path=None):
    cutoff = cutoff or PRODUCTION_CONFIG["target_date"]
    rho = DC_RHO_INIT if rho is None else float(rho)

    metrics_path = OUTPUTS / f"metrics_2026_iter{iteration}.csv"
    group_sheet_path = OUTPUTS / f"group_prediction_sheet_2026_iter{iteration}.csv"
    stage_probs_path = OUTPUTS / f"tournament_stage_probs_2026_iter{iteration}.csv"

    for path in (metrics_path, group_sheet_path, stage_probs_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path}")

    metrics = safe_read_csv(metrics_path)
    group_sheet = safe_read_csv(group_sheet_path)
    stage_probs = safe_read_csv(stage_probs_path)
    schedule = prepare_schedule_2026(SCHEDULE_2026)
    groups = groups_from_schedule(schedule)

    participating = sorted({team for grp_teams in groups.values() for team in grp_teams})

    teams = []
    for _, row in metrics.iterrows():
        if row["Team_Name"] not in participating:
            continue
        teams.append(
            {
                "id": int(row["Team_ID"]),
                "name": row["Team_Name"],
                "att": float(row["Attacking_Strength"]),
                "def": float(row["Defensive_Weakness"]),
                "elo": float(row["Elo_Rating"]) if "Elo_Rating" in row and pd.notna(row["Elo_Rating"]) else None,
            }
        )

    model_picks = {}
    for _, row in group_sheet.iterrows():
        grp = str(row["Group"]).strip()
        model_picks[grp] = {
            "first": row["1st (model pick)"],
            "p_first": float(row["P(1st)"]),
            "second": row["2nd (model pick)"],
            "p_second": float(row["P(2nd)"]),
            "third_qualifier": row["3rd (best P among top-8 thirds)"],
            "p_third_qualifies": float(row["P(3rd qualifies)"]),
            "teams": [t.strip() for t in str(row["All teams"]).split(",")],
        }

    stage_prob_map = {}
    for _, row in stage_probs.iterrows():
        if row["team"] not in participating:
            continue
        stage_prob_map[row["team"]] = {
            "p_reach_qf": float(row["p_reach_qf"]),
            "p_reach_sf": float(row["p_reach_sf"]),
            "p_reach_final": float(row["p_reach_final"]),
            "model_win_pct": float(row["model_win_pct"]),
        }

    schedule_rows = []
    for _, row in schedule.iterrows():
        grp = row["group"] if pd.notna(row.get("group")) else None
        stage = row["stage"]
        round_label = str(grp) if stage == "knockout" and grp is not None else None
        host_city = str(row["host_city"]) if pd.notna(row.get("host_city")) else None
        date_s = str(row["date"])
        time_s = str(row["time_local"]) if pd.notna(row.get("time_local")) else None
        kickoff_utc = None
        venue_tz = None
        if host_city and time_s:
            kickoff_utc, venue_tz = parse_schedule_datetime(date_s, time_s, host_city)

        schedule_rows.append(
            {
                "match_number": int(row["match_number"]),
                "date": date_s,
                "day": str(row["day"]) if pd.notna(row.get("day")) else None,
                "time_local": time_s,
                "kickoff_utc": kickoff_utc,
                "venue_timezone": venue_tz,
                "team1": row["team1"],
                "team2": row["team2"],
                "group": str(grp) if grp is not None else None,
                "round": round_label,
                "stage": stage,
                "host_city": host_city,
                "stadium": str(row["stadium"]) if pd.notna(row.get("stadium")) else None,
            }
        )

    bundle = {
        "metadata": {
            "iteration": iteration,
            "cutoff": cutoff,
            "rho": rho,
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "teams": sorted(teams, key=lambda t: t["name"]),
        "groups": groups,
        "schedule": schedule_rows,
        "model_picks": model_picks,
        "stage_probs": stage_prob_map,
    }

    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / "web" / "data" / f"iter{iteration}_bundle.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export web UI JSON bundle")
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--cutoff", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    export_bundle(
        iteration=args.iteration,
        rho=args.rho,
        cutoff=args.cutoff,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
