"""Cross-check FIFA official local schedule vs DataCamp submission-format fixtures (match_id / UTC)."""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schedule_datacamp_sync import (
    group_pairing_mismatch,
    is_group_match,
    local_fields_from_utc,
    match_group_rows,
    match_knockout_rows,
    utc_from_dc,
)
from src.venue_timezones import parse_schedule_datetime

LOCAL = ROOT / "data" / "2026_FIFA_World_Cup_Schedule.csv.csv"
DC_GROUP = ROOT / "general" / "datacamp_data_group_fixtures.csv"
DC_KO = ROOT / "general" / "datacamp_data_knockout_slots.csv"
OUT = ROOT / "outputs" / "schedule_time_mismatches.csv"
PAIR_OUT = ROOT / "outputs" / "schedule_pairing_mismatches.csv"


def compare_row(row, date_utc, stage, dc_match_id=None):
    iso, _ = parse_schedule_datetime(row["date"], row["time_local"], row["host_city"])
    u_assumed = datetime.fromisoformat(iso)
    u_official = utc_from_dc(date_utc)
    diff_h = (u_assumed - u_official).total_seconds() / 3600
    correct_date, correct_time, correct_day = local_fields_from_utc(date_utc, row["host_city"])
    return {
        "match_number": int(row["match_number"]),
        "stage": stage,
        "dc_match_id": dc_match_id,
        "host_city": row["host_city"],
        "team1": row["team1"],
        "team2": row["team2"],
        "current_date": row["date"],
        "current_time_local": row["time_local"],
        "current_day": row["day"],
        "correct_date": correct_date,
        "correct_time_local": correct_time,
        "correct_day": correct_day,
        "official_utc": date_utc,
        "offset_hours_error": round(diff_h, 2),
    }


def main():
    local = pd.read_csv(LOCAL)
    dc_group = pd.read_csv(DC_GROUP)
    dc_ko = pd.read_csv(DC_KO)

    rows = []
    pairing_rows = []

    local_g = local[local["group"].apply(is_group_match)].copy()
    group_map = match_group_rows(local_g, dc_group)
    for _, r in local_g.iterrows():
        local_no = int(r["match_number"])
        dc_row = group_map.get(local_no)
        if dc_row is None:
            pairing_rows.append(
                {
                    "match_number": local_no,
                    "group": r["group"],
                    "local_team1": r["team1"],
                    "local_team2": r["team2"],
                    "issue": "no_datacamp_fixture",
                }
            )
            continue

        mismatch = group_pairing_mismatch(r, dc_row)
        if mismatch:
            pairing_rows.append(
                {
                    "match_number": local_no,
                    "group": r["group"],
                    "local_team1": r["team1"],
                    "local_team2": r["team2"],
                    "dc_match_id": int(dc_row["match_id"]),
                    "dc_home": dc_row["home_team"],
                    "dc_away": dc_row["away_team"],
                    "issue": mismatch,
                }
            )

        rows.append(
            compare_row(r, dc_row["date_utc"], "group", dc_match_id=int(dc_row["match_id"]))
        )

    local_ko = local[~local["group"].apply(is_group_match)].copy()
    ko_map = match_knockout_rows(local_ko, dc_ko)
    for local_no, dc_no in sorted(ko_map.items()):
        r = local[local["match_number"] == local_no].iloc[0]
        dc_row = dc_ko[dc_ko["match_id"] == dc_no].iloc[0]
        rows.append(compare_row(r, dc_row["date_utc"], "knockout", dc_match_id=dc_no))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    pair_df = pd.DataFrame(pairing_rows)
    if not pair_df.empty:
        pair_df.to_csv(PAIR_OUT, index=False)
    elif PAIR_OUT.exists():
        PAIR_OUT.unlink()

    bad = df[df["offset_hours_error"].abs() > 0.01]
    day_bad = df[df["current_day"] != df["correct_day"]]
    group_bad = bad[bad["stage"] == "group"]
    ko_bad = bad[bad["stage"] == "knockout"]

    print(f"Compared {len(df)} matches ({len(group_map)} group, {len(ko_map)} knockout)")
    print(f"Time mismatches: {len(bad)} (group: {len(group_bad)}, knockout: {len(ko_bad)})")
    print(f"Day mismatches: {len(day_bad)}")
    print(f"Team pairing mismatches: {len(pairing_rows)}")

    unmatched_group = set(local_g["match_number"].astype(int)) - set(group_map.keys())
    if unmatched_group:
        print(f"Unmatched local group rows (skipped): {sorted(unmatched_group)}")

    unmatched = set(local_ko["match_number"].astype(int)) - set(ko_map.keys())
    if unmatched:
        print(f"Unmatched local knockout rows (skipped): {sorted(unmatched)}")

    if len(bad):
        print(
            bad[
                [
                    "match_number",
                    "stage",
                    "host_city",
                    "current_time_local",
                    "correct_time_local",
                    "offset_hours_error",
                ]
            ].to_string(index=False)
        )
    else:
        print("All paired times match DataCamp submission-format UTC when interpreted as venue-local.")

    if len(day_bad):
        print("\nDay name mismatches:")
        print(day_bad[["match_number", "current_day", "correct_day"]].to_string(index=False))

    if len(pairing_rows):
        print("\nTeam pairing mismatches:")
        print(
            pair_df[
                [
                    "match_number",
                    "group",
                    "local_team1",
                    "local_team2",
                    "dc_home",
                    "dc_away",
                    "issue",
                ]
            ].to_string(index=False)
        )
    else:
        print("All group pairings match DataCamp submission-format fixtures (with playoff resolution).")


if __name__ == "__main__":
    main()
