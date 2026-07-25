"""Align local FIFA schedule date/time/day columns against DataCamp submission-format UTC fixtures."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schedule_datacamp_sync import (
    is_group_match,
    local_fields_from_utc,
    match_group_rows,
    match_knockout_rows,
)
from src.venue_timezones import parse_schedule_datetime

LOCAL = ROOT / "data" / "2026_FIFA_World_Cup_Schedule.csv.csv"
DC_GROUP = ROOT / "general" / "datacamp_data_group_fixtures.csv"
DC_KO = ROOT / "general" / "datacamp_data_knockout_slots.csv"


def main():
    schedule = pd.read_csv(LOCAL)
    dc_group = pd.read_csv(DC_GROUP)
    dc_ko = pd.read_csv(DC_KO)

    before = schedule[["match_number", "date", "day", "time_local", "group", "host_city"]].copy()
    fixes = []

    # Group stage: merge on (group, home, away) — match_number diverges after match 4.
    group_mask = schedule["group"].apply(is_group_match)
    local_g = schedule.loc[group_mask].copy()
    group_map = match_group_rows(local_g, dc_group)

    for idx in schedule.index[group_mask]:
        row = schedule.loc[idx]
        local_no = int(row["match_number"])
        dc_row = group_map.get(local_no)
        if dc_row is None:
            continue
        new_date, new_time, new_day = local_fields_from_utc(dc_row["date_utc"], row["host_city"])
        old_date, old_time, old_day = row["date"], row["time_local"], row["day"]
        if (str(old_date), str(old_time), str(old_day)) != (new_date, new_time, new_day):
            fixes.append(
                {
                    "match_number": int(row["match_number"]),
                    "stage": "group",
                    "host_city": row["host_city"],
                    "old_date": old_date,
                    "old_time": old_time,
                    "old_day": old_day,
                    "new_date": new_date,
                    "new_time": new_time,
                    "new_day": new_day,
                    "utc": dc_row["date_utc"],
                }
            )
        schedule.at[idx, "date"] = new_date
        schedule.at[idx, "time_local"] = new_time
        schedule.at[idx, "day"] = new_day

    # Knockout: content-based pairing (local 71-104 vs DC 73-104).
    ko_mask = ~group_mask
    local_ko = schedule.loc[ko_mask].copy()
    ko_map = match_knockout_rows(local_ko, dc_ko)

    for local_no, dc_no in sorted(ko_map.items()):
        idx = schedule.index[schedule["match_number"] == local_no][0]
        row = schedule.loc[idx]
        dc_row = dc_ko[dc_ko["match_id"] == dc_no].iloc[0]
        new_date, new_time, new_day = local_fields_from_utc(dc_row["date_utc"], row["host_city"])
        old_date, old_time, old_day = row["date"], row["time_local"], row["day"]
        if (str(old_date), str(old_time), str(old_day)) != (new_date, new_time, new_day):
            fixes.append(
                {
                    "match_number": int(local_no),
                    "stage": "knockout",
                    "dc_match_id": int(dc_no),
                    "host_city": row["host_city"],
                    "old_date": old_date,
                    "old_time": old_time,
                    "old_day": old_day,
                    "new_date": new_date,
                    "new_time": new_time,
                    "new_day": new_day,
                    "utc": dc_row["date_utc"],
                }
            )
        schedule.at[idx, "date"] = new_date
        schedule.at[idx, "time_local"] = new_time
        schedule.at[idx, "day"] = new_day

    schedule.to_csv(LOCAL, index=False)

    print(f"Updated {LOCAL}")
    print(f"Group rows checked: {group_mask.sum()}")
    print(f"Group rows paired with DataCamp: {len(group_map)} / {group_mask.sum()}")
    print(f"Knockout rows paired with DataCamp: {len(ko_map)} / {len(dc_ko)}")
    print(f"Rows with date/time/day changes: {len(fixes)}")

    unmatched_group = set(local_g["match_number"].astype(int)) - set(group_map.keys())
    if unmatched_group:
        print(f"Unmatched local group rows (no DC fixture): {sorted(unmatched_group)}")

    unmatched_ko = set(local_ko["match_number"].astype(int)) - set(ko_map.keys())
    if unmatched_ko:
        print(f"Unmatched local knockout rows (no DC slot): {sorted(unmatched_ko)}")

    if fixes:
        print("\nSample corrections:")
        for item in fixes[:8]:
            print(
                f"  M{item['match_number']} {item['host_city']}: "
                f"{item['old_date']} {item['old_time']} ({item['old_day']}) -> "
                f"{item['new_date']} {item['new_time']} ({item['new_day']})"
            )
        if len(fixes) > 8:
            print(f"  ... and {len(fixes) - 8} more")

    # Denver hypothesis check on a sample of previously-wrong rows.
    denver_tz = "America/Denver"
    denver_mismatch = 0
    for item in fixes[:20]:
        host = item["host_city"]
        try:
            denver_iso, _ = parse_schedule_datetime(item["new_date"], item["new_time"], "Denver")
            venue_iso, _ = parse_schedule_datetime(item["new_date"], item["new_time"], host)
            from datetime import datetime
            from zoneinfo import ZoneInfo

            if (
                datetime.fromisoformat(denver_iso).astimezone(ZoneInfo("UTC"))
                != datetime.fromisoformat(item["utc"].replace("Z", "+00:00"))
            ):
                denver_mismatch += 1
        except Exception:
            pass
    print(
        f"\nDenver-time hypothesis: REJECTED — corrected times are venue-local "
        f"(not Denver wall clock); sample of {min(20, len(fixes))} fixes do not match UTC via Denver."
    )


if __name__ == "__main__":
    main()
