"""One-off fix: add missing Group I MD3 fixtures and remove bogus knockout rows."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schedule_datacamp_sync import is_group_match

LOCAL = ROOT / "data" / "2026_FIFA_World_Cup_Schedule.csv.csv"
BOGUS_KO = {86, 87}


def main():
    df = pd.read_csv(LOCAL)
    group = df[df["group"].apply(is_group_match)].copy()
    ko = df[~df["group"].apply(is_group_match) & ~df["match_number"].isin(BOGUS_KO)].copy()

    new_group = pd.DataFrame(
        [
            {
                "match_number": 71,
                "date": "6/26/2026",
                "day": "Friday",
                "time_local": "15:00",
                "team1": "Norway",
                "team2": "France",
                "group": "I",
                "host_city": "Foxborough",
                "stadium": "Gillette Stadium",
                "country": "USA",
            },
            {
                "match_number": 72,
                "date": "6/26/2026",
                "day": "Friday",
                "time_local": "15:00",
                "team1": "Senegal",
                "team2": "Iraq",
                "group": "I",
                "host_city": "Toronto",
                "stadium": "BMO Field",
                "country": "Canada",
            },
        ]
    )

    ko = ko.sort_values("match_number").reset_index(drop=True)
    ko["match_number"] = range(73, 73 + len(ko))

    out = pd.concat([group, new_group, ko], ignore_index=True)
    out = out.sort_values("match_number").reset_index(drop=True)
    out.to_csv(LOCAL, index=False)

    print(f"Wrote {LOCAL}")
    print(f"Group rows: {len(group) + len(new_group)} (added Norway-France, Senegal-Iraq)")
    print(f"Knockout rows: {len(ko)} (removed bogus local 86-87, renumbered 73-104)")
    print(f"Total: {len(out)}")


if __name__ == "__main__":
    main()
