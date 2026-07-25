"""Build DataCamp competition fixture files from the local 2026 schedule."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BASE, SCHEDULE_2026
from src.simulator import prepare_schedule_2026
from src.venue_timezones import parse_schedule_datetime

OUT_DIR = BASE / "data" / "datacamp"

# DataCamp competition round labels and multipliers (from notebook scoring table).
ROUND_LABEL = {
    "R16": "Round of 32",  # local schedule labels first KO block as R16; verify vs DataLab
    "QF": "Quarter-final",
    "SF": "Semi-final",
    "3P": "Third-place playoff",
    "Final": "Final",
}

ROUND_MULTIPLIER = {
    "R16": 1,   # Round of 32 ×1 — split R16 block in DataLab if slots differ
    "QF": 4,
    "SF": 8,
    "3P": 8,
    "Final": 16,
}


def _round_key(code: str) -> str:
    return str(code).strip()


def _venue_string(row) -> str:
    stadium = str(row.get("stadium", "") or "").strip()
    city = str(row.get("host_city", "") or "").strip()
    if stadium and city:
        return f"{stadium}, {city}"
    return stadium or city or ""


def _utc_date(row) -> str:
    date_s = str(row.get("date", ""))
    time_s = str(row.get("time_local", "12:00"))
    city = row.get("host_city")
    try:
        iso, _ = parse_schedule_datetime(date_s, time_s, city)
        return iso
    except Exception:
        return date_s


def build_group_fixtures(schedule: pd.DataFrame) -> pd.DataFrame:
    group = schedule[schedule["stage"] == "group"].copy()
    rows = []
    for _, row in group.iterrows():
        rows.append(
            {
                "match_id": int(row["match_number"]),
                "group": str(row["group"]),
                "home_team": row["team1"],
                "away_team": row["team2"],
                "date": _utc_date(row),
                "venue": _venue_string(row),
            }
        )
    return pd.DataFrame(rows)


def build_knockout_slots(schedule: pd.DataFrame) -> pd.DataFrame:
    ko = schedule[schedule["stage"] == "knockout"].copy()
    rows = []
    for _, row in ko.iterrows():
        code = _round_key(row["group"])
        round_name = ROUND_LABEL.get(code, code)
        mult = ROUND_MULTIPLIER.get(code, 1)
        rows.append(
            {
                "match_id": int(row["match_number"]),
                "round": round_name,
                "multiplier": mult,
                "slot_home": row["team1"],
                "slot_away": row["team2"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=SCHEDULE_2026,
        help="Path to 2026 schedule CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for DataCamp CSVs",
    )
    args = parser.parse_args()

    schedule = prepare_schedule_2026(args.schedule)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gf = build_group_fixtures(schedule)
    ks = build_knockout_slots(schedule)

    gf_path = args.out_dir / "group_fixtures.csv"
    ks_path = args.out_dir / "knockout_slots.csv"
    gf.to_csv(gf_path, index=False)
    ks.to_csv(ks_path, index=False)

    print(f"Wrote {gf_path} ({len(gf)} group matches)")
    print(f"Wrote {ks_path} ({len(ks)} knockout slots)")
    print("\nKnockout multipliers (verify in DataLab notebook):")
    print(ks[["round", "multiplier"]].drop_duplicates().to_string(index=False))

    if len(gf) != 72:
        print(
            f"\nNote: competition expects 72 group matches; local schedule has {len(gf)}."
            " Group I may be missing 2 fixtures in your source file."
        )
    if len(ks) != 32:
        print(
            f"\nNote: competition expects 32 knockout slots; local schedule has {len(ks)}."
        )


if __name__ == "__main__":
    main()
