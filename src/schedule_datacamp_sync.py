"""Helpers to pair FIFA official schedule rows with DataCamp submission-format fixtures."""

from __future__ import annotations

import re
from datetime import datetime

from zoneinfo import ZoneInfo

from src.config import PLAYOFF_RESOLUTION, TEAM_ALIASES
from src.venue_timezones import HOST_CITY_TIMEZONE

GROUP_PATTERN = re.compile(r"^[A-L]$")

DC_ROUND_TO_LOCAL_GROUP = {
    "Round of 32": "R16",
    "Round of 16": "QF",
    "Quarter-final": "SF",
    "Semi-final": "SF",
    "Third-place playoff": "3P",
    "Final": "Final",
}

# Local SF rows 97-100 are quarter-finals; 101-102 are semi-finals.
LOCAL_SF_QF_MATCH_NUMBERS = {97, 98, 99, 100}
LOCAL_SF_SF_MATCH_NUMBERS = {101, 102}

# DC venue suffixes vs local host_city labels (same IANA zone).
HOST_CITY_ALIASES = {
    "los angeles": {"inglewood", "los angeles"},
    "boston": {"foxborough", "boston"},
    "santa clara": {"santa clara", "san francisco bay"},
    "monterrey": {"monterrey", "guadalupe"},
    "east rutherford": {"east rutherford"},
    "miami": {"miami gardens", "miami"},
    "dallas": {"dallas", "arlington"},
    "philadelphia": {"philadelphia"},
    "kansas city": {"kansas city"},
    "houston": {"houston"},
    "atlanta": {"atlanta"},
    "seattle": {"seattle"},
    "toronto": {"toronto"},
    "vancouver": {"vancouver"},
    "mexico city": {"mexico city"},
}


# DataCamp fixture labels that differ from local schedule naming.
DC_SCHEDULE_ALIASES = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Curaçao": "Curacao",
}


def normalize_team_for_match(name: str) -> str:
    """Canonical team label for pairing local schedule rows with DataCamp fixtures."""
    text = str(name).strip()
    text = PLAYOFF_RESOLUTION.get(text, text)
    text = DC_SCHEDULE_ALIASES.get(text, text)
    return TEAM_ALIASES.get(text, text)


def group_match_key(group: str, home: str, away: str) -> tuple[str, frozenset[str]]:
    """Order-independent pairing key within a group."""
    return (
        str(group).strip(),
        frozenset({normalize_team_for_match(home), normalize_team_for_match(away)}),
    )


def build_dc_group_lookup(dc_group) -> dict[tuple[str, frozenset[str]], object]:
    lookup: dict[tuple[str, frozenset[str]], object] = {}
    for _, row in dc_group.iterrows():
        key = group_match_key(row["group"], row["home_team"], row["away_team"])
        lookup[key] = row
    return lookup


def match_group_rows(local_group, dc_group) -> dict[int, object]:
    """Map local match_number -> DataCamp group row by (group, home, away) content."""
    lookup = build_dc_group_lookup(dc_group)
    mapping: dict[int, object] = {}
    for _, row in local_group.iterrows():
        key = group_match_key(row["group"], row["team1"], row["team2"])
        dc_row = lookup.get(key)
        if dc_row is not None:
            mapping[int(row["match_number"])] = dc_row
    return mapping


def group_pairing_mismatch(local_row, dc_row) -> str | None:
    """Return a mismatch description when resolved teams differ, else None."""
    local_home = normalize_team_for_match(local_row["team1"])
    local_away = normalize_team_for_match(local_row["team2"])
    dc_home = normalize_team_for_match(dc_row["home_team"])
    dc_away = normalize_team_for_match(dc_row["away_team"])
    if {local_home, local_away} == {dc_home, dc_away}:
        return None
    return (
        f"local {local_home} vs {local_away}; "
        f"datacamp {dc_row['home_team']} vs {dc_row['away_team']} "
        f"(resolved {dc_home} vs {dc_away})"
    )


def is_group_match(group: str) -> bool:
    return bool(GROUP_PATTERN.match(str(group).strip()))


def utc_from_dc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def local_fields_from_utc(date_utc: str, host_city: str) -> tuple[str, str, str]:
    """Return (date M/D/Y, time HH:MM, weekday name) in venue-local time."""
    utc = utc_from_dc(date_utc)
    tz_name = HOST_CITY_TIMEZONE.get(str(host_city).strip(), "UTC")
    local = utc.astimezone(ZoneInfo(tz_name))
    date_str = f"{local.month}/{local.day}/{local.year}"
    return date_str, local.strftime("%H:%M"), local.strftime("%A")


def normalize_ko_slot(slot: str, match_offset: int = 0) -> str:
    text = str(slot).strip()
    if re.search(r"third.?place|best 3rd", text, re.IGNORECASE):
        return "BEST3RD"
    match_ref = re.match(r"(Winner|Loser) Match (\d+)", text, re.IGNORECASE)
    if match_ref:
        role = match_ref.group(1).title()
        number = int(match_ref.group(2)) + match_offset
        return f"{role} Match {number}"
    return re.sub(r"\s+", " ", text)


def ko_slot_pair(home: str, away: str, match_offset: int = 0) -> tuple[str, str]:
    return (
        normalize_ko_slot(home, match_offset),
        normalize_ko_slot(away, match_offset),
    )


def _local_ko_bucket(row) -> str:
    code = str(row["group"]).strip()
    match_no = int(row["match_number"])
    if code == "SF":
        if match_no in LOCAL_SF_QF_MATCH_NUMBERS:
            return "Quarter-final"
        if match_no in LOCAL_SF_SF_MATCH_NUMBERS:
            return "Semi-final"
    mapping = {
        "R16": "Round of 32",
        "QF": "Round of 16",
        "3P": "Third-place playoff",
        "Final": "Final",
    }
    return mapping.get(code, code)


def _is_r32_style_row(row) -> bool:
    """Round-of-32 rows use group winner/runner-up slots (not Winner Match refs)."""
    home = str(row["team1"])
    away = str(row["team2"])
    if "Match" in home or "Match" in away:
        return False
    return _local_ko_bucket(row) == "Round of 32"


def _host_city_bonus(local_city: str, dc_venue: str) -> int:
    local = str(local_city).strip().lower()
    venue_city = str(dc_venue).split(",")[-1].strip().lower()
    if local == venue_city:
        return 15
    for _venue_key, aliases in HOST_CITY_ALIASES.items():
        if venue_city == _venue_key and local in aliases:
            return 15
    return 0


def score_ko_pair(local_row, dc_row, match_offset: int) -> int:
    local_pair = ko_slot_pair(local_row["team1"], local_row["team2"], match_offset)
    dc_pair = ko_slot_pair(dc_row["slot_home"], dc_row["slot_away"], 0)
    score = 0
    if local_pair[0] == dc_pair[0]:
        score += 60
    elif local_pair[0] == "BEST3RD" or dc_pair[0] == "BEST3RD":
        if local_pair[0] == "BEST3RD" and dc_pair[0] == "BEST3RD":
            score += 60
        elif local_pair[0] == "BEST3RD" or dc_pair[0] == "BEST3RD":
            score += 30
    if local_pair[1] == dc_pair[1]:
        score += 60
    elif local_pair[1] == "BEST3RD" or dc_pair[1] == "BEST3RD":
        if local_pair[1] == "BEST3RD" and dc_pair[1] == "BEST3RD":
            score += 60
        elif local_pair[1] == "BEST3RD" or dc_pair[1] == "BEST3RD":
            score += 30
    score += _host_city_bonus(local_row["host_city"], dc_row["venue"])
    return score


def _best_pair_score(local_row, dc_row, offsets: tuple[int, ...]) -> int:
    return max(score_ko_pair(local_row, dc_row, offset) for offset in offsets)


def _assign_round(local_rows, dc_rows, offsets: tuple[int, ...], min_score: int) -> dict[int, int]:
    """Greedy maximum-weight matching for one knockout round."""
    pairs: list[tuple[int, int, int]] = []
    for local_row in local_rows:
        local_no = int(local_row["match_number"])
        for dc_row in dc_rows:
            dc_no = int(dc_row["match_id"])
            score = _best_pair_score(local_row, dc_row, offsets)
            if score >= min_score:
                pairs.append((score, local_no, dc_no))
    pairs.sort(reverse=True)

    mapping: dict[int, int] = {}
    used_local: set[int] = set()
    used_dc: set[int] = set()
    for score, local_no, dc_no in pairs:
        if local_no in used_local or dc_no in used_dc:
            continue
        mapping[local_no] = dc_no
        used_local.add(local_no)
        used_dc.add(dc_no)
    return mapping


def _sort_dc_rows(dc_rows):
    return sorted(dc_rows, key=lambda r: (utc_from_dc(r["date_utc"]), int(r["match_id"])))


def _sort_local_rows(local_rows):
    return sorted(local_rows, key=lambda r: int(r["match_number"]))


def _positional_round_map(local_rows, dc_rows) -> dict[int, int]:
    local_sorted = _sort_local_rows(local_rows)
    dc_sorted = _sort_dc_rows(dc_rows)
    if len(local_sorted) != len(dc_sorted):
        return {}
    return {
        int(local_row["match_number"]): int(dc_row["match_id"])
        for local_row, dc_row in zip(local_sorted, dc_sorted)
    }


def match_knockout_rows(local_ko, dc_ko) -> dict[int, int]:
    """Map local match_number -> DataCamp match_id."""
    dc_by_round: dict[str, list] = {}
    for _, row in dc_ko.iterrows():
        dc_by_round.setdefault(str(row["round"]), []).append(row)

    local_candidates = []
    for _, row in local_ko.iterrows():
        if _local_ko_bucket(row) == "Round of 32" and not _is_r32_style_row(row):
            continue
        local_candidates.append(row)

    local_by_round: dict[str, list] = {}
    for row in local_candidates:
        local_by_round.setdefault(_local_ko_bucket(row), []).append(row)

    mapping: dict[int, int] = {}

    # Round of 32: slot content matching (local order permuted vs DataCamp).
    r32_local = local_by_round.get("Round of 32", [])
    r32_dc = dc_by_round.get("Round of 32", [])
    mapping.update(_assign_round(r32_local, r32_dc, (0,), min_score=90))

    # Later rounds: stable local numbering aligns with DataCamp when sorted by kickoff.
    for round_name in (
        "Round of 16",
        "Quarter-final",
        "Semi-final",
        "Third-place playoff",
        "Final",
    ):
        local_rows = local_by_round.get(round_name, [])
        dc_rows = dc_by_round.get(round_name, [])
        mapping.update(_positional_round_map(local_rows, dc_rows))

    return mapping


def dc_row_for_match(dc_df, match_id: int):
    rows = dc_df[dc_df["match_id"] == match_id]
    if rows.empty:
        return None
    return rows.iloc[0]
