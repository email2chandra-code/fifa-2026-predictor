"""
Monte Carlo tournament simulator: group stage + knockout bracket.

Supports 32-team World Cups (2018/2022) and the 48-team 2026 schedule format.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import standardize_team_name
from .io_utils import safe_read_csv
from .shootouts import blend_knockout_tie_prob

DEFAULT_SIMULATIONS = 10_000
HOME_ADV = 0.15
GROUP_STAGE_MATCHES_32 = 48

# Standard FIFA R16 pairings for 8 groups (confirmed against Qatar 2022)
R16_SLOTS_8_GROUPS = [
    ("1B", "2A"),
    ("1A", "2B"),
    ("1D", "2C"),
    ("1C", "2D"),
    ("1E", "2F"),
    ("1F", "2E"),
    ("1G", "2H"),
    ("1H", "2G"),
]


@dataclass
class Standing:
    team: str
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self):
        return self.gf - self.ga


@dataclass
class KnockoutResult:
    """Teams advancing at each knockout stage in one simulation."""

    r16_winners: list  # 8 teams (reach QF)
    qf_winners: list  # 4 teams (reach SF)
    sf_winners: list  # 2 teams (reach Final)
    champion: str


def infer_groups_from_matches(group_matches):
    """Cluster teams into groups of 4 from round-robin group-stage fixtures."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    for _, row in group_matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        union(t1, t2)

    buckets = defaultdict(list)
    for team in parent:
        buckets[find(team)].append(team)

    groups = sorted(buckets.values(), key=lambda g: sorted(g)[0])
    if not all(len(g) == 4 for g in groups):
        raise ValueError(f"Expected groups of 4, got sizes {[len(g) for g in groups]}")
    return groups


def build_team_to_group_map(groups):
    labels = "ABCDEFGHIJKL"[: len(groups)]
    mapping = {}
    for label, teams in zip(labels, groups):
        for team in teams:
            mapping[team] = label
    return mapping


def compute_lambdas(h_att, h_def, a_att, a_def, iteration, neutral=True, home_adv=HOME_ADV):
    ha = 0.0 if neutral else home_adv
        if iteration == 1:
        lh = max(h_att * a_def + ha, 1e-6)
        la = max(a_att * h_def, 1e-6)
        else:
        lh = np.exp(h_att - a_def + ha)
        la = np.exp(a_att - h_def)
    return float(lh), float(la)


def simulate_score(lambda_home, lambda_away):
    return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))


def knockout_winner(
    team1,
    team2,
    lambda_1,
    lambda_2,
    att_dict,
    shootout_rates=None,
    shootout_counts=None,
):
    """Simulate a knockout tie; draws resolved by shootout blend or attack ratio."""
    g1, g2 = simulate_score(lambda_1, lambda_2)
    if g1 > g2:
        return team1
    if g2 > g1:
        return team2
    if shootout_rates is not None:
        p_win = blend_knockout_tie_prob(
            team1, team2, att_dict, shootout_rates, shootout_counts
        )
    else:
        a1 = max(att_dict.get(team1, 1.0), 1e-6)
        a2 = max(att_dict.get(team2, 1.0), 1e-6)
        p_win = a1 / (a1 + a2)
    return team1 if np.random.random() < p_win else team2


def rank_group_standings(standings_dict):
    """Return ordered list of Standing objects (1st, 2nd, 3rd, 4th)."""
    rows = list(standings_dict.values())
    return sorted(rows, key=lambda s: (s.points, s.gd, s.gf), reverse=True)


def update_standings(standings, team1, team2, g1, g2):
    for team, gf, ga, pts in [
        (team1, g1, g2, 3 if g1 > g2 else (1 if g1 == g2 else 0)),
        (team2, g2, g1, 3 if g2 > g1 else (1 if g1 == g2 else 0)),
    ]:
        if team not in standings:
            standings[team] = Standing(team=team)
        standings[team].points += pts
        standings[team].gf += gf
        standings[team].ga += ga


def resolve_group_slot(slot, group_tables):
    """Resolve labels like 1A or 2B to a team name."""
    pos = int(slot[0])
    group = slot[1]
    ranked = group_tables[group]
    if pos == 1:
        return ranked[0].team
    if pos == 2:
        return ranked[1].team
    if pos == 3:
        return ranked[2].team
    raise ValueError(f"Unknown slot {slot}")


def simulate_group_stage_matches(matches, att_dict, def_dict, iteration, neutral=True):
    """Simulate a list of group fixtures; return per-group Standing dicts."""
    group_tables = defaultdict(dict)

    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        grp = row.get("group")
        if pd.isna(grp) or str(grp) in ("KO", "knockout"):
            continue

        h_att, h_def = att_dict.get(t1, 1.0), def_dict.get(t1, 1.0)
        a_att, a_def = att_dict.get(t2, 1.0), def_dict.get(t2, 1.0)
        l1, l2 = compute_lambdas(h_att, h_def, a_att, a_def, iteration, neutral=neutral)
        g1, g2 = simulate_score(l1, l2)

        if grp not in group_tables:
            group_tables[grp] = {}
        update_standings(group_tables[grp], t1, t2, g1, g2)

    ranked = {}
    for grp, standings in group_tables.items():
        ranked[grp] = rank_group_standings(standings)
    return ranked


def simulate_knockout_8_groups(
    group_tables, att_dict, def_dict, iteration, neutral=True, shootout_rates=None, shootout_counts=None
):
    """R16 -> QF -> SF -> Final for 8-group / 32-team format."""
    r16_winners = []
    for slot1, slot2 in R16_SLOTS_8_GROUPS:
        t1 = resolve_group_slot(slot1, group_tables)
        t2 = resolve_group_slot(slot2, group_tables)
        l1, l2 = _match_lambdas(t1, t2, att_dict, def_dict, iteration, neutral)
        r16_winners.append(
            knockout_winner(
                t1, t2, l1, l2, att_dict, shootout_rates, shootout_counts
            )
        )

    qf_winners = _pairwise_knockout(
        r16_winners, att_dict, def_dict, iteration, neutral, shootout_rates, shootout_counts
    )
    sf_winners = _pairwise_knockout(
        qf_winners, att_dict, def_dict, iteration, neutral, shootout_rates, shootout_counts
    )
    champion = _pairwise_knockout(
        sf_winners, att_dict, def_dict, iteration, neutral, shootout_rates, shootout_counts
    )[0]
    return KnockoutResult(r16_winners, qf_winners, sf_winners, champion)


def _match_lambdas(t1, t2, att_dict, def_dict, iteration, neutral):
    h_att, h_def = att_dict.get(t1, 1.0), def_dict.get(t1, 1.0)
    a_att, a_def = att_dict.get(t2, 1.0), def_dict.get(t2, 1.0)
    return compute_lambdas(h_att, h_def, a_att, a_def, iteration, neutral=neutral)


def _pairwise_knockout(
    teams, att_dict, def_dict, iteration, neutral, shootout_rates=None, shootout_counts=None
):
    if len(teams) == 1:
        return teams
    winners = []
    for i in range(0, len(teams), 2):
        t1, t2 = teams[i], teams[i + 1]
        l1, l2 = _match_lambdas(t1, t2, att_dict, def_dict, iteration, neutral)
        winners.append(
            knockout_winner(
                t1, t2, l1, l2, att_dict, shootout_rates, shootout_counts
            )
        )
    return winners


def parse_placeholder_slot(name):
    """Parse 2026 schedule placeholders like 'Winner Group A'."""
    name = str(name).strip()
    if name.startswith("Winner Group "):
        return ("1", name.replace("Winner Group ", "").strip())
    if name.startswith("Runner-up Group "):
        return ("2", name.replace("Runner-up Group ", "").strip())
    if name == "Third-place qualifier":
        return ("3Q", None)
    if name.startswith("Winner Match "):
        return ("WM", int(name.replace("Winner Match ", "").strip()))
    if name.startswith("Loser Match "):
        return ("LM", int(name.replace("Loser Match ", "").strip()))
    return ("TEAM", name)


def rank_third_place_teams(group_tables):
    thirds = []
    for grp, ranked in group_tables.items():
        third = ranked[2]
        thirds.append((third.points, third.gd, third.gf, third.team, grp))
    thirds.sort(reverse=True)
    return [t[3] for t in thirds[:8]]


# 2026 schedule match ranges (from data/2026_FIFA_World_Cup_Schedule.csv.csv)
R16_MATCH_NUMBERS_2026 = frozenset(range(71, 89))
QF_MATCH_NUMBERS_2026 = frozenset(range(89, 97))
SF_FINAL_FEED_MATCH_NUMBERS_2026 = frozenset((101, 102))
FINAL_MATCH_NUMBER_2026 = 104


def simulate_knockout_from_schedule(
    knockout_rows,
    group_tables,
    third_place_pool,
    att_dict,
    def_dict,
    iteration,
    shootout_rates=None,
    shootout_counts=None,
):
    """Resolve a 2026-style placeholder knockout path."""
    match_winners = {}
    r16_winners = []
    qf_winners = []
    sf_winners = []
    third_idx = [0]
    neutral = True
    ko_sorted = knockout_rows.sort_values("match_number")

    for _, row in ko_sorted.iterrows():
        raw1, raw2 = row["team1"], row["team2"]
        s1, s2 = parse_placeholder_slot(raw1), parse_placeholder_slot(raw2)
        t1 = _resolve_placeholder(s1, group_tables, third_place_pool, third_idx, match_winners)
        t2 = _resolve_placeholder(s2, group_tables, third_place_pool, third_idx, match_winners)

        l1, l2 = _match_lambdas(t1, t2, att_dict, def_dict, iteration, neutral)
        winner = knockout_winner(
            t1, t2, l1, l2, att_dict, shootout_rates, shootout_counts
        )
        mnum = int(row["match_number"])
        match_winners[mnum] = winner

        if mnum in R16_MATCH_NUMBERS_2026:
            r16_winners.append(winner)
        elif mnum in QF_MATCH_NUMBERS_2026:
            qf_winners.append(winner)
        elif mnum in SF_FINAL_FEED_MATCH_NUMBERS_2026:
            sf_winners.append(winner)

    champion = match_winners.get(FINAL_MATCH_NUMBER_2026) or match_winners.get(max(match_winners))
    return KnockoutResult(r16_winners, qf_winners, sf_winners, champion)


def _resolve_placeholder(slot, group_tables, third_pool, third_idx, match_winners):
    kind, val = slot
    if kind == "TEAM":
        return standardize_team_name(val)
    if kind in ("1", "2"):
        return resolve_group_slot(f"{kind}{val}", group_tables)
    if kind == "3Q":
        idx = min(third_idx[0], max(len(third_pool) - 1, 0))
        team = third_pool[idx] if third_pool else "Unknown"
        third_idx[0] += 1
        return team
    if kind == "WM":
        if val not in match_winners:
            raise KeyError(f"Winner Match {val} not yet played")
        return match_winners[val]
    if kind == "LM":
        return match_winners.get(val, "Unknown")
    raise ValueError(f"Cannot resolve slot {slot}")


def prepare_schedule_32_team(wc_matches):
    """Attach group labels to a 32-team World Cup (2018/2022) from results.csv."""
    schedule = wc_matches.copy()
    group_matches = schedule.iloc[:GROUP_STAGE_MATCHES_32]
    groups = infer_groups_from_matches(group_matches)
    team_to_group = build_team_to_group_map(groups)

    schedule["group"] = schedule.apply(
        lambda r: team_to_group.get(r["team1"], "KO")
        if r.name < GROUP_STAGE_MATCHES_32
        else "KO",
        axis=1,
    )
    schedule["stage"] = np.where(
        schedule.index < GROUP_STAGE_MATCHES_32, "group", "knockout"
    )
    return schedule, team_to_group


def prepare_schedule_2026(schedule_path):
    """Load and normalise the 2026 schedule CSV."""
    df = safe_read_csv(schedule_path)
    for col in ["team1", "team2"]:
        df[col] = df[col].apply(standardize_team_name)

    df["stage"] = df["group"].apply(
        lambda g: "group" if isinstance(g, str) and len(g) == 1 and g.isalpha() else "knockout"
    )
    return df


def _metrics_lookup(df_metrics):
    att = dict(zip(df_metrics["Team_Name"], df_metrics["Attacking_Strength"]))
    defn = dict(zip(df_metrics["Team_Name"], df_metrics["Defensive_Weakness"]))
    return att, defn


def simulate_one_tournament(schedule, df_metrics, iteration, format_32=True):
    """Run a single tournament simulation; return champion team name."""
    att_dict, def_dict = _metrics_lookup(df_metrics)
    group_rows = schedule[schedule["stage"] == "group"]

    group_tables = simulate_group_stage_matches(
        group_rows, att_dict, def_dict, iteration, neutral=True
    )

    if format_32:
        result = simulate_knockout_8_groups(
            group_tables, att_dict, def_dict, iteration, neutral=True
        )
        return result.champion

    ko_rows = schedule[schedule["stage"] == "knockout"].copy()
    third_pool = rank_third_place_teams(group_tables)
    result = simulate_knockout_from_schedule(
        ko_rows, group_tables, third_pool, att_dict, def_dict, iteration
    )
    return result.champion


def _count_knockout_stages(result, qf_counts, sf_counts, final_counts, win_counts):
    """Accumulate per-stage reach counts from one KnockoutResult."""
    for team in result.r16_winners:
        qf_counts[team] += 1
    for team in result.qf_winners:
        sf_counts[team] += 1
    for team in result.sf_winners:
        final_counts[team] += 1
    if result.champion:
        win_counts[result.champion] += 1


def run_monte_carlo_simulation(
    schedule,
    df_metrics,
    iteration=4,
    simulations=DEFAULT_SIMULATIONS,
    format_32=True,
    shootout_rates=None,
    shootout_counts=None,
):
    """
    Run Monte Carlo tournament simulations.

    Returns
    -------
    win_probs : dict
        team -> P(win tournament)
    top4_probs : dict
        team -> P(reach SF) — winners of quarter-finals (4 teams per sim).
        Alias for stage_probs["p_reach_sf"]; kept for backward compatibility.
    stage_probs : dict
        p_reach_qf, p_reach_sf, p_reach_final — per-team probabilities.
    """
    qf_counts = Counter()
    sf_counts = Counter()
    final_counts = Counter()
    win_counts = Counter()

    for _ in range(simulations):
        att_dict, def_dict = _metrics_lookup(df_metrics)
        group_rows = schedule[schedule["stage"] == "group"]
        group_tables = simulate_group_stage_matches(
            group_rows, att_dict, def_dict, iteration, neutral=True
        )

        if format_32:
            result = simulate_knockout_8_groups(
                group_tables,
                att_dict,
                def_dict,
                iteration,
                neutral=True,
                shootout_rates=shootout_rates,
                shootout_counts=shootout_counts,
            )
        else:
            third_pool = rank_third_place_teams(group_tables)
            ko_rows = schedule[schedule["stage"] == "knockout"]
            result = simulate_knockout_from_schedule(
                ko_rows,
                group_tables,
                third_pool,
                att_dict,
                def_dict,
                iteration,
                shootout_rates=shootout_rates,
                shootout_counts=shootout_counts,
            )

        _count_knockout_stages(result, qf_counts, sf_counts, final_counts, win_counts)

    win_probs = {team: c / simulations for team, c in win_counts.items()}
    top4_probs = {team: c / simulations for team, c in sf_counts.items()}
    stage_probs = {
        "p_reach_qf": {team: c / simulations for team, c in qf_counts.items()},
        "p_reach_sf": top4_probs,
        "p_reach_final": {team: c / simulations for team, c in final_counts.items()},
    }
    return win_probs, top4_probs, stage_probs


def win_probs_to_dataframe(prob_dict, value_col="model_win_pct"):
    """Convert a probability dict to a merge-friendly DataFrame."""
    if not prob_dict:
        return pd.DataFrame(columns=["team", value_col])
    return (
        pd.DataFrame(list(prob_dict.items()), columns=["team", value_col])
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def build_stage_probs_dataframe(df_metrics, stage_probs, win_probs):
    """Merge QF/SF/Final/Win probabilities for all teams in metrics."""
    teams = df_metrics["Team_Name"].tolist()
    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "p_reach_qf": stage_probs["p_reach_qf"].get(team, 0.0),
                "p_reach_sf": stage_probs["p_reach_sf"].get(team, 0.0),
                "p_reach_final": stage_probs["p_reach_final"].get(team, 0.0),
                "model_win_pct": win_probs.get(team, 0.0),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("model_win_pct", ascending=False)
        .reset_index(drop=True)
    )


def run_monte_carlo_from_metrics_csv(
    schedule,
    metrics_csv_path,
    iteration=4,
    simulations=DEFAULT_SIMULATIONS,
    format_32=True,
    shootout_rates=None,
    shootout_counts=None,
):
    """Convenience wrapper reading metrics CSV from disk."""
    df_metrics = safe_read_csv(metrics_csv_path)
    return run_monte_carlo_simulation(
        schedule,
        df_metrics,
        iteration=iteration,
        simulations=simulations,
        format_32=format_32,
        shootout_rates=shootout_rates,
        shootout_counts=shootout_counts,
    )


def groups_from_schedule(schedule):
    """Return {group_letter: [team, ...]} for 2026 group-stage fixtures."""
    group_rows = schedule[schedule["stage"] == "group"]
    mapping = {}
    for grp in sorted(group_rows["group"].dropna().unique()):
        rows = group_rows[group_rows["group"] == grp]
        teams = sorted(set(rows["team1"]).union(rows["team2"]))
        mapping[str(grp)] = teams
    return mapping


def run_group_stage_monte_carlo(
    schedule,
    df_metrics,
    iteration=5,
    simulations=DEFAULT_SIMULATIONS,
):
    """
    Estimate P(finish 1st / 2nd / 3rd) and P(advance to round of 32) per team.

    2026 format: 12 groups of 4 → top 2 auto-qualify + 8 best third-place teams (32 total).
    """
    att_dict, def_dict = _metrics_lookup(df_metrics)
    group_rows = schedule[schedule["stage"] == "group"]
    group_teams = groups_from_schedule(schedule)

    first = Counter()
    second = Counter()
    third = Counter()
    third_qualify = Counter()
    advance = Counter()
    group_first = {g: Counter() for g in group_teams}
    group_second = {g: Counter() for g in group_teams}

    for _ in range(simulations):
        group_tables = simulate_group_stage_matches(
            group_rows, att_dict, def_dict, iteration, neutral=True
        )
        qualifying_thirds = set(rank_third_place_teams(group_tables))

        for grp, ranked in group_tables.items():
            for pos, standing in enumerate(ranked):
                team = standing.team
                if pos == 0:
                    first[team] += 1
                    group_first[grp][team] += 1
                    advance[team] += 1
                elif pos == 1:
                    second[team] += 1
                    group_second[grp][team] += 1
                    advance[team] += 1
                elif pos == 2:
                    third[team] += 1
                    if team in qualifying_thirds:
                        third_qualify[team] += 1
                        advance[team] += 1

    rows = []
    for grp, teams in group_teams.items():
        for team in teams:
            rows.append(
                {
                    "group": grp,
                    "team": team,
                    "p_finish_1st": first[team] / simulations,
                    "p_finish_2nd": second[team] / simulations,
                    "p_finish_3rd": third[team] / simulations,
                    "p_third_qualifies": third_qualify[team] / simulations,
                    "p_advance_r32": advance[team] / simulations,
                }
            )

    detail = pd.DataFrame(rows).sort_values(["group", "p_advance_r32"], ascending=[True, False])

    summary_rows = []
    for grp, teams in sorted(group_teams.items()):
        sub = detail[detail["group"] == grp]
        pick1 = sub.loc[sub["p_finish_1st"].idxmax()]
        remaining = sub[sub["team"] != pick1["team"]]
        pick2 = remaining.loc[remaining["p_finish_2nd"].idxmax()]
        remaining2 = remaining[remaining["team"] != pick2["team"]]
        pick3q = remaining2.loc[remaining2["p_third_qualifies"].idxmax()]
        summary_rows.append(
            {
                "group": grp,
                "teams": ", ".join(teams),
                "predicted_winner": pick1["team"],
                "p_winner": pick1["p_finish_1st"],
                "predicted_runner_up": pick2["team"],
                "p_runner_up": pick2["p_finish_2nd"],
                "best_third_qualifier_chance": pick3q["team"],
                "p_third_qualifies": pick3q["p_third_qualifies"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    qualifiers = (
        detail.sort_values("p_advance_r32", ascending=False)
        .head(32)
        .reset_index(drop=True)
    )
    qualifiers["qualifier_rank"] = qualifiers.index + 1

    return detail, summary, qualifiers


def build_prediction_sheet_table(summary):
    """Wide table aligned with a typical printable group prediction sheet."""
    return summary[
        [
            "group",
            "predicted_winner",
            "p_winner",
            "predicted_runner_up",
            "p_runner_up",
            "best_third_qualifier_chance",
            "p_third_qualifies",
            "teams",
        ]
    ].rename(
        columns={
            "group": "Group",
            "predicted_winner": "1st (model pick)",
            "p_winner": "P(1st)",
            "predicted_runner_up": "2nd (model pick)",
            "p_runner_up": "P(2nd)",
            "best_third_qualifier_chance": "3rd (best P among top-8 thirds)",
            "p_third_qualifies": "P(3rd qualifies)",
            "teams": "All teams",
        }
    )
