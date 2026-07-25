"""Generate Executive_Report.html from outputs/ (open in browser → Print to PDF)."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import BASE, OUTPUTS, SCHEDULE_2026
from src.io_utils import safe_read_csv

TEAM_ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bolivia": "bo", "Bosnia and Herzegovina": "ba", "Brazil": "br", "Canada": "ca",
    "Cape Verde": "cv", "Colombia": "co", "Croatia": "hr", "Curacao": "cw",
    "Czechia": "cz", "DR Congo": "cd", "Ecuador": "ec", "Egypt": "eg",
    "England": "gb-eng", "France": "fr", "Germany": "de", "Ghana": "gh",
    "Haiti": "ht", "Iran": "ir", "Iraq": "iq", "Italy": "it", "Ivory Coast": "ci",
    "Japan": "jp", "Jordan": "jo", "Kosovo": "xk", "Mexico": "mx",
    "Morocco": "ma", "Netherlands": "nl", "New Zealand": "nz", "Norway": "no",
    "Panama": "pa", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa",
    "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn",
    "South Africa": "za", "South Korea": "kr", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "Turkey": "tr", "Tunisia": "tn", "USA": "us", "United States": "us",
    "Uruguay": "uy", "Uzbekistan": "uz",
}


def participating_teams():
    df = safe_read_csv(SCHEDULE_2026)
    skip = ("Winner", "Runner", "Third", "Loser")
    names = set()
    for col in ("team1", "team2"):
        for t in df[col].dropna().astype(str):
            if not any(t.startswith(s) for s in skip):
                names.add("USA" if t == "United States" else t)
    return sorted(names)


def prediction_order_by_group(iter_2026):
    """
    Return [(group, rank_label, team), ...] in model pick order:
    groups A–L, each with 1st → 2nd → Best 3rd → 4th.
    """
    sheet_path = OUTPUTS / f"group_prediction_sheet_2026_iter{iter_2026}.csv"
    if not sheet_path.exists():
        return [(None, "", t) for t in participating_teams()]

    df = safe_read_csv(sheet_path)
    ordered = []
    for _, row in df.sort_values("Group").iterrows():
        grp = str(row["Group"]).strip()
        first = str(row["1st (model pick)"]).strip()
        second = str(row["2nd (model pick)"]).strip()
        third = str(row["3rd (best P among top-8 thirds)"]).strip()
        all_teams = [t.strip() for t in str(row["All teams"]).split(",")]
        picks = {first, second, third}
        fourths = [t for t in all_teams if t not in picks]

        for rank, team in (
            ("1st", first),
            ("2nd", second),
            ("Best 3rd", third),
        ):
            if team:
                ordered.append((grp, rank, team))
        for team in fourths:
            ordered.append((grp, "4th", team))
    return ordered


def flag_collage_html(teams):
    """Alphabetical fallback collage."""
    cells = []
    for team in teams:
        iso = TEAM_ISO.get(team, "un")
        url = f"https://flagcdn.com/w80/{iso}.png"
        cells.append(
            f'<div class="flag-cell">'
            f'<img src="{url}" alt="{team} flag" loading="lazy" />'
            f'<span>{team}</span></div>'
        )
    return "\n".join(cells)


def flag_collage_by_prediction(iter_2026):
    """Flags grouped A–L in model prediction order (1st, 2nd, Best 3rd, 4th)."""
    blocks = []
    current_grp = None
    cells = []

    def flush_group():
        if current_grp is not None and cells:
            blocks.append(
                f'<div class="flag-group">'
                f'<h3 class="flag-group-title">Group {current_grp}</h3>'
                f'<div class="flags">{"".join(cells)}</div></div>'
            )

    for grp, rank, team in prediction_order_by_group(iter_2026):
        if grp != current_grp:
            flush_group()
            cells = []
            current_grp = grp
        iso = TEAM_ISO.get(team, "un")
        url = f"https://flagcdn.com/w80/{iso}.png"
        cells.append(
            f'<div class="flag-cell">'
            f'<img src="{url}" alt="{team} flag" loading="lazy" />'
            f'<span class="flag-rank">{rank}</span>'
            f'<span>{team}</span></div>'
        )
    flush_group()
    return "\n".join(blocks) if blocks else flag_collage_html(participating_teams())


def brier_table_html(summary):
    rows = []
    for year in sorted(summary["year"].unique()):
        sub = summary[summary["year"] == year].sort_values("iteration")
        for _, r in sub.iterrows():
            rows.append(
                f"<tr><td>{int(r['year'])}</td><td>{int(r['iteration'])}</td>"
                f"<td>{r['brier_score']:.4f}</td>"
                f"<td>{r['model_top_pick']}</td>"
                f"<td>{100*r['model_top_pick_pct']:.1f}%</td>"
                f"<td>{r['actual_winner']}</td></tr>"
            )
    return "\n".join(rows)


def group_summary_html(path, n=12):
    if not path.exists():
        return "<p><em>Run <code>run_group_predictions.py</code> to generate group forecasts.</em></p>"
    df = safe_read_csv(path).head(n)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td><strong>{r['group']}</strong></td>"
            f"<td>{r['predicted_winner']}</td>"
            f"<td>{r['predicted_runner_up']}</td>"
            f"<td>{r['best_third_qualifier_chance']}</td></tr>"
        )
    return "\n".join(rows)


def stage_probs_html(path, n=10):
    if not path.exists():
        return "<p><em>Run <code>run_stage_probabilities.py</code> to generate knockout reach probabilities.</em></p>"
    df = safe_read_csv(path).sort_values("model_win_pct", ascending=False).head(n)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td><strong>{r['team']}</strong></td>"
            f"<td>{100*r['p_reach_qf']:.1f}%</td>"
            f"<td>{100*r['p_reach_sf']:.1f}%</td>"
            f"<td>{100*r['p_reach_final']:.1f}%</td>"
            f"<td>{100*r['model_win_pct']:.1f}%</td></tr>"
        )
    return "\n".join(rows)


def comparison_table_html(cmp, n=12):
    cmp = cmp.dropna(subset=["model_win_pct"]).sort_values("model_win_pct", ascending=False)
    rows = []
    for _, r in cmp.head(n).iterrows():
        rank = int(r["opta_rank"]) if pd.notna(r.get("opta_rank")) else "—"
        opta = f"{100*r['opta_win_pct']:.1f}%" if pd.notna(r["opta_win_pct"]) else "—"
        model = f"{100*r['model_win_pct']:.1f}%"
        if pd.notna(r["opta_win_pct"]):
            delta = 100 * (r["model_win_pct"] - r["opta_win_pct"])
            delta_s = f"{delta:+.1f}pp"
        else:
            delta_s = "—"
        rows.append(
            f"<tr><td>{rank}</td><td><strong>{r['team']}</strong></td>"
            f"<td>{opta}</td><td>{model}</td><td>{delta_s}</td></tr>"
        )
    return "\n".join(rows)


def main(iter_2026=5):
    teams = participating_teams()
    summary = safe_read_csv(OUTPUTS / "backtest_summary_all.csv")
    cmp_path = OUTPUTS / f"comparison_2026_iter{iter_2026}.csv"
    cmp = safe_read_csv(cmp_path)

    best_2018 = summary[summary["year"] == 2018].loc[
        summary[summary["year"] == 2018]["brier_score"].idxmin()
    ]
    best_2022 = summary[summary["year"] == 2022].loc[
        summary[summary["year"] == 2022]["brier_score"].idxmin()
    ]

    group_path = OUTPUTS / f"group_summary_2026_iter{iter_2026}.csv"
    stage_path = OUTPUTS / f"tournament_stage_probs_2026_iter{iter_2026}.csv"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FIFA World Cup 2026 — Tournament Forecast Report</title>
  <style>
    :root {{
      --ink: #1a1a2e;
      --muted: #4a5568;
      --accent: #0f4c5c;
      --gold: #bc6c25;
      --bg: #faf9f7;
      --card: #ffffff;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background: var(--bg);
      margin: 0;
      line-height: 1.5;
    }}
    .page {{
      max-width: 900px;
      margin: 0 auto;
      padding: 48px 40px;
      background: var(--card);
      min-height: 100vh;
    }}
    @media print {{
      body {{ background: white; }}
      .page {{ padding: 24px; max-width: 100%; }}
      .no-print {{ display: none; }}
    }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0 0 8px;
      color: var(--accent);
      letter-spacing: -0.02em;
    }}
    .subtitle {{ color: var(--muted); font-size: 1rem; margin-bottom: 32px; }}
    h2 {{
      font-size: 1.1rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
      border-bottom: 2px solid var(--gold);
      padding-bottom: 6px;
      margin: 36px 0 16px;
    }}
    .exec-box {{
      background: #f0f7f9;
      border-left: 4px solid var(--accent);
      padding: 20px 24px;
      margin: 24px 0;
      font-size: 0.95rem;
    }}
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin: 24px 0;
    }}
    .kpi {{
      border: 1px solid var(--border);
      padding: 16px;
      text-align: center;
    }}
    .kpi .val {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
    .kpi .lbl {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }}
    .flag-groups {{
      margin: 20px 0;
    }}
    .flag-group {{
      margin-bottom: 18px;
    }}
    .flag-group-title {{
      font-size: 0.85rem;
      color: var(--accent);
      margin: 0 0 8px;
      letter-spacing: 0.04em;
    }}
    .flags {{
      display: grid;
      grid-template-columns: repeat(4, minmax(88px, 1fr));
      gap: 12px;
    }}
    .flag-cell {{
      text-align: center;
      font-size: 0.65rem;
      color: var(--muted);
    }}
    .flag-rank {{
      display: block;
      font-size: 0.58rem;
      font-weight: 600;
      color: var(--gold);
      text-transform: uppercase;
      margin-bottom: 2px;
    }}
    .flag-cell img {{
      width: 48px;
      height: 32px;
      object-fit: cover;
      border: 1px solid var(--border);
      border-radius: 2px;
      display: block;
      margin: 0 auto 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin: 12px 0;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
    }}
    th {{ background: #edf2f7; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .footnote {{
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 40px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
    }}
    ul {{ margin: 8px 0; padding-left: 20px; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <div class="page">
    <p class="no-print" style="background:#fff3cd;padding:10px;font-size:0.85rem;">
      Open in Chrome/Edge → <strong>Ctrl+P</strong> → Save as PDF for portfolio or sharing.
      To paste into PowerPoint: Insert → Pictures → this PDF, or copy sections manually.
    </p>

    <h1>FIFA World Cup 2026 Predictor</h1>
    <p class="subtitle">Tournament forecast report · Academic analytics project · Generated {date.today().isoformat()}</p>

    <div class="exec-box">
      <strong>Summary.</strong> A walk-forward Poisson/Dixon-Coles model estimates international
      A-team goal rates from a five-year rolling window of match history (recent matches
      upweighted; club football excluded). Match-level probabilities are validated on held-out
      2018 and 2022 World Cup finals (Brier score). Because the 2026 format links group
      standings to a 32-team knockout bracket — including best-third-place slot assignment —
      tournament-stage outcomes are estimated with Monte Carlo simulation (10,000 runs).
      This report presents iteration&nbsp;{iter_2026} forecasts against the FIFA official
      48-team schedule; external win-probability benchmarks appear later for comparison only.
    </div>

    <div class="kpi-row">
      <div class="kpi">
        <div class="val">{len(teams)}</div>
        <div class="lbl">Teams (2026 schedule)</div>
      </div>
      <div class="kpi">
        <div class="val">{best_2018['brier_score']:.3f}</div>
        <div class="lbl">Best 2018 Brier (iter {int(best_2018['iteration'])})</div>
      </div>
      <div class="kpi">
        <div class="val">{best_2022['brier_score']:.3f}</div>
        <div class="lbl">Best 2022 Brier (iter {int(best_2022['iteration'])})</div>
      </div>
    </div>

    <h2>Participating nations — model prediction order</h2>
    <p style="font-size:0.9rem;color:var(--muted);">
      48 teams arranged by group (A–L), in model pick order: 1st, 2nd, Best 3rd, then 4th.
      Flags via <a href="https://flagcdn.com">flagcdn.com</a> — not official FIFA branding.
    </p>
    <div class="flag-groups">
      {flag_collage_by_prediction(iter_2026)}
    </div>

    <h2>Methodology</h2>
    <ul>
      <li><strong>Training data:</strong> Kaggle martj42 international A-team results; five-year window before each cutoff; 2-year half-life decay.</li>
      <li><strong>Fixtures:</strong> FIFA.com official 2026 schedule (local CSV), with playoff slots resolved from UEFA/FIFA results.</li>
      <li><strong>Model:</strong> Log-linear Poisson goals with tier weights, Elo anchor, Dixon-Coles τ, opponent difficulty, and competition-context weights (iterations 3–7); optional squad-value prior blend (iteration 8 prototype).</li>
      <li><strong>Validation:</strong> Walk-forward — tune on 2018 World Cup, confirm on 2022; 64 finals matches held out per year.</li>
      <li><strong>DataCamp:</strong> competition submission format only (match_id, schema) — not used as fixture authority.</li>
    </ul>

    <h2>Why Monte Carlo simulation?</h2>
    <p style="font-size:0.92rem;">
      Dixon-Coles Poisson yields <strong>match-level</strong> score probabilities, not tournament-stage
      outcomes directly. The 2026 format adds twelve groups, eight best-third-place qualifiers mapped
      to fixed knockout slots, and bracket dependencies that have no closed-form winner probability.
      We simulate many independent tournaments (10,000), sample scores from the model, apply FIFA
      tiebreak rules, and aggregate P(win), P(reach each stage), and group-advancement rates.
    </p>

    <h2>Brier score &amp; Dixon-Coles</h2>
    <p style="font-size:0.92rem;">
      <strong>Brier score.</strong> For each World Cup match we predict three probabilities
      (home win, draw, away win). The Brier score is the mean squared error between those
      probabilities and the actual result (expressed as a one-hot vector). Lower is better:
      0 is perfect; about <strong>0.67</strong> is the benchmark from naively guessing one-third
      for every outcome. We average over all 64 held-out tournament matches to compare
      iterations and years. Iteration choice for development uses the lowest 2018 Brier;
      2022 is reported as confirmation without re-tuning.
    </p>
    <p style="font-size:0.92rem;">
      <strong>Dixon-Coles.</strong> Classical Poisson models assume home and away goals are
      independent; real football shows extra correlation on low scores (0–0, 1–0, 0–1, 1–1),
      especially draws. Dixon and Coles (1997) multiply independent Poisson probabilities by
      a correction factor τ (rho, ρ) on those scorelines. Our pipeline learns team attack and
      defense rates via Poisson negative log-likelihood (iterations 1–4); <strong>iteration 5</strong>
      adds the τ term in training and when converting goal rates to win/draw/loss probabilities.
      World Cup match scoring uses neutral venues (no home advantage). Elo anchoring (iter 4+)
      shrinks team strengths toward a stable rating so single shock results do not dominate.
    </p>

    <h2>Backtest — Brier score (lower is better)</h2>
    <table>
      <thead>
        <tr><th>Year</th><th>Iter</th><th>Brier</th><th>Model favourite</th><th>Win %</th><th>Actual winner</th></tr>
      </thead>
      <tbody>
        {brier_table_html(summary)}
      </tbody>
    </table>

    <h2>External benchmark — OPTA pre-tournament win % (iteration {iter_2026})</h2>
    <p style="font-size:0.88rem;color:var(--muted);margin:0 0 8px;">
      Post-hoc comparison only; OPTA probabilities are not training inputs.
    </p>
    <table>
      <thead>
        <tr><th>OPTA rank</th><th>Team</th><th>OPTA win %</th><th>Model win %</th><th>Delta</th></tr>
      </thead>
      <tbody>
        {comparison_table_html(cmp)}
      </tbody>
    </table>

    <h2>Group-stage outlook (12 groups → round of 32)</h2>
    <p style="font-size:0.9rem;color:var(--muted);">
      Model picks for group winner, runner-up, and best third-place qualifier candidate.
      48-team format: 24 auto-qualifiers + 8 best third-place teams.
    </p>
    <table>
      <thead>
        <tr><th>Group</th><th>1st</th><th>2nd</th><th>Best 3rd</th></tr>
      </thead>
      <tbody>
        {group_summary_html(group_path)}
      </tbody>
    </table>

    <h2>Knockout reach probabilities</h2>
    <p style="font-size:0.9rem;color:var(--muted);">
      Monte Carlo estimates of advancing through each knockout round (iteration {iter_2026}).
    </p>
    <table>
      <thead>
        <tr><th>Team</th><th>P(QF)</th><th>P(SF)</th><th>P(Final)</th><th>P(Win)</th></tr>
      </thead>
      <tbody>
        {stage_probs_html(stage_path)}
      </tbody>
    </table>

    <h2>Key findings &amp; limitations</h2>
    <ul>
      <li>Elo anchoring (iter 4+) limits overreaction to isolated shock results in training.</li>
      <li>Dixon-Coles τ (iter 5+) improves low-score and draw calibration on held-out World Cup matches.</li>
      <li>Iteration 8 adds Transfermarkt squad-value priors; modest Brier improvement on 2018/2022 — retained for sensitivity analysis.</li>
      <li>No xG or injury data; neutral venues at World Cup inference; live updates require re-running with a new cutoff.</li>
    </ul>

    <p class="footnote">
      Independent research project by <span style="font-size:0.85em">Chandra Sekhar</span>. Not affiliated with FIFA, OPTA, or any cohort program.
      Do not use FIFA copyrighted logos without permission. Team flags from flagcdn.com.
      Project: fifa_2026_predictor · Outputs: outputs/ · Refresh: python generate_executive_report.py
    </p>
  </div>
</body>
</html>
"""

    for out in (BASE / "Executive_Report.html", BASE / "web" / "Executive_Report.html"):
        out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    print("Open in browser -> Ctrl+P -> Save as PDF")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, default=8)
    args = p.parse_args()
    main(iter_2026=args.iteration)
