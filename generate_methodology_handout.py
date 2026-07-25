"""Generate academic methodology handout (HTML + PDF)."""

from datetime import date
from pathlib import Path

from src.config import BACKTEST_CONFIGS, HORIZON_DAYS, PRODUCTION_CONFIG, RESULTS_PATH
from src.features import load_and_weight_data

BASE = Path(__file__).resolve().parent
OUT_HTML = BASE / "Methodology_Handout.html"
OUT_PDF = BASE / "Methodology_Handout.pdf"
OUT_WEB = BASE / "web" / "Methodology_Handout.html"

CUTS = [
    ("2018 development", BACKTEST_CONFIGS[2018]["target_date"]),
    ("2022 confirmation", BACKTEST_CONFIGS[2022]["target_date"]),
    ("2026 production", PRODUCTION_CONFIG["target_date"]),
]


def training_rows():
    rows = []
    for label, target_date in CUTS:
        df, _, _, _, _, _ = load_and_weight_data(RESULTS_PATH, target_date, iteration=7)
        rows.append(
            {
                "label": label,
                "cutoff": target_date,
                "matches": f"{len(df):,}",
                "from": str(df["date"].min().date()),
                "to": str(df["date"].max().date()),
            }
        )
    return rows


def build_html(rows):
    table_rows = "\n".join(
        f"<tr><td>{r['label']}</td><td>{r['cutoff']}</td><td>{r['matches']}</td>"
        f"<td>{r['from']}</td><td>{r['to']}</td></tr>"
        for r in rows
    )
    today = date.today().strftime("%B %d, %Y")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FIFA 2026 Predictor — Training Data & Methodology</title>
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
      max-width: 820px;
      margin: 0 auto;
      padding: 40px 36px;
      background: var(--card);
    }}
    @media print {{
      body {{ background: white; }}
      .page {{ padding: 20px; max-width: 100%; }}
      .no-print {{ display: none; }}
    }}
    h1 {{
      font-size: 1.55rem;
      font-weight: 700;
      margin: 0 0 6px;
      color: var(--accent);
    }}
    .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 24px; }}
    h2 {{
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
      border-bottom: 2px solid var(--gold);
      padding-bottom: 4px;
      margin: 28px 0 12px;
    }}
    .exec-box {{
      background: #f0f7f9;
      border-left: 4px solid var(--accent);
      padding: 16px 20px;
      margin: 16px 0;
      font-size: 0.92rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin: 10px 0;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 7px 9px;
      text-align: left;
    }}
    th {{ background: #edf2f7; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    ul {{ margin: 8px 0; padding-left: 1.2rem; }}
    li {{ margin: 4px 0; font-size: 0.9rem; }}
    .compare {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 12px 0;
    }}
    .compare div {{
      border: 1px solid var(--border);
      padding: 12px;
      font-size: 0.85rem;
    }}
    .compare h3 {{
      margin: 0 0 8px;
      font-size: 0.8rem;
      text-transform: uppercase;
      color: var(--gold);
    }}
    .footnote {{
      font-size: 0.72rem;
      color: var(--muted);
      margin-top: 28px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }}
    .no-print {{
      background: #fff8e6;
      border: 1px solid #f0d78c;
      padding: 10px 14px;
      font-size: 0.85rem;
      margin-bottom: 20px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <p class="no-print">Tip: Chrome/Edge → <strong>Ctrl+P</strong> → Save as PDF</p>
    <h1>FIFA World Cup 2026 Predictor</h1>
    <p class="subtitle">Academic methodology brief — training data &amp; methods (for methodological review)<br/>
    Author: Chandra Sekhar · Model: Iteration 7 · Generated {today}</p>

    <div class="exec-box">
      <strong>Summary.</strong> We train on <em>international A-team matches only</em> (country vs country),
      using a <strong>5-year rolling window</strong> before each World Cup cutoff. Recent matches are
      <strong>upweighted</strong> (2-year half-life decay), not excluded. The model does <strong>not</strong>
      use club football (Premier League, Champions League, etc.).
    </div>

    <h2>Data sources</h2>
    <ul>
      <li><strong>Historical training:</strong> Kaggle martj42 international results (<code>results.csv</code>) — A-team match outcomes from 1872 onward.</li>
      <li><strong>2026 fixtures:</strong> FIFA.com official schedule/PDF (<code>2026_FIFA_World_Cup_Schedule.csv.csv</code> locally), with playoff slots resolved from UEFA/FIFA playoff results.</li>
      <li><strong>DataCamp files:</strong> used only for competition <em>submission format</em> (match_id alignment, output schema, placeholder team labels for grading) — not authoritative for kickoff times, qualification, or real-world truth.</li>
      <li><strong>Secondary checks (optional):</strong> OpenFootball or TheStatsAPI as machine-readable cross-checks against the FIFA schedule.</li>
    </ul>

    <h2>What matches are included?</h2>
    <p style="font-size:0.9rem;margin-top:0;">
      Training rows come from the Kaggle martj42 dataset above, extended through 2026.
      Teams are <strong>nations</strong> (France, England, Brazil) — never clubs.
    </p>
    <ul>
      <li><strong>Included:</strong> World Cup &amp; continental qualifiers, continental finals (Euro, Copa, AFCON, Asian Cup),
          Nations League, Gold Cup, friendlies, and regional A-team cups — all confederations (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC).</li>
      <li><strong>Excluded:</strong> Domestic leagues, UEFA Champions League/Europa League, any club match; matches on/after the forecast cutoff; matches older than 5 years.</li>
      <li><strong>Validation:</strong> Walk-forward — tune on 2018 World Cup (cutoff opening day), confirm on 2022; 64 finals matches held out per year for Brier scoring.</li>
    </ul>

    <h2>Training windows (verified counts)</h2>
    <table>
      <tr><th>Run</th><th>Cutoff</th><th>Matches</th><th>From</th><th>To</th></tr>
      {table_rows}
    </table>
    <p style="font-size:0.82rem;color:var(--muted);margin:6px 0 0;">
      Each run uses only the last {HORIZON_DAYS} days (~5 years) with <code>date &lt; cutoff</code>.
      The full archive spans 1872–2026 (~49k matches); training never uses the entire history.
    </p>

    <h2>Club vs national team — common clarification</h2>
    <div class="compare">
      <div>
        <h3>Not in our data</h3>
        PSG → France, Arsenal → England (club → nation mapping)<br/>
        Premier League, Ligue 1, Champions League<br/>
        Club form / domestic league progression
      </div>
      <div>
        <h3>What we actually use</h3>
        France vs Brazil, England vs Germany (A-team fixtures)<br/>
        UEFA = <em>national-team</em> Euro, qualifiers, Nations League<br/>
        Strength learned from weighted international results + Elo
      </div>
    </div>

    <h2>How matches are weighted (Iteration 7)</h2>
    <table>
      <tr><th>Component</th><th>Setting</th><th>Meaning</th></tr>
      <tr><td>Time decay</td><td>2-year half-life</td><td>Matches in the last 2 years get the highest weight; older matches within the 5-year window count less.</td></tr>
      <tr><td>Tournament tier</td><td>A=1.00, B=0.85, C=0.50</td><td>WC/qualifiers/finals (A); Nations League, Gold Cup (B); friendlies (C).</td></tr>
      <tr><td>Opponent difficulty</td><td>D<sub>opp</sub></td><td>Discounts goals vs weak opponents (walk-forward Elo).</td></tr>
      <tr><td>Competition context</td><td>C<sub>conf</sub> (Opta-inspired)</td><td>FIFA 1.05, UEFA 1.00, CONMEBOL 0.95, CAF 0.88, AFC 0.80, CONCACAF 0.78 — discounts weaker competition <em>contexts</em>, not teams.</td></tr>
    </table>
    <p style="font-size:0.85rem;margin:8px 0 0;">
      Final sample weight = time × tournament tier × D<sub>opp</sub> × C<sub>conf</sub>.
      Model: Dixon-Coles Poisson with Elo anchor; neutral venues at World Cup inference.
    </p>

    <h2>Why Monte Carlo simulation?</h2>
    <p style="font-size:0.9rem;margin-top:0;">
      Dixon-Coles Poisson yields <strong>match-level</strong> score probabilities — not tournament-stage
      outcomes directly. The 2026 format adds structural complexity: 12 groups of four, eight
      best-third-place qualifiers assigned to predetermined knockout slots, and a 32-team bracket
      where later opponents depend on earlier group results.
    </p>
    <ul>
      <li><strong>Bracket dependencies:</strong> Knockout pairings follow group standings and best-third tiebreak rules; a closed-form tournament-winner probability is not available.</li>
      <li><strong>Quantities estimated:</strong> P(win), P(reach semifinals), group advancement rates, and uncertainty in best-third qualification.</li>
      <li><strong>Method:</strong> Many independent simulated tournaments (10,000 in production) sample match scores from the Poisson model, apply FIFA tiebreak rules, and aggregate stage outcomes.</li>
    </ul>

    <h2>Opta-inspired weights vs OPTA benchmark</h2>
    <div class="compare">
      <div>
        <h3>Opta-inspired (in training)</h3>
        C<sub>conf</sub> multipliers on each <em>match</em> by tournament confederation context.<br/>
        Tuned on 2018 Brier. Does <strong>not</strong> import OPTA team ratings.
      </div>
      <div>
        <h3>OPTA CSV (benchmark only)</h3>
        Pre-tournament win % from OPTA used only for post-hoc comparison tables — never a training feature.
      </div>
    </div>

    <h2>Backtest performance (Brier — lower is better)</h2>
    <table>
      <tr><th>Year</th><th>Iteration 7 Brier</th><th>Role</th></tr>
      <tr><td>2018</td><td>0.739</td><td>Development / hyperparameter tuning</td></tr>
      <tr><td>2022</td><td>0.726</td><td>Holdout confirmation</td></tr>
    </table>

    <h2>Iteration 8 (prototype)</h2>
    <p style="font-size:0.9rem;margin-top:0;">
      An exploratory branch blends Transfermarkt squad-value priors with walk-forward Elo before Poisson training.
      Production forecasts remain on iteration 7; iteration 8 is retained for sensitivity analysis only.
    </p>

    <p class="footnote">
      Project: fifa_2026_predictor · Training data: {RESULTS_PATH.name} · Fixtures: FIFA official schedule ·
      Live app: fifa2026-chandra.netlify.app ·
      For full predictions see Executive_Report.html.
    </p>
  </div>
</body>
</html>"""


def build_pdf(rows):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(14, 14, 14)
    usable = pdf.w - pdf.l_margin - pdf.r_margin

    def writeln(text, size=9, bold=False, color=(26, 26, 46), h=5):
        style = "B" if bold else ""
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable, h, text)
        pdf.ln(1)

    pdf.add_page()
    writeln("FIFA World Cup 2026 Predictor", size=16, bold=True, color=(15, 76, 92), h=8)
    writeln("Academic Methodology Brief - Training Data & Methods", size=10, color=(74, 85, 104))
    writeln(f"Author: Chandra Sekhar | Model: Iteration 7 | {date.today()}", size=10, color=(74, 85, 104))
    pdf.ln(2)

    pdf.set_fill_color(240, 247, 249)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(26, 26, 46)
    pdf.multi_cell(
        usable,
        5,
        (
            "Summary. We train on international A-team matches only (country vs country), "
            "using a 5-year rolling window before each World Cup cutoff. Recent matches are "
            "upweighted (2-year half-life decay), not excluded. The model does not use club "
            "football (Premier League, Champions League, etc.)."
        ),
        fill=True,
    )
    pdf.ln(3)

    def section(title):
        writeln(title, size=11, bold=True, color=(15, 76, 92), h=7)
        y = pdf.get_y()
        pdf.set_draw_color(188, 108, 37)
        pdf.line(pdf.l_margin, y, pdf.l_margin + usable, y)
        pdf.ln(3)

    section("Data sources")
    for b in [
        "Historical training: Kaggle martj42 international results (results.csv).",
        "2026 fixtures: FIFA.com official schedule/PDF; playoff slots from UEFA/FIFA results.",
        "DataCamp files: competition submission format only (match_id, schema, placeholder labels) - not schedule authority.",
        "Optional cross-checks: OpenFootball or TheStatsAPI vs FIFA schedule.",
    ]:
        writeln(f"- {b}")

    section("What matches are included?")
    for b in [
        "Training rows: Kaggle martj42 results.csv. Teams are nations, never clubs.",
        "Included: WC/continental qualifiers and finals, Nations League, Gold Cup, friendlies, all confederations.",
        "Excluded: Domestic leagues, UEFA club competitions, matches on/after cutoff, matches older than 5 years.",
        "Validation: Walk-forward 2018 (tune) and 2022 (holdout); 64 WC finals held out per year.",
    ]:
        writeln(f"- {b}")

    section("Training windows (verified counts)")
    col_w = [40, 26, 20, 30, 30]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_x(pdf.l_margin)
    for i, h in enumerate(["Run", "Cutoff", "Matches", "From", "To"]):
        pdf.cell(col_w[i], 7, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in rows:
        pdf.set_x(pdf.l_margin)
        for i, v in enumerate([r["label"], r["cutoff"], r["matches"], r["from"], r["to"]]):
            pdf.cell(col_w[i], 7, v, border=1)
        pdf.ln()
    pdf.ln(2)

    section("Club vs national team")
    writeln(
        "NOT in data: PSG/Arsenal club matches, Premier League, Champions League, club-to-nation mapping."
    )
    writeln(
        "IN data: France vs Brazil, England vs Germany. UEFA in code means national-team Euro, qualifiers, Nations League."
    )

    section("Match weighting (Iteration 7)")
    wcols = [42, 34, usable - 76]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_x(pdf.l_margin)
    for i, h in enumerate(["Component", "Setting", "Meaning"]):
        pdf.cell(wcols[i], 7, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for row in [
        ("Time decay", "2-year half-life", "Recent matches weighted highest within 5-year window"),
        ("Tournament tier", "A=1.0 B=0.85 C=0.5", "Qualifiers/finals vs Nations League vs friendlies"),
        ("Opponent difficulty", "D_opp", "Discount goals vs weak opponents"),
        ("Competition context", "C_conf (Opta-inspired)", "FIFA 1.05, UEFA 1.0, CAF 0.88, AFC 0.80"),
    ]:
        pdf.set_x(pdf.l_margin)
        x0 = pdf.l_margin
        pdf.cell(wcols[0], 7, row[0], border=1)
        pdf.cell(wcols[1], 7, row[1], border=1)
        y0 = pdf.get_y()
        pdf.set_xy(x0 + wcols[0] + wcols[1], y0)
        pdf.multi_cell(wcols[2], 7, row[2], border=1)
        pdf.set_y(y0 + 7)
    pdf.ln(2)

    section("Why Monte Carlo simulation?")
    writeln(
        "Dixon-Coles Poisson yields match-level score probabilities, not tournament-stage outcomes. "
        "The 2026 format: 12 groups of four, eight best-third qualifiers into predetermined knockout slots, "
        "and a 32-team bracket with slot dependencies on group results."
    )
    for b in [
        "Bracket dependencies: knockout pairings follow group standings and best-third tiebreaks; no closed-form winner probability.",
        "Quantities estimated: P(win), P(reach semifinals), group advancement, best-third qualification uncertainty.",
        "Method: 10,000 independent simulated tournaments sample Poisson scores, apply FIFA tiebreak rules, aggregate outcomes.",
    ]:
        writeln(f"- {b}")

    section("Opta-inspired weights vs OPTA benchmark")
    writeln("IN TRAINING: C_conf multipliers on each match by tournament confederation context (Opta-inspired index).")
    writeln("BENCHMARK ONLY: OPTA pre-tournament win % from CSV - never fed into training.")

    section("Backtest (Brier - lower is better)")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(24, 7, "2018", border=1)
    pdf.cell(24, 7, "0.739", border=1)
    pdf.cell(usable - 48, 7, "Development / tuning", border=1)
    pdf.ln()
    pdf.set_x(pdf.l_margin)
    pdf.cell(24, 7, "2022", border=1)
    pdf.cell(24, 7, "0.726", border=1)
    pdf.cell(usable - 48, 7, "Holdout confirmation", border=1)
    pdf.ln(4)

    section("Iteration 8 (prototype)")
    writeln(
        "Exploratory branch: Transfermarkt squad-value priors blended with walk-forward Elo. "
        "Production remains iteration 7; iteration 8 kept for sensitivity analysis only."
    )
    pdf.ln(2)

    writeln(
        "Project: fifa_2026_predictor | Training: results.csv | Fixtures: FIFA official schedule | "
        "Live: fifa2026-chandra.netlify.app | See Executive_Report.html for full predictions.",
        size=7,
        color=(74, 85, 104),
        h=4,
    )
    pdf.output(str(OUT_PDF))


def main():
    rows = training_rows()
    html = build_html(rows)
    OUT_HTML.write_text(html, encoding="utf-8")
    OUT_WEB.parent.mkdir(parents=True, exist_ok=True)
    OUT_WEB.write_text(html, encoding="utf-8")

    try:
        build_pdf(rows)
        print(f"Wrote {OUT_PDF}")
    except ImportError:
        print("fpdf2 not installed; HTML only. Run: pip install fpdf2")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {OUT_WEB}")


if __name__ == "__main__":
    main()
