"""Scrape Transfermarkt top-23 squad values and write cutoff CSVs."""

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BACKTEST_CONFIGS, PRODUCTION_CONFIG, SQUAD_DIR
from src.features import load_results, standardize_team_name
from src.squad_features import TM_NATIONAL_TEAMS

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def parse_eur(text):
    s = re.sub(r"[^\d.mkbn]", "", str(text).lower())
    if not s or s == "-":
        return 0.0
    mult = 1.0
    if s.endswith("bn"):
        mult = 1e9
        s = s[:-2]
    elif s.endswith("m"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("k"):
        mult = 1e3
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def fetch_top23_value(slug, tm_id, pause=0.6):
    url = f"https://www.transfermarkt.com/{slug}/startseite/verein/{tm_id}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    vals = [
        parse_eur(td.get_text(strip=True))
        for td in soup.select("table.items td.rechts.hauptlink")
    ]
    vals = sorted([v for v in vals if v > 0], reverse=True)
    time.sleep(pause)
    return sum(vals[:23]), len(vals)


def teams_for_cutoff(cutoff_date):
    df = load_results(ROOT / "general" / "Kaggle_martj42international-football-results-from-1872-to-2017" / "results.csv")
    if not df.columns.isin(["date"]).any():
        from src.config import RESULTS_PATH
        df = load_results(RESULTS_PATH)
    cutoff = pd.to_datetime(cutoff_date)
    window = df[(df["date"] < cutoff) & (df["date"] >= cutoff - pd.Timedelta(days=1825))]
    names = set(window["home_team"].tolist() + window["away_team"].tolist())
    return sorted(standardize_team_name(t) for t in names)


def build_for_teams(teams, snapshot_date):
    rows = []
    for team in teams:
        if team not in TM_NATIONAL_TEAMS:
            continue
        slug, tm_id = TM_NATIONAL_TEAMS[team]
        try:
            value, n_players = fetch_top23_value(slug, tm_id)
            rows.append(
                {
                    "team": team,
                    "squad_value_eur": value,
                    "n_players_scraped": n_players,
                    "snapshot_date": snapshot_date,
                    "source": "transfermarkt_top23",
                }
            )
            print(f"  {team:28} {value/1e6:8.1f}M EUR ({n_players} players)")
        except Exception as exc:
            print(f"  {team:28} SKIP ({exc})")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="2018,2022,2026")
    args = parser.parse_args()

    SQUAD_DIR.mkdir(parents=True, exist_ok=True)
    cutoffs = {
        "2018": BACKTEST_CONFIGS[2018]["target_date"],
        "2022": BACKTEST_CONFIGS[2022]["target_date"],
        "2026": PRODUCTION_CONFIG["target_date"],
    }
    snapshot = str(date.today())
    all_priority = set()
    for label in args.labels.split(","):
        label = label.strip()
        if label in cutoffs:
            teams = teams_for_cutoff(cutoffs[label])
            all_priority.update(t for t in teams if t in TM_NATIONAL_TEAMS)

    print(f"\n=== Scraping {len(all_priority)} national teams (Transfermarkt top-23) ===")
    df = build_for_teams(sorted(all_priority), snapshot)

    for label in args.labels.split(","):
        label = label.strip()
        if label not in cutoffs:
            continue
        out = SQUAD_DIR / f"squad_values_{label}.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows -> {out}")

    # Combined fallback file (2026 participants focus)
    combined = SQUAD_DIR / "squad_values.csv"
    latest = SQUAD_DIR / "squad_values_2026.csv"
    if latest.exists():
        pd.read_csv(latest).to_csv(combined, index=False)
        print(f"Wrote fallback {combined}")


if __name__ == "__main__":
    main()
