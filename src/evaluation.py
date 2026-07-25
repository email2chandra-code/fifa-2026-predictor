import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BACKTEST_CONFIGS, OPTA_PATH, OPTA_YEAR_COLUMNS, OUTPUTS, TEAM_ALIASES
from .dixon_coles import _poisson_pmf, dixon_coles_outcome_probs
from .features import standardize_team_name
from .io_utils import safe_read_csv, write_csv

OPTA_TEAM_ALIASES = {
    **TEAM_ALIASES,
    "United States": "USA",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
}


def poisson_outcome_probs(lambda_home, lambda_away, max_goals=10):
    """Return (p_home, p_draw, p_away) from independent Poisson scorelines."""
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            probs[h, a] = _poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)

    p_home = float(probs[np.triu_indices(max_goals + 1, k=1)].sum())
    p_draw = float(np.trace(probs))
    p_away = float(probs[np.tril_indices(max_goals + 1, k=-1)].sum())

    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    return p_home, p_draw, p_away


def actual_outcome_onehot(home_score, away_score):
    if home_score > away_score:
        return np.array([1.0, 0.0, 0.0])
    if home_score < away_score:
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def brier_score(prob_rows, actual_rows):
    """
    Mean Brier score over matches for 3-way outcomes (home / draw / away).

    prob_rows: list of (p_home, p_draw, p_away)
    actual_rows: list of (home_score, away_score)
    """
    scores = []
    for (ph, pd_, pa), (hs, as_) in zip(prob_rows, actual_rows):
        pred = np.array([ph, pd_, pa])
        actual = actual_outcome_onehot(hs, as_)
        scores.append(float(np.sum((pred - actual) ** 2)))
    return float(np.mean(scores)) if scores else math.nan


def match_outcome_probs(lambda_home, lambda_away, iteration=4, rho=-0.03):
    """3-way outcome probabilities; Dixon-Coles tau for iteration 5+."""
    if iteration >= 5:
        return dixon_coles_outcome_probs(lambda_home, lambda_away, rho=float(rho))
    return poisson_outcome_probs(lambda_home, lambda_away)


def predict_world_cup_matches(model, wc_matches, team_to_id, iteration=4):
    """Score every World Cup match at neutral venues (no home advantage)."""
    import torch

    model.eval()
    rows = []
    rho = float(model.rho.item()) if iteration >= 5 else -0.03

    with torch.no_grad():
        for _, match in wc_matches.iterrows():
            t1, t2 = match["team1"], match["team2"]
            h_id = team_to_id.get(t1)
            a_id = team_to_id.get(t2)

            if h_id is None or a_id is None:
                rows.append(
                    {
                        "date": match["date"],
                        "team1": t1,
                        "team2": t2,
                        "p_home": np.nan,
                        "p_draw": np.nan,
                        "p_away": np.nan,
                        "skipped": True,
                    }
                )
                continue

            lambdas = model(
                torch.tensor([h_id], dtype=torch.long),
                torch.tensor([a_id], dtype=torch.long),
                iteration=iteration,
                neutral=True,
            )
            lh, la = lambdas[0, 0].item(), lambdas[0, 1].item()
            ph, pd_, pa = match_outcome_probs(lh, la, iteration=iteration, rho=rho)

            rows.append(
                {
                    "date": match["date"],
                    "team1": t1,
                    "team2": t2,
                    "team1_score": match["team1_score"],
                    "team2_score": match["team2_score"],
                    "lambda_team1": lh,
                    "lambda_team2": la,
                    "p_home": ph,
                    "p_draw": pd_,
                    "p_away": pa,
                    "skipped": False,
                }
            )

    return pd.DataFrame(rows)


def extract_team_metrics(model, id_to_team, elo_ratings=None):
    attack = model.attack_embeddings.weight.detach().numpy().flatten()
    defense = model.defense_embeddings.weight.detach().numpy().flatten()
    metrics = pd.DataFrame(
        {
            "Team_ID": list(id_to_team.keys()),
            "Team_Name": list(id_to_team.values()),
            "Attacking_Strength": attack,
            "Defensive_Weakness": defense,
        }
    )
    if elo_ratings is not None:
        metrics["Elo_Rating"] = metrics["Team_Name"].map(elo_ratings)
    return metrics.sort_values("Attacking_Strength", ascending=False)


def normalize_opta_team(name):
    """Strip emoji/flags and align OPTA team labels with results.csv names."""
    if pd.isna(name):
        return None
    cleaned = str(name).strip()
    if cleaned in ("—", "(Field of 32 Max)", "nan", ""):
        return None
    # Remove non-Latin symbols (flags, etc.) while keeping letters and spaces
    cleaned = "".join(ch for ch in cleaned if ch.isascii() and (ch.isalnum() or ch.isspace()))
    cleaned = " ".join(cleaned.split())
    return OPTA_TEAM_ALIASES.get(cleaned, standardize_team_name(cleaned))


def _parse_opta_pct(value):
    text = str(value).strip().replace("%", "")
    if text in ("—", "", "nan"):
        return np.nan
    return float(text) / 100.0


def load_opta_win_probs(year=2022):
    """Parse OPTA pre-tournament win probabilities for 2018, 2022, or 2026."""
    columns = ["team", "opta_win_pct", "opta_rank"]
    if not OPTA_PATH.exists():
        return pd.DataFrame(columns=columns)

    if year not in OPTA_YEAR_COLUMNS:
        raise ValueError(f"year must be one of {list(OPTA_YEAR_COLUMNS)}; got {year}")

    team_col, pct_col = OPTA_YEAR_COLUMNS[year]
    raw = safe_read_csv(OPTA_PATH, header=None, skiprows=2)

    df = raw.iloc[:, [0, team_col, pct_col]].copy()
    df.columns = ["opta_rank", "team_raw", "opta_win_pct_raw"]
    df["team"] = df["team_raw"].apply(normalize_opta_team)
    df["opta_win_pct"] = df["opta_win_pct_raw"].apply(_parse_opta_pct)
    df = df.dropna(subset=["team"]).copy()
    df["opta_rank"] = pd.to_numeric(df["opta_rank"], errors="coerce")
    return df[columns].reset_index(drop=True)


def build_comparison_table(year, model_win_probs=None, iteration=None):
    """
    Build OPTA vs Model vs Actual table for a tournament year.

    model_win_probs: optional DataFrame or dict with keys team -> win probability
                     (from Monte Carlo once simulator is wired; None leaves blank).
    """
    if year not in BACKTEST_CONFIGS and year != 2026:
        raise ValueError(f"No comparison config for year {year}")

    opta = load_opta_win_probs(year)
    actual_winner = BACKTEST_CONFIGS.get(year, {}).get("actual_winner")

    table = opta.copy()
    table["year"] = year
    if iteration is not None:
        table["iteration"] = iteration

    if model_win_probs is None:
        table["model_win_pct"] = np.nan
    elif isinstance(model_win_probs, dict):
        table["model_win_pct"] = table["team"].map(model_win_probs)
    else:
        mdf = model_win_probs.copy()
        if "team" not in mdf.columns:
            mdf = mdf.rename(columns={"Team_Name": "team"})
        table = table.merge(mdf[["team", "model_win_pct"]], on="team", how="left")

    table["actual_winner"] = table["team"] == actual_winner if actual_winner else False
    table["delta_model_minus_opta"] = table["model_win_pct"] - table["opta_win_pct"]
    return table


def save_comparison_table(year, iteration=None, model_win_probs=None, outputs_dir=OUTPUTS):
    """Write OPTA vs model comparison CSV for presentation."""
    table = build_comparison_table(year, model_win_probs=model_win_probs, iteration=iteration)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_iter{iteration}" if iteration is not None else ""
    out_path = outputs_dir / f"comparison_{year}{suffix}.csv"
    write_csv(table, out_path)
    return out_path


def summarize_backtest(predictions_df, year, iteration, brier):
    """Build a one-row summary for a single year × iteration run."""
    cfg = BACKTEST_CONFIGS[year]
    scored = predictions_df[~predictions_df["skipped"]].copy()
    return {
        "year": year,
        "iteration": iteration,
        "role": cfg["role"],
        "target_date": cfg["target_date"],
        "brier_score": brier,
        "matches_scored": len(scored),
        "actual_winner": cfg["actual_winner"],
    }


def save_backtest_outputs(predictions_df, year, iteration, outputs_dir=OUTPUTS):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    pred_path = outputs_dir / f"backtest_{year}_iter{iteration}.csv"
    write_csv(predictions_df, pred_path)
    return pred_path
