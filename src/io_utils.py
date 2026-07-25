"""Safe CSV loading — avoids numpy/pandas copyto issues in some Jupyter kernels."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def safe_read_csv(path, **kwargs):
    """
    Read CSV with pandas; fall back to engine='python' if numpy copyto fails.

    The copyto AttributeError usually means a stale or mixed numpy install in
    a long-running Jupyter kernel — restart kernel after fixing the env.
    """
    path = Path(path)
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except AttributeError as exc:
        if "copyto" not in str(exc):
            raise
        return pd.read_csv(path, engine="python", **kwargs)


def write_csv(df, path, **kwargs):
    """Write CSV; raise a clear error if the file is locked (Excel, VS Code, OneDrive)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, **kwargs)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write '{path}'. Close it if open in Excel, VS Code, or "
            f"presentation.ipynb, then retry. OneDrive sync can also lock files."
        ) from exc


def resolve_output_file(outputs_dir, pattern, preferred_suffix):
    """
    Resolve an outputs file when the preferred iteration may not exist.

    Example: preferred comparison_2026_iter1.csv missing, but iter4 exists
    after production was only run for iteration 4.
    """
    outputs_dir = Path(outputs_dir)
    preferred = outputs_dir / pattern.format(preferred_suffix)
    if preferred.exists():
        return preferred, preferred_suffix

    stem = pattern.split("{")[0]
    tail = pattern.split("}")[-1]
    glob_pattern = f"{stem}*{tail}"
    def _iter_num(path):
        stem = path.stem
        if "_iter" in stem:
            return int(stem.rsplit("_iter", 1)[-1])
        return 0

    matches = sorted(outputs_dir.glob(glob_pattern), key=_iter_num)
    if not matches:
        raise FileNotFoundError(
            f"No file matching {outputs_dir / glob_pattern} "
            f"(preferred: {preferred.name})"
        )

    chosen = matches[-1]
    # Extract iteration number from e.g. comparison_2026_iter4.csv
    name = chosen.stem
    iter_part = name.split("_iter")[-1] if "_iter" in name else preferred_suffix
    return chosen, iter_part
