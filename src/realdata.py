"""Real-data base-strategy library from Ken French 49 value-weighted industry
portfolios (daily, CRSP 2026-08 vintage).

Each base strategy trades one industry *hedged against the market* (industry minus
Mkt) with a position decided from information up to t-1:
  hold_i        : always long industry-minus-market
  tsmom_i_L     : sign(sum of hedged returns over the previous L days), L in LOOKBACKS
  ma_i_s_l      : sign(MA_s - MA_l) of the hedged cumulative return index
Splits: train 1970-1999, validation 2000-2012, test 2013-2026.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
LOOKBACKS = [5, 21, 63, 126, 252]
MAS = [(5, 21), (21, 63), (50, 200)]
SPLITS = {"train": ("1970-01-01", "1999-12-31"),
          "val": ("2000-01-01", "2012-12-31"),
          "test": ("2013-01-01", "2026-12-31")}


def _read_french(path: Path, header_marker: str) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(header_marker)) + 1
    rows = []
    for l in lines[start:]:
        parts = l.split(",")
        if not parts[0].strip().isdigit():
            break
        rows.append(parts)
    header = [c.strip() for c in lines[start - 1].split(",")] if "," in lines[start - 1] else None
    df = pd.DataFrame(rows)
    df.columns = ["date"] + (header[1:] if header else list(range(1, df.shape[1])))
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d")
    df = df.set_index("date").astype(float)
    return df.replace([-99.99, -999], np.nan) / 100.0


@lru_cache
def load_panels():
    ind = _read_french(DATA / "49_Industry_Portfolios_Daily.csv", ",Agric")
    ff = _read_french(DATA / "F-F_Research_Data_Factors_daily.csv", ",Mkt-RF")
    return ind, ff


@lru_cache
def base_library(start="1969-01-01"):
    """Returns (R, names, dates): R is (T, k) daily base-strategy returns."""
    ind, ff = load_panels()
    ind = ind.loc[start:].dropna(axis=1)
    mkt = (ff["Mkt-RF"] + ff["RF"]).reindex(ind.index)
    hedged = ind.sub(mkt, axis=0)
    cols, names = [], []
    cum = hedged.cumsum()
    for c in hedged.columns:
        h = hedged[c]
        cols.append(h); names.append(f"hold_{c}")
        for L in LOOKBACKS:
            pos = np.sign(h.rolling(L).sum().shift(1))
            cols.append(pos * h); names.append(f"tsmom_{c}_{L}")
        for s, l in MAS:
            pos = np.sign(cum[c].rolling(s).mean() - cum[c].rolling(l).mean()).shift(1)
            cols.append(pos * h); names.append(f"ma_{c}_{s}_{l}")
    R = pd.concat(cols, axis=1)
    R.columns = names
    R = R.loc["1970-01-01":].fillna(0.0)
    return R


def split(R: pd.DataFrame):
    return {k: R.loc[a:b].to_numpy() for k, (a, b) in SPLITS.items()}


def stationary_bootstrap_null(R: np.ndarray, rng, mean_block=21):
    """Null-ified copy of R: demean every column (no true edge), then jointly resample
    time with a stationary bootstrap (keeps cross-strategy correlation, vol clustering
    within blocks, fat tails).  Any 'discovery' on this matrix is false."""
    X = R - R.mean(axis=0, keepdims=True)
    T = X.shape[0]
    idx = np.empty(T, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(T)
    for t in range(T):
        idx[t] = i
        i = rng.integers(T) if rng.random() < p else (i + 1) % T
    return X[idx]
