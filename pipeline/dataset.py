"""Data access for DeepHarvest — pure pandas/numpy.

Deliberately imports NO modelling library. LightGBM and PyTorch each ship their own
libomp on macOS and segfault when loaded into the same process (in either order), so
this module stays neutral and lets the caller pick a framework.

Everything the notebook and the training scripts need to agree on lives here:
column layout, the weather tensor, year-blocked splits, and metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import N_TIMESTEPS, PROCESSED, TIMESTEP_FEATURES

# Per-county-year scalars. All yield-derived entries are strictly backward-looking;
# see pipeline/features.py for how the expanding windows are built.
STATIC_COLS = [
    "yield_lag1",
    "yield_lag2",
    "yield_lag3",
    "yield_prior_mean",
    "yield_prior_std",
    "n_prior_years",
    "trend_slope",
    "trend_pred",
    "lat",
    "lon",
    "land_sqmi",
    "year",
]

ID_COLS = ["fips", "year", "state_alpha", "county_name", "crop"]


def weather_cols(n_bins: int = N_TIMESTEPS) -> list[str]:
    """Flat weather column names, feature-major: gdd_t0..gdd_tN, precip_t0.., ..."""
    return [f"{f}_t{b}" for f in TIMESTEP_FEATURES for b in range(n_bins)]


def load_features(
    crop: str, as_of_bin: int = N_TIMESTEPS
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load one crop's features.

    Returns
    -------
    df : DataFrame, one row per county-year, sorted by (fips, year).
    W  : float32 [N, as_of_bin, 8] weather tensor. Channel order = TIMESTEP_FEATURES.
    S  : float32 [N, len(STATIC_COLS)] static/scalar features.

    `as_of_bin` truncates the growing season to its first N biweekly bins. Use it to
    match what is actually observable mid-season: training on full seasons and
    predicting a half-observed one is a train/inference mismatch.
    """
    path = PROCESSED / f"features_{crop}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run:\n"
            "  python pipeline/fetch_nass.py\n"
            "  python pipeline/fetch_weather.py --only-crop-counties\n"
            "  python pipeline/features.py"
        )
    df = pd.read_parquet(path).sort_values(["fips", "year"]).reset_index(drop=True)

    W = np.stack(
        [
            df[[f"{f}_t{b}" for b in range(as_of_bin)]].to_numpy(dtype=np.float32)
            for f in TIMESTEP_FEATURES
        ],
        axis=-1,
    )
    S = df[STATIC_COLS].to_numpy(dtype=np.float32)
    return df, W, S


def usable_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows fit for supervised training: have a label AND at least one prior year."""
    return df["yield"].notna().to_numpy() & df["yield_lag1"].notna().to_numpy()


def year_split(
    df: pd.DataFrame, test_year: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Year-blocked split: train on years < test_year, test on test_year.

    Random k-fold is not offered on purpose — it leaks future harvests into the past
    and yields scores that look good but cannot forecast.
    """
    m = usable_mask(df) if mask is None else mask
    train = np.where(m & (df["year"] < test_year).to_numpy())[0]
    test = np.where(m & (df["year"] == test_year).to_numpy())[0]
    return train, test


def holdout_tail(
    df: pd.DataFrame, train_rows: np.ndarray, n_years: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Carve the most recent `n_years` off a training set for early stopping."""
    years = df["year"].to_numpy()[train_rows]
    cut = years.max() - n_years + 1
    fit = train_rows[years < cut]
    val = train_rows[years >= cut]
    if len(val) == 0 or len(fit) < 500:
        return train_rows, train_rows[:0]
    return fit, val


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """RMSE / MAE / R2 / Pearson r, ignoring positions where either side is NaN."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "corr": np.nan}
    resid = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt((resid**2).mean())),
        "mae": float(np.abs(resid).mean()),
        "r2": float(1 - float((resid**2).sum()) / ss_tot) if ss_tot > 0 else np.nan,
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else np.nan,
    }


def acre_weighted(g: pd.DataFrame, value: str = "yield") -> float:
    """Production-weighted mean. State/national yield is NOT a plain county average."""
    w = g["acres"].fillna(0).to_numpy(dtype=float)
    v = g[value].to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w)
    if not ok.any() or w[ok].sum() <= 0:
        return float(np.nanmean(v)) if ok.any() else np.nan
    return float(np.average(v[ok], weights=w[ok]))
