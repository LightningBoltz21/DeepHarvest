"""Turn daily weather + NASS yields into model-ready features.

Two feature groups:

* Per-timestep weather, shaped [T=14, F=8] per county-year. The growing season
  (Apr 1 - Oct 31) is split into 14 biweekly bins. This is the temporal axis the
  MMST-ViT-Lite temporal transformer attends over, and the GBM consumes flattened.

* Per-county-year scalars: lagged yields, an expanding-window trend estimate, county
  mean, lat/lon and year.

LEAKAGE: every yield-derived feature for target year Y uses only years strictly
before Y, via expanding windows shifted by one observation. Trend lines are refit
per year from prior data only. Getting this wrong is the easiest way to produce
impressive-looking but meaningless scores, so it is enforced here, once, centrally.

Partial seasons are expected and fine: the current year has no data past today, so
its late bins are NaN and carry n_days=0. train.py truncates every year to the same
bin cutoff when forecasting in-season, keeping train and inference distributions
aligned.

Usage:
    python pipeline/features.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import (
    CROPS,
    N_TIMESTEPS,
    PROCESSED,
    RAW,
    SEASON_END_DOY,
    SEASON_START_DOY,
    TIMESTEP_FEATURES,
    WEATHER_RAW,
)

GDD_BASE = 10.0   # degC, standard for corn/soy
GDD_CAP = 30.0    # development plateaus above this
HEAT_THRESHOLD = 30.0  # degC; days above this drive the classic yield penalty

SEASON_DAYS = SEASON_END_DOY - SEASON_START_DOY + 1
BIN_WIDTH = SEASON_DAYS / N_TIMESTEPS


def season_bins(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one county's daily weather into per-year, per-bin season features."""
    df = weather.copy()
    doy = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    in_season = (doy >= SEASON_START_DOY) & (doy <= SEASON_END_DOY)
    df = df[in_season].copy()
    df["bin"] = np.minimum(
        ((doy[in_season] - SEASON_START_DOY) / BIN_WIDTH).astype(int),
        N_TIMESTEPS - 1,
    )

    tmax_c = df["tmax"].clip(upper=GDD_CAP)
    tmin_c = df["tmin"].clip(lower=GDD_BASE)
    df["gdd_d"] = np.clip((tmax_c + tmin_c) / 2.0 - GDD_BASE, 0.0, None)
    df["heat_d"] = (df["tmax"] > HEAT_THRESHOLD).astype(float)
    df["wb_d"] = df["precip"] - df["et0"]

    g = df.groupby(["year", "bin"])
    out = g.agg(
        gdd=("gdd_d", "sum"),
        precip=("precip", "sum"),
        tmax_mean=("tmax", "mean"),
        tmax_max=("tmax", "max"),
        heat_days=("heat_d", "sum"),
        et0=("et0", "sum"),
        radiation=("srad", "sum"),
        water_balance=("wb_d", "sum"),
        n_days=("tmax", "size"),
    ).reset_index()
    out["fips"] = weather["fips"].iloc[0]
    return out


def build_weather_features() -> pd.DataFrame:
    """Season-binned weather for every cached county."""
    cache = PROCESSED / "weather_bins.parquet"
    files = sorted(WEATHER_RAW.glob("*.parquet"))
    if cache.exists():
        cached = pd.read_parquet(cache)
        if cached["fips"].nunique() == len(files):
            print(f"Using cached weather bins ({len(files):,} counties)")
            return cached

    print(f"Binning weather for {len(files):,} counties ...")
    frames = []
    for i, f in enumerate(files, 1):
        frames.append(season_bins(pd.read_parquet(f)))
        if i % 500 == 0:
            print(f"  {i:,}/{len(files):,}")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    print(f"  wrote {cache.name}: {len(out):,} county-year-bins")
    return out


def wide_weather(bins: pd.DataFrame) -> pd.DataFrame:
    """Long bins -> one row per county-year with <feature>_t<bin> columns."""
    wide = bins.pivot_table(
        index=["fips", "year"], columns="bin", values=TIMESTEP_FEATURES
    )
    wide.columns = [f"{feat}_t{b}" for feat, b in wide.columns]
    wide = wide.reset_index()

    coverage = (
        bins.pivot_table(index=["fips", "year"], columns="bin", values="n_days")
        .reindex(columns=range(N_TIMESTEPS))
        .fillna(0)
    )
    coverage.columns = [f"n_days_t{b}" for b in coverage.columns]
    return wide.merge(coverage.reset_index(), on=["fips", "year"], how="left")


def add_yield_history(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, expanding-mean and expanding-trend features, all strictly backward-looking.

    The trend is an OLS line refit for every target year using only prior observations,
    computed in closed form from expanding sums so it stays O(n) rather than refitting
    a regression per row.
    """
    df = df.sort_values(["fips", "year"]).copy()
    g = df.groupby("fips", sort=False)
    y = df["yield"]

    for lag in (1, 2, 3):
        df[f"yield_lag{lag}"] = g["yield"].shift(lag)

    # Expanding stats over strictly prior years.
    df["yield_prior_mean"] = g["yield"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["yield_prior_std"] = g["yield"].transform(
        lambda s: s.shift(1).expanding().std()
    )
    df["n_prior_years"] = g["yield"].transform(lambda s: s.shift(1).expanding().count())

    # Closed-form expanding OLS of yield on year, using prior years only.
    x = df["year"].astype(float)
    df["_x"], df["_y"], df["_xy"], df["_xx"] = x, y, x * y, x * x
    for col in ("_x", "_y", "_xy", "_xx"):
        df[f"{col}_cs"] = g[col].transform(lambda s: s.shift(1).expanding().sum())
    n = df["n_prior_years"]
    denom = n * df["_xx_cs"] - df["_x_cs"] ** 2
    slope = (n * df["_xy_cs"] - df["_x_cs"] * df["_y_cs"]) / denom.replace(0, np.nan)
    intercept = (df["_y_cs"] - slope * df["_x_cs"]) / n
    df["trend_slope"] = slope
    df["trend_pred"] = intercept + slope * x
    # With <3 prior years a trend line is noise; fall back to the prior mean.
    thin = df["n_prior_years"] < 3
    df.loc[thin, "trend_pred"] = df.loc[thin, "yield_prior_mean"]
    df.loc[thin, "trend_slope"] = np.nan

    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def build(crop: str) -> pd.DataFrame:
    nass = pd.read_parquet(RAW / f"nass_{crop}.parquet")
    cents = pd.read_parquet(RAW / "county_centroids.parquet")[
        ["fips", "lat", "lon", "land_sqmi"]
    ]
    bins = build_weather_features()
    weather = wide_weather(bins)

    # Outer-join on the weather side for the current season: NASS has no rows for the
    # in-progress year, but we still need its weather row to predict into.
    df = weather.merge(nass, on=["fips", "year"], how="left").merge(
        cents, on="fips", how="left"
    )
    df = df[df["lat"].notna()]
    df = add_yield_history(df)
    df["crop"] = crop
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=list(CROPS))
    args = ap.parse_args()

    for crop in [args.crop] if args.crop else list(CROPS):
        df = build(crop)
        out = PROCESSED / f"features_{crop}.parquet"
        df.to_parquet(out, index=False)
        labeled = df["yield"].notna().sum()
        print(
            f"{crop}: {len(df):,} county-years ({labeled:,} labeled), "
            f"{df.fips.nunique():,} counties, {df.year.min()}-{df.year.max()} "
            f"-> {out.name}"
        )


if __name__ == "__main__":
    main()
