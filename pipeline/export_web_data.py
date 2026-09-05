"""Compact per-county-year data for the web frontend.

Joins observed NASS yields (data/processed/features_{crop}.parquet) with the
current-season LSTM forecast (data/processed/predictions_{crop}.parquet, see
predict_lstm.py) into one record per crop. Only the columns the frontend
actually uses are kept -- the source parquet also carries ~130 weather
feature columns that have no reason to ship to the browser.

Output shape is normalized rather than one flat list of county-year objects:
county identity (state, display name) is repeated ~35x per county across
years in a flat structure, which is most of the payload for no reason. So
each crop file is { fips: { state, county, years: [...], yield: [...],
predicted: [...], acres: [...] } }, with the three parallel arrays aligned
to `years`. This is still plain JSON (no new dependency) but roughly halves
the encoded size versus one row per county-year.

Deliberately NOT precomputed here: five-year mean, diff, %diff, production.
Those are cheap, well-defined derivations (web/src/lib/metrics.ts) and
keeping one source of truth for "5 preceding valid years, excluding the
selected year" avoids a second, possibly-drifting implementation server-side.

Usage:
    python pipeline/export_web_data.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import CROPS, PROCESSED, RAW, WEB_PUBLIC

OUT_DIR = WEB_PUBLIC / "data"


def export_crop(crop: str, centroids: pd.DataFrame) -> tuple[dict, dict]:
    df = pd.read_parquet(PROCESSED / f"features_{crop}.parquet")
    df = df[["fips", "year", "yield", "acres"]]

    pred_path = PROCESSED / f"predictions_{crop}.parquet"
    preds = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame(
        columns=["fips", "year", "predicted_yield", "baseline_5yr_mean", "val_mae"]
    )
    df = df.merge(
        preds[["fips", "year", "predicted_yield"]], on=["fips", "year"], how="left"
    )
    df = df[df["yield"].notna() | df["predicted_yield"].notna()]
    df = df.merge(centroids[["fips", "county_name", "state_alpha"]], on="fips", how="inner")
    df = df.sort_values(["fips", "year"])

    # NASS publishes harvested acreage only after harvest, so the forecast year
    # has none -- and without acreage there is no production to show. Carry the
    # county's most recent reported acreage forward as an explicit estimate.
    # This is an assumption, not an observation, so it ships in a separate field
    # and the UI labels any production derived from it as estimated.
    df["acres_est"] = np.nan
    need = df["acres"].isna()
    if need.any():
        last_acres = (
            df[df["acres"].notna()]
            .sort_values("year")
            .groupby("fips")["acres"]
            .last()
        )
        df.loc[need, "acres_est"] = df.loc[need, "fips"].map(last_acres)

    counties: dict[str, dict] = {}
    for fips, g in df.groupby("fips", sort=False):
        counties[fips] = {
            "state": g["state_alpha"].iloc[0],
            "county": g["county_name"].iloc[0],
            "years": g["year"].astype(int).tolist(),
            "yield": [round(float(v), 2) if pd.notna(v) else None for v in g["yield"]],
            "predicted": [round(float(v), 2) if pd.notna(v) else None for v in g["predicted_yield"]],
            "acres": [round(float(v), 1) if pd.notna(v) else None for v in g["acres"]],
            "acresEst": [round(float(v), 1) if pd.notna(v) else None for v in g["acres_est"]],
        }

    val_mae = None
    if not preds.empty and preds["val_mae"].notna().any():
        val_mae = round(float(preds["val_mae"].iloc[0]), 2)

    years = sorted(df["year"].unique().tolist())
    return counties, {"years": years, "valMae": val_mae}


def main() -> None:
    centroids = pd.read_parquet(RAW / "county_centroids.parquet")[
        ["fips", "county_name", "state_alpha"]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "crops": list(CROPS.keys()),
        "cropLabels": {k: v["label"] for k, v in CROPS.items()},
        "unit": "bu/acre",
        "years": {},
        "valMae": {},
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }

    for crop in CROPS:
        counties, info = export_crop(crop, centroids)
        path = OUT_DIR / f"{crop}.json"
        path.write_text(json.dumps(counties, separators=(",", ":")))
        meta["years"][crop] = info["years"]
        meta["valMae"][crop] = info["valMae"]
        print(f"{crop}: {len(counties):,} counties -> {path.name}")

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("meta.json written")


if __name__ == "__main__":
    main()
