"""Fetch county-level yields and harvested acres from the USDA NASS Quick Stats API.

Harvested acres are not optional: they are the weights used for the production-weighted
state and national rollups. A plain mean over counties is materially wrong.

The API refuses any query returning more than 50k records, so we chunk by year.
Each (crop, year) response is cached under data/raw/nass/ and skipped on re-run.

Usage:
    python pipeline/fetch_nass.py --crop corn
    python pipeline/fetch_nass.py            # both crops
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date

import pandas as pd
import requests

from config import CROPS, EXCLUDED_STATE_FIPS, RAW, START_YEAR, nass_api_key

API = "https://quickstats.nass.usda.gov/api/api_GET/"
CACHE = RAW / "nass"
CACHE.mkdir(parents=True, exist_ok=True)


def _clean_value(v: str) -> float | None:
    """NASS suppresses values as '(D)', '(Z)', '(NA)' and thousands-separates numbers."""
    if v is None:
        return None
    v = str(v).strip().replace(",", "")
    if not v or v.startswith("("):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _get(params: dict, retries: int = 6) -> list[dict]:
    """GET with backoff. NASS sits behind an Azure gateway that throttles with 403,
    not 429, so 403 must be treated as rate limiting and waited out, not failed."""
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=120)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"    network error ({exc}); retrying")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json().get("data", [])
        # NASS returns 400 with an explanatory body when a query matches no rows.
        if r.status_code == 400:
            body = r.text[:300]
            if "exceeds" in body.lower():
                raise RuntimeError(f"Query too large, needs finer chunking: {body}")
            return []
        if r.status_code in (403, 429, 503):
            wait = 30 * (attempt + 1)
            print(f"    throttled (HTTP {r.status_code}); sleeping {wait}s")
            time.sleep(wait)
            continue
        if attempt == retries - 1:
            raise RuntimeError(f"NASS HTTP {r.status_code}: {r.text[:300]}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("NASS request failed after retries (still throttled)")


def fetch_crop(crop: str, key: str, end_year: int) -> pd.DataFrame:
    spec = CROPS[crop]
    frames = []

    for year in range(START_YEAR, end_year + 1):
        cache_file = CACHE / f"{crop}_{year}.json"
        if cache_file.exists():
            rows = json.loads(cache_file.read_text())
        else:
            rows = []
            for stat in ("YIELD", "AREA HARVESTED"):
                rows += _get(
                    {
                        "key": key,
                        "source_desc": "SURVEY",
                        "sector_desc": "CROPS",
                        "commodity_desc": spec["commodity_desc"],
                        "statisticcat_desc": stat,
                        "agg_level_desc": "COUNTY",
                        "year": year,
                        "format": "JSON",
                    }
                )
                time.sleep(1.0)
            cache_file.write_text(json.dumps(rows))
            print(f"  {crop} {year}: {len(rows)} rows fetched")

        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Keep only the two exact short_desc series we want; NASS returns irrigated /
    # non-irrigated and other practice splits under the same statisticcat_desc.
    df = df[df["short_desc"].isin([spec["yield_stat"], spec["area_stat"]])].copy()

    df["fips"] = (
        df["state_fips_code"].astype(str).str.zfill(2)
        + df["county_code"].astype(str).str.zfill(3)
    )
    df["year"] = df["year"].astype(int)
    df["value"] = df["Value"].map(_clean_value)
    df = df.dropna(subset=["value"])

    # Drop non-CONUS and the '998' combined-counties pseudo-county NASS emits.
    df = df[~df["fips"].str[:2].isin(EXCLUDED_STATE_FIPS)]
    df = df[df["county_code"].astype(str).str.zfill(3) != "998"]

    df["metric"] = df["short_desc"].map(
        {spec["yield_stat"]: "yield", spec["area_stat"]: "acres"}
    )

    wide = (
        df.pivot_table(
            index=["fips", "year", "state_alpha", "county_name"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for col in ("yield", "acres"):
        if col not in wide.columns:
            wide[col] = pd.NA

    wide = wide.dropna(subset=["yield"])
    wide["crop"] = crop
    return wide[
        ["fips", "year", "state_alpha", "county_name", "crop", "yield", "acres"]
    ].sort_values(["fips", "year"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=list(CROPS), help="default: all crops")
    ap.add_argument("--end-year", type=int, default=date.today().year)
    args = ap.parse_args()

    key = nass_api_key()
    crops = [args.crop] if args.crop else list(CROPS)

    for crop in crops:
        print(f"Fetching {crop} {START_YEAR}-{args.end_year} ...")
        df = fetch_crop(crop, key, args.end_year)
        if df.empty:
            print(f"  no data for {crop}")
            continue
        out = RAW / f"nass_{crop}.parquet"
        df.to_parquet(out, index=False)
        n_recent = df[df.year == df.year.max()].shape[0]
        print(
            f"  wrote {out.name}: {len(df):,} county-years, "
            f"{df.fips.nunique():,} counties, {df.year.min()}-{df.year.max()} "
            f"({n_recent:,} counties in {df.year.max()})"
        )


if __name__ == "__main__":
    main()
