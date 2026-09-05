"""Fetch daily weather per county centroid from NASA POWER (community=AG).

POWER is keyless, purpose-built for agroclimatology, and returns a full multi-decade
daily record for one point in about a second, so ~2.7k counties take minutes.

Reference evapotranspiration is not requested from the API; it is computed locally
with the Hargreaves equation from tmax/tmin plus extraterrestrial radiation derived
from latitude and day-of-year. That is pure math, so it costs no extra requests.

County centroids come from the Census Gazetteer (GEOID = 5-digit FIPS).

Usage:
    python pipeline/fetch_weather.py --limit 20            # smoke test
    python pipeline/fetch_weather.py --only-crop-counties  # full run
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import requests

from config import (
    EXCLUDED_STATE_FIPS,
    POWER_FILL,
    POWER_PARAMS,
    RAW,
    WEATHER_RAW,
    WEATHER_START_YEAR,
)

POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
GAZETTEER = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)
CENTROIDS = RAW / "county_centroids.parquet"


def load_centroids() -> pd.DataFrame:
    """County FIPS -> lat/lon, cached locally after the first download."""
    if CENTROIDS.exists():
        return pd.read_parquet(CENTROIDS)

    print("Downloading Census county gazetteer ...")
    r = requests.get(GAZETTEER, timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        raw = zf.read(name).decode("latin-1")

    df = pd.read_csv(io.StringIO(raw), sep="\t", dtype={"GEOID": str})
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "GEOID": "fips",
            "INTPTLAT": "lat",
            "INTPTLONG": "lon",
            "NAME": "county_name",
            "USPS": "state_alpha",
            "ALAND_SQMI": "land_sqmi",
        }
    )
    df["fips"] = df["fips"].str.zfill(5)
    df = df[~df["fips"].str[:2].isin(EXCLUDED_STATE_FIPS)]
    df = df[["fips", "county_name", "state_alpha", "lat", "lon", "land_sqmi"]]
    df.to_parquet(CENTROIDS, index=False)
    print(f"  {len(df):,} CONUS counties")
    return df


def hargreaves_et0(
    tmax: np.ndarray, tmin: np.ndarray, doy: np.ndarray, lat_deg: float
) -> np.ndarray:
    """FAO-56 Hargreaves reference ET (mm/day).

    Ra (extraterrestrial radiation) is derived from latitude and day-of-year, so no
    additional API call is needed. Standard form:
        ET0 = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * Ra_mm
    """
    phi = np.deg2rad(lat_deg)
    j = 2.0 * np.pi * doy / 365.0
    dr = 1.0 + 0.033 * np.cos(j)                       # inverse relative distance
    decl = 0.409 * np.sin(j - 1.39)                    # solar declination
    # Clip guards the polar edge case where the sun never sets / never rises.
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
    ra_mj = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws))
    )
    ra_mm = ra_mj / 2.45                               # MJ/m2/day -> mm/day equivalent
    trange = np.clip(tmax - tmin, 0.0, None)
    tmean = (tmax + tmin) / 2.0
    return np.clip(0.0023 * (tmean + 17.8) * np.sqrt(trange) * ra_mm, 0.0, None)


def fetch_county(fips: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": "AG",
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": start,
        "end": end,
        "format": "JSON",
    }
    last_err = ""
    for attempt in range(5):
        try:
            r = requests.get(POWER_API, params=params, timeout=300)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            block = r.json()["properties"]["parameter"]
            df = pd.DataFrame({dst: block[src] for src, dst in POWER_PARAMS.items()})
            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

            # POWER encodes missing data as -999; treat as NaN then interpolate the
            # short gaps so downstream binning does not see sentinel values as real.
            for col in POWER_PARAMS.values():
                df[col] = df[col].replace(POWER_FILL, np.nan)
            df[list(POWER_PARAMS.values())] = (
                df[list(POWER_PARAMS.values())].interpolate(limit=5).ffill().bfill()
            )

            df["et0"] = hargreaves_et0(
                df["tmax"].to_numpy(),
                df["tmin"].to_numpy(),
                df["date"].dt.dayofyear.to_numpy(),
                lat,
            )
            df["fips"] = fips
            return df
        if r.status_code in (429, 503):
            time.sleep(20 * (attempt + 1))
            continue
        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"POWER fetch failed for {fips} — {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only fetch N counties (smoke test)")
    ap.add_argument("--end-year", type=int, default=date.today().year)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--only-crop-counties",
        action="store_true",
        help="restrict to counties present in the fetched NASS data",
    )
    args = ap.parse_args()

    cents = load_centroids()

    if args.only_crop_counties:
        keep: set[str] = set()
        for f in RAW.glob("nass_*.parquet"):
            keep |= set(pd.read_parquet(f, columns=["fips"])["fips"])
        if keep:
            cents = cents[cents["fips"].isin(keep)]
            print(f"Restricted to {len(cents):,} counties present in NASS data")

    todo = [
        row
        for row in cents.itertuples()
        if not (WEATHER_RAW / f"{row.fips}.parquet").exists()
    ]
    cached = len(cents) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    start = f"{WEATHER_START_YEAR}0101"
    end = min(date(args.end_year, 12, 31), date.today()).strftime("%Y%m%d")
    print(f"{cached:,} cached, {len(todo):,} to fetch  (range {start} -> {end})")
    if not todo:
        return

    t0 = time.time()
    failures: list[str] = []

    def work(row):
        df = fetch_county(row.fips, row.lat, row.lon, start, end)
        df.to_parquet(WEATHER_RAW / f"{row.fips}.parquet", index=False)
        return row.fips

    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(work, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # keep going; report at the end
                failures.append(row.fips)
                print(f"    FAILED {row.fips} ({row.county_name}): {exc}")
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(
                    f"  {i:,}/{len(todo):,}  ({i / el:.1f}/s, "
                    f"ETA {(len(todo) - i) / max(i / el, 1e-6) / 60:.1f} min)"
                )

    print(f"Done. {len(list(WEATHER_RAW.glob('*.parquet'))):,} counties cached.")
    if failures:
        print(f"{len(failures)} failures (re-run to retry): {failures[:20]}")


if __name__ == "__main__":
    main()
