"""Shared paths, constants and crop definitions for the DeepHarvest pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
WEATHER_RAW = RAW / "weather"
WEB_PUBLIC = ROOT / "web" / "public"

for _d in (RAW, PROCESSED, WEATHER_RAW):
    _d.mkdir(parents=True, exist_ok=True)


def nass_api_key() -> str:
    """NASS key from env, falling back to a .env file at the repo root."""
    key = os.environ.get("NASS_API_KEY")
    if key:
        return key.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("NASS_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(
        "NASS_API_KEY is not set.\n"
        "Get a free key at https://quickstats.nass.usda.gov/api then either:\n"
        "  export NASS_API_KEY=your-key\n"
        "or write it to .env as NASS_API_KEY=your-key"
    )


# NASS commodity spec per crop we support.
CROPS = {
    "corn": {
        "commodity_desc": "CORN",
        "yield_stat": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
        "area_stat": "CORN, GRAIN - ACRES HARVESTED",
        "unit": "bu/acre",
        "label": "Corn (grain)",
    },
    "soybeans": {
        "commodity_desc": "SOYBEANS",
        "yield_stat": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
        "area_stat": "SOYBEANS - ACRES HARVESTED",
        "unit": "bu/acre",
        "label": "Soybeans",
    },
}

START_YEAR = 1980

# Weather history can start later than NASS: ERA5 payloads are large and the
# Open-Meteo free tier is volume-limited. Yields before ~1990 also reflect very
# different genetics, and the per-county trend feature absorbs that era anyway.
WEATHER_START_YEAR = 1990

# Growing season binned into 14 biweekly steps: Apr 1 (DOY 91) -> Oct 31 (DOY 304).
SEASON_START_DOY = 91
SEASON_END_DOY = 304
N_TIMESTEPS = 14

# Daily weather comes from NASA POWER (community=AG), which is keyless and serves a
# full 37-year daily record per point in ~1s. Open-Meteo was the first choice but its
# free tier is volume-billed, and 2.7k counties x 37 years would have taken ~10 days.
# POWER parameter -> our column name.
POWER_PARAMS = {
    "T2M_MAX": "tmax",           # degC
    "T2M_MIN": "tmin",           # degC
    "PRECTOTCORR": "precip",     # mm/day
    "ALLSKY_SFC_SW_DWN": "srad", # MJ/m^2/day
}
POWER_FILL = -999.0  # POWER's missing-data sentinel

# Per-timestep engineered features, fixed order -> the [T, F] tensor channel layout.
TIMESTEP_FEATURES = [
    "gdd",
    "precip",
    "tmax_mean",
    "tmax_max",
    "heat_days",
    "et0",
    "radiation",
    "water_balance",
]

# Non-contiguous / non-CONUS FIPS state prefixes excluded from the map and model.
EXCLUDED_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
