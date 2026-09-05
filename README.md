# DeepHarvest

County-level crop yield data and modelling workbench for US corn and soybeans.

Everything here is built on free, public data. The only credential needed is a USDA NASS
API key, which is free and issued instantly.

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "NASS_API_KEY=your-key" > .env          # free: https://quickstats.nass.usda.gov/api

.venv/bin/python pipeline/fetch_nass.py                          # ~10 min
.venv/bin/python pipeline/fetch_weather.py --only-crop-counties  # ~25 min
.venv/bin/python pipeline/features.py                            # ~5 min

.venv/bin/jupyter lab notebooks/01_data.ipynb
```

Data is already fetched in this working copy; the steps above are for rebuilding.

`data/` and `notebooks/data.csv` are not in the repo — they are large and fully
regenerable by the commands above. `02_model.ipynb` reads `notebooks/data.csv`,
which is a flat copy of the corn feature table:

```bash
.venv/bin/python -c "import pandas as pd; \
  pd.read_parquet('data/processed/features_corn.parquet') \
    .to_csv('notebooks/data.csv', index=False)"
```

## The data

| Source | Content | Auth |
|---|---|---|
| [USDA NASS Quick Stats](https://quickstats.nass.usda.gov/api) | County yield + acres harvested, 1980–2025 | Free key |
| [NASA POWER](https://power.larc.nasa.gov/) (`community=AG`) | Daily tmax/tmin/precip/solar, 1990–today | None |
| [Census Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html) | County centroids | None |

**Coverage:** 2,695 counties · 99,715 county-years per crop · 64,057 labeled corn,
53,231 labeled soybeans.

Reference evapotranspiration (`et0`) is computed locally with the Hargreaves equation
from tmax/tmin plus latitude-derived extraterrestrial radiation, so it costs no requests.

### Why NASA POWER and not Open-Meteo

Open-Meteo was the first choice and works, but its free tier is volume-billed: 2,695
counties × 37 years would have taken roughly 10 days of quota. POWER returns a full
37-year daily record per point in about a second and sustains ~7 req/s, so the whole
country takes minutes.

## Layout

```
pipeline/
  config.py         paths, crop specs, season/bin constants
  fetch_nass.py     NASS -> data/raw/nass_{crop}.parquet
  fetch_weather.py  POWER -> data/raw/weather/{fips}.parquet
  features.py       daily weather + yields -> data/processed/features_{crop}.parquet
  dataset.py        loading, splits, metrics  (NO ML imports — see landmine below)
  train.py          year-blocked CV driver
  predict_lstm.py   in-season forecast -> data/processed/predictions_{crop}.parquet
  export_web_data.py  actuals + forecast -> web/public/data/{crop}.json
  omp_guard.py      import-order shim, see below
models/
  gbm.py            LightGBM wrapper
  mmstvit_lite.py   MMST-ViT-inspired spatial-temporal transformer
notebooks/
  01_data.ipynb     data workbench — start here
web/                static site (Vite + React) — see "The website" below
```

## The website

`web/` is a static site: county choropleth of yields, anomalies and production,
with no backend and no runtime data fetching. Everything it renders is a file in
`web/public/data/`, produced by the pipeline and committed to the repo.

```bash
.venv/bin/python pipeline/predict_lstm.py      # forecast the current season (~2 min)
.venv/bin/python pipeline/export_web_data.py   # -> web/public/data/{crop}.json + meta.json

cd web
npm install
npm run build:topo    # county/state geometry -> public/data/us.topojson
npm run dev           # http://localhost:5173
npm run build         # static build in web/dist, deployable as-is
```

Re-run the two Python steps plus `build:topo` whenever the underlying NASS data
changes; the JSON files under `web/public/data/` are build artifacts but are
committed, because the deploy builds only the frontend.

### What the map shows

Historical years (1990–2025) are observed NASS yields. The most recent year has
no NASS labels yet, so it is a forecast from `predict_lstm.py` — the same LSTM
and the same as-of-bin truncation as the 2026 map in `notebooks/02_model.ipynb`,
generalized to both crops. Held-out MAE is shown next to the map title.

Three map views:

- **Yield per acre** — bu/acre, the quantity the model predicts.
- **% vs. 5-year mean** — the selected year against the mean of the **observed**
  yields in the five preceding years, excluding the selected year. A county with
  no observations in that window reads "insufficient history" rather than
  showing a number.
- **Total production** — bushels, derived as yield x harvested acres. NASS
  publishes acreage only after harvest, so for the forecast year the county's
  most recently reported acreage is carried forward and both the acreage and the
  production are labelled as estimates in the tooltip. Production spans three
  orders of magnitude and is heavily right-skewed, so it is drawn on a
  square-root scale; the legend ticks are positioned to match.

Corn is drawn in gold and soybeans in green, so the two are never mistaken for
each other; the percentage view uses a red-to-green diverging scale centred on
zero.

Geometry comes from the `us-atlas` package (the maintained TopoJSON build of the
Census cartographic files), filtered to the counties that appear in the crop
data — which is what drops Alaska, Hawaii and the territories.

## Features

Per county-year:

- **Weather tensor `[14, 8]`** — the Apr 1 – Oct 31 season in 14 biweekly bins, with
  channels `gdd, precip, tmax_mean, tmax_max, heat_days, et0, radiation, water_balance`.
  Also available flattened as `gdd_t0 … water_balance_t13`.
- **Static scalars** — `yield_lag1/2/3`, `yield_prior_mean`, `yield_prior_std`,
  `n_prior_years`, `trend_slope`, `trend_pred`, `lat`, `lon`, `land_sqmi`, `year`.
- **`n_days_t{b}`** — days actually observed in bin `b`; `0` means unobserved.

### Two correctness properties

1. **No leakage.** Every yield-derived feature for target year Y is computed from years
   strictly before Y, using expanding windows. Trend lines are refit per year in closed
   form from prior data only. The notebook re-derives one by hand and asserts on it.
2. **Year-blocked validation only.** `dataset.year_split` trains on `year < Y` and tests
   on `Y`. Random k-fold is deliberately not provided — it leaks future harvests
   backwards and produces scores that look good but cannot forecast.

### The current season

NASS publishes county yields only after harvest, so **2026 has zero labels** and a
partially observed season (10 of 14 bins complete as of early September). To forecast it,
load features truncated to the same cutoff so training matches inference:

```python
df, W, S = load_features("corn", 10)
```

## ⚠️ Environment landmine: lightgbm + torch

On macOS, `lightgbm` and `torch` each ship their own `libomp` and **segfault when
imported into the same process** — in either order. The symptom is a bare SIGSEGV with no
Python traceback:

- torch first → LightGBM's C `Dataset` constructor crashes
- lightgbm first → torch's `nn.TransformerEncoder` crashes during layer deepcopy

`KMP_DUPLICATE_LIB_OK=TRUE` does **not** fix it.

`pipeline/dataset.py` and `notebooks/01_data.ipynb` import neither, so they are safe. Use
one framework per process. `pipeline/omp_guard.py` forces the lightgbm-first order for
scripts that only need LightGBM.

Separately, **LightGBM 4.7 segfaults on a pandas 3.0 DataFrame** — pass a contiguous
numpy array instead. `models/gbm.py` handles this conversion internally.

## Models

`models/gbm.py` is a LightGBM wrapper; `models/mmstvit_lite.py` is a compact
spatial-temporal transformer.

`pipeline/train.py --crop corn --skip-nn` runs year-blocked CV for the GBM and works
today. The combined GBM + NN path hits the libomp conflict above and needs the two
models split across processes.

### Status

Verified GBM baseline, corn, year-blocked CV on 2023–2025:

| Model | R² | RMSE |
|---|---|---|
| trend-only baseline | −0.40 | 44.4 |
| LightGBM | **0.718** | **20.5** |

### On MMST-ViT

The original goal was [MMST-ViT (ICCV 2023)](https://github.com/fudong03/MMST-ViT). It
was not used because it ships **no pretrained weights**, needs the **2.61 TB** CropNet
corpus plus multi-GPU self-supervised pretraining, is validated on only **~200 counties**
in IA/IL/MS/LA, has data ending in **2022**, and is **CC-BY-NC 4.0**.

`models/mmstvit_lite.py` keeps the paper's three-stage structure — multi-modal fusion →
temporal transformer → spatial attention over k-nearest counties — but runs on per-county
time series instead of 384×384 Sentinel-2 tiles. It uses **no satellite imagery** and is
**not** a reproduction.

For a like-for-like comparison, the paper reports 2021 corn R²=0.811 / soybean R²=0.843
on its ~200-county subset. Score on IA/IL to compare fairly; national numbers across all
2,695 counties are a harder problem and will read lower.

## Rebuilding

All fetch steps cache to disk and are resumable — re-running skips what already exists.
To force a rebuild, delete the relevant cache:

```bash
rm -rf data/raw/nass/          # re-fetch NASS
rm -rf data/raw/weather/       # re-fetch weather
rm  data/processed/weather_bins.parquet   # re-bin the season
```
