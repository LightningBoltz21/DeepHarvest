# DeepHarvest

County-level crop yield data and modelling platform for US corn and soybeans.

Everything here is built on free, public data. The only credential needed is a USDA NASS
API key, which is free and issued instantly.

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

### What the map shows

Historical years (1990–2025) are observed NASS yields. The most recent year has
no NASS labels yet, so it is a forecast from `predict_lstm.py` — the same LSTM
and the same as-of-bin truncation as the 2026 map in `notebooks/02_model.ipynb`,
generalized to both crops. Held-out MAE is shown next to the map title.

Three map views:

- **Total production** — bushels, derived as yield x harvested acres. NASS
  publishes acreage only after harvest, so for the forecast year the county's
  most recently reported acreage is carried forward and both the acreage and the
  production are labelled as estimates in the tooltip. Production spans three
  orders of magnitude and is heavily right-skewed, so it is drawn on a
  square-root scale; the legend ticks are positioned to match.
- **Yield per acre** — bu/acre, the quantity the model predicts.
- **% vs. 5-year mean** — the selected year against the mean of the **observed**
  yields in the five preceding years, excluding the selected year. A county with
  no observations in that window reads "insufficient history" rather than
  showing a number.

Geometry comes from the `us-atlas` package (the maintained TopoJSON build of the
Census cartographic files), filtered to the counties that appear in the crop
data — which is what drops Alaska, Hawaii and the territories.
