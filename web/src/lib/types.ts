/* Schema of the static data files in public/data, written by
 * pipeline/export_web_data.py and pipeline/export_geo.py + scripts/build-topo.mjs.
 *
 * Everything the app renders comes from these files. There are no runtime API
 * calls: the site is fully static once built. */

/** One county's full time series for a single crop.
 *
 * `years` is ascending, and `yield`/`predicted`/`acres` are parallel to it.
 * A county-year only appears if it has an observed yield or a forecast, so
 * gaps in a county's history are gaps in `years`, not nulls. */
export interface CountySeries {
  /** two-letter USPS state code, e.g. "IA" */
  state: string;
  /** display name including the "County"/"Parish" suffix, e.g. "Story County" */
  county: string;
  years: number[];
  /** observed NASS yield in bu/acre; null for the unlabeled forecast year */
  yield: (number | null)[];
  /** model forecast in bu/acre; non-null only for the forecast year */
  predicted: (number | null)[];
  /** harvested acres as reported by NASS; null where unpublished */
  acres: (number | null)[];
  /** the county's most recent reported acreage, carried forward. Only used
   * where `acres` is null (NASS publishes acreage after harvest, so the
   * forecast year has none). Production from this is an estimate. */
  acresEst: (number | null)[];
}

/** public/data/{crop}.json — keyed by 5-digit zero-padded county FIPS. */
export type CropData = Record<string, CountySeries>;

/** public/data/meta.json */
export interface Meta {
  crops: string[];
  cropLabels: Record<string, string>;
  unit: string;
  years: Record<string, number[]>;
  /** held-out MAE of the forecast model, per crop, in `unit` */
  valMae: Record<string, number | null>;
  generatedAt: string;
}

export type Metric = "yield" | "pctDiff" | "production";

export interface FilterState {
  crop: string;
  year: number;
  metric: Metric;
  /** two-letter state code, or "ALL" */
  state: string;
  search: string;
}

/** Everything the map and tooltip need for one county under current filters. */
export interface CountyValue {
  fips: string;
  state: string;
  county: string;
  /** the yield used for this year: forecast if present, else observed */
  value: number | null;
  isForecast: boolean;
  acres: number | null;
  /** true when `acres` was carried forward rather than reported, which makes
   * `production` an estimate */
  acresEstimated: boolean;
  production: number | null;
  baseline: number | null;
  /** how many prior years the baseline averaged (0 => not enough history) */
  baselineYears: number;
  diff: number | null;
  pctDiff: number | null;
  /** the number the active metric maps to color; null renders as "no data" */
  metricValue: number | null;
}
