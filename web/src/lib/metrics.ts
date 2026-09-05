/* Derived metrics: five-year baseline, anomalies, production.
 *
 * These are computed in the browser rather than baked into the static files so
 * that there is exactly one definition of the baseline window. */

import type { CountySeries, CountyValue, CropData, Metric } from "./types";

/** How many preceding years the baseline averages over. */
export const BASELINE_WINDOW = 5;

/** Mean observed yield over the `BASELINE_WINDOW` years *preceding* `year`.
 *
 * The selected year is excluded, so a forecast is never compared against
 * itself. Only observed (NASS) yields count -- a forecast is not evidence
 * about its own baseline. Years inside the window with no observation are
 * skipped rather than treated as zero, and `n` reports how many actually
 * contributed so callers can refuse to show a one-year "five-year mean". */
export function baseline(
  series: CountySeries,
  year: number,
): { mean: number | null; n: number } {
  const lo = year - BASELINE_WINDOW;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < series.years.length; i++) {
    const y = series.years[i];
    if (y >= lo && y < year) {
      const v = series.yield[i];
      if (v !== null) {
        sum += v;
        n++;
      }
    }
  }
  return n === 0 ? { mean: null, n: 0 } : { mean: sum / n, n };
}

/** Total production in bushels, derived as yield (bu/acre) x harvested acres.
 *
 * NASS publishes production directly, but not for the forecast year, so it is
 * derived here for consistency across years. Null when acreage is unpublished. */
export function production(value: number | null, acres: number | null): number | null {
  if (value === null || acres === null) return null;
  return value * acres;
}

/** Resolve one county for one crop-year into everything the UI needs. */
export function countyValue(
  fips: string,
  series: CountySeries,
  year: number,
  metric: Metric,
): CountyValue | null {
  const i = series.years.indexOf(year);
  if (i === -1) return null;

  const observed = series.yield[i];
  const forecast = series.predicted[i];
  const value = forecast ?? observed;
  if (value === null) return null;

  // Fall back to carried-forward acreage so the forecast year, which has no
  // reported acreage yet, still has a production figure.
  const reported = series.acres[i];
  const acres = reported ?? series.acresEst[i];
  const { mean, n } = baseline(series, year);

  const diff = mean === null ? null : value - mean;
  // Guard the ratio: a zero baseline would make the percentage infinite.
  const pctDiff = mean === null || mean === 0 ? null : ((value - mean) / mean) * 100;
  const prod = production(value, acres);

  const metricValue =
    metric === "yield" ? value : metric === "pctDiff" ? pctDiff : prod;

  return {
    fips,
    state: series.state,
    county: series.county,
    value,
    isForecast: forecast !== null,
    acres,
    acresEstimated: reported === null && acres !== null,
    production: prod,
    baseline: mean,
    baselineYears: n,
    diff,
    pctDiff,
    metricValue,
  };
}

/** Every county that has data for this crop-year, keyed by FIPS. */
export function countyValues(
  data: CropData,
  year: number,
  metric: Metric,
): Map<string, CountyValue> {
  const out = new Map<string, CountyValue>();
  for (const fips in data) {
    const v = countyValue(fips, data[fips], year, metric);
    if (v !== null) out.set(fips, v);
  }
  return out;
}

/** True when the metric diverges around zero and needs a centered scale. */
export function isDiverging(metric: Metric): boolean {
  return metric === "pctDiff";
}

/* Total production spans three orders of magnitude across counties and is
 * heavily right-skewed -- half of them fall in the bottom tenth of the range,
 * so a linear ramp renders most of the map as the darkest shade. Compressing
 * with a square root spreads those counties across the ramp while keeping the
 * ordering and the zero point intact. Yield per acre is near-symmetric and
 * needs no such treatment. */
export function isCompressed(metric: Metric): boolean {
  return metric === "production";
}

/** Position of a value in 0..1 within a domain, applying the metric's scale. */
export function scalePosition(
  value: number,
  domain: [number, number],
  metric: Metric,
): number {
  const [lo, hi] = domain;
  if (hi === lo) return 0.5;
  if (!isCompressed(metric)) return (value - lo) / (hi - lo);
  const f = (v: number) => Math.sqrt(Math.max(0, v - lo));
  return f(value) / f(hi);
}

/** Color-scale domain for the active metric.
 *
 * Sequential metrics span [min, max] of the data. Diverging metrics are forced
 * symmetric around zero so that equal departures above and below the baseline
 * get equally intense color; the extent uses a high percentile rather than the
 * outright max so a handful of extreme counties don't flatten the whole map. */
export function metricDomain(
  values: Iterable<CountyValue>,
  metric: Metric,
): [number, number] | null {
  const nums: number[] = [];
  for (const v of values) {
    if (v.metricValue !== null && Number.isFinite(v.metricValue)) nums.push(v.metricValue);
  }
  if (nums.length === 0) return null;
  nums.sort((a, b) => a - b);

  if (isDiverging(metric)) {
    const extent = Math.max(
      Math.abs(quantile(nums, 0.02)),
      Math.abs(quantile(nums, 0.98)),
    );
    return extent === 0 ? [-1, 1] : [-extent, extent];
  }
  const lo = quantile(nums, 0.02);
  const hi = quantile(nums, 0.98);
  return lo === hi ? [lo, lo + 1] : [lo, hi];
}

/** Linear-interpolated quantile of a pre-sorted array. */
function quantile(sorted: number[], p: number): number {
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}
