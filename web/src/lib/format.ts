/* Number and label formatting for the map, legend and tooltip. */

import type { Metric } from "./types";

const NUM = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const INT = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function bushelsPerAcre(v: number | null): string {
  return v === null ? "—" : `${NUM.format(v)} bu/ac`;
}

export function signedBushels(v: number | null): string {
  if (v === null) return "—";
  return `${v > 0 ? "+" : ""}${NUM.format(v)} bu/ac`;
}

export function signedPercent(v: number | null): string {
  if (v === null) return "—";
  return `${v > 0 ? "+" : ""}${NUM.format(v)}%`;
}

export function acres(v: number | null): string {
  return v === null ? "—" : `${INT.format(v)} ac`;
}

/** Bushels, abbreviated -- county production runs to eight figures. */
export function bushels(v: number | null): string {
  if (v === null) return "—";
  if (Math.abs(v) >= 1e6) return `${NUM.format(v / 1e6)}M bu`;
  if (Math.abs(v) >= 1e3) return `${NUM.format(v / 1e3)}K bu`;
  return `${INT.format(v)} bu`;
}

/** Format a value in the units of the given metric. */
export function metricValue(v: number | null, metric: Metric): string {
  switch (metric) {
    case "yield":
      return bushelsPerAcre(v);
    case "pctDiff":
      return signedPercent(v);
    case "production":
      return bushels(v);
  }
}

/** Compact axis label for the legend, where space is tight. */
export function metricTick(v: number, metric: Metric): string {
  switch (metric) {
    case "pctDiff":
      return `${v > 0 ? "+" : ""}${NUM.format(v)}%`;
    case "production":
      return bushels(v).replace(" bu", "");
    default:
      return NUM.format(v);
  }
}

export const METRIC_LABELS: Record<Metric, string> = {
  yield: "Yield per Acre",
  pctDiff: "% vs. 5-Year Mean",
  production: "Total Production",
};

export const METRIC_UNITS: Record<Metric, string> = {
  yield: "bu/acre",
  pctDiff: "% vs. baseline",
  production: "bushels",
};
