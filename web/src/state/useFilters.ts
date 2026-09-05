/* Filter state, mirrored into the URL query string so a particular view can be
 * linked or reloaded without losing the selection. */

import { useCallback, useEffect, useState } from "react";
import type { FilterState, Meta, Metric } from "../lib/types";

const METRICS: Metric[] = ["production", "yield", "pctDiff"];

function readURL(meta: Meta): FilterState {
  const params = new URLSearchParams(window.location.search);

  const crop = meta.crops.includes(params.get("crop") ?? "")
    ? (params.get("crop") as string)
    : meta.crops[0];

  const years = meta.years[crop] ?? [];
  const urlYear = Number(params.get("year"));
  // Default to the most recent year, which is the in-season forecast.
  const year = years.includes(urlYear) ? urlYear : years[years.length - 1];

  const urlMetric = params.get("metric") as Metric | null;
  const metric = urlMetric && METRICS.includes(urlMetric) ? urlMetric : METRICS[0];

  return {
    crop,
    year,
    metric,
    state: params.get("state") ?? "ALL",
    search: "",
  };
}

export function useFilters(meta: Meta) {
  const [filters, setFilters] = useState<FilterState>(() => readURL(meta));

  useEffect(() => {
    const params = new URLSearchParams();
    params.set("crop", filters.crop);
    params.set("year", String(filters.year));
    params.set("metric", filters.metric);
    if (filters.state !== "ALL") params.set("state", filters.state);
    const qs = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}?${qs}`);
  }, [filters]);

  const update = useCallback(
    (patch: Partial<FilterState>) => {
      setFilters((prev) => {
        const next = { ...prev, ...patch };
        // Not every crop covers every year; keep the selection valid.
        if (patch.crop && patch.crop !== prev.crop) {
          const years = meta.years[patch.crop] ?? [];
          if (!years.includes(next.year)) next.year = years[years.length - 1];
        }
        return next;
      });
    },
    [meta],
  );

  return [filters, update] as const;
}
