/* Filter panel: crop, year, metric, geography. */

import type { FilterState, Meta, Metric } from "../lib/types";
import { METRIC_LABELS } from "../lib/format";

interface Props {
  meta: Meta;
  filters: FilterState;
  /** selectable states, as {fips prefix, USPS code} */
  states: { fips: string; code: string }[];
  forecastYear: number | null;
  matchCount: number | null;
  onChange: (patch: Partial<FilterState>) => void;
}

const METRICS: Metric[] = ["production", "yield", "pctDiff"];

export function Sidebar({
  meta,
  filters,
  states,
  forecastYear,
  matchCount,
  onChange,
}: Props) {
  const years = meta.years[filters.crop] ?? [];

  return (
    <div className="sidebar-inner">
      <Field label="Crop" htmlFor="crop">
        <select
          id="crop"
          value={filters.crop}
          onChange={(e) => onChange({ crop: e.target.value })}
        >
          {meta.crops.map((c) => (
            <option key={c} value={c}>
              {meta.cropLabels[c] ?? c}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Year" htmlFor="year">
        <select
          id="year"
          value={filters.year}
          onChange={(e) => onChange({ year: Number(e.target.value) })}
        >
          {[...years].reverse().map((y) => (
            <option key={y} value={y}>
              {y}
              {y === forecastYear ? " — forecast" : ""}
            </option>
          ))}
        </select>
      </Field>

      <fieldset className="field metric-group">
        <legend className="label">Map view</legend>
        <div className="metric-options">
          {METRICS.map((m) => (
            <label key={m} className={filters.metric === m ? "metric active" : "metric"}>
              <input
                type="radio"
                name="metric"
                value={m}
                checked={filters.metric === m}
                onChange={() => onChange({ metric: m })}
              />
              <span>{METRIC_LABELS[m]}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="sidebar-divider" />

      <Field label="State" htmlFor="state">
        <select
          id="state"
          value={filters.state}
          onChange={(e) => onChange({ state: e.target.value })}
        >
          <option value="ALL">All United States</option>
          {states.map((s) => (
            <option key={s.fips} value={s.fips}>
              {s.code}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Find county" htmlFor="search">
        <input
          id="search"
          type="search"
          placeholder="e.g. Story"
          autoComplete="off"
          value={filters.search}
          onChange={(e) => onChange({ search: e.target.value })}
        />
        {filters.search.trim() !== "" && matchCount !== null && (
          <p className="hint" role="status">
            {matchCount === 0
              ? "No counties match."
              : `${matchCount} ${matchCount === 1 ? "county" : "counties"} outlined.`}
          </p>
        )}
      </Field>
    </div>
  );
}

function Field({
  label,
  htmlFor,
  children,
}: {
  label: string;
  htmlFor: string;
  children: React.ReactNode;
}) {
  return (
    <div className="field">
      <label className="label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
    </div>
  );
}
