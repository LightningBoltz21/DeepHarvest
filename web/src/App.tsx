import { useCallback, useEffect, useMemo, useState } from "react";
import { Header, type Tab } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { CountyMap } from "./components/CountyMap";
import { Legend } from "./components/Legend";
import { RankPanel } from "./components/RankPanel";
import { Tooltip } from "./components/Tooltip";
import { PricesPlaceholder } from "./components/PricesPlaceholder";
import {
  loadCrop,
  loadGeography,
  loadMeta,
  type CountyFeatures,
  type StateFeatures,
} from "./lib/dataLoader";
import { countyValues, metricDomain } from "./lib/metrics";
import { METRIC_LABELS } from "./lib/format";
import { useFilters } from "./state/useFilters";
import type { CountyValue, CropData, Meta } from "./lib/types";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMeta().then(setMeta).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <FatalError message={error} />;
  if (!meta) return <Booting />;
  return <Ready meta={meta} />;
}

function Ready({ meta }: { meta: Meta }) {
  const [tab, setTab] = useState<Tab>("yields");
  const [filters, update] = useFilters(meta);
  const [geo, setGeo] = useState<{ counties: CountyFeatures; states: StateFeatures } | null>(
    null,
  );
  const [crop, setCrop] = useState<CropData | null>(null);
  const [hover, setHover] = useState<{ value: CountyValue | null; x: number; y: number }>({
    value: null,
    x: 0,
    y: 0,
  });
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    loadGeography().then(setGeo).catch(() => setGeo(null));
  }, []);

  useEffect(() => {
    let live = true;
    loadCrop(filters.crop).then((d) => {
      if (live) setCrop(d);
    });
    return () => {
      live = false;
    };
  }, [filters.crop]);

  const values = useMemo(
    () => (crop ? countyValues(crop, filters.year, filters.metric) : new Map()),
    [crop, filters.year, filters.metric],
  );

  /* When a state is selected the map only shows that state, so the color scale
   * and the county count should describe what is actually on screen rather
   * than the whole country. */
  const visible = useMemo(() => {
    if (filters.state === "ALL") return [...values.values()];
    return [...values.values()].filter((v) => v.fips.startsWith(filters.state));
  }, [values, filters.state]);

  const domain = useMemo(
    () => metricDomain(visible, filters.metric),
    [visible, filters.metric],
  );

  /* State options come from the crop data itself, pairing the USPS code the
   * user sees with the FIPS prefix the map filters on. */
  const states = useMemo(() => {
    if (!crop) return [];
    const byFips = new Map<string, string>();
    for (const fips in crop) byFips.set(fips.slice(0, 2), crop[fips].state);
    return [...byFips]
      .map(([fips, code]) => ({ fips, code }))
      .sort((a, b) => a.code.localeCompare(b.code));
  }, [crop]);

  /* The forecast year is the last year in the crop's range: the in-season one
   * NASS has not published yields for yet. Derived from the data rather than
   * hardcoded so it stays right after the next pipeline run. */
  const forecastYear = useMemo(() => {
    const years = meta.years[filters.crop] ?? [];
    return years.length > 0 ? years[years.length - 1] : null;
  }, [meta, filters.crop]);

  const highlighted = useMemo(() => {
    const q = filters.search.trim().toLowerCase();
    if (q === "" || !crop) return new Set<string>();
    const out = new Set<string>();
    for (const fips in crop) {
      const c = crop[fips];
      if (c.county.toLowerCase().includes(q) && values.has(fips)) out.add(fips);
    }
    return out;
  }, [filters.search, crop, values]);

  const onHover = useCallback((value: CountyValue | null, x: number, y: number) => {
    setHover({ value, x, y });
  }, []);

  /* Hovering a row in the ranking outlines that county on the map. It sets no
   * pointer position, so the floating tooltip stays out of the way -- the row
   * itself already shows the number. */
  const [rowHover, setRowHover] = useState<string | null>(null);
  const onHoverRow = useCallback((value: CountyValue | null) => {
    setRowHover(value?.fips ?? null);
  }, []);

  const emphasized = useMemo(() => {
    if (highlighted.size > 0) return highlighted;
    return rowHover ? new Set([rowHover]) : new Set<string>();
  }, [highlighted, rowHover]);

  /* Counties drawn on the map with no value for this crop-year -- what the
   * legend's "no data" key accounts for. */
  const noDataCount = useMemo(() => {
    if (!geo) return 0;
    const inView = geo.counties.features.filter((f) =>
      filters.state === "ALL" ? true : String(f.id).startsWith(filters.state),
    );
    return inView.filter((f) => !values.has(String(f.id))).length;
  }, [geo, values, filters.state]);

  const ready = geo && crop;

  return (
    <div className="app">
      <Header tab={tab} onTab={setTab} />

      {tab === "prices" ? (
        <PricesPlaceholder />
      ) : (
        <main className="layout">
          <button
            type="button"
            className="sidebar-toggle"
            aria-expanded={sidebarOpen}
            onClick={() => setSidebarOpen((v) => !v)}
          >
            {sidebarOpen ? "Hide filters" : "Filters"}
          </button>

          <aside className={sidebarOpen ? "sidebar open" : "sidebar"}>
            <Sidebar
              meta={meta}
              filters={filters}
              states={states}
              forecastYear={forecastYear}
              matchCount={filters.search.trim() === "" ? null : highlighted.size}
              onChange={update}
            />
          </aside>

          <section className="map-panel">
            <div className="map-head">
              <h1>
                {METRIC_LABELS[filters.metric]}
                <span className="map-head-sub">
                  {meta.cropLabels[filters.crop] ?? filters.crop} · {filters.year}
                  {forecastYear === filters.year && meta.valMae[filters.crop] !== null && (
                    <> · forecast, held-out MAE ±{meta.valMae[filters.crop]} bu/ac</>
                  )}
                </span>
              </h1>
            </div>

            {ready ? (
              visible.length === 0 ? (
                <div className="map-empty">
                  <p>
                    No counties reported {meta.cropLabels[filters.crop]} in {filters.year}
                    {filters.state === "ALL" ? "" : " for this state"}.
                  </p>
                </div>
              ) : (
                <CountyMap
                  counties={geo.counties}
                  states={geo.states}
                  values={values}
                  domain={domain}
                  metric={filters.metric}
                  crop={filters.crop}
                  stateFilter={filters.state}
                  highlighted={emphasized}
                  onHover={onHover}
                />
              )
            ) : (
              <div className="map-empty">
                <p className="label">Loading county data…</p>
              </div>
            )}

            <Legend
              domain={domain}
              metric={filters.metric}
              crop={filters.crop}
              countyCount={visible.length}
              noDataCount={noDataCount}
            />
          </section>

          <RankPanel
            values={visible}
            domain={domain}
            metric={filters.metric}
            crop={filters.crop}
            highlighted={highlighted}
            hoveredFips={hover.value?.fips ?? null}
            onHoverRow={onHoverRow}
          />
        </main>
      )}

      <Tooltip
        value={hover.value}
        x={hover.x}
        y={hover.y}
        cropLabel={meta.cropLabels[filters.crop] ?? filters.crop}
        year={filters.year}
      />
    </div>
  );
}

function Booting() {
  return (
    <div className="boot">
      <span className="label">Loading DeepHarvest…</span>
    </div>
  );
}

function FatalError({ message }: { message: string }) {
  return (
    <div className="boot">
      <p className="label">Could not load data</p>
      <p className="mono">{message}</p>
    </div>
  );
}
