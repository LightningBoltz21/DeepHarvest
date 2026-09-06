/* Ranked list of the counties currently on the map.
 *
 * Shares the exact `CountyValue[]` the choropleth is drawing, so the ranking,
 * the coloring and the legend can never disagree. The list can run to ~1,800
 * rows, which is too many DOM nodes to rebuild on every filter change, so only
 * the visible window is rendered. */

import { useEffect, useMemo, useRef, useState } from "react";
import { colorFor } from "../lib/colorScales";
import { METRIC_LABELS, metricValue } from "../lib/format";
import type { CountyValue, Metric } from "../lib/types";

interface Props {
  values: CountyValue[];
  domain: [number, number] | null;
  metric: Metric;
  crop: string;
  /** FIPS matching the county search, if any */
  highlighted: Set<string>;
  hoveredFips: string | null;
  onHoverRow: (value: CountyValue | null) => void;
}

/* Must match the row height the stylesheet lays out, since the virtual list
 * positions rows arithmetically rather than by measuring them. */
const ROW_HEIGHT = 44;
const OVERSCAN = 8;

export function RankPanel({
  values,
  domain,
  metric,
  crop,
  highlighted,
  hoveredFips,
  onHoverRow,
}: Props) {
  const [descending, setDescending] = useState(true);
  /* On a short screen (a phone in landscape) the list and the map cannot both
   * be useful, so the list starts folded and the map gets the space. */
  const [collapsed, setCollapsed] = useState(
    () => typeof window !== "undefined" && window.innerHeight < 520,
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  // filter() already returns a new array, so sorting it does not disturb the
  // caller's list -- which the map is drawing from.
  const ranked = useMemo(
    () =>
      values
        .filter((v) => v.metricValue !== null)
        .sort((a, b) =>
          descending
            ? (b.metricValue as number) - (a.metricValue as number)
            : (a.metricValue as number) - (b.metricValue as number),
        ),
    [values, descending],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setViewportHeight(entry.contentRect.height),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Changing sort or filters should bring the reader back to the top.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
    setScrollTop(0);
  }, [descending, metric, crop]);

  const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
  const last = Math.min(
    ranked.length,
    Math.ceil((scrollTop + viewportHeight) / ROW_HEIGHT) + OVERSCAN,
  );
  const rows = ranked.slice(first, last);

  return (
    <section
      className={collapsed ? "rank collapsed" : "rank"}
      aria-label={`Counties ranked by ${METRIC_LABELS[metric]}`}
    >
      <header className="rank-head">
        <div className="rank-title">
          <span className="label">Ranking</span>
          <p className="rank-sub">
            {ranked.length.toLocaleString()} counties · {METRIC_LABELS[metric]}
          </p>
        </div>
        <button
          type="button"
          className="rank-sort"
          onClick={() => setDescending((d) => !d)}
          aria-label={descending ? "Sort ascending" : "Sort descending"}
        >
          {descending ? "High → low" : "Low → high"}
        </button>
        {/* Phone-only: the list and the map compete for a short screen, so the
            list can be folded away to give the map the whole viewport. */}
        <button
          type="button"
          className="rank-collapse"
          aria-expanded={!collapsed}
          onClick={() => setCollapsed((c) => !c)}
        >
          {collapsed ? "Show" : "Hide"}
        </button>
      </header>

      <div
        className="rank-scroll"
        ref={scrollRef}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      >
        {ranked.length === 0 ? (
          <p className="rank-empty">No counties to rank.</p>
        ) : (
          <ol
            className="rank-list"
            style={{ height: ranked.length * ROW_HEIGHT }}
            // The list is absolutely positioned inside a spacer of the full
            // height, so the scrollbar reflects the whole ranking.
          >
            {rows.map((v, i) => {
              const rank = first + i + 1;
              const isHovered = v.fips === hoveredFips;
              const isMatch = highlighted.has(v.fips);
              return (
                <li
                  key={v.fips}
                  className={
                    "rank-row" +
                    (isHovered ? " hovered" : "") +
                    (isMatch ? " matched" : "")
                  }
                  style={{ top: (first + i) * ROW_HEIGHT, height: ROW_HEIGHT }}
                  onMouseEnter={() => onHoverRow(v)}
                  onMouseLeave={() => onHoverRow(null)}
                >
                  <span className="rank-n mono">{rank}</span>
                  <span
                    className="rank-swatch"
                    style={{ background: colorFor(v.metricValue, domain, metric, crop) }}
                    aria-hidden="true"
                  />
                  <span className="rank-name" title={`${v.county}, ${v.state}`}>
                    {v.county}
                  </span>
                  <span className="rank-state mono">{v.state}</span>
                  <span className="rank-value mono">
                    {metricValue(v.metricValue, metric)}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}
