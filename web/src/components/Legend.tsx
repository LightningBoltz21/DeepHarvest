/* Legend for the active metric: a continuous ramp with end/mid ticks, plus an
 * explicit no-data key so grey is never left to be guessed at. */

import { useMemo } from "react";
import { rampColor, NO_DATA_COLOR } from "../lib/colorScales";
import { isCompressed, isDiverging, scalePosition } from "../lib/metrics";
import { METRIC_UNITS, metricTick } from "../lib/format";
import type { Metric } from "../lib/types";

interface Props {
  domain: [number, number] | null;
  metric: Metric;
  crop: string;
  countyCount: number;
  noDataCount: number;
}

const STEPS = 28;

export function Legend({ domain, metric, crop, countyCount, noDataCount }: Props) {
  const swatches = useMemo(
    () => Array.from({ length: STEPS }, (_, i) => rampColor(i / (STEPS - 1), metric, crop)),
    [metric, crop],
  );

  if (!domain) {
    return (
      <div className="legend">
        <p className="legend-empty">No data for this selection.</p>
      </div>
    );
  }

  // A diverging ramp needs its midpoint labelled as zero, which is the whole
  // point of centring it. A compressed ramp is labelled at the value that
  // actually sits halfway along the bar, not the arithmetic midpoint, so the
  // numbers line up with the colors above them.
  const ticks = isDiverging(metric)
    ? [domain[0], 0, domain[1]]
    : isCompressed(metric)
      ? [domain[0], domain[0] + (domain[1] - domain[0]) / 4, domain[1]]
      : [domain[0], (domain[0] + domain[1]) / 2, domain[1]];

  return (
    <div className="legend">
      <div className="legend-head">
        <span className="label">{METRIC_UNITS[metric]}</span>
        <span className="legend-count mono">
          {countyCount.toLocaleString()} counties
        </span>
      </div>

      <div className="legend-ramp" aria-hidden="true">
        {swatches.map((c, i) => (
          <span key={i} style={{ background: c }} />
        ))}
      </div>

      {/* Ticks are positioned by where their value falls on the ramp, so a
          compressed scale's labels stay aligned with the colors. */}
      <div className="legend-ticks mono">
        {ticks.map((t, i) => (
          <span
            key={i}
            style={{ left: `${scalePosition(t, domain, metric) * 100}%` }}
          >
            {metricTick(t, metric)}
          </span>
        ))}
      </div>

      <div className="legend-keys">
        <span className="legend-key">
          <span className="legend-swatch" style={{ background: NO_DATA_COLOR }} />
          No data
          {noDataCount > 0 && (
            <span className="legend-key-count mono">
              {noDataCount.toLocaleString()}
            </span>
          )}
        </span>
        <span className="legend-key-note">
          counties that did not report this crop
        </span>
      </div>
    </div>
  );
}
