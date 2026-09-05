/* Hover readout. Follows the cursor, flipping sides near the viewport edge. */

import { useEffect, useState } from "react";
import { acres, bushels, bushelsPerAcre, signedBushels, signedPercent } from "../lib/format";
import { BASELINE_WINDOW } from "../lib/metrics";
import type { CountyValue } from "../lib/types";

interface Props {
  value: CountyValue | null;
  x: number;
  y: number;
  cropLabel: string;
  year: number;
}

const OFFSET = 16;
const WIDTH = 380;

export function Tooltip({ value, x, y, cropLabel, year }: Props) {
  const [flip, setFlip] = useState(false);

  useEffect(() => {
    setFlip(x + OFFSET + WIDTH > window.innerWidth);
  }, [x]);

  if (!value) return null;

  const left = flip ? x - OFFSET - WIDTH : x + OFFSET;
  const top = Math.min(y + OFFSET, window.innerHeight - 240);

  return (
    <div className="tooltip" style={{ left, top, width: WIDTH }} role="tooltip">
      <div className="tooltip-head">
        <strong>{value.county}</strong>
        <span className="mono">{value.state}</span>
      </div>
      <div className="tooltip-sub">
        {cropLabel} · {year}
        {value.isForecast && <span className="tag">Forecast</span>}
      </div>

      <dl className="tooltip-rows">
        <Row
          label={value.isForecast ? "Predicted yield" : "Yield"}
          value={bushelsPerAcre(value.value)}
        />
        <Row
          label={
            value.baseline !== null && value.baselineYears < BASELINE_WINDOW
              ? `Mean of ${value.baselineYears} prior yr`
              : `${BASELINE_WINDOW}-year mean`
          }
          value={
            value.baseline === null
              ? "Insufficient history"
              : bushelsPerAcre(value.baseline)
          }
        />
        {value.baseline !== null && (
          <>
            <Row label="vs. mean" value={signedBushels(value.diff)} trend={value.diff} />
            <Row
              label="vs. mean %"
              value={signedPercent(value.pctDiff)}
              trend={value.pctDiff}
            />
          </>
        )}
        {value.acres !== null && (
          <Row
            label={value.acresEstimated ? "Harvested (est.)" : "Harvested"}
            value={acres(value.acres)}
          />
        )}
        {value.production !== null && (
          <Row
            label={value.acresEstimated ? "Production (est.)" : "Production"}
            value={bushels(value.production)}
          />
        )}
      </dl>

      {value.acresEstimated && (
        <p className="tooltip-note">
          Acreage carried forward from the last reported year; production is an estimate.
        </p>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  trend,
}: {
  label: string;
  value: string;
  /** when present, drives an up/down ticker arrow */
  trend?: number | null;
}) {
  const dir = trend === null || trend === undefined ? 0 : Math.sign(trend);
  return (
    <div className="tooltip-row">
      <dt>{label}</dt>
      <dd className={dir > 0 ? "mono up" : dir < 0 ? "mono down" : "mono"}>
        {dir !== 0 && (
          <span className="arrow" aria-hidden="true">
            {dir > 0 ? "▲" : "▼"}
          </span>
        )}
        {value}
      </dd>
    </div>
  );
}
