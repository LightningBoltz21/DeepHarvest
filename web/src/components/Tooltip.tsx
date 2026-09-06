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
/** Keeps the card off the very edge of the viewport. */
const MARGIN = 8;

export function Tooltip({ value, x, y, cropLabel, year }: Props) {
  const [box, setBox] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const measure = () => setBox({ w: window.innerWidth, h: window.innerHeight });
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  if (!value) return null;

  /* Prefer the right of the pointer, flip to the left when that would overflow,
   * then clamp: on a narrow screen the card is wider than the space on either
   * side, so flipping alone would push it off the opposite edge. */
  const width = Math.min(WIDTH, box.w - MARGIN * 2);
  const fitsRight = x + OFFSET + width <= box.w - MARGIN;
  const rawLeft = fitsRight ? x + OFFSET : x - OFFSET - width;
  const left = Math.max(MARGIN, Math.min(rawLeft, box.w - width - MARGIN));

  /* Below the pointer normally; above it when the card would run off the
   * bottom, so a tap near the foot of a phone screen stays readable. */
  const estHeight = 260;
  const below = y + OFFSET;
  const top =
    below + estHeight > box.h - MARGIN
      ? Math.max(MARGIN, y - OFFSET - estHeight)
      : below;

  return (
    <div className="tooltip" style={{ left, top, width }} role="tooltip">
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
