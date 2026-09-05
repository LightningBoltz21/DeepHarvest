/* Map color ramps.
 *
 * The UI chrome is a single gold accent, which cannot encode an ordered
 * quantity on its own and has nothing to encode direction, so the map carries
 * its own scales:
 *
 *   yield / production   sequential, per crop -- gold for corn, green for
 *                        soybeans, so the two crops are never confused for
 *                        each other at a glance. Both run dark -> saturated,
 *                        so intensity reads as magnitude in either.
 *
 *   % difference         diverging red -> neutral -> green, the conventional
 *                        "below / above normal" pairing. Lightness carries
 *                        most of the signal (both ends are darker than the
 *                        middle), so the direction survives red-green color
 *                        vision deficiency even where the hues do not.
 *
 * Interpolated in Oklab so equal steps look equally spaced. */

import { scalePosition } from "./metrics";
import type { Metric } from "./types";

type RGB = [number, number, number];

/* Multi-hue sequential ramps, in the manner of viridis/magma: lightness climbs
 * monotonically from bottom to top (so the ordering survives greyscale and
 * color vision deficiency) while the hue travels across the spectrum, which is
 * what makes the map read as colorful rather than as one tinted wash.
 *
 * Each crop ends on its own signature hue -- corn gold, soybeans green -- so
 * the two are still tellable apart at a glance. */
const SEQUENTIAL_BY_CROP: Record<string, RGB[]> = {
  // deep indigo -> violet -> magenta -> orange -> gold
  corn: [
    [22, 26, 58],
    [58, 42, 108],
    [116, 52, 122],
    [173, 63, 100],
    [215, 96, 62],
    [238, 152, 48],
    [248, 206, 92],
    [252, 240, 168],
  ],
  // deep navy -> blue -> teal -> green -> chartreuse
  soybeans: [
    [20, 28, 60],
    [30, 62, 118],
    [28, 105, 130],
    [34, 140, 116],
    [62, 170, 90],
    [124, 196, 72],
    [186, 216, 84],
    [232, 240, 150],
  ],
};

/* Diverging: deep red through a near-neutral zero to deep green, with the
 * shoulders pushed toward magenta and teal so the mid-range values carry hue
 * instead of washing out to grey. Zero stays the lightest point, which is what
 * makes the direction of a departure readable at a glance. */
const DIVERGING: RGB[] = [
  [116, 18, 60],
  [176, 34, 48],
  [216, 92, 62],
  [238, 166, 124],
  [242, 232, 214],
  [150, 206, 150],
  [70, 168, 118],
  [28, 122, 104],
  [16, 78, 88],
];

/** Fill for counties with no value under the current filters. */
export const NO_DATA_COLOR = "#161c26";

function stopsFor(metric: Metric, crop: string): RGB[] {
  if (metric === "pctDiff") return DIVERGING;
  return SEQUENTIAL_BY_CROP[crop] ?? SEQUENTIAL_BY_CROP.corn;
}

export function rampColor(t: number, metric: Metric, crop: string): string {
  const stops = stopsFor(metric, crop);
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  return oklabMix(stops[i], stops[i + 1], scaled - i);
}

export function colorFor(
  value: number | null,
  domain: [number, number] | null,
  metric: Metric,
  crop: string,
): string {
  if (value === null || domain === null || !Number.isFinite(value)) return NO_DATA_COLOR;
  return rampColor(scalePosition(value, domain, metric), metric, crop);
}

/* --- Oklab interpolation ------------------------------------------------- */
/* sRGB mixing muddies the midpoint of a ramp; Oklab is close enough to
 * perceptually uniform that evenly spaced values look evenly spaced. */

function oklabMix(a: RGB, b: RGB, t: number): string {
  const [l1, a1, b1] = rgbToOklab(a);
  const [l2, a2, b2] = rgbToOklab(b);
  return oklabToCss([
    l1 + (l2 - l1) * t,
    a1 + (a2 - a1) * t,
    b1 + (b2 - b1) * t,
  ]);
}

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function linearToSrgb(v: number): number {
  const c = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.round(Math.min(1, Math.max(0, c)) * 255);
}

function rgbToOklab([r, g, b]: RGB): RGB {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);

  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);

  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToCss([L, A, B]: RGB): string {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;

  const r = linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s);
  const g = linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s);
  const b = linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s);
  return `rgb(${r},${g},${b})`;
}
