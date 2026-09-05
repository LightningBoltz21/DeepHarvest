/* County choropleth with zoom and pan.
 *
 * Canvas rather than SVG: ~2,500 county paths as DOM nodes makes filter changes
 * and zooming visibly janky, and none of them need to be individually styled or
 * focusable. Hit testing for hover uses a second, offscreen canvas where each
 * county is filled with a color encoding its index, so a pointer lookup is one
 * getImageData of a single pixel rather than a point-in-polygon scan. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { geoAlbersUsa, geoPath } from "d3-geo";
import type { GeoPath, GeoProjection } from "d3-geo";
import { select } from "d3-selection";
import "d3-transition"; // augments selection.transition(), used by the zoom buttons
import { zoom as d3zoom, zoomIdentity, type ZoomTransform } from "d3-zoom";
import { colorFor, NO_DATA_COLOR } from "../lib/colorScales";
import type { CountyFeatures, StateFeatures } from "../lib/dataLoader";
import type { CountyValue, Metric } from "../lib/types";

interface Props {
  counties: CountyFeatures;
  states: StateFeatures;
  values: Map<string, CountyValue>;
  domain: [number, number] | null;
  metric: Metric;
  crop: string;
  /** two-digit state FIPS prefix to zoom to, or "ALL" */
  stateFilter: string;
  /** FIPS of counties matching the search box; empty means no search active */
  highlighted: Set<string>;
  onHover: (value: CountyValue | null, x: number, y: number) => void;
}

const BORDER = "#0a0e14";
const STATE_BORDER = "#7b8494";
const HIGHLIGHT = "#c8a951";
const MAX_ZOOM = 12;

export function CountyMap({
  counties,
  states,
  values,
  domain,
  metric,
  crop,
  stateFilter,
  highlighted,
  onHover,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pickRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [hoveredFips, setHoveredFips] = useState<string | null>(null);
  const [transform, setTransform] = useState<ZoomTransform>(zoomIdentity);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  /* Fit the projection to whichever geography is in view. Filtering to a state
   * refits the map to that state instead of leaving it national. */
  const projection = useMemo<GeoProjection | null>(() => {
    if (size.width === 0 || size.height === 0) return null;
    const fitTo =
      stateFilter === "ALL"
        ? states
        : {
            type: "FeatureCollection" as const,
            features: states.features.filter((f) => f.id === stateFilter),
          };
    if (fitTo.features.length === 0) return null;
    return geoAlbersUsa().fitExtent(
      [
        [12, 12],
        [size.width - 12, size.height - 12],
      ],
      fitTo,
    );
  }, [size, states, stateFilter]);

  const path = useMemo<GeoPath | null>(
    () => (projection ? geoPath(projection) : null),
    [projection],
  );

  // Changing what the map is fitted to invalidates the current pan/zoom.
  useEffect(() => {
    setTransform(zoomIdentity);
  }, [stateFilter]);

  const drawList = useMemo(() => counties.features, [counties]);

  /* --- zoom behaviour ---------------------------------------------------- */
  const zoomRef = useRef<ReturnType<typeof d3zoom<HTMLCanvasElement, unknown>> | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width === 0) return;

    const behavior = d3zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([1, MAX_ZOOM])
      // Keep the map from being dragged off-screen.
      .translateExtent([
        [0, 0],
        [size.width, size.height],
      ])
      .on("zoom", (event) => setTransform(event.transform));

    zoomRef.current = behavior;
    const sel = select(canvas);
    sel.call(behavior);
    // Double-click-to-zoom fights with rapid hovering on a dense map.
    sel.on("dblclick.zoom", null);

    return () => {
      sel.on(".zoom", null);
    };
  }, [size]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !path || size.width === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const ctxPath = geoPath(path.projection(), ctx);
    const dimmed = stateFilter !== "ALL";
    // Borders are drawn in screen pixels, so they thin out as the map zooms in
    // rather than growing into slabs.
    const px = 1 / transform.k;

    for (const f of drawList) {
      const fips = String(f.id);
      const inState = !dimmed || fips.slice(0, 2) === stateFilter;
      const v = values.get(fips);
      ctx.beginPath();
      ctxPath(f);
      ctx.fillStyle = v ? colorFor(v.metricValue, domain, metric, crop) : NO_DATA_COLOR;
      ctx.globalAlpha = inState ? 1 : 0.18;
      ctx.fill();
      ctx.lineWidth = 0.3 * px;
      ctx.strokeStyle = BORDER;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    ctx.beginPath();
    for (const f of states.features) ctxPath(f);
    ctx.lineWidth = 0.7 * px;
    ctx.strokeStyle = STATE_BORDER;
    ctx.stroke();

    if (highlighted.size > 0) {
      ctx.beginPath();
      for (const f of drawList) {
        if (highlighted.has(String(f.id))) ctxPath(f);
      }
      ctx.lineWidth = 1.6 * px;
      ctx.strokeStyle = HIGHLIGHT;
      ctx.stroke();
    }

    if (hoveredFips) {
      const f = drawList.find((d) => String(d.id) === hoveredFips);
      if (f) {
        ctx.beginPath();
        ctxPath(f);
        ctx.lineWidth = 1.4 * px;
        ctx.strokeStyle = "#edf1f6";
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [
    path,
    size,
    drawList,
    values,
    domain,
    metric,
    crop,
    states,
    stateFilter,
    highlighted,
    hoveredFips,
    transform,
  ]);

  useEffect(draw, [draw]);

  /* The pick canvas holds identity, not appearance, so it is rebuilt only when
   * the geometry or viewport changes -- never on a filter or zoom change. The
   * pointer position is mapped back through the zoom transform instead. */
  useEffect(() => {
    if (!path || size.width === 0) return;
    const pick = document.createElement("canvas");
    pick.width = size.width;
    pick.height = size.height;
    const ctx = pick.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const pickPath = geoPath(path.projection(), ctx);
    drawList.forEach((f, i) => {
      ctx.beginPath();
      pickPath(f);
      ctx.fillStyle = indexToColor(i + 1);
      ctx.fill();
      // Stroke too, so hairline-thin counties stay hoverable.
      ctx.strokeStyle = indexToColor(i + 1);
      ctx.lineWidth = 1;
      ctx.stroke();
    });
    pickRef.current = pick;
  }, [path, size, drawList]);

  const handleMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const pick = pickRef.current;
      const canvas = canvasRef.current;
      if (!pick || !canvas) return;
      const rect = canvas.getBoundingClientRect();
      const [x, y] = transform.invert([e.clientX - rect.left, e.clientY - rect.top]);
      if (x < 0 || y < 0 || x >= pick.width || y >= pick.height) {
        setHoveredFips(null);
        onHover(null, 0, 0);
        return;
      }
      const ctx = pick.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;
      const [r, g, b] = ctx.getImageData(Math.round(x), Math.round(y), 1, 1).data;
      const index = colorToIndex(r, g, b) - 1;
      const f = index >= 0 ? drawList[index] : undefined;
      const fips = f ? String(f.id) : null;
      setHoveredFips(fips);
      onHover(fips ? (values.get(fips) ?? null) : null, e.clientX, e.clientY);
    },
    [drawList, values, onHover, transform],
  );

  const handleLeave = useCallback(() => {
    setHoveredFips(null);
    onHover(null, 0, 0);
  }, [onHover]);

  const zoomBy = useCallback((factor: number) => {
    const canvas = canvasRef.current;
    const behavior = zoomRef.current;
    if (!canvas || !behavior) return;
    behavior.scaleBy(select(canvas).transition().duration(180) as never, factor);
  }, []);

  const resetZoom = useCallback(() => {
    const canvas = canvasRef.current;
    const behavior = zoomRef.current;
    if (!canvas || !behavior) return;
    behavior.transform(select(canvas).transition().duration(200) as never, zoomIdentity);
  }, []);

  const zoomed = transform.k !== 1 || transform.x !== 0 || transform.y !== 0;

  return (
    <div className="map-wrap" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height }}
        onPointerMove={handleMove}
        onPointerLeave={handleLeave}
      />
      <div className="map-controls">
        <button type="button" onClick={() => zoomBy(1.6)} aria-label="Zoom in">
          +
        </button>
        <button type="button" onClick={() => zoomBy(1 / 1.6)} aria-label="Zoom out">
          −
        </button>
        <button
          type="button"
          className="map-reset"
          onClick={resetZoom}
          disabled={!zoomed}
        >
          Reset
        </button>
      </div>
    </div>
  );
}

function indexToColor(i: number): string {
  return `rgb(${(i >> 16) & 255},${(i >> 8) & 255},${i & 255})`;
}

function colorToIndex(r: number, g: number, b: number): number {
  return (r << 16) | (g << 8) | b;
}
