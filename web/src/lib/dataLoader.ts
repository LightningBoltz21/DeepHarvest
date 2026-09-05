/* Loading of the static data files.
 *
 * All fetches are same-origin requests for files in public/data. Crop data is
 * cached after first load so switching back to a crop is instant. */

import { feature } from "topojson-client";
import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";
import type { Topology } from "topojson-specification";
import type { CropData, Meta } from "./types";

const BASE = `${import.meta.env.BASE_URL}data`;

/** Feature ids are FIPS codes: 5 digits for counties, 2 for states. */
export type CountyFeatures = FeatureCollection<Polygon | MultiPolygon>;
export type StateFeatures = FeatureCollection<Polygon | MultiPolygon>;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}/${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export function loadMeta(): Promise<Meta> {
  return getJSON<Meta>("meta.json");
}

const cropCache = new Map<string, Promise<CropData>>();

export function loadCrop(crop: string): Promise<CropData> {
  let pending = cropCache.get(crop);
  if (!pending) {
    pending = getJSON<CropData>(`${crop}.json`);
    cropCache.set(crop, pending);
  }
  return pending;
}

export async function loadGeography(): Promise<{
  counties: CountyFeatures;
  states: StateFeatures;
}> {
  // One file: the two layers share the same topology arcs.
  const topo = await getJSON<Topology>("us.topojson");
  return {
    counties: feature(topo, topo.objects.counties) as CountyFeatures,
    states: feature(topo, topo.objects.states) as StateFeatures,
  };
}
