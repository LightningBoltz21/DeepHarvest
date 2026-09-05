// County/state geometry for the map, from the us-atlas package.
//
// us-atlas is the maintained TopoJSON build of the Census cartographic
// boundary files -- the same source as data/raw/*_boundaries.npz, but with
// polygon topology (shared borders, and holes for the counties that contain
// an independent city) already correct. Deriving the geometry here rather
// than from the .npz keeps that correctness out of this project's hands.
//
// The output is filtered to the counties that actually appear in the crop
// data, which is what drops Alaska, Hawaii and the territories: NASS county
// yields do not cover them.
//
// Usage: node scripts/build-topo.mjs

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const DATA = resolve(here, "../public/data");

mkdirSync(DATA, { recursive: true });

const meta = JSON.parse(readFileSync(resolve(DATA, "meta.json"), "utf8"));
const keep = new Set();
for (const crop of meta.crops) {
  const rows = JSON.parse(readFileSync(resolve(DATA, `${crop}.json`), "utf8"));
  for (const fips of Object.keys(rows)) keep.add(fips);
}

const atlas = JSON.parse(
  readFileSync(require.resolve("us-atlas/counties-10m.json"), "utf8"),
);

const counties = atlas.objects.counties.geometries.filter((g) => keep.has(g.id));
const statePrefixes = new Set([...keep].map((f) => f.slice(0, 2)));
const states = atlas.objects.states.geometries.filter((g) => statePrefixes.has(g.id));

// Both layers share the parent topology's arcs, so they are written as one
// file: splitting them would duplicate every coastline and state border.
const topo = {
  type: "Topology",
  bbox: atlas.bbox,
  transform: atlas.transform,
  arcs: atlas.arcs,
  objects: {
    counties: { type: "GeometryCollection", geometries: counties },
    states: { type: "GeometryCollection", geometries: states },
  },
};

const out = resolve(DATA, "us.topojson");
writeFileSync(out, JSON.stringify(topo));
console.log(`us.topojson: ${counties.length} counties, ${states.length} states`);
