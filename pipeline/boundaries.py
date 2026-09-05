"""US county/state boundary geometry — pure stdlib, no geo dependencies.

The project deliberately avoids geopandas/shapely: those pull GDAL/GEOS native
libraries into an environment that already has a documented libomp conflict
between lightgbm and torch (see README). Census cartographic boundary files are
simple enough to read directly, so this module parses the ESRI shapefile and DBF
formats with the standard library and caches the result as a .npz.

Source: Census cartographic boundary files, 20m (least detailed) resolution —
the same Census lineage as data/raw/county_centroids.parquet.
"""
from __future__ import annotations

import io
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BASE = "https://www2.census.gov/geo/tiger/GENZ2021/shp"
LAYERS = {
    "county": ("cb_2021_us_county_20m", "GEOID"),
    "state":  ("cb_2021_us_state_20m",  "STUSPS"),
}


def _read_dbf(buf: bytes) -> list[dict[str, str]]:
    """Minimal dBASE III reader — enough for Census attribute tables."""
    n_recs, hdr_len, rec_len = struct.unpack("<IHH", buf[4:12])
    fields = []
    pos = 32
    while buf[pos] != 0x0D:
        name = buf[pos:pos + 11].split(b"\0")[0].decode("latin-1")
        size = buf[pos + 16]
        fields.append((name, size))
        pos += 32
    rows = []
    pos = hdr_len
    for _ in range(n_recs):
        off = pos + 1                      # skip deletion flag
        row = {}
        for name, size in fields:
            row[name] = buf[off:off + size].decode("latin-1").strip()
            off += size
        rows.append(row)
        pos += rec_len
    return rows


def _read_shp(buf: bytes) -> list[list[np.ndarray]]:
    """Read polygon shapes. Returns one list of [n,2] rings per record."""
    shapes: list[list[np.ndarray]] = []
    pos = 100                                          # file header
    end = len(buf)
    while pos < end:
        _, clen = struct.unpack(">II", buf[pos:pos + 8])
        rec = buf[pos + 8: pos + 8 + clen * 2]
        pos += 8 + clen * 2
        (shp_type,) = struct.unpack("<I", rec[:4])
        if shp_type != 5:                              # 5 = Polygon
            shapes.append([])
            continue
        n_parts, n_pts = struct.unpack("<II", rec[36:44])
        parts = struct.unpack(f"<{n_parts}I", rec[44:44 + 4 * n_parts])
        pts_off = 44 + 4 * n_parts
        pts = np.frombuffer(rec, dtype="<f8", count=n_pts * 2,
                            offset=pts_off).reshape(-1, 2)
        bounds = list(parts) + [n_pts]
        shapes.append([pts[bounds[i]:bounds[i + 1]] for i in range(n_parts)])
    return shapes


def load(layer: str, cache_dir: Path) -> dict[str, list[np.ndarray]]:
    """Return {key: [rings]} for 'county' (keyed by 5-digit FIPS) or 'state'.

    Downloads and caches on first use; afterwards reads the local .npz.
    """
    name, key_field = LAYERS[layer]
    cache = Path(cache_dir) / f"{layer}_boundaries.npz"

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: list(v) for k, v in zip(z["keys"], z["rings"])}

    cache.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{BASE}/{name}.zip", timeout=120) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    shapes = _read_shp(zf.read(f"{name}.shp"))
    attrs = _read_dbf(zf.read(f"{name}.dbf"))

    out = {a[key_field]: s for a, s in zip(attrs, shapes) if s}
    np.savez_compressed(
        cache,
        keys=np.array(list(out.keys())),
        rings=np.array([np.array(v, dtype=object) for v in out.values()],
                       dtype=object),
    )
    return out
