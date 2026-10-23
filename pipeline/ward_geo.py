"""
Ward-polygon helpers for stage 3 and the repair script.

Pure Python (json + math only) so they run in any environment. Everything is
in lon/lat degrees unless a function says km.
"""

import json
import math


# --------------------------------------------------------------------------- #
#  loading                                                                     #
# --------------------------------------------------------------------------- #
def load_wards(path, ward_field):
    """Return {ward_name: {"polys": [[ring, hole, ...], ...], "props": {...}}}.

    Each polygon is a list of rings; ring 0 is the exterior. Coordinates are
    [lon, lat]. GeoJSON produced by geopandas .to_crs(4326) or CRS84 both work."""
    gj = json.load(open(path, encoding="utf-8"))
    out = {}
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        if g["type"] == "MultiPolygon":
            polys = g["coordinates"]
        elif g["type"] == "Polygon":
            polys = [g["coordinates"]]
        else:
            continue
        name = str(f["properties"].get(ward_field))
        e = out.setdefault(name, {"polys": [], "props": f["properties"]})
        e["polys"].extend(polys)
    if not out:
        raise SystemExit(f"no polygons found in {path}")
    sx, sy = next(iter(out.values()))["polys"][0][0][0][:2]
    if not (60 < sx < 100 and 0 < sy < 40):
        raise SystemExit(f"{path} does not look like lon/lat degrees (first vertex {sx}, {sy}). "
                         "Re-export it in EPSG:4326.")
    return out


# --------------------------------------------------------------------------- #
#  area                                                                        #
# --------------------------------------------------------------------------- #
def _ring_area_km2(ring, lat0):
    kx, ky = 111.320 * math.cos(math.radians(lat0)), 110.574
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        a += (x1 * kx) * (y2 * ky) - (x2 * kx) * (y1 * ky)
    return abs(a) / 2.0


def area_km2(polys):
    """Equirectangular shoelace area (within ~0.5 % of a projected area at this
    latitude and size). Holes are subtracted."""
    lat0 = sum(p[0][0][1] for p in polys) / len(polys)
    tot = 0.0
    for rings in polys:
        tot += _ring_area_km2([tuple(v[:2]) for v in rings[0]], lat0)
        for h in rings[1:]:
            tot -= _ring_area_km2([tuple(v[:2]) for v in h], lat0)
    return tot


# --------------------------------------------------------------------------- #
#  simplification (Douglas-Peucker, iterative, closed-ring aware)              #
# --------------------------------------------------------------------------- #
def _dp_open(pts, tol):
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    t2 = tol * tol
    while stack:
        a, b = stack.pop()
        ax, ay = pts[a]
        bx, by = pts[b]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        best, bi = -1.0, -1
        for i in range(a + 1, b):
            px, py = pts[i]
            if L2 == 0:
                d2 = (px - ax) ** 2 + (py - ay) ** 2
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                d2 = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2
            if d2 > best:
                best, bi = d2, i
        if best > t2:
            keep[bi] = True
            stack.append((a, bi))
            stack.append((bi, b))
    return [p for p, k in zip(pts, keep) if k]


def simplify_ring(ring, tol):
    """Simplify a closed ring. Returns an open list of (x, y) (closing vertex
    dropped) or None if it collapses below a triangle."""
    pts = [tuple(v[:2]) for v in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 4:
        return pts if len(pts) == 3 else None
    x0, y0 = pts[0]
    k = max(range(len(pts)), key=lambda i: (pts[i][0] - x0) ** 2 + (pts[i][1] - y0) ** 2)
    a = _dp_open(pts[:k + 1], tol)
    b = _dp_open(pts[k:] + [pts[0]], tol)
    out = a[:-1] + b[:-1]
    return out if len(out) >= 3 else None


def _signed_area(r):
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1])) / 2.0


# --------------------------------------------------------------------------- #
#  point in polygon / zone borders                                             #
# --------------------------------------------------------------------------- #
def _pip(x, y, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def zone_segments(rings_by_ward, zone_of, eps_list=(0.0006, 0.0015)):
    """Line segments (lon1, lat1, lon2, lat2) that separate wards of different
    zones, or a ward from the outside. rings_by_ward: {name: [exterior rings]}.
    Works on the simplified rings, so it does not rely on the source polygons
    sharing vertices."""
    names = list(rings_by_ward)
    bbox = {}
    for n in names:
        xs = [p[0] for r in rings_by_ward[n] for p in r]
        ys = [p[1] for r in rings_by_ward[n] for p in r]
        bbox[n] = (min(xs), min(ys), max(xs), max(ys))

    def owner(x, y):
        for n in names:
            b = bbox[n]
            if x < b[0] or x > b[2] or y < b[1] or y > b[3]:
                continue
            if any(_pip(x, y, r) for r in rings_by_ward[n]):
                return n
        return None

    segs = []
    for i, n in enumerate(names):
        for r in rings_by_ward[n]:
            ccw = _signed_area(r) > 0
            for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1]):
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)
                if L < 1e-7:
                    continue
                nx, ny = (dy / L, -dx / L) if ccw else (-dy / L, dx / L)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                other, found = None, False
                for eps in eps_list:
                    o = owner(mx + nx * eps, my + ny * eps)
                    if o is not None:
                        other, found = o, True
                        break
                if found and other == n:
                    continue                       # concave notch of the same ward
                if found and zone_of.get(other) == zone_of.get(n):
                    continue                       # interior ward boundary
                if found and names.index(other) < i:
                    continue                       # the neighbour emits this border
                segs.append((x1, y1, x2, y2))
    return segs
