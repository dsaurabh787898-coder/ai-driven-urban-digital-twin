"""
Stage 1b - Ward adjacency and a coarse flow topology for the flood module.

WHY THIS FILE EXISTS
--------------------
Heat is a local property: a ward is hot because of what is inside it, so zonal
statistics over a ward polygon are a fair summary. Flooding is not local. A ward
floods because of what is upstream of it, and water crosses ward boundaries in
minutes. Any flood module built on this project's ward units therefore needs one
thing the heat module never needed: who drains into whom.

WHAT THIS PRODUCES, AND WHAT IT IS NOT
--------------------------------------
It produces a directed graph over the 198 ward polygons, edges pointing from
higher mean elevation to lower mean elevation across a shared boundary, weighted
by the elevation drop. That is a cascade topology, not a hydrological one:

  - Real flow direction comes from a DEM at 10-30 m, not from the mean elevation
    of a 3 km2 polygon. A ward whose mean sits high can still contain the lowest
    point in its neighbourhood.
  - Real urban routing follows the storm drain network and the rajakaluve
    channels, which do not respect ward boundaries and in Bengaluru frequently
    run against the apparent surface gradient because they are culverted.
  - No channel capacity, no travel time, no backwater effects.

Use it for "which wards accumulate water from their neighbours", which it answers
defensibly. Do not use it for "how deep will Koramangala get", which needs a
hydrodynamic model on a real DEM.

    python pipeline/flood_topology.py --wards data/bbmp_wards.geojson \
        --ward-field WARD_NAME            # -> data/ward_topology.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ward_geo

QUANT = 5          # decimal degrees of rounding for vertex matching (~1 m)
MIN_SHARED = 2     # shared vertices needed to call two wards adjacent
PROX_M = 120.0     # fallback: boundary vertices this close also count


def _quant(x, y):
    return (round(x, QUANT), round(y, QUANT))


def adjacency_by_vertex(wards):
    """Wards that share at least MIN_SHARED identical boundary vertices.

    Works when the source polygons were cut from a single coverage, which is the
    usual case for an administrative layer. Returns {ward: set(ward)}.
    """
    owners = defaultdict(set)
    for name, w in wards.items():
        for rings in w["polys"]:
            for ring in rings:
                for v in ring:
                    owners[_quant(v[0], v[1])].add(name)
    shared = defaultdict(int)
    for names in owners.values():
        if len(names) < 2:
            continue
        ns = sorted(names)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                shared[(ns[i], ns[j])] += 1
    adj = defaultdict(set)
    for (a, b), n in shared.items():
        if n >= MIN_SHARED:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def adjacency_by_proximity(wards, adj, lat0):
    """Fill in wards the vertex test missed, by looking for boundary vertices
    within PROX_M of each other. Quadratic over a coarse spatial hash, so it is
    only run for wards that came out with too few neighbours."""
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110574.0
    cell = PROX_M * 2

    pts = {}
    for name, w in wards.items():
        p = []
        for rings in w["polys"]:
            for ring in rings:
                p.extend((v[0] * kx, v[1] * ky) for v in ring[::2])
        pts[name] = p

    grid = defaultdict(set)
    for name, p in pts.items():
        for x, y in p:
            grid[(int(x // cell), int(y // cell))].add(name)

    lonely = [n for n in wards if len(adj.get(n, ())) < 2]
    added = 0
    for name in lonely:
        near = set()
        for x, y in pts[name]:
            gx, gy = int(x // cell), int(y // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    near |= grid.get((gx + dx, gy + dy), set())
        near.discard(name)
        for other in near:
            if other in adj[name]:
                continue
            hit = any((x1 - x2) ** 2 + (y1 - y2) ** 2 < PROX_M ** 2
                      for x1, y1 in pts[name] for x2, y2 in pts[other])
            if hit:
                adj[name].add(other)
                adj[other].add(name)
                added += 1
    return added


def build(wards_path, ward_field, cells_csv, out_path):
    wards = ward_geo.load_wards(wards_path, ward_field)

    import pandas as pd
    df = pd.read_csv(cells_csv)
    missing = sorted(set(df.ward) - set(wards))
    if missing:
        raise SystemExit(f"{len(missing)} wards in {cells_csv} are absent from "
                         f"{wards_path}, e.g. {missing[:3]}")
    names = sorted(df.ward.unique())
    idx = {n: i for i, n in enumerate(names)}

    lat0 = float(df.lat.mean())
    adj = adjacency_by_vertex(wards)
    for n in names:
        adj.setdefault(n, set())
    added = adjacency_by_proximity(wards, adj, lat0)

    elev = df.set_index("ward").elevation.to_dict()
    area = df.set_index("ward").area_km2.to_dict()

    # Directed edges high -> low, weighted by drop. A ward with no lower
    # neighbour is a terminal sink and retains whatever reaches it.
    edges = []
    for a in names:
        for b in sorted(adj[a]):
            if b not in idx or a == b:
                continue
            drop = elev[a] - elev[b]
            if drop > 0:
                edges.append([idx[a], idx[b], round(drop, 3)])

    outdeg = defaultdict(int)
    for u, _, _ in edges:
        outdeg[u] += 1
    sinks = [n for n in names if outdeg[idx[n]] == 0]

    degrees = [len(adj[n]) for n in names]
    topo = {
        "unit": "ward",
        "wards": names,
        "elevation": [round(float(elev[n]), 2) for n in names],
        "area_km2": [round(float(area[n]), 4) for n in names],
        "neighbours": [[idx[b] for b in sorted(adj[n]) if b in idx] for n in names],
        "edges": edges,
        "sinks": [idx[n] for n in sinks],
        "method": {
            "adjacency": f"shared boundary vertices (>= {MIN_SHARED} at {QUANT} dp) "
                         f"plus a {PROX_M:.0f} m proximity fallback for "
                         f"{added} additional pairs",
            "direction": "mean ward elevation, high -> low; weight = elevation drop (m)",
            "caveat": "Cascade topology over administrative polygons. Not a DEM flow "
                      "network and not the storm drain network. Ward mean elevation "
                      "cannot see the low point inside a ward, and culverted "
                      "rajakaluves in Bengaluru frequently run against the apparent "
                      "surface gradient.",
        },
        "stats": {
            "n_wards": len(names),
            "n_edges": len(edges),
            "mean_neighbours": round(sum(degrees) / len(degrees), 2),
            "min_neighbours": min(degrees),
            "isolated": [n for n in names if not adj[n]],
            "n_sinks": len(sinks),
            "elevation_range_m": [round(min(elev.values()), 1), round(max(elev.values()), 1)],
        },
    }
    json.dump(topo, open(out_path, "w"), separators=(",", ":"))
    return topo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wards", default="data/bbmp_wards.geojson")
    ap.add_argument("--ward-field", default="WARD_NAME")
    ap.add_argument("--cells", default="data/grid_cells.csv")
    ap.add_argument("--out", default="data/ward_topology.json")
    a = ap.parse_args()

    t = build(a.wards, a.ward_field, a.cells, a.out)
    s = t["stats"]
    print(f"wrote {a.out}")
    print(f"  wards            {s['n_wards']}")
    print(f"  mean neighbours  {s['mean_neighbours']}  (min {s['min_neighbours']})")
    print(f"  directed edges   {s['n_edges']}")
    print(f"  terminal sinks   {s['n_sinks']}")
    print(f"  elevation range  {s['elevation_range_m'][0]}-{s['elevation_range_m'][1]} m")
    if s["isolated"]:
        print(f"  ISOLATED (no neighbour found): {s['isolated']}")
