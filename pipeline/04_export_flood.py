"""
Stage 4 - Bundle everything the flood simulator needs into one JSON.

Reuses the ward geometry already simplified by stage 3 rather than re-simplifying
it, so the flood map and the heat map are pixel-identical and a ward index means
the same thing in both.

    python pipeline/04_export_flood.py        # -> web/flood_bundle.json
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flood_model


def build(bundle_path, topo_path, cells_path, out_path):
    bundle = json.load(open(bundle_path))
    topo = json.load(open(topo_path))
    df = pd.read_csv(cells_path)

    bw, tw = bundle["wards"], topo["wards"]
    if bw != tw:
        raise SystemExit(
            "ward order differs between the stage-3 bundle and the topology; "
            "the geometry would be attached to the wrong wards. "
            f"first mismatch: {next(a for a, b in zip(bw, tw) if a != b)}")

    by_ward = df.set_index("ward")
    need = ["impervious", "tree_canopy", "water", "builtup", "ndvi", "elevation",
            "area_km2", "population", "pop_density", "vulnerability", "lst"]
    units = [{k: float(by_ward.loc[w, k]) for k in need} for w in tw]

    for u in units:
        u["cn"] = round(flood_model.curve_number(
            u["impervious"], u["tree_canopy"], u["water"]), 1)

    zone_of = by_ward.zone.to_dict()
    zones = sorted(set(zone_of.values()))

    out = {
        "meta": {
            "city": bundle["meta"].get("city"),
            "unit": "ward",
            "n_wards": len(tw),
            "mode": bundle["meta"].get("mode"),
            "provenance": bundle["meta"].get("provenance"),
            "built_from": os.path.basename(bundle_path),
        },
        "wards": tw,
        "zones": zones,
        "zoneOf": [zones.index(zone_of[w]) for w in tw],
        "wardGeo": bundle["wardGeo"],
        "zoneSegs": bundle["zoneSegs"],
        "units": [{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in u.items()} for u in units],
        "topology": {
            "elevation": topo["elevation"],
            "area_km2": topo["area_km2"],
            "edges": topo["edges"],
            "neighbours": topo["neighbours"],
            "sinks": topo["sinks"],
            "method": topo["method"],
            "stats": topo["stats"],
        },
        "floodConfig": flood_model.FLOOD_CONFIG,
    }
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="web/bundle.json")
    ap.add_argument("--topology", default="data/ward_topology.json")
    ap.add_argument("--cells", default="data/grid_cells.csv")
    ap.add_argument("--out", default="web/flood_bundle.json")
    a = ap.parse_args()
    o = build(a.bundle, a.topology, a.cells, a.out)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")
    print(f"  wards {o['meta']['n_wards']}  edges {len(o['topology']['edges'])}  "
          f"sinks {len(o['topology']['sinks'])}")
