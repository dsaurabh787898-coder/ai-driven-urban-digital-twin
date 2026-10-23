"""
Stage 5 - Real flood labels from the BBMP / KSRSAC point layers on OpenCity.

Source: https://data.opencity.in/dataset/flooding-locations-in-bengaluru-urban
Licence: Other (Public Domain). Three KML point layers:

    flood_vulnerable_points   "Map of Locations Vulnerable to Flooding" (has BBMP
                              ward name / number / zone for each point)
    flood_prone_locations     "Flood Prone Locations Map"
    lowlying_locations        "BBMP Low Lying Areas"

WHAT THESE ARE, AND WHAT THEY ARE NOT
-------------------------------------
They are an inventory of places BBMP has flagged. They carry no date, no rainfall
and no depth, and they list only places that were flagged, never places confirmed
dry. So they are PRESENCE-ONLY labels:

  - a ward with a listed point is evidence that the ward has a flooding problem
  - a ward with no listed point is NOT evidence that it does not (it may simply
    never have been surveyed, or the point may sit in a neighbour)

and they cannot teach a model how a given amount of rain turns into flooding,
because no rainfall is attached to any point. The label is therefore
"BBMP-flagged ward", and everything trained on it should be described that way.

    python pipeline/05_flood_labels.py
        -> data/flood_labels.csv      per-ward counts per layer
        -> data/flood_points.json     every point with its ward, for the map overlay
"""

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ward_geo

NS = {"k": "http://www.opengis.net/kml/2.2"}

LAYERS = {
    "vulnerable": "data/raw/flood_vulnerable_points.kml",
    "prone": "data/raw/flood_prone_locations.kml",
    "lowlying": "data/raw/lowlying_locations.kml",
}


def parse_kml(path):
    """Yield dicts: name, lon, lat (None when the file stores nan), fields."""
    root = ET.parse(path).getroot()
    for pm in root.iter("{%s}Placemark" % NS["k"]):
        name = (pm.findtext("k:name", default="", namespaces=NS) or "").strip()
        fields = {sd.get("name"): (sd.text or "").strip() for sd in pm.iter("{%s}SimpleData" % NS["k"])}
        c = pm.findtext(".//k:Point/k:coordinates", default="", namespaces=NS).strip()
        lon = lat = None
        try:
            a, b = (float(v) for v in c.split(",")[:2])
            if a == a and b == b:                     # nan != nan
                lon, lat = a, b
        except ValueError:
            pass
        yield {"name": name or fields.get("LocationName", ""), "lon": lon, "lat": lat, "fields": fields}


def _pip(x, y, ring):
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class WardIndex:
    def __init__(self, wards):
        self.names = list(wards)
        self.polys = {n: wards[n]["polys"] for n in self.names}
        self.bbox = {}
        for n in self.names:
            xs = [v[0] for p in self.polys[n] for r in p for v in r]
            ys = [v[1] for p in self.polys[n] for r in p for v in r]
            self.bbox[n] = (min(xs), min(ys), max(xs), max(ys))

    def locate(self, x, y):
        hits = []
        for n in self.names:
            b = self.bbox[n]
            if x < b[0] or x > b[2] or y < b[1] or y > b[3]:
                continue
            for poly in self.polys[n]:
                inside = _pip(x, y, poly[0]) and not any(_pip(x, y, h) for h in poly[1:])
                if inside:
                    hits.append(n)
                    break
        return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wards", default="data/bbmp_wards.geojson")
    ap.add_argument("--ward-field", default="WARD_NAME")
    ap.add_argument("--cells", default="data/grid_cells.csv")
    ap.add_argument("--out", default="data/flood_labels.csv")
    ap.add_argument("--points-out", default="data/flood_points.json")
    a = ap.parse_args()

    wards = ward_geo.load_wards(a.wards, a.ward_field)
    idx = WardIndex(wards)
    model_wards = sorted({r["ward"] for r in csv.DictReader(open(a.cells))})
    unknown = [w for w in model_wards if w not in wards]
    if unknown:
        raise SystemExit(f"{len(unknown)} wards in {a.cells} are not in {a.wards}, e.g. {unknown[:3]}")

    counts = {w: defaultdict(int) for w in model_wards}
    points, report = [], {}
    # BBMP's own ward number -> ward name, for validating the join and as a fallback
    num2name = {}
    for n, w in wards.items():
        v = w["props"].get("WARD_NO")
        if v not in (None, ""):
            num2name[int(float(v))] = n
    for layer, path in LAYERS.items():
        if not os.path.exists(path):
            print(f"[skip] {layer}: {path} not found")
            continue
        n = bad = outside = dup = rescued = 0
        seen = set()
        agree = checked = 0
        for p in parse_kml(path):
            n += 1
            if p["lon"] is None:
                bad += 1
                continue
            key = (round(p["lon"], 5), round(p["lat"], 5))
            if key in seen:
                dup += 1                               # same coordinate listed twice
                continue
            seen.add(key)
            w = idx.locate(p["lon"], p["lat"])
            if w is None and p["fields"].get("WARDNO"):
                # point sits just outside every polygon (boundary / digitising gap);
                # fall back to the ward number BBMP recorded for it
                w = num2name.get(int(float(p["fields"]["WARDNO"])))
                rescued += int(w is not None)
            if w is None:
                outside += 1
                continue
            counts[w][layer] += 1
            points.append({"layer": layer, "name": p["name"], "lon": round(p["lon"], 5),
                           "lat": round(p["lat"], 5), "ward": w})
            if p["fields"].get("WARDNO"):
                # compare by ward NUMBER: names differ in spelling between the two
                # layers (Chamrajpet / Chamarajpet, Binnypete / Binnipete)
                checked += 1
                agree += int(num2name.get(int(float(p["fields"]["WARDNO"]))) == w)
        report[layer] = dict(placemarks=n, no_coordinates=bad, duplicate_coordinates=dup,
                             outside_all_wards=outside, rescued_by_ward_no=rescued,
                             kept=n - bad - dup - outside,
                             ward_name_agreement=(agree, checked) if checked else None)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ward"] + [f"n_{k}" for k in LAYERS])
        for name in model_wards:
            w.writerow([name] + [counts[name].get(k, 0) for k in LAYERS])
    json.dump(points, open(a.points_out, "w"), separators=(",", ":"))

    print(f"wrote {a.out} and {a.points_out}\n")
    for layer, r in report.items():
        print(f"{layer:11} placemarks {r['placemarks']:4d}  no-coords {r['no_coordinates']}  "
              f"dup-coords {r['duplicate_coordinates']}  outside {r['outside_all_wards']}  "
              f"(rescued via ward no. {r['rescued_by_ward_no']})  kept {r['kept']}")
        if r["ward_name_agreement"]:
            ag, ch = r["ward_name_agreement"]
            print(f"{'':11} BBMP ward number agrees with my point-in-polygon join: {ag}/{ch} ({100*ag/ch:.1f}%)")
    print()
    for k in LAYERS:
        nz = sum(1 for wd in model_wards if counts[wd].get(k, 0) > 0)
        print(f"wards with >=1 {k:10} point: {nz:3d} / {len(model_wards)}")
    both = sum(1 for wd in model_wards if counts[wd].get("vulnerable", 0) + counts[wd].get("prone", 0) > 0)
    print(f"wards with >=1 vulnerable-or-prone point: {both} / {len(model_wards)}")


if __name__ == "__main__":
    main()
