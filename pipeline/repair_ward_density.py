"""
One-off repair for a ward-polygon grid_cells.csv produced before the density fix.

Stage 01 used to compute  pop_density = population / CELL_KM**2  even when the
units were wards, so 'pop_density' was the ward's TOTAL population and the
vulnerability proxy (which uses it) ranked wards by size, not by crowding.

This script adds area_km2 (from the ward polygons), recomputes pop_density as
people per km2, and recomputes vulnerability with the same formula as stage 01.
It does NOT touch any satellite-derived column, so there is no need to re-run
Earth Engine.

    python pipeline/repair_ward_density.py --wards data/bbmp_wards.geojson --ward-field WARD_NAME
"""

import argparse
import numpy as np
import pandas as pd
import ward_geo


def _minmax(s):
    s = np.asarray(s, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo + 1e-9)


def vulnerability(df, density):
    return (0.45 * _minmax(np.log1p(density))
            + 0.35 * (1 - _minmax(df.ndvi))
            + 0.20 * _minmax(df.impervious)).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wards", required=True)
    ap.add_argument("--ward-field", default="WARD_NAME")
    ap.add_argument("--csv", default="data/grid_cells.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    wards = ward_geo.load_wards(a.wards, a.ward_field)
    missing = sorted(set(df.ward) - set(wards))
    if missing:
        raise SystemExit(f"{len(missing)} wards in the CSV are not in {a.wards}, e.g. {missing[:3]}")

    # Sanity: reproducing the OLD vulnerability from the OLD density must match
    # the file, otherwise the formula here has drifted from stage 01.
    drift = np.abs(vulnerability(df, df.population) - df.vulnerability).max()
    if "area_km2" not in df.columns and drift > 2e-3:
        raise SystemExit(f"vulnerability formula does not reproduce the file (max diff {drift:.4f}); "
                         "stage 01 changed - not repairing blindly.")

    df["area_km2"] = df.ward.map({n: ward_geo.area_km2(e["polys"]) for n, e in wards.items()}).round(3)
    df["pop_density"] = (df.population / df.area_km2).round(1)
    old = df.vulnerability.copy()
    df["vulnerability"] = vulnerability(df, df.pop_density)

    cols = [c for c in df.columns if c not in ("clear_obs", "area_km2")]
    df = df[cols + ["area_km2", "clear_obs"]] if "clear_obs" in df.columns else df
    df.to_csv(a.csv, index=False)

    print(f"wrote {a.csv}: {len(df)} wards, total area {df.area_km2.sum():,.0f} km2")
    print(f"density  min/median/max : {df.pop_density.min():,.0f} / {df.pop_density.median():,.0f} / "
          f"{df.pop_density.max():,.0f} people per km2")
    print(f"vulnerability changed for {(abs(df.vulnerability - old) > 1e-4).sum()} wards "
          f"(mean shift {abs(df.vulnerability - old).mean():.3f})")


if __name__ == "__main__":
    main()
