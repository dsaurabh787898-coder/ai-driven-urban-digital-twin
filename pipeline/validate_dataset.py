"""
Validate a cell table before it reaches the model.

Real Earth-observation data fails in ways synthetic data never does: whole units
masked by cloud, NDVI saturating in a wet season, a population raster that sums to
a third of the census, an albedo band that silently came back as raw integers. Any
of those will train quietly and produce a confident, wrong map. This script is the
gate.

    python pipeline/validate_dataset.py [data/grid_cells.csv]

Exit code 0 = safe to train. 1 = hard failures. Warnings do not block, but every
one of them changes how the output should be read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["row", "col", "lat", "lon", "zone", "ward",
            "ndvi", "tree_canopy", "builtup", "impervious", "water", "albedo",
            "bldg_height", "elevation", "lst", "population", "vulnerability"]

# (column, hard_min, hard_max, plausible_min, plausible_max, note)
RANGES = [
    ("ndvi",        -1.0, 1.0,   0.0,  0.85, "Sentinel-2 NDVI over a tropical city"),
    ("tree_canopy",  0.0, 1.0,   0.0,  0.60, "WorldCover tree fraction"),
    ("builtup",      0.0, 1.0,   0.0,  1.00, "WorldCover built-up fraction"),
    ("impervious",   0.0, 1.0,   0.0,  1.00, "rescaled NDBI or sealed fraction"),
    ("water",        0.0, 1.0,   0.0,  0.40, "WorldCover water fraction"),
    ("albedo",       0.0, 1.0,   0.08, 0.40, "broadband shortwave albedo"),
    ("bldg_height",  0.0, 400.0, 0.0,  60.0, "GHSL mean built height, metres"),
    ("elevation",  -500.0, 9000.0, 700.0, 1100.0, "Bengaluru plateau, ~900 m"),
    ("lst",        -50.0, 80.0,  20.0, 55.0, "Landsat C2 surface temperature, deg C"),
]

# Pairs that must not be near-duplicates, or single-lever what-if breaks.
COLLINEAR_GUARD = [("ndvi", "tree_canopy", 0.93), ("builtup", "impervious", 0.93),
                   ("builtup", "bldg_height", 0.93)]

CENSUS_2011_BBMP = 8_443_675


def main(path="data/grid_cells.csv"):
    p = Path(path)
    if not p.exists():
        sys.exit(f"missing {p}. Run stage 01 first.")
    df = pd.read_csv(p)
    prov = {}
    pp = Path("data/provenance.json")
    if pp.exists():
        prov = json.loads(pp.read_text())

    fails, warns = [], []
    mode = prov.get("mode", "demo (no data/provenance.json found)")
    print(f"file      : {p}  ({len(df)} units, {len(df.columns)} columns)")
    print(f"mode      : {mode}")
    if prov.get("landsat"):
        L = prov["landsat"]
        print(f"landsat   : {L['scenes']} scenes, most recent {L['most_recent_scene']}, "
              f"median {L['median_clear_obs_per_unit']:.0f} clear obs/unit")

    # --- schema ----------------------------------------------------------- #
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        fails.append(f"missing required columns: {missing}")

    # --- completeness ----------------------------------------------------- #
    for c in [c for c in REQUIRED if c in df.columns]:
        n = int(df[c].isna().sum())
        if n:
            frac = n / len(df)
            (fails if frac > 0.10 else warns).append(
                f"{c}: {n} null ({frac:.1%}) — cloud-masked units, or a band that "
                f"did not resolve")

    # --- ranges ----------------------------------------------------------- #
    for col, hmin, hmax, pmin, pmax, note in RANGES:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            fails.append(f"{col}: entirely null")
            continue
        if s.min() < hmin or s.max() > hmax:
            fails.append(f"{col}: {s.min():.3g}..{s.max():.3g} outside physical "
                         f"[{hmin}, {hmax}] — almost always an unapplied scale factor ({note})")
        elif s.min() < pmin or s.max() > pmax:
            warns.append(f"{col}: {s.min():.3g}..{s.max():.3g} outside plausible "
                         f"[{pmin}, {pmax}] for {note}")

    # --- degenerate variance ---------------------------------------------- #
    for col in ["ndvi", "builtup", "lst", "albedo"]:
        if col in df.columns and df[col].std() < 1e-6:
            fails.append(f"{col}: zero variance — the model cannot use a constant")

    # --- collinearity ------------------------------------------------------ #
    for a, b, lim in COLLINEAR_GUARD:
        if a in df.columns and b in df.columns:
            r = abs(df[a].corr(df[b]))
            if r > lim:
                fails.append(
                    f"corr({a}, {b}) = {r:.3f} > {lim}. These are effectively the "
                    f"same variable, so the model will split their coefficients "
                    f"arbitrarily and single-lever scenarios become meaningless. "
                    f"Derive {b} from an independent product, not from {a}.")

    if all(c in df.columns for c in ["ndvi", "builtup", "impervious", "water",
                                     "albedo", "bldg_height", "elevation", "tree_canopy"]):
        F = ["ndvi", "tree_canopy", "builtup", "impervious", "water", "albedo",
             "bldg_height", "elevation"]
        X = df[F].dropna().to_numpy(float)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        try:
            vif = np.diag(np.linalg.inv(np.corrcoef(Xs.T)))
            for f, v in zip(F, vif):
                if v > 10:
                    fails.append(f"VIF({f}) = {v:.1f} — severe multicollinearity")
                elif v > 5:
                    warns.append(f"VIF({f}) = {v:.1f} — coefficients will be unstable")
        except np.linalg.LinAlgError:
            fails.append("feature correlation matrix is singular — duplicate columns?")

    # --- expected physical signs ------------------------------------------ #
    if {"lst", "ndvi", "builtup"} <= set(df.columns):
        r_v, r_b = df.lst.corr(df.ndvi), df.lst.corr(df.builtup)
        print(f"signs     : corr(LST, NDVI) {r_v:+.3f}, corr(LST, built-up) {r_b:+.3f}")
        if r_v > -0.2:
            warns.append(f"corr(LST, NDVI) = {r_v:+.3f}. Urban heat islands show a "
                         f"clearly negative relationship; a flat or positive one "
                         f"usually means a wet-season composite, or LST and the "
                         f"indices coming from different seasons")
        if r_b < 0.2:
            warns.append(f"corr(LST, built-up) = {r_b:+.3f}, weaker than expected")

    # --- population sanity ------------------------------------------------- #
    if "population" in df.columns:
        tot = df.population.sum()
        ratio = tot / CENSUS_2011_BBMP
        print(f"population: {tot:,.0f} ({ratio:.2f}x the 2011 BBMP census)")
        if not 0.7 <= ratio <= 2.5:
            warns.append(f"population total is {ratio:.2f}x the 2011 census — check "
                         f"the raster is constrained, summed not averaged, and that "
                         f"the AOI matches the boundary you think it does")

    # --- spatial coverage --------------------------------------------------- #
    if "clear_obs" in df.columns:
        thin = int((df.clear_obs < 3).sum())
        if thin:
            warns.append(f"{thin} units built from fewer than 3 clear observations — "
                         f"their LST is a single-scene value with no outlier "
                         f"protection; consider masking them out entirely")

    # --- report ------------------------------------------------------------ #
    print()
    for w in warns:
        print(f"  WARN  {w}")
    for f in fails:
        print(f"  FAIL  {f}")
    if not warns and not fails:
        print("  all checks passed")
    print()

    if fails:
        print(f"{len(fails)} hard failure(s). Fix stage 01 before training — a model "
              f"trained on this will produce a confident, wrong map.")
        return 1
    if warns:
        print(f"{len(warns)} warning(s). Safe to train, but each one changes how the "
              f"output should be read. Carry them into the method section.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/grid_cells.csv"))
