"""
Stage 3 - Export the fitted forest and the cell table into a single JSON bundle
that the browser digital twin evaluates directly.

The forest is exported as flat arrays so the front end can run inference and
exact interventional Shapley values client-side, which is what makes the what-if
simulator instant and the explanations live rather than precomputed.
"""

import json
import pickle
import numpy as np
import pandas as pd

FEATURES = ["ndvi", "tree_canopy", "builtup", "impervious",
            "water", "albedo", "bldg_height", "elevation"]


def export_tree(t):
    tt = t.tree_
    return {
        "f": [int(v) for v in tt.feature],
        "t": [round(float(v), 5) for v in tt.threshold],
        "l": [int(v) for v in tt.children_left],
        "r": [int(v) for v in tt.children_right],
        "v": [round(float(v[0][0]), 4) for v in tt.value],
    }


def export_model(m):
    """Serialise a fitted regressor into something the browser can evaluate."""
    from sklearn.pipeline import Pipeline
    if isinstance(m, Pipeline):
        sc, reg = m.steps[0][1], m.steps[-1][1]
        return {"type": "linear",
                "mean": [round(float(v), 6) for v in sc.mean_],
                "scale": [round(float(v), 6) for v in sc.scale_],
                "coef": [round(float(v), 6) for v in reg.coef_],
                "intercept": round(float(reg.intercept_), 6)}
    if hasattr(m, "estimators_") and hasattr(m, "init_"):          # gradient boosting
        return {"type": "gb", "lr": float(m.learning_rate),
                "init": round(float(m.init_.constant_[0][0]), 6),
                "trees": [export_tree(e[0]) for e in m.estimators_]}
    return {"type": "forest", "trees": [export_tree(e) for e in m.estimators_]}


def main():
    bundle_model = pickle.load(open("models/lst_model.pkl", "rb"))
    model, name = bundle_model["model"], bundle_model["name"]
    df = pd.read_csv("data/grid_cells.csv")
    metrics = json.load(open("models/metrics.json"))
    import os
    prov = (json.load(open("data/provenance.json"))
            if os.path.exists("data/provenance.json") else {"mode": "unknown"})

    models = {nm: export_model(m) for nm, m in bundle_model["all"].items()}

    # Background sample for interventional Shapley: a stratified draw across the
    # LST distribution so the reference distribution covers the whole city.
    q = pd.qcut(df.lst, 12, labels=False)
    bg = df.groupby(q, group_keys=False).apply(
        lambda g: g.sample(min(3, len(g)), random_state=3), include_groups=False)
    background = bg[FEATURES].round(4).values.tolist()

    cells = []
    for _, r in df.iterrows():
        cells.append([
            int(r.row), int(r.col), round(r.lat, 5), round(r.lon, 5),
            int(ZONE_IDX[r.zone]), int(WARD_IDX[r.ward]),
            round(r.ndvi, 4), round(r.tree_canopy, 4), round(r.builtup, 4),
            round(r.impervious, 4), round(r.water, 4), round(r.albedo, 4),
            round(r.bldg_height, 1), round(r.elevation, 0),
            round(r.lst, 2), int(r.population), round(r.vulnerability, 4),
        ])

    out = {
        "meta": {
            "city": "Bengaluru (BBMP jurisdiction)",
            "crs_analysis": "EPSG:32643 / UTM 43N",
            "crs_display": "EPSG:4326",
            "cell_km": 1.0,
            "n_cells": len(df),
            "generated": "pipeline stages 01-03",
            "model": name,
            "mode": prov.get("mode", "unknown"),
            "provenance": prov,
        },
        "features": FEATURES,
        "featureLabels": ["Vegetation index (NDVI)", "Tree canopy", "Built-up fraction",
                          "Impervious surface", "Water coverage", "Surface albedo",
                          "Mean building height", "Elevation"],
        "featureUnits": ["", "fraction", "fraction", "fraction", "fraction", "", "m", "m"],
        "zones": ZONES,
        "wards": WARDS,
        "metrics": metrics,
        "models": models,
        "selectedModel": name,
        "background": background,
        "cells": cells,
        "cellSchema": ["row", "col", "lat", "lon", "zone", "ward",
                       *FEATURES, "lst", "population", "vulnerability"],
    }
    js = json.dumps(out, separators=(",", ":"))
    open("web/bundle.json", "w").write(js)
    for nm, m in models.items():
        extra = f'{len(m["trees"])} trees' if "trees" in m else "coefficients"
        print(f"  {'*' if nm == name else ' '} {nm:20s} {m['type']:7s} {extra}")
    print(f"background : {len(background)} rows")
    print(f"bundle     : web/bundle.json  {len(js)/1024:.0f} KB")


if __name__ == "__main__":
    _df = pd.read_csv("data/grid_cells.csv")
    ZONES = sorted(_df.zone.unique().tolist())
    WARDS = sorted(_df.ward.unique().tolist())
    ZONE_IDX = {z: i for i, z in enumerate(ZONES)}
    WARD_IDX = {w: i for i, w in enumerate(WARDS)}
    main()
