"""
FastAPI inference service for the Bengaluru urban heat digital twin.

    uvicorn backend.main:app --reload

Everything it serves is derived from data/grid_cells.csv and models/lst_model.pkl,
which stages 01-03 of the pipeline produce. The service holds no state of its own.

Design note: /api/predict returns a land surface temperature, because that is what
the model was trained on. The heat-risk score is assembled from LST, exposure and
vulnerability by `risk_score()` below, which is plain arithmetic with the weights
exposed in /api/config. Keeping the learned part and the value-laden part in
separate, separately auditable places is deliberate.
"""

from __future__ import annotations

import math
import pickle
from functools import lru_cache
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

FEATURES = ["ndvi", "tree_canopy", "builtup", "impervious",
            "water", "albedo", "bldg_height", "elevation"]
FEATURE_LABELS = {
    "ndvi": "Vegetation (NDVI)", "tree_canopy": "Tree canopy",
    "builtup": "Built-up fraction", "impervious": "Impervious surface",
    "water": "Water coverage", "albedo": "Surface albedo",
    "bldg_height": "Building height", "elevation": "Elevation",
}
CAPS = {"ndvi": (0.02, 0.80), "tree_canopy": (0.0, 0.70), "builtup": (0.0, 1.0),
        "impervious": (0.01, 0.95), "water": (0.0, 0.35), "albedo": (0.08, 0.38),
        "bldg_height": (0.0, 100.0), "elevation": (0.0, 3000.0)}

CONFIG = {"lst_lo": 30.0, "lst_hi": 48.0,
          "w_hazard": 0.60, "w_exposure": 0.25, "w_vulnerability": 0.15,
          "bands": [20, 40, 60, 80],
          "band_names": ["Very low", "Low", "Moderate", "High", "Very high"]}

app = FastAPI(title="Urban Heat Digital Twin", version="0.3",
              description="Prototype. All surface and temperature values are "
                          "simulated demo data, not satellite observations.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Flood scenario module. Optional: it needs data/ward_topology.json from
# pipeline/flood_topology.py, and the heat endpoints work without it.
try:
    from backend.flood import router as flood_router
    app.include_router(flood_router)
except (ImportError, FileNotFoundError) as exc:      # noqa: BLE001
    print(f"flood module not mounted ({exc}); "
          f"run: python pipeline/flood_topology.py")

DF = pd.read_csv("data/grid_cells.csv")
_BUNDLE = pickle.load(open("models/lst_model.pkl", "rb"))
MODELS = _BUNDLE.get("all", {_BUNDLE["name"]: _BUNDLE["model"]})
DEFAULT_MODEL = _BUNDLE["name"]

_logd = np.log1p(DF.population)
LD_MIN, LD_MAX = float(_logd.min()), float(_logd.max())

# Background sample for interventional Shapley, stratified across the LST range.
_q = pd.qcut(DF.lst, 12, labels=False)
BACKGROUND = (DF.groupby(_q, group_keys=False)
                .apply(lambda g: g.sample(min(3, len(g)), random_state=3),
                       include_groups=False)[FEATURES].to_numpy(float))


# --------------------------------------------------------------------------- #
#  index arithmetic (no ML)                                                    #
# --------------------------------------------------------------------------- #
def hazard(lst: float) -> float:
    return float(np.clip((lst - CONFIG["lst_lo"]) /
                         (CONFIG["lst_hi"] - CONFIG["lst_lo"]), 0, 1) * 100)


def exposure(pop_density: float) -> float:
    return float(np.clip((math.log1p(pop_density) - LD_MIN) / (LD_MAX - LD_MIN), 0, 1) * 100)


def risk_score(lst: float, pop_density: float, vulnerability: float) -> float:
    return (CONFIG["w_hazard"] * hazard(lst)
            + CONFIG["w_exposure"] * exposure(pop_density)
            + CONFIG["w_vulnerability"] * vulnerability * 100)


def band(risk: float) -> str:
    i = sum(risk >= b for b in CONFIG["bands"])
    return CONFIG["band_names"][i]


def predict_lst(X: np.ndarray, model: str | None = None) -> np.ndarray:
    return MODELS[model or DEFAULT_MODEL].predict(np.asarray(X, float))


# --------------------------------------------------------------------------- #
#  units                                                                       #
# --------------------------------------------------------------------------- #
def subset(level: str, ident: str | None) -> pd.DataFrame:
    if level == "city" or ident is None:
        return DF
    col = {"ward": "ward", "zone": "zone"}.get(level)
    if col is None:
        raise HTTPException(400, f"unknown level '{level}'")
    g = DF[DF[col] == ident]
    if g.empty:
        raise HTTPException(404, f"no {level} named '{ident}'")
    return g


def summarise(g: pd.DataFrame, name: str, model: str | None = None) -> dict:
    feats = g[FEATURES].mean()
    lst = float(g.lst.mean())
    popd = float(g.population.mean())
    vul = float(g.vulnerability.mean())
    r = risk_score(lst, popd, vul)
    return {
        "name": name, "cells": int(len(g)), "area_km2": float(len(g)),
        "population": int(g.population.sum()), "population_density": round(popd, 1),
        "lst_observed_c": round(lst, 2),
        "lst_modelled_c": round(float(predict_lst(feats.to_numpy()[None, :], model)[0]), 2),
        "heat_risk": round(r, 1), "risk_category": band(r),
        "hazard": round(hazard(lst), 1), "exposure": round(exposure(popd), 1),
        "vulnerability": round(vul * 100, 1),
        "features": {k: round(float(v), 4) for k, v in feats.items()},
        "provenance": "lst_observed_c is SIMULATED demo data; lst_modelled_c is a MODEL PREDICTION",
    }


# --------------------------------------------------------------------------- #
#  exact interventional Shapley over 2**8 coalitions                           #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=512)
def _shap(x_key: tuple, model: str) -> tuple:
    x = np.array(x_key, float)
    n = len(FEATURES)
    vals = {}
    for k in range(n + 1):
        for S in combinations(range(n), k):
            Z = BACKGROUND.copy()
            for i in S:
                Z[:, i] = x[i]
            vals[S] = float(predict_lst(Z, model).mean())
    fact = [math.factorial(i) for i in range(n + 1)]
    phi = np.zeros(n)
    for i in range(n):
        for k in range(n):
            for S in combinations([j for j in range(n) if j != i], k):
                w = fact[k] * fact[n - k - 1] / fact[n]
                phi[i] += w * (vals[tuple(sorted(S + (i,)))] - vals[S])
    return tuple(phi), vals[()], vals[tuple(range(n))]


def explain(feats: np.ndarray, model: str | None = None) -> dict:
    phi, base, pred = _shap(tuple(np.round(feats, 5)), model or DEFAULT_MODEL)
    k = CONFIG["w_hazard"] * 100 / (CONFIG["lst_hi"] - CONFIG["lst_lo"])
    items = sorted(
        [{"feature": f, "label": FEATURE_LABELS[f], "value": round(float(feats[i]), 4),
          "contribution_c": round(float(phi[i]), 3),
          "contribution_risk_points": round(float(phi[i]) * k, 2)}
         for i, f in enumerate(FEATURES)],
        key=lambda d: -abs(d["contribution_c"]))
    return {"method": "exact interventional Shapley values, 256 coalitions, "
                      f"{len(BACKGROUND)}-row stratified background",
            "units": "degrees C of land surface temperature",
            "baseline_c": round(base, 3), "prediction_c": round(pred, 3),
            "contributions": items,
            "note": "Contributions sum to prediction_c - baseline_c. Positive means "
                    "the feature pushes surface temperature up at this location."}


# --------------------------------------------------------------------------- #
#  routes                                                                      #
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config():
    return {"index": CONFIG, "features": FEATURES,
            "models": list(MODELS), "selected_model": DEFAULT_MODEL,
            "caps": CAPS}


@app.get("/api/city")
def get_city():
    out = summarise(DF, "Bengaluru (BBMP)")
    out["zones"] = sorted(DF.zone.unique().tolist())
    out["crs"] = {"analysis": "EPSG:32643", "display": "EPSG:4326"}
    out["warning"] = ("Demo dataset. Surface variables, temperature and the "
                      "intra-city population distribution are simulated.")
    return out


@app.get("/api/wards")
def list_wards(sort: str = "heat_risk", desc: bool = True, limit: int = 100):
    rows = [summarise(g, w) for w, g in DF.groupby("ward")]
    if sort not in rows[0]:
        raise HTTPException(400, f"cannot sort by '{sort}'")
    rows.sort(key=lambda r: r[sort], reverse=desc)
    return {"count": len(rows), "wards": rows[:limit],
            "note": "Ordering is an analytical view of model output, not a priority list."}


@app.get("/api/wards/{ward_id}")
def get_ward(ward_id: str):
    return summarise(subset("ward", ward_id), ward_id)


@app.get("/api/heat-risk")
def heat_risk_grid(level: Literal["cell", "ward", "zone"] = "ward"):
    if level == "cell":
        return {"level": "cell", "cells": [
            {"row": int(r.row), "col": int(r.col), "lat": r.lat, "lon": r.lon,
             "lst_c": r.lst, "heat_risk": round(risk_score(r.lst, r.population, r.vulnerability), 1)}
            for r in DF.itertuples()]}
    col = "ward" if level == "ward" else "zone"
    return {"level": level, "units": [summarise(g, k) for k, g in DF.groupby(col)]}


@app.get("/api/xai/{level}/{ident}")
def get_xai(level: Literal["ward", "zone"], ident: str, model: str | None = None):
    g = subset(level, ident)
    return {"unit": ident, "model": model or DEFAULT_MODEL,
            **explain(g[FEATURES].mean().to_numpy(), model)}


@app.get("/api/exposure/{level}/{ident}")
def get_exposure(level: Literal["ward", "zone"], ident: str):
    g = subset(level, ident)
    risks = g.apply(lambda r: risk_score(r.lst, r.population, r.vulnerability), axis=1)
    return {
        "unit": ident,
        "population": int(g.population.sum()),
        "population_density": round(float(g.population.mean()), 1),
        "hazard": round(hazard(float(g.lst.mean())), 1),
        "exposure": round(exposure(float(g.population.mean())), 1),
        "population_above_risk_60": int(g.population[risks >= 60].sum()),
        "share_of_city": round(float(g.population.sum()) / float(DF.population.sum()), 4),
        "note": "Hazard and exposure are reported separately on purpose. A hot "
                "industrial cell with few residents and a moderately hot dense "
                "residential cell are not the same planning problem.",
    }


class SimulationRequest(BaseModel):
    level: Literal["ward", "zone", "city"] = "ward"
    ident: str | None = Field(None, description="ward or zone name")
    ndvi_change: float = Field(0.0, ge=-0.5, le=0.5)
    tree_cover_change: float = Field(0.0, ge=-0.5, le=0.5)
    albedo_change: float = Field(0.0, ge=-0.2, le=0.2)
    impervious_change: float = Field(0.0, ge=-0.5, le=0.5)
    water_change: float = Field(0.0, ge=-0.2, le=0.2)
    model: str | None = None


@app.post("/api/simulation")
def simulate(req: SimulationRequest):
    g = subset(req.level, req.ident)
    X = g[FEATURES].to_numpy(float)
    Z = X.copy()
    for col, delta in (("ndvi", req.ndvi_change), ("tree_canopy", req.tree_cover_change),
                       ("albedo", req.albedo_change), ("impervious", req.impervious_change),
                       ("water", req.water_change)):
        i = FEATURES.index(col)
        Z[:, i] = np.clip(Z[:, i] + delta, *CAPS[col])

    popd, vul = float(g.population.mean()), float(g.vulnerability.mean())
    base_lst = float(predict_lst(X, req.model).mean())
    new_lst = float(predict_lst(Z, req.model).mean())
    base_r, new_r = risk_score(base_lst, popd, vul), risk_score(new_lst, popd, vul)
    mae = 1.20  # spatial-block MAE of the selected model, degrees C
    return {
        "unit": req.ident or "city", "model": req.model or DEFAULT_MODEL,
        "baseline_risk": round(base_r, 1), "scenario_risk": round(new_r, 1),
        "change": round(new_r - base_r, 1),
        "baseline_lst_c": round(base_lst, 2), "scenario_lst_c": round(new_lst, 2),
        "lst_change_c": round(new_lst - base_lst, 2),
        "result_type": "SCENARIO ESTIMATE",
        "interpretation": (
            f"Model-estimated change in heat-risk score across {len(g)} km². This is not a "
            f"predicted real-world outcome: it assumes the intervention moves the surface "
            f"variables exactly as specified and that the cross-sectional relationship the "
            f"model learned across the city also describes change over time in one place. "
            f"Model MAE is about {mae:.2f} °C, which bounds how finely the difference should "
            f"be read."),
    }


class PredictRequest(BaseModel):
    features: dict[str, float]
    population_density: float | None = None
    vulnerability: float | None = None
    model: str | None = None


@app.post("/api/predict")
def predict(req: PredictRequest):
    missing = [f for f in FEATURES if f not in req.features]
    if missing:
        raise HTTPException(400, f"missing features: {missing}")
    x = np.array([[req.features[f] for f in FEATURES]], float)
    lst = float(predict_lst(x, req.model)[0])
    out = {"lst_c": round(lst, 2), "hazard": round(hazard(lst), 1),
           "result_type": "MODEL PREDICTION", "model": req.model or DEFAULT_MODEL}
    if req.population_density is not None:
        r = risk_score(lst, req.population_density, req.vulnerability or 0.5)
        out |= {"heat_risk": round(r, 1), "risk_category": band(r)}
    return out


ACTIONS = [
    ("Street-tree and canopy program", {"ndvi": 0.06, "tree_canopy": 0.10},
     ("ndvi", "tree_canopy"), "canopy shades surfaces and adds evapotranspiration"),
    ("Neighbourhood parks and open green", {"ndvi": 0.12, "tree_canopy": 0.05, "impervious": -0.05},
     ("ndvi", "impervious"), "converts sealed ground to vegetated ground"),
    ("Cool and reflective roof retrofit", {"albedo": 0.10},
     ("albedo", "builtup"), "raises albedo so less shortwave radiation is absorbed"),
    ("Permeable and lighter paving", {"impervious": -0.10, "albedo": 0.02},
     ("impervious", "builtup"), "reduces sealed, heat-storing surface"),
    ("Lake restoration and water features", {"water": 0.03, "ndvi": 0.02},
     ("water",), "open water and wet soil buffer daytime heating"),
]


@app.post("/api/recommendations")
def recommendations(level: Literal["ward", "zone"] = "ward",
                    ident: str = Query(...), model: str | None = None):
    g = subset(level, ident)
    X = g[FEATURES].to_numpy(float)
    popd, vul = float(g.population.mean()), float(g.vulnerability.mean())
    base_lst = float(predict_lst(X, model).mean())
    base_r = risk_score(base_lst, popd, vul)
    xai = explain(g[FEATURES].mean().to_numpy(), model)
    phi = {c["feature"]: c["contribution_c"] for c in xai["contributions"]}

    out = []
    for label, dose, drivers, why in ACTIONS:
        Z = X.copy()
        for col, d in dose.items():
            i = FEATURES.index(col)
            Z[:, i] = np.clip(Z[:, i] + d, *CAPS[col])
        lst = float(predict_lst(Z, model).mean())
        out.append({
            "intervention": label, "mechanism": why,
            "targets": [{"feature": d, "contribution_c": phi[d]} for d in drivers],
            "dose": dose,
            "modelled_lst_change_c": round(lst - base_lst, 3),
            "modelled_risk_change": round(risk_score(lst, popd, vul) - base_r, 2),
            "result_type": "SCENARIO ESTIMATE",
        })
    out.sort(key=lambda d: d["modelled_risk_change"])
    return {"unit": ident, "baseline_risk": round(base_r, 1), "options": out,
            "note": "Ranked by modelled effect for a fixed dose in this unit only. Says "
                    "nothing about cost, land availability or delivery time."}


CLIM = [-2.0, 0.2, 2.9, 4.3, 3.4, -0.6, -2.6, -2.7, -1.7, -1.4, -2.2, -2.4]
TREND_C_PER_YEAR = 0.030


@app.get("/api/forecast")
def forecast(level: Literal["ward", "zone"] = "ward", ident: str = Query(...),
             horizons: str = "1,3,6,12"):
    import datetime as dt
    g = subset(level, ident)
    lst0 = float(g.lst.mean())
    popd, vul = float(g.population.mean()), float(g.vulnerability.mean())
    m0 = dt.date.today().month - 1
    ref = max(CLIM)
    pts = []
    for h in [int(v) for v in horizons.split(",")]:
        m = (m0 + h) % 12
        yrs = h / 12
        mean = lst0 - (ref - CLIM[m]) + TREND_C_PER_YEAR * yrs
        sd = 0.9 + 1.3 * math.sqrt(yrs)
        lo, hi = mean - 1.2816 * sd, mean + 1.2816 * sd
        pts.append({"horizon_months": h, "month": m + 1,
                    "lst_c": round(mean, 2), "interval_80": [round(lo, 2), round(hi, 2)],
                    "heat_risk": round(risk_score(mean, popd, vul), 1),
                    "risk_interval_80": [round(risk_score(lo, popd, vul), 1),
                                         round(risk_score(hi, popd, vul), 1)]})
    return {
        "unit": ident, "reference": "peak-season daytime LST for this unit",
        "basis": f"demo monthly climatology plus a warming trend of "
                 f"{TREND_C_PER_YEAR * 10:.1f} C/decade",
        "result_type": "SCENARIO ESTIMATE",
        "points": pts,
        "caveat": "Carries no weather forecast and no land-cover change. Intervals widen "
                  "with horizon and exceed 2 C at one year; read the 12-month figure as a "
                  "direction of travel, not a planning number. Replace the climatology with "
                  "a reanalysis series and the trend with a fitted Landsat time series "
                  "before any operational use.",
    }


@app.get("/api/weather")
def get_weather():
    """
    Live meteorological conditions. This is the only realtime endpoint in the API:
    everything else describes slow-moving surface structure derived from satellite
    composites. Air temperature at 2 m is not land surface temperature.
    """
    try:
        from pipeline import weather
    except ImportError:
        raise HTTPException(503, "pipeline/weather.py not importable")
    w = weather.current()
    if "error" in w:
        raise HTTPException(502, w["error"])
    return w


@app.get("/api/advisory/{level}/{ident}")
def get_advisory(level: Literal["ward", "zone"], ident: str):
    """
    Combine a unit's structural heat-risk score with today's conditions.

    The score is a property of the place and moves over months; the weather is a
    property of the day. The result is an advisory about whether a known
    disadvantage is being stressed right now, not a prediction of health outcomes -
    that would need surveillance data this system does not have.
    """
    try:
        from pipeline import weather
    except ImportError:
        raise HTTPException(503, "pipeline/weather.py not importable")
    g = subset(level, ident)
    r = risk_score(float(g.lst.mean()), float(g.population.mean()),
                   float(g.vulnerability.mean()))
    out = weather.advisory(r)
    if "error" in out:
        raise HTTPException(502, out["error"])
    return {"unit": ident, **out}


@app.get("/api/provenance")
def get_provenance():
    """Where every number in this deployment came from, and how current it is."""
    import json as _json
    import os
    if os.path.exists("data/provenance.json"):
        return _json.load(open("data/provenance.json"))
    return {"mode": "unknown", "note": "data/provenance.json missing; rerun stage 01"}


@app.get("/api/statistics")
def statistics():
    risks = DF.apply(lambda r: risk_score(r.lst, r.population, r.vulnerability), axis=1)
    hottest = max(DF.groupby("ward"), key=lambda kv: kv[1].lst.mean())[0]
    return {
        "area_km2": int(len(DF)), "wards": int(DF.ward.nunique()),
        "zones": int(DF.zone.nunique()), "population": int(DF.population.sum()),
        "mean_lst_c": round(float(DF.lst.mean()), 2),
        "lst_range_c": [round(float(DF.lst.min()), 2), round(float(DF.lst.max()), 2)],
        "mean_heat_risk": round(float(risks.mean()), 1),
        "hottest_ward": hottest,
        "population_above_risk_60": int(DF.population[risks >= 60].sum()),
        "band_counts": {CONFIG["band_names"][i]: int(((risks >= ([0] + CONFIG["bands"])[i]) &
                        (risks < (CONFIG["bands"] + [101])[i])).sum()) for i in range(5)},
    }
