"""
Flood-scenario endpoints, mounted by backend/main.py.

The split mirrors the heat side deliberately:

    learned, falsifiable  :  urban form -> land surface temperature   (stage 2)
    stated, auditable     :  rainfall -> runoff -> routing -> ponding (here)

There is no model file behind these endpoints and no accuracy figure, because
flood risk has no observable ground truth to score against. Everything is the
arithmetic in pipeline/flood_model.py with the constants served at
GET /api/flood/config so a caller can disagree with them explicitly.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import flood_model as fm

router = APIRouter(prefix="/api/flood", tags=["flood"])

TOPO = json.load(open("data/ward_topology.json"))
_DF = pd.read_csv("data/grid_cells.csv").set_index("ward").loc[TOPO["wards"]].reset_index()
UNITS = _DF.to_dict("records")
_IDX = {w: i for i, w in enumerate(TOPO["wards"])}

_logd = np.log1p(_DF.pop_density)
_LD_MIN, _LD_MAX = float(_logd.min()), float(_logd.max())


def _exposure(i: int) -> float:
    v = (math.log1p(UNITS[i]["pop_density"]) - _LD_MIN) / (_LD_MAX - _LD_MIN)
    return float(np.clip(v, 0, 1) * 100)


class Storm(BaseModel):
    rain_mm: float = Field(110, ge=0, le=600, description="Depth applied to the selected wards")
    duration_hr: float = Field(3.0, gt=0, le=72)
    drainage_mm_hr: float | None = Field(None, ge=0, le=100,
                                         description="Scenario parameter, not measured data")
    wards: list[str] | None = Field(None, description="Wards it rains on. Omit for the whole city.")
    per_ward_mm: dict[str, float] | None = Field(None, description="Overrides rain_mm per ward")


@router.get("/config")
def config():
    return {"config": fm.FLOOD_CONFIG,
            "topology": {k: TOPO[k] for k in ("method", "stats")},
            "wards": TOPO["wards"],
            "note": "No model and no accuracy metric. Flood risk has no observable "
                    "ground truth in this project, so this module is stated "
                    "arithmetic, not a fitted model."}


@router.get("/topology")
def topology():
    return TOPO


@router.post("/simulate")
def simulate(storm: Storm):
    rain: dict[int, float] = {}
    if storm.per_ward_mm:
        for w, v in storm.per_ward_mm.items():
            if w not in _IDX:
                raise HTTPException(404, f"no ward named '{w}'")
            rain[_IDX[w]] = float(v)
    else:
        targets = storm.wards if storm.wards is not None else TOPO["wards"]
        for w in targets:
            if w not in _IDX:
                raise HTTPException(404, f"no ward named '{w}'")
            rain[_IDX[w]] = storm.rain_mm

    out = fm.simulate(TOPO, UNITS, rain, storm.duration_hr, storm.drainage_mm_hr)
    for u in out["units"]:
        i = u["i"]
        u["exposure"] = round(_exposure(i), 1)
        u["vulnerability"] = round(float(UNITS[i]["vulnerability"]) * 100, 1)
        u["flood_risk"] = round(fm.flood_risk(u["flood_hazard"], u["exposure"],
                                              float(UNITS[i]["vulnerability"])), 1)
        u["risk_category"] = fm.band(u["flood_risk"])
        u["population"] = int(UNITS[i]["population"])
    out["units"].sort(key=lambda u: -u["flood_hazard"])
    out["interpretation"] = (
        "Ponding is retained runoff volume spread evenly over the ward, in mm. It is an "
        "accumulation index, not a water level: real flooding concentrates in a small "
        "fraction of the ward area. Use the ordering, not the magnitude.")
    return out


@router.get("/upstream/{ward}")
def upstream(ward: str):
    """Which wards drain through this one. The point of the flood module: a ward's
    exposure is not a property of the ward."""
    if ward not in _IDX:
        raise HTTPException(404, f"no ward named '{ward}'")
    target = _IDX[ward]
    outs: dict[int, list[int]] = {}
    for u, v, _ in TOPO["edges"]:
        outs.setdefault(u, []).append(v)

    seen, frontier = set(), [target]
    while frontier:
        nxt = []
        for v in frontier:
            for u, vs in outs.items():
                if v in vs and u not in seen and u != target:
                    seen.add(u)
                    nxt.append(u)
        frontier = nxt
    direct = sorted(TOPO["wards"][u] for u, vs in outs.items() if target in vs)
    return {
        "ward": ward,
        "direct_contributors": direct,
        "full_catchment": sorted(TOPO["wards"][u] for u in seen),
        "catchment_area_km2": round(sum(TOPO["area_km2"][u] for u in seen), 2),
        "own_area_km2": round(TOPO["area_km2"][target], 2),
        "is_domain_outlet": target in TOPO["sinks"],
    }


@router.get("/combined/{ward}")
def combined(ward: str, rain_mm: float = 110, duration_hr: float = 3):
    """Both hazards for one ward, deliberately kept as two numbers.

    A single composite index would score 'hot and dry' the same as 'cool and
    flooded', and those need opposite interventions.
    """
    if ward not in _IDX:
        raise HTTPException(404, f"no ward named '{ward}'")
    i = _IDX[ward]
    sim = fm.simulate(TOPO, UNITS, {k: rain_mm for k in range(len(UNITS))}, duration_hr)
    f = sim["units"][i]
    lst = float(UNITS[i]["lst"])
    heat_hazard = float(np.clip((lst - 30.0) / 18.0, 0, 1) * 100)
    heat = 0.60 * heat_hazard + 0.25 * _exposure(i) + 0.15 * float(UNITS[i]["vulnerability"]) * 100
    flood = fm.flood_risk(f["flood_hazard"], _exposure(i), float(UNITS[i]["vulnerability"]))
    return {
        "ward": ward,
        "heat_risk": round(heat, 1), "heat_category": fm.band(heat),
        "flood_risk": round(flood, 1), "flood_category": fm.band(flood),
        "worst": "heat" if heat >= flood else "flood",
        "both_high": heat >= 60 and flood >= 60,
        "scenario": {"rain_mm": rain_mm, "duration_hr": duration_hr},
        "note": "Two hazards, two numbers. They are not averaged into one index.",
    }
