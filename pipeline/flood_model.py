"""
Urban flood scenario model: rainfall -> runoff -> routing -> ponding -> risk.

DESIGN, AND WHY THERE IS NO ML IN HERE
--------------------------------------
Stage 2 trains a model because LST is an observable target. Flood risk is not.
Inventing a susceptibility index from the features and then fitting a model to
reproduce it would make every accuracy figure circular, which is the exact trap
`02_train_models.py` was written to avoid. So this module is the flood-side
equivalent of `risk_score()` in the backend: stated arithmetic, every constant in
one place, no accuracy claim attached.

Two of the three steps are published method, one is a knob:

  runoff      NRCS curve number (TR-55). Standard, citable, and the curve number
              is derived from land cover this project already measures.
  routing     mass balance down the ward cascade from flood_topology.py.
  drainage    a user-set removal rate. There is no storm-drain capacity dataset
              in this repository, so this is a scenario parameter, not data.
              It is deliberately exposed in the UI rather than buried.

WHAT A RESULT MEANS
-------------------
"Ponding depth" is retained runoff volume spread evenly over the ward. It is an
accumulation index expressed in millimetres, not a predicted water level. Real
depth is concentrated in a few percent of the ward area, so the true depth at the
places that actually flood is many times this number, and the rest of the ward is
dry. Read the ranking, not the value.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
#  constants, all in one place so they can be argued with                      #
# --------------------------------------------------------------------------- #
FLOOD_CONFIG = {
    # NRCS curve numbers, hydrologic soil group C (Bengaluru red loamy soils
    # sit between B and C; C is the more conservative choice for an urban study)
    "cn_impervious": 98.0,      # connected impervious, TR-55 table 2-2a
    "cn_pervious": 79.0,        # open space, fair condition, HSG C
    "cn_canopy": 70.0,          # woods, fair condition, HSG C
    "cn_water": 100.0,          # standing water contributes directly

    "ia_ratio": 0.20,           # initial abstraction as a fraction of S

    # cascade routing
    "retention_base": 0.25,     # share of throughput a ward holds back
    "retention_water_gain": 0.35,   # water bodies detain more
    "retention_slope_penalty": 0.5,  # steep wards hold back less

    # drainage removal, mm/hr of the ponded column
    "drainage_default": 12.0,
    "drainage_builtup_penalty": 0.45,  # dense built-up areas drain worse

    # A ward with no lower neighbour is the edge of the modelled domain, not a
    # closed basin. Bengaluru drains out along the Vrishabhavathi, Arkavathi and
    # Dakshina Pinakini valleys, so water reaching a terminal ward mostly leaves.
    # Retaining it all made those 16 wards absorb the whole city and saturate.
    "sink_export": 0.85,

    # hazard scaling: ponding depth (mm) -> 0-100
    "depth_hi_mm": 250.0,

    # shared with the heat index so the two hazards stay commensurable
    "w_hazard": 0.60, "w_exposure": 0.25, "w_vulnerability": 0.15,
    "bands": [20, 40, 60, 80],
    "band_names": ["Very low", "Low", "Moderate", "High", "Very high"],
}


# --------------------------------------------------------------------------- #
#  1. curve number from land cover                                             #
# --------------------------------------------------------------------------- #
def curve_number(impervious: float, tree_canopy: float, water: float,
                 cfg: dict = FLOOD_CONFIG) -> float:
    """Area-weighted composite CN. Fractions are clamped and renormalised so a
    ward whose measured fractions sum past 1 (they are independent products)
    does not produce a CN outside the physical range."""
    imp = max(0.0, min(1.0, impervious))
    can = max(0.0, min(1.0, tree_canopy))
    wat = max(0.0, min(1.0, water))
    total = imp + can + wat
    if total > 1.0:
        imp, can, wat = imp / total, can / total, wat / total
    per = max(0.0, 1.0 - imp - can - wat)
    cn = (imp * cfg["cn_impervious"] + can * cfg["cn_canopy"]
          + wat * cfg["cn_water"] + per * cfg["cn_pervious"])
    return max(30.0, min(100.0, cn))


def runoff_depth_mm(rain_mm: float, cn: float, cfg: dict = FLOOD_CONFIG) -> float:
    """NRCS runoff equation. Returns the depth of rainfall that becomes runoff."""
    if rain_mm <= 0:
        return 0.0
    s = 25400.0 / cn - 254.0
    ia = cfg["ia_ratio"] * s
    if rain_mm <= ia:
        return 0.0
    return (rain_mm - ia) ** 2 / (rain_mm - ia + s)


# --------------------------------------------------------------------------- #
#  2. cascade routing over the ward graph                                      #
# --------------------------------------------------------------------------- #
def _retention(water: float, slope_proxy: float, cfg: dict) -> float:
    r = (cfg["retention_base"]
         + cfg["retention_water_gain"] * max(0.0, min(1.0, water))
         - cfg["retention_slope_penalty"] * slope_proxy)
    return max(0.05, min(0.95, r))


def simulate(topology: dict, units: list[dict], rain_mm: dict[int, float],
             duration_hr: float = 3.0, drainage_mm_hr: float | None = None,
             cfg: dict = FLOOD_CONFIG) -> dict:
    """Run one rainfall scenario.

    topology  output of flood_topology.build()
    units     per-ward dicts with impervious, tree_canopy, water, elevation,
              area_km2, population, pop_density, vulnerability
    rain_mm   {ward index: rainfall depth in mm} for the wards being rained on
    returns   per-ward runoff, inflow, ponding depth, hazard and flood risk
    """
    n = len(units)
    elev = topology["elevation"]
    area = topology["area_km2"]
    drainage_mm_hr = cfg["drainage_default"] if drainage_mm_hr is None else drainage_mm_hr

    # slope proxy: mean elevation drop to downhill neighbours, normalised
    drops = [0.0] * n
    outs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in topology["edges"]:
        outs[u].append((v, w))
    for i in range(n):
        if outs[i]:
            drops[i] = sum(w for _, w in outs[i]) / len(outs[i])
    dmax = max(drops) or 1.0
    slope = [min(1.0, d / dmax) for d in drops]

    # local runoff volume, m3.  1 mm over 1 km2 = 1000 m3
    cn = [curve_number(u["impervious"], u["tree_canopy"], u["water"], cfg) for u in units]
    local_mm = [runoff_depth_mm(rain_mm.get(i, 0.0), cn[i], cfg) for i in range(n)]
    local_vol = [local_mm[i] * area[i] * 1000.0 for i in range(n)]

    # route strictly downhill, so processing in descending elevation order is a
    # valid topological order and no iteration is needed
    order = sorted(range(n), key=lambda i: -elev[i])
    inflow = [0.0] * n
    retained = [0.0] * n
    outflow = [0.0] * n

    exported = 0.0
    for i in order:
        through = local_vol[i] + inflow[i]
        if through <= 0:
            continue
        r = _retention(units[i]["water"], slope[i], cfg)
        if not outs[i]:
            # domain outlet: most of what arrives leaves the city entirely
            leaves = through * cfg["sink_export"]
            exported += leaves
            retained[i] = through - leaves
            outflow[i] = leaves
            continue
        retained[i] = through * r
        pass_on = through - retained[i]
        outflow[i] = pass_on
        wsum = sum(w for _, w in outs[i]) or 1.0
        for v, w in outs[i]:
            inflow[v] += pass_on * (w / wsum)

    # drainage removes a column over the storm duration, degraded by built-up
    out = []
    for i in range(n):
        depth_raw = retained[i] / (area[i] * 1000.0) if area[i] > 0 else 0.0
        cap = drainage_mm_hr * duration_hr * (
            1.0 - cfg["drainage_builtup_penalty"] * max(0.0, min(1.0, units[i]["builtup"])))
        depth = max(0.0, depth_raw - cap)
        hazard = min(100.0, 100.0 * math.sqrt(depth / cfg["depth_hi_mm"])) if depth > 0 else 0.0
        out.append({
            "i": i,
            "ward": topology["wards"][i],
            "rain_mm": round(rain_mm.get(i, 0.0), 1),
            "cn": round(cn[i], 1),
            "runoff_mm": round(local_mm[i], 1),
            "inflow_m3": round(inflow[i], 0),
            "outflow_m3": round(outflow[i], 0),
            "ponding_mm": round(depth, 1),
            "ponding_before_drainage_mm": round(depth_raw, 1),
            "flood_hazard": round(hazard, 1),
            "is_sink": not outs[i],
            "upstream_share": round(inflow[i] / (local_vol[i] + inflow[i]), 3)
                              if (local_vol[i] + inflow[i]) > 0 else 0.0,
        })
    return {
        "units": out,
        "total_rain_m3": round(sum(rain_mm.get(i, 0.0) * area[i] * 1000.0 for i in range(n)), 0),
        "total_runoff_m3": round(sum(local_vol), 0),
        "exported_m3": round(exported, 0),
        "retained_m3": round(sum(retained), 0),
        "duration_hr": duration_hr,
        "drainage_mm_hr": drainage_mm_hr,
        "result_type": "SCENARIO ESTIMATE",
    }


# --------------------------------------------------------------------------- #
#  3. index arithmetic, mirroring the heat side                                #
# --------------------------------------------------------------------------- #
def flood_risk(hazard: float, exposure_score: float, vulnerability: float,
               cfg: dict = FLOOD_CONFIG) -> float:
    return (cfg["w_hazard"] * hazard
            + cfg["w_exposure"] * exposure_score
            + cfg["w_vulnerability"] * vulnerability * 100.0)


def band(score: float, cfg: dict = FLOOD_CONFIG) -> str:
    return cfg["band_names"][sum(score >= b for b in cfg["bands"])]
