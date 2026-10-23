"""
Live meteorological forcing from Open-Meteo.

This is the only part of the system that is genuinely realtime. Satellite land
surface temperature has an 8-day nominal revisit and needs a multi-week clear-sky
composite to be usable, so it describes slow-moving surface structure. Air
temperature, humidity and wind change hourly and are what decides whether today is
dangerous.

Keeping them separate matters. A ward's heat-risk score describes its persistent
disadvantage - how it will behave whenever conditions are hot. A heat *alert*
describes today. Multiplying them into a single number produces something that
looks precise and means nothing, so this module returns the two side by side and
labels a combined index explicitly as an advisory, not as a prediction of harm.

Open-Meteo needs no API key and no account. Be a good citizen: cache, and do not
hammer it with one request per grid cell. Bengaluru's whole extent sits inside a
single 11 km model cell for most purposes, so a handful of sample points is plenty.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
CACHE = Path("data/weather_cache.json")
CACHE_TTL_S = 900          # 15 minutes, matching Open-Meteo's own update cadence

# A few points spanning the city rather than one centroid, so the sampling picks up
# the north-south gradient without pretending to per-cell resolution.
SAMPLE_POINTS = [
    ("north", 13.10, 77.59), ("centre", 12.976, 77.594),
    ("south", 12.89, 77.60), ("east", 12.99, 77.71), ("west", 12.95, 77.50),
]


@dataclass
class Conditions:
    time: str
    temperature_c: float
    apparent_temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    precipitation_mm: float


def _get(lat: float, lon: float, forecast_days: int = 3) -> dict:
    r = requests.get(ENDPOINT, params={
        "latitude": lat, "longitude": lon,
        "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "wind_speed_10m,precipitation,weather_code"),
        "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m",
        "daily": "temperature_2m_max,apparent_temperature_max,uv_index_max",
        "forecast_days": forecast_days, "timezone": "auto",
    }, timeout=20)
    r.raise_for_status()
    return r.json()


def current(use_cache: bool = True) -> dict:
    """Current conditions at each sample point, plus a city summary."""
    if use_cache and CACHE.exists():
        blob = json.loads(CACHE.read_text())
        if time.time() - blob.get("_fetched", 0) < CACHE_TTL_S:
            return blob

    points = {}
    for name, lat, lon in SAMPLE_POINTS:
        try:
            d = _get(lat, lon)
        except requests.RequestException as e:
            points[name] = {"error": str(e)}
            continue
        c = d["current"]
        points[name] = asdict(Conditions(
            time=c["time"], temperature_c=c["temperature_2m"],
            apparent_temperature_c=c["apparent_temperature"],
            humidity_pct=c["relative_humidity_2m"],
            wind_speed_kmh=c["wind_speed_10m"],
            precipitation_mm=c["precipitation"]))
        points[name]["daily_max_c"] = d["daily"]["temperature_2m_max"]
        points[name]["daily_apparent_max_c"] = d["daily"]["apparent_temperature_max"]
        points[name]["uv_index_max"] = d["daily"]["uv_index_max"]

    ok = [p for p in points.values() if "error" not in p]
    if not ok:
        return {"error": "all Open-Meteo requests failed", "points": points}

    out = {
        "_fetched": time.time(),
        "source": "Open-Meteo (DWD/NOAA/Météo-France models), no API key",
        "result_type": "REAL OBSERVATION / FORECAST",
        "points": points,
        "city": {
            "temperature_c": round(sum(p["temperature_c"] for p in ok) / len(ok), 1),
            "apparent_temperature_c": round(
                sum(p["apparent_temperature_c"] for p in ok) / len(ok), 1),
            "humidity_pct": round(sum(p["humidity_pct"] for p in ok) / len(ok)),
            "wind_speed_kmh": round(sum(p["wind_speed_kmh"] for p in ok) / len(ok), 1),
            "spread_c": round(max(p["temperature_c"] for p in ok)
                              - min(p["temperature_c"] for p in ok), 1),
        },
        "note": ("Air temperature at 2 m, not land surface temperature. The two "
                 "differ by many degrees on a clear afternoon and are not "
                 "interchangeable."),
    }
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------- #
#  advisory combination                                                        #
# --------------------------------------------------------------------------- #
# India's national heat-action framework works off absolute air temperature
# thresholds and relative departures from normal. These are working defaults for
# a plains city, expressed in apparent temperature because humidity is what makes
# Bengaluru's pre-monsoon afternoons bite. Set them with your local health
# department before anyone acts on them.
ADVISORY_BANDS = [(38, "Watch"), (41, "Alert"), (44, "Severe")]


def advisory(structural_risk: float, use_cache: bool = True) -> dict:
    """
    Combine today's conditions with a ward's structural heat-risk score.

    `structural_risk` is the 0-100 score from the digital twin: a property of the
    place, stable over months. Today's weather is a property of the day. The output
    is an ADVISORY - a statement about whether this place's known disadvantage is
    being stressed right now - and deliberately not a prediction of health outcomes,
    which would need surveillance data this system does not have.
    """
    w = current(use_cache)
    if "error" in w:
        return {"error": w["error"]}
    at = w["city"]["apparent_temperature_c"]

    level = "None"
    for thresh, name in ADVISORY_BANDS:
        if at >= thresh:
            level = name

    # Structural risk raises the effective exposure of a place at a given city-wide
    # condition. A ward at 90 reaches a given stress earlier than one at 40.
    offset = (structural_risk - 50) / 50 * 2.0          # +/- 2 C of effective shift
    effective = at + offset
    eff_level = "None"
    for thresh, name in ADVISORY_BANDS:
        if effective >= thresh:
            eff_level = name

    return {
        "result_type": "ADVISORY",
        "observed_apparent_c": at,
        "city_level": level,
        "structural_risk": round(structural_risk, 1),
        "effective_apparent_c": round(effective, 1),
        "ward_level": eff_level,
        "thresholds_c": {name: t for t, name in ADVISORY_BANDS},
        "weather_time": w["points"]["centre"]["time"],
        "caveat": ("The structural offset is a working heuristic, not a calibrated "
                   "relationship: it has never been validated against measured "
                   "intra-city air temperature differences in Bengaluru, let alone "
                   "against health outcomes. Calibrate it against a local sensor "
                   "network before attaching any operational consequence to it."),
    }


if __name__ == "__main__":
    import sys
    w = current(use_cache=False)
    if "error" in w:
        sys.exit(w["error"])
    c = w["city"]
    print(f"Bengaluru now: {c['temperature_c']} C air, feels {c['apparent_temperature_c']} C, "
          f"{c['humidity_pct']}% RH, wind {c['wind_speed_kmh']} km/h")
    print(f"north-south spread across sample points: {c['spread_c']} C")
    for r in (30, 60, 90):
        a = advisory(r)
        print(f"  structural risk {r:3d} -> effective {a['effective_apparent_c']} C, "
              f"level {a['ward_level']}")
