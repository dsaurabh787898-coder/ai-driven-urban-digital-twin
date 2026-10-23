"""
Stage 1 (REAL DATA) - build the cell table from actual Earth observation data.

This is a drop-in replacement for 01_build_dataset.py. It writes exactly the same
schema to data/grid_cells.csv, so stages 02 and 03, the API and the interface need
no changes. Run one or the other, never both.

    pip install earthengine-api geemap geopandas shapely pyproj requests
    earthengine authenticate                  # one-time, opens a browser
    python pipeline/01_ingest_real.py --project YOUR_GEE_PROJECT

Earth Engine is free for research and non-commercial use but needs a registered
Cloud project. There is no way around the account: the alternative is downloading
Landsat and Sentinel scenes yourself, which is the same work with more steps.

WHAT IS AND IS NOT REALTIME
---------------------------
Land surface temperature is not a realtime quantity. Landsat 8 and 9 together give
a nominal 8-day revisit at ~10:30 local overpass, and monsoon cloud over Bengaluru
routinely kills several consecutive passes, so a usable clear-sky composite needs a
multi-week window. This script therefore builds a *seasonal composite* and records
how many clear scenes went into it and when the most recent one was. Treat that
date as the true currency of the map.

What genuinely updates in realtime is the meteorological forcing - air temperature,
humidity, wind - which `weather.py` pulls from Open-Meteo every time it is called.
The honest architecture is: slow-moving surface structure from satellites, fast-
moving atmospheric state from weather APIs, and the two labelled separately rather
than blended into one number that implies a freshness it does not have.

A NOTE ON EMISSIVITY
--------------------
Landsat Collection 2 Level-2 ST_B10 is already an emissivity-corrected surface
temperature product. Many tutorials apply a further NDVI-threshold emissivity
correction on top of it. That is double-counting and biases the result. Here the
band is scaled to Kelvin and converted to Celsius, and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# --- Bengaluru extent. Same grid as the demo build, so results are comparable. ---
LAT_MIN, LAT_MAX = 12.830, 13.140
LON_MIN, LON_MAX = 77.440, 77.780
CELL_KM = 1.0

OUT_SCHEMA = ["row", "col", "lat", "lon", "zone", "ward", "dist_cbd_km",
              "ndvi", "tree_canopy", "builtup", "impervious", "water", "albedo",
              "bldg_height", "elevation", "lst", "population", "pop_density",
              "vulnerability"]

# Asset IDs drift as catalogues are versioned. Check
# https://developers.google.com/earth-engine/datasets before a long run.
ASSETS = {
    "l8": "LANDSAT/LC08/C02/T1_L2",
    "l9": "LANDSAT/LC09/C02/T1_L2",
    "s2": "COPERNICUS/S2_SR_HARMONIZED",
    "worldcover": "ESA/WorldCover/v200",
    "built_h": "JRC/GHSL/P2023A/GHS_BUILT_H/2018",
    "albedo": "MODIS/061/MCD43A3",
    # "dem": "COPERNICUS/DEM/GLO30",
    "dem": "COPERNICUS/DEM/GLO30_2024_1",
    "pop": "WorldPop/GP/100m/pop",
}


# --------------------------------------------------------------------------- #
#  Earth Engine layers                                                         #
# --------------------------------------------------------------------------- #
def landsat_lst(ee, aoi, start, end, season):
    """Clear-sky LST composite in degrees C, plus scene-count and latest-date bands."""

    def prep(img):
        qa = img.select("QA_PIXEL")
        # Collection 2 QA_PIXEL: bit 1 dilated cloud, 3 cloud, 4 cloud shadow, 5 snow.
        clear = (qa.bitwiseAnd(1 << 1).eq(0)
                 .And(qa.bitwiseAnd(1 << 3).eq(0))
                 .And(qa.bitwiseAnd(1 << 4).eq(0))
                 .And(qa.bitwiseAnd(1 << 5).eq(0)))
        # ST_B10 -> Kelvin -> Celsius. Already emissivity-corrected; see module docstring.
        lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
        return (lst.rename("lst").updateMask(clear)
                .addBands(ee.Image.constant(1).rename("n").updateMask(clear))
                .copyProperties(img, ["system:time_start"]))

    col = (ee.ImageCollection(ASSETS["l8"]).filterBounds(aoi).filterDate(start, end)
           .merge(ee.ImageCollection(ASSETS["l9"]).filterBounds(aoi).filterDate(start, end))
           .filter(season)
           .filter(ee.Filter.lt("CLOUD_COVER", 60))
           .map(prep))

    n = col.size().getInfo()
    if n == 0:
        raise SystemExit(f"No Landsat scenes over the AOI between {start} and {end}. "
                         "Widen the window or relax the CLOUD_COVER filter.")
    latest = ee.Date(col.sort("system:time_start", False).first()
                     .get("system:time_start")).format("YYYY-MM-dd").getInfo()

    # Median over clear observations resists residual cloud edges better than mean.
    composite = col.select("lst").median().rename("lst")
    counts = col.select("n").sum().rename("clear_obs")
    return composite.addBands(counts), n, latest
def sentinel_indices(ee, aoi, start, end, season):
    """NDVI, NDWI and NDBI from a cloud-screened Sentinel-2 median composite."""

    def mask(img):
        scl = img.select("SCL")
        # SCL classes to drop: 3 shadow, 8/9/10 cloud med/high/cirrus, 11 snow.
        bad = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11)))
        return img.updateMask(bad.Not()).divide(10000)

    s2 = (ee.ImageCollection(ASSETS["s2"]).filterBounds(aoi).filterDate(start, end)
          .filter(season)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask).median())
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("ndwi")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("ndbi")
    return ndvi.addBands(ndwi).addBands(ndbi)

def landcover_fractions(ee):
    """
    Tree canopy, built-up and water as land-cover fractions from ESA WorldCover.

    Tree canopy comes from the classified product, NOT from NDVI. Deriving canopy
    from NDVI makes the two collinear and destroys single-lever what-if analysis -
    lawns, parks and cropland are green without casting shade.
    """
    wc = ee.ImageCollection(ASSETS["worldcover"]).first().select("Map")
    return (wc.eq(10).rename("tree_canopy")          # 10 tree cover
            .addBands(wc.eq(50).rename("builtup"))   # 50 built-up
            .addBands(wc.eq(80).rename("water")))    # 80 permanent water


def static_layers(ee):
    height = ee.Image(ASSETS["built_h"]).rename("bldg_height")
    dem = ee.ImageCollection(ASSETS["dem"]).select("DEM").mosaic().rename("elevation")
    return height.addBands(dem)


def albedo_layer(ee, start, end, season):
    """Broadband white-sky shortwave albedo, MODIS MCD43A3, scale factor 0.001."""
    return (ee.ImageCollection(ASSETS["albedo"]).filterDate(start, end)
            .filter(season)
            .select("Albedo_WSA_shortwave").mean().multiply(0.001).rename("albedo"))

def population_layer(ee, year=2020):
    return (ee.ImageCollection(ASSETS["pop"])
            .filter(ee.Filter.eq("country", "IND"))
            .filter(ee.Filter.eq("year", year))
            .mosaic().rename("population"))


# --------------------------------------------------------------------------- #
#  analysis units                                                              #
# --------------------------------------------------------------------------- #
def build_grid_fc(ee):
    """A regular 1 km grid over the city extent, as an ee.FeatureCollection."""
    nr = int(round((LAT_MAX - LAT_MIN) * 110.574 / CELL_KM))
    nc = int(round((LON_MAX - LON_MIN) * 111.320 * math.cos(math.radians(12.98)) / CELL_KM))
    dlat = (LAT_MAX - LAT_MIN) / nr
    dlon = (LON_MAX - LON_MIN) / nc
    feats = []
    for i in range(nr):
        for j in range(nc):
            y0, x0 = LAT_MIN + i * dlat, LON_MIN + j * dlon
            feats.append(ee.Feature(
                ee.Geometry.Rectangle([x0, y0, x0 + dlon, y0 + dlat]),
                {"row": i, "col": j, "lat": y0 + dlat / 2, "lon": x0 + dlon / 2}))
    return ee.FeatureCollection(feats)


def load_wards_fc(ee, path, name_field, zone_field):
    """
    Load real ward boundaries from a local GeoJSON.

    Bengaluru's administrative geography changed under you: the Greater Bengaluru
    Governance Act 2024 came into force in 2025, abolished BBMP and replaced it with
    five City Corporations under the Greater Bengaluru Authority, with the old 198
    wards redelimited into 369. Both vintages are in circulation:

      BBMP 198 wards (2009 delimitation, well-mirrored, stable)
        data.opencity.in -> "BBMP Ward Information" -> BBMP Ward Map 2015 (KML)
        github.com/datameet/Municipal_Spatial_Data/tree/master/Bangalore
      BBMP 225 wards (2023 delimitation, never used for an election)
        data.opencity.in -> BBMP Final Wards Map 2023
      GBA 369 wards across 5 corporations (current)
        check data.opencity.in and OSM admin boundaries; still settling

    Pick deliberately and record which you used. Historical LST composites predate
    the new boundaries, so joining a 2019 composite to 2025 wards is defensible only
    if you say so out loud.
    """
    
    import geopandas as gpd
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    
    if name_field not in gdf.columns:
        raise SystemExit(f"'{name_field}' not in {list(gdf.columns)}")
        
    feats = []
    for _, r in gdf.iterrows():
        # 1. Skip null or empty geometries
        if r.geometry is None or r.geometry.is_empty:
            continue
            
        try:
            # 2. Attempt to parse and append the geometry
            geom = ee.Geometry(r.geometry.__geo_interface__)
            feats.append(ee.Feature(geom, {
                "ward": str(r[name_field]),
                "zone": str(r[zone_field]) if zone_field in gdf.columns else "n/a",
                "lat": r.geometry.centroid.y, "lon": r.geometry.centroid.x,
                "row": 0, "col": 0}))
        except Exception as e:
            # 3. Catch and log any remaining unparseable artifacts without crashing
            print(f"Skipping invalid geometry for ward '{r.get(name_field, 'Unknown')}': {e}")
            
    return ee.FeatureCollection(feats), gdf


# --------------------------------------------------------------------------- #
#  main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Google Cloud project with Earth Engine enabled")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD, default: 18 months ago")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD, default: today")
    ap.add_argument("--season", default="3,4,5",
                    help="months to composite, comma separated. Default Mar-May, "
                         "Bengaluru's pre-monsoon heat peak")
    ap.add_argument("--wards", default=None, help="path to ward boundary GeoJSON/KML")
    ap.add_argument("--ward-field", default="Ward_Name")
    ap.add_argument("--zone-field", default="Zone")
    ap.add_argument("--scale", type=int, default=100, help="zonal stats scale, metres")
    ap.add_argument("--out", default="data/grid_cells.csv")
    args = ap.parse_args()

    try:
        import ee
    except ImportError:
        sys.exit("pip install earthengine-api")

    ee.Initialize(project=args.project)

    end = args.end or date.today().isoformat()
    start = args.start or (date.today() - timedelta(days=548)).isoformat()
    months = [int(m) for m in args.season.split(",")]
    aoi = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])

    print(f"AOI     : {LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E")
    print(f"window  : {start} to {end}, months {months}")
    season = ee.Filter.calendarRange(min(months), max(months), "month")
    
    # PASS SEASON HERE
    lst_img, n_scenes, latest = landsat_lst(
        ee, aoi, start, end, season)
    print(f"landsat : {n_scenes} scenes, most recent {latest}")

    stack = (lst_img
             # PASS SEASON HERE
             .addBands(sentinel_indices(ee, aoi, start, end, season))
             .addBands(landcover_fractions(ee))
             .addBands(static_layers(ee))
             # PASS SEASON HERE
             .addBands(albedo_layer(ee, start, end, season))
             .addBands(population_layer(ee)))

    if args.wards:
        units, gdf = load_wards_fc(ee, args.wards, args.ward_field, args.zone_field)
        print(f"units   : {len(gdf)} ward polygons from {args.wards}")
        unit_kind = "ward polygons"
    else:
        units = build_grid_fc(ee)
        print(f"units   : regular {CELL_KM} km grid")
        unit_kind = "regular grid"

    # Population must be SUMMED over a unit; everything else is a MEAN.
    reducer = (ee.Reducer.mean().forEachBand(stack.select(
                   ["lst", "clear_obs", "ndvi", "ndwi", "ndbi", "tree_canopy",
                    "builtup", "water", "bldg_height", "elevation", "albedo"]))
               .combine(ee.Reducer.sum().setOutputs(["population_sum"]), sharedInputs=False))

    stats = stack.reduceRegions(collection=units,
                                reducer=ee.Reducer.mean(),
                                scale=args.scale)
    pop_stats = stack.select("population").reduceRegions(
        collection=units, reducer=ee.Reducer.sum().setOutputs(["population"]),
        scale=100)

    print("exporting… (large AOIs may need ee.batch.Export.table.toDrive instead)")
    rows = stats.getInfo()["features"]
    pops = {f["properties"].get("ward", f"{f['properties'].get('row')}_{f['properties'].get('col')}"):
            f["properties"].get("population", 0) for f in pop_stats.getInfo()["features"]}

    recs = []
    for f in rows:
        p = f["properties"]
        key = p.get("ward", f"{p.get('row')}_{p.get('col')}")
        if p.get("lst") is None:
            continue                      # unit fully masked by cloud
        recs.append(dict(
            row=p.get("row", 0), col=p.get("col", 0),
            lat=p["lat"], lon=p["lon"],
            zone=p.get("zone", "n/a"), ward=p.get("ward", key),
            ndvi=p.get("ndvi"), tree_canopy=p.get("tree_canopy"),
            builtup=p.get("builtup"),
            # NDBI is a built-up *index*, not a fraction. Used here only as the
            # impervious proxy, min-max rescaled below. If you have OSM road and
            # building footprints, compute real sealed-surface fraction instead.
            impervious=p.get("ndbi"),
            water=p.get("water"), albedo=p.get("albedo"),
            bldg_height=p.get("bldg_height"), elevation=p.get("elevation"),
            lst=p.get("lst"), clear_obs=p.get("clear_obs"),
            population=pops.get(key, 0),
        ))

    df = pd.DataFrame(recs).dropna(subset=["lst", "ndvi"])
    if df.empty:
        sys.exit("Every unit came back masked. Widen the date window.")

    df["impervious"] = _minmax(df.impervious).clip(0.02, 0.95)
    df["bldg_height"] = df.bldg_height.fillna(df.bldg_height.median())
    df["albedo"] = df.albedo.fillna(df.albedo.median())
    if args.wards:
        # Ward polygons vary from <1 km2 to >30 km2, so density must use each ward's own
        # area (projected to UTM 43N). Dividing by CELL_KM**2 would just return the total.
        area = (gdf.assign(_w=gdf[args.ward_field].astype(str)).set_index("_w")
                   .to_crs("EPSG:32643").geometry.area / 1e6)
        df["area_km2"] = df.ward.map(area.groupby(level=0).sum()).round(3)
        df["pop_density"] = (df.population / df.area_km2).round(1)
    else:
        df["pop_density"] = df.population / (CELL_KM ** 2)
    df["dist_cbd_km"] = np.hypot((df.lat - 12.976) * 110.574,
                                 (df.lon - 77.594) * 108.5).round(2)
    df["vulnerability"] = (0.45 * _minmax(np.log1p(df.pop_density))
                           + 0.35 * (1 - _minmax(df.ndvi))
                           + 0.20 * _minmax(df.impervious)).round(4)

    df = df.reindex(columns=[c for c in OUT_SCHEMA if c in df.columns]
                            + [c for c in ("area_km2", "clear_obs") if c in df.columns])
    df.to_csv(args.out, index=False)

    prov = {
        "mode": "real",
        "city": "Bengaluru",
        "generated_utc": pd.Timestamp.now('UTC').isoformat(),
        "units": unit_kind, "n_units": int(len(df)),
        "zonal_scale_m": args.scale,
        "composite_window": {"start": start, "end": end, "months": months},
        "landsat": {"scenes": n_scenes, "most_recent_scene": latest,
                    "median_clear_obs_per_unit": float(df.clear_obs.median())},
        "sources": {
            "lst": "Landsat 8/9 C2 L2 ST_B10, emissivity-corrected, cloud-masked median",
            "ndvi/ndwi/ndbi": "Sentinel-2 L2A SCL-masked median",
            "tree_canopy/builtup/water": "ESA WorldCover v200 class fractions",
            "bldg_height": "JRC GHSL GHS_BUILT_H 2018",
            "albedo": "MODIS MCD43A3 white-sky shortwave",
            "elevation": "Copernicus DEM GLO-30",
            "population": "WorldPop 100 m constrained",
        },
        "caveats": [
            "LST is a seasonal clear-sky composite, not a realtime observation. Its "
            f"currency is the most recent clear scene: {latest}.",
            "Impervious fraction is a min-max rescaled NDBI, not a measured sealed-"
            "surface fraction. Replace with OSM footprints for operational use.",
            "Vulnerability is a proxy from density, greenness and imperviousness. It "
            "contains no age, income, housing or health data.",
            "Cloud-masked medians over the monsoon months are unreliable; the default "
            "window deliberately composites the pre-monsoon peak only.",
        ],
    }
    json.dump(prov, open("data/provenance.json", "w"), indent=2)

    print(f"\nwrote {args.out}: {len(df)} units")
    print(f"LST  min/mean/max : {df.lst.min():.1f} / {df.lst.mean():.1f} / {df.lst.max():.1f} C")
    print(f"NDVI min/mean/max : {df.ndvi.min():.2f} / {df.ndvi.mean():.2f} / {df.ndvi.max():.2f}")
    print(f"population        : {df.population.sum():,.0f}")
    print(f"clear obs / unit  : median {df.clear_obs.median():.0f}")
    print("\nNext: python pipeline/validate_dataset.py, then stages 02 and 03.")
    print("Expect R2 to fall well below the demo build's 0.89 - that is the point.")


def _minmax(s):
    s = np.asarray(s, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo + 1e-9)


if __name__ == "__main__":
    main()
