# Urban Heat Digital Twin — Bengaluru

A decision-support prototype for urban heat: it locates heat risk, explains what is
driving it, says who is exposed, lets a planner test interventions before committing
to them, and projects the near-term trajectory with its uncertainty attached.

**Read this first.** Every land-surface-temperature value, vegetation index,
built-up fraction and intra-city population figure in this repository is
**simulated**. Nothing here is a satellite retrieval or a census enumeration. What
is real is the analysis grid over the BBMP bounding box, the eight BBMP zone names,
the 2011 census population total used to scale the population field, and the
published Bengaluru summer statistics used to calibrate the synthetic temperature
field to a plausible range (roughly 29.9–48 °C, city mean about 38.6 °C). The
pipeline is built so that replacing stage 1 with real rasters changes nothing
downstream.

---

## Two design decisions worth arguing about

**The model predicts temperature, not risk.** There is no measurable ground truth
called "heat risk". If you invent a risk index from your features and then train a
model to reproduce it, the model learns your own formula and every accuracy figure
becomes circular. So the pipeline splits the problem in two:

```
learned, falsifiable   :  urban form  ->  land surface temperature
stated, auditable      :  LST + exposure + vulnerability  ->  risk score 0-100
```

The second step is plain arithmetic with the weights in one place
(`CONFIG` in `backend/main.py`, mirrored in the front end), so anyone can disagree
with the value judgement without having to unpick a model.

**Metrics come from a spatial block split.** Neighbouring 1 km cells are strongly
autocorrelated, so a random train/test split lets a model memorise each test cell's
surroundings. Stage 2 therefore holds out 16 contiguous blocks, repeated across
five different ways of cutting the city, and reports the random-split numbers
alongside purely to show the size of the optimism.

A consequence worth noting: under spatial extrapolation the ridge baseline is
competitive with, and sometimes beats, both tree ensembles. Trees cannot extrapolate
past the leaf ranges they were grown on, which is exactly what a held-out block
demands. All three models ship to the browser and the interface lets you switch
between them mid-analysis, because a scenario estimate that only survives under one
model class is not a robust finding.

---

## Layout

```
pipeline/01_build_dataset.py   DEMO stage 1: grid, synthetic surface, LST, population
pipeline/01_ingest_real.py     REAL stage 1: Landsat/Sentinel-2/WorldCover/GHSL/WorldPop
pipeline/validate_dataset.py   schema, range, collinearity and sanity gate
pipeline/weather.py            live Open-Meteo forcing and heat advisories
pipeline/02_train_models.py    model comparison, repeated spatial-block CV, selection
pipeline/03_export_bundle.py   serialise models + cell table to web/bundle.json
backend/main.py                FastAPI inference service
web/template.html              the digital twin interface (bundle injected at build)
data/  models/                 pipeline outputs
```

## Running it

```bash
pip install -r requirements.txt

python pipeline/01_build_dataset.py     # -> data/grid_cells.csv
python pipeline/02_train_models.py      # -> models/lst_model.pkl, models/metrics.json
python pipeline/03_export_bundle.py     # -> web/bundle.json

# single-file interface
python -c "
t=open('web/template.html').read(); b=open('web/bundle.json').read()
open('dist.html','w').write(t.replace('/*__BUNDLE__*/{}', b))"

# API
uvicorn backend.main:app --reload       # http://127.0.0.1:8000/docs
```

The interface is one self-contained HTML file with no server dependency: the
models are exported as coefficients and flat tree arrays, so inference, Shapley
attribution and what-if simulation all run client-side.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | index weights, band thresholds, feature list, available models |
| `GET /api/city` · `/api/statistics` | city rollup, band counts |
| `GET /api/wards` · `/api/wards/{id}` | ward table with sorting, single ward |
| `GET /api/heat-risk?level=cell\|ward\|zone` | risk surface at a chosen resolution |
| `GET /api/xai/{level}/{id}` | exact Shapley attribution, °C and risk points |
| `GET /api/exposure/{level}/{id}` | population exposure, hazard and exposure kept separate |
| `POST /api/simulation` | what-if scenario |
| `POST /api/predict` | LST for an arbitrary feature vector |
| `POST /api/recommendations` | interventions ranked by modelled effect |
| `GET /api/forecast` | 1/3/6/12-month outlook with 80% intervals |
| `GET /api/weather` | live conditions from Open-Meteo — the only realtime endpoint |
| `GET /api/advisory/{level}/{id}` | structural risk combined with today's conditions |
| `GET /api/provenance` | where every number came from and how current it is |

```jsonc
// POST /api/simulation
{"level":"ward","ident":"Bengaluru South W4","ndvi_change":0.20,
 "tree_cover_change":0.15,"albedo_change":0.10,"impervious_change":-0.10}

{"baseline_risk":91.2,"scenario_risk":70.4,"change":-20.8,
 "lst_change_c":-6.2,"result_type":"SCENARIO ESTIMATE","interpretation":"..."}
```

## Explainability

Attribution uses **exact interventional Shapley values** over all 2^8 coalitions
against a stratified background sample — not an approximation, and not
`feature_importances_`. The same routine runs in Python and in JavaScript and the
two agree to floating-point precision. Because it is exact, contributions sum
exactly to `prediction − baseline`, which is the property that makes the waterfall
readable. It is also cheap enough (about 100 ms) to recompute live as scenario
sliders move, so explanations track the scenario instead of going stale.

## Features

Eight predictors: NDVI, tree canopy, built-up fraction, impervious surface, water
coverage, surface albedo, mean building height, elevation. Population density and
the vulnerability proxy are deliberately excluded from the temperature model —
people do not cause surface temperature — and enter only through the risk index.

Stage 1 gives each predictor an independent source of spatial variation. This is
not cosmetic. An earlier version derived tree canopy directly from NDVI and
impervious surface directly from built-up fraction; the resulting collinearity
(|r| > 0.97, VIF > 20) made the linear model assign tree canopy a positive
coefficient, and a tree-planting scenario came out as *increasing* heat risk.
High NDVI is genuinely not the same thing as canopy — lawns, parks and cropland are
green without casting shade — and modelling them as the same thing breaks
single-lever what-if analysis. Current worst pairwise correlation is 0.79
(built-up vs impervious, which really are related) and all VIFs are below 4.



## Running it on real Bengaluru data

`pipeline/01_ingest_real.py` is a drop-in replacement for stage 1. It writes the
same schema to `data/grid_cells.csv`, so stages 2 and 3, the API and the interface
are untouched. Run one or the other, never both.

```bash
pip install earthengine-api geemap geopandas shapely pyproj requests
earthengine authenticate                          # one-time

python pipeline/01_ingest_real.py --project rock-abbey-509207-a2 --season 3,4,5 --wards data/bbmp_wards.geojson --ward-field WARD_NAME --zone-field ASS_CONST1

python pipeline/validate_dataset.py               # gate: exit 1 blocks training
python pipeline/02_train_models.py
python pipeline/03_export_bundle.py
```

Earth Engine is free for research and non-commercial use but needs a registered
Cloud project. The alternative is downloading the scenes yourself, which is the
same work with more steps.

| Field | Asset | Note |
|---|---|---|
| `lst` | `LANDSAT/LC08/C02/T1_L2` + `LC09` | `ST_B10` × 0.00341802 + 149.0 − 273.15, `QA_PIXEL` cloud mask |
| `ndvi`, `ndwi`, `ndbi` | `COPERNICUS/S2_SR_HARMONIZED` | SCL-masked median |
| `tree_canopy`, `builtup`, `water` | `ESA/WorldCover/v200` | class fractions 10 / 50 / 80 |
| `bldg_height` | `JRC/GHSL/P2023A/GHS_BUILT_H/2018` | mean built height, m |
| `albedo` | `MODIS/061/MCD43A3` | white-sky shortwave × 0.001 |
| `elevation` | `COPERNICUS/DEM/GLO30` | |
| `population` | `WorldPop/GP/100m/pop` | **summed** over a unit, not averaged |

Two correctness traps worth naming. Landsat Collection 2 `ST_B10` is **already**
emissivity-corrected; many tutorials apply an NDVI-threshold emissivity correction
on top of it, which double-counts and biases the result — the ingester scales to
Kelvin and stops. And tree canopy must come from the classified land-cover product,
not from NDVI, or it becomes collinear with NDVI and single-lever what-if analysis
stops meaning anything (see the Features section above for what that failure looks
like in practice).

### Boundaries: pick a vintage deliberately

Bengaluru's administrative geography moved under this project. The Greater
Bengaluru Governance Act 2024 came into force in 2025, abolished BBMP and replaced
it with five City Corporations under the Greater Bengaluru Authority; the 198 wards
were redelimited into 369. Three vintages are in circulation:

- **BBMP 198 wards** (2009 delimitation) — best-mirrored and most stable. On
  `data.opencity.in` under "BBMP Ward Information", and in
  `datameet/Municipal_Spatial_Data`.
- **BBMP 225 wards** (2023 delimitation) — notified, never used for an election.
- **GBA 369 wards across 5 corporations** — current, still settling.

The demo build uses the old BBMP zone names because that is what the stable open
data uses. Whichever you pick, record it: joining a 2019 temperature composite to
2025 ward boundaries is defensible, but only if you say so.

### What "realtime" can actually mean

Land surface temperature is not a realtime quantity. Landsat 8 and 9 together give
a nominal 8-day revisit at a ~10:30 local overpass, and monsoon cloud over
Bengaluru removes several consecutive passes, so a usable clear-sky field needs a
multi-week composite. The ingester records the scene count and the date of the most
recent clear scene, and the interface displays that date — it is the true currency
of the map, and the header badge turns green and shows it once you are on real data.

What genuinely updates continuously is the atmosphere. `pipeline/weather.py` pulls
current air temperature, apparent temperature, humidity and wind from Open-Meteo
(no key, no account, 15-minute cadence) and `GET /api/advisory/{level}/{id}`
combines it with a unit's structural score:

```
structural risk   property of the place, moves over months
today's weather   property of the day, moves hourly
      -> advisory: is this place's known disadvantage being stressed right now?
```

Deliberately an advisory, not a prediction of harm — that would need health
surveillance data this system does not have. The structural offset (±2 °C of
effective apparent temperature across the risk range) is an uncalibrated heuristic;
validate it against a local sensor network before anyone acts on it. Air
temperature at 2 m and land surface temperature differ by many degrees on a clear
afternoon and are never interchangeable.

The published single-file interface cannot reach Open-Meteo — hosted pages are
blocked from calling external hosts — so the live layer runs against the API
service only.

### Expect the numbers to get worse

The demo build reports spatial-block R² around 0.89 because its temperature field
was generated from its own features plus noise. Real LST carries wind, soil
moisture, anthropogenic heat, thermal admittance and retrieval error that eight
surface variables cannot explain. **Expect R² in the 0.5–0.7 range**, wider scenario
intervals, and possibly a different model winning. If your real-data R² comes out
near 0.9, something has leaked — check that LST and the indices come from the same
season and that units are not overlapping.

## Real satellite imagery, not just satellite-derived numbers

Everything above turns satellite data into numbers — NDVI, LST, a risk score. It
never shows you a satellite *picture*. `web/satellite_viewer.html` does: the same
grid, ward outlines and risk layers as the main twin, drawn over a real image
basemap.

```
python3 -c "
t=open('web/satellite_viewer.html').read(); b=open('web/bundle.json').read()
open('satellite-viewer.html','w').write(t.replace('/*__BUNDLE__*/{}', b))"
```

Then open `satellite-viewer.html` directly in a browser, or serve it
(`python3 -m http.server`) and browse to it. Five basemaps, switchable:

| Basemap | Source | Currency | Resolution |
|---|---|---|---|
| Esri World Imagery | `server.arcgisonline.com` | periodic refresh, undated | ~1 m in cities |
| MODIS Terra true colour | NASA GIBS | daily, ~10:30 local pass | 250 m |
| MODIS Aqua true colour | NASA GIBS | daily, ~13:30 local pass | 250 m |
| VIIRS true colour | NASA GIBS | daily | 375 m |
| MODIS LST (day) | NASA GIBS | daily | ~1 km, colourised |

All five are free and need no account or API key. The MODIS LST layer is a second,
independent satellite temperature product — useful as a rough cross-check against
this project's own Landsat-based pipeline, not the same measurement and not
expected to match it exactly.



## Reference: swapping data sources


| Field | Source |
|---|---|
| `lst` | Landsat 8/9 Collection 2 L2 `ST_B10`, cloud-masked, peak-season composite |
| `ndvi`, `water` | Sentinel-2 L2A, NDVI and NDWI |
| `builtup`, `impervious` | Sentinel-2 NDBI plus OSM building footprints and road network |
| `tree_canopy` | high-resolution canopy product or classified Sentinel-2, **not** NDVI |
| `albedo` | Landsat narrow-to-broadband conversion |
| `bldg_height` | OSM `building:levels` where tagged, or a global height raster |
| `elevation` | SRTM / Copernicus DEM |
| `population` | WorldPop, or census at ward level disaggregated by built-up area |
| ward geometry | see the boundary-vintage note above |

Reproject everything to EPSG:32643 before zonal statistics; the display layer stays
EPSG:4326. Once real LST is in place, refit and re-read the metrics honestly: R²
against real satellite data will be lower than what this demo reports, and the
scenario numbers should widen accordingly.








---

## The flood module

The architecture note above said a second hazard needs a new target column, an
entry in `MODELS` and a front-end layer. That turned out to be wrong for flooding,
in two ways worth recording.

**There is no `MODELS` entry, because there is no target.** LST is observable, so
stage 2 can be scored honestly. Flood risk is not. Fitting a model to a
susceptibility index assembled from the same features reproduces the formula and
returns a meaningless R². So the flood module is the hazard-side equivalent of
`risk_score()`: stated arithmetic, constants in one place, no accuracy claim.
Runoff uses the NRCS curve number method (TR-55), which is published and citable;
the routing and drainage terms are not, and are labelled as such in the interface.

**Heat is local, flooding is not.** A ward is hot because of what is inside it, so
zonal statistics over a ward polygon are a fair summary. A ward floods because of
what is upstream of it. Nayandahalli is 2.05 km² and sits below a 289 km²
contributing catchment; under a citywide 110 mm storm, 96 % of the water reaching
it fell somewhere else. No per-ward feature vector can express that, which is why
`pipeline/flood_topology.py` exists: it is the only part of this project that
needs to know how units relate to each other rather than just what is in them.

```
pipeline/flood_topology.py   ward adjacency + downhill cascade -> data/ward_topology.json
pipeline/flood_model.py      curve number -> runoff -> routing -> ponding -> risk
pipeline/04_export_flood.py  bundle for the browser         -> web/flood_bundle.json
backend/flood.py             /api/flood/* endpoints
web/flood_simulator.html     paint-rain-on-wards scenario interface
```

```bash
python pipeline/flood_topology.py     # -> data/ward_topology.json
python pipeline/04_export_flood.py    # -> web/flood_bundle.json

python3 -c "
t=open('web/flood_simulator.html').read(); b=open('web/flood_bundle.json').read()
open('flood-simulator.html','w').write(t.replace('/*__FLOODBUNDLE__*/{}', b))"
```

| Endpoint | Purpose |
|---|---|
| `GET /api/flood/config` | runoff and routing constants, topology method and caveats |
| `GET /api/flood/topology` | the ward cascade graph |
| `POST /api/flood/simulate` | a rainfall scenario, per-ward or citywide |
| `GET /api/flood/upstream/{ward}` | direct contributors and the full catchment draining through a ward |
| `GET /api/flood/combined/{ward}` | heat and flood risk side by side, deliberately not averaged |



## Merged twin and learned flood susceptibility

`python pipeline/05_flood_labels.py` joins the three BBMP/KSRSAC point layers (OpenCity, public domain) to wards.
`python pipeline/06_train_flood.py` trains a Poisson ridge model of flagged-point density with spatial-block CV.
`python pipeline/07_build_twin.py` assembles `web/urban-climate-twin.html`: Heat, Flood and Heat x flood tabs in one file.

Labels are presence-only (no date, rainfall or depth), so the learned layer is a rain-independent susceptibility map
and cannot learn rain-to-flood response; the cascade supplies that. Skill is modest: D2 0.224 over an area-only null,
AUC 0.64 for flagged wards (area alone 0.615), cascade ponding alone 0.639 (area 0.623). Treat both layers as hypotheses.
