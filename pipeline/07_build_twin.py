"""
Stage 7 - assemble the merged Urban Climate Twin (heat + flood + heat x flood) into one HTML file.

    python pipeline/07_build_twin.py  ->  web/urban-climate-twin.html

The heat app (web/template.html) is patched at a handful of exact points; every patch asserts
that its anchor occurs exactly once so a change to the template fails loudly instead of silently.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
rd = lambda p: open(p, encoding="utf-8").read()

def sub(t, old, new, label):
    assert t.count(old) == 1, f"patch '{label}': anchor found {t.count(old)}x"
    return t.replace(old, new)

tpl = rd(P("web/template.html"))
heat = json.load(open(P("web/bundle.json")))
fb = json.load(open(P("web/flood_bundle.json")))
fm = json.load(open(P("data/flood_model.json")))
assert fb["wards"] == heat["wards"], "flood and heat bundles disagree on ward order"
assert fm["wards"] == heat["wards"], "flood model and heat bundle disagree on ward order"

code = {"vulnerable": 0, "prone": 1, "lowlying": 2}
pts = [[p["lon"], p["lat"], code[p["layer"]]] for p in json.load(open(P("data/flood_points.json")))]
keep = ["target", "label_caveat", "n_points_raw", "n_points_unique", "features", "ref_storm", "susc_percentile_oof",
        "listed_points", "lowlying", "unlisted_candidates", "cv"]
FLOOD = {"units": fb["units"], "topology": fb["topology"], "floodConfig": fb["floodConfig"],
         "model": {k: fm[k] for k in keep}, "points": pts}
heat["zoneOf"] = fb["zoneOf"]                       # used by the flood brush ("whole zone")

t = tpl
# P1 title + header tabs
t = sub(t, "<title>Bengaluru Urban Heat Digital Twin</title>", "<title>Bengaluru Urban Climate Digital Twin</title>", "title")
t = sub(t, "<h1>Bengaluru Urban Heat Digital Twin</h1>", "<h1>Bengaluru Urban Climate Digital Twin</h1>", "h1")
t = sub(t, '    <div class="spacer"></div>\n    <a class="chip" id="provChip"',
        '    <div class="seg vtabs" id="hazardTabs" role="group" aria-label="Hazard view">'
        '<button data-view="heat" aria-pressed="true">Heat</button>'
        '<button data-view="flood" aria-pressed="false">Flood</button>'
        '<button data-view="both" aria-pressed="false">Heat × flood</button></div>\n'
        '    <div class="spacer"></div>\n    <a class="chip" id="provChip"', "tabs")
# P2 views
t = sub(t, '<div class="kpis" id="kpis"></div>', '<div id="view-heat">\n<div class="kpis" id="kpis"></div>', "view-heat open")
flood_html, both_html = rd(P("web/twin/flood.html")), rd(P("web/twin/both.html"))
t = sub(t, "  <footer>", "</div>\n" + flood_html + "\n" + both_html + "\n  <footer>", "view-heat close")
t = sub(t, "Scenario outputs are model\n    estimates under assumptions stated in the method section, not predicted real-world outcomes.",
        "Scenario outputs are model\n    estimates under assumptions stated in the method section, not predicted real-world outcomes. "
        "Flood figures come from a cascade mass-balance model plus a susceptibility model trained on BBMP-flagged locations "
        "(presence-only, no rainfall attached); neither is calibrated against observed inundation.", "footer")
# P3 CSS
t = sub(t, "</style>", rd(P("web/twin/shell.css")) + "\n</style>", "css")
# P4 data + shell before the heat script
t = sub(t, "const BUNDLE = /*__BUNDLE__*/{};\n</script>",
        "const BUNDLE = " + json.dumps(heat, separators=(",", ":")) + ";\n</script>\n<script>\nconst FLOOD = "
        + json.dumps(FLOOD, separators=(",", ":")) + ";\n</script>\n<script>\n" + rd(P("web/twin/shell.js")) + "\n</script>", "data")
# P5 heat <-> bus
t = sub(t, "  selection=unitFor(c); lev={ndvi:0,tree:0,roof:0,imp:0,water:0}; render(); renderTable(); draw()});",
        "  selection=unitFor(c); lev={ndvi:0,tree:0,roof:0,imp:0,water:0}; render(); renderTable(); draw();\n"
        '  if(unitMode==="ward") TWIN.emit("heat",c[S.ward])});', "map click")
t = sub(t, "    lev={ndvi:0,tree:0,roof:0,imp:0,water:0}; render(); renderTable(); draw();});\n}",
        "    lev={ndvi:0,tree:0,roof:0,imp:0,water:0}; render(); renderTable(); draw();\n"
        "    TWIN.emit(\"heat\",D.wards.indexOf(selection.name));});\n}", "table click")
api = '''window.HEAT_API={
  wardRisk:i=>WARD_AGG[i].risk,
  scenarioDelta:(i,L)=>{const u=WARD_AGG[i]; return scenarioRisk(u,L).risk-baselineModelRisk(u).risk;},
  selected:()=>selection&&unitMode==="ward"?D.wards.indexOf(selection.name):-1,
  select:i=>{unitMode="ward"; syncSeg("unitSeg","u","ward"); selection=WARD_AGG[i];
    lev={ndvi:0,tree:0,roof:0,imp:0,water:0}; render(); renderTable(); draw();},
  onShow:()=>{resizeMap();}
};
TWIN.register("heat",HEAT_API);
TWIN.emit("heat",D.wards.indexOf(selection.name));
'''
t = sub(t, "setTimeout(renderGlobal, 60);", "setTimeout(renderGlobal, 60);\n" + api, "heat api")
# P6 flood + both scripts and boot
t = sub(t, "</script>\n</body>", "</script>\n<script>\n" + rd(P("web/twin/flood.js")) + "\n</script>\n<script>\n" + rd(P("web/twin/both.js"))
        + "\n</script>\n<script>TWIN.show((location.hash||'#heat').slice(1));</script>\n</body>", "scripts")
out = P("web/urban-climate-twin.html")
open(out, "w", encoding="utf-8").write(t)
print(f"wrote {out}  ({len(t)/1e6:.2f} MB)")
