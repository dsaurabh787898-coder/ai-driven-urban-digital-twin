"""
Stage 6 - Flood-susceptibility model trained on real BBMP flood-point labels.

TARGET
------
Number of BBMP-flagged flood locations per ward (the "vulnerable" and "prone"
layers, de-duplicated across layers). The "low-lying" layer is NOT used for
training: it is a different BBMP inventory, so it serves as a held-out label type.

WHY A POISSON MODEL WITH AN AREA OFFSET
---------------------------------------
"Ward contains at least one point" is partly a statement about ward size: a
larger polygon has more chances to contain one. Ward area alone scores ROC-AUC
~0.62 against that label. Modelling the *density* of flagged points (count per
km2, Poisson, log-area offset) removes the size effect, and every candidate is
compared against a null model that knows nothing but area. A feature only counts
if it beats that null under spatial validation.

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
The labels are presence-only and carry no rainfall or date. So:
  - it predicts WHERE BBMP is likely to flag flooding (a susceptibility ranking),
  - it cannot learn how much rain produces how much flooding,
  - a ward with no listed point is unlabelled, not confirmed dry.
Predictions are reported out-of-fold, so no ward is scored by a model that saw it.

    python pipeline/06_train_flood.py
        -> data/flood_model.json   coefficients, per-ward out-of-fold scores, metrics
        -> models/flood_metrics.json
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flood_model as fm

warnings.filterwarnings("ignore")

SEEDS = [0, 1, 2, 3, 4]
N_BLOCKS, N_FOLDS = 16, 8
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
DEDUPE_M = 120.0
REF_STORM = dict(rain=110.0, duration=3.0, drainage=12.0)   # reference storm for the cascade feature


# --------------------------------------------------------------------------- #
#  data                                                                        #
# --------------------------------------------------------------------------- #
def dedupe_union(points, lat0):
    """Union of the vulnerable and prone layers with points closer than DEDUPE_M
    merged, so a location listed in both inventories counts once."""
    kx, ky = 111320.0 * np.cos(np.radians(lat0)), 110574.0
    kept = []
    for p in points:
        x, y = p["lon"] * kx, p["lat"] * ky
        if all((x - q[0]) ** 2 + (y - q[1]) ** 2 > DEDUPE_M ** 2 for q in kept):
            kept.append((x, y, p["ward"]))
    return kept


def load():
    topo = json.load(open("data/ward_topology.json"))
    names = topo["wards"]
    df = pd.read_csv("data/grid_cells.csv").set_index("ward").loc[names].reset_index()
    pts = json.load(open("data/flood_points.json"))
    lab = pd.read_csv("data/flood_labels.csv").set_index("ward").loc[names]

    union_in = [p for p in pts if p["layer"] in ("vulnerable", "prone")]
    raw_n = len(union_in)
    kept = dedupe_union(union_in, float(df.lat.mean()))
    count = pd.Series(0, index=names, dtype=float)
    for _, _, w in kept:
        count[w] += 1
    low = (lab.n_lowlying > 0).astype(int).values
    return topo, df, count.values, low, raw_n, len(kept)


def catchment_area(topo):
    n = len(topo["wards"])
    ins = {i: [] for i in range(n)}
    for u, v, _ in topo["edges"]:
        ins[v].append(u)
    out = np.zeros(n)
    for t in range(n):
        seen, fr = set(), [t]
        while fr:
            nx = []
            for v in fr:
                for u in ins[v]:
                    if u not in seen and u != t:
                        seen.add(u)
                        nx.append(u)
            fr = nx
        out[t] = sum(topo["area_km2"][u] for u in seen)
    return out


def features(topo, df):
    n = len(topo["wards"])
    units = df.to_dict("records")
    sim = fm.simulate(topo, units, {i: REF_STORM["rain"] for i in range(n)},
                      REF_STORM["duration"], REF_STORM["drainage"])["units"]
    area = np.array(topo["area_km2"])
    catch = catchment_area(topo)
    X = pd.DataFrame({
        "elevation": df.elevation.values,
        "water": df.water.values,
        "impervious": df.impervious.values,
        "builtup": df.builtup.values,
        "tree_canopy": df.tree_canopy.values,
        "log_pop_density": np.log1p(df.pop_density.values),
        "log_catchment_km2": np.log1p(catch),
        "log_catchment_ratio": np.log1p(catch / area),
        "log_ponding_ref": np.log1p([u["ponding_mm"] for u in sim]),
        "upstream_share_ref": [u["upstream_share"] for u in sim],
    })
    return X, area, catch


FEATURE_SETS = {
    "null (area only)": [],
    "catchment only": ["log_catchment_km2"],
    "cascade only": ["log_ponding_ref", "upstream_share_ref"],
    "static only": ["elevation", "water", "impervious", "builtup", "tree_canopy", "log_pop_density"],
    "full (ridge)": None,       # all columns
    "full (boosted trees)": None,
}


# --------------------------------------------------------------------------- #
#  fitting                                                                     #
# --------------------------------------------------------------------------- #
def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-9)
    term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def fit_predict(kind, Xtr, ytr, atr, Xte, ate, groups_tr=None):
    """Return expected counts (mu) for the test wards. Rate formulation: target is
    count/area with sample_weight=area, equivalent to a Poisson model with a
    log-area offset."""
    rate = ytr / atr
    if kind == "null":
        lam = ytr.sum() / atr.sum()
        return lam * ate, None
    if kind == "boost":
        m = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05,
                                          max_iter=150, min_samples_leaf=8,
                                          l2_regularization=1.0, random_state=0)
        m.fit(Xtr, rate, sample_weight=atr)
        return m.predict(Xte) * ate, None
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    best_a, best_d = ALPHAS[0], np.inf
    if groups_tr is not None and len(set(groups_tr)) >= 4:
        for a in ALPHAS:                                   # inner spatial CV, no peeking at the test blocks
            dev = 0.0
            for tr, va in GroupKFold(n_splits=4).split(Ztr, rate, groups_tr):
                mm = PoissonRegressor(alpha=a, max_iter=500).fit(Ztr[tr], rate[tr], sample_weight=atr[tr])
                dev += poisson_deviance(ytr[va], mm.predict(Ztr[va]) * atr[va])
            if dev < best_d:
                best_a, best_d = a, dev
    m = PoissonRegressor(alpha=best_a, max_iter=500).fit(Ztr, rate, sample_weight=atr)
    return m.predict(Zte) * ate, best_a


def blocks(df, seed):
    kx, ky = 111.32 * np.cos(np.radians(df.lat.mean())), 110.574
    xy = np.c_[df.lon.values * kx, df.lat.values * ky]
    return KMeans(n_clusters=N_BLOCKS, n_init=10, random_state=seed).fit_predict(xy)


def run_cv(X, y, area, low, df):
    results = {name: [] for name in FEATURE_SETS}
    oof_store = {name: [] for name in FEATURE_SETS}
    present = (y > 0).astype(int)
    null_dev_by_seed = {}
    for seed in SEEDS:
        g = blocks(df, seed)
        folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, g))
        preds = {}
        for name, cols in FEATURE_SETS.items():
            mu = np.zeros(len(y))
            for tr, te in folds:
                if name.startswith("null"):
                    mu[te], _ = fit_predict("null", None, y[tr], area[tr], None, area[te])
                    continue
                use = list(X.columns) if cols is None else cols
                kind = "boost" if "boosted" in name else "ridge"
                mu[te], _ = fit_predict(kind, X.iloc[tr][use].values, y[tr], area[tr],
                                        X.iloc[te][use].values, area[te], groups_tr=g[tr])
            preds[name] = mu
        null_dev = poisson_deviance(y, preds["null (area only)"])
        for name, mu in preds.items():
            p_any = 1.0 - np.exp(-mu)
            results[name].append({
                "d2_vs_null": 1.0 - poisson_deviance(y, mu) / null_dev,
                "auc_flagged": roc_auc_score(present, p_any),
                "auc_lowlying_heldout": roc_auc_score(low, p_any),
                # a constant-density null has no ranking; per-fold lambda differences would
                # otherwise produce a spurious correlation, so it is reported as undefined
                "spearman_density": (float("nan") if name.startswith("null")
                                     else spearmanr(mu / area, y / area)[0]),
            })
            oof_store[name].append(mu)
    return results, {k: np.mean(v, axis=0) for k, v in oof_store.items()}


def summarise(results):
    out = {}
    for name, rs in results.items():
        out[name] = {k: (float(np.nanmean([r[k] for r in rs])) if not all(r[k] != r[k] for r in rs) else float("nan"),
                         float(np.nanstd([r[k] for r in rs])) if not all(r[k] != r[k] for r in rs) else float("nan"))
                     for k in rs[0]}
    return out


# --------------------------------------------------------------------------- #
def main():
    topo, df, y, low, raw_n, kept_n = load()
    names = topo["wards"]
    X, area, catch = features(topo, df)
    present = (y > 0).astype(int)

    print(f"points: {raw_n} in vulnerable+prone layers -> {kept_n} after merging duplicates within {DEDUPE_M:.0f} m")
    print(f"wards with >=1 flagged point: {present.sum()}/{len(y)}   max per ward: {int(y.max())}   "
          f"low-lying (held-out label): {low.sum()}/{len(y)}\n")

    print(f"spatial-block CV: {N_BLOCKS} blocks, {N_FOLDS} folds, {len(SEEDS)} partitions. "
          f"Every ward is scored by a model that never saw its block.\n")
    results, oof = run_cv(X, y, area, low, df)
    S = summarise(results)
    base_auc = S["null (area only)"]["auc_flagged"][0]

    print(f"{'model':22} {'D2 vs null':>14} {'AUC flagged':>14} {'AUC low-lying*':>16} {'rho density':>14}")
    for name, m in S.items():
        def f(k):
            if m[k][0] != m[k][0]:
                return "—"
            return f"{m[k][0]:+.3f}±{m[k][1]:.3f}" if k in ("d2_vs_null", "spearman_density") else f"{m[k][0]:.3f}±{m[k][1]:.3f}"
        print(f"{name:22} {f('d2_vs_null'):>14} {f('auc_flagged'):>14} {f('auc_lowlying_heldout'):>16} {f('spearman_density'):>14}")
    print("  * low-lying layer was never used for training\n")

    # ---- final linear model on all data, for the browser ----------------------
    cols = list(X.columns)
    sc = StandardScaler().fit(X.values)
    Z = sc.transform(X.values)
    g0 = blocks(df, 0)
    rate = y / area
    best_a, best_d = ALPHAS[0], np.inf
    for a in ALPHAS:
        dev = 0.0
        for tr, va in GroupKFold(n_splits=N_FOLDS).split(Z, rate, g0):
            mm = PoissonRegressor(alpha=a, max_iter=500).fit(Z[tr], rate[tr], sample_weight=area[tr])
            dev += poisson_deviance(y[va], mm.predict(Z[va]) * area[va])
        if dev < best_d:
            best_a, best_d = a, dev
    final = PoissonRegressor(alpha=best_a, max_iter=500).fit(Z, rate, sample_weight=area)

    print(f"final ridge-Poisson alpha = {best_a:g}")
    print(f"{'feature':24} {'coef (per SD)':>14}   density multiplier per +1 SD")
    for c, w in sorted(zip(cols, final.coef_), key=lambda t: -abs(t[1])):
        print(f"{c:24} {w:+14.3f}   x{np.exp(w):.2f}")

    # out-of-fold susceptibility, averaged over the 5 partitions, for display
    mu_oof = oof["full (ridge)"]
    dens = mu_oof / area
    pct = pd.Series(dens).rank(pct=True).values * 100.0
    unlisted = [(names[i], float(dens[i]), float(pct[i]), float(catch[i]))
                for i in np.argsort(-dens) if present[i] == 0][:12]
    print("\nhighest out-of-fold susceptibility among wards with NO listed point (candidates for a survey):")
    for n_, d_, p_, c_ in unlisted[:8]:
        print(f"   {n_:28} density {d_:5.2f}/km2  percentile {p_:4.0f}  catchment {c_:6.1f} km2")

    out = {
        "target": "BBMP-flagged flood locations per ward (vulnerable + prone, de-duplicated)",
        "label_caveat": "Presence-only. A ward with no listed point is unlabelled, not confirmed dry. "
                        "No rainfall or date is attached to any point.",
        "n_points_raw": raw_n, "n_points_unique": kept_n,
        "features": cols, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": final.coef_.tolist(), "intercept": float(final.intercept_), "alpha": best_a,
        "ref_storm": REF_STORM,
        "wards": names,
        "susc_density_oof": [round(float(v), 5) for v in dens],
        "susc_percentile_oof": [round(float(v), 1) for v in pct],
        "listed_points": [int(v) for v in y],
        "lowlying": [int(v) for v in low],
        "unlisted_candidates": [{"ward": n_, "density": round(d_, 3), "percentile": round(p_, 1),
                                 "catchment_km2": round(c_, 1)} for n_, d_, p_, c_ in unlisted],
        "cv": {"scheme": f"spatial blocks (KMeans, {N_BLOCKS} blocks, {N_FOLDS} folds) x {len(SEEDS)} partitions",
               "metrics": {k: {m: ({"mean": None, "sd": None} if a != a else {"mean": round(a, 4), "sd": round(b, 4)})
                                for m, (a, b) in v.items()}
                           for k, v in S.items()},
               "null_auc_flagged": round(base_auc, 4)},
    }
    os.makedirs("models", exist_ok=True)
    json.dump(out, open("data/flood_model.json", "w"), indent=1)
    json.dump(out["cv"], open("models/flood_metrics.json", "w"), indent=1)
    print("\nwrote data/flood_model.json and models/flood_metrics.json")


if __name__ == "__main__":
    main()
