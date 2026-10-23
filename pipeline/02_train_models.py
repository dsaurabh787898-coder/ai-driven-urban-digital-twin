"""
Stage 2 - Train and compare models that predict Land Surface Temperature from
urban-form features.

WHY LST AND NOT "HEAT RISK" DIRECTLY
------------------------------------
There is no observable ground-truth label called "heat risk". If you invent a
risk index from the features and then train a model to reproduce it, the model
learns your own formula and every accuracy figure is circular. So this pipeline
splits the problem:

    ML  : urban form  ->  LST            (a real, measurable target)
    Index: LST + exposure + vulnerability -> risk score 0-100  (transparent, no ML)

That keeps the learned part falsifiable and keeps the value-laden part auditable.

DATA LEAKAGE
------------
Neighbouring 1 km cells are strongly spatially autocorrelated. A random
train/test split lets the model memorise the neighbourhood of every test cell.
The headline metrics below therefore come from a SPATIAL BLOCK split: the grid is
cut into 16 contiguous blocks and whole blocks are held out. The random-split
metrics are reported alongside purely to show the size of the optimism.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ndvi", "tree_canopy", "builtup", "impervious",
            "water", "albedo", "bldg_height", "elevation"]
TARGET = "lst"
N_BLOCKS = 4  # -> 16 spatial blocks


def spatial_blocks(df, n=N_BLOCKS, seed=0):
    """Assign every unit to one of n*n contiguous spatial blocks.

    Works for a regular grid AND for irregular units such as wards: blocks come
    from clustering the unit centroids (lat/lon converted to km), not from the
    row/col indices, which are all 0 for polygon units. Changing `seed` moves the
    block boundaries, so repeats do not all test the same arbitrary partition."""
    from sklearn.cluster import KMeans
    lat0 = df.lat.mean()
    xy = np.column_stack([(df.lon - df.lon.mean()) * 111.32 * np.cos(np.radians(lat0)),
                          (df.lat - lat0) * 110.57])
    k = min(n * n, len(df) // 8)          # keep >= ~8 units per block
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy)
    return pd.Series(km.labels_, index=df.index)


def evaluate(model_fn, X, y, splitter, groups=None):
    preds = np.zeros(len(y))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = model_fn()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return dict(
        MAE=mean_absolute_error(y, preds),
        RMSE=root_mean_squared_error(y, preds),
        R2=r2_score(y, preds),
    ), preds


MODELS = {
    "Ridge (baseline)": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=150, max_depth=9, min_samples_leaf=6,
        max_features=0.7, random_state=7, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.85, min_samples_leaf=8, random_state=7),
}


def main():
    df = pd.read_csv("data/grid_cells.csv")
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    groups = spatial_blocks(df).to_numpy()

    # Repeated spatial CV: the block origin is shifted between repeats so that model
    # selection is not decided by one arbitrary way of cutting the city up.
    rows = []
    best, best_r2 = None, -1e9
    for name, fn in MODELS.items():
        reps = []
        for shift in range(5):
            g = spatial_blocks(df, seed=shift)
            sp, _ = evaluate(fn, X, y, GroupKFold(n_splits=5), g.to_numpy())
            reps.append(sp)
        sp = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        sd = {k: float(np.std([r[k] for r in reps])) for k in reps[0]}
        rd, _ = evaluate(fn, X, y, KFold(n_splits=5, shuffle=True, random_state=7))
        rows.append(dict(model=name, **{f"spatial_{k}": v for k, v in sp.items()},
                         spatial_R2_sd=sd["R2"],
                         **{f"random_{k}": v for k, v in rd.items()}))
        if sp["R2"] > best_r2:
            best, best_r2 = name, sp["R2"]

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:8.3f}")
    print("\nCross-validated performance (target: LST, degrees C)\n")
    print(res.to_string(index=False))
    print(f"\nSelected model: {best}  (chosen on spatial-block R2)")
    print("Note the gap between random_R2 and spatial_R2 - that gap is the leakage\n"
          "a naive random split would have hidden.\n")

    # Fit the selected model on everything and save metrics + the fitted object.
    model = MODELS[best]()
    model.fit(X, y)

    # Risk-category agreement, using out-of-fold spatial predictions of the winner.
    sp_metrics, oof = evaluate(MODELS[best], X, y, GroupKFold(n_splits=5), groups)
    sp_metrics = {k: float(v) for k, v in
                  next(r for r in rows if r["model"] == best).items() if k.startswith("spatial_")
                  and k != "spatial_R2_sd"} | {"MAE": float(sp_metrics["MAE"])}
    edges = [0, 20, 40, 60, 80, 100]
    def cat(v):
        s = np.clip((v - 29.0) / (48.0 - 29.0) * 100, 0, 100)
        return np.digitize(s, edges[1:-1])
    agree = (cat(oof) == cat(y)).mean()
    print(f"Hazard-band agreement (5 bands, spatial OOF): {agree:.3f}")

    # Keep every candidate: the front end lets a planner switch between them, because
    # a scenario estimate that only holds for one model class is not a robust finding.
    fitted = {}
    for nm, fn in MODELS.items():
        m = fn(); m.fit(X, y); fitted[nm] = m
    import pickle
    with open("models/lst_model.pkl", "wb") as f:
        pickle.dump({"model": model, "all": fitted, "features": FEATURES, "name": best}, f)
    res.to_csv("data/model_comparison.csv", index=False)
    json.dump({"selected": best, "metrics": sp_metrics, "band_agreement": float(agree),
               "comparison": rows},
              open("models/metrics.json", "w"), indent=2)
    print("saved -> models/lst_model.pkl, models/metrics.json")


if __name__ == "__main__":
    main()
