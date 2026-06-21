import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C

FEATURE_COLS = ["mean_log_ppf", "mean_facility_density", "women_secondary_edu"]
ALPHAS = np.logspace(-2, 3, 40)


def aggregate_to_states(lgas):
    df = lgas.copy()
    ppf = df["pop_per_facility"].replace([np.inf, -np.inf], np.nan)
    df["log_ppf"] = np.log10(ppf.where(ppf > 0))
    g = df.groupby("state_name")
    agg = pd.DataFrame({
        "mean_log_ppf": g["log_ppf"].mean(),
        "mean_facility_density": g["facility_density"].mean(),
        "mean_pop_per_facility": g["pop_per_facility"]
            .apply(lambda s: s.replace([np.inf, -np.inf], np.nan).mean()),
    })
    if "women_secondary_edu" in df.columns:
        agg["women_secondary_edu"] = g["women_secondary_edu"].first()
    for short in C.MODEL_TARGET_PREFERENCE:
        if short in df.columns:
            agg[short] = g[short].first()
    return agg.reset_index()


def choose_target(state_df):
    for short in C.MODEL_TARGET_PREFERENCE:
        if short in state_df.columns and state_df[short].notna().sum() >= C.MIN_STATES_FOR_MODEL:
            return short
    return None


def representative_slope(model, state_df):
    ppf0 = float(np.nanmedian(state_df["mean_pop_per_facility"]))
    ppf1 = ppf0 * 0.9
    means = {f: float(np.nanmean(state_df[f])) for f in FEATURE_COLS}

    def vec(log_ppf):
        v = dict(means)
        v["mean_log_ppf"] = log_ppf
        return [[v[f] for f in FEATURE_COLS]]

    util0 = model.predict(vec(np.log10(ppf0)))[0]
    util1 = model.predict(vec(np.log10(ppf1)))[0]
    return float((util1 - util0) / (ppf0 - ppf1)), ppf0


def train():
    if not C.FEATURES_GEOJSON.exists():
        raise SystemExit("Run build_features.py first.")

    lgas = gpd.read_file(C.FEATURES_GEOJSON)
    state_df = aggregate_to_states(lgas)

    target = choose_target(state_df)
    if target is None:
        raise SystemExit("No DHS target has enough state coverage to model.")
    print(f"Target: {target}  ({state_df[target].notna().sum()} states)\n")

    d = state_df.dropna(subset=FEATURE_COLS + [target]).copy()
    X, y = d[FEATURE_COLS].values, d[target].values

    print("Feature <-> target correlations:")
    for i, f in enumerate(FEATURE_COLS):
        r = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {f:24s} r = {r:+.2f}")
    print()

    loo, preds, truth = LeaveOneOut(), [], []
    for tr, te in loo.split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        m.fit(X[tr], y[tr])
        preds.append(m.predict(X[te])[0])
        truth.append(y[te][0])
    preds, truth = np.array(preds), np.array(truth)

    mae = mean_absolute_error(truth, preds)
    r2 = r2_score(truth, preds)
    resid_sd = float(np.std(truth - preds))
    naive_mae = mean_absolute_error(truth, np.full_like(truth, truth.mean()))

    print(f"  LOO MAE       = {mae:5.2f} pp   (naive mean-predictor = {naive_mae:5.2f} pp)")
    print(f"  LOO R2        = {r2:+.2f}")
    print(f"  residual sd   = {resid_sd:5.2f} pp  -> drives simulator uncertainty")
    if r2 < 0:
        print("\n  [!] R2 still <= 0: access structure alone barely predicts "
              "utilisation.\n      Next move: add a socioeconomic confounder "
              "(women's secondary\n      education) from DHS -- the dominant driver. "
              "Say the word and I'll wire it in.")
    elif r2 < 0.25:
        print("\n  Modest fit: treat the slope as a calibrated prior with wide "
              "bands\n  (the simulator represents this via Monte Carlo).")

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)
    chosen_alpha = float(model.named_steps["ridgecv"].alpha_)
    coefs = dict(zip(FEATURE_COLS, model.named_steps["ridgecv"].coef_.tolist()))
    slope_util_per_ppf, ppf_ref = representative_slope(model, d)

    joblib.dump({"model": model, "target": target,
                 "feature_cols": FEATURE_COLS}, C.MODEL_PATH)
    metrics = {
        "target": target, "n_states": int(len(d)),
        "loo_mae": float(mae), "naive_mae": float(naive_mae),
        "loo_r2": float(r2), "residual_sd": resid_sd,
        "chosen_alpha": chosen_alpha,
        "coefficients_per_feature_sd": coefs,
        "baseline_target_mean": float(y.mean()),
        "slope_util_per_ppf": slope_util_per_ppf,
        "slope_reference_ppf": ppf_ref,
        "model_rel_uncertainty": float(min(0.9, resid_sd / max(y.mean(), 1))),
    }
    C.MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\n  slope: +{slope_util_per_ppf*1000:.3f} utilisation pp per 1,000-person "
          f"drop in people/facility")
    print(f"  chosen alpha = {chosen_alpha:.2f}")
    print(f"\nSaved model -> {C.MODEL_PATH}")
    print(f"Saved metrics -> {C.MODEL_METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()