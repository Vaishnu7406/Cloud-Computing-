"""Train, compare, evaluate and save the indoor-position regression model.

Usage (from the project folder):
    python train_model.py                      # uses / generates the synthetic dataset
    python train_model.py --csv my.csv --features L1 L2 L3 L4 --targets X Y \
        --source "Where the data came from" --real
"""
import argparse
import json
import platform
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import data_utils as du

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def make_pipeline(estimator):
    """log1p compresses the large dynamic range of optical power; the scaler
    puts all LEDs on the same scale. Both are fitted inside the pipeline so
    training and prediction always use identical preprocessing."""
    return Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def candidate_models():
    return {
        "Linear Regression (baseline)": make_pipeline(LinearRegression()),
        "K-Nearest Neighbours": make_pipeline(KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                                             random_state=SEED, n_jobs=-1)),
        "Extra Trees": make_pipeline(ExtraTreesRegressor(n_estimators=150, min_samples_leaf=3,
                                                         random_state=SEED, n_jobs=-1)),
        "Neural Network (MLP)": make_pipeline(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1500,
                                                           early_stopping=True, random_state=SEED)),
    }


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))          # Euclidean error per sample (m)
    return {
        "mae_x": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "mae_y": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        "r2_x": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_y": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error_m": float(err.mean()),
        "median_error_m": float(np.median(err)),
        "p90_error_m": float(np.percentile(err, 90)),
        "max_error_m": float(err.max()),
    }


def cross_validate_candidates(models, X_train, y_train):
    """K-fold CV on the TRAINING split only (the test split is never touched)."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for name, pipe in models.items():
        fold_metrics = []
        for tr, va in kf.split(X_train):
            m = clone(pipe).fit(X_train.iloc[tr], y_train.iloc[tr])
            fold_metrics.append(compute_metrics(y_train.iloc[va].to_numpy(), m.predict(X_train.iloc[va])))
        row = {"model": name}
        for key in ("mean_error_m", "mae", "rmse", "r2"):
            vals = [f[key] for f in fold_metrics]
            row[f"cv_{key}"] = float(np.mean(vals))
            row[f"cv_{key}_std"] = float(np.std(vals))
        rows.append(row)
        print(f"  {name:32s} CV mean position error = {row['cv_mean_error_m']:.4f} m")
    return rows


def main(csv=None, features=None, targets=None, source=None, is_synthetic=None, units="m"):
    du.MODEL_DIR.mkdir(exist_ok=True)
    csv = du.Path(csv) if csv else du.DEFAULT_CSV
    using_default = csv == du.DEFAULT_CSV
    features = features or du.SYN_FEATURES
    targets = targets or du.SYN_TARGETS
    if len(targets) != 2:
        raise ValueError("Exactly two target columns (X and Y) are required.")
    if is_synthetic is None:
        is_synthetic = using_default

    if not csv.exists():
        if not using_default:
            raise FileNotFoundError(f"Dataset file not found: {csv}")
        print("Dataset not found - generating synthetic dataset ...")
        du.generate_synthetic_dataset()

    print(f"Loading {csv}")
    raw = pd.read_csv(csv)
    df, report = du.validate_dataframe(raw, features, targets)
    print(f"Rows: raw={report['rows_raw']}, usable={report['rows_clean']}, "
          f"missing={report['missing_values_raw']}, duplicates={report['duplicate_rows_raw']}")

    X, y = df[features], df[targets]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"Train={len(X_train)}  Test={len(X_test)}")

    print("Cross-validating candidates on the training split:")
    models = candidate_models()
    cv_rows = cross_validate_candidates(models, X_train, y_train)
    best = min(cv_rows, key=lambda r: r["cv_mean_error_m"])["model"]
    print(f"Selected model (lowest CV mean position error): {best}")

    final = clone(models[best]).fit(X_train, y_train)
    pred = final.predict(X_test)
    test_metrics = compute_metrics(y_test.to_numpy(), pred)

    # Test-set scores of every candidate (reported for information only; selection used CV).
    test_all = {}
    for name, pipe in models.items():
        p = final.predict(X_test) if name == best else clone(pipe).fit(X_train, y_train).predict(X_test)
        test_all[name] = compute_metrics(y_test.to_numpy(), p)

    joblib.dump(final, du.MODEL_DIR / "position_model.joblib", compress=3)
    out = X_test.copy()
    out[f"{targets[0]}_actual"], out[f"{targets[1]}_actual"] = y_test.to_numpy().T
    out[f"{targets[0]}_pred"], out[f"{targets[1]}_pred"] = pred.T
    out.to_csv(du.MODEL_DIR / "test_predictions.csv", index=False)

    try:
        data_file = str(csv.resolve().relative_to(du.BASE_DIR))
    except ValueError:
        data_file = str(csv.resolve())

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "data_file": data_file,
        "is_synthetic": bool(is_synthetic),
        "source": source or ("Synthetic data generated by data_utils.py (Lambertian LED simulation). "
                             "NOT real experimental data." if is_synthetic else "User-supplied dataset"),
        "features": list(features),
        "targets": list(targets),
        "units": {"signal": "uW (relative received optical power)" if is_synthetic else "as in source dataset",
                  "position": units},
        "validation_report": report,
        "feature_stats": {f: {"min": float(df[f].min()), "max": float(df[f].max()),
                              "median": float(df[f].median())} for f in features},
        "target_stats": {t: {"min": float(df[t].min()), "max": float(df[t].max())} for t in targets},
        "split": {"test_size": TEST_SIZE, "seed": SEED, "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "cv_folds": CV_FOLDS,
        "selection_criterion": "lowest mean cross-validated Euclidean position error (training split only)",
        "cv_results": cv_rows,
        "best_model": best,
        "best_params": {k: str(v) for k, v in final.named_steps["model"].get_params().items()},
        "test_metrics": test_metrics,
        "test_metrics_all_models": test_all,
        "saved_model_note": "The saved model was fitted on the training split only, so the reported "
                            "test metrics describe exactly this saved model.",
    }
    if is_synthetic:
        meta["simulation"] = {"room_m": [du.ROOM_W, du.ROOM_L, du.ROOM_H], "receiver_height_m": du.RX_HEIGHT,
                              "led_xy_m": du.LED_XY, "lambertian_order": du.LAMBERT_ORDER,
                              "peak_power_uw": du.PEAK_POWER_UW}
    with open(du.MODEL_DIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\nTest-set results for the selected model:")
    for k, v in test_metrics.items():
        print(f"  {k:15s} {v:.4f}")
    print("\nSaved: models/position_model.joblib, models/metadata.json, models/test_predictions.csv")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Path to a CSV dataset (default: bundled synthetic dataset)")
    ap.add_argument("--features", nargs="+", help="Signal column names, in order")
    ap.add_argument("--targets", nargs=2, help="X and Y column names")
    ap.add_argument("--source", help="Citation / description of the dataset")
    ap.add_argument("--units", default="m", help="Position units in the dataset (default m)")
    ap.add_argument("--real", action="store_true", help="Mark a custom CSV as real (non-synthetic) data")
    a = ap.parse_args()
    try:
        main(a.csv, a.features, a.targets, a.source, False if a.real else None, a.units)
    except Exception as exc:  # friendly message instead of a long traceback
        print(f"\nTraining failed: {exc}")
        sys.exit(1)
