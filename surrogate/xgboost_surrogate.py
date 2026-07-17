#!/usr/bin/env python3
"""
Train an XGBoost surrogate model using FLAML on the target-encoded dataset,
then save it to disk for use with NSGA-II.
"""

import os
import sys
import re
import copy
import warnings
import logging
import contextlib
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logging.getLogger("flaml").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)

from flaml import AutoML
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
TARGET_COL  = "ETS"
TIME_BUDGET = 300             
N_FOLDS     = 10              
DATASET     = "dataset_one_hot_encoded.csv"
OUTPUT_PATH = "xgboost_surrogate_ohe.pkl"

# ── XGBoost sklearn baseline (choose-better fallback) ─────────────────────────
XGBOOST_BASELINE = XGBRegressor(
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = SEED,
    verbosity         = 0,
    n_jobs            = -1,
)

# ── Stderr suppressor ─────────────────────────────────────────────────────────
@contextlib.contextmanager
def suppress_stderr():
    original_fd = sys.stderr.fileno()
    saved_fd    = os.dup(original_fd)
    devnull_fd  = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, original_fd)
        yield
    finally:
        os.dup2(saved_fd, original_fd)
        os.close(saved_fd)
        os.close(devnull_fd)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for col in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
        if clean and clean[0].isdigit():
            clean = "f_" + clean
        if not clean:
            clean = "feature"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols
    return df


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target '{TARGET_COL}' in {filepath}")
    df = df.drop(columns=["RMSE"], errors="ignore")   # drop RMSE, keep ETS
    df = sanitize_feature_names(df)
    num_cols = [c for c in df.columns if c != TARGET_COL]
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    print(f"  ✓ Loaded  : {df.shape[0]} samples × {df.shape[1]-1} features")
    print(f"  ✓ Target  : {TARGET_COL}  |  range [{df[TARGET_COL].min():.4f}, {df[TARGET_COL].max():.4f}]")
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}


def print_metrics(label: str, metrics: dict):
    print(f"\n  {'─'*48}")
    print(f"  {label}")
    print(f"  {'─'*48}")
    print(f"    RMSE      : {metrics['RMSE']:.4f}")
    print(f"    MAE       : {metrics['MAE']:.4f}")
    print(f"    MAPE      : {metrics['MAPE(%)']:.2f}%")


def cv_rmse(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """Quick 5-fold CV RMSE used for the choose-better comparison."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmses = []
    for tr, te in kf.split(X):
        est = copy.deepcopy(estimator)
        with suppress_stderr():
            est.fit(X[tr], y[tr])
        rmses.append(float(np.sqrt(mean_squared_error(y[te], est.predict(X[te])))))
    return float(np.mean(rmses))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("XGBOOST SURROGATE TRAINING  —  FLAML HPO + Choose-Better")
    print(f"Target     : {TARGET_COL}")
    print(f"Strategy   : {N_FOLDS}-Fold CV  (80% train / 20% test per fold)")
    print(f"HPO metric : RMSE  (selection criterion for NSGA-II ranking)")
    print(f"Budget     : {TIME_BUDGET}s")
    print(f"Output     : {OUTPUT_PATH}")
    print("=" * 62)

    # 1. Load data ─────────────────────────────────────────────────────────────
    df           = load_dataset(DATASET)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X            = df[feature_cols].values
    y            = df[TARGET_COL].values
    total        = len(X)
    fold_test_sz = total // N_FOLDS

    print(f"\n  ✓ Total samples  : {total}")
    print(f"  ✓ Per fold       : ~{total - fold_test_sz} train / ~{fold_test_sz} test")

    # 2. Baseline XGBoost CV RMSE (choose-better reference) ───────────────────
    print(f"\n Step 1 — Baseline XGBoost (300 trees, lr=0.05, depth=6) ...")
    baseline_rmse = cv_rmse(XGBOOST_BASELINE, X, y)
    print(f"   Baseline CV RMSE : {baseline_rmse:.4f}")

    # 3. FLAML HPO — restricted to XGBoost ────────────────────────────────────
    print(f"\n► Step 2 — FLAML HPO (budget={TIME_BUDGET}s, metric=rmse, estimator=xgboost) ...")
    automl = AutoML()
    with suppress_stderr():
        automl.fit(
            X_train        = X,
            y_train        = y,
            task           = "regression",
            metric         = "rmse",           # selection criterion
            estimator_list = ["xgboost"],
            time_budget    = TIME_BUDGET,
            eval_method    = "cv",
            n_splits       = N_FOLDS,
            seed           = SEED,
            verbose        = 0,
        )

    flaml_surrogate = automl.model.estimator

    # Enforce verbosity=0 on the FLAML-returned XGBoost estimator
    if hasattr(flaml_surrogate, "set_params"):
        try:
            flaml_surrogate.set_params(verbosity=0, n_jobs=-1)
        except Exception:
            pass

    flaml_rmse = cv_rmse(flaml_surrogate, X, y)

    print(f"   FLAML best config : {automl.best_config}")
    print(f"   FLAML CV RMSE     : {flaml_rmse:.4f}")

    # 4. Choose-better rule ────────────────────────────────────────────────────
    print(f"\n Step 3 — Choose-better rule (criterion: lowest CV RMSE) ...")
    if flaml_rmse <= baseline_rmse:
        surrogate   = flaml_surrogate
        method_used = "flaml_tuned"
        chosen_rmse = flaml_rmse
        print(f"   FLAML selected  ({flaml_rmse:.4f} ≤ baseline {baseline_rmse:.4f})")
    else:
        surrogate   = copy.deepcopy(XGBOOST_BASELINE)
        method_used = "baseline_chosen"
        chosen_rmse = baseline_rmse
        print(f"   Baseline selected ({baseline_rmse:.4f} < FLAML {flaml_rmse:.4f})")
        print(f"    FLAML did not improve over baseline — using default XGBoost")

    # 5. Full 10-fold CV on chosen surrogate ───────────────────────────────────
    print(f"\n  {'─'*62}")
    print(f"  {N_FOLDS}-FOLD CV on selected surrogate  [{method_used}]")
    print(f"  {'─'*62}")
    print(
        f"  {'Fold':>5} | {'Train RMSE':>11} | {'Test RMSE':>10} | "
        f"{'Train MAE':>10} | {'Test MAE':>9} | "
        f"{'Train R²':>9} | {'Test R²':>8} | {'MAPE':>8}"
    )
    print(f"  {'─'*62}")

    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_records = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        est = copy.deepcopy(surrogate)
        with suppress_stderr():
            est.fit(X_tr, y_tr)

        train_m = compute_metrics(y_tr, est.predict(X_tr))
        test_m  = compute_metrics(y_te, est.predict(X_te))

        cv_records.append({
            "fold"       : fold + 1,
            "train_size" : len(tr_idx),
            "test_size"  : len(te_idx),
            "train_R2"   : train_m["R2"],
            "test_R2"    : test_m["R2"],
            "train_RMSE" : train_m["RMSE"],
            "test_RMSE"  : test_m["RMSE"],
            "train_MAE"  : train_m["MAE"],
            "test_MAE"   : test_m["MAE"],
            "test_MAPE"  : test_m["MAPE(%)"],
        })

        print(
            f"  {fold+1:>5} | "
            f"{train_m['RMSE']:>11.4f} | {test_m['RMSE']:>10.4f} | "
            f"{train_m['MAE']:>10.4f} | {test_m['MAE']:>9.4f} | "
            f"{train_m['R2']:>9.4f} | {test_m['R2']:>8.4f} | "
            f"{test_m['MAPE(%)']:>7.2f}%"
        )

    cv_df = pd.DataFrame(cv_records).set_index("fold")

    # 6. Summary ───────────────────────────────────────────────────────────────
    print(f"  {'─'*62}")
    print(
        f"  {'Mean':>5} | "
        f"{cv_df['train_RMSE'].mean():>11.4f} | {cv_df['test_RMSE'].mean():>10.4f} | "
        f"{cv_df['train_MAE'].mean():>10.4f} | {cv_df['test_MAE'].mean():>9.4f} | "
        f"{cv_df['train_R2'].mean():>9.4f} | {cv_df['test_R2'].mean():>8.4f} | "
        f"{cv_df['test_MAPE'].mean():>7.2f}%"
    )
    print(
        f"  {'Std':>5} | "
        f"{cv_df['train_RMSE'].std():>11.4f} | {cv_df['test_RMSE'].std():>10.4f} | "
        f"{cv_df['train_MAE'].std():>10.4f} | {cv_df['test_MAE'].std():>9.4f} | "
        f"{cv_df['train_R2'].std():>9.4f} | {cv_df['test_R2'].std():>8.4f} | "
        f"{cv_df['test_MAPE'].std():>7.2f}%"
    )
    print(f"  {'─'*62}")

    # 7. Overfitting check ─────────────────────────────────────────────────────
    r2_gap   = cv_df['train_R2'].mean()   - cv_df['test_R2'].mean()
    rmse_gap = cv_df['test_RMSE'].mean()  - cv_df['train_RMSE'].mean()
    print(f"\n OVERFITTING CHECK")
    print(f"    R²   gap (train−test) : {r2_gap:+.4f}  "
          f"{' possible overfit' if r2_gap > 0.10 else ' OK'}")
    print(f"    RMSE gap (test−train) : {rmse_gap:+.4f}  "
          f"{' possible overfit' if rmse_gap > 0.05 else ' OK'}")

    # Note: XGBoost typically fits training data tightly (near-zero train RMSE
    # is possible with deep trees). A train-test R² gap up to 0.20 is normal
    # for gradient boosting on small datasets. Monitor test_RMSE stability
    # across folds — if std is small the model generalises well regardless of gap.

    # 8. Save per-fold metrics ─────────────────────────────────────────────────
    cv_df.to_csv("xgboost_cv_metrics.csv")
    print(f"\n   Per-fold metrics saved -> xgboost_cv_metrics.csv")

    # 9. Final refit on ALL data ───────────────────────────────────────────────
    print(f"\n► Final refit on full dataset ({total} samples) ...")
    with suppress_stderr():
        surrogate.fit(X, y)

    final_preds   = surrogate.predict(X)
    final_metrics = compute_metrics(y, final_preds)
    print_metrics("FINAL XGBOOST MODEL (full-data refit)", final_metrics)

    # 10. Save surrogate + metadata ────────────────────────────────────────────
    target_range = float(y.max() - y.min())
    rel_rmse     = cv_df['test_RMSE'].mean() / target_range * 100 if target_range > 0 else None

    payload = {
        "surrogate"    : surrogate,
        "model_type"   : "xgboost",
        "method_used"  : method_used,
        "feature_cols" : feature_cols,
        "target_col"   : TARGET_COL,
        "X_min"        : X.min(axis=0),
        "X_max"        : X.max(axis=0),
        "X_mean"       : X.mean(axis=0),
        "target_range" : (float(y.min()), float(y.max())),
        "metrics"      : {
            "cv_test_RMSE_mean"  : round(cv_df['test_RMSE'].mean(), 4),
            "cv_test_RMSE_std"   : round(cv_df['test_RMSE'].std(),  4),
            "cv_test_MAE_mean"   : round(cv_df['test_MAE'].mean(),  4),
            "cv_test_MAE_std"    : round(cv_df['test_MAE'].std(),   4),
            "cv_test_R2_mean"    : round(cv_df['test_R2'].mean(),   4),
            "cv_test_R2_std"     : round(cv_df['test_R2'].std(),    4),
            "cv_test_MAPE_mean"  : round(cv_df['test_MAPE'].mean(), 4),
            "relative_RMSE_pct"  : round(rel_rmse, 2) if rel_rmse else None,
            "final_refit"        : final_metrics,
        },
        "flaml_config" : automl.best_config,
        "baseline_rmse": round(baseline_rmse, 4),
        "flaml_rmse"   : round(flaml_rmse,    4),
    }

    joblib.dump(payload, OUTPUT_PATH)

    print(f"\n{'='*62}")
    print(f" XGBoost surrogate saved  ->  {OUTPUT_PATH}")
    print(f"  Features ({len(feature_cols)}) : {feature_cols}")
    print(f"\n  CV summary (selection criterion = test RMSE):")
    print(f"    Test RMSE  : {cv_df['test_RMSE'].mean():.4f} ± {cv_df['test_RMSE'].std():.4f}")
    print(f"    Test MAE   : {cv_df['test_MAE'].mean():.4f} ± {cv_df['test_MAE'].std():.4f}")
    if rel_rmse:
        print(f"    Rel. RMSE  : {rel_rmse:.1f}% of target range "
              f"[{y.min():.4f}, {y.max():.4f}]")
    print(f"    Method     : {method_used}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
