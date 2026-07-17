#!/usr/bin/env python3
"""
FLAML Surrogate Benchmark — Paper Figure Edition
=================================================
  All figures saved to paper_figures/ at 300 dpi.
  Run this script once for RMSE target (S1) and once for ETS target (S2)
  by changing TARGET_COL below.
"""

import os
import sys
import re
import copy
import time
import logging
import warnings
import contextlib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)
logging.getLogger("flaml").setLevel(logging.ERROR)
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

from flaml import AutoML
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# ── Configuration ─────────────────────────────────────────────────────────────

SEED        = 42
TARGET_COL  = "RMSE"       # change to "ETS" for Surrogate S2
N_FOLDS     = 10
TIME_BUDGET = 300

FIGURES_DIR = "paper_figures_surrogte_rmse"
DPI         = 300
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
PURPLE = "#9467bd"
GRAY   = "#7f7f7f"
TEAL   = "#17becf"
COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, TEAL]

MODEL_LABELS: Dict[str, str] = {
    "dt"       : "Decision Tree",
    "rf"       : "Random Forest",
    "gbr"      : "Gradient Boost",
    "et"       : "Extra Trees",
    "xgboost"  : "XGBoost",
    "catboost" : "CatBoost",
    "lightgbm" : "LightGBM",
}

MODEL_ORDER   = ["dt", "rf", "gbr", "et", "xgboost", "catboost", "lightgbm"]
MODEL_DISPLAY = [MODEL_LABELS[k] for k in MODEL_ORDER]

ENC_ORDER   = ["target", "binary", "ohe"]
ENC_DISPLAY = {"target": "Target\nEncoding", "binary": "Binary\nEncoding", "ohe": "One-Hot\nEncoding"}
ENC_COLORS  = {enc: c for enc, c in zip(ENC_ORDER, [BLUE, ORANGE, GREEN])}

FLAML_UNSUPPORTED = {"gbr"}

FLAML_ESTIMATOR_MAP: Dict[str, str] = {
    "dt"       : "xgb_limitdepth",
    "rf"       : "rf",
    "et"       : "extra_tree",
    "xgboost"  : "xgboost",
    "catboost" : "catboost",
    "lightgbm" : "lgbm",
}

SKLEARN_ESTIMATORS: Dict[str, object] = {
    "dt"       : DecisionTreeRegressor(random_state=SEED),
    "rf"       : RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "gbr"      : GradientBoostingRegressor(random_state=SEED),
    "et"       : ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "xgboost"  : XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1),
    "catboost" : CatBoostRegressor(random_state=SEED, verbose=0),
    "lightgbm" : LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=-1),
}

DATASETS = {
    "target" : "dataset_encoded_target.csv",
    "binary" : "dataset_encoded_binary.csv",
    "ohe"    : "dataset_one_hot_encoded.csv",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for col in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
        if clean and clean[0].isdigit(): clean = "f_" + clean
        if not clean: clean = "feature"
        if clean in seen:
            seen[clean] += 1; clean = f"{clean}_{seen[clean]}"
        else: seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols
    return df


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    other = "ETS" if TARGET_COL == "RMSE" else "RMSE"
    df = df.drop(columns=[other], errors="ignore")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target '{TARGET_COL}' in {filepath}")
    df = sanitize_feature_names(df)
    num_cols = [c for c in df.columns if c != TARGET_COL]
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    n, f = df.shape[0], df.shape[1] - 1
    print(f"  ✓ {filepath.split('/')[-1]}: {n} samples × {f} features")
    return df


def cross_validate_model(estimator, X, y, n_folds=N_FOLDS, seed=SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    records = []
    for fold_idx, (tr, val) in enumerate(kf.split(X)):
        est = copy.deepcopy(estimator)
        est.fit(X[tr], y[tr])
        preds = est.predict(X[val])
        records.append({
            "fold": fold_idx,
            "R2"  : r2_score(y[val], preds),
            "RMSE": float(np.sqrt(mean_squared_error(y[val], preds))),
            "MAE" : mean_absolute_error(y[val], preds),
            "y_true": y[val].tolist(),
            "y_pred": preds.tolist(),
        })
    fold_df = pd.DataFrame(records).set_index("fold")
    return (fold_df,
            float(fold_df["R2"].mean()),   float(fold_df["R2"].std()),
            float(fold_df["RMSE"].mean()), float(fold_df["RMSE"].std()),
            float(fold_df["MAE"].mean()),  float(fold_df["MAE"].std()))


def flaml_tune_model(model_key, X, y, time_budget=TIME_BUDGET, seed=SEED):
    automl = AutoML()
    with suppress_stderr():
        automl.fit(X_train=X, y_train=y, task="regression",
                   metric="mae", estimator_list=[FLAML_ESTIMATOR_MAP[model_key]],
                   time_budget=time_budget, seed=seed, verbose=0,
                   eval_method="cv", n_splits=N_FOLDS)
    return automl.model.estimator


def run_global_flaml(X, y, enc_name, time_budget=TIME_BUDGET * 2, seed=SEED):
    print(f"\n  [global FLAML] unrestricted (budget={time_budget}s, metric=rmse) ...")
    automl = AutoML()
    with suppress_stderr():
        automl.fit(X_train=X, y_train=y, task="regression",
                   metric="mae", time_budget=time_budget,
                   seed=seed, verbose=0, eval_method="cv", n_splits=N_FOLDS)
    best_est = automl.model.estimator
    fold_df, r2, sr2, rmse, srmse, mae, smae = cross_validate_model(best_est, X, y)
    result = {"encoding": enc_name, "best_estimator": automl.best_estimator,
              "best_config": str(automl.best_config),
              "avg_r2": round(r2, 4), "std_r2": round(sr2, 4),
              "avg_rmse": round(rmse, 4), "std_rmse": round(srmse, 4),
              "avg_mae": round(mae, 4), "std_mae": round(smae, 4)}
    print(f"    → {automl.best_estimator}  R2={r2:.4f}±{sr2:.4f} | "
          f"RMSE={rmse:.4f}±{srmse:.4f} | MAE={mae:.4f}±{smae:.4f}")
    return result


def save_model_csv(fold_df, enc_name, model_key):
    os.makedirs("results", exist_ok=True)
    label = MODEL_LABELS[model_key]
    out   = fold_df.drop(columns=["y_true", "y_pred"], errors="ignore").copy()
    out.insert(0, "encoding", enc_name)
    out.insert(1, "model", label)
    out.index.name = "fold"
    filename = f"results/{enc_name}_{label.replace(' ', '_')}.csv"
    out.to_csv(filename)
    return filename


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_flaml_benchmark(df, enc_name):
    results      = []
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    global_result = run_global_flaml(X, y, enc_name)

    print(f"\n  [baseline CV — {N_FOLDS}-fold] ...")
    baseline_rmse = {}
    for model_key, estimator in SKLEARN_ESTIMATORS.items():
        label = MODEL_LABELS[model_key]
        try:
            with suppress_stderr():
                fd, r2, sr2, rmse, srmse, mae, smae = cross_validate_model(estimator, X, y)
            baseline_rmse[model_key] = rmse
            print(f"    · {label:18s}  R2={r2:.4f}±{sr2:.4f} | RMSE={rmse:.4f}±{srmse:.4f} | MAE={mae:.4f}±{smae:.4f}")
        except Exception as e:
            print(f"   {label} failed: {e}")
            baseline_rmse[model_key] = np.inf

    sorted_keys = sorted(baseline_rmse, key=lambda k: baseline_rmse[k])

    print(f"\n  [FLAML HPO — {TIME_BUDGET}s/model, metric=rmse] ...")
    for model_key in sorted_keys:
        label = MODEL_LABELS[model_key]
        t0    = time.time()
        try:
            with suppress_stderr():
                if model_key in FLAML_UNSUPPORTED:
                    print(f"    · {label:18s}  [sklearn baseline — no FLAML proxy]")
                    fd, r2, sr2, rmse, srmse, mae, smae = cross_validate_model(SKLEARN_ESTIMATORS[model_key], X, y)
                    method_used = "sklearn_baseline"
                else:
                    best_est = flaml_tune_model(model_key, X, y)
                    fd, r2, sr2, rmse, srmse, mae, smae = cross_validate_model(best_est, X, y)
                    if rmse > baseline_rmse[model_key]:
                        fd, r2, sr2, rmse, srmse, mae, smae = cross_validate_model(SKLEARN_ESTIMATORS[model_key], X, y)
                        method_used = "baseline_chosen"
                    else:
                        method_used = "flaml_tuned"

            elapsed  = time.time() - t0
            csv_file = save_model_csv(fd, enc_name, model_key)

            # collect all y_true / y_pred across folds for scatter plot
            all_true = np.concatenate([row for row in fd["y_true"]])
            all_pred = np.concatenate([row for row in fd["y_pred"]])

            results.append({
                "encoding"  : enc_name,
                "model_key" : model_key,
                "model"     : label,
                "method"    : method_used,
                "avg_r2"    : round(r2,    4), "std_r2"   : round(sr2,   4),
                "avg_rmse"  : round(rmse,  4), "std_rmse" : round(srmse, 4),
                "avg_mae"   : round(mae,   4), "std_mae"  : round(smae,  4),
                "time_s"    : round(elapsed, 1),
                "csv_file"  : csv_file,
                "y_true"    : all_true,
                "y_pred"    : all_pred,
                "X"         : X,
                "y"         : y,
            })
            print(f"   {label:18s}  R2={r2:.4f}±{sr2:.4f} | RMSE={rmse:.4f}±{srmse:.4f} | MAE={mae:.4f}±{smae:.4f}  ({elapsed:.0f}s) [{method_used}]")
        except Exception as e:
            print(f"    ✗ {label} failed: {e}")
    return results, global_result


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3a — R² HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig3a_r2_heatmap(summary: pd.DataFrame, target: str):
    """
    Fig 3a: R² heatmap — 7 models × 3 encodings.
    Labelled as reporting metric (not selection).
    """
    mat = np.full((len(MODEL_ORDER), len(ENC_ORDER)), np.nan)
    for i, mk in enumerate(MODEL_ORDER):
        label = MODEL_LABELS[mk]
        for j, enc in enumerate(ENC_ORDER):
            row = summary[(summary["model"] == label) & (summary["encoding"] == enc)]
            if not row.empty:
                mat[i, j] = row["avg_r2"].values[0]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle(f"Figure 3a — Surrogate R² Heatmap  [{target} objective — reporting metric]",
                 fontsize=11, fontweight="bold", y=1.01)

    vmin = max(0, np.nanmin(mat) - 0.02)
    im   = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(ENC_ORDER)))
    ax.set_xticklabels([ENC_DISPLAY[e] for e in ENC_ORDER], fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_DISPLAY, fontsize=10)
    ax.set_xlabel("Encoding strategy", fontsize=11)
    ax.set_ylabel("Algorithm", fontsize=11)
    plt.colorbar(im, ax=ax, label="Mean R² (10-fold CV)", fraction=0.046, pad=0.04)

    for i in range(len(MODEL_ORDER)):
        for j in range(len(ENC_ORDER)):
            val = mat[i, j]
            if not np.isnan(val):
                tc = "white" if val > 0.78 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold")

    # mark best cell
    bi, bj = divmod(int(np.nanargmax(mat)), len(ENC_ORDER))
    rect = plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                          fill=False, edgecolor="red", linewidth=2.5)
    ax.add_patch(rect)
    ax.text(bj, bi + 0.42, " best", ha="center", va="center",
            fontsize=7.5, color="red", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"Fig3a_R2_heatmap_{target}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  ✓ Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3b — RMSE HEATMAP  (selection basis)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig3b_rmse_heatmap(summary: pd.DataFrame, target: str):
    """
    Fig 3b: RMSE heatmap — 7 models × 3 encodings.
    This is the SELECTION metric: lowest RMSE = selected surrogate.
    Selected cell marked with green star.
    """
    mat = np.full((len(MODEL_ORDER), len(ENC_ORDER)), np.nan)
    for i, mk in enumerate(MODEL_ORDER):
        label = MODEL_LABELS[mk]
        for j, enc in enumerate(ENC_ORDER):
            row = summary[(summary["model"] == label) & (summary["encoding"] == enc)]
            if not row.empty:
                mat[i, j] = row["avg_rmse"].values[0]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle(f"Figure 3b — Surrogate RMSE Heatmap  [{target} objective — SELECTION metric ★]",
                 fontsize=11, fontweight="bold", y=1.01)

    # Invert colormap: lower RMSE = darker = better
    vmax = np.nanmax(mat) + 0.01
    vmin = max(0, np.nanmin(mat) - 0.01)
    im   = ax.imshow(mat, cmap="Reds_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(ENC_ORDER)))
    ax.set_xticklabels([ENC_DISPLAY[e] for e in ENC_ORDER], fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_DISPLAY, fontsize=10)
    ax.set_xlabel("Encoding strategy", fontsize=11)
    ax.set_ylabel("Algorithm", fontsize=11)
    plt.colorbar(im, ax=ax, label="Mean Surrogate RMSE (10-fold CV)", fraction=0.046, pad=0.04)

    for i in range(len(MODEL_ORDER)):
        for j in range(len(ENC_ORDER)):
            val = mat[i, j]
            if not np.isnan(val):
                # dark cells get white text
                norm_val = (val - vmin) / (vmax - vmin + 1e-9)
                tc = "white" if norm_val < 0.35 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold")

    # mark best (lowest RMSE) cell with green star
    bi, bj = divmod(int(np.nanargmin(mat)), len(ENC_ORDER))
    rect = plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                          fill=False, edgecolor=GREEN, linewidth=3.0)
    ax.add_patch(rect)
    ax.text(bj, bi + 0.42, "★ selected", ha="center", va="center",
            fontsize=7.5, color=GREEN, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"Fig3b_RMSE_heatmap_{target}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"   Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ENCODING EFFECT BOXPLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4_encoding_boxplot(summary: pd.DataFrame, target: str):
    """
    Fig 4: Side-by-side boxplots showing R² and RMSE distribution
    across 7 models for each encoding strategy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Figure 4 — Effect of Encoding Strategy on Surrogate Performance  [{target} objective]",
                 fontsize=12, fontweight="bold", y=1.01)

    metrics = [("avg_r2", "Mean R² (10-fold CV)", True),
               ("avg_rmse", "Mean Surrogate RMSE (10-fold CV)", False)]

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        data   = [summary[summary["encoding"] == enc][metric].dropna().values
                  for enc in ENC_ORDER]
        colors = [ENC_COLORS[enc] for enc in ENC_ORDER]

        bp = ax.boxplot(data,
                        labels=[ENC_DISPLAY[e] for e in ENC_ORDER],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2.2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=5, alpha=0.5))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color + "88")

        for j, (vals, color) in enumerate(zip(data, colors)):
            jitter = np.random.default_rng(j).uniform(-0.13, 0.13, len(vals))
            ax.scatter(np.ones(len(vals)) * (j + 1) + jitter, vals,
                       color=color, s=45, alpha=0.75, zorder=4,
                       edgecolors="k", linewidths=0.4)

        # Annotate medians
        for j, vals in enumerate(data):
            if len(vals) > 0:
                med = np.median(vals)
                ax.text(j + 1, med + (0.003 if higher_better else -0.003),
                        f"{med:.3f}", ha="center",
                        va="bottom" if higher_better else "top",
                        fontsize=8.5, fontweight="bold", color="black")

        direction = "↑ higher = better" if higher_better else "↓ lower = better"
        ax.set_ylabel(f"{ylabel}\n({direction})", fontsize=10)
        ax.set_title(f"({'a' if higher_better else 'b'}) {ylabel.split('(')[0].strip()}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"Fig4_encoding_boxplot_{target}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"   Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — RADAR CHART PER ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig5_radar(summary: pd.DataFrame, target: str):
    """
    Fig 5: Radar chart showing R², 1-RMSE_norm, 1-MAE_norm for all 7 models
    per encoding strategy. Three panels side by side (one per encoding).
    Higher values on all axes = better.
    """
    metrics_raw = ["avg_r2", "avg_rmse", "avg_mae"]
    metric_labels = ["R²", "1 − RMSE\n(normalized)", "1 − MAE\n(normalized)"]
    N = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                             subplot_kw=dict(polar=True))
    fig.suptitle(f"Figure 5 — Multi-Metric Radar Chart per Encoding  [{target} objective]",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, enc in zip(axes, ENC_ORDER):
        sub = summary[summary["encoding"] == enc].copy()

        # Normalize RMSE and MAE to [0,1] then invert so higher = better
        rmse_min, rmse_max = sub["avg_rmse"].min(), sub["avg_rmse"].max()
        mae_min,  mae_max  = sub["avg_mae"].min(),  sub["avg_mae"].max()
        sub["rmse_norm"] = 1 - (sub["avg_rmse"] - rmse_min) / (rmse_max - rmse_min + 1e-9)
        sub["mae_norm"]  = 1 - (sub["avg_mae"]  - mae_min)  / (mae_max  - mae_min  + 1e-9)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="gray")
        ax.grid(color="gray", alpha=0.3)
        ax.set_title(f"{ENC_DISPLAY[enc].replace(chr(10),' ')}", fontsize=10,
                     fontweight="bold", pad=15)

        legend_handles = []
        for idx, mk in enumerate(MODEL_ORDER):
            label = MODEL_LABELS[mk]
            row   = sub[sub["model"] == label]
            if row.empty: continue
            vals = [float(row["avg_r2"]),
                    float(row["rmse_norm"]),
                    float(row["mae_norm"])]
            vals += vals[:1]
            color = COLORS[idx % len(COLORS)]
            ax.plot(angles, vals, color=color, linewidth=1.8, linestyle="solid")
            ax.fill(angles, vals, color=color, alpha=0.08)
            legend_handles.append(Line2D([0], [0], color=color, linewidth=2,
                                         label=label))

        if enc == ENC_ORDER[-1]:
            ax.legend(handles=legend_handles, loc="upper left",
                      bbox_to_anchor=(1.25, 1.1), fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"Fig5_radar_chart_{target}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"   Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig6_learning_curve(results: list, target: str):
    """
    Fig 6: Learning curve for the best model (lowest avg_rmse) per encoding.
    Shows how surrogate RMSE evolves as training size increases.
    Answers: is 700 samples sufficient?
    """
    # Find best model per encoding
    summary_tmp = pd.DataFrame([{k: v for k, v in r.items()
                                  if k not in ("y_true","y_pred","X","y")}
                                  for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f"Figure 6 — Learning Curve: Surrogate RMSE vs Training Size  [{target} objective]",
                 fontsize=12, fontweight="bold", y=1.01)

    train_sizes = np.linspace(0.10, 1.0, 10)

    for ax, enc in zip(axes, ENC_ORDER):
        sub = summary_tmp[summary_tmp["encoding"] == enc]
        if sub.empty: continue
        best_row = sub.loc[sub["avg_rmse"].idxmin()]
        mk       = best_row["model_key"]
        label    = best_row["model"]

        # Get X, y from results list
        res_row  = next(r for r in results
                        if r["encoding"] == enc and r["model_key"] == mk)
        X, y     = res_row["X"], res_row["y"]
        estimator = copy.deepcopy(SKLEARN_ESTIMATORS[mk])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr_sizes, tr_scores, val_scores = learning_curve(
                estimator, X, y,
                train_sizes=train_sizes,
                cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

        tr_mean  = -tr_scores.mean(axis=1)
        tr_std   =  tr_scores.std(axis=1)
        val_mean = -val_scores.mean(axis=1)
        val_std  =  val_scores.std(axis=1)

        ax.plot(tr_sizes, tr_mean,  color=BLUE,   linewidth=2, label="Training RMSE")
        ax.fill_between(tr_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color=BLUE)
        ax.plot(tr_sizes, val_mean, color=ORANGE, linewidth=2,
                linestyle="--", label="Validation RMSE")
        ax.fill_between(tr_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color=ORANGE)

        # Mark convergence: where improvement < 1% between last two points
        diffs = np.abs(np.diff(val_mean)) / (val_mean[:-1] + 1e-9)
        converge_idx = next((i+1 for i, d in enumerate(diffs) if d < 0.01), None)
        if converge_idx:
            ax.axvline(tr_sizes[converge_idx], color=GREEN, linewidth=1.5,
                       linestyle=":", alpha=0.8)
            ax.text(tr_sizes[converge_idx] + len(X) * 0.01,
                    val_mean.max() * 0.98,
                    f"~{tr_sizes[converge_idx]} samples\n(convergence)",
                    fontsize=7.5, color=GREEN, va="top")

        enc_title = ENC_DISPLAY[enc].replace("\n", " ")
        ax.set_title(f"{enc_title}\nBest model: {label}", fontsize=9.5,
                     fontweight="bold")
        ax.set_xlabel("Number of training samples", fontsize=10)
        ax.set_ylabel("Surrogate RMSE", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"Fig6_learning_curve_{target}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — PREDICTED vs TRUE SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig7_predicted_vs_true(results: list, summary: pd.DataFrame, target: str):
    """
    Fig 7: Predicted vs True scatter for the best model per encoding.
    One panel per encoding strategy (3 panels).
    Shows surrogate fidelity — the most direct evidence for reviewers.
    Points coloured by density. 1:1 reference line shown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(f"Figure 7 — Surrogate Predicted vs WRF-Computed {target}  [best model per encoding]",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, enc in zip(axes, ENC_ORDER):
        sub = summary[summary["encoding"] == enc]
        if sub.empty: continue
        best_row = sub.loc[sub["avg_rmse"].idxmin()]
        mk       = best_row["model_key"]
        label    = best_row["model"]
        r2_val   = best_row["avg_r2"]
        rmse_val = best_row["avg_rmse"]

        res_row  = next((r for r in results
                         if r["encoding"] == enc and r["model_key"] == mk), None)
        if res_row is None: continue

        y_true = np.array(res_row["y_true"])
        y_pred = np.array(res_row["y_pred"])

        # Scatter with density coloring
        from matplotlib.colors import Normalize as MNorm
        from scipy.stats import gaussian_kde
        try:
            xy  = np.vstack([y_true, y_pred])
            kde = gaussian_kde(xy)(xy)
            idx = kde.argsort()
            sc  = ax.scatter(y_true[idx], y_pred[idx], c=kde[idx],
                             cmap="plasma", s=18, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Density", fraction=0.046, pad=0.04)
        except Exception:
            ax.scatter(y_true, y_pred, color=ENC_COLORS[enc],
                       s=18, alpha=0.6, edgecolors="none")

        # 1:1 line
        mn = min(y_true.min(), y_pred.min()) * 0.98
        mx = max(y_true.max(), y_pred.max()) * 1.02
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.5, label="1:1 line")

        # ±10% band
        ax.fill_between([mn, mx], [mn*0.9, mx*0.9], [mn*1.1, mx*1.1],
                        alpha=0.08, color="gray", label="±10% band")

        enc_title = ENC_DISPLAY[enc].replace("\n", " ")
        ax.set_title(f"{enc_title}\n{label}",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel(f"WRF-computed {target} (true)", fontsize=10)
        ax.set_ylabel(f"Surrogate-predicted {target}", fontsize=10)
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # Annotate metrics
        ax.text(0.04, 0.95,
                f"R² = {r2_val:.3f}\nRMSE = {rmse_val:.4f}",
                transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"Fig7_predicted_vs_true_{target}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"   Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    all_results     = []
    all_global_best = []

    print(f"\n{'='*72}")
    print(f"SURROGATE BENCHMARK  --  FLAML Paper Figure Edition")
    print(f"Target          : {TARGET_COL}  (change to 'ETS' for Surrogate S2)")
    print(f"Models          : {', '.join(MODEL_LABELS.values())}")
    print(f"CV              : {N_FOLDS}-fold | HPO: FLAML {TIME_BUDGET}s | Seed: {SEED}")
    print(f"Selection crit. : lowest external CV RMSE")
    print(f"Figures         : {FIGURES_DIR}/")
    print(f"{'='*72}")

    # ── Run benchmark ──────────────────────────────────────────────────────────
    for enc_name, filepath in DATASETS.items():
        print(f"\n► ENCODING: {enc_name.upper()}")
        try:
            df = load_dataset(filepath)
        except Exception as e:
            print(f"  ✗ Load failed: {e}"); continue
        try:
            results, global_result = run_flaml_benchmark(df, enc_name)
            all_results.extend(results)
            all_global_best.append(global_result)
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}"); continue

    if not all_results:
        print("\n✗ No results — check dataset paths."); return

    # ── Build summary ──────────────────────────────────────────────────────────
    summary = pd.DataFrame([{k: v for k, v in r.items()
                              if k not in ("y_true","y_pred","X","y")}
                              for r in all_results])
    summary = summary.sort_values("avg_rmse").reset_index(drop=True)

    summary["selected"] = False
    for enc in summary["encoding"].unique():
        best_idx = summary[summary["encoding"] == enc]["avg_rmse"].idxmin()
        summary.loc[best_idx, "selected"] = True

    summary.to_csv("flaml_benchmark_results.csv", index=False)

    # Print full table
    print(f"\n{'='*72}")
    print("FULL RESULTS (sorted by avg_rmse):")
    print(summary[["encoding","model","method",
                   "avg_r2","std_r2","avg_rmse","std_rmse",
                   "avg_mae","std_mae","selected","time_s"]].to_string(index=False))

    # Selection summary
    sel = summary[summary["selected"]]
    print(f"\n{'='*72}")
    print("SELECTED SURROGATE PER ENCODING  (criterion: lowest RMSE):")
    print(f"  {'Encoding':10} | {'Model':18} | {'RMSE':>10} ± {'std':>8} | {'R²':>8}")
    print(f"  {'-'*65}")
    for _, row in sel.iterrows():
        print(f"  {row['encoding']:10} | {row['model']:18} | "
              f"{row['avg_rmse']:>10.4f} ± {row['std_rmse']:>8.4f} | {row['avg_r2']:>8.4f}")

    best = summary.iloc[0]
    print(f"\n  ★ Overall best: [{best['encoding']}] {best['model']}")
    print(f"    RMSE={best['avg_rmse']:.4f}±{best['std_rmse']:.4f}  "
          f"R²={best['avg_r2']:.4f}  MAE={best['avg_mae']:.4f}")
    print(f"    → Will be used as Surrogate for {TARGET_COL} in NSGA-II")

    # Save global best
    if all_global_best:
        pd.DataFrame(all_global_best).to_csv("flaml_global_best.csv", index=False)

    # ── Generate all paper figures ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Generating paper figures -> {FIGURES_DIR}/")

    plot_fig3a_r2_heatmap(summary, TARGET_COL)
    plot_fig3b_rmse_heatmap(summary, TARGET_COL)
    plot_fig4_encoding_boxplot(summary, TARGET_COL)
    plot_fig5_radar(summary, TARGET_COL)
    plot_fig6_learning_curve(all_results, TARGET_COL)
    plot_fig7_predicted_vs_true(all_results, summary, TARGET_COL)

    print(f"\n{'='*72}")
    print(f"All figures saved to {FIGURES_DIR}/:")
    print(f"  Fig3a  — R² heatmap         (reporting metric)")
    print(f"  Fig3b  — RMSE heatmap        (SELECTION metric ★)")
    print(f"  Fig4   — Encoding boxplot    (R² and RMSE distributions)")
    print(f"  Fig5   — Radar chart         (multi-metric per encoding)")
    print(f"  Fig6   — Learning curve      (RMSE vs training size)")
    print(f"  Fig7   — Predicted vs True   (surrogate fidelity scatter)")
    print(f"\nRun again with TARGET_COL = 'ETS' for Surrogate S2 figures.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
