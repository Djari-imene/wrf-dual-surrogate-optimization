#!/usr/bin/env python3
"""
NSGA-II Multi-Objective Genetic Algorithm — Final Paper Edition
===============================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Global figure style — larger fonts for readability across ALL figures ─────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 19,
    "axes.linewidth": 1.1,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#BBBBBB",
})
GOLD = "#E8A33D"


# ── Constants ─────────────────────────────────────────────────────────────────
RMSE_SURROGATE_PATH    = "catboost_surrogate_target.pkl"  # CatBoost + target encoding
ETS_SURROGATE_PATH     = "xgboost_surrogate_ohe.pkl"      # XGBoost + OHE encoding
RMSE_LOOKUP_PATH       = "encoding_lookup_table_rmse.csv"
# ETS uses OHE — no lookup table needed
CV_METRICS_RMSE_PATH   = "catboost_cv_metrics.csv"   # from catboost_surrogate.py
CV_METRICS_ETS_PATH    = "xgboost_cv_metrics.csv"    # from xgboost_surrogate.py
PARETO_OUTPUT          = "nsga2_pareto_front.csv"
HISTORY_OUTPUT         = "nsga2_history.csv"
FIGURES_DIR            = "paper_figures"

# NSGA-II Hyperparameters
POP_SIZE        = 100
N_GENERATIONS   = 200
CROSSOVER_RATE  = 0.7
MUTATION_RATE   = 0.143
TOURNAMENT_SIZE = 2
SEED            = 42
N_VERIFY        = 10      # Pareto solutions selected for WRF re-verification

# Publication figure settings
DPI        = 300
FIG_WIDTH  = 10           # inches — single-column-friendly width
FIG_WIDTH2 = 14           # wider for multi-panel figures

rng = np.random.default_rng(SEED)

# ── Physics options ───────────────────────────────────────────────────────────
PHYSICS_OPTIONS = {
    "mp_physics"        : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17],
    "cu_physics"        : [1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 99],
    "bl_pbl_physics"    : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99],
    "ra_lw_physics"     : [1, 3, 4, 5, 7, 99],
    "ra_sw_physics"     : [1, 2, 3, 4, 5, 7, 99],
    "sf_sfclay_physics" : [1, 2, 3, 4, 10],
    "sf_surface_physics": [1, 2, 3, 4, 5],
}

PARAM_LABELS = {
    "mp_physics"        : "Microphysics",
    "cu_physics"        : "Cumulus",
    "bl_pbl_physics"    : "PBL",
    "ra_lw_physics"     : "LW Radiation",
    "ra_sw_physics"     : "SW Radiation",
    "sf_sfclay_physics" : "Surface Layer",
    "sf_surface_physics": "Land Surface",
}

# ── PBL ↔ surface-layer compatibility rule (WRF 4.2.1) ────────────────────────
# If bl_pbl_physics is 2 (MYJ), 3 (GFS), 4 (QNSE) or 10 (TEMF), set
# sf_sfclay_physics equal to bl_pbl_physics (these PBL schemes require their
# own surface layer scheme). Otherwise, set sf_sfclay_physics to 1.
PBL_TIED_SFCLAY = {2, 3, 4, 10}

def repair_sfclay(individual, feature_cols):
    """Enforce the PBL/surface-layer compatibility rule on one individual (in place)."""
    cols    = list(feature_cols)
    pbl_idx = cols.index("bl_pbl_physics")
    sfc_idx = cols.index("sf_sfclay_physics")
    pbl     = int(individual[pbl_idx])
    individual[sfc_idx] = pbl if pbl in PBL_TIED_SFCLAY else 1
    return individual

MODEL_ORDER = [
    "decision_tree", "random_forest", "gradient_boosting",
    "extra_trees", "xgboost", "catboost", "lightgbm"
]
MODEL_DISPLAY = {
    "decision_tree"    : "DT",
    "random_forest"    : "RF",
    "gradient_boosting": "GBR",
    "extra_trees"      : "ET",
    "xgboost"          : "XGB",
    "catboost"         : "CAT",
    "lightgbm"         : "LGB",
}
ENC_ORDER   = ["target", "binary", "ohe"]
ENC_DISPLAY = {"target": "Target\nEncoding", "binary": "Binary\nEncoding", "ohe": "One-Hot\nEncoding"}

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
GREY   = "#aaaaaa"
PURPLE = "#9467bd"

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(fig, name, tight=True):
    path = os.path.join(FIGURES_DIR, name)
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"   Saved -> {path}")
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# SURROGATE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_surrogate(path: str, label: str):
    payload      = joblib.load(path)
    surrogate    = payload["surrogate"]
    feature_cols = payload["feature_cols"]
    print(f"   [{label}] {type(surrogate).__name__}  |  features: {feature_cols}")
    return surrogate, feature_cols

def load_lookup_table(path: str, feature_cols: list, label: str) -> dict:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for c in df.columns:
        if "param"   in c:              col_map[c] = "parameter"
        elif "scheme" in c or c == "id": col_map[c] = "scheme_id"
        elif "encoded" in c:             col_map[c] = "encoded_value"
    df = df.rename(columns=col_map)
    required = {"parameter", "scheme_id", "encoded_value"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"[{label}] Missing columns: {missing}")
    lookup = {}
    for param in feature_cols:
        sub = df[df["parameter"] == param]
        if sub.empty:
            raise ValueError(f"[{label}] '{param}' not in lookup table.")
        lookup[param] = dict(zip(sub["scheme_id"].astype(int),
                                 sub["encoded_value"].astype(float)))
    print(f"   [{label}] Lookup loaded ({len(lookup)} params)")
    return lookup

# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING — two strategies
#   S1 (CatBoost): TARGET ENCODING  — uses lookup table (mean RMSE per scheme)
#   S2 (XGBoost):  OHE ENCODING     — binary indicator columns (no lookup table)
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_NAMES = list(PHYSICS_OPTIONS.keys())   # ordered list of 7 parameter names

# ── Target encoding (S1 — CatBoost) ──────────────────────────────────────────
def encode_individual_target(individual, feature_cols, lookup):
    """Replace each scheme ID with its mean target value from the lookup table."""
    encoded = np.zeros(len(feature_cols), dtype=float)
    for i, param in enumerate(feature_cols):
        sid = int(individual[i])
        if sid not in lookup[param]:
            available = np.array(list(lookup[param].keys()))
            sid = int(available[np.argmin(np.abs(available - sid))])
        encoded[i] = lookup[param][sid]
    return encoded

def encode_population_target(population, feature_cols, lookup):
    return np.array([encode_individual_target(ind, feature_cols, lookup)
                     for ind in population])

# ── OHE encoding (S2 — XGBoost) ──────────────────────────────────────────────
def build_ohe_column_index(feature_cols):
    
    index = {}
    for col_idx, col_name in enumerate(feature_cols):
        # Parse 'param_name_schemeID' — split on last underscore
        parts = col_name.rsplit("_", 1)
        if len(parts) == 2:
            try:
                param   = parts[0]   # e.g. 'mp_physics'
                scheme  = int(parts[1])
                index[(param, scheme)] = col_idx
            except ValueError:
                pass   # column not a physics OHE column — skip
    return index

def encode_individual_ohe(individual, feature_cols, ohe_index):
    
    encoded = np.zeros(len(feature_cols), dtype=float)
    for param_pos, param in enumerate(PARAM_NAMES):
        sid = int(individual[param_pos])
        key = (param, sid)
        if key in ohe_index:
            encoded[ohe_index[key]] = 1.0
        else:
            # Nearest valid scheme fallback
            valid = [k[1] for k in ohe_index if k[0] == param]
            if valid:
                nearest = min(valid, key=lambda x: abs(x - sid))
                encoded[ohe_index[(param, nearest)]] = 1.0
    return encoded

def encode_population_ohe(population, feature_cols, ohe_index):
    return np.array([encode_individual_ohe(ind, feature_cols, ohe_index)
                     for ind in population])

# Keep legacy name for backwards compatibility
def encode_individual(individual, feature_cols, lookup):
    return encode_individual_target(individual, feature_cols, lookup)

def encode_population(population, feature_cols, lookup):
    return encode_population_target(population, feature_cols, lookup)

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_population(population, sur_rmse, fc_rmse, lk_rmse,
                                     sur_ets,  fc_ets,  ohe_index_ets):
    # S1: CatBoost uses target encoding → lookup table
    enc_rmse  = encode_population_target(population, fc_rmse, lk_rmse)
    # S2: XGBoost uses OHE encoding → binary indicator vector
    enc_ets   = encode_population_ohe(population, fc_ets, ohe_index_ets)
    rmse_vals =  sur_rmse.predict(enc_rmse).astype(float)
    ets_vals  = -sur_ets.predict(enc_ets).astype(float)
    return np.column_stack([rmse_vals, ets_vals])

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERVOLUME (2D sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hypervolume_2d(objectives, front_idx, reference_point):
    front_obj = objectives[front_idx]
    valid     = np.all(front_obj < reference_point, axis=1)
    front_obj = front_obj[valid]
    if len(front_obj) == 0:
        return 0.0
    sort_order = np.argsort(front_obj[:, 0])
    front_obj  = front_obj[sort_order]
    hv, prev_x = 0.0, reference_point[0]
    for i in range(len(front_obj) - 1, -1, -1):
        width  = prev_x - front_obj[i, 0]
        height = reference_point[1] - front_obj[i, 1]
        hv    += width * height
        prev_x = front_obj[i, 0]
    return float(hv)

# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II CORE
# ═══════════════════════════════════════════════════════════════════════════════

def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))

def fast_non_dominated_sort(objectives):
    N = len(objectives)
    domination_count = np.zeros(N, dtype=int)
    dominated_by     = [[] for _ in range(N)]
    fronts           = [[]]
    for i in range(N):
        for j in range(i + 1, N):
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j); domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_by[j].append(i); domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)
    return [f for f in fronts if f]

def crowding_distance(objectives, front):
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    distances = np.zeros(n)
    obj_front = objectives[front]
    for m in range(obj_front.shape[1]):
        sorted_idx = np.argsort(obj_front[:, m])
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
        obj_range = obj_front[sorted_idx[-1], m] - obj_front[sorted_idx[0], m]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                obj_front[sorted_idx[k + 1], m] - obj_front[sorted_idx[k - 1], m]
            ) / obj_range
    return distances

def nsga2_selection(population, objectives, fronts, pop_size):
    new_pop_idx = []
    for front in fronts:
        if len(new_pop_idx) + len(front) <= pop_size:
            new_pop_idx.extend(front)
        else:
            needed   = pop_size - len(new_pop_idx)
            cd       = crowding_distance(objectives, front)
            cd_order = np.argsort(-cd)
            new_pop_idx.extend([front[i] for i in cd_order[:needed]])
            break
    return population[new_pop_idx].copy(), objectives[new_pop_idx].copy()

def binary_tournament(population, objectives, fronts):
    N    = len(population)
    rank = np.zeros(N, dtype=int)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r
    cd_all = np.zeros(N)
    for front in fronts:
        if len(front) < 2:
            for idx in front:
                cd_all[idx] = np.inf
            continue
        cd = crowding_distance(objectives, front)
        for k, idx in enumerate(front):
            cd_all[idx] = cd[k]
    i, j = rng.integers(0, N, size=2)
    if   rank[i] < rank[j]:    return population[i].copy()
    elif rank[j] < rank[i]:    return population[j].copy()
    elif cd_all[i] >= cd_all[j]: return population[i].copy()
    else:                        return population[j].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# GENETIC OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_population(pop_size, feature_cols):
    pop = np.zeros((pop_size, len(feature_cols)), dtype=int)
    for i, param in enumerate(feature_cols):
        pop[:, i] = rng.choice(PHYSICS_OPTIONS[param], size=pop_size)
    for k in range(pop_size):
        repair_sfclay(pop[k], feature_cols)
    return pop

def crossover(p1, p2):
    if rng.random() < CROSSOVER_RATE:
        mask = rng.random(len(p1)) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)
    return p1.copy(), p2.copy()

def mutate(individual, feature_cols):
    ind = individual.copy()
    for i, param in enumerate(feature_cols):
        if rng.random() < MUTATION_RATE:
            ind[i] = rng.choice(PHYSICS_OPTIONS[param])
    return repair_sfclay(ind, feature_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_nsga2(sur_rmse, fc_rmse, lk_rmse, sur_ets, fc_ets, ohe_index_ets, feature_cols):
    print(f"\n Running NSGA-II  (pop={POP_SIZE}, gen={N_GENERATIONS})")
    population = initialize_population(POP_SIZE, feature_cols)
    objectives = evaluate_population(population, sur_rmse, fc_rmse, lk_rmse,
                                                  sur_ets,  fc_ets,  ohe_index_ets)
    ref_rmse        = float(objectives[:, 0].max()) * 1.05
    ref_ets         = float(objectives[:, 1].max()) * 1.05
    reference_point = np.array([ref_rmse, ref_ets])
    print(f"  Reference point: RMSE={ref_rmse:.4f}, -ETS={ref_ets:.4f}")
    print(f"\n  {'Gen':>5} | {'Pareto':>7} | {'Unique':>7} | {'Min RMSE':>10} | {'Max ETS':>10} | {'Hypervolume':>14}")
    print(f"  {'-'*68}")
    history = []

    # ── Max attempts to generate a unique offspring ────────────────────────
    MAX_UNIQUE_TRIES = 5
    # ── Diversity restart threshold: if this fraction are duplicates → inject randoms
    DIVERSITY_THRESHOLD = 0.5

    for gen in range(N_GENERATIONS):
        fronts = fast_non_dominated_sort(objectives)

        # ── Count unique individuals in current population ─────────────────
        pop_keys   = [tuple(ind) for ind in population]
        n_unique   = len(set(pop_keys))
        dup_ratio  = 1.0 - n_unique / len(pop_keys)

        # ── Diversity restart: replace duplicate slots with random configs ──
        if dup_ratio > DIVERSITY_THRESHOLD:
            seen   = set()
            new_pop = []
            for ind in population:
                key = tuple(ind)
                if key not in seen:
                    seen.add(key)
                    new_pop.append(ind)
            # Fill remaining slots with fresh random individuals
            n_fill = POP_SIZE - len(new_pop)
            for _ in range(n_fill):
                rand_ind = np.array([rng.choice(PHYSICS_OPTIONS[p])
                                     for p in feature_cols])
                repair_sfclay(rand_ind, feature_cols)
                new_pop.append(rand_ind)
            population = np.array(new_pop)
            objectives = evaluate_population(population, sur_rmse, fc_rmse, lk_rmse,
                                             sur_ets, fc_ets, ohe_index_ets)
            fronts = fast_non_dominated_sort(objectives)

        # ── Generate offspring with duplicate prevention ───────────────────
        existing_keys = set(tuple(ind) for ind in population)
        offspring = []
        attempts  = 0

        while len(offspring) < POP_SIZE:
            p1 = binary_tournament(population, objectives, fronts)
            p2 = binary_tournament(population, objectives, fronts)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, feature_cols)
            c2 = mutate(c2, feature_cols)

            for child in [c1, c2]:
                key = tuple(child)
                if key not in existing_keys:
                    offspring.append(child)
                    existing_keys.add(key)
                else:
                    # Try extra mutation to escape the duplicate
                    attempts += 1
                    if attempts <= MAX_UNIQUE_TRIES * POP_SIZE:
                        for i, param in enumerate(feature_cols):
                            if rng.random() < 0.5:
                                child[i] = rng.choice(PHYSICS_OPTIONS[param])
                        repair_sfclay(child, feature_cols)
                        key2 = tuple(child)
                        if key2 not in existing_keys:
                            offspring.append(child)
                            existing_keys.add(key2)
                        else:
                            offspring.append(child)   # accept duplicate as last resort
                    else:
                        offspring.append(child)       # give up after too many tries

            if len(offspring) >= POP_SIZE:
                break

        offspring    = np.array(offspring[:POP_SIZE])
        off_obj      = evaluate_population(offspring, sur_rmse, fc_rmse, lk_rmse,
                                                       sur_ets,  fc_ets,  ohe_index_ets)
        combined_pop    = np.vstack([population, offspring])
        combined_obj    = np.vstack([objectives, off_obj])
        combined_fronts = fast_non_dominated_sort(combined_obj)
        population, objectives = nsga2_selection(combined_pop, combined_obj,
                                                 combined_fronts, POP_SIZE)
        pareto_front = fast_non_dominated_sort(objectives)[0]
        pareto_rmse  = objectives[pareto_front, 0]
        pareto_ets   = -objectives[pareto_front, 1]
        hv = compute_hypervolume_2d(objectives, pareto_front, reference_point)

        # Count unique in current population for monitoring
        n_unique_now = len(set(tuple(ind) for ind in population))

        history.append({
            "generation"  : gen,
            "pareto_size" : len(pareto_front),
            "n_unique"    : n_unique_now,
            "dup_ratio"   : round(1 - n_unique_now / POP_SIZE, 3),
            "min_rmse"    : float(pareto_rmse.min()),
            "max_ets"     : float(pareto_ets.max()),
            "mean_rmse"   : float(pareto_rmse.mean()),
            "mean_ets"    : float(pareto_ets.mean()),
            "hypervolume" : hv,
        })
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            print(f"  {gen:>5} | {len(pareto_front):>7} | "
                  f"{n_unique_now:>7} | "
                  f"{pareto_rmse.min():>10.6f} | {pareto_ets.max():>10.6f} | {hv:>14.6f}")

    return population, objectives, pd.DataFrame(history), reference_point

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED VERIFICATION SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_verification_candidates(pareto_df, feature_cols, n_verify=N_VERIFY):
    
    df = pareto_df.copy()

    # Step 1: create config key
    df["config_key"] = df.apply(
        lambda row: tuple(int(row[p]) for p in feature_cols), axis=1
    )

    # Step 2: deduplicate — keep first occurrence per unique config (RMSE order)
    df_unique = (
        df.sort_values("predicted_RMSE")
          .drop_duplicates(subset="config_key", keep="first")
          .reset_index(drop=True)
    )

    n_unique = len(df_unique)
    print(f"\n  Pareto front: {len(df)} solutions total, "
          f"{n_unique} physically unique configurations")

    if n_unique <= n_verify:
        selected_keys = set(df_unique["config_key"].tolist())
        print(f"  Using all {n_unique} unique configurations for verification")
    else:
        # Step 3: uniform spacing along deduplicated front
        indices       = np.linspace(0, n_unique - 1, n_verify, dtype=int)
        selected_keys = set(df_unique.iloc[indices]["config_key"].tolist())
        print(f"  Selected {n_verify} solutions by uniform spacing along "
              f"deduplicated Pareto front")

    # Step 4: mark selected in original dataframe (first occurrence per config)
    pareto_df["config_key"]               = df["config_key"]
    pareto_df["selected_for_verification"] = False
    seen = set()
    for idx, row in pareto_df.iterrows():
        key = row["config_key"]
        if key in selected_keys and key not in seen:
            pareto_df.loc[idx, "selected_for_verification"] = True
            seen.add(key)

    print(f"\n  Selected solutions for WRF re-verification "
          f"(uniform spacing, deduplicated front):")
    print(f"  {'#':>4} | {'RMSE':>10} | {'ETS':>10} | Physics Configuration")
    print(f"  {'-'*80}")
    selected_df = pareto_df[pareto_df["selected_for_verification"]].sort_values("predicted_RMSE")
    for k, (_, row) in enumerate(selected_df.iterrows()):
        config = "  ".join([f"{p.split('_')[0]}={int(row[p])}" for p in feature_cols])
        print(f"  {k+1:>4} | {row['predicted_RMSE']:>10.6f} | "
              f"{row['predicted_ETS']:>10.6f} | {config}")

    return pareto_df

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Surrogate Benchmark Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig3_surrogate_performance(cv_rmse_path, cv_ets_path):
    
    missing = [p for p in [cv_rmse_path, cv_ets_path] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"  ⚠ {p} not found — Fig3 skipped.")
        return

    df_rmse = pd.read_csv(cv_rmse_path)
    df_ets  = pd.read_csv(cv_ets_path)

    # Normalise column names (handle both index and explicit fold column)
    for df in [df_rmse, df_ets]:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "fold" not in df.columns:
            df.insert(0, "fold", range(1, len(df) + 1))

    def safe_col(df, *names):
        """Return first column name that exists in df, else None."""
        for n in names:
            if n in df.columns:
                return n
        return None

    rmse_col_rmse = safe_col(df_rmse, "test_rmse", "rmse")
    mae_col_rmse  = safe_col(df_rmse, "test_mae",  "mae")
    rmse_col_ets  = safe_col(df_ets,  "test_rmse", "rmse")
    mae_col_ets   = safe_col(df_ets,  "test_mae",  "mae")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH2, 5.5))
    fig.suptitle(
        "Figure 3 — Surrogate Model CV Performance (10-fold)\n"
        "S1: CatBoost + Target Encoding (RMSE)  |  S2: XGBoost + OHE (ETS)",
        fontsize=16, fontweight="bold", y=1.03
    )

    configs = [
        (axes[0], df_rmse, rmse_col_rmse, mae_col_rmse,
         "(a) S1 — CatBoost / Target Encoding (RMSE)",
         "Prediction error (mm)", BLUE, ORANGE),
        (axes[1], df_ets,  rmse_col_ets,  mae_col_ets,
         "(b) S2 — XGBoost / OHE Encoding (ETS)",
         "Prediction error (ETS units)", GREEN, PURPLE),
    ]

    for ax, df, rc, mc, title, ylabel, c1, c2 in configs:
        folds = df["fold"].values
        n     = len(folds)
        x     = np.arange(n)
        w     = 0.35

        rmse_vals = df[rc].values if rc else np.full(n, np.nan)
        mae_vals  = df[mc].values if mc else np.full(n, np.nan)

        b1 = ax.bar(x - w/2, rmse_vals, w, label="Test RMSE",
                    color=c1, alpha=0.80, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + w/2, mae_vals,  w, label="Test MAE",
                    color=c2, alpha=0.80, edgecolor="white", linewidth=0.5)

        # Mean ± std reference lines
        if not np.all(np.isnan(rmse_vals)):
            m, s = np.nanmean(rmse_vals), np.nanstd(rmse_vals)
            ax.axhline(m, color=c1, linewidth=1.8, linestyle="--", alpha=0.9)
            ax.fill_between([-0.5, n - 0.5], m - s, m + s,
                            color=c1, alpha=0.10)
            ax.text(n - 0.5, m, f" μ={m:.4f}±{s:.4f}",
                    va="center", ha="left", fontsize=12, color=c1)

        if not np.all(np.isnan(mae_vals)):
            m, s = np.nanmean(mae_vals), np.nanstd(mae_vals)
            ax.axhline(m, color=c2, linewidth=1.8, linestyle=":",  alpha=0.9)
            ax.text(n - 0.5, m, f" μ={m:.4f}±{s:.4f}",
                    va="center", ha="left", fontsize=12, color=c2)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {int(f)}" for f in folds], fontsize=13,
                           rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.legend(fontsize=13, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.6, n - 0.4 + 1.5)   # room for annotations

        # Annotate target range info
        rng_txt = None
        if "target_range" in df.columns:
            lo, hi = df["target_range"].iloc[0], df["target_range"].iloc[-1]
            span   = abs(hi - lo)
            rng_txt = f"Target range: [{lo:.3f}, {hi:.3f}]"
            if not np.all(np.isnan(rmse_vals)):
                rel = np.nanmean(rmse_vals) / span * 100 if span > 0 else 0
                rng_txt += f"\nRelative RMSE: {rel:.1f}%"
        if rng_txt:
            ax.text(0.02, 0.97, rng_txt, transform=ax.transAxes,
                    fontsize=12, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="grey", alpha=0.8))

    plt.tight_layout()
    savefig(fig, "Fig3_surrogate_cv_performance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Encoding Comparison Boxplot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4_cv_distribution(cv_rmse_path, cv_ets_path):
    
    missing = [p for p in [cv_rmse_path, cv_ets_path] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"  ⚠ {p} not found — Fig4 skipped.")
        return

    df_rmse = pd.read_csv(cv_rmse_path)
    df_ets  = pd.read_csv(cv_ets_path)

    for df in [df_rmse, df_ets]:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def safe_col(df, *names):
        for n in names:
            if n in df.columns: return n
        return None

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH2, 5.5))
    fig.suptitle(
        "Figure 4 — Per-Fold CV Metric Distribution\n"
        "S1: CatBoost + Target Encoding (RMSE)  |  S2: XGBoost + OHE (ETS)",
        fontsize=16, fontweight="bold", y=1.03
    )

    configs = [
        (axes[0], df_rmse, "mm",        "(a) S1 — CatBoost / Target Encoding (RMSE)",  BLUE, ORANGE),
        (axes[1], df_ets,  "ETS units", "(b) S2 — XGBoost / OHE Encoding (ETS)",    GREEN, PURPLE),
    ]

    for ax, df, unit, title, c1, c2 in configs:
        rc = safe_col(df, "test_rmse", "rmse")
        mc = safe_col(df, "test_mae",  "mae")

        rmse_vals = df[rc].dropna().values if rc else np.array([])
        mae_vals  = df[mc].dropna().values if mc else np.array([])

        data   = [v for v in [rmse_vals, mae_vals] if len(v) > 0]
        labels = []
        colors = []
        if len(rmse_vals): labels.append("Test RMSE"); colors.append(c1)
        if len(mae_vals):  labels.append("Test MAE");  colors.append(c2)

        if not data:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="red")
            ax.set_title(title, fontsize=15, fontweight="bold")
            continue

        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        # Overlay individual fold points with jitter
        for j, (vals, color) in enumerate(zip(data, colors)):
            jitter = np.random.default_rng(j + 10).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.ones(len(vals)) * (j + 1) + jitter, vals,
                       color=color, alpha=0.85, s=55, zorder=4,
                       edgecolors="white", linewidths=0.6)

        # Annotate mean ± std above each box
        for j, (vals, color) in enumerate(zip(data, colors)):
            m, s = np.mean(vals), np.std(vals)
            ax.text(j + 1, np.max(vals) + 0.002,
                    f"μ={m:.4f}\n±{s:.4f}",
                    ha="center", va="bottom", fontsize=12,
                    color=color, fontweight="bold")

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylabel(f"Prediction error ({unit})", fontsize=16)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        n_folds = len(rmse_vals) if len(rmse_vals) else len(mae_vals)
        ax.text(0.97, 0.03, f"n = {n_folds} folds",
                transform=ax.transAxes, fontsize=12,
                ha="right", va="bottom", color="grey")

    plt.tight_layout()
    savefig(fig, "Fig4_surrogate_cv_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — NSGA-II Convergence
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig5_convergence(history):
   
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5.5))

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.16))

    l1, = ax.plot(history["generation"],  history["min_rmse"],
                  color=BLUE,   linewidth=3.0, label="Min RMSE (mm)")
    l2, = ax2.plot(history["generation"], history["max_ets"],
                   color=ORANGE, linewidth=3.0, linestyle="--", label="Max ETS")
    l3, = ax3.plot(history["generation"], history["hypervolume"],
                   color=PURPLE, linewidth=3.0, linestyle=":",  label="Hypervolume")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Min RMSE (mm)",  color=BLUE)
    ax2.set_ylabel("Max ETS",        color=ORANGE)
    ax3.set_ylabel("Hypervolume",    color=PURPLE)
    ax.tick_params(axis="y",  labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax3.tick_params(axis="y", labelcolor=PURPLE)
    ax.legend(handles=[l1, l2, l3], loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, "Fig5_nsga2_convergence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Pareto Front Size
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig6_pareto_size(history):
    """
    Fig6: Pareto front size (number of non-dominated solutions) per generation.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5))
    ax.fill_between(history["generation"], history["pareto_size"],
                    alpha=0.25, color=GREEN)
    ax.plot(history["generation"], history["pareto_size"],
            color=GREEN, linewidth=3.0)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Non-dominated solutions")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, "Fig6_pareto_front_size.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Pareto Front Scatter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig7_pareto_front(pareto_df):
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 6.5))

    srt = pareto_df.sort_values("predicted_RMSE")
    ax.plot(srt["predicted_RMSE"], srt["predicted_ETS"],
            color=BLUE, linewidth=2.2, alpha=0.55, zorder=2, label="Pareto front")
    ax.scatter(pareto_df["predicted_RMSE"], pareto_df["predicted_ETS"],
               color=BLUE, s=150, zorder=4, edgecolors="black", linewidths=0.9,
               label=f"Pareto solutions (n={len(pareto_df)})")

    if "selected_for_verification" in pareto_df.columns:
        vdf = pareto_df[pareto_df["selected_for_verification"]]
        ax.scatter(vdf["predicted_RMSE"], vdf["predicted_ETS"], s=300,
                   facecolors="none", edgecolors=GOLD, linewidths=3.0, zorder=5,
                   label=f"WRF verification (n={len(vdf)})")

    best_r = pareto_df["predicted_RMSE"].idxmin()
    best_e = pareto_df["predicted_ETS"].idxmax()
    ax.scatter(pareto_df.loc[best_r, "predicted_RMSE"],
               pareto_df.loc[best_r, "predicted_ETS"],
               marker="*", s=520, color=RED, edgecolors="black", linewidths=0.8,
               zorder=6, label="Best RMSE")
    ax.scatter(pareto_df.loc[best_e, "predicted_RMSE"],
               pareto_df.loc[best_e, "predicted_ETS"],
               marker="*", s=520, color=GREEN, edgecolors="black", linewidths=0.8,
               zorder=6, label="Best ETS")

    ax.set_xlabel("Surrogate-predicted RMSE (mm)  \u2193 lower is better")
    ax.set_ylabel("Surrogate-predicted ETS  \u2191 higher is better")
    ax.legend(loc="lower left", handletextpad=0.6)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, "Fig7_pareto_front_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Physics Scheme Frequency Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig8_scheme_frequency(pareto_df, feature_cols):
    
    all_schemes = sorted(set(sid for opts in PHYSICS_OPTIONS.values() for sid in opts))
    freq_matrix = np.zeros((len(feature_cols), len(all_schemes)))
    for i, param in enumerate(feature_cols):
        for j, sid in enumerate(all_schemes):
            freq_matrix[i, j] = (pareto_df[param] == sid).sum()

    row_sums              = freq_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    freq_norm             = freq_matrix / row_sums

    fig, ax = plt.subplots(figsize=(FIG_WIDTH2, 5.5))

    im = ax.imshow(freq_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_schemes)))
    ax.set_xticklabels(all_schemes, fontsize=12, rotation=90)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in feature_cols], fontsize=15)
    ax.set_xlabel("WRF scheme ID")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Relative frequency on Pareto front", fontsize=15)
    cb.ax.tick_params(labelsize=13)

    for i in range(len(feature_cols)):
        for j in range(len(all_schemes)):
            count = int(freq_matrix[i, j])
            if count > 0:
                color = "white" if freq_norm[i, j] > 0.6 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    plt.tight_layout()
    savefig(fig, "Fig8_scheme_frequency_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — REMOVED
# The surrogate "uncertainty" figure was removed: per-tree std is invalid for the
# CatBoost/XGBoost (boosting) surrogates and evaluated to all zeros. This keeps the
# code consistent with the manuscript, where the uncertainty analysis was dropped.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Final Population vs Pareto Front
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig10_population(final_obj):
    
    pareto_mask = np.zeros(len(final_obj), dtype=bool)
    pareto_idxs = fast_non_dominated_sort(final_obj)[0]
    pareto_mask[pareto_idxs] = True

    all_rmse = final_obj[:, 0]
    all_ets  = -final_obj[:, 1]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 6.5))

    ax.scatter(all_rmse[~pareto_mask], all_ets[~pareto_mask],
               color=GREY, s=35, alpha=0.4, label="Dominated solutions", zorder=2)
    ax.scatter(all_rmse[pareto_mask],  all_ets[pareto_mask],
               color=RED,  s=160, alpha=0.95, edgecolors="k", linewidths=0.9, zorder=4,
               label=f"Pareto front (n={pareto_mask.sum()})")

    ax.set_xlabel("Surrogate-predicted RMSE (mm)  \u2193")
    ax.set_ylabel("Surrogate-predicted ETS  \u2191")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(fig, "Fig10_population_vs_pareto.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig11_shap(sur_rmse, sur_ets, fc_rmse, fc_ets,
                    lk_rmse, ohe_index_ets, pareto_df, feature_cols):
    
    if not SHAP_AVAILABLE:
        print("  \u26a0 SHAP not available — Fig11 skipped.")
        return

    # ── S1: target-encoded RMSE surrogate (7 columns = 7 categories) ──────────
    print("  Computing SHAP for S1 (RMSE surrogate) ...")
    enc_rmse = np.array([
        encode_individual(
            np.array([int(pareto_df.iloc[i][p]) for p in fc_rmse]),
            fc_rmse, lk_rmse)
        for i in range(len(pareto_df))
    ])
    shap_rmse_cols = np.abs(shap.TreeExplainer(sur_rmse).shap_values(enc_rmse)).mean(axis=0)
    imp_rmse = {PARAM_LABELS.get(p, p): shap_rmse_cols[i] for i, p in enumerate(fc_rmse)}

    # ── S2: OHE-encoded ETS surrogate (many columns → aggregate per category) ──
    print("  Computing SHAP for S2 (ETS surrogate) ...")
    enc_ets = np.array([
        encode_individual_ohe(
            np.array([int(pareto_df.iloc[i][p]) for p in PARAM_NAMES]),
            fc_ets, ohe_index_ets)
        for i in range(len(pareto_df))
    ])
    shap_ets_cols = np.abs(shap.TreeExplainer(sur_ets).shap_values(enc_ets)).mean(axis=0)
    imp_ets = {PARAM_LABELS.get(p, p): 0.0 for p in PARAM_NAMES}
    for (param, scheme), col in ohe_index_ets.items():
        if col < len(shap_ets_cols):
            imp_ets[PARAM_LABELS.get(param, param)] += shap_ets_cols[col]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH2, 6))

    for ax, imp, title, color in zip(
        axes,
        [imp_rmse, imp_ets],
        ["(a) S1 — CatBoost + Target Encoding (RMSE)",
         "(b) S2 — XGBoost + OHE Encoding (ETS)"],
        [BLUE, ORANGE],
    ):
        items  = sorted(imp.items(), key=lambda kv: kv[1])
        labels = [k for k, _ in items]
        vals   = [v for _, v in items]
        ax.barh(range(len(labels)), vals, color=color, edgecolor="k", linewidth=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        for j, v in enumerate(vals):
            ax.text(v, j, f" {v:.3f}", va="center", fontsize=11)

    plt.tight_layout()
    savefig(fig, "Fig11_shap_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("NSGA-II  |  Multi-Objective WRF Physics Optimisation")
    print(f"  Obj 1 : Minimise RMSE  ({RMSE_SURROGATE_PATH})")
    print(f"  Obj 2 : Maximise ETS   ({ETS_SURROGATE_PATH})")
    print(f"  Space : 15\u00D712\u00D713\u00D76\u00D77\u00D75 = 491,400 configurations "
          f"(sf_sfclay tied to bl_pbl: 2/3/4/10 \u2192 same ID, else 1)")
    print(f"  Figs  : {FIGURES_DIR}/")
    print("=" * 65)

    # ── 1. Generate paper figures from benchmark CSVs (Figs 3, 4) ─────────────
    print("\n Generating surrogate benchmark figures (Figs 3 & 4) ...")
    plot_fig3_surrogate_performance(CV_METRICS_RMSE_PATH, CV_METRICS_ETS_PATH)
    plot_fig4_cv_distribution(CV_METRICS_RMSE_PATH, CV_METRICS_ETS_PATH)

    # ── 2. Load surrogates and lookup tables ───────────────────────────────────
    print("\n Loading surrogates ...")
    sur_rmse, fc_rmse = load_surrogate(RMSE_SURROGATE_PATH, "RMSE-CatBoost")
    sur_ets,  fc_ets  = load_surrogate(ETS_SURROGATE_PATH,  "ETS-XGBoost")
    print("\n Loading lookup tables ...")
    # S1 (CatBoost): load target encoding lookup table
    lk_rmse = load_lookup_table(RMSE_LOOKUP_PATH, fc_rmse, "RMSE")
    # S2 (XGBoost): build OHE column index from feature_cols in the surrogate payload
    ohe_index_ets = build_ohe_column_index(fc_ets)
    print(f"   [ETS-XGBoost] OHE index built: {len(ohe_index_ets)} (param, scheme) entries")
    feature_cols = fc_rmse

    # ── 3. Run NSGA-II ─────────────────────────────────────────────────────────
    final_pop, final_obj, history, reference_point = run_nsga2(
        sur_rmse, fc_rmse, lk_rmse,
        sur_ets,  fc_ets,  ohe_index_ets,
        feature_cols
    )

    # ── 4. Extract Pareto front ────────────────────────────────────────────────
    pareto_idx = fast_non_dominated_sort(final_obj)[0]
    pareto_pop = final_pop[pareto_idx]
    pareto_obj = final_obj[pareto_idx]
    sort_order = np.argsort(pareto_obj[:, 0])
    pareto_pop = pareto_pop[sort_order]
    pareto_obj = pareto_obj[sort_order]

    # Print Pareto front summary
    print(f"\n{'='*65}")
    print(f"PARETO FRONT  ({len(pareto_pop)} solutions):")
    print(f"  {'#':>4} | {'RMSE':>10} | {'ETS':>10} | Physics Configuration")
    print(f"  {'-'*65}")
    for k, (ind, obj) in enumerate(zip(pareto_pop, pareto_obj)):
        config = "  ".join([f"{p.split('_')[0]}={int(v)}"
                            for p, v in zip(feature_cols, ind)])
        print(f"  {k+1:>4} | {obj[0]:>10.6f} | {-obj[1]:>10.6f} | {config}")

    # ── 5. Build Pareto DataFrame ────────────────────────────────────────────────────────────────────────
    rows = []
    for ind, obj in zip(pareto_pop, pareto_obj):
        row      = {param: int(sid) for param, sid in zip(feature_cols, ind)}
        enc_rmse = encode_individual(ind, fc_rmse, lk_rmse)
        enc_ets  = encode_individual_ohe(ind, fc_ets, ohe_index_ets)
        for param, ev in zip(fc_rmse, enc_rmse):
            row[f"{param}_enc_rmse"] = round(ev, 6)
        for param, ev in zip(fc_ets,  enc_ets):
            row[f"{param}_enc_ets"]  = round(ev, 6)
        row["predicted_RMSE"]   = round(float(obj[0]),  6)
        row["predicted_ETS"]    = round(float(-obj[1]), 6)
        rows.append(row)

    pareto_df = pd.DataFrame(rows)

    # ── 6. FIXED verification selection ───────────────────────────────────────
    pareto_df = select_verification_candidates(pareto_df, feature_cols, N_VERIFY)

    # ── 7. Save outputs ────────────────────────────────────────────────────────
    pareto_df.to_csv(PARETO_OUTPUT, index=False)
    history.to_csv(HISTORY_OUTPUT, index=False)
    print(f"\n Pareto front    -> {PARETO_OUTPUT}  ({len(pareto_df)} solutions)")
    print(f" NSGA-II history -> {HISTORY_OUTPUT}")

    # ── 8. Hypervolume summary ─────────────────────────────────────────────────
    final_hv = history["hypervolume"].iloc[-1]
    best_hv  = history["hypervolume"].max()
    best_gen = history["hypervolume"].idxmax()
    print(f"\n Hypervolume summary:")
    print(f"  Best observed    : {best_hv:.6f}  (generation {best_gen})")
    print(f"  Final generation : {final_hv:.6f}")
    print(f"  Reference point  : RMSE={reference_point[0]:.4f}, "
          f"-ETS={reference_point[1]:.4f}")

    # ── 9. Generate all paper figures ─────────────────────────────────────────
    print(f"\n Generating paper figures -> {FIGURES_DIR}/")
    plot_fig5_convergence(history)
    plot_fig6_pareto_size(history)
    plot_fig7_pareto_front(pareto_df)
    plot_fig8_scheme_frequency(pareto_df, feature_cols)
    plot_fig10_population(final_obj)
    plot_fig11_shap(sur_rmse, sur_ets, fc_rmse, fc_ets,
                    lk_rmse, ohe_index_ets, pareto_df, feature_cols)

    print(f"\n{'='*65}")
    print(f"All paper figures saved to: {FIGURES_DIR}/")
    print(f"  Fig3  — Surrogate CV performance (CatBoost S1 vs XGBoost S2)")
    print(f"  Fig4  — Per-fold CV metric distributions")
    print(f"  Fig5  — NSGA-II convergence (RMSE + ETS + Hypervolume)")
    print(f"  Fig6  — Pareto front size over generations")
    print(f"  Fig7  — Pareto front scatter (RMSE vs ETS)")
    print(f"  Fig8  — Physics scheme frequency heatmap")
    print(f"  Fig10 — Final population vs Pareto front")
    print(f"  Fig11 — SHAP feature importance (S1 and S2)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
