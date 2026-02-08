import numpy as np
import json
import itertools
import lightgbm as lgb
import xgboost as xgb  # Für optional XGBoost
import xgboost.callback  # Für EarlyStopping in XGB
from sklearn.ensemble import RandomForestRegressor  # Für optional Random Forest
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import warnings
import os
import argparse
from imblearn.over_sampling import SMOTE  # Für SMOTE-Variante (installiere via pip install imbalanced-learn)
from sklearn.metrics import r2_score, mean_absolute_error
import psutil
import subprocess
import random
import math
import time  

# Suppress sklearn UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

sys.path.append("../")
from utils.features import *

def compute_features(structure: str) -> np.ndarray:
    """Berechnet alle 36 Features plus 9 Bigram-Features."""
    length = len(structure)
    # (hier alle Feature-Berechnungen wie gehabt) …
    bulges = count_bulges(structure)
    internal_loops = count_internal_loops(structure)
    max_loop = max_loop_size(structure)
    multiloops = count_multiloops(structure)
    centrality = graph_centrality(structure)
    depth = tree_depth(structure)
    mean_depth, var_depth, peaks, mean_depth_paired, var_depth_paired, mean_depth_unpaired, var_depth_unpaired = get_depth_features(structure)
    num_stems, max_stem_length, avg_stem_length, var_stem_length = get_stem_features(structure)
    mean_loop, var_loop = get_loop_features(structure)
    bigram_features = get_ngram_features(structure, n=2)
    hairpin_loops = count_hairpin_loops(structure)
    stacked_pairs = count_stacked_pairs(structure)
    avg_bp_dist, _ = get_base_pair_distances(structure)
    paired_bases = num_paired_bases(structure)
    unpaired_bases = num_unpaired_bases(structure)
    hairpin_sizes = get_hairpin_loop_sizes(structure)
    mean_hairpin_size = np.mean(hairpin_sizes) if hairpin_sizes else 0
    max_hairpin_size = max(hairpin_sizes) if hairpin_sizes else 0
    internal_loop_sizes = get_internal_loop_sizes(structure)
    mean_internal_loop_size = np.mean(internal_loop_sizes) if internal_loop_sizes else 0
    max_internal_loop_size = max(internal_loop_sizes) if internal_loop_sizes else 0
    bulge_sizes = get_bulge_sizes(structure)
    mean_bulge_size = np.mean(bulge_sizes) if bulge_sizes else 0
    max_bulge_size = max(bulge_sizes) if bulge_sizes else 0

    features = [
        internal_loops,
        var_depth_paired,
        multiloops,
        max_loop,
        length,
        mean_loop,
        depth,
        mean_depth_unpaired,
        bulges,
        var_loop,
        centrality,
        var_stem_length,
        max_stem_length,
        avg_stem_length,
        mean_depth_paired,
        peaks,
        num_stems,
        var_depth_unpaired,
        var_depth,
        mean_depth,
        hairpin_loops,
        stacked_pairs,
        avg_bp_dist,
        paired_bases,
        unpaired_bases,
        mean_hairpin_size,
        max_hairpin_size,
        mean_internal_loop_size,
        max_internal_loop_size,
        mean_bulge_size,
        max_bulge_size
    ] + bigram_features

    return np.array(features, dtype=np.float32)

def msle_objective(y_true, y_pred):
    """Custom MSLE objective for relative errors: Gradient und Hessian."""
    epsilon = 1e-6
    y_pred = np.maximum(y_pred, epsilon)  # Vermeide log(0) oder negativ
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = 1 / ((y_pred + 1) ** 2)
    return grad, hess

def _inverse_log_transform(y, use_log: bool):
    return np.expm1(y) if use_log else y

def _pearson_r_ci(r, n, alpha=0.05):
    # Fisher-Z Konfidenzintervall
    if n <= 3 or np.isnan(r):
        return [float('nan'), float('nan')]
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    zcrit = 1.959963984540054  # ~95%
    lo = z - zcrit * se
    hi = z + zcrit * se
    rlo = (np.exp(2 * lo) - 1) / (np.exp(2 * lo) + 1)
    rhi = (np.exp(2 * hi) - 1) / (np.exp(2 * hi) + 1)
    return [rlo, rhi]

def build_pred_matrix_from_pairs(pred_pairs, pair_idx, N):
    """
    pred_pairs: 1D array der Vorhersagen für 'pairs'
    pair_idx: Liste von (i,j) mit i<j in gleicher Reihenfolge wie X/pairs
    N: Anzahl Strukturen
    Rückgabe: NxN symmetrische Matrix (diagonale 0)
    """
    M = np.zeros((N, N), dtype=np.float32)
    for val, (i, j) in zip(pred_pairs, pair_idx):
        M[i, j] = val
        M[j, i] = val
    return M

def topk_indices_from_distance_row(row, k, self_index=None):
    # Maskiere Selbsttreffer
    if self_index is not None and 0 <= self_index < len(row):
        row = row.copy()
        row[self_index] = np.inf
    k = min(int(k), len(row))
    idx = np.argpartition(row, kth=min(k-1, len(row)-1))[:k]
    return idx


def recall_at_k(true_mat, pred_mat, ks=(10, 20, 50)):
    """
    true_mat, pred_mat: NxN Distanzen (0 auf Diagonale)
    Gibt mean Recall@K über K in ks (und pro-K Vektoren) zurück
    """
    N = true_mat.shape[0]
    results = {}
    for K in ks:
        per_node = []
        for i in range(N):
            # kleinste K im Ground Truth / in der Vorhersage
            true_nn = topk_indices_from_distance_row(true_mat[i], K)
            pred_nn = topk_indices_from_distance_row(pred_mat[i], K)
            inter = len(set(true_nn.tolist()) & set(pred_nn.tolist()))
            per_node.append(inter / K)
        results[K] = {
            "mean": float(np.mean(per_node)),
            "median": float(np.median(per_node)),
            "p25": float(np.percentile(per_node, 25)),
            "p75": float(np.percentile(per_node, 75)),
            "per_node": per_node  # bei Bedarf groß; sonst weglassen
        }
    return results

def pruning_from_topk(pred_mat, K):
    """
    Erzeuge die Menge der (i,j)-Paare, die durch Top-K je Knoten weitergereicht würden.
    """
    N = pred_mat.shape[0]
    forwarded = set()
    for i in range(N):
        nn = topk_indices_from_distance_row(pred_mat[i], K)
        for j in nn:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            forwarded.add((a, b))
    total_pairs = N * (N - 1) // 2
    forwarded_pairs = len(forwarded)
    pruned_percent = 100.0 * (1 - forwarded_pairs / total_pairs)
    return {
        "total_pairs": total_pairs,
        "forwarded_pairs": forwarded_pairs,
        "pruned_percent": pruned_percent,
        "candidates_per_query_median": K  # nach Definition Top-K
    }

def time_pred_inference(reg, X_all):
    t0 = time.perf_counter()
    _ = reg.predict(X_all)
    dt = time.perf_counter() - t0
    return dt

def sample_exact_runtime_with_rnadistance(structures, sample_pairs=2000, metric='T'):
    """
    Optional: misst RNAdistance-Laufzeit auf Stichprobe.
    Erfordert, dass 'RNAdistance' im PATH liegt.
    metric: 'T' = tree edit distance; passe nach deinem Setup an.
    """
    try:
        random.seed(42)
        m = len(structures)
        idx = random.sample(list(itertools.combinations(range(m), 2)), k=min(sample_pairs, m*(m-1)//2))
        t0 = time.perf_counter()
        for i, j in idx:
            s1 = structures[i]
            s2 = structures[j]
            # RNAdistance erwartet Zeilen mit Strukturen, Option -D für Distanz nur
            # Beispiel-Aufruf: echo -e "(((...)))\n((..))" | RNAdistance -D -T
            p = subprocess.run(
                ["RNAdistance", "-D", f"-{metric}"],
                input=(s1 + "\n" + s2 + "\n").encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            # Ausgabe wird nicht weiter verwendet
        dt = time.perf_counter() - t0
        per_pair = dt / len(idx)
        return {"ok": True, "sample_seconds": dt, "pairs": len(idx), "seconds_per_pair": per_pair}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main(
    structures_file: str = '../data/structures.txt',
    ted_matrix_file: str = '../data/ted_matrix.txt',
    corr_threshold: float = 0
):
    # 0) Parsers
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--use_log', action='store_true', help='Use log transformation on target (TED)')
    parser.add_argument('--use_custom_loss', action='store_true', help='Use custom MSLE loss function for relative errors')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE oversampling for imbalanced regression (focus on small TED)')
    parser.add_argument('--use_xgboost', action='store_true', help='Use XGBoost instead of LightGBM')
    parser.add_argument('--use_random_forest', action='store_true', help='Use Random Forest in ensemble with main model')
    parser.add_argument('--reporting', action='store_true', help="Generate Reporting metrics")
    parser.add_argument('--rich_features', action='store_true',
                        help='Use rich pairwise features: diff + sum + min + max (144 features instead of 36)')
    parser.add_argument('--use_lambdarank', action='store_true',
                        help='Use LambdaRank objective (optimises NDCG directly)')
    parser.add_argument('--float_output', action='store_true',
                        help='Skip integer rounding in evaluation (float predictions)')
    args = parser.parse_args()

    # 1) Daten einlesen
    with open(structures_file, 'r') as f:
        structures = [line.strip() for line in f if line.strip()]
    ted_matrix = np.loadtxt(ted_matrix_file, delimiter=' ')
    if len(structures) != ted_matrix.shape[0]:
        print("Error: Number of structures does not match TED matrix dimensions.")
        return

    # 2) Feature-Matrix berechnen
    print("Computing feature matrix…")
    feature_matrix = np.array([compute_features(s) for s in tqdm(structures)])

    # 3) Paare & Differenzen
    m = len(structures)
    pairs = list(itertools.combinations(range(m), 2))
    y = np.array([ted_matrix[i, j] for i, j in pairs])

    if args.rich_features:
        print("Using rich pairwise features: diff + sum + min + max (144 features)")
        X_list = []
        for i, j in tqdm(pairs, desc="Building rich features"):
            fi, fj = feature_matrix[i], feature_matrix[j]
            X_list.append(np.concatenate([
                np.abs(fi - fj),          # diff (36)
                fi + fj,                  # sum  (36)
                np.minimum(fi, fj),       # min  (36)
                np.maximum(fi, fj),       # max  (36)
            ]))
        X = np.array(X_list, dtype=np.float32)
        del X_list
        print(f"  Feature matrix shape: {X.shape}")
    else:
        X = np.array([np.abs(feature_matrix[i] - feature_matrix[j]) for i, j in pairs])

    # Optional: Log-Transformation des Targets
    if args.use_log:
        print("Using log transformation")
        y_transformed = np.log1p(y)  # log(y + 1) für TED >= 0
    else:
        y_transformed = y

    # 4) Konstanten entfernen
    variances = np.var(X, axis=0)
    non_const = np.where(variances > 0)[0]
    X = X[:, non_const]

    # 5) (Optional) Feature-Selektion per Korrelation
    feature_names = []  # hier könntest du deine ursprünglichen Namen einfügen
    corrs = [pearsonr(X[:,i], y)[0] for i in range(X.shape[1])]  # Korrelation auf original y, für Konsistenz
    selected = [i for i,c in enumerate(corrs) if abs(c) >= corr_threshold]
    X_sel = X[:, selected]

    # 6) Split in Training / Validation / Test (auf transformiertem y)
    # Erst: Train+Valid vs. Test
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_sel, y_transformed, test_size=0.20, random_state=42
    )
    # Dann: Train vs. Valid aus Train+Valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.20, random_state=42
    )


    # Optional: SMOTE Oversampling für kleine TED-Werte (Regression-Approximation)
    if args.use_smote:
        print("Using SMOTE oversampling for imbalanced regression")
        # Dynamische Threshold für Klassen (Median, um immer zwei Klassen zu haben)
        median_y = np.median(y_train)
        y_train_classes = np.where(y_train < median_y, 0, 1)  # Klasse 0: kleine TED (underrepresented)
        unique_classes = np.unique(y_train_classes)
        if len(unique_classes) < 2:
            print("Only one class detected (all y same side of median), skipping SMOTE")
        else:
            # Bestimme minority class
            class_counts = np.bincount(y_train_classes)
            minority_class = np.argmin(class_counts)
            minority_y = y_train[y_train_classes == minority_class]

            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_resampled, y_train_classes_resampled = smote.fit_resample(X_train, y_train_classes)

            # Anzahl synthetischer Samples
            num_synthetic = len(y_train_classes_resampled) - len(y_train_classes)

            # Approximiere y für synthetische Samples (random choice aus minority y)
            synthetic_y = np.random.choice(minority_y, num_synthetic, replace=True)

            # Kombiniere original y mit synthetic y
            y_train_resampled = np.concatenate((y_train, synthetic_y))
            X_train = X_train_resampled
            y_train = y_train_resampled  # Fix: Aktualisiere y_train auf resampled Version
            print(f"After SMOTE: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")  # Debug-Log
    else:
        y_train_resampled = y_train  # Für Konsistenz
        y_train = y_train_resampled  # Sicherstellen, dass y_train aktualisiert wird

    # 7) Hyperparameter-Tuning (falls gewünscht)
    # Modell-spezifische Params-Datei
    if args.use_xgboost:
        params_file = 'best_params_xgb.json'
    else:
        params_file = 'best_params.json'

    if args.tuning or not os.path.exists(params_file):
        if args.use_xgboost:
            param_grid = {
                'max_depth': [5, 7, 9],  # Für XGB
                'n_estimators': [300, 400, 500, 600],
                'learning_rate': [0.1]
            }
            grid_model = xgb.XGBRegressor()
        else:
            param_grid = {
                'num_leaves': [31, 100, 250],
                'max_depth': [-1, 5, 10],
                'n_estimators': [200]
            }
            grid_model = lgb.LGBMRegressor()
        grid = GridSearchCV(
            grid_model, param_grid, cv=3, scoring='r2', verbose=1
        )
        grid.fit(X_train, y_train)
        best = grid.best_params_
        with open(params_file, 'w') as f:
            json.dump(best, f, indent=4)
        print(f"Best params saved to {params_file}:", best)
    else:
        with open(params_file, 'r') as f:
            best = json.load(f)
        print(f"Loaded best params from {params_file}:", best)

    # 8) Finales Modell trainieren (alle Samples gleich gewichtet)

    # ---- LambdaRank path ----
    if args.use_lambdarank:
        print("Using LambdaRank objective (optimises NDCG directly)")
        # LambdaRank requires query groups: each query = one structure, its pairs = candidates
        # Build group arrays from pair indices
        from collections import Counter

        # Map pairs back to query groups (structure i → all pairs containing i)
        # We use the first element of each pair as the query
        pair_indices_train = []
        pair_indices_valid = []

        # We need to reconstruct which pairs went into train/valid/test
        # Since train_test_split uses indices, we track them
        all_indices = np.arange(len(y_transformed))
        idx_train_valid, idx_test = train_test_split(
            all_indices, test_size=0.20, random_state=42
        )
        idx_train, idx_valid = train_test_split(
            idx_train_valid, test_size=0.20, random_state=42
        )

        # Build query groups: group by first structure in pair
        def build_query_groups(pair_subset_indices):
            """Return (sorted_X, sorted_y, group_sizes) sorted by query structure."""
            query_ids = np.array([pairs[pi][0] for pi in pair_subset_indices])
            sort_order = np.argsort(query_ids)
            sorted_indices = pair_subset_indices[sort_order]
            sorted_queries = query_ids[sort_order]
            # Count consecutive groups
            groups = []
            current_q = sorted_queries[0]
            count = 1
            for qi in sorted_queries[1:]:
                if qi == current_q:
                    count += 1
                else:
                    groups.append(count)
                    current_q = qi
                    count = 1
            groups.append(count)
            return sorted_indices, np.array(groups)

        train_sorted_idx, train_groups = build_query_groups(idx_train)
        valid_sorted_idx, valid_groups = build_query_groups(idx_valid)

        X_train_rank = X_sel[train_sorted_idx]
        y_train_rank = y_transformed[train_sorted_idx]
        X_valid_rank = X_sel[valid_sorted_idx]
        y_valid_rank = y_transformed[valid_sorted_idx]

        # For ranking, higher relevance = smaller TED → invert: relevance = max_ted - ted
        max_ted = max(np.max(y_train_rank), np.max(y_valid_rank))
        y_train_rel = max_ted - y_train_rank
        y_valid_rel = max_ted - y_valid_rank

        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=best.get('n_estimators', 200),
            num_leaves=best.get('num_leaves', 250),
            learning_rate=0.05,
            min_child_samples=20,
            lambdarank_truncation_level=20,
        )
        ranker.fit(
            X_train_rank, y_train_rel,
            group=train_groups,
            eval_set=[(X_valid_rank, y_valid_rel)],
            eval_group=[valid_groups],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=10)
            ]
        )
        # LambdaRank scores are not TED values — we need a regression head on top
        # Strategy: use ranker scores as additional feature, retrain a small regressor
        print("Training regression head on ranker scores...")
        rank_scores_train = ranker.predict(X_train)
        rank_scores_valid = ranker.predict(X_valid)
        X_train_aug = np.column_stack([X_train, rank_scores_train])
        X_valid_aug = np.column_stack([X_valid, rank_scores_valid])

        reg = lgb.LGBMRegressor(n_estimators=100, num_leaves=31)
        reg.fit(
            X_train_aug, y_train,
            eval_set=[(X_valid_aug, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
        )
        # Store ranker for later use
        reg._lambdarank_ranker = ranker
        reg._lambdarank_mode = True
        print("LambdaRank + regression head trained.")

    # ---- Standard regression path ----
    elif args.use_xgboost:
        print("Using XGBoost instead of LightGBM")

        reg = xgb.XGBRegressor(
            **best,
            objective=msle_objective if args.use_custom_loss else None,
            eval_metric="rmse",      
            early_stopping_rounds=30      
        )
        reg.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=10             
        )

    else:
        if args.use_custom_loss:
            print("Using custom MSLE loss function")
            reg = lgb.LGBMRegressor(**best, objective=msle_objective)
        else:
            reg = lgb.LGBMRegressor(**best)
        reg.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=10)
            ]
        )

    # Optional: Random Forest für Ensemble
    if args.use_random_forest:
        print("Using Random Forest in ensemble")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Einfache Params, tune bei Bedarf
        rf.fit(X_train, y_train)

    # Helper: predict with LambdaRank augmentation if needed
    def _predict(X_input):
        """Predict TED values, handling LambdaRank augmentation transparently."""
        if args.use_lambdarank and hasattr(reg, '_lambdarank_ranker'):
            rank_scores = reg._lambdarank_ranker.predict(X_input)
            X_aug = np.column_stack([X_input, rank_scores])
            return reg.predict(X_aug)
        return reg.predict(X_input)

    # 9) Cross-Validation (auf transformiertem y, falls aktiviert)
    if not args.use_lambdarank:
        if args.use_log:
            cv_y = np.log1p(y)
        else:
            cv_y = y
        cv_model = reg.__class__(**best) if not args.use_custom_loss else reg.__class__(**best, objective=msle_objective)
        scores = cross_val_score(cv_model, X_sel, cv_y, cv=5, scoring='r2')
        print("CV R² scores (main model):", scores, "Mean:", np.mean(scores))
    else:
        print("Skipping CV for LambdaRank (requires query groups)")

    # 10) Evaluation auf TEST-Set (keine Leakage)
    test_pred_tr = _predict(X_test)
    test_pred = _inverse_log_transform(test_pred_tr, args.use_log)
    test_true = _inverse_log_transform(y_test, args.use_log)
    test_pred = np.maximum(test_pred, 0)

    if args.use_random_forest:
        rf_pred = rf.predict(X_test)
        rf_pred = np.maximum(rf_pred, 0)
        test_pred = (test_pred + rf_pred) / 2  # Ensemble-Durchschnitt

    corr, _ = pearsonr(test_true, test_pred)
    print(f"Test corr: {corr:.4f}")

    mape = np.mean(np.abs((test_true - test_pred) / (test_true + 1e-6))) * 100
    print(f"Test MAPE: {mape:.2f}%")

    # Scatter-Plot
    plt.figure(figsize=(5,3))
    plt.scatter(test_pred, test_true, alpha=0.01)
    plt.xlabel("Predicted TED")
    plt.ylabel("True TED")

    m, b = np.polyfit(test_pred, test_true, 1)
    plt.plot(test_pred, m * test_pred + b, "r-", linewidth=1)

    plt.text(
        0.05, 0.95,
        f"$r = {corr:.4f}$",
        transform=plt.gca().transAxes, 
        ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0)
    )
    plt.tight_layout()
    plt.savefig("ted_prediction_plot.pdf", dpi=300, format="pdf")
    plt.close()

    # Histogram (Test-Targets)
    plt.figure(figsize=(8,6))
    plt.hist(test_true, bins=50)
    plt.xlabel('True TED (test)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('ted_distribution_test.png')

    # Prozentuale Abweichung (Test)
    percentage_deviation = np.abs((test_pred - test_true) / (test_true + 1e-6)) * 100
    plt.figure(figsize=(8,6))
    plt.scatter(test_true, percentage_deviation, alpha=0.01)
    plt.xlabel('True TED (test)')
    plt.ylabel('Absolute Percentage Deviation (%)')
    plt.title('Test: Prozentuale Abweichung')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('ted_percentage_deviation_test.png')


    # Feature-Importances speichern
    imps = reg.feature_importances_
    with open('feature_importances.json','w') as f:
        json.dump({'importances': imps.tolist()}, f, indent=2)

    # Modell speichern
    if args.use_lambdarank:
        # Save both ranker and regression head
        reg._lambdarank_ranker.booster_.save_model('../model_ranker.txt')
        reg.booster_.save_model('../model_reghead.txt')
        print("Saved ranker to ../model_ranker.txt and regression head to ../model_reghead.txt")
    elif args.use_xgboost:
        reg.save_model('../model.txt')
    else:
        reg.booster_.save_model('../model.txt')


    if args.reporting:
        # ===== Reporting & Metriken =====
        REPORT = {}

        # Datensatz & Split
        N_total = len(structures)
        total_pairs = N_total * (N_total - 1) // 2
        REPORT["dataset"] = {
            "N_total": int(N_total),
            "num_pairs": int(total_pairs)
        }
        # train/valid/test aus Shapes ableiten
        n_train = len(y_train)
        n_valid = len(y_valid)
        n_test  = len(y_test)
        n_all   = n_train + n_valid + n_test
        REPORT["split"] = {
            "train_count": int(n_train),
            "valid_count": int(n_valid),
            "test_count":  int(n_test),
            "train_percent": round(100 * n_train / max(n_all,1), 2),
            "valid_percent": round(100 * n_valid / max(n_all,1), 2),
            "test_percent":  round(100 * n_test  / max(n_all,1), 2)
        }

        # Test-Metriken auf TEST-Set: auf Originalskala auswerten
        test_pred_tr = _predict(X_test)
        test_pred = _inverse_log_transform(test_pred_tr, args.use_log)
        test_true = _inverse_log_transform(y_test, args.use_log)
        test_pred = np.maximum(test_pred, 0)

        R2 = r2_score(test_true, test_pred)
        MAE = mean_absolute_error(test_true, test_pred)
        MAPE = float(np.mean(np.abs((test_true - test_pred) / (test_true + 1e-6))) * 100)
        r_val, _ = pearsonr(test_true, test_pred)
        r_lo, r_hi = _pearson_r_ci(r_val, len(test_true))
        resid = test_true - test_pred
        IQR = float(np.percentile(resid, 75) - np.percentile(resid, 25))
        REPORT["test_metrics"] = {
            "R2": float(R2),
            "MAE": float(MAE),
            "MAPE_percent": float(MAPE),
            "pearson_r": float(r_val),
            "pearson_r_CI95": [float(r_lo), float(r_hi)],
            "residual_IQR": IQR
        }


        # Vollständige Vorhersagen für ALLE Paare (für Recall@K usw.)
        # Nutzt X_sel (alle Paare nach Feature-Selektion)
        t_pred0 = time.perf_counter()
        yhat_all_tr = _predict(X_sel)
        t_pred = time.perf_counter() - t_pred0
        yhat_all = _inverse_log_transform(yhat_all_tr, args.use_log)
        yhat_all = np.maximum(yhat_all, 0)

        pred_mat = build_pred_matrix_from_pairs(yhat_all, pairs, N_total)
        true_mat = ted_matrix.astype(np.float32)

        # Recall@K & Pruning
        K_LIST = [10, 20, 50]  # -> passe nach Bedarf an
        recall = recall_at_k(true_mat, pred_mat, ks=K_LIST)
        REPORT["recall_at_k"] = {
            "k_values": K_LIST,
            "summary": {int(k): {k2: (float(v[k2]) if isinstance(v[k2], (int, float)) else v[k2]) for k2 in ("mean","median","p25","p75")}
                        for k, v in recall.items()}
        }
        # Eine K-Größe für Pruning-Statistik auswählen (z. B. bestes/gewünschtes K)
        K_for_pruning = K_LIST[1] if len(K_LIST) > 1 else K_LIST[0]
        prune_stats = pruning_from_topk(pred_mat, K_for_pruning)
        REPORT["pruning"] = {
            "K_used": int(K_for_pruning),
            **{k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in prune_stats.items()}
        }

        # Laufzeit & Durchsatz (PredTED-Inferenz über alle Paare)
        pairs_count = X_sel.shape[0]
        throughput = pairs_count / t_pred if t_pred > 0 else float('inf')
        # Strukturen/s: grob über Paare -> ~2*Paare/N
        structures_per_sec = (2 * throughput) / max(N_total, 1)
        REPORT["timing"] = {
            "predTED_predict_seconds": float(t_pred),
            "throughput_pairs_per_sec": float(throughput),
            "throughput_structures_per_sec_est": float(structures_per_sec)
        }

        # Optional: RNAdistance Stichprobenzeit (externe Binärdatei benötigt)
        exact_probe = sample_exact_runtime_with_rnadistance(structures, sample_pairs=2000, metric='T')
        if exact_probe.get("ok", False):
            seconds_per_pair = exact_probe["seconds_per_pair"]
            exact_all_seconds = seconds_per_pair * total_pairs
            exact_forwarded_seconds = seconds_per_pair * prune_stats["forwarded_pairs"]
            REPORT["timing"].update({
                "exact_sample_seconds": float(exact_probe["sample_seconds"]),
                "exact_seconds_per_pair_est": float(seconds_per_pair),
                "exact_extrapolated_seconds_allpairs": float(exact_all_seconds),
                "exact_extrapolated_seconds_forwarded": float(exact_forwarded_seconds),
                "speedup_x_predTED_vs_exact_allpairs": float(exact_all_seconds / max(t_pred, 1e-9))
            })
        else:
            REPORT["timing"]["exact_probe_error"] = exact_probe.get("error", "unknown")

        # Trade-off n_estimators (Inference-Zeit + R2 auf VALID)
        trade = []
        for ne in [50, 100, 200, 400]:
            try:
                clone = reg.__class__(**{**best, "n_estimators": ne})
                clone.fit(X_train, y_train)
                t0 = time.perf_counter()
                pr_tr = clone.predict(X_valid)
                t1 = time.perf_counter() - t0
                pr = _inverse_log_transform(pr_tr, args.use_log)
                R2_ne = r2_score(valid_true, pr)
                trade.append({"n_estimators": ne, "inference_ms": 1000.0 * t1, "R2_test": float(R2_ne)})
            except Exception:
                pass
        REPORT["tradeoff_n_estimators"] = trade

        # Ablation (trage deine Feature-Gruppen ein!)
        # Beispiel: indices als Liste der Spalten von X_sel, die zur Gruppe gehören
        feature_groups = {
            # "depth": [...],
            # "loop":  [...],
            # "stem":  [...],
            # "bigrams": [...]
        }
        ablation = []
        for name, idxs in feature_groups.items():
            if not idxs:
                continue
            keep = [j for j in range(X_sel.shape[1]) if j not in idxs]
            X_train_k = X_train[:, keep]
            X_valid_k = X_valid[:, keep]
            model_k = reg.__class__(**best)
            model_k.fit(X_train_k, y_train)
            pred_k = _inverse_log_transform(model_k.predict(X_valid_k), args.use_log)
            R2_k = r2_score(valid_true, pred_k)
            # Recall@K-Delta auf kleiner Stichprobe (optional)
            ablation.append({"group": name, "delta_R2": float(R2_k - R2)})
        REPORT["ablation"] = ablation

        # JSON speichern
        with open("results_summary.json", "w") as f:
            json.dump(REPORT, f, indent=2)
        print("Saved metrics to results_summary.json")

        # Zusätzliche Plots: Recall@K Kurve
        try:
            plt.figure(figsize=(7,5))
            xs = list(REPORT["recall_at_k"]["summary"].keys())
            ys = [REPORT["recall_at_k"]["summary"][k]["mean"] for k in xs]
            plt.plot(xs, ys, marker='o')
            plt.xlabel('K')
            plt.ylabel('Mean Recall@K')
            plt.title('Recall@K (predTED pre-filter)')
            plt.tight_layout()
            plt.savefig('predted_recall_curve.png')
        except Exception:
            pass

    print("Done. Model saved to ../model.txt")

if __name__ == "__main__":
    main()
