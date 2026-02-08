import numpy as np
import json
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from utils.features import *

def compute_features(structure: str) -> np.ndarray:
    length = len(structure)
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
    trigram_features = get_ngram_features(structure, n=3)
    hairpin_loops = count_hairpin_loops(structure)
    stacked_pairs = count_stacked_pairs(structure)
    avg_bp_dist, max_bp_dist = get_base_pair_distances(structure)
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
        max_bp_dist,
        paired_bases,
        unpaired_bases,
        mean_hairpin_size,
        max_hairpin_size,
        mean_internal_loop_size,
        max_internal_loop_size,
        mean_bulge_size,
        max_bulge_size
    ] + bigram_features + trigram_features
    return np.array(features, dtype=np.float32)

def main(structures_file: str = '../data/structures.txt', ted_matrix_file: str = '../data/ted_matrix.txt', corr_threshold: float = 0):
    # Read structures and TED matrix
    with open(structures_file, 'r') as f:
        structures = [line.strip() for line in f if line.strip()]
    ted_matrix = np.loadtxt(ted_matrix_file, delimiter=' ')
    
    if len(structures) != ted_matrix.shape[0]:
        print("Error: Number of structures does not match TED matrix dimensions.")
        return
    
    # Define feature names
    other_feature_names = [
        "internal_loops",
        "var_depth_paired",
        "multiloops",
        "max_loop",
        "length",
        "mean_loop",
        "depth",
        "mean_depth_unpaired",
        "bulges",
        "var_loop",
        "centrality",
        "var_stem_length",
        "max_stem_length",
        "avg_stem_length",
        "mean_depth_paired",
        "num_peaks",
        "num_stems",
        "var_depth_unpaired",
        "var_depth",
        "mean_depth",
        "hairpin_loops",
        "stacked_pairs",
        "avg_bp_dist",
        "max_bp_dist",
        "paired_bases",
        "unpaired_bases",
        "mean_hairpin_size",
        "max_hairpin_size",
        "mean_internal_loop_size",
        "max_internal_loop_size",
        "mean_bulge_size",
        "max_bulge_size"
    ]
    bigram_names = [f"bigram_{''.join(comb)}" for comb in itertools.product('().', repeat=2)]
    trigram_names = [f"trigram_{''.join(comb)}" for comb in itertools.product('().', repeat=3)]
    feature_names = other_feature_names + bigram_names + trigram_names
    
    # Compute feature matrix
    print("Computing feature matrix...")
    feature_matrix = np.array([compute_features(struct) for struct in tqdm(structures)])
    
    # Generate pairs and compute differences
    m = len(structures)
    pairs = list(itertools.combinations(range(m), 2))
    X = np.array([np.abs(feature_matrix[i] - feature_matrix[j]) for i, j in pairs])
    y = np.array([ted_matrix[i, j] for i, j in pairs])
    
    # Filter out constant features
    variances = np.var(X, axis=0)
    non_constant_indices = np.where(variances > 0)[0]
    X = X[:, non_constant_indices]
    feature_names = [feature_names[i] for i in non_constant_indices]
    
    # Compute correlations for feature selection
    print("Computing correlations between feature differences and TED...")
    corrs = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]
    selected_indices = [i for i, corr in enumerate(corrs) if abs(corr) >= corr_threshold]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    print(f"Selected features: {selected_feature_names}")
    
    if not selected_indices:
        print("No features selected. Exiting.")
        return
    
    # Select features
    X_selected = X[:, selected_indices]
    
    # Scale selected features
    scaler = StandardScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    # Fit LinearRegression model with intercept
    print("Fitting LinearRegression model...")
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_selected_scaled, y)
    weights = reg.coef_
    intercept = reg.intercept_
    
    # Compute predicted TED
    predicted_TED = reg.predict(X_selected_scaled)
    
    # Clip negative predictions
    predicted_TED = np.maximum(predicted_TED, 0)
    
    # Compute correlation
    corr, _ = pearsonr(y, predicted_TED)
    print(f"Correlation between predicted and true TED: {corr:.4f}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_TED, y, alpha=0.01, label=f'Correlation: {corr:.4f}')
    plt.xlabel('Predicted TED')
    plt.ylabel('True TED')
    plt.title('True vs Predicted Tree Edit Distance')
    m, b = np.polyfit(predicted_TED, y, 1)
    plt.plot(predicted_TED, m * predicted_TED + b, color='red', label='Regression Line')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ted_prediction_plot.png')
    print("Scatter plot saved as 'ted_prediction_plot.png'")
    
    # Create histogram of TED values
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, color='blue', alpha=0.7)
    plt.xlabel('True TED')
    plt.ylabel('Frequency')
    plt.title('Distribution of True Tree Edit Distance Values')
    plt.tight_layout()
    plt.savefig('ted_distribution.png')
    print("Histogram saved as 'ted_distribution.png'")
    
    # Save weights to JSON
    weights_dict = {selected_feature_names[i]: float(weights[i]) for i in range(len(weights))}
    sorted_weights = sorted(weights_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_weights_dict = dict(sorted_weights)
    with open('feature_weights.json', 'w') as f:
        json.dump(sorted_weights_dict, f, indent=4)
    print("Feature weights saved to 'feature_weights.json'")

if __name__ == "__main__":
    main()