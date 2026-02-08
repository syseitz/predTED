"""Feature computation for predTED, matching the C implementation exactly.

All 36 features are computed in the same order as predTED.c's
``compute_selected_features`` so that the LightGBM model produces
identical predictions.

When the C extension is available (built via ``pip install -e .``),
:func:`compute_features` delegates to the compiled code for speed.
Otherwise, the pure-Python fallback below is used transparently.

No ViennaRNA dependency — pair tables are computed in pure Python.
"""

from typing import List, Tuple

import numpy as np

NUM_FEATURES: int = 36

# Try to load the C extension built from c_src/predted_features.c
try:
    from predted._features_c import compute_features as _c_compute_features
    USING_C_BACKEND: bool = True
except ImportError:
    _c_compute_features = None
    USING_C_BACKEND: bool = False


# ---------------------------------------------------------------------------
# Pair table (1-based, matching the C implementation)
# ---------------------------------------------------------------------------

def create_pair_table(structure: str) -> List[int]:
    """Create a 1-based pair table.

    ``pt[0]`` is unused (set to 0).  For *i* in 1..n, ``pt[i]`` is the
    1-based index of the pairing partner, or 0 if position *i* is unpaired.
    """
    n = len(structure)
    pt = [0] * (n + 1)
    stack: List[int] = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i + 1)
        elif c == ')' and stack:
            j = stack.pop()
            pt[j] = i + 1
            pt[i + 1] = j
    return pt


# ---------------------------------------------------------------------------
# Depth profile
# ---------------------------------------------------------------------------

def compute_depth_profile(structure: str) -> List[int]:
    depth = 0
    profile: List[int] = []
    for c in structure:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        profile.append(depth)
    return profile


def get_depth_features(
    structure: str,
) -> Tuple[float, float, int, float, float, float, float]:
    """Return depth statistics matching the C implementation.

    Returns
    -------
    mean_depth, var_depth, peaks,
    mean_depth_paired, var_depth_paired,
    mean_depth_unpaired, var_depth_unpaired
    """
    pt = create_pair_table(structure)
    profile = compute_depth_profile(structure)
    n = len(structure)

    sum_all = sum_sq_all = 0.0
    sum_p = sum_sq_p = 0.0
    count_p = 0
    sum_u = sum_sq_u = 0.0
    count_u = 0
    peaks = 0

    for i in range(n):
        d = profile[i]
        sum_all += d
        sum_sq_all += d * d

        if pt[i + 1] > 0:  # paired ('(' or ')')
            sum_p += d
            sum_sq_p += d * d
            count_p += 1
        elif structure[i] == '.':  # unpaired
            sum_u += d
            sum_sq_u += d * d
            count_u += 1

        if 0 < i < n - 1 and profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
            peaks += 1

    mean_d = sum_all / n if n > 0 else 0.0
    var_d = (sum_sq_all / n - mean_d * mean_d) if n > 1 else 0.0

    mean_pd = sum_p / count_p if count_p > 0 else 0.0
    var_pd = (sum_sq_p / count_p - mean_pd * mean_pd) if count_p > 1 else 0.0

    mean_ud = sum_u / count_u if count_u > 0 else 0.0
    var_ud = (sum_sq_u / count_u - mean_ud * mean_ud) if count_u > 1 else 0.0

    return mean_d, var_d, peaks, mean_pd, var_pd, mean_ud, var_ud


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------

def find_stems(structure: str) -> List[int]:
    """Return a list of stem lengths (only stems with length > 1)."""
    pt = create_pair_table(structure)
    n = len(structure)
    stems: List[int] = []
    visited = set()
    i = 1
    while i < n:  # C: for (i = 1; i < len; i++)
        if pt[i] > i and i not in visited and pt[i] not in visited:
            j = pt[i]
            length = 1
            while i + 1 <= n and pt[i + 1] == j - 1:
                i += 1
                j -= 1
                length += 1
                visited.add(i)
                visited.add(j)
            if length > 1:
                stems.append(length)
        i += 1
    return stems


def get_stem_features(structure: str) -> Tuple[int, float, float, float]:
    """Return (num_stems, max_stem_length, avg_stem_length, var_stem_length)."""
    stems = find_stems(structure)
    count = len(stems)
    if count == 0:
        return 0, 0.0, 0.0, 0.0
    s = sum(stems)
    ssq = sum(x * x for x in stems)
    mx = max(stems)
    avg = s / count
    var = (ssq / count - avg * avg) if count > 1 else 0.0
    return count, float(mx), avg, var


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

def get_loop_features(structure: str) -> Tuple[float, float]:
    """Return (mean_loop_size, var_loop_size)."""
    sizes: List[int] = []
    current = 0
    for c in structure:
        if c == '.':
            current += 1
        else:
            if current > 0:
                sizes.append(current)
                current = 0
    if current > 0:
        sizes.append(current)

    if not sizes:
        return 0.0, 0.0
    s = sum(sizes)
    ssq = sum(x * x for x in sizes)
    count = len(sizes)
    mean = s / count
    var = (ssq / count - mean * mean) if count > 1 else 0.0
    return mean, var


def max_loop_size(structure: str) -> int:
    mx = 0
    current = 0
    for c in structure:
        if c == '.':
            current += 1
        else:
            if current > mx:
                mx = current
            current = 0
    return max(mx, current)


def count_internal_loops(structure: str) -> int:
    n = len(structure)
    internal_loops = 0
    i = 0
    while i < n - 1:
        if structure[i] == ')' and structure[i + 1] == '.':
            j = i + 1
            while j < n and structure[j] == '.':
                j += 1
            if j < n and structure[j] == '(':
                internal_loops += 1
            i = j
        else:
            i += 1
    return internal_loops


def count_multiloops(structure: str) -> int:
    n = len(structure)
    stack: List[int] = []
    multiloops = 0
    for i in range(n):
        if structure[i] == '(':
            stack.append(i)
        elif structure[i] == ')' and stack:
            stack.pop()
            if stack and i + 1 < n and structure[i + 1] == '(':
                multiloops += 1
    return multiloops


def count_bulges(structure: str) -> int:
    n = len(structure)
    stack: List[int] = []
    bulges = 0
    for i in range(n):
        if structure[i] == '(':
            stack.append(i)
        elif structure[i] == ')' and stack:
            stack.pop()
        elif structure[i] == '.' and stack:
            if i + 1 < n and structure[i + 1] == ')':
                bulges += 1
    return bulges


# ---------------------------------------------------------------------------
# Hairpin loops
# ---------------------------------------------------------------------------

def count_hairpin_loops(structure: str) -> int:
    pt = create_pair_table(structure)
    n = len(structure)
    count = 0
    for i in range(1, n + 1):
        if pt[i] > i:
            j = pt[i]
            if all(pt[k] == 0 for k in range(i + 1, j)):
                count += 1
    return count


def get_hairpin_loop_sizes(structure: str) -> List[int]:
    pt = create_pair_table(structure)
    n = len(structure)
    sizes: List[int] = []
    for i in range(1, n + 1):
        if pt[i] > i:
            j = pt[i]
            if all(pt[k] == 0 for k in range(i + 1, j)):
                sizes.append(j - i - 1)
    return sizes


# ---------------------------------------------------------------------------
# Internal loop sizes (matching C implementation exactly)
# ---------------------------------------------------------------------------

def get_internal_loop_sizes(structure: str) -> List[int]:
    pt = create_pair_table(structure)
    n = len(structure)
    sizes: List[int] = []
    i = 1
    while i <= n:
        if pt[i] > i:
            j = pt[i]
            for k in range(i + 1, j):
                if pt[k] > k and pt[k] < j:
                    m = pt[k]
                    left_unpaired = k - i - 1
                    right_unpaired = j - m - 1
                    if left_unpaired > 0 and right_unpaired > 0:
                        sizes.append(left_unpaired + right_unpaired)
                    i = m  # C: i = m, then for-loop does i++
                    break
        i += 1
    return sizes


# ---------------------------------------------------------------------------
# Stacked pairs & base pair distances
# ---------------------------------------------------------------------------

def count_stacked_pairs(structure: str) -> int:
    pt = create_pair_table(structure)
    n = len(structure)
    stacked = 0
    for i in range(1, n):
        if pt[i] > i and pt[i + 1] == pt[i] - 1:
            stacked += 1
    return stacked


def get_base_pair_distances(structure: str) -> Tuple[float, int]:
    """Return (avg_bp_distance, max_bp_distance)."""
    pt = create_pair_table(structure)
    n = len(structure)
    total = 0
    count = 0
    mx = 0
    for i in range(1, n + 1):
        if pt[i] > i:
            dist = pt[i] - i
            total += dist
            count += 1
            if dist > mx:
                mx = dist
    avg = total / count if count > 0 else 0.0
    return avg, mx


# ---------------------------------------------------------------------------
# Simple counts
# ---------------------------------------------------------------------------

def num_paired_bases(structure: str) -> int:
    return 2 * structure.count('(')


def num_unpaired_bases(structure: str) -> int:
    return structure.count('.')


def graph_centrality(structure: str) -> float:
    pairs = structure.count('(')
    return pairs / len(structure) if structure else 0.0


def tree_depth(structure: str) -> int:
    depth = 0
    mx = 0
    for c in structure:
        if c == '(':
            depth += 1
            if depth > mx:
                mx = depth
        elif c == ')':
            depth -= 1
    return mx


# ---------------------------------------------------------------------------
# N-gram features
# ---------------------------------------------------------------------------

def get_ngram_features(structure: str, n: int = 2) -> List[float]:
    """Return n-gram frequencies in the order ``(().`` × ``(().``.

    For bigrams (n=2) this gives 9 values in the order:
    ``((``, ``()``, ``(. ``, ``)(``, ``))``, ``). ``, ``.(``, ``.)``, ``..``
    """
    length = len(structure)
    symbols = '().'
    if n == 2:
        possible = [a + b for a in symbols for b in symbols]
    else:
        possible = [a + b + c for a in symbols for b in symbols for c in symbols]

    total = length - n + 1
    if total <= 0:
        return [0.0] * len(possible)

    counts = {ng: 0 for ng in possible}
    for i in range(total):
        gram = structure[i:i + n]
        if gram in counts:
            counts[gram] += 1

    return [counts[ng] / total for ng in possible]


# ---------------------------------------------------------------------------
# Combined feature vector (36 features, exact C order)
# ---------------------------------------------------------------------------

def _compute_features_python(structure: str) -> np.ndarray:
    """Pure-Python fallback: compute the 36-element feature vector."""
    features = np.zeros(NUM_FEATURES, dtype=np.float64)

    features[0] = count_internal_loops(structure)

    mean_d, var_d, peaks, mean_pd, var_pd, mean_ud, var_ud = get_depth_features(structure)
    features[1] = var_pd
    features[2] = count_multiloops(structure)
    features[3] = max_loop_size(structure)
    features[4] = len(structure)

    mean_loop, var_loop = get_loop_features(structure)
    features[5] = mean_loop
    features[6] = tree_depth(structure)
    features[7] = mean_ud
    features[8] = count_bulges(structure)
    features[9] = var_loop
    features[10] = graph_centrality(structure)

    n_stems, max_stem, avg_stem, var_stem = get_stem_features(structure)
    features[11] = var_stem
    features[12] = max_stem
    features[13] = avg_stem
    features[14] = mean_pd
    features[15] = n_stems
    features[16] = var_ud
    features[17] = var_d
    features[18] = mean_d
    features[19] = count_hairpin_loops(structure)
    features[20] = count_stacked_pairs(structure)

    avg_bp, max_bp = get_base_pair_distances(structure)
    features[21] = avg_bp
    features[22] = num_paired_bases(structure)
    features[23] = num_unpaired_bases(structure)

    hairpin_sizes = get_hairpin_loop_sizes(structure)
    features[24] = (sum(hairpin_sizes) / len(hairpin_sizes)) if hairpin_sizes else 0.0
    features[25] = max(hairpin_sizes) if hairpin_sizes else 0

    internal_sizes = get_internal_loop_sizes(structure)
    features[26] = (sum(internal_sizes) / len(internal_sizes)) if internal_sizes else 0.0
    features[27] = max(internal_sizes) if internal_sizes else 0

    # Bigrams — skip index 1 ("()") to match C code
    bigrams = get_ngram_features(structure, n=2)
    features[28] = bigrams[0]   # "(("
    features[29] = bigrams[2]   # "(."
    features[30] = bigrams[3]   # ")("
    features[31] = bigrams[4]   # "))"
    features[32] = bigrams[5]   # ")."
    features[33] = bigrams[6]   # ".("
    features[34] = bigrams[7]   # ".)"
    features[35] = bigrams[8]   # ".."

    return features


def compute_features(structure: str) -> np.ndarray:
    """Compute the 36-element feature vector matching ``compute_selected_features`` in C.

    Uses the compiled C extension when available, otherwise falls back
    to the pure-Python implementation above.
    """
    if _c_compute_features is not None:
        return np.array(_c_compute_features(structure), dtype=np.float64)
    return _compute_features_python(structure)
