# predTED Performance Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximise predTED speed (35-50x on 32-core cluster) and minimise RAM for pairwise distance matrix computation.

**Architecture:** Optimise the C feature computation (StructureContext, n-gram), parallelise the pairwise loop (one LightGBM booster per thread), add SIMD for pairwise feature building, stream the Python API, and tune batch sizes.

**Tech Stack:** C11, OpenMP, ARM NEON / x86 SSE, LightGBM C API, Python/NumPy

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `c_src/predted_features.h` | Modify | Add `StructureContext` typedef, `create_context`/`destroy_context` |
| `c_src/predted_features.c` | Modify | Refactor sub-functions to accept context, optimise n-grams |
| `c_src/predTED.c` | Modify | Multi-threaded pairwise loop, SIMD feature building, float32 pre-conversion |
| `predted/__init__.py` | Modify | Streaming `predict_matrix`, vectorised feature building |
| `Makefile` | Modify | Add `-march=native`, `bench-batch` target |
| `tests/test_predict.py` | Modify | Add regression tests for streaming predict_matrix |
| `tests/test_features_regression.py` | Create | Capture + verify feature values survive refactoring |
| `benchmarks/bench_batch_size.sh` | Create | Batch-size sweep script |

---

### Task 1: Baseline — Capture Regression Data

**Files:**
- Create: `tests/test_features_regression.py`

This captures current feature outputs and prediction values so we can verify correctness after every refactoring step.

- [ ] **Step 1: Write the feature regression test**

```python
"""Regression tests: feature values must not change after refactoring."""

import numpy as np
import pytest

from predted.features import compute_features, NUM_FEATURES

# Structures covering: simple hairpin, multi-loop, internal loop, stacked, unpaired-only
STRUCTURES = [
    "((..))",
    "(())..",
    "...((..))...",
    "((((...))))",
    "(((..(((....)))..((....))..)))",
    "................",
    "((((((((((...((.(((...)).))).)))))))))",
]

# Captured from current (pre-optimisation) code — filled in Step 3
EXPECTED_FEATURES = None


class TestFeatureRegression:
    """Verify features match pre-optimisation baseline."""

    @pytest.fixture(autouse=True)
    def _load_expected(self):
        if EXPECTED_FEATURES is None:
            pytest.skip("Run capture_baseline() first to populate EXPECTED_FEATURES")

    @pytest.mark.parametrize("idx", range(len(STRUCTURES)))
    def test_features_match_baseline(self, idx: int):
        actual = compute_features(STRUCTURES[idx])
        expected = np.array(EXPECTED_FEATURES[idx], dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=1e-10,
                                   err_msg=f"Feature mismatch for structure {idx}: {STRUCTURES[idx][:40]}")


def capture_baseline():
    """Run once to print the baseline feature values for copy-paste into EXPECTED_FEATURES."""
    print("EXPECTED_FEATURES = [")
    for s in STRUCTURES:
        feats = compute_features(s)
        print(f"    {feats.tolist()},")
    print("]")


if __name__ == "__main__":
    capture_baseline()
```

- [ ] **Step 2: Run the capture script to get baseline values**

Run: `cd /Volumes/Masterarbeit/predTED && python tests/test_features_regression.py`

This prints the `EXPECTED_FEATURES` array. Copy the output.

- [ ] **Step 3: Paste captured values into the test file**

Replace `EXPECTED_FEATURES = None` with the captured output (the `EXPECTED_FEATURES = [...]` block).

- [ ] **Step 4: Run the regression test to verify it passes**

Run: `cd /Volumes/Masterarbeit/predTED && python -m pytest tests/test_features_regression.py -v`

Expected: All 7 parametrised tests PASS.

- [ ] **Step 5: Run full existing test suite as baseline**

Run: `cd /Volumes/Masterarbeit/predTED && python -m pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add tests/test_features_regression.py
git commit -m "test: add feature regression tests for optimisation baseline"
```

---

### Task 2: StructureContext Refactoring (`predted_features.c/h`)

**Files:**
- Modify: `c_src/predted_features.h`
- Modify: `c_src/predted_features.c`

**Goal:** Compute pair table and length once per structure instead of 7 times.

- [ ] **Step 1: Add StructureContext to the header**

Add after the `#define NUM_FEATURES_BASE 36` line in `c_src/predted_features.h`:

```c
/* Pre-computed context for a single structure (avoids redundant work). */
typedef struct {
    const char *structure;
    int         len;
    short      *pair_table;   /* 1-based, allocated; caller frees via destroy_context */
} StructureContext;

/* Create/destroy a context (pair_table allocated on heap). */
StructureContext create_context(const char *structure);
void             destroy_context(StructureContext *ctx);
```

- [ ] **Step 2: Implement context creation/destruction in `predted_features.c`**

Add after the existing `create_pair_table` function (after line 42):

```c
StructureContext create_context(const char *structure) {
    StructureContext ctx;
    ctx.structure = structure;
    ctx.len = (int)strlen(structure);
    ctx.pair_table = create_pair_table(structure);
    return ctx;
}

void destroy_context(StructureContext *ctx) {
    free(ctx->pair_table);
    ctx->pair_table = NULL;
}
```

- [ ] **Step 3: Refactor `get_depth_features` to accept context**

Replace the existing `get_depth_features` function (lines 59-96) with:

```c
void get_depth_features_ctx(const StructureContext *ctx, double* mean_depth, double* var_depth, int* peaks, double* mean_depth_paired, double* var_depth_paired, double* mean_depth_unpaired, double* var_depth_unpaired) {
    const int len = ctx->len;
    const short *pt = ctx->pair_table;
    const char *structure = ctx->structure;
    int* profile = compute_depth_profile(structure);
    double sum = 0, sum_sq = 0;
    int count = 0;
    double sum_paired = 0, sum_sq_paired = 0;
    int count_paired = 0;
    double sum_unpaired = 0, sum_sq_unpaired = 0;
    int count_unpaired = 0;
    int peak_count = 0;
    for (int i = 0; i < len; i++) {
        sum += profile[i];
        sum_sq += profile[i] * profile[i];
        count++;
        if (pt[i + 1] > 0) {
            sum_paired += profile[i];
            sum_sq_paired += profile[i] * profile[i];
            count_paired++;
        } else if (structure[i] == '.') {
            sum_unpaired += profile[i];
            sum_sq_unpaired += profile[i] * profile[i];
            count_unpaired++;
        }
        if (i > 0 && i < len - 1 && profile[i] > profile[i - 1] && profile[i] > profile[i + 1]) {
            peak_count++;
        }
    }
    *mean_depth = count > 0 ? sum / count : 0;
    *var_depth = count > 1 ? (sum_sq / count - (*mean_depth) * (*mean_depth)) : 0;
    *peaks = peak_count;
    *mean_depth_paired = count_paired > 0 ? sum_paired / count_paired : 0;
    *var_depth_paired = count_paired > 1 ? (sum_sq_paired / count_paired - (*mean_depth_paired) * (*mean_depth_paired)) : 0;
    *mean_depth_unpaired = count_unpaired > 0 ? sum_unpaired / count_unpaired : 0;
    *var_depth_unpaired = count_unpaired > 1 ? (sum_sq_unpaired / count_unpaired - (*mean_depth_unpaired) * (*mean_depth_unpaired)) : 0;
    free(profile);
}

void get_depth_features(const char* structure, double* mean_depth, double* var_depth, int* peaks, double* mean_depth_paired, double* var_depth_paired, double* mean_depth_unpaired, double* var_depth_unpaired) {
    StructureContext ctx = create_context(structure);
    get_depth_features_ctx(&ctx, mean_depth, var_depth, peaks, mean_depth_paired, var_depth_paired, mean_depth_unpaired, var_depth_unpaired);
    destroy_context(&ctx);
}
```

- [ ] **Step 4: Refactor `find_stems` to accept context**

Replace `find_stems` (lines 98-124) with:

```c
int* find_stems_ctx(const StructureContext *ctx) {
    const short *pt = ctx->pair_table;
    const int len = ctx->len;
    int* stems = (int*)malloc((len + 1) * sizeof(int));
    int stem_count = 0;
    int* visited = (int*)calloc(len + 1, sizeof(int));
    for (int i = 1; i < len; i++) {
        if (pt[i] > i && !visited[i] && !visited[pt[i]]) {
            int j = pt[i];
            int stem_length = 1;
            while (i + 1 <= len && pt[i + 1] == j - 1) {
                i++;
                j--;
                stem_length++;
                visited[i] = 1;
                visited[j] = 1;
            }
            if (stem_length > 1) {
                stems[stem_count++] = stem_length;
            }
        }
    }
    stems[stem_count] = -1;
    free(visited);
    return stems;
}

int* find_stems(const char* structure) {
    StructureContext ctx = create_context(structure);
    int* result = find_stems_ctx(&ctx);
    destroy_context(&ctx);
    return result;
}
```

Update `get_stem_features` to also have a `_ctx` variant:

```c
void get_stem_features_ctx(const StructureContext *ctx, int* num_stems, double* max_stem_length, double* avg_stem_length, double* var_stem_length) {
    int* stems = find_stems_ctx(ctx);
    int count = 0;
    double sum = 0, sum_sq = 0;
    int max_len = 0;
    for (int i = 0; stems[i] != -1; i++) {
        count++;
        sum += stems[i];
        sum_sq += stems[i] * stems[i];
        max_len = max_len > stems[i] ? max_len : stems[i];
    }
    *num_stems = count;
    *max_stem_length = max_len;
    *avg_stem_length = count > 0 ? sum / count : 0;
    *var_stem_length = count > 1 ? (sum_sq / count - (*avg_stem_length) * (*avg_stem_length)) : 0;
    free(stems);
}

void get_stem_features(const char* structure, int* num_stems, double* max_stem_length, double* avg_stem_length, double* var_stem_length) {
    StructureContext ctx = create_context(structure);
    get_stem_features_ctx(&ctx, num_stems, max_stem_length, avg_stem_length, var_stem_length);
    destroy_context(&ctx);
}
```

- [ ] **Step 5: Refactor remaining pair-table functions to accept context**

Apply the same pattern to these functions — each gets a `_ctx` variant that uses `ctx->pair_table` instead of calling `create_pair_table`, and the old function becomes a wrapper:

**`count_hairpin_loops`** (lines 220-241):

```c
int count_hairpin_loops_ctx(const StructureContext *ctx) {
    const short *pt = ctx->pair_table;
    const int len = ctx->len;
    int hairpin_loops = 0;
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int j = pt[i];
            int unpaired = 1;
            for (int k = i + 1; k < j; k++) {
                if (pt[k] != 0) { unpaired = 0; break; }
            }
            if (unpaired) hairpin_loops++;
        }
    }
    return hairpin_loops;
}

int count_hairpin_loops(const char* structure) {
    StructureContext ctx = create_context(structure);
    int result = count_hairpin_loops_ctx(&ctx);
    destroy_context(&ctx);
    return result;
}
```

**`count_stacked_pairs`** (lines 243-254):

```c
int count_stacked_pairs_ctx(const StructureContext *ctx) {
    const short *pt = ctx->pair_table;
    const int len = ctx->len;
    int stacked = 0;
    for (int i = 1; i < len; i++) {
        if (pt[i] > i && pt[i + 1] == pt[i] - 1) stacked++;
    }
    return stacked;
}

int count_stacked_pairs(const char* structure) {
    StructureContext ctx = create_context(structure);
    int result = count_stacked_pairs_ctx(&ctx);
    destroy_context(&ctx);
    return result;
}
```

**`get_base_pair_distances`** (lines 256-273):

```c
void get_base_pair_distances_ctx(const StructureContext *ctx, double* avg_bp_dist, int* max_bp_dist) {
    const short *pt = ctx->pair_table;
    const int len = ctx->len;
    double sum = 0;
    int count = 0, max_dist = 0;
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int dist = pt[i] - i;
            sum += dist;
            count++;
            max_dist = max_dist > dist ? max_dist : dist;
        }
    }
    *avg_bp_dist = count > 0 ? sum / count : 0;
    *max_bp_dist = max_dist;
}

void get_base_pair_distances(const char* structure, double* avg_bp_dist, int* max_bp_dist) {
    StructureContext ctx = create_context(structure);
    get_base_pair_distances_ctx(&ctx, avg_bp_dist, max_bp_dist);
    destroy_context(&ctx);
}
```

**`get_hairpin_loop_sizes`** (lines 283-306):

```c
int* get_hairpin_loop_sizes_ctx(const StructureContext *ctx) {
    const short *pt = ctx->pair_table;
    const int len = ctx->len;
    int* sizes = (int*)malloc((len + 1) * sizeof(int));
    int count = 0;
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int j = pt[i];
            int unpaired = 1;
            for (int k = i + 1; k < j; k++) {
                if (pt[k] != 0) { unpaired = 0; break; }
            }
            if (unpaired) sizes[count++] = j - i - 1;
        }
    }
    sizes[count] = -1;
    return sizes;
}

int* get_hairpin_loop_sizes(const char* structure) {
    StructureContext ctx = create_context(structure);
    int* result = get_hairpin_loop_sizes_ctx(&ctx);
    destroy_context(&ctx);
    return result;
}
```

**`get_internal_loop_sizes`** (lines 414-439):

```c
int* get_internal_loop_sizes_ctx(const StructureContext *ctx) {
    const int len = ctx->len;
    const short *pt = ctx->pair_table;
    int* sizes = (int*)malloc((len + 1) * sizeof(int));
    int count = 0;
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int j = pt[i];
            for (int k = i + 1; k < j; k++) {
                if (pt[k] > k && pt[k] < j) {
                    int m = pt[k];
                    int left_unpaired = k - i - 1;
                    int right_unpaired = j - m - 1;
                    if (left_unpaired > 0 && right_unpaired > 0) {
                        sizes[count++] = left_unpaired + right_unpaired;
                    }
                    i = m;
                    break;
                }
            }
        }
    }
    sizes[count] = -1;
    return sizes;
}

int* get_internal_loop_sizes(const char* structure) {
    StructureContext ctx = create_context(structure);
    int* result = get_internal_loop_sizes_ctx(&ctx);
    destroy_context(&ctx);
    return result;
}
```

- [ ] **Step 6: Rewrite `compute_selected_features` to use context**

Replace the existing `compute_selected_features` function (lines 488-542) with:

```c
void compute_selected_features(const char* structure, double* features) {
    StructureContext ctx = create_context(structure);

    features[0] = count_internal_loops(structure);  /* no pair_table needed */

    double mean_depth, var_depth, mean_depth_paired, var_depth_paired, mean_depth_unpaired, var_depth_unpaired;
    int peaks;
    get_depth_features_ctx(&ctx, &mean_depth, &var_depth, &peaks, &mean_depth_paired, &var_depth_paired, &mean_depth_unpaired, &var_depth_unpaired);
    features[1] = var_depth_paired;
    features[2] = count_multiloops(structure);       /* no pair_table needed */
    features[3] = max_loop_size(structure);           /* no pair_table needed */
    features[4] = ctx.len;

    double mean_loop, var_loop;
    get_loop_features(structure, &mean_loop, &var_loop);  /* no pair_table needed */
    features[5] = mean_loop;
    features[6] = tree_depth(structure);              /* no pair_table needed */
    features[7] = mean_depth_unpaired;
    features[8] = count_bulges(structure);            /* no pair_table needed */
    features[9] = var_loop;
    features[10] = graph_centrality(structure);       /* no pair_table needed */

    int num_stems;
    double max_stem_length, avg_stem_length, var_stem_length;
    get_stem_features_ctx(&ctx, &num_stems, &max_stem_length, &avg_stem_length, &var_stem_length);
    features[11] = var_stem_length;
    features[12] = max_stem_length;
    features[13] = avg_stem_length;
    features[14] = mean_depth_paired;
    features[15] = num_stems;
    features[16] = var_depth_unpaired;
    features[17] = var_depth;
    features[18] = mean_depth;
    features[19] = count_hairpin_loops_ctx(&ctx);
    features[20] = count_stacked_pairs_ctx(&ctx);

    double avg_bp_dist;
    int max_bp_dist;
    get_base_pair_distances_ctx(&ctx, &avg_bp_dist, &max_bp_dist);
    features[21] = avg_bp_dist;
    features[22] = num_paired_bases(structure);       /* no pair_table needed */
    features[23] = num_unpaired_bases(structure);     /* no pair_table needed */

    int* hairpin_sizes = get_hairpin_loop_sizes_ctx(&ctx);
    features[24] = compute_mean(hairpin_sizes);
    features[25] = compute_max(hairpin_sizes);
    free(hairpin_sizes);

    int* internal_loop_sizes = get_internal_loop_sizes_ctx(&ctx);
    features[26] = compute_mean(internal_loop_sizes);
    features[27] = compute_max(internal_loop_sizes);
    free(internal_loop_sizes);

    double* bigram_features = get_ngram_features(structure, 2);
    features[28] = bigram_features[0];
    features[29] = bigram_features[2];
    features[30] = bigram_features[3];
    features[31] = bigram_features[4];
    features[32] = bigram_features[5];
    features[33] = bigram_features[6];
    features[34] = bigram_features[7];
    features[35] = bigram_features[8];
    free(bigram_features);

    destroy_context(&ctx);
}
```

- [ ] **Step 7: Build and test**

Run: `cd /Volumes/Masterarbeit/predTED && pip install -e . && python -m pytest tests/ -v`

Expected: All tests PASS (features identical to baseline).

- [ ] **Step 8: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add c_src/predted_features.h c_src/predted_features.c
git commit -m "perf: refactor feature computation — single pair table per structure"
```

---

### Task 3: N-gram Optimisation (`predted_features.c`)

**Files:**
- Modify: `c_src/predted_features.c`

**Goal:** Replace strcmp-based n-gram matching with direct character indexing.

- [ ] **Step 1: Replace `get_ngram_features` with optimised version**

Replace the existing `get_ngram_features` function (lines 172-218 approximately, after Task 2 edits look for the function) with:

```c
static inline int char_idx(char c) {
    /* '(' -> 0, ')' -> 1, '.' -> 2 */
    return c == '(' ? 0 : (c == ')' ? 1 : 2);
}

double* get_ngram_features(const char* structure, int n) {
    int len = (int)strlen(structure);
    int num_ngrams = n == 2 ? 9 : 27;
    double* frequencies = (double*)calloc(num_ngrams, sizeof(double));
    int total = len - n + 1;
    if (total <= 0) return frequencies;

    if (n == 2) {
        for (int i = 0; i < total; i++) {
            int idx = char_idx(structure[i]) * 3 + char_idx(structure[i + 1]);
            frequencies[idx]++;
        }
    } else {
        for (int i = 0; i < total; i++) {
            int idx = char_idx(structure[i]) * 9
                    + char_idx(structure[i + 1]) * 3
                    + char_idx(structure[i + 2]);
            frequencies[idx]++;
        }
    }

    for (int j = 0; j < num_ngrams; j++) {
        frequencies[j] /= total;
    }
    return frequencies;
}
```

- [ ] **Step 2: Build and test**

Run: `cd /Volumes/Masterarbeit/predTED && pip install -e . && python -m pytest tests/ -v`

Expected: All tests PASS (n-gram values unchanged — same index order as before).

- [ ] **Step 3: Also build the CLI to verify it compiles**

Run: `cd /Volumes/Masterarbeit/predTED && make cli`

Expected: Compiles without errors.

- [ ] **Step 4: Quick CLI smoke test**

Run: `cd /Volumes/Masterarbeit/predTED && echo -e "((..))\n(())..\n...((..))..." | bin/predted`

Expected: Prints a 3×3 distance matrix (same values as before).

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add c_src/predted_features.c
git commit -m "perf: optimise n-gram computation — direct char indexing instead of strcmp"
```

---

### Task 4: Multi-threaded Pairwise Loop (`predTED.c`)

**Files:**
- Modify: `c_src/predTED.c`

This is the largest and most impactful change. The current single-threaded pairwise loop becomes parallelised with one LightGBM booster per thread.

- [ ] **Step 1: Add multi-booster loading after single booster load**

In `predTED.c`, replace the single booster loading section (lines 242-259 approximately) with code that loads one booster per thread. Find the section starting with `// Load LightGBM model from embedded bytes` and replace through `free(model_str);`:

```c
    // Load LightGBM model — one booster per thread for parallel prediction
    char *model_str = (char*)malloc((size_t)model_txt_len + 1);
    if (!model_str) {
        fprintf(stderr, "Out of memory allocating model string\n");
        return 1;
    }
    memcpy(model_str, model_txt, (size_t)model_txt_len);
    model_str[model_txt_len] = '\0';

    BoosterHandle *boosters = (BoosterHandle*)malloc((size_t)num_threads * sizeof(BoosterHandle));
    if (!boosters) {
        fprintf(stderr, "Out of memory allocating boosters array\n");
        free(model_str);
        return 1;
    }
    for (int t = 0; t < num_threads; t++) {
        int total_iterations;
        if (LGBM_BoosterLoadModelFromString(model_str, &total_iterations, &boosters[t]) != 0) {
            fprintf(stderr, "Error loading model for thread %d\n", t);
            free(model_str);
            return 1;
        }
    }
    free(model_str);
    fprintf(stderr, "[predTED] loaded %d booster instances\n", num_threads);
```

- [ ] **Step 2: Replace per-thread buffer allocation**

Replace the single-thread buffer allocation section (the `row_int`/`row_flt`, `batch_diff_features`, `out_results`, `pairs`, `row_f32` allocations) with thread-indexed arrays. Find and replace from `// Row buffer:` through the `row_f32` allocation:

```c
    // Thread-local buffers: one set per thread, indexed by tid
    float  *all_batch = (float*)  malloc((size_t)num_threads * BATCH_SIZE * num_feat * sizeof(float));
    double *all_out   = (double*) malloc((size_t)num_threads * BATCH_SIZE * sizeof(double));
    int    *all_pairs = (int*)    malloc((size_t)num_threads * BATCH_SIZE * sizeof(int));
    if (!all_batch || !all_out || !all_pairs) {
        fprintf(stderr, "Out of memory allocating thread batch buffers\n");
        return 1;
    }

    uint16_t *all_row_int = NULL;
    double   *all_row_flt = NULL;
    float    *all_row_f32 = NULL;
    if (float_output) {
        all_row_flt = (double*)calloc((size_t)num_threads * num_structures, sizeof(double));
        if (!all_row_flt) { fprintf(stderr, "Out of memory\n"); return 1; }
    } else {
        all_row_int = (uint16_t*)malloc((size_t)num_threads * num_structures * sizeof(uint16_t));
        if (!all_row_int) { fprintf(stderr, "Out of memory\n"); return 1; }
    }
    if (binary_output && float_output) {
        all_row_f32 = (float*)malloc((size_t)num_threads * num_structures * sizeof(float));
        if (!all_row_f32) { fprintf(stderr, "Out of memory\n"); return 1; }
    }
```

- [ ] **Step 3: Rewrite the main pairwise loop with OpenMP**

Replace the entire `for (int i = 0; ...)` loop (from `for (int i = 0; i < num_structures; ++i)` through the progress reporting closing brace) with:

```c
    // Per-thread LightGBM params (no shared state)
    char lgbm_params_single[64];
    snprintf(lgbm_params_single, sizeof(lgbm_params_single), "num_threads=1");

    int completed_rows = 0;

    // KNN per-thread buffers (allocated outside parallel region)
    int32_t  *all_knn_idx = NULL;
    uint16_t *all_knn_dst = NULL;
    if (topk > 0) {
        all_knn_idx = (int32_t*)malloc((size_t)num_threads * topk * sizeof(int32_t));
        all_knn_dst = (uint16_t*)malloc((size_t)num_threads * topk * sizeof(uint16_t));
        if (!all_knn_idx || !all_knn_dst) {
            fprintf(stderr, "Out of memory allocating KNN thread buffers\n");
            return 1;
        }
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        BoosterHandle my_booster = boosters[tid];
        float  *my_batch = &all_batch[(size_t)tid * BATCH_SIZE * num_feat];
        double *my_out   = &all_out[(size_t)tid * BATCH_SIZE];
        int    *my_pairs = &all_pairs[(size_t)tid * BATCH_SIZE];
        uint16_t *my_row_int = all_row_int ? &all_row_int[(size_t)tid * num_structures] : NULL;
        double   *my_row_flt = all_row_flt ? &all_row_flt[(size_t)tid * num_structures] : NULL;
        float    *my_row_f32 = all_row_f32 ? &all_row_f32[(size_t)tid * num_structures] : NULL;
        int32_t  *my_knn_idx = all_knn_idx ? &all_knn_idx[(size_t)tid * topk] : NULL;
        uint16_t *my_knn_dst = all_knn_dst ? &all_knn_dst[(size_t)tid * topk] : NULL;

        #pragma omp for schedule(dynamic, 1) ordered
        for (int i = 0; i < num_structures; ++i) {
            /* --- Compute row i (thread-local, no shared writes) --- */
            if (float_output)
                memset(my_row_flt, 0, (size_t)num_structures * sizeof(double));
            else
                memset(my_row_int, 0, (size_t)num_structures * sizeof(uint16_t));

            int batch_count = 0;
            const int len_i = lengths[i];

            for (int j = i + 1; j < num_structures; ++j) {
                const int len_j = lengths[j];

                if (max_len_diff >= 0 && abs(len_i - len_j) > max_len_diff) {
                    if (float_output) my_row_flt[j] = 301.0;
                    else              my_row_int[j] = 301;
                    continue;
                }

                if (subsample > 1) {
                    if (((i + j) % subsample) != 0) {
                        if (float_output) my_row_flt[j] = -1.0;
                        else              my_row_int[j] = (uint16_t)MISS_UINT16;
                        continue;
                    }
                }

                const int offset = batch_count * num_feat;
                const double *fi = &features[i * NUM_FEATURES_BASE];
                const double *fj = &features[j * NUM_FEATURES_BASE];

                for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
                    my_batch[offset + k] = (float)fabs(fi[k] - fj[k]);
                }
                if (rich_features) {
                    for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
                        my_batch[offset + NUM_FEATURES_BASE     + k] = (float)(fi[k] + fj[k]);
                        my_batch[offset + NUM_FEATURES_BASE * 2 + k] = (float)(fi[k] < fj[k] ? fi[k] : fj[k]);
                        my_batch[offset + NUM_FEATURES_BASE * 3 + k] = (float)(fi[k] > fj[k] ? fi[k] : fj[k]);
                    }
                }
                my_pairs[batch_count] = j;
                batch_count++;

                if (batch_count == BATCH_SIZE || j == num_structures - 1) {
                    if (batch_count > 0) {
                        int64_t out_len = 0;
                        if (LGBM_BoosterPredictForMat(
                                my_booster,
                                (const void*)my_batch,
                                C_API_DTYPE_FLOAT32,
                                batch_count,
                                num_feat,
                                1,
                                C_API_PREDICT_NORMAL,
                                -1, 0,
                                lgbm_params_single,
                                &out_len,
                                my_out) != 0) {
                            fprintf(stderr, "LightGBM prediction failed on thread %d\n", tid);
                        }

                        for (int b = 0; b < batch_count; ++b) {
                            int col = my_pairs[b];
                            double val = my_out[b];
                            if (val < 0) val = 0;
                            if (float_output) {
                                my_row_flt[col] = val;
                            } else {
                                int pred_ted = (int)llround(val);
                                if (pred_ted > 65535) pred_ted = 65535;
                                my_row_int[col] = (uint16_t)pred_ted;
                            }
                        }
                        batch_count = 0;
                    }
                }
            }

            /* --- Output (ordered to preserve row sequence) --- */
            #pragma omp ordered
            {
                if (topk > 0) {
                    for (int k = 0; k < topk; k++) {
                        my_knn_idx[k] = -1;
                        my_knn_dst[k] = UINT16_MAX;
                    }
                    int knn_count = 0;
                    uint16_t knn_worst = 0;
                    int knn_worst_pos = 0;
                    for (int j = i + 1; j < num_structures; ++j) {
                        uint16_t d = my_row_int[j];
                        if (d == 0 || d == MISS_UINT16 || d > (uint16_t)tau) continue;
                        if (knn_count < topk) {
                            my_knn_idx[knn_count] = (int32_t)j;
                            my_knn_dst[knn_count] = d;
                            if (d > knn_worst) { knn_worst = d; knn_worst_pos = knn_count; }
                            knn_count++;
                        } else if (d < knn_worst) {
                            my_knn_idx[knn_worst_pos] = (int32_t)j;
                            my_knn_dst[knn_worst_pos] = d;
                            knn_worst = 0;
                            for (int k = 0; k < topk; k++) {
                                if (my_knn_dst[k] > knn_worst && my_knn_idx[k] >= 0) {
                                    knn_worst = my_knn_dst[k];
                                    knn_worst_pos = k;
                                }
                            }
                        }
                    }
                    fwrite(my_knn_idx, sizeof(int32_t),  (size_t)topk, knn_idx_fp);
                    fwrite(my_knn_dst, sizeof(uint16_t), (size_t)topk, knn_dst_fp);

                } else if (binary_output) {
                    if (upper_only) {
                        int count = num_structures - i - 1;
                        if (count > 0) {
                            if (float_output) {
                                for (int j = 0; j < count; j++)
                                    my_row_f32[j] = (float)my_row_flt[i + 1 + j];
                                fwrite(my_row_f32, sizeof(float), (size_t)count, stdout);
                            } else {
                                fwrite(&my_row_int[i + 1], sizeof(uint16_t), (size_t)count, stdout);
                            }
                        }
                    } else {
                        if (float_output) {
                            for (int j = 0; j < num_structures; j++)
                                my_row_f32[j] = (float)my_row_flt[j];
                            fwrite(my_row_f32, sizeof(float), (size_t)num_structures, stdout);
                        } else {
                            fwrite(my_row_int, sizeof(uint16_t), (size_t)num_structures, stdout);
                        }
                    }

                } else {
                    if (float_output) {
                        if (upper_only) {
                            for (int j = i + 1; j < num_structures; ++j) {
                                if (j + 1 < num_structures) printf("%.4f ", my_row_flt[j]);
                                else                        printf("%.4f", my_row_flt[j]);
                            }
                            printf("\n");
                        } else {
                            for (int j = 0; j < num_structures; ++j)
                                printf("%.4f ", my_row_flt[j]);
                            printf("\n");
                        }
                    } else {
                        if (upper_only) {
                            for (int j = i + 1; j < num_structures; ++j) {
                                if (j + 1 < num_structures) printf("%" PRIu16 " ", my_row_int[j]);
                                else                        printf("%" PRIu16, my_row_int[j]);
                            }
                            printf("\n");
                        } else {
                            for (int j = 0; j < num_structures; ++j)
                                printf("%" PRIu16 " ", my_row_int[j]);
                            printf("\n");
                        }
                    }
                }

                /* Progress (inside ordered = serialised, safe to print) */
                completed_rows++;
                int percentage = (completed_rows * 100) / (num_structures > 0 ? num_structures : 1);
                if (percentage != last_percentage) {
                    time_t now = time(NULL);
                    double elapsed = difftime(now, start_time);
                    double est_total = percentage ? elapsed / (percentage / 100.0) : 0.0;
                    double remaining = est_total - elapsed;
                    if (is_tty)
                        fprintf(stderr, "\rProgress: %d%%, Elapsed: %.0f s, Remaining: %.0f s",
                                percentage, elapsed, remaining);
                    else
                        fprintf(stderr, "Progress: %d%%, Elapsed: %.0f s, Remaining: %.0f s\n",
                                percentage, elapsed, remaining);
                    last_percentage = percentage;
                }
            } /* end omp ordered */
        } /* end omp for */
    } /* end omp parallel */

    fprintf(stderr, "\n");
```

- [ ] **Step 4: Update cleanup section**

Replace the old cleanup (from `// Close KNN output files` to end of main) with:

```c
    // Close KNN output files
    if (knn_idx_fp) fclose(knn_idx_fp);
    if (knn_dst_fp) fclose(knn_dst_fp);
    free(all_knn_idx);
    free(all_knn_dst);

    if (topk > 0) {
        fprintf(stderr, "[predTED] KNN complete: m=%d K=%d tau=%d\n",
                num_structures, topk, tau);
        printf("%d %d\n", num_structures, topk);
    }

    free(lengths);
    free(all_batch);
    free(all_out);
    free(all_pairs);
    free(all_row_int);
    free(all_row_flt);
    free(all_row_f32);

    for (int i = 0; i < num_structures; i++) free(structures[i]);
    free(structures);
    free(features);

    for (int t = 0; t < num_threads; t++)
        LGBM_BoosterFree(boosters[t]);
    free(boosters);

    return 0;
}
```

- [ ] **Step 5: Remove old single-thread variables that are no longer used**

Remove these declarations that were replaced by thread-local versions:
- `uint16_t *row_int`, `double *row_flt`, `float *row_f32`
- `float *batch_diff_features`, `double *out_results`, `int *pairs`
- `BoosterHandle booster` and `int total_iterations`
- `char lgbm_params[64]` (replaced by `lgbm_params_single`)
- `int32_t *knn_row_idx`, `uint16_t *knn_row_dst` (replaced by `all_knn_idx`/`all_knn_dst`)

Also note: each booster uses `num_threads=1` in its params (`lgbm_params_single`) to prevent nested OpenMP parallelism within LightGBM.

- [ ] **Step 6: Build CLI and verify**

Run: `cd /Volumes/Masterarbeit/predTED && make cli`

Expected: Compiles without errors.

- [ ] **Step 7: Test with small input (correctness)**

Run: `cd /Volumes/Masterarbeit/predTED && echo -e "((..))\n(())..\n...((..))..." | bin/predted -t 1`

Capture output. Then run with multiple threads:

Run: `cd /Volumes/Masterarbeit/predTED && echo -e "((..))\n(())..\n...((..))..." | bin/predted -t 4`

Expected: Both produce identical output (same distance matrix).

- [ ] **Step 8: Test binary, float, and upper-only modes**

Run:
```bash
cd /Volumes/Masterarbeit/predTED
echo -e "((..))\n(())..\n...((..))..." | bin/predted -f -t 2
echo -e "((..))\n(())..\n...((..))..." | bin/predted -u -t 2
```

Expected: Same output as single-threaded runs.

- [ ] **Step 9: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add c_src/predTED.c
git commit -m "perf: parallelise pairwise loop — one LightGBM booster per thread"
```

---

### Task 5: SIMD Pairwise Feature Building (`predTED.c`)

**Files:**
- Modify: `c_src/predTED.c`
- Modify: `Makefile`

**Goal:** Vectorise the diff/sum/min/max computation over 36 floats using NEON (ARM) and SSE (x86).

- [ ] **Step 1: Add SIMD headers and helper at the top of `predTED.c`**

Add after the existing `#include` block (after `#include "predted_features.h"`):

```c
/* SIMD: build 144 pairwise features from two 36-element double arrays
   into a float32 output buffer.  Falls back to scalar when unavailable. */

#if defined(__ARM_NEON)
  #include <arm_neon.h>
  #define PREDTED_SIMD_NEON 1
#elif defined(__SSE__)
  #include <xmmintrin.h>
  #define PREDTED_SIMD_SSE 1
#endif

static inline void build_rich_features_simd(
    const double * restrict fi,
    const double * restrict fj,
    float * restrict out,
    int rich)
{
    /* Convert double->float and compute diff; optionally sum/min/max.
       NUM_FEATURES_BASE (36) is divisible by 4 — perfect for SSE/NEON. */

#if defined(PREDTED_SIMD_NEON)
    for (int k = 0; k < NUM_FEATURES_BASE; k += 4) {
        /* Load 4 doubles as 2×float64x2, convert each to float32x2, combine to float32x4 */
        float32x2_t a_lo = vcvt_f32_f64(vld1q_f64(&fi[k]));
        float32x2_t a_hi = vcvt_f32_f64(vld1q_f64(&fi[k + 2]));
        float32x4_t a = vcombine_f32(a_lo, a_hi);

        float32x2_t b_lo = vcvt_f32_f64(vld1q_f64(&fj[k]));
        float32x2_t b_hi = vcvt_f32_f64(vld1q_f64(&fj[k + 2]));
        float32x4_t b = vcombine_f32(b_lo, b_hi);

        vst1q_f32(&out[k], vabsq_f32(vsubq_f32(a, b)));  /* diff */
        if (rich) {
            vst1q_f32(&out[NUM_FEATURES_BASE + k],     vaddq_f32(a, b));  /* sum */
            vst1q_f32(&out[NUM_FEATURES_BASE * 2 + k], vminq_f32(a, b));  /* min */
            vst1q_f32(&out[NUM_FEATURES_BASE * 3 + k], vmaxq_f32(a, b));  /* max */
        }
    }

#elif defined(PREDTED_SIMD_SSE)
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    for (int k = 0; k < NUM_FEATURES_BASE; k += 4) {
        __m128 a = _mm_set_ps((float)fi[k+3], (float)fi[k+2], (float)fi[k+1], (float)fi[k]);
        __m128 b = _mm_set_ps((float)fj[k+3], (float)fj[k+2], (float)fj[k+1], (float)fj[k]);
        __m128 diff = _mm_andnot_ps(sign_mask, _mm_sub_ps(a, b));  /* fabsf */
        _mm_storeu_ps(&out[k], diff);
        if (rich) {
            _mm_storeu_ps(&out[NUM_FEATURES_BASE + k],     _mm_add_ps(a, b));
            _mm_storeu_ps(&out[NUM_FEATURES_BASE * 2 + k], _mm_min_ps(a, b));
            _mm_storeu_ps(&out[NUM_FEATURES_BASE * 3 + k], _mm_max_ps(a, b));
        }
    }

#else
    /* Scalar fallback */
    for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
        out[k] = (float)fabs(fi[k] - fj[k]);
    }
    if (rich) {
        for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
            out[NUM_FEATURES_BASE     + k] = (float)(fi[k] + fj[k]);
            out[NUM_FEATURES_BASE * 2 + k] = (float)(fi[k] < fj[k] ? fi[k] : fj[k]);
            out[NUM_FEATURES_BASE * 3 + k] = (float)(fi[k] > fj[k] ? fi[k] : fj[k]);
        }
    }
#endif
}
```

- [ ] **Step 2: Replace the scalar feature building in the pairwise inner loop**

In the parallel pairwise loop (Task 4), find the section that builds features:

```c
                for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
                    my_batch[offset + k] = (float)fabs(fi[k] - fj[k]);
                }
                if (rich_features) {
                    for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
                        my_batch[offset + NUM_FEATURES_BASE     + k] = (float)(fi[k] + fj[k]);
                        my_batch[offset + NUM_FEATURES_BASE * 2 + k] = (float)(fi[k] < fj[k] ? fi[k] : fj[k]);
                        my_batch[offset + NUM_FEATURES_BASE * 3 + k] = (float)(fi[k] > fj[k] ? fi[k] : fj[k]);
                    }
                }
```

Replace with:

```c
                build_rich_features_simd(fi, fj, &my_batch[offset], rich_features);
```

- [ ] **Step 3: Update Makefile with `-march=native`**

In `Makefile`, change the CFLAGS line:

```makefile
CFLAGS   ?= -O2 -Wall -Wno-deprecated-declarations -march=native
```

- [ ] **Step 4: Build and verify correctness**

Run: `cd /Volumes/Masterarbeit/predTED && make clean && make cli`

Then test:

```bash
cd /Volumes/Masterarbeit/predTED
echo -e "((..))\n(())..\n...((..))..." | bin/predted -t 1
echo -e "((..))\n(())..\n...((..))..." | bin/predted -t 4
```

Expected: Same output as before SIMD changes.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add c_src/predTED.c Makefile
git commit -m "perf: SIMD pairwise feature building (NEON + SSE with scalar fallback)"
```

---

### Task 6: Python API Streaming (`predted/__init__.py`)

**Files:**
- Modify: `predted/__init__.py`
- Modify: `tests/test_predict.py`

**Goal:** Replace O(N^2) memory `predict_matrix` with row-streaming + NumPy vectorisation.

- [ ] **Step 1: Write test for large-N streaming behaviour**

Add to `tests/test_predict.py` at the end:

```python
class TestPredictMatrixStreaming:
    """Verify the streaming implementation matches pair-by-pair computation."""

    def test_medium_matrix_consistent(self):
        """20 structures: matrix entries must match individual predict() calls."""
        structs = [
            "((..))", "(())..", "...((..))...", "((((...))))",
            "(((..(((....)))..((....))..)))",
            "((((((((((...((.(((...)).))).)))))))))",
            "................",
            "((((....))))", ".((..((.....))..)).",
            "(((....)))(((....)))",
            "..((..))..", "(((..((..))..)))",
            "((((((......)))))).........",
            "......(((((((.......)))))))",
            "((..((..((.....))..))..)).",
            "(((...)))(((...)))(((...)))",
            "..(((....)))...(((....))).",
            "((((....((((....))))...))))",
            "((....))((....))((....))",
            "((((((((....)))))))).",
        ]
        matrix = predted.predict_matrix(structs, dtype=int)
        for i in range(len(structs)):
            for j in range(i + 1, len(structs)):
                single = predted.predict(structs[i], structs[j])
                assert matrix[i, j] == single, (
                    f"Mismatch at [{i},{j}]: matrix={matrix[i,j]}, single={single}"
                )

    def test_dtype_float_streaming(self):
        structs = ["((..))", "(())..", "...((..))...", "((((...))))"]
        mat_f = predted.predict_matrix(structs, dtype=float)
        mat_i = predted.predict_matrix(structs, dtype=int)
        # Float rounding should match int
        np.testing.assert_array_equal(
            np.round(mat_f).astype(int),
            mat_i,
        )
```

- [ ] **Step 2: Run the new test to verify it passes with current code**

Run: `cd /Volumes/Masterarbeit/predTED && python -m pytest tests/test_predict.py::TestPredictMatrixStreaming -v`

Expected: PASS (the test is compatible with both old and new implementation).

- [ ] **Step 3: Rewrite `predict_matrix` with streaming + vectorised features**

Replace the entire `predict_matrix` function in `predted/__init__.py` (lines 80-134) with:

```python
def predict_matrix(
    structures: List[str],
    *,
    dtype: type = int,
) -> np.ndarray:
    """Compute the pairwise predicted-TED matrix for a list of structures.

    Parameters
    ----------
    structures
        List of dot-bracket notation strings.
    dtype
        ``int`` (default) for rounded integers, ``float`` for raw predictions.

    Returns
    -------
    np.ndarray
        Symmetric N*N distance matrix with zeros on the diagonal.
    """
    n = len(structures)
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)

    # Pre-compute per-structure features — O(N * 36)
    all_features = np.array([compute_features(s) for s in structures])

    use_float = (dtype is float or dtype is np.float64 or dtype is np.float32)
    matrix = np.zeros((n, n), dtype=np.float64 if use_float else int)

    if n < 2:
        return matrix

    booster = _get_booster()

    # Row-by-row streaming: O(N) memory per iteration instead of O(N^2) total
    for i in range(n - 1):
        fi = all_features[i]             # (36,)
        fj = all_features[i + 1:]       # (N-i-1, 36)

        # Vectorised pairwise feature building (NumPy broadcasts fi over rows of fj)
        diff = np.abs(fi - fj)
        sums = fi + fj
        mins = np.minimum(fi, fj)
        maxs = np.maximum(fi, fj)
        row_features = np.concatenate([diff, sums, mins, maxs], axis=1).astype(np.float32)

        preds = booster.predict(row_features)
        preds = np.clip(preds, 0, None)

        if not use_float:
            preds = np.round(preds).astype(int)

        matrix[i, i + 1:] = preds
        matrix[i + 1:, i] = preds

    return matrix
```

- [ ] **Step 4: Run full test suite**

Run: `cd /Volumes/Masterarbeit/predTED && python -m pytest tests/ -v`

Expected: All tests PASS, including the new streaming tests and the old `test_consistent_with_predict`.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add predted/__init__.py tests/test_predict.py
git commit -m "perf: streaming predict_matrix — O(N) RAM with NumPy vectorisation"
```

---

### Task 7: Batch-Size Tuning

**Files:**
- Create: `benchmarks/bench_batch_size.sh`
- Modify: `Makefile`

**Goal:** Find optimal BATCH_SIZE per platform and add a benchmark target.

- [ ] **Step 1: Create the batch-size sweep script**

```bash
#!/usr/bin/env bash
# benchmarks/bench_batch_size.sh — Sweep BATCH_SIZE for optimal throughput.
# Usage: bash benchmarks/bench_batch_size.sh [structures_file]

set -euo pipefail

STRUCTURES="${1:-data/structures.txt}"
N_LINES=$(wc -l < "$STRUCTURES")
echo "Structures: $N_LINES from $STRUCTURES"
echo ""

BEST_TIME=999999
BEST_BS=8192

for BS in 1024 2048 4096 8192 16384 32768 65536; do
    echo -n "BATCH_SIZE=$BS ... "

    # Build with this batch size
    make cli CFLAGS="-O2 -Wall -Wno-deprecated-declarations -march=native -DBATCH_SIZE=$BS" 2>/dev/null

    # Time it (3 runs, take best)
    BEST_RUN=999999
    for run in 1 2 3; do
        T=$( { time bin/predted -u < "$STRUCTURES" > /dev/null ; } 2>&1 | grep real | awk '{print $2}' | sed 's/[ms]/ /g' | awk '{print $1*60+$2}' )
        if (( $(echo "$T < $BEST_RUN" | bc -l) )); then
            BEST_RUN=$T
        fi
    done

    echo "${BEST_RUN}s"

    if (( $(echo "$BEST_RUN < $BEST_TIME" | bc -l) )); then
        BEST_TIME=$BEST_RUN
        BEST_BS=$BS
    fi
done

echo ""
echo "=== Best: BATCH_SIZE=$BEST_BS (${BEST_TIME}s) ==="
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /Volumes/Masterarbeit/predTED/benchmarks/bench_batch_size.sh`

- [ ] **Step 3: Add Makefile target**

Add before the `clean:` target in `Makefile`:

```makefile
bench-batch:
	bash benchmarks/bench_batch_size.sh data/structures.txt
```

- [ ] **Step 4: Make BATCH_SIZE overridable in `predTED.c`**

In `c_src/predTED.c`, change the BATCH_SIZE define:

```c
#ifndef BATCH_SIZE
#define BATCH_SIZE 8192  /* overridable via -DBATCH_SIZE=N at compile time */
#endif
```

- [ ] **Step 5: Build default CLI and verify**

Run: `cd /Volumes/Masterarbeit/predTED && make clean && make cli`

Expected: Compiles successfully.

- [ ] **Step 6: Commit**

```bash
cd /Volumes/Masterarbeit/predTED
git add benchmarks/bench_batch_size.sh Makefile c_src/predTED.c
git commit -m "perf: add batch-size tuning benchmark and make BATCH_SIZE overridable"
```

---

### Task 8: Final Benchmark + Verification

**Files:**
- No new files — verification only.

- [ ] **Step 1: Run feature regression tests**

Run: `cd /Volumes/Masterarbeit/predTED && python -m pytest tests/test_features_regression.py -v`

Expected: All PASS.

- [ ] **Step 2: Run full Python test suite**

Run: `cd /Volumes/Masterarbeit/predTED && python -m pytest tests/ -v`

Expected: All PASS.

- [ ] **Step 3: CLI correctness — compare 1-thread vs multi-thread**

```bash
cd /Volumes/Masterarbeit/predTED
head -100 data/structures.txt | bin/predted -t 1 -u > /tmp/predted_1t.txt
head -100 data/structures.txt | bin/predted -t 4 -u > /tmp/predted_4t.txt
diff /tmp/predted_1t.txt /tmp/predted_4t.txt
```

Expected: No diff (identical output).

- [ ] **Step 4: CLI correctness — binary mode**

```bash
cd /Volumes/Masterarbeit/predTED
head -100 data/structures.txt | bin/predted -t 1 -u -b > /tmp/predted_1t.bin
head -100 data/structures.txt | bin/predted -t 4 -u -b > /tmp/predted_4t.bin
cmp /tmp/predted_1t.bin /tmp/predted_4t.bin
```

Expected: Files are identical.

- [ ] **Step 5: Performance benchmark — CLI (N=1500)**

```bash
cd /Volumes/Masterarbeit/predTED
echo "--- 1 thread ---"
time bin/predted -t 1 -u < data/structures.txt > /dev/null
echo "--- 2 threads ---"
time bin/predted -t 2 -u < data/structures.txt > /dev/null
echo "--- 4 threads ---"
time bin/predted -t 4 -u < data/structures.txt > /dev/null
echo "--- 8 threads ---"
time bin/predted -t 8 -u < data/structures.txt > /dev/null
```

Record timings. Expected: near-linear speedup with thread count.

- [ ] **Step 6: Performance benchmark — Python (N=500)**

```bash
cd /Volumes/Masterarbeit/predTED
python -c "
import time, predted
structs = open('data/structures.txt').read().strip().splitlines()[:500]
start = time.perf_counter()
m = predted.predict_matrix(structs)
elapsed = time.perf_counter() - start
pairs = len(structs) * (len(structs) - 1) // 2
print(f'N={len(structs)}, {pairs:,} pairs, {elapsed:.2f}s ({pairs/elapsed:,.0f} pairs/s)')
"
```

Expected: Completes without OOM, significantly faster than before.

- [ ] **Step 7: Clean up temporary files**

```bash
rm -f /tmp/predted_*.txt /tmp/predted_*.bin
```
