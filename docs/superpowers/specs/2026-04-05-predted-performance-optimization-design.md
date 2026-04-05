# predTED Performance Optimization

**Date:** 2026-04-05
**Goal:** Maximise speed and minimise RAM usage for pairwise distance matrix computation, targeting 500K+ structures.

---

## Context

predTED predicts Tree Edit Distances (TED) between RNA secondary structures using 36 structural features + a LightGBM regression model. The C CLI (`bin/predted`) streams row-by-row to avoid O(N^2) memory. The Python API (`predted.predict_matrix`) currently builds all pairs in RAM.

**Current bottlenecks:**
- Pair table computed 7x per structure in `compute_selected_features()`
- `strlen()` called ~15x per structure
- N-gram computation uses string allocation + `strcmp` in inner loop
- Pairwise loop is single-threaded (LightGBM Booster not thread-safe)
- Python API allocates O(N^2) pairs in memory
- BATCH_SIZE=8192 never benchmarked

**Target environments:** macOS Apple Silicon (M4, 16 GB) + SLURM cluster (32+ cores, x86_64).

---

## 1. Feature Computation Refactoring (`predted_features.c`)

### 1.1 StructureContext

Introduce a context struct computed once per structure:

```c
typedef struct {
    const char *structure;
    int len;
    short *pair_table;
    int *depth_profile;
} StructureContext;
```

All sub-functions (`get_depth_features`, `find_stems`, `count_hairpin_loops`, `count_stacked_pairs`, `get_base_pair_distances`, `get_hairpin_loop_sizes`, `get_internal_loop_sizes`) receive `StructureContext*` instead of `const char*`.

### 1.2 Orchestrator pattern

`compute_selected_features()` becomes:

1. Compute `len = strlen(structure)` once
2. Compute `pair_table = create_pair_table(structure)` once
3. Compute `depth_profile = compute_depth_profile(structure)` once
4. Call all sub-functions with context pointer
5. Free pair_table and depth_profile

### 1.3 N-gram without strcmp

Replace string-based n-gram matching with direct character-to-index mapping:

```c
static inline int char_idx(char c) {
    // '(' -> 0, ')' -> 1, '.' -> 2
    return c == '(' ? 0 : (c == ')' ? 1 : 2);
}

// Bigram index: 0..8
int idx = char_idx(s[i]) * 3 + char_idx(s[i + 1]);
frequencies[idx]++;
```

Eliminates 9 malloc + 9 free + 9 strcmp calls per structure.

### 1.4 Backward compatibility

The public C function signature remains:

```c
void compute_selected_features(const char* structure, double* features);
```

This wrapper creates the context internally, calls the optimised path, and frees. No breaking changes for the Python C extension or CLI.

### Expected effect

~2-3x faster feature preprocessing. At 500K structures: ~5s instead of ~20-30s.

---

## 2. Multi-threaded Pairwise Loop (`predTED.c`)

### 2.1 One Booster per thread

```c
BoosterHandle *boosters = malloc(num_threads * sizeof(BoosterHandle));
for (int t = 0; t < num_threads; t++)
    LGBM_BoosterLoadModelFromString(model_str, &iters, &boosters[t]);
```

RAM overhead: ~2 MB per booster (model is small).

### 2.2 Row-parallel with OpenMP

```c
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    BoosterHandle my_booster = boosters[tid];
    // Thread-local buffers: batch_features, out_results, pairs, row_buf
    float  *my_batch = &all_batches[tid * BATCH_SIZE * num_feat];
    double *my_out   = &all_outs[tid * BATCH_SIZE];
    int    *my_pairs = &all_pairs[tid * BATCH_SIZE];
    uint16_t *my_row = &all_rows[tid * num_structures];

    #pragma omp for schedule(dynamic, 1) ordered
    for (int i = 0; i < num_structures; i++) {
        // Build pairwise features for row i
        // Predict in batches using my_booster
        // Fill my_row

        #pragma omp ordered
        {
            // Write my_row to stdout (text, binary, or KNN)
        }
    }
}
```

### 2.3 Thread-local buffer allocation

All buffers allocated once before the parallel region, indexed by thread ID:

| Buffer | Size per thread | 32 threads |
|--------|----------------|------------|
| batch_diff_features | BATCH_SIZE * 144 * 4 = 4.5 MB | 144 MB |
| out_results | BATCH_SIZE * 8 = 64 KB | 2 MB |
| pairs | BATCH_SIZE * 4 = 32 KB | 1 MB |
| row_int | N * 2 (N=500K = 1 MB) | 32 MB |
| **Total** | **~5.6 MB** | **~179 MB** |

### 2.4 Progress reporting

```c
int completed_rows = 0;
// Inside ordered block, after output:
#pragma omp atomic
completed_rows++;
// Print progress from ordered block (already serialised)
```

### 2.5 KNN mode compatibility

KNN mode already uses `upper_only` and writes per-row. The parallel structure is identical; each thread fills its own KNN row buffer and flushes in the ordered section.

### Expected effect

Near-linear speedup with thread count:

| Threads | Speedup |
|---------|---------|
| 1 | 1x (baseline) |
| 8 | ~8x |
| 32 | ~30x |

---

## 3. SIMD Feature Building

### 3.1 Platform-specific intrinsics

The pairwise feature loop (diff/sum/min/max over 36 floats) is vectorised:

**ARM NEON (M4 Mac):**
```c
#if defined(__ARM_NEON)
#include <arm_neon.h>
for (int k = 0; k < NUM_FEATURES_BASE; k += 4) {
    float32x4_t va = vld1q_f32(&fi[k]);
    float32x4_t vb = vld1q_f32(&fj[k]);
    vst1q_f32(&out[offset + k],            vabsq_f32(vsubq_f32(va, vb)));
    vst1q_f32(&out[offset + 36 + k],       vaddq_f32(va, vb));
    vst1q_f32(&out[offset + 72 + k],       vminq_f32(va, vb));
    vst1q_f32(&out[offset + 108 + k],      vmaxq_f32(va, vb));
}
#endif
```

**x86 SSE (Cluster):**
```c
#if defined(__SSE__)
#include <xmmintrin.h>
// Same pattern with _mm_load_ps, _mm_sub_ps, _mm_add_ps, _mm_min_ps, _mm_max_ps
// fabsf via _mm_andnot_ps with sign mask
#endif
```

**AVX (if available):** Processes 8 floats at once with `_mm256_*` intrinsics.

### 3.2 Compile-time selection

Preprocessor macros select the implementation at compile time. No runtime dispatch overhead. The Makefile sets appropriate flags:

- macOS: `-march=native` (enables NEON automatically on ARM)
- Cluster: `-march=native -mavx2` (if supported) or `-msse4.2`

### 3.3 Aligned allocation

Feature arrays allocated with 32-byte alignment for optimal SIMD loads:

```c
double *features = aligned_alloc(32, num_structures * NUM_FEATURES_BASE * sizeof(double));
```

Note: The pairwise loop converts double features to float32 for LightGBM. The SIMD path operates on float32 batch buffers which are already contiguous and aligned.

### 3.4 Scalar fallback

The existing code remains as the default path when no SIMD is detected.

### Expected effect

~3-4x faster feature building inner loop. This is a secondary optimisation since LightGBM prediction dominates batch time, but it reduces the non-LightGBM overhead per pair.

---

## 4. Python API Streaming

### 4.1 Row-wise predict_matrix

Replace the current O(N^2) memory approach with row-streaming:

```python
def predict_matrix(structures, *, dtype=int, chunk_size=8192):
    n = len(structures)
    all_features = np.array([compute_features(s) for s in structures])
    matrix = np.zeros((n, n), dtype=np.float64 if dtype is float else int)
    booster = _get_booster()

    for i in range(n):
        if i + 1 >= n:
            continue
        fi = all_features[i]          # (36,)
        fj = all_features[i + 1:]    # (N-i-1, 36)

        # Vectorised pairwise features (NumPy)
        diff = np.abs(fi - fj)
        sums = fi + fj
        mins = np.minimum(fi, fj)
        maxs = np.maximum(fi, fj)
        row_features = np.concatenate([diff, sums, mins, maxs], axis=1).astype(np.float32)

        preds = np.clip(booster.predict(row_features), 0, None)
        if dtype is not float and dtype is not np.float64 and dtype is not np.float32:
            preds = np.round(preds).astype(int)

        matrix[i, i + 1:] = preds
        matrix[i + 1:, i] = preds

    return matrix
```

### 4.2 Generator variant for very large N

```python
def predict_matrix_rows(structures, *, dtype=int):
    """Yield (row_index, row_array) tuples. O(N) RAM."""
    ...
    for i in range(n):
        yield i, row
```

Useful when even the output matrix doesn't fit in RAM.

### 4.3 RAM comparison

| N | Current | Streaming |
|---|---------|-----------|
| 1,000 | ~27 MB | ~0.6 MB |
| 10,000 | ~2.7 GB | ~5.5 MB |
| 50,000 | ~67 GB (OOM) | ~27 MB |
| 500,000 | impossible | ~270 MB |

Note: The output matrix itself is N^2 * dtype_size. For N=500K with int (8 bytes), that's 2 TB. The generator variant avoids even this.

### Expected effect

- Works at any N instead of OOM at N~5000
- ~10-50x faster than the current Python loop due to NumPy vectorisation

---

## 5. Batch-Size Tuning

### 5.1 Benchmark target

Add `make bench-batch` that runs predTED with BATCH_SIZE values from 1024 to 65536 (powers of 2) on a fixed test set (~5000 structures) and reports throughput (pairs/second).

### 5.2 Implementation

BATCH_SIZE becomes a compile-time constant (as now) but with a well-benchmarked default. If benchmarks show significant platform differences, provide two presets:

```c
#ifndef BATCH_SIZE
  #if defined(__APPLE__)
    #define BATCH_SIZE 16384
  #else
    #define BATCH_SIZE 32768
  #endif
#endif
```

### Expected effect

~10-30% throughput improvement depending on platform and cache hierarchy.

---

## Summary

### Combined speedup estimate (C CLI)

| Component | Factor |
|-----------|--------|
| Feature preprocessing | 2-3x |
| Multi-threading (8 cores) | ~8x |
| Multi-threading (32 cores) | ~30x |
| SIMD feature building | 3-4x on inner loop |
| Batch-size tuning | 1.1-1.3x |
| **Combined (8 cores)** | **~10-15x** |
| **Combined (32 cores)** | **~35-50x** |

### RAM profile (32 threads, N=500K)

| Component | Size |
|-----------|------|
| Feature array | 138 MB |
| Thread-local buffers (32x) | ~179 MB |
| Boosters (32x) | ~64 MB |
| Row buffers (32x) | ~32 MB |
| **Total** | **~413 MB** |

### What does NOT change

- LightGBM model, features, accuracy (no retraining)
- CLI interface (flags, input/output formats)
- KNN mode (benefits automatically from multi-threading)
- Python public API signatures (`predict`, `predict_float`, `predict_matrix`)
- Binary output format

### Risks

- Booster loading per thread: ~1s startup per booster (one-time cost, negligible for large N)
- `omp ordered` may slightly reduce parallelism for the last few rows (mitigated by dynamic scheduling)
- SIMD code increases maintenance surface (3 code paths) but is isolated behind preprocessor guards
