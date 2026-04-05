#include <LightGBM/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <getopt.h>
#include <time.h>
#include "model.h"
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>

#include "predted_features.h"

/* SIMD: build pairwise features from two double[36] arrays into float32 output */

#if defined(__ARM_NEON)
  #include <arm_neon.h>
  #define PREDTED_SIMD_NEON 1
#elif defined(__SSE__)
  #include <xmmintrin.h>
  #define PREDTED_SIMD_SSE 1
#endif

static inline void build_rich_features_simd(
    const double * __restrict__ fi,
    const double * __restrict__ fj,
    float * __restrict__ out,
    int rich)
{
#if defined(PREDTED_SIMD_NEON)
    for (int k = 0; k < NUM_FEATURES_BASE; k += 4) {
        float32x2_t a_lo = vcvt_f32_f64(vld1q_f64(&fi[k]));
        float32x2_t a_hi = vcvt_f32_f64(vld1q_f64(&fi[k + 2]));
        float32x4_t a = vcombine_f32(a_lo, a_hi);

        float32x2_t b_lo = vcvt_f32_f64(vld1q_f64(&fj[k]));
        float32x2_t b_hi = vcvt_f32_f64(vld1q_f64(&fj[k + 2]));
        float32x4_t b = vcombine_f32(b_lo, b_hi);

        vst1q_f32(&out[k], vabsq_f32(vsubq_f32(a, b)));
        if (rich) {
            vst1q_f32(&out[NUM_FEATURES_BASE + k],     vaddq_f32(a, b));
            vst1q_f32(&out[NUM_FEATURES_BASE * 2 + k], vminq_f32(a, b));
            vst1q_f32(&out[NUM_FEATURES_BASE * 3 + k], vmaxq_f32(a, b));
        }
    }

#elif defined(PREDTED_SIMD_SSE)
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    for (int k = 0; k < NUM_FEATURES_BASE; k += 4) {
        __m128 a = _mm_set_ps((float)fi[k+3], (float)fi[k+2], (float)fi[k+1], (float)fi[k]);
        __m128 b = _mm_set_ps((float)fj[k+3], (float)fj[k+2], (float)fj[k+1], (float)fj[k]);
        __m128 diff = _mm_andnot_ps(sign_mask, _mm_sub_ps(a, b));
        _mm_storeu_ps(&out[k], diff);
        if (rich) {
            _mm_storeu_ps(&out[NUM_FEATURES_BASE + k],     _mm_add_ps(a, b));
            _mm_storeu_ps(&out[NUM_FEATURES_BASE * 2 + k], _mm_min_ps(a, b));
            _mm_storeu_ps(&out[NUM_FEATURES_BASE * 3 + k], _mm_max_ps(a, b));
        }
    }

#else
    for (int k = 0; k < NUM_FEATURES_BASE; ++k)
        out[k] = (float)fabs(fi[k] - fj[k]);
    if (rich) {
        for (int k = 0; k < NUM_FEATURES_BASE; ++k) {
            out[NUM_FEATURES_BASE     + k] = (float)(fi[k] + fj[k]);
            out[NUM_FEATURES_BASE * 2 + k] = (float)(fi[k] < fj[k] ? fi[k] : fj[k]);
            out[NUM_FEATURES_BASE * 3 + k] = (float)(fi[k] > fj[k] ? fi[k] : fj[k]);
        }
    }
#endif
}

#define MISS_UINT16 65535

#define NUM_FEATURES_RICH (NUM_FEATURES_BASE * 4)  /* diff + sum + min + max */
#define VERSION "0.1.0"
#define INITIAL_CAPACITY 100
#ifndef BATCH_SIZE
#define BATCH_SIZE 8192  /* overridable via -DBATCH_SIZE=N at compile time */
#endif

int predict_TED(const char* struct1, const char* struct2, BoosterHandle booster) {
    double features1[NUM_FEATURES_BASE];
    double features2[NUM_FEATURES_BASE];
    compute_selected_features(struct1, features1);
    compute_selected_features(struct2, features2);

    double diff_features[NUM_FEATURES_BASE];
    for (int i = 0; i < NUM_FEATURES_BASE; i++) {
        diff_features[i] = fabs(features1[i] - features2[i]);
    }

    double out_result;
    int64_t out_len;
    LGBM_BoosterPredictForMat(booster, diff_features, C_API_DTYPE_FLOAT64, 1, NUM_FEATURES_BASE, 1, C_API_PREDICT_NORMAL, -1, 0, "", &out_len, &out_result);

    double predicted_TED = out_result;
    if (predicted_TED < 0) predicted_TED = 0;
    return (int)round(predicted_TED);
}

int main(int argc, char* argv[]) {
    int opt;
    int is_tty = isatty(fileno(stderr));
    int num_threads = 0;            // decide below
    int threads_from_cli = 0;       // set if -t is used
    int upper_only = 0;
    int subsample = 1;           // NEU
    int max_len_diff = -1;       // NEU (negativ = aus)
    int rich_features = 1;       // v0.4: rich features on by default (144 feat)
    int float_output = 0;        // v0.2: output floats instead of uint16
    int binary_output = 0;       // v0.5: raw binary bytes instead of text
    int topk = 0;                // v0.5: KNN mode — keep K closest per row (0 = off)
    int tau = 300;               // v0.5: distance threshold for KNN
    char *knn_prefix = NULL;     // v0.5: file prefix for KNN output

    // Options
    struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"threads", required_argument, 0, 't'},
        {"upper-only", no_argument, 0, 'u'},
        {"subsample",  required_argument, 0, 's'},
        {"max-len-diff", required_argument, 0, 'L'},
        {"rich-features", no_argument, 0, 'r'},
        {"float", no_argument, 0, 'f'},
        {"binary", no_argument, 0, 'b'},
        {"topk", required_argument, 0, 'k'},
        {"tau", required_argument, 0, 'T'},
        {"knn-prefix", required_argument, 0, 'K'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "hvt:us:L:rfbk:T:K:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  --help, -h           Display this help message\n");
                printf("  --version, -v        Display version information\n");
                printf("  --threads, -t N      Set number of threads (default: env/auto)\n");
                printf("  --upper-only, -u     Print only upper triangle per row (j>i)\n");
                printf("  --rich-features, -r  Use rich features: diff+sum+min+max (144 feat)\n");
                printf("  --float, -f          Output float values instead of integers\n");
                printf("  --binary, -b         Output raw binary bytes instead of text\n");
                printf("  --topk K, -k K       KNN mode: keep only K closest neighbours per row\n");
                printf("  --tau T, -T T        Distance threshold for KNN (default: 300)\n");
                printf("  --knn-prefix P, -K P File prefix for KNN output (.idx + .dst memmaps)\n");
                return 0;
            case 'v':
                printf("Version: %s\n", VERSION);
                return 0;
            case 't':
                num_threads = atoi(optarg);
                if (num_threads <= 0) {
                    fprintf(stderr, "Invalid number of threads: %s\n", optarg);
                    return 1;
                }
                threads_from_cli = 1;
                break;
            case 'u':
                upper_only = 1;
                break;
            case 's':
                subsample = atoi(optarg);
                if (subsample < 1) subsample = 1;
                break;
            case 'L':
                max_len_diff = atoi(optarg);
                if (max_len_diff < 0) max_len_diff = -1;
                break;
            case 'r':
                rich_features = 1;
                break;
            case 'f':
                float_output = 1;
                break;
            case 'b':
                binary_output = 1;
                break;
            case 'k':
                topk = atoi(optarg);
                if (topk < 1) { fprintf(stderr, "Invalid --topk: %s\n", optarg); return 1; }
                break;
            case 'T':
                tau = atoi(optarg);
                if (tau < 0) { fprintf(stderr, "Invalid --tau: %s\n", optarg); return 1; }
                break;
            case 'K':
                knn_prefix = optarg;
                break;
            default:
                fprintf(stderr, "Invalid option. Use --help for help.\n");
                return 1;
        }
    }

    // Validate KNN options
    if (topk > 0 && !knn_prefix) {
        fprintf(stderr, "--topk requires --knn-prefix PATH\n");
        return 1;
    }
    if (knn_prefix && topk == 0) {
        fprintf(stderr, "--knn-prefix requires --topk K\n");
        return 1;
    }
    if (topk > 0) {
        upper_only = 1;   // KNN always uses upper-only mode
        float_output = 0; // KNN memmaps are uint16 — force integer mode
    }

    // If not given via -t, derive from SLURM_CPUS_PER_TASK, OMP_NUM_THREADS, else hw
    if (!threads_from_cli) {
        const char* s = getenv("SLURM_CPUS_PER_TASK");
        if (s && atoi(s) > 0) num_threads = atoi(s);
        else {
            s = getenv("OMP_NUM_THREADS");
            if (s && atoi(s) > 0) num_threads = atoi(s);
            else num_threads = omp_get_max_threads();
        }
    }

    // Stabilize OpenMP scheduling / binding (only if not already set)
    if (!getenv("OMP_PROC_BIND"))  setenv("OMP_PROC_BIND",  "close", 0);
    if (!getenv("OMP_PLACES"))     setenv("OMP_PLACES",     "cores", 0);
    if (!getenv("OMP_WAIT_POLICY"))setenv("OMP_WAIT_POLICY","PASSIVE", 0);

    // Avoid nested/dynamic oversubscription
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    // Larger buffers: speed up stdout matrix + line-buffered stderr progress
    setvbuf(stdout, NULL, _IOFBF, 1<<20);
    setvbuf(stderr, NULL, _IOLBF, 0);

    // Log node + threads
    char host[256]; host[255]='\0'; gethostname(host, sizeof(host)-1);
    fprintf(stderr, "[predTED] host=%s threads=%d\n", host, num_threads);


    // Read structures from stdin
    char** structures = (char**)malloc(INITIAL_CAPACITY * sizeof(char*));
    int capacity = INITIAL_CAPACITY;
    int num_structures = 0;
    char line[1024];

    while (fgets(line, sizeof(line), stdin) != NULL) {
        line[strcspn(line, "\n")] = 0; // Remove newline
        if (num_structures >= capacity) {
            capacity *= 2;
            structures = (char**)realloc(structures, capacity * sizeof(char*));
        }
        structures[num_structures] = (char*)malloc(strlen(line) + 1);
        strcpy(structures[num_structures], line);
        num_structures++;
    }
    structures = (char**)realloc(structures, num_structures * sizeof(char*));

    // Check if structures are provided
    if (num_structures == 0) {
        fprintf(stderr, "No structures provided.\n");
        free(structures);
        return 1;
    }

    // Precompute structure lengths (used for cheap pair prefiltering)
    int *lengths = (int*)malloc((size_t)num_structures * sizeof(int));
    if (!lengths) {
        fprintf(stderr, "Out of memory allocating lengths\n");
        for (int i = 0; i < num_structures; i++) free(structures[i]);
        free(structures);
        return 1;
    }
    for (int i = 0; i < num_structures; ++i) {
        lengths[i] = (int)strlen(structures[i]);
    }


    // Determine number of pairwise features
    const int num_feat = rich_features ? NUM_FEATURES_RICH : NUM_FEATURES_BASE;

    fprintf(stderr, "[predTED] features_per_pair=%d (%s) float_output=%d\n",
            num_feat, rich_features ? "rich" : "base", float_output);

    // Compute per-structure features in parallel
    double* features = (double*)malloc(num_structures * NUM_FEATURES_BASE * sizeof(double));
    if (!features) {
        fprintf(stderr, "Out of memory allocating features\n");
        for (int i = 0; i < num_structures; i++) free(structures[i]);
        free(structures);
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_structures; i++) {
        compute_selected_features(structures[i], &features[i * NUM_FEATURES_BASE]);
    }

    // Load LightGBM model — one booster per thread for thread safety
    char *model_str = (char*)malloc((size_t)model_txt_len + 1);
    if (!model_str) { fprintf(stderr, "Out of memory allocating model string\n"); return 1; }
    memcpy(model_str, model_txt, (size_t)model_txt_len);
    model_str[model_txt_len] = '\0';

    BoosterHandle *boosters = (BoosterHandle*)malloc((size_t)num_threads * sizeof(BoosterHandle));
    if (!boosters) { fprintf(stderr, "Out of memory allocating boosters\n"); free(model_str); return 1; }
    for (int t = 0; t < num_threads; t++) {
        int total_iterations;
        if (LGBM_BoosterLoadModelFromString(model_str, &total_iterations, &boosters[t]) != 0) {
            fprintf(stderr, "Error loading model for thread %d\n", t);
            free(model_str); return 1;
        }
    }
    free(model_str);
    fprintf(stderr, "[predTED] loaded %d booster instances\n", num_threads);


    // Chunk buffer: ROW_CHUNK rows computed in parallel, then flushed sequentially.
    // This avoids omp-ordered (which serialises the loop) while keeping RAM bounded.
    #ifndef ROW_CHUNK
    #define ROW_CHUNK 256
    #endif
    const int chunk_size = (ROW_CHUNK < num_structures) ? ROW_CHUNK : num_structures;

    uint16_t *chunk_int = NULL;
    double   *chunk_flt = NULL;
    float    *out_f32   = NULL;

    if (float_output) {
        chunk_flt = (double*)calloc((size_t)chunk_size * num_structures, sizeof(double));
        if (!chunk_flt) { fprintf(stderr, "Out of memory allocating chunk buffer\n"); return 1; }
    } else {
        chunk_int = (uint16_t*)malloc((size_t)chunk_size * num_structures * sizeof(uint16_t));
        if (!chunk_int) { fprintf(stderr, "Out of memory allocating chunk buffer\n"); return 1; }
    }
    if (binary_output && float_output) {
        out_f32 = (float*)malloc((size_t)num_structures * sizeof(float));
        if (!out_f32) { fprintf(stderr, "Out of memory allocating float32 buffer\n"); return 1; }
    }

    fprintf(stderr, "[predTED] chunk_size=%d (%.1f MB row buffer)\n", chunk_size,
            (double)chunk_size * num_structures * (float_output ? 8 : 2) / (1024.0 * 1024.0));

    // Thread-local batch buffers (for LightGBM prediction only)
    float  *all_batch = (float*)malloc((size_t)num_threads * BATCH_SIZE * num_feat * sizeof(float));
    double *all_out   = (double*)malloc((size_t)num_threads * BATCH_SIZE * sizeof(double));
    int    *all_pairs = (int*)malloc((size_t)num_threads * BATCH_SIZE * sizeof(int));
    if (!all_batch || !all_out || !all_pairs) {
        fprintf(stderr, "Out of memory allocating batch buffers\n"); return 1;
    }

    // KNN mode: open output files
    FILE *knn_idx_fp = NULL, *knn_dst_fp = NULL;

    if (topk > 0) {
        char path_buf[PATH_MAX];
        snprintf(path_buf, sizeof(path_buf), "%s.knn.idx.i32.mmap", knn_prefix);
        knn_idx_fp = fopen(path_buf, "wb");
        if (!knn_idx_fp) {
            fprintf(stderr, "Cannot open %s for writing\n", path_buf);
            return 1;
        }
        snprintf(path_buf, sizeof(path_buf), "%s.knn.dst.u16.mmap", knn_prefix);
        knn_dst_fp = fopen(path_buf, "wb");
        if (!knn_dst_fp) {
            fprintf(stderr, "Cannot open %s for writing\n", path_buf);
            fclose(knn_idx_fp);
            return 1;
        }
        fprintf(stderr, "[predTED] KNN mode: K=%d tau=%d prefix=%s\n", topk, tau, knn_prefix);
    }

    char lgbm_params_single[64];
    snprintf(lgbm_params_single, sizeof(lgbm_params_single), "num_threads=1");

    time_t start_time = time(NULL);
    int completed_rows = 0;
    int last_percentage = -1;

    // KNN buffers (single set — used only in sequential output phase)
    int32_t  *knn_idx_buf = NULL;
    uint16_t *knn_dst_buf = NULL;
    if (topk > 0) {
        knn_idx_buf = (int32_t*)malloc((size_t)topk * sizeof(int32_t));
        knn_dst_buf = (uint16_t*)malloc((size_t)topk * sizeof(uint16_t));
        if (!knn_idx_buf || !knn_dst_buf) {
            fprintf(stderr, "Out of memory allocating KNN buffers\n");
            return 1;
        }
    }

    /* ================================================================
     * Chunk-based parallel pairwise computation.
     *
     * Instead of omp-ordered (which serialises the entire loop),
     * we process ROW_CHUNK rows at a time:
     *   Phase 1 — all threads compute chunk rows in parallel (no ordering)
     *   Phase 2 — one thread writes output sequentially (microseconds)
     *
     * The implicit barrier between phases ensures correctness.
     * RAM: chunk_size × N × element_size  (e.g. 256 × 500K × 2 = 256 MB)
     * ================================================================ */

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        BoosterHandle my_booster = boosters[tid];
        float  *my_batch = &all_batch[(size_t)tid * BATCH_SIZE * num_feat];
        double *my_out   = &all_out[(size_t)tid * BATCH_SIZE];
        int    *my_pairs = &all_pairs[(size_t)tid * BATCH_SIZE];

        for (int chunk_start = 0; chunk_start < num_structures; chunk_start += chunk_size) {
            const int chunk_end = (chunk_start + chunk_size < num_structures)
                                ? chunk_start + chunk_size : num_structures;
            const int chunk_len = chunk_end - chunk_start;

            /* --- Phase 1: compute all rows in this chunk (fully parallel) --- */
            #pragma omp for schedule(dynamic, 1)
            for (int ci = 0; ci < chunk_len; ++ci) {
                const int i = chunk_start + ci;
                uint16_t *row_int = chunk_int ? &chunk_int[(size_t)ci * num_structures] : NULL;
                double   *row_flt = chunk_flt ? &chunk_flt[(size_t)ci * num_structures] : NULL;

                if (float_output)
                    memset(row_flt, 0, (size_t)num_structures * sizeof(double));
                else
                    memset(row_int, 0, (size_t)num_structures * sizeof(uint16_t));

                int batch_count = 0;
                const int len_i = lengths[i];

                for (int j = i + 1; j < num_structures; ++j) {
                    const int len_j = lengths[j];

                    if (max_len_diff >= 0 && abs(len_i - len_j) > max_len_diff) {
                        if (float_output) row_flt[j] = 301.0;
                        else              row_int[j] = 301;
                        continue;
                    }

                    if (subsample > 1) {
                        if (((i + j) % subsample) != 0) {
                            if (float_output) row_flt[j] = -1.0;
                            else              row_int[j] = (uint16_t)MISS_UINT16;
                            continue;
                        }
                    }

                    const int offset = batch_count * num_feat;
                    const double *fi = &features[i * NUM_FEATURES_BASE];
                    const double *fj = &features[j * NUM_FEATURES_BASE];

                    build_rich_features_simd(fi, fj, &my_batch[offset], rich_features);
                    my_pairs[batch_count] = j;
                    batch_count++;

                    if (batch_count == BATCH_SIZE || j == num_structures - 1) {
                        if (batch_count > 0) {
                            int64_t out_len = 0;
                            if (LGBM_BoosterPredictForMat(
                                    my_booster,
                                    (const void*)my_batch,
                                    C_API_DTYPE_FLOAT32,
                                    batch_count, num_feat, 1,
                                    C_API_PREDICT_NORMAL, -1, 0,
                                    lgbm_params_single,
                                    &out_len, my_out) != 0) {
                                fprintf(stderr, "LightGBM prediction failed (thread %d)\n", tid);
                            }
                            for (int b = 0; b < batch_count; ++b) {
                                int col = my_pairs[b];
                                double val = my_out[b];
                                if (val < 0) val = 0;
                                if (float_output) {
                                    row_flt[col] = val;
                                } else {
                                    int pred_ted = (int)llround(val);
                                    if (pred_ted > 65535) pred_ted = 65535;
                                    row_int[col] = (uint16_t)pred_ted;
                                }
                            }
                            batch_count = 0;
                        }
                    }
                }
            }
            /* implicit barrier — all chunk rows computed */

            /* --- Phase 2: output chunk rows sequentially (single thread) --- */
            #pragma omp single
            {
                for (int ci = 0; ci < chunk_len; ++ci) {
                    const int i = chunk_start + ci;
                    uint16_t *row_int = chunk_int ? &chunk_int[(size_t)ci * num_structures] : NULL;
                    double   *row_flt = chunk_flt ? &chunk_flt[(size_t)ci * num_structures] : NULL;

                    if (topk > 0) {
                        for (int k = 0; k < topk; k++) {
                            knn_idx_buf[k] = -1;
                            knn_dst_buf[k] = UINT16_MAX;
                        }
                        int knn_count = 0;
                        uint16_t knn_worst = 0;
                        int knn_worst_pos = 0;

                        for (int j = i + 1; j < num_structures; ++j) {
                            uint16_t d = row_int[j];
                            if (d == 0 || d == MISS_UINT16 || d > (uint16_t)tau) continue;
                            if (knn_count < topk) {
                                knn_idx_buf[knn_count] = (int32_t)j;
                                knn_dst_buf[knn_count] = d;
                                if (d > knn_worst) { knn_worst = d; knn_worst_pos = knn_count; }
                                knn_count++;
                            } else if (d < knn_worst) {
                                knn_idx_buf[knn_worst_pos] = (int32_t)j;
                                knn_dst_buf[knn_worst_pos] = d;
                                knn_worst = 0;
                                for (int k = 0; k < topk; k++) {
                                    if (knn_dst_buf[k] > knn_worst && knn_idx_buf[k] >= 0) {
                                        knn_worst = knn_dst_buf[k];
                                        knn_worst_pos = k;
                                    }
                                }
                            }
                        }
                        fwrite(knn_idx_buf, sizeof(int32_t),  (size_t)topk, knn_idx_fp);
                        fwrite(knn_dst_buf, sizeof(uint16_t), (size_t)topk, knn_dst_fp);

                    } else if (binary_output) {
                        if (upper_only) {
                            int count = num_structures - i - 1;
                            if (count > 0) {
                                if (float_output) {
                                    for (int j = 0; j < count; j++)
                                        out_f32[j] = (float)row_flt[i + 1 + j];
                                    fwrite(out_f32, sizeof(float), (size_t)count, stdout);
                                } else {
                                    fwrite(&row_int[i + 1], sizeof(uint16_t), (size_t)count, stdout);
                                }
                            }
                        } else {
                            if (float_output) {
                                for (int j = 0; j < num_structures; j++)
                                    out_f32[j] = (float)row_flt[j];
                                fwrite(out_f32, sizeof(float), (size_t)num_structures, stdout);
                            } else {
                                fwrite(row_int, sizeof(uint16_t), (size_t)num_structures, stdout);
                            }
                        }

                    } else {
                        if (float_output) {
                            if (upper_only) {
                                if (i + 1 < num_structures) {
                                    for (int j = i + 1; j < num_structures; ++j) {
                                        if (j + 1 < num_structures) printf("%.4f ", row_flt[j]);
                                        else                        printf("%.4f", row_flt[j]);
                                    }
                                }
                                printf("\n");
                            } else {
                                for (int j = 0; j < num_structures; ++j)
                                    printf("%.4f ", row_flt[j]);
                                printf("\n");
                            }
                        } else {
                            if (upper_only) {
                                if (i + 1 < num_structures) {
                                    for (int j = i + 1; j < num_structures; ++j) {
                                        if (j + 1 < num_structures) printf("%" PRIu16 " ", row_int[j]);
                                        else                        printf("%" PRIu16, row_int[j]);
                                    }
                                }
                                printf("\n");
                            } else {
                                for (int j = 0; j < num_structures; ++j)
                                    printf("%" PRIu16 " ", row_int[j]);
                                printf("\n");
                            }
                        }
                    }

                    completed_rows++;
                    int percentage = (completed_rows * 100) / (num_structures > 0 ? num_structures : 1);
                    if (percentage != last_percentage) {
                        time_t now = time(NULL);
                        double elapsed = difftime(now, start_time);
                        double est_total = percentage ? elapsed / (percentage / 100.0) : 0.0;
                        double remaining = est_total - elapsed;
                        if (is_tty) {
                            fprintf(stderr, "\rProgress: %d%%, Elapsed: %.0f s, Remaining: %.0f s",
                                    percentage, elapsed, remaining);
                            fflush(stderr);
                        } else {
                            fprintf(stderr, "Progress: %d%%, Elapsed: %.0f s, Remaining: %.0f s\n",
                                    percentage, elapsed, remaining);
                        }
                        last_percentage = percentage;
                    }
                } /* end chunk output loop */
            } /* end omp single */
            /* implicit barrier — output done, safe to reuse chunk buffer */
        } /* end chunk loop */
    } /* end omp parallel */

    fprintf(stderr, "\n");

    // Close KNN output files
    if (knn_idx_fp) fclose(knn_idx_fp);
    if (knn_dst_fp) fclose(knn_dst_fp);
    free(knn_idx_buf);
    free(knn_dst_buf);

    if (topk > 0) {
        fprintf(stderr, "[predTED] KNN complete: m=%d K=%d tau=%d\n",
                num_structures, topk, tau);
        printf("%d %d\n", num_structures, topk);
    }

    free(lengths);
    free(all_batch);
    free(all_out);
    free(all_pairs);
    free(chunk_int);
    free(chunk_flt);
    free(out_f32);

    for (int i = 0; i < num_structures; i++) free(structures[i]);
    free(structures);
    free(features);
    for (int t = 0; t < num_threads; t++) LGBM_BoosterFree(boosters[t]);
    free(boosters);

    return 0;
}