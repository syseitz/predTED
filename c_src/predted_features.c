/*
 * predted_features.c — Structural feature computation for RNA dot-bracket strings.
 *
 * This file is the SINGLE SOURCE OF TRUTH for all 36 features used by predTED.
 * It is shared between:
 *   - The Python package (via _features_module.c → predted._features_c)
 *   - The C CLI binary    (via #include "predted_features.h" in predTED.c)
 *
 * Dependencies: only stdlib, string.h, math.h — no LightGBM, no OpenMP.
 */

#include "predted_features.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

int count_char(const char* structure, char c) {
    int count = 0;
    for (int i = 0; structure[i]; i++) {
        if (structure[i] == c) count++;
    }
    return count;
}

short* create_pair_table(const char* structure) {
    int len = strlen(structure);
    short* pt = (short*)calloc(len + 1, sizeof(short));
    int* stack = (int*)malloc(len * sizeof(int));
    int top = -1;
    for (int i = 0; i < len; i++) {
        if (structure[i] == '(') {
            stack[++top] = i + 1;
        } else if (structure[i] == ')' && top >= 0) {
            int j = stack[top--];
            pt[j] = i + 1;
            pt[i + 1] = j;
        }
    }
    free(stack);
    return pt;
}

int* compute_depth_profile(const char* structure) {
    int len = strlen(structure);
    int* profile = (int*)malloc(len * sizeof(int));
    int depth = 0;
    for (int i = 0; i < len; i++) {
        if (structure[i] == '(') {
            depth++;
        } else if (structure[i] == ')') {
            depth--;
        }
        profile[i] = depth;
    }
    return profile;
}

void get_depth_features(const char* structure, double* mean_depth, double* var_depth, int* peaks, double* mean_depth_paired, double* var_depth_paired, double* mean_depth_unpaired, double* var_depth_unpaired) {
    int len = strlen(structure);
    short* pt = create_pair_table(structure);
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
    free(pt);
}

int* find_stems(const char* structure) {
    short* pt = create_pair_table(structure);
    int len = strlen(structure);
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
    free(pt);
    free(visited);
    return stems;
}

void get_stem_features(const char* structure, int* num_stems, double* max_stem_length, double* avg_stem_length, double* var_stem_length) {
    int* stems = find_stems(structure);
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

void get_loop_features(const char* structure, double* mean_loop, double* var_loop) {
    int len = strlen(structure);
    int* loop_sizes = (int*)malloc(len * sizeof(int));
    int count = 0;
    int current_loop = 0;
    for (int i = 0; i < len; i++) {
        if (structure[i] == '.') {
            current_loop++;
        } else {
            if (current_loop > 0) {
                loop_sizes[count++] = current_loop;
                current_loop = 0;
            }
        }
    }
    if (current_loop > 0) {
        loop_sizes[count++] = current_loop;
    }
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < count; i++) {
        sum += loop_sizes[i];
        sum_sq += loop_sizes[i] * loop_sizes[i];
    }
    *mean_loop = count > 0 ? sum / count : 0;
    *var_loop = count > 1 ? (sum_sq / count - (*mean_loop) * (*mean_loop)) : 0;
    free(loop_sizes);
}

double* get_ngram_features(const char* structure, int n) {
    int len = strlen(structure);
    int num_ngrams = n == 2 ? 9 : 27;
    double* frequencies = (double*)calloc(num_ngrams, sizeof(double));
    if (len < n) return frequencies;
    char* ngram = (char*)malloc((n + 1) * sizeof(char));
    int total = len - n + 1;
    char symbols[] = "().";
    char** possible_ngrams = (char**)malloc(num_ngrams * sizeof(char*));
    int idx = 0;
    if (n == 2) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                possible_ngrams[idx] = (char*)malloc(3 * sizeof(char));
                sprintf(possible_ngrams[idx], "%c%c", symbols[i], symbols[j]);
                idx++;
            }
        }
    } else {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    possible_ngrams[idx] = (char*)malloc(4 * sizeof(char));
                    sprintf(possible_ngrams[idx], "%c%c%c", symbols[i], symbols[j], symbols[k]);
                    idx++;
                }
            }
        }
    }
    for (int i = 0; i <= len - n; i++) {
        strncpy(ngram, structure + i, n);
        ngram[n] = '\0';
        for (int j = 0; j < num_ngrams; j++) {
            if (strcmp(ngram, possible_ngrams[j]) == 0) {
                frequencies[j]++;
                break;
            }
        }
    }
    for (int j = 0; j < num_ngrams; j++) {
        frequencies[j] /= total;
        free(possible_ngrams[j]);
    }
    free(possible_ngrams);
    free(ngram);
    return frequencies;
}

int count_hairpin_loops(const char* structure) {
    short* pt = create_pair_table(structure);
    int hairpin_loops = 0;
    int len = strlen(structure);
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int j = pt[i];
            int unpaired = 1;
            for (int k = i + 1; k < j; k++) {
                if (pt[k] != 0) {
                    unpaired = 0;
                    break;
                }
            }
            if (unpaired) {
                hairpin_loops++;
            }
        }
    }
    free(pt);
    return hairpin_loops;
}

int count_stacked_pairs(const char* structure) {
    short* pt = create_pair_table(structure);
    int len = strlen(structure);
    int stacked = 0;
    for (int i = 1; i < len; i++) {
        if (pt[i] > i && pt[i + 1] == pt[i] - 1) {
            stacked++;
        }
    }
    free(pt);
    return stacked;
}

void get_base_pair_distances(const char* structure, double* avg_bp_dist, int* max_bp_dist) {
    short* pt = create_pair_table(structure);
    int len = strlen(structure);
    double sum = 0;
    int count = 0;
    int max_dist = 0;
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
    free(pt);
}

int num_paired_bases(const char* structure) {
    return 2 * count_char(structure, '(');
}

int num_unpaired_bases(const char* structure) {
    return count_char(structure, '.');
}

int* get_hairpin_loop_sizes(const char* structure) {
    short* pt = create_pair_table(structure);
    int len = strlen(structure);
    int* sizes = (int*)malloc((len + 1) * sizeof(int));
    int count = 0;
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int j = pt[i];
            int unpaired = 1;
            for (int k = i + 1; k < j; k++) {
                if (pt[k] != 0) {
                    unpaired = 0;
                    break;
                }
            }
            if (unpaired) {
                sizes[count++] = j - i - 1;
            }
        }
    }
    sizes[count] = -1;
    free(pt);
    return sizes;
}

int count_bulges(const char* structure) {
    int len = strlen(structure);
    int* stack = (int*)malloc(len * sizeof(int));
    int top = -1;
    int bulges = 0;
    for (int i = 0; i < len; i++) {
        if (structure[i] == '(') {
            stack[++top] = i;
        } else if (structure[i] == ')' && top >= 0) {
            top--;
        } else if (structure[i] == '.' && top >= 0) {
            if (i + 1 < len && structure[i + 1] == ')') {
                bulges++;
            }
        }
    }
    free(stack);
    return bulges;
}

int count_internal_loops(const char* structure) {
    int len = strlen(structure);
    int internal_loops = 0;
    int i = 0;
    while (i < len - 1) {
        if (structure[i] == ')' && structure[i + 1] == '.') {
            int j = i + 1;
            while (j < len && structure[j] == '.') {
                j++;
            }
            if (j < len && structure[j] == '(') {
                internal_loops++;
            }
            i = j;
        } else {
            i++;
        }
    }
    return internal_loops;
}

int max_loop_size(const char* structure) {
    int max_size = 0;
    int current_size = 0;
    int in_loop = 0;
    for (int i = 0; structure[i]; i++) {
        if (structure[i] == '.') {
            if (!in_loop) {
                in_loop = 1;
                current_size = 1;
            } else {
                current_size++;
            }
        } else {
            if (in_loop) {
                max_size = max_size > current_size ? max_size : current_size;
                in_loop = 0;
                current_size = 0;
            }
        }
    }
    return in_loop ? (max_size > current_size ? max_size : current_size) : max_size;
}

int count_multiloops(const char* structure) {
    int len = strlen(structure);
    int* stack = (int*)malloc(len * sizeof(int));
    int top = -1;
    int multiloops = 0;
    for (int i = 0; i < len; i++) {
        if (structure[i] == '(') {
            stack[++top] = i;
        } else if (structure[i] == ')' && top >= 0) {
            top--;
            if (top >= 0 && i + 1 < len && structure[i + 1] == '(') {
                multiloops++;
            }
        }
    }
    free(stack);
    return multiloops;
}

double graph_centrality(const char* structure) {
    int pairs = 0;
    for (int i = 0; structure[i]; i++) {
        if (structure[i] == '(') pairs++;
    }
    int len = strlen(structure);
    return len > 0 ? (double)pairs / len : 0.0;
}

int tree_depth(const char* structure) {
    int depth = 0;
    int max_depth = 0;
    for (int i = 0; structure[i]; i++) {
        if (structure[i] == '(') {
            depth++;
            max_depth = max_depth > depth ? max_depth : depth;
        } else if (structure[i] == ')') {
            depth--;
        }
    }
    return max_depth;
}

int* get_internal_loop_sizes(const char* structure) {
    int len = strlen(structure);
    int* sizes = (int*)malloc((len + 1) * sizeof(int));
    int count = 0;
    short* pt = create_pair_table(structure);
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
    free(pt);
    return sizes;
}

int* get_bulge_sizes(const char* structure) {
    int len = strlen(structure);
    int* sizes = (int*)malloc((len + 1) * sizeof(int));
    int count = 0;
    short* pt = create_pair_table(structure);
    for (int i = 1; i <= len; i++) {
        if (pt[i] > i) {
            int j = pt[i];
            for (int k = i + 1; k < j; k++) {
                if (structure[k - 1] == '.') {
                    int bulge_size = 0;
                    while (k < j && structure[k - 1] == '.') {
                        bulge_size++;
                        k++;
                    }
                    if (bulge_size > 0) {
                        sizes[count++] = bulge_size;
                    }
                }
            }
        }
    }
    sizes[count] = -1;
    free(pt);
    return sizes;
}

double compute_mean(int* sizes) {
    int count = 0;
    double sum = 0;
    while (sizes[count] != -1) {
        sum += sizes[count];
        count++;
    }
    return count > 0 ? sum / count : 0;
}

int compute_max(int* sizes) {
    int max_size = 0;
    int i = 0;
    while (sizes[i] != -1) {
        if (sizes[i] > max_size) max_size = sizes[i];
        i++;
    }
    return max_size;
}

void compute_selected_features(const char* structure, double* features) {
    features[0] = count_internal_loops(structure);
    double mean_depth, var_depth, mean_depth_paired, var_depth_paired, mean_depth_unpaired, var_depth_unpaired;
    int peaks;
    get_depth_features(structure, &mean_depth, &var_depth, &peaks, &mean_depth_paired, &var_depth_paired, &mean_depth_unpaired, &var_depth_unpaired);
    features[1] = var_depth_paired;
    features[2] = count_multiloops(structure);
    features[3] = max_loop_size(structure);
    features[4] = strlen(structure);
    double mean_loop, var_loop;
    get_loop_features(structure, &mean_loop, &var_loop);
    features[5] = mean_loop;
    features[6] = tree_depth(structure);
    features[7] = mean_depth_unpaired;
    features[8] = count_bulges(structure);
    features[9] = var_loop;
    features[10] = graph_centrality(structure);
    int num_stems;
    double max_stem_length, avg_stem_length, var_stem_length;
    get_stem_features(structure, &num_stems, &max_stem_length, &avg_stem_length, &var_stem_length);
    features[11] = var_stem_length;
    features[12] = max_stem_length;
    features[13] = avg_stem_length;
    features[14] = mean_depth_paired;
    features[15] = num_stems;
    features[16] = var_depth_unpaired;
    features[17] = var_depth;
    features[18] = mean_depth;
    features[19] = count_hairpin_loops(structure);
    features[20] = count_stacked_pairs(structure);
    double avg_bp_dist;
    int max_bp_dist;
    get_base_pair_distances(structure, &avg_bp_dist, &max_bp_dist);
    features[21] = avg_bp_dist;
    features[22] = num_paired_bases(structure);
    features[23] = num_unpaired_bases(structure);
    int* hairpin_sizes = get_hairpin_loop_sizes(structure);
    features[24] = compute_mean(hairpin_sizes);
    features[25] = compute_max(hairpin_sizes);
    free(hairpin_sizes);
    int* internal_loop_sizes = get_internal_loop_sizes(structure);
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
}
