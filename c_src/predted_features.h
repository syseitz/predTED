#ifndef PREDTED_FEATURES_H
#define PREDTED_FEATURES_H

#define NUM_FEATURES_BASE 36

/* Compute the 36 structural features for a dot-bracket string.
 * `features` must point to an array of at least NUM_FEATURES_BASE doubles. */
void compute_selected_features(const char* structure, double* features);

/* Individual feature functions (also usable standalone) */
int    count_char(const char* structure, char c);
short* create_pair_table(const char* structure);
int*   compute_depth_profile(const char* structure);
void   get_depth_features(const char* structure, double* mean_depth, double* var_depth, int* peaks, double* mean_depth_paired, double* var_depth_paired, double* mean_depth_unpaired, double* var_depth_unpaired);
int*   find_stems(const char* structure);
void   get_stem_features(const char* structure, int* num_stems, double* max_stem_length, double* avg_stem_length, double* var_stem_length);
void   get_loop_features(const char* structure, double* mean_loop, double* var_loop);
double* get_ngram_features(const char* structure, int n);
int    count_hairpin_loops(const char* structure);
int    count_stacked_pairs(const char* structure);
void   get_base_pair_distances(const char* structure, double* avg_bp_dist, int* max_bp_dist);
int    num_paired_bases(const char* structure);
int    num_unpaired_bases(const char* structure);
int*   get_hairpin_loop_sizes(const char* structure);
int    count_bulges(const char* structure);
int    count_internal_loops(const char* structure);
int    max_loop_size(const char* structure);
int    count_multiloops(const char* structure);
double graph_centrality(const char* structure);
int    tree_depth(const char* structure);
int*   get_internal_loop_sizes(const char* structure);
int*   get_bulge_sizes(const char* structure);
double compute_mean(int* sizes);
int    compute_max(int* sizes);

#endif
