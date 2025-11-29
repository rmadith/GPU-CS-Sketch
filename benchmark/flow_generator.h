#ifndef FLOW_GENERATOR_H
#define FLOW_GENERATOR_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/**
 * Flow Distribution Types for Benchmarking
 * 
 * DIST_UNIFORM:       All flows have similar weight (1-100)
 * DIST_ZIPF:          Power-law distribution (realistic traffic)
 * DIST_HEAVY_ELEPHANT: 10% of flows get 90% of traffic
 * DIST_HEAVY_MICE:    90% small flows, 10% elephant flows
 * DIST_BIMODAL:       Two distinct clusters (1-10 and 10000-50000)
 */
typedef enum {
    DIST_UNIFORM = 0,
    DIST_ZIPF = 1,
    DIST_HEAVY_ELEPHANT = 2,
    DIST_HEAVY_MICE = 3,
    DIST_BIMODAL = 4,
    DIST_COUNT = 5
} DistributionType;

static const char* distribution_names[] = {
    "uniform",
    "zipf",
    "heavy_elephant",
    "heavy_mice",
    "bimodal"
};

// Zipf distribution parameters
#define ZIPF_ALPHA 1.2
#define ZIPF_MAX_RANK 10000

// Weight ranges
#define ELEPHANT_WEIGHT_MIN 10000
#define ELEPHANT_WEIGHT_MAX 100000
#define MICE_WEIGHT_MIN 1
#define MICE_WEIGHT_MAX 100

// Bimodal parameters
#define BIMODAL_LOW_MIN 1
#define BIMODAL_LOW_MAX 10
#define BIMODAL_HIGH_MIN 10000
#define BIMODAL_HIGH_MAX 50000

// Global state for Zipf distribution (precomputed)
static double zipf_cdf[ZIPF_MAX_RANK + 1];
static int zipf_initialized = 0;

/**
 * Initialize Zipf CDF for sampling
 */
static inline void init_zipf_cdf(double alpha, int max_rank) {
    if (zipf_initialized) return;
    
    double sum = 0.0;
    for (int i = 1; i <= max_rank; ++i) {
        sum += 1.0 / pow((double)i, alpha);
    }
    
    double cumsum = 0.0;
    zipf_cdf[0] = 0.0;
    for (int i = 1; i <= max_rank; ++i) {
        cumsum += (1.0 / pow((double)i, alpha)) / sum;
        zipf_cdf[i] = cumsum;
    }
    zipf_cdf[max_rank] = 1.0;
    
    zipf_initialized = 1;
}

/**
 * Sample from Zipf distribution using inverse CDF
 * Returns a rank from 1 to max_rank
 */
static inline int sample_zipf_rank(void) {
    if (!zipf_initialized) {
        init_zipf_cdf(ZIPF_ALPHA, ZIPF_MAX_RANK);
    }
    
    double u = (double)rand() / (double)RAND_MAX;
    
    // Binary search for the rank
    int lo = 1, hi = ZIPF_MAX_RANK;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (zipf_cdf[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/**
 * Generate weight based on Zipf rank
 * Lower ranks (more popular) get higher weights
 */
static inline uint32_t zipf_rank_to_weight(int rank) {
    // Weight inversely proportional to rank
    // Top flows get weights up to 100000, tail flows get weights ~1
    double weight = 100000.0 / pow((double)rank, ZIPF_ALPHA);
    if (weight < 1.0) weight = 1.0;
    if (weight > 100000.0) weight = 100000.0;
    return (uint32_t)weight;
}

/**
 * Generate uniform random weight in [1, 100]
 */
static inline uint32_t generate_uniform_weight(void) {
    return (uint32_t)(MICE_WEIGHT_MIN + (rand() % (MICE_WEIGHT_MAX - MICE_WEIGHT_MIN + 1)));
}

/**
 * Generate Zipf-distributed weight
 */
static inline uint32_t generate_zipf_weight(void) {
    int rank = sample_zipf_rank();
    return zipf_rank_to_weight(rank);
}

/**
 * Generate weight for heavy-elephant scenario
 * 10% of flows are elephants, 90% are mice
 * But elephants dominate total traffic (~90%)
 */
static inline uint32_t generate_heavy_elephant_weight(int flow_index, int total_flows) {
    // First 10% of flows are elephants
    int elephant_count = total_flows / 10;
    if (elephant_count < 1) elephant_count = 1;
    
    if (flow_index < elephant_count) {
        // Elephant flow: large weight
        return (uint32_t)(ELEPHANT_WEIGHT_MIN + 
            (rand() % (ELEPHANT_WEIGHT_MAX - ELEPHANT_WEIGHT_MIN + 1)));
    } else {
        // Mice flow: small weight
        return (uint32_t)(MICE_WEIGHT_MIN + 
            (rand() % (MICE_WEIGHT_MAX - MICE_WEIGHT_MIN + 1)));
    }
}

/**
 * Generate weight for heavy-mice scenario
 * 90% of flows are mice, 10% are elephants
 * Traffic is more balanced between groups
 */
static inline uint32_t generate_heavy_mice_weight(int flow_index, int total_flows) {
    // Last 10% of flows are elephants
    int mice_count = (total_flows * 9) / 10;
    
    if (flow_index < mice_count) {
        // Mice flow: small weight
        return (uint32_t)(MICE_WEIGHT_MIN + 
            (rand() % (MICE_WEIGHT_MAX - MICE_WEIGHT_MIN + 1)));
    } else {
        // Elephant flow: large weight
        return (uint32_t)(ELEPHANT_WEIGHT_MIN + 
            (rand() % (ELEPHANT_WEIGHT_MAX - ELEPHANT_WEIGHT_MIN + 1)));
    }
}

/**
 * Generate bimodal weight
 * 50% chance of being in low cluster, 50% in high cluster
 */
static inline uint32_t generate_bimodal_weight(void) {
    if (rand() % 2 == 0) {
        // Low cluster
        return (uint32_t)(BIMODAL_LOW_MIN + 
            (rand() % (BIMODAL_LOW_MAX - BIMODAL_LOW_MIN + 1)));
    } else {
        // High cluster
        return (uint32_t)(BIMODAL_HIGH_MIN + 
            (rand() % (BIMODAL_HIGH_MAX - BIMODAL_HIGH_MIN + 1)));
    }
}

/**
 * Main weight generation function - dispatches based on distribution type
 * 
 * @param dist_type     The distribution to use
 * @param flow_index    Current flow index (needed for elephant/mice)
 * @param total_flows   Total number of flows (needed for elephant/mice)
 * @return              Generated weight
 */
static inline uint32_t generate_weight_for_distribution(
    DistributionType dist_type,
    int flow_index,
    int total_flows
) {
    switch (dist_type) {
        case DIST_UNIFORM:
            return generate_uniform_weight();
        
        case DIST_ZIPF:
            return generate_zipf_weight();
        
        case DIST_HEAVY_ELEPHANT:
            return generate_heavy_elephant_weight(flow_index, total_flows);
        
        case DIST_HEAVY_MICE:
            return generate_heavy_mice_weight(flow_index, total_flows);
        
        case DIST_BIMODAL:
            return generate_bimodal_weight();
        
        default:
            return generate_uniform_weight();
    }
}

/**
 * Parse distribution type from string
 */
static inline DistributionType parse_distribution(const char* name) {
    for (int i = 0; i < DIST_COUNT; ++i) {
        // Simple string comparison
        const char* expected = distribution_names[i];
        const char* p = name;
        const char* q = expected;
        int match = 1;
        while (*p && *q) {
            if (*p != *q) {
                match = 0;
                break;
            }
            p++;
            q++;
        }
        if (match && *p == '\0' && *q == '\0') {
            return (DistributionType)i;
        }
    }
    return DIST_UNIFORM; // Default
}

/**
 * Get statistics about generated weights (for debugging/verification)
 */
static inline void compute_weight_stats(
    const uint32_t* weights,
    int count,
    double* out_mean,
    double* out_min,
    double* out_max,
    double* out_stddev
) {
    if (count <= 0) {
        *out_mean = *out_min = *out_max = *out_stddev = 0.0;
        return;
    }
    
    double sum = 0.0;
    double min_val = (double)weights[0];
    double max_val = (double)weights[0];
    
    for (int i = 0; i < count; ++i) {
        double w = (double)weights[i];
        sum += w;
        if (w < min_val) min_val = w;
        if (w > max_val) max_val = w;
    }
    
    double mean = sum / (double)count;
    
    double var_sum = 0.0;
    for (int i = 0; i < count; ++i) {
        double diff = (double)weights[i] - mean;
        var_sum += diff * diff;
    }
    
    *out_mean = mean;
    *out_min = min_val;
    *out_max = max_val;
    *out_stddev = sqrt(var_sum / (double)count);
}

#endif // FLOW_GENERATOR_H

