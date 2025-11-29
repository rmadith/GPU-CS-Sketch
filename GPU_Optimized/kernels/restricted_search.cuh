#ifndef RESTRICTED_SEARCH_CUH
#define RESTRICTED_SEARCH_CUH

#include <cuda_runtime.h>
#include "../server_gpu_opt.h"
#include "reduction.cuh"

// =============================================================================
// Restricted Search Space Optimization
// 
// Instead of scanning all N flows for argmax, we:
// 1. Find the top-K buckets with highest |residual|
// 2. Build a candidate set from flows mapping to those buckets
// 3. Run argmax only on the candidate set
//
// Complexity: O(M) + O(K * bucket_load) instead of O(N)
// =============================================================================

// =============================================================================
// Kernel 1: Find Heavy Buckets
// =============================================================================

/**
 * Fused kernel: Compute |r[i]| AND find block-level max in one pass.
 * Eliminates redundant memory access from separate abs_r computation.
 * 
 * @param r           Residual vector (length M)
 * @param m           Number of buckets
 * @param abs_r       Output: |r[i]| for each bucket (for later threshold selection)
 * @param block_max   Output: max |r| per block
 */
__global__ void compute_abs_and_max(const float* __restrict__ r,
                                    int m,
                                    float* abs_r,
                                    float* block_max) {
    __shared__ float s_val[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float val = -1.0f;
    if (tid < m) {
        val = fabsf(r[tid]);
        abs_r[tid] = val;  // Store for later use
    }
    
    // Warp-level max reduction
    val = warpReduceMax(val);
    
    if (lane == 0) {
        s_val[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_val[lane] : -1.0f;
        val = warpReduceMax(val);
        
        if (lane == 0) {
            block_max[blockIdx.x] = val;
        }
    }
}

/**
 * Partial sort to find top-K buckets using bitonic sort within blocks.
 * Each block processes a portion and outputs its local top-K.
 * 
 * For simplicity, we use a threshold-based approach instead:
 * Select buckets where |r[b]| > threshold, up to max K buckets.
 * 
 * @param abs_r           |r[i]| values
 * @param bucket_idx      Bucket indices
 * @param m               Number of buckets
 * @param threshold       Minimum |r| to be considered heavy
 * @param heavy_buckets   Output: indices of heavy buckets
 * @param heavy_count     Output: number of heavy buckets found (atomic)
 * @param max_heavy       Maximum number of heavy buckets to select
 */
__global__ void select_heavy_buckets_threshold(
    const float* __restrict__ abs_r,
    int m,
    float threshold,
    int* heavy_buckets,
    int* heavy_count,
    int max_heavy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m && abs_r[i] > threshold) {
        int pos = atomicAdd(heavy_count, 1);
        if (pos < max_heavy) {
            heavy_buckets[pos] = i;
        }
    }
}

/**
 * Block-level top-K selection (alternative to threshold-based approach).
 * Each block finds its local top-K elements and writes them to output.
 * Host-side merge required for global top-K.
 * 
 * NOTE: Currently unused. Kept as alternative to threshold-based selection.
 * 
 * @param abs_r       |r[i]| values for all buckets
 * @param m           Number of buckets
 * @param block_vals  Output: top values per block (block_count * K)
 * @param block_idx   Output: top indices per block
 * @param k           Number of top elements per block
 */
__global__ void find_top_k_per_block(const float* __restrict__ abs_r,
                                     int m,
                                     float* block_vals,
                                     int* block_idx,
                                     int k) {
    extern __shared__ char shmem_raw[];
    float* s_val = reinterpret_cast<float*>(shmem_raw);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load value
    float val = -1.0f;
    int idx = -1;
    if (tid < m) {
        val = abs_r[tid];
        idx = tid;
    }
    
    // Store to shared memory for sorting
    s_val[threadIdx.x] = val;
    s_idx[threadIdx.x] = idx;
    __syncthreads();
    
    // Simple selection: find top-k using repeated argmax
    // For small k, this is efficient enough
    int out_base = blockIdx.x * k;
    
    if (threadIdx.x == 0) {
        for (int ki = 0; ki < k && ki < (int)blockDim.x; ++ki) {
            // Find max in remaining elements
            float max_val = -1.0f;
            int max_pos = -1;
            for (int j = 0; j < (int)blockDim.x; ++j) {
                if (s_val[j] > max_val) {
                    max_val = s_val[j];
                    max_pos = j;
                }
            }
            
            if (max_pos >= 0) {
                block_vals[out_base + ki] = max_val;
                block_idx[out_base + ki] = s_idx[max_pos];
                s_val[max_pos] = -1.0f;  // Mark as used
            }
        }
    }
}

/**
 * Merge block-level top-K results to get global top-K (alternative approach).
 * Single block kernel for final merge.
 * 
 * NOTE: Currently unused. Kept as alternative to threshold-based selection.
 * 
 * @param block_vals    Top-K values from each block
 * @param block_idx     Top-K indices from each block
 * @param num_blocks    Number of blocks that contributed
 * @param k             K value (top-K per block and output)
 * @param global_vals   Output: global top-K values
 * @param global_idx    Output: global top-K indices
 */
__global__ void merge_top_k(const float* __restrict__ block_vals,
                            const int* __restrict__ block_idx,
                            int num_blocks,
                            int k,
                            float* global_vals,
                            int* global_idx) {
    extern __shared__ char shmem_raw[];
    float* s_val = reinterpret_cast<float*>(shmem_raw);
    int* s_idx = reinterpret_cast<int*>(s_val + 1024);
    
    int total = num_blocks * k;
    
    // Load all block results into shared memory
    for (int i = threadIdx.x; i < total && i < 1024; i += blockDim.x) {
        s_val[i] = block_vals[i];
        s_idx[i] = block_idx[i];
    }
    __syncthreads();
    
    // Thread 0 does final selection
    if (threadIdx.x == 0) {
        for (int ki = 0; ki < k; ++ki) {
            float max_val = -1.0f;
            int max_pos = -1;
            int count = (total < 1024) ? total : 1024;
            
            for (int j = 0; j < count; ++j) {
                if (s_val[j] > max_val) {
                    max_val = s_val[j];
                    max_pos = j;
                }
            }
            
            if (max_pos >= 0) {
                global_vals[ki] = max_val;
                global_idx[ki] = s_idx[max_pos];
                s_val[max_pos] = -1.0f;
            }
        }
    }
}

// =============================================================================
// Kernel 2: Build Candidate Set
// =============================================================================

/**
 * Clear the candidate bitmap.
 */
__global__ void clear_bitmap(unsigned int* bitmap, int n_words) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) {
        bitmap[i] = 0;
    }
}

/**
 * Mark candidates in bitmap from heavy buckets.
 * Uses the inverse index to find all flows in each heavy bucket.
 * 
 * @param heavy_buckets   Array of heavy bucket indices
 * @param num_heavy       Number of heavy buckets
 * @param inv_offset      CSR offset array for inverse index
 * @param inv_flows       CSR flow indices
 * @param bitmap          Output: bit i set if flow i is candidate
 */
__global__ void mark_candidates_from_buckets(
    const int* __restrict__ heavy_buckets,
    int num_heavy,
    const int* __restrict__ inv_offset,
    const int* __restrict__ inv_flows,
    unsigned int* bitmap) {
    
    // Each block handles one heavy bucket
    int bucket_id = blockIdx.x;
    if (bucket_id >= num_heavy) return;
    
    int bucket = heavy_buckets[bucket_id];
    int start = inv_offset[bucket];
    int end = inv_offset[bucket + 1];
    
    // Mark all flows in this bucket
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int flow = inv_flows[i];
        // Atomic set bit
        int word = flow >> 5;  // flow / 32
        int bit = flow & 31;   // flow % 32
        atomicOr(&bitmap[word], 1u << bit);
    }
}

/**
 * Compact bitmap to dense candidate array with bounds checking.
 * 
 * @param bitmap          Candidate bitmap
 * @param n_flows         Total number of flows
 * @param candidates      Output: dense array of candidate flow indices
 * @param candidate_count Output: number of candidates (atomic counter)
 * @param max_candidates  Maximum candidates to write (buffer size)
 */
__global__ void compact_candidates(const unsigned int* __restrict__ bitmap,
                                   int n_flows,
                                   int* candidates,
                                   int* candidate_count,
                                   int max_candidates) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread checks one flow
    if (tid < n_flows) {
        int word = tid >> 5;
        int bit = tid & 31;
        
        if (bitmap[word] & (1u << bit)) {
            int pos = atomicAdd(candidate_count, 1);
            // Bounds check to prevent buffer overflow
            if (pos < max_candidates) {
                candidates[pos] = tid;
            }
        }
    }
}

// =============================================================================
// Kernel 3: Restricted ArgMax
// =============================================================================

/**
 * Find argmax of |correlation| over candidate set only.
 * Much faster than scanning all N flows when candidate set is small.
 * 
 * @param corr            Correlation array (all flows)
 * @param candidates      Array of candidate flow indices
 * @param n_candidates    Number of candidates
 * @param block_max       Output: block-level max values
 * @param block_idx       Output: block-level argmax (original flow index)
 */
__global__ void restricted_argmax_kernel(const float* __restrict__ corr,
                                         const int* __restrict__ candidates,
                                         int n_candidates,
                                         float* block_max,
                                         int* block_idx) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    // Load correlation for this candidate
    float val = -1.0f;
    int idx = -1;
    
    if (tid < n_candidates) {
        int flow = candidates[tid];
        val = fabsf(corr[flow]);
        idx = flow;  // Store original flow index, not candidate index
    }
    
    // Warp-level reduction
    warpReduceArgMax(val, idx);
    
    if (lane == 0) {
        s_val[warp_id] = val;
        s_idx[warp_id] = idx;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_val[lane] : -1.0f;
        idx = (lane < num_warps) ? s_idx[lane] : -1;
        warpReduceArgMax(val, idx);
        
        if (lane == 0) {
            block_max[blockIdx.x] = val;
            block_idx[blockIdx.x] = idx;
        }
    }
}

/**
 * Final reduction for restricted argmax across blocks.
 * Handles arbitrary number of blocks by having each thread process multiple elements.
 */
__global__ void restricted_argmax_final(const float* __restrict__ block_vals,
                                        const int* __restrict__ block_idx,
                                        int n_blocks,
                                        float* out_val,
                                        int* out_idx) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    // Each thread finds max across multiple blocks if needed
    float val = -1.0f;
    int idx = -1;
    
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        float block_val = block_vals[i];
        if (block_val > val) {
            val = block_val;
            idx = block_idx[i];
        }
    }
    
    warpReduceArgMax(val, idx);
    
    if (lane == 0) {
        s_val[warp_id] = val;
        s_idx[warp_id] = idx;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_val[lane] : -1.0f;
        idx = (lane < num_warps) ? s_idx[lane] : -1;
        warpReduceArgMax(val, idx);
        
        if (lane == 0) {
            *out_val = val;
            *out_idx = idx;
        }
    }
}

// =============================================================================
// Host Helper: Compute Max for Heavy Bucket Threshold
// =============================================================================

/**
 * Block-level max reduction - first pass.
 * Finds max |r| value to use for threshold computation.
 * 
 * @param abs_r       |r[i]| values (length M)
 * @param m           Number of buckets
 * @param block_max   Output: max value per block
 */
__global__ void reduce_max_first(const float* __restrict__ abs_r,
                                 int m,
                                 float* block_max) {
    __shared__ float s_val[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float val = -1.0f;
    if (tid < m) {
        val = abs_r[tid];
    }
    
    // Warp-level max
    val = warpReduceMax(val);
    
    if (lane == 0) {
        s_val[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_val[lane] : -1.0f;
        val = warpReduceMax(val);
        
        if (lane == 0) {
            block_max[blockIdx.x] = val;
        }
    }
}

/**
 * Block-level max reduction - subsequent passes.
 * Reduces block-level max values to find global max.
 * 
 * @param vals        Input values to reduce
 * @param n           Number of input values
 * @param out         Output: max values per block
 */
__global__ void reduce_max_final(const float* __restrict__ vals,
                                 int n,
                                 float* out) {
    __shared__ float s_val[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float val = -1.0f;
    if (tid < n) {
        val = vals[tid];
    }
    
    val = warpReduceMax(val);
    
    if (lane == 0) {
        s_val[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_val[lane] : -1.0f;
        val = warpReduceMax(val);
        
        if (lane == 0) {
            out[blockIdx.x] = val;
        }
    }
}

#endif // RESTRICTED_SEARCH_CUH

