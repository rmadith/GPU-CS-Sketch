/**
 * Multi-GPU Optimized OMP Implementation for Count-Min Sketch with NCCL
 * 
 * Strategy:
 * 1. Data parallelism: Each GPU handles a partition of flows
 * 2. NCCL for efficient collective communication (broadcast, allreduce)
 * 3. All-reduce for finding global best flow
 * 
 * Build: nvcc -O2 -std=c++14 -DUSE_CUDA -DUSE_NCCL -gencode arch=compute_75,code=sm_75 \
 *        -I/usr/local/nccl/include -L/usr/local/nccl/lib -lnccl \
 *        server_gpu_multi.cu -o multi-gpu-omp
 */


#include "server_gpu_multi.h"
#include "../GPU_Optimized/kernels/reduction.cuh"
#include "../GPU_Optimized/kernels/correlation.cuh"
#include "../GPU_Optimized/kernels/cholesky.cuh"
#include "../GPU_Optimized/kernels/restricted_search.cuh"

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <limits> 

// Import CPU structures
extern "C" {
#include "../CPU/server.h"
#include "../CPU/switch.h"
}

// =============================================================================
// Error Checking Macros
// =============================================================================

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                 \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(expr)                                                       \
    do {                                                                       \
        ncclResult_t _r = (expr);                                              \
        if (_r != ncclSuccess) {                                               \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,      \
                    ncclGetErrorString(_r));                                   \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

// =============================================================================
// Utility Functions
// =============================================================================

// Host helper: reinterpret float bits to uint32
static inline uint32_t float_as_uint32(float f) {
    union { float f; uint32_t u; } cv;
    cv.f = f;
    return cv.u;
}
static inline float uint32_as_float(uint32_t u) {
    union { uint32_t u; float f; } cv;
    cv.u = u;
    return cv.f;
}

/**
 * Map IEEE-754 float bits to lexicographically ordered uint32 such that
 * standard unsigned integer comparison corresponds to float comparison.
 *
 * Encoding:
 *  - If sign bit == 1 (negative), encode as bitwise NOT of raw bits.
 *  - If sign bit == 0 (positive or zero), raw_bits XOR 0x80000000u.
 *
 * This map orders floats from -inf .. +inf as increasing unsigned ints.
 */
static inline uint32_t float_to_ordered_uint32(float f) {
    uint32_t u = float_as_uint32(f);
    if (u & 0x80000000u) {
        return ~u;
    } else {
        return u ^ 0x80000000u;
    }
}

/** Inverse of the above mapping */
static inline float ordered_uint32_to_float(uint32_t ord) {
    uint32_t raw;
    if (ord & 0x80000000u) {
        raw = ord ^ 0x80000000u;
    } else {
        raw = ~ord;
    }
    return uint32_as_float(raw);
}

/**
 * Pack (float_value, global_index) into a uint64_t for a single ncclMax.
 * High 32 bits = ordered uint32 representation of float (so ncclMax => max float).
 * Low 32 bits  = global index (use UINT32_MAX for invalid/negative index).
 */
static inline uint64_t pack_val_idx(float value, int global_idx) {
    uint32_t ord = float_to_ordered_uint32(value);
    uint32_t gi  = (global_idx >= 0) ? (uint32_t)global_idx : UINT32_MAX;
    return ( (uint64_t)ord << 32 ) | (uint64_t)gi;
}

/** Unpack uint64_t -> (float, int) */
static inline void unpack_val_idx(uint64_t packed, float *out_val, int *out_idx) {
    uint32_t ord = (uint32_t)(packed >> 32);
    uint32_t gi  = (uint32_t)(packed & 0xFFFFFFFFu);
    *out_val = ordered_uint32_to_float(ord);
    *out_idx = (gi == UINT32_MAX) ? -1 : (int)gi;
}


static inline int div_up(int a, int b) { return (a + b - 1) / b; }

/**
 * Host-side argmax reduction using warp shuffle kernels.
 */
static bool device_argmax_warp(const float* d_vals, int n,
                               float* scratch_val, int* scratch_idx,
                               int block_size,
                               float* h_val, int* h_idx) {
    const float* cur_vals = d_vals;
    const int* cur_idx = nullptr;
    int cur_n = n;
    bool first = true;
    
    while (true) {
        int blocks = div_up(cur_n, block_size);
        
        if (first) {
            argmax_warp_first<<<blocks, block_size>>>(cur_vals, cur_n,
                                                       scratch_val, scratch_idx);
            first = false;
        } else {
            argmax_warp_reduce<<<blocks, block_size>>>(cur_vals, cur_idx, cur_n,
                                                        scratch_val, scratch_idx);
        }
        
        if (cudaGetLastError() != cudaSuccess) return false;
        
        if (blocks == 1) break;
        
        cur_vals = scratch_val;
        cur_idx = scratch_idx;
        cur_n = blocks;
    }
    
    if (cudaMemcpy(h_val, scratch_val, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (cudaMemcpy(h_idx, scratch_idx, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    
    return true;
}

/**
 * Compute L2 norm of residual using warp shuffle reductions.
 */
static bool device_l2_norm_warp(const float* d_r, int m,
                                float* scratch, int block_size,
                                float* h_out) {
    int blocks = div_up(m, block_size);
    
    l2_norm_partial<<<blocks, block_size>>>(d_r, m, scratch);
    if (cudaGetLastError() != cudaSuccess) return false;
    
    int cur_n = blocks;
    const float* cur_vals = scratch;
    
    while (cur_n > 1) {
        int next_blocks = div_up(cur_n, block_size);
        reduce_sum_final<<<next_blocks, block_size>>>(cur_vals, cur_n, scratch);
        if (cudaGetLastError() != cudaSuccess) return false;
        cur_vals = scratch;
        cur_n = next_blocks;
    }
    
    float sum = 0.0f;
    if (cudaMemcpy(&sum, cur_vals, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    
    *h_out = sqrtf(sum);
    return true;
}

#if GPU_OPT_USE_INCREMENTAL_CORR
/**
 * Build inverse index on host and upload to device.
 * Returns offsets and flows arrays in CSR format.
 */
static bool build_inverse_index_host(const int* h_idx, int n, int m,
                                     int** d_inv_offset, int** d_inv_flows) {
    // Count flows per counter
    std::vector<int> counts(m + 1, 0);
    for (int flow = 0; flow < n; ++flow) {
        for (int d = 0; d < GPU_OPT_D; ++d) {
            int counter = h_idx[flow * GPU_OPT_D + d];
            counts[counter + 1]++;
        }
    }
    
    // Prefix sum to get offsets
    for (int i = 1; i <= m; ++i) {
        counts[i] += counts[i - 1];
    }
    
    // Fill flows
    std::vector<int> flows_arr(n * GPU_OPT_D);
    std::vector<int> positions(m, 0);
    for (int flow = 0; flow < n; ++flow) {
        for (int d = 0; d < GPU_OPT_D; ++d) {
            int counter = h_idx[flow * GPU_OPT_D + d];
            int pos = counts[counter] + positions[counter];
            flows_arr[pos] = flow;
            positions[counter]++;
        }
    }
    
    // Upload to device
    if (cudaMalloc(d_inv_offset, (m + 1) * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(d_inv_flows, n * GPU_OPT_D * sizeof(int)) != cudaSuccess) {
        cudaFree(*d_inv_offset);
        return false;
    }
    
    if (cudaMemcpy(*d_inv_offset, counts.data(), (m + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_inv_offset);
        cudaFree(*d_inv_flows);
        return false;
    }
    if (cudaMemcpy(*d_inv_flows, flows_arr.data(), n * GPU_OPT_D * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_inv_offset);
        cudaFree(*d_inv_flows);
        return false;
    }
    
    return true;
}
#endif


// =============================================================================
// Main OMP Reconstruction Function
// =============================================================================

void server_reconstruct_omp_multi_gpu(int K_max, int flow_count, int device_id, int num_gpus, ncclComm_t comm) {
    if (flow_count <= 0) return;
    if (flow_count > N) flow_count = N;
    if (K_max > flow_count) K_max = flow_count;
    
    const int block_size = GPU_OPT_CORR_BLOCK_SIZE;
    const int reduce_block = GPU_OPT_REDUCE_BLOCK_SIZE;

    // partition flow set 
    int N_total = flow_count;
    // Simple block-partitioning
    int N_local = N_total / num_gpus;
    int N_start_idx = device_id * N_local;
    if (device_id == num_gpus - 1) {
        N_local += (N_total % num_gpus); // Last GPU takes the remainder
    }
    int N_end_idx = N_start_idx + N_local;

    // =========================================================================
    // Host Buffers
    // =========================================================================
    std::vector<float> h_y(GPU_OPT_M);
    std::vector<int> h_idx(N_local * GPU_OPT_D);
    std::vector<int> h_support(K_max, -1);
    std::vector<float> h_support_x(K_max, 0.0f);
    std::vector<float> h_r(GPU_OPT_M, 0.0f);
    
    // Flatten CMS sketch to measurement vector
    for (int row = 0; row < GPU_OPT_D; ++row) {
        for (int col = 0; col < GPU_OPT_W; ++col) {
            int pos = row * GPU_OPT_W + col;
            h_y[pos] = (float)cms_sketch[row][col];
        }
    }
    
    // Copy to CPU residual
    for (int i = 0; i < GPU_OPT_M; ++i) {
        r[i] = (double)h_y[i];
    }
    
    // Build flow index array
    for (int i = 0; i < N_local; ++i) {
        int global_i = N_start_idx + i;
        for (int j = 0; j < GPU_OPT_D; ++j) {
            h_idx[i * GPU_OPT_D + j] = j * GPU_OPT_W + (int)flows[global_i].idx[j];
        }
    }
    
    // Reset CPU outputs
    for (int i = 0; i < N; ++i) {
        selected[i] = 0;
        x[i] = 0.0;
    }
    
    // =========================================================================
    // Device Buffers
    // =========================================================================
    float* d_y = nullptr;
    float* d_r = nullptr;
    int* d_idx = nullptr;
    int* d_S = nullptr;
    float* d_L = nullptr;
    float* d_x = nullptr;
    float* d_tmp_val = nullptr;
    int* d_tmp_idx = nullptr;
    float* d_tmp_norm = nullptr;
    signed char* d_overlap = nullptr;
#if GPU_OPT_USE_SUPPORT_OVERLAP_CACHE
    signed char* d_support_overlap = nullptr;  // K x K support overlap cache
#endif
#if GPU_OPT_USE_INCREMENTAL_CORR
    float* d_corr = nullptr;
    float* d_x_old = nullptr;  // Store old coefficients for delta computation
    int* d_inv_offset = nullptr;
    int* d_inv_flows = nullptr;
#endif

#if GPU_OPT_USE_RESTRICTED_SEARCH && GPU_OPT_USE_INCREMENTAL_CORR
    // Restricted search buffers
    float* d_abs_r = nullptr;           // |r[i]| for all buckets
    int* d_heavy_buckets = nullptr;     // Indices of heavy buckets
    int* d_heavy_count = nullptr;       // Number of heavy buckets found (atomic counter)
    unsigned int* d_candidate_bitmap = nullptr;  // Bitmap for candidate deduplication
    int* d_candidates = nullptr;        // Dense array of candidate flow indices
    int* d_candidate_count = nullptr;   // Number of candidates (atomic counter)
    float* d_restricted_tmp_val = nullptr;  // Scratch for restricted argmax
    int* d_restricted_tmp_idx = nullptr;
#endif
    
    // Sizes
    size_t y_bytes = GPU_OPT_M * sizeof(float);
    size_t idx_bytes = (size_t)N_local * GPU_OPT_D * sizeof(int);
    int corr_grid = div_up(N_local, block_size);
    int tmp_len = div_up(N_local, reduce_block);
    int norm_tmp_len = div_up(GPU_OPT_M, reduce_block);
    int support_size = 0;

    
#if GPU_OPT_USE_RESTRICTED_SEARCH && GPU_OPT_USE_INCREMENTAL_CORR
    // Restricted search sizes
    // Max candidates = heavy_buckets * avg_bucket_load = K * (N*D/M)
    int avg_bucket_load = (N_local * GPU_OPT_D + GPU_OPT_M - 1) / GPU_OPT_M;
    int max_candidates = GPU_OPT_HEAVY_BUCKET_COUNT * avg_bucket_load * 2;  // 2x safety margin
    if (max_candidates > N_local) max_candidates = N_local;
    int bitmap_words = (N_local + 31) / 32;
    int restricted_tmp_len = div_up(max_candidates, reduce_block);
#endif
    

    // Device buffers for NCCL AllReduce
    uint64_t* d_local_packed = nullptr;
    uint64_t* d_global_packed = nullptr;

    CUDA_CHECK(cudaSetDevice(device_id));

    // Allocate device memory (e.g., in the Device Buffers section)
    CUDA_CHECK(cudaMalloc(&d_local_packed, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_global_packed, sizeof(uint64_t)));

    // =========================================================================
    // Allocate Device Memory
    // =========================================================================
    CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
    CUDA_CHECK(cudaMalloc(&d_r, y_bytes));
    CUDA_CHECK(cudaMalloc(&d_idx, idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_S, K_max * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_L, (size_t)K_max * K_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, K_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_val, tmp_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_idx, tmp_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tmp_norm, norm_tmp_len * sizeof(float)));
    
#if GPU_OPT_PRECOMPUTE_OVERLAPS
    // Allocate overlap matrix (N x N int8)
    CUDA_CHECK(cudaMalloc(&d_overlap, (size_t)N_local * N_local * sizeof(signed char)));
#endif

#if GPU_OPT_USE_SUPPORT_OVERLAP_CACHE
    // Allocate support overlap cache (K x K int8) - much smaller than N x N
    CUDA_CHECK(cudaMalloc(&d_support_overlap, (size_t)K_max * K_max * sizeof(signed char)));
    CUDA_CHECK(cudaMemset(d_support_overlap, 0, (size_t)K_max * K_max * sizeof(signed char)));
#endif
    
#if GPU_OPT_USE_INCREMENTAL_CORR
    CUDA_CHECK(cudaMalloc(&d_corr, N_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_old, K_max * sizeof(float)));
    // Build inverse index
    if (!build_inverse_index_host(h_idx.data(), N_local, GPU_OPT_M,
                                   &d_inv_offset, &d_inv_flows)) {
        fprintf(stderr, "Failed to build inverse index\n");
        goto cleanup;
    }
#endif

#if GPU_OPT_USE_RESTRICTED_SEARCH && GPU_OPT_USE_INCREMENTAL_CORR
    // Allocate restricted search buffers
    CUDA_CHECK(cudaMalloc(&d_abs_r, GPU_OPT_M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_heavy_buckets, GPU_OPT_HEAVY_BUCKET_COUNT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_heavy_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_bitmap, bitmap_words * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_candidates, max_candidates * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_restricted_tmp_val, restricted_tmp_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_restricted_tmp_idx, restricted_tmp_len * sizeof(int)));
#endif
    
    // =========================================================================
    // Initialize Device Memory
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), y_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_y.data(), y_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_L, 0, (size_t)K_max * K_max * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x, 0, K_max * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_S, 0xff, K_max * sizeof(int)));
#if GPU_OPT_USE_INCREMENTAL_CORR
    CUDA_CHECK(cudaMemset(d_corr, 0, N_local * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x_old, 0, K_max * sizeof(float)));
#endif
    
#if GPU_OPT_PRECOMPUTE_OVERLAPS
    // Precompute overlap matrix
    {
        dim3 block2d(16, 16);
        dim3 grid2d(div_up(N_local, 16), div_up(N_local, 16));
        precompute_overlaps<GPU_OPT_D><<<grid2d, block2d>>>(
            d_idx, N_local, d_overlap);
        CUDA_CHECK(cudaGetLastError());
    }
#endif
    
    // =========================================================================
    // OMP Main Loop
    // =========================================================================
    while (support_size < K_max) {
        float best_val = 0.0f;
        int best_idx = -1;

        int local_best_rel_idx = -1;
        int local_best_global_idx = -1;
        uint64_t h_local_packed = 0;
        uint64_t h_global_packed = 0;
        float global_best_val = 0.0f;
        int global_best_idx = -1;
        
#if GPU_OPT_USE_INCREMENTAL_CORR
        if (support_size == 0) {
            // First iteration: compute correlations from y and store
            correlate_store_and_reduce<GPU_OPT_D><<<corr_grid, block_size>>>(
                d_y, d_idx, N_local, d_corr, d_tmp_val, d_tmp_idx);
            CUDA_CHECK(cudaGetLastError());
        } else {
#if GPU_OPT_USE_RESTRICTED_SEARCH
            // Decide whether to use restricted search or full scan
            bool use_restricted = (GPU_OPT_FULL_SCAN_INTERVAL == 0) ||
                                  (support_size % GPU_OPT_FULL_SCAN_INTERVAL != 0);
            
            if (use_restricted) {
                // ===============================================================
                // Restricted Search Path: Only scan flows in high-energy buckets
                // ===============================================================
                
                // Step 1+2 Fused: Compute |r[b]| AND find max in one pass
                float max_abs_r = 0.0f;
                {
                    int fused_grid = div_up(GPU_OPT_M, reduce_block);
                    compute_abs_and_max<<<fused_grid, reduce_block>>>(
                        d_r, GPU_OPT_M, d_abs_r, d_tmp_norm);
                    CUDA_CHECK(cudaGetLastError());
                    
                    // Reduce block-level max values to find global max
                    int cur_n = fused_grid;
                    while (cur_n > 1) {
                        int next_blocks = div_up(cur_n, reduce_block);
                        reduce_max_final<<<next_blocks, reduce_block>>>(
                            d_tmp_norm, cur_n, d_tmp_norm);
                        CUDA_CHECK(cudaGetLastError());
                        cur_n = next_blocks;
                    }
                    CUDA_CHECK(cudaMemcpy(&max_abs_r, d_tmp_norm, sizeof(float),
                                          cudaMemcpyDeviceToHost));
                }
                
                // Step 3: Select heavy buckets using threshold
                // Set threshold at fraction of max (select buckets with significant residual)
                float threshold = max_abs_r * 0.1f;  // Top ~10% by magnitude
                
                // Reset heavy bucket count
                CUDA_CHECK(cudaMemset(d_heavy_count, 0, sizeof(int)));
                
                int heavy_grid = div_up(GPU_OPT_M, block_size);
                select_heavy_buckets_threshold<<<heavy_grid, block_size>>>(
                    d_abs_r, GPU_OPT_M, threshold, d_heavy_buckets,
                    d_heavy_count, GPU_OPT_HEAVY_BUCKET_COUNT);
                CUDA_CHECK(cudaGetLastError());
                
                // Get number of heavy buckets found
                int h_heavy_count = 0;
                CUDA_CHECK(cudaMemcpy(&h_heavy_count, d_heavy_count, sizeof(int),
                                      cudaMemcpyDeviceToHost));
                
                if (h_heavy_count == 0) {
                    // No heavy buckets found, fall back to full scan
                    goto full_scan_path;
                }
                if (h_heavy_count > GPU_OPT_HEAVY_BUCKET_COUNT) {
                    h_heavy_count = GPU_OPT_HEAVY_BUCKET_COUNT;
                }
                
                // Step 4: Build candidate set from heavy buckets
                // Clear bitmap
                CUDA_CHECK(cudaMemset(d_candidate_bitmap, 0,
                                      bitmap_words * sizeof(unsigned int)));
                
                // Mark candidates in bitmap
                mark_candidates_from_buckets<<<h_heavy_count, 256>>>(
                    d_heavy_buckets, h_heavy_count, d_inv_offset, d_inv_flows,
                    d_candidate_bitmap);
                CUDA_CHECK(cudaGetLastError());
                
                // Compact to dense array (with bounds check)
                CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));
                int compact_grid = div_up(N_local, block_size);
                compact_candidates<<<compact_grid, block_size>>>(
                    d_candidate_bitmap, N_local, d_candidates, d_candidate_count,
                    max_candidates);
                CUDA_CHECK(cudaGetLastError());
                
                int h_candidate_count = 0;
                CUDA_CHECK(cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int),
                                      cudaMemcpyDeviceToHost));
                
                if (h_candidate_count == 0) {
                    // No candidates, fall back to full scan
                    goto full_scan_path;
                }
                if (h_candidate_count > max_candidates) {
                    h_candidate_count = max_candidates;
                }
                
                // Step 5: Restricted argmax over candidates only
                int restricted_grid = div_up(h_candidate_count, reduce_block);
                restricted_argmax_kernel<<<restricted_grid, reduce_block>>>(
                    d_corr, d_candidates, h_candidate_count,
                    d_restricted_tmp_val, d_restricted_tmp_idx);
                CUDA_CHECK(cudaGetLastError());
                
                // Final reduction if needed
                if (restricted_grid > 1) {
                    restricted_argmax_final<<<1, reduce_block>>>(
                        d_restricted_tmp_val, d_restricted_tmp_idx, restricted_grid,
                        d_restricted_tmp_val, d_restricted_tmp_idx);
                    CUDA_CHECK(cudaGetLastError());
                }
                CUDA_CHECK(cudaMemcpy(&best_val, d_restricted_tmp_val, sizeof(float),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&best_idx, d_restricted_tmp_idx, sizeof(int),
                                      cudaMemcpyDeviceToHost));
                
                goto check_stopping;
            }
            
full_scan_path:
#endif // GPU_OPT_USE_RESTRICTED_SEARCH
            // Full scan: argmax on all stored correlations
            if (!device_argmax_warp(d_corr, N_local, d_tmp_val, d_tmp_idx,
                                    reduce_block, &best_val, &best_idx)) {
                fprintf(stderr, "argmax failed\n");
                break;
            }
            // Skip the block reduction below
            goto check_stopping;
        }
#else
        // Non-incremental: compute correlations from residual each time
        correlate_and_reduce<GPU_OPT_D><<<corr_grid, block_size>>>(
            d_r, d_idx, N_local, d_tmp_val, d_tmp_idx);
        CUDA_CHECK(cudaGetLastError());
#endif
        
        // Final reduction for argmax across blocks
        if (corr_grid > 1) {
            if (!device_argmax_warp(d_tmp_val, corr_grid, d_tmp_val, d_tmp_idx,
                                    reduce_block, &best_val, &best_idx)) {
                fprintf(stderr, "argmax reduction failed\n");
                break;
            }
        } 
        // else {
        //     CUDA_CHECK(cudaMemcpy(&best_val, d_tmp_val, sizeof(float), cudaMemcpyDeviceToHost));
        //     CUDA_CHECK(cudaMemcpy(&best_idx, d_tmp_idx, sizeof(int), cudaMemcpyDeviceToHost));
        // }

        // MULTI-GPU STEP 1: Final Local ArgMax and Index Translation (GPU side)
        local_best_rel_idx = best_idx;
        local_best_global_idx = (local_best_rel_idx >= 0)
                                ? (N_start_idx + local_best_rel_idx)
                                : -1;

        // Pack to uint64_t using ordered float representation
        h_local_packed = pack_val_idx(best_val, local_best_global_idx);
        // Copy packed value to device
        CUDA_CHECK(cudaMemcpy(d_local_packed, &h_local_packed, sizeof(uint64_t),
                                cudaMemcpyHostToDevice));
    
        // MULTI-GPU STEP 2: Global ArgMax Reduction (NCCL)
        CUDA_CHECK(cudaStreamSynchronize(0));
        NCCL_CHECK(ncclAllReduce(
            d_local_packed,   // input buffer (local best flow)
            d_global_packed,  // output buffer (global best flow)
            1,
            ncclUint64,
            ncclMax,
            comm,           // NCCL communicator
            0               // CUDA stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(0)); // Wait for reduction to complete
        // MULTI-GPU STEP 3: Copy Global Result to Host & Check Stopping
        h_global_packed = 0;
        CUDA_CHECK(cudaMemcpy(&h_global_packed, d_global_packed, sizeof(uint64_t),
                                cudaMemcpyDeviceToHost));

        global_best_val = 0.0f;
        global_best_idx = -1;
        unpack_val_idx(h_global_packed, &global_best_val, &global_best_idx);

        // Update host variables
        best_val = global_best_val;
        best_idx = (global_best_idx >= 0) ? (global_best_idx - N_start_idx) : -1;

        
#if GPU_OPT_USE_INCREMENTAL_CORR
check_stopping:
#endif
        // Check stopping conditions
        if (best_idx < 0 || best_val <= GPU_OPT_CORR_THRESHOLD) {
            break;
        }
        
#if GPU_OPT_USE_INCREMENTAL_CORR
        // Save old coefficients BEFORE Cholesky update
        if (support_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_x_old, d_x, support_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
        // For the new flow, x_old is 0 (already initialized)
#endif
        
        // =====================================================================
        // Update Cholesky and Residual
        // =====================================================================
#if GPU_OPT_USE_SUPPORT_OVERLAP_CACHE
        // Compute overlaps between new flow and existing support (parallel)
        // Even for first flow (support_size=0), we launch with 1 block to compute diagonal
        {
            int overlap_threads = (support_size > 0) ? support_size : 1;
            int overlap_blocks = div_up(overlap_threads, 256);
            compute_support_flow_overlap<GPU_OPT_D><<<overlap_blocks, 256>>>(
                d_idx, best_idx, d_S, support_size, d_support_overlap, K_max);
            CUDA_CHECK(cudaGetLastError());
        }
#endif
        {
            size_t shmem = (size_t)K_max * 4 * sizeof(float);
            update_cholesky_and_residual<GPU_OPT_D><<<1, 256, shmem>>>(
                d_y, d_r, d_idx, best_idx, d_S, support_size, d_L, d_x, K_max,
#if GPU_OPT_PRECOMPUTE_OVERLAPS
                d_overlap, N_local,
#else
                nullptr, 0,
#endif
#if GPU_OPT_USE_SUPPORT_OVERLAP_CACHE
                d_support_overlap
#else
                nullptr
#endif
            );
            CUDA_CHECK(cudaGetLastError());
        }
        
#if GPU_OPT_USE_INCREMENTAL_CORR
        // Update correlations based on coefficient DELTAS for ALL selected flows
        // Grid: (support_size + 1) * D blocks, each handles one (flow, counter) pair
        {
            int new_support_size = support_size + 1;
            int update_grid = new_support_size * GPU_OPT_D;
            update_correlations_all_deltas<GPU_OPT_D><<<update_grid, 256>>>(
                d_inv_offset, d_inv_flows, d_idx, d_S, d_x, d_x_old,
                new_support_size, d_corr);
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Zero out selected flow's correlation so it won't be selected again
        if (best_idx >= 0 && best_idx < N_local) {
            CUDA_CHECK(cudaMemset(&d_corr[best_idx], 0, sizeof(float)));
        }
#endif
        
        // Compute residual norm for stopping criterion
        float norm_r = 0.0f;
        if (!device_l2_norm_warp(d_r, GPU_OPT_M, d_tmp_norm, reduce_block, &norm_r)) {
            fprintf(stderr, "L2 norm failed\n");
            break;
        }
        
        support_size++;
        
        if (norm_r < GPU_OPT_RESIDUAL_THRESHOLD) {
            break;
        }
    }
    
    // =========================================================================
    // Copy Results Back to Host
    // =========================================================================
    if (support_size > 0) {
        CUDA_CHECK(cudaMemcpy(h_support.data(), d_S, support_size * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_support_x.data(), d_x, support_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, y_bytes, cudaMemcpyDeviceToHost));
        
        // Update CPU residual
        for (int i = 0; i < GPU_OPT_M; ++i) {
            r[i] = (double)h_r[i];
        }
        
        // Update CPU flow outputs
        for (int p = 0; p < support_size; ++p) {
            int fi = h_support[p];
            if (fi >= 0 && fi < N) {
                selected[fi] = 1;
                x[fi] = (double)h_support_x[p];
            }
        }
    }
    
cleanup:
    // =========================================================================
    // Cleanup
    // =========================================================================

    if (d_global_packed) cudaFree(d_global_packed);
    if (d_local_packed)  cudaFree(d_local_packed);
    if (comm)          ncclCommDestroy(comm);

#if GPU_OPT_USE_RESTRICTED_SEARCH && GPU_OPT_USE_INCREMENTAL_CORR
    if (d_restricted_tmp_idx) cudaFree(d_restricted_tmp_idx);
    if (d_restricted_tmp_val) cudaFree(d_restricted_tmp_val);
    if (d_candidate_count) cudaFree(d_candidate_count);
    if (d_candidates) cudaFree(d_candidates);
    if (d_candidate_bitmap) cudaFree(d_candidate_bitmap);
    if (d_heavy_count) cudaFree(d_heavy_count);
    if (d_heavy_buckets) cudaFree(d_heavy_buckets);
    if (d_abs_r) cudaFree(d_abs_r);
#endif
#if GPU_OPT_USE_INCREMENTAL_CORR
    if (d_inv_flows) cudaFree(d_inv_flows);
    if (d_inv_offset) cudaFree(d_inv_offset);
    if (d_x_old) cudaFree(d_x_old);
    if (d_corr) cudaFree(d_corr);
#endif
#if GPU_OPT_USE_SUPPORT_OVERLAP_CACHE
    if (d_support_overlap) cudaFree(d_support_overlap);
#endif
    if (d_overlap) cudaFree(d_overlap);
    if (d_tmp_norm) cudaFree(d_tmp_norm);
    if (d_tmp_idx) cudaFree(d_tmp_idx);
    if (d_tmp_val) cudaFree(d_tmp_val);
    if (d_x) cudaFree(d_x);
    if (d_L) cudaFree(d_L);
    if (d_S) cudaFree(d_S);
    if (d_idx) cudaFree(d_idx);
    if (d_r) cudaFree(d_r);
    if (d_y) cudaFree(d_y);
}