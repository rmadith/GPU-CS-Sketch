/**
 * GPU-Optimized OMP Implementation for Count-Min Sketch
 * 
 * Optimizations:
 * 1. Warp shuffle reductions (no shared memory bank conflicts)
 * 2. Read-only cache (__ldg) for residual access
 * 3. Fused correlation + argmax kernel
 * 4. Precomputed flow overlap matrix
 * 5. Incremental correlation updates via inverse index (with coefficient deltas)
 * 
 * Build: nvcc -O2 -std=c++14 -DUSE_CUDA -gencode arch=compute_75,code=sm_75 \
 *        CPU/run.c CPU/server.c CPU/switch.c GPU_Optimized/server_gpu_opt.cu -o gpu-opt-run
 */

#include "server_gpu_opt.h"
#include "kernels/reduction.cuh"
#include "kernels/correlation.cuh"
#include "kernels/cholesky.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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

// =============================================================================
// Utility Functions
// =============================================================================

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

extern "C" void server_reconstruct_omp_gpu(int K_max, int flow_count) {
    if (flow_count <= 0) return;
    if (flow_count > N) flow_count = N;
    if (K_max > flow_count) K_max = flow_count;
    
    const int block_size = GPU_OPT_CORR_BLOCK_SIZE;
    const int reduce_block = GPU_OPT_REDUCE_BLOCK_SIZE;
    
    // =========================================================================
    // Host Buffers
    // =========================================================================
    std::vector<float> h_y(GPU_OPT_M);
    std::vector<int> h_idx(flow_count * GPU_OPT_D);
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
    for (int i = 0; i < flow_count; ++i) {
        for (int j = 0; j < GPU_OPT_D; ++j) {
            h_idx[i * GPU_OPT_D + j] = j * GPU_OPT_W + (int)flows[i].idx[j];
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
#if GPU_OPT_USE_INCREMENTAL_CORR
    float* d_corr = nullptr;
    float* d_x_old = nullptr;  // Store old coefficients for delta computation
    int* d_inv_offset = nullptr;
    int* d_inv_flows = nullptr;
#endif
    
    // Sizes
    size_t y_bytes = GPU_OPT_M * sizeof(float);
    size_t idx_bytes = (size_t)flow_count * GPU_OPT_D * sizeof(int);
    int corr_grid = div_up(flow_count, block_size);
    int tmp_len = div_up(flow_count, reduce_block);
    int norm_tmp_len = div_up(GPU_OPT_M, reduce_block);
    int support_size = 0;
    
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
    CUDA_CHECK(cudaMalloc(&d_overlap, (size_t)flow_count * flow_count * sizeof(signed char)));
#endif
    
#if GPU_OPT_USE_INCREMENTAL_CORR
    CUDA_CHECK(cudaMalloc(&d_corr, flow_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_old, K_max * sizeof(float)));
    // Build inverse index
    if (!build_inverse_index_host(h_idx.data(), flow_count, GPU_OPT_M,
                                   &d_inv_offset, &d_inv_flows)) {
        fprintf(stderr, "Failed to build inverse index\n");
        goto cleanup;
    }
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
    CUDA_CHECK(cudaMemset(d_corr, 0, flow_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x_old, 0, K_max * sizeof(float)));
#endif
    
#if GPU_OPT_PRECOMPUTE_OVERLAPS
    // Precompute overlap matrix
    {
        dim3 block2d(16, 16);
        dim3 grid2d(div_up(flow_count, 16), div_up(flow_count, 16));
        precompute_overlaps<GPU_OPT_D><<<grid2d, block2d>>>(
            d_idx, flow_count, d_overlap);
        CUDA_CHECK(cudaGetLastError());
    }
#endif
    
    // =========================================================================
    // OMP Main Loop
    // =========================================================================
    while (support_size < K_max) {
        float best_val = 0.0f;
        int best_idx = -1;
        
#if GPU_OPT_USE_INCREMENTAL_CORR
        if (support_size == 0) {
            // First iteration: compute correlations from y and store
            correlate_store_and_reduce<GPU_OPT_D><<<corr_grid, block_size>>>(
                d_y, d_idx, flow_count, d_corr, d_tmp_val, d_tmp_idx);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // Subsequent iterations: correlations were updated incrementally
            // Just do argmax on stored correlations
            if (!device_argmax_warp(d_corr, flow_count, d_tmp_val, d_tmp_idx,
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
            d_r, d_idx, flow_count, d_tmp_val, d_tmp_idx);
        CUDA_CHECK(cudaGetLastError());
#endif
        
        // Final reduction for argmax across blocks
        if (corr_grid > 1) {
            if (!device_argmax_warp(d_tmp_val, corr_grid, d_tmp_val, d_tmp_idx,
                                    reduce_block, &best_val, &best_idx)) {
                fprintf(stderr, "argmax reduction failed\n");
                break;
            }
        } else {
            CUDA_CHECK(cudaMemcpy(&best_val, d_tmp_val, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&best_idx, d_tmp_idx, sizeof(int), cudaMemcpyDeviceToHost));
        }
        
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
        {
            size_t shmem = (size_t)K_max * 4 * sizeof(float);
            update_cholesky_and_residual<GPU_OPT_D><<<1, 256, shmem>>>(
                d_y, d_r, d_idx, best_idx, d_S, support_size, d_L, d_x, K_max,
#if GPU_OPT_PRECOMPUTE_OVERLAPS
                d_overlap, flow_count
#else
                nullptr, 0
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
        CUDA_CHECK(cudaMemset(&d_corr[best_idx], 0, sizeof(float)));
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
#if GPU_OPT_USE_INCREMENTAL_CORR
    if (d_inv_flows) cudaFree(d_inv_flows);
    if (d_inv_offset) cudaFree(d_inv_offset);
    if (d_x_old) cudaFree(d_x_old);
    if (d_corr) cudaFree(d_corr);
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

