#ifndef CORRELATION_CUH
#define CORRELATION_CUH

#include <cuda_runtime.h>
#include "../server_gpu_opt.h"
#include "reduction.cuh"

// =============================================================================
// Correlation Computation Kernels
// =============================================================================

/**
 * Basic correlation kernel with read-only cache optimization.
 * Computes c[i] = sum of residual values at flow i's hash positions.
 * 
 * Uses __ldg() for cached reads and __restrict__ for compiler optimization.
 * 
 * @param r    Residual vector (length M)
 * @param idx  Flow index array (N * D entries, each flow has D positions)
 * @param n    Number of flows
 * @param c    Output correlation array (length N)
 */
template<int D_VAL>
__global__ void correlate_gather_opt(const float* __restrict__ r,
                                     const int* __restrict__ idx,
                                     int n, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        int base = i * D_VAL;
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            int pos = idx[base + d];
            sum += __ldg(&r[pos]);  // Read-only cache
        }
        c[i] = sum;
    }
}

// Explicit instantiation for D=4
template __global__ void correlate_gather_opt<4>(const float* __restrict__,
                                                  const int* __restrict__,
                                                  int, float*);

/**
 * Fused correlation + partial argmax kernel.
 * Computes correlation and performs block-level argmax reduction in one pass.
 * Reduces memory bandwidth by avoiding intermediate correlation array write/read.
 * 
 * @param r          Residual vector (length M)
 * @param idx        Flow index array (N * D entries)
 * @param n          Number of flows
 * @param block_max  Output: one max correlation per block
 * @param block_idx  Output: one argmax index per block
 */
template<int D_VAL>
__global__ void correlate_and_reduce(const float* __restrict__ r,
                                     const int* __restrict__ idx,
                                     int n,
                                     float* block_max, int* block_idx) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    // Compute correlation for this thread's flow
    float corr = -1.0f;
    int my_idx = -1;
    
    if (tid < n) {
        float sum = 0.0f;
        int base = tid * D_VAL;
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            sum += __ldg(&r[idx[base + d]]);
        }
        corr = fabsf(sum);
        my_idx = tid;
    }
    
    // Warp-level argmax reduction
    warpReduceArgMax(corr, my_idx);
    
    // Store warp results to shared memory
    if (lane == 0) {
        s_val[warp_id] = corr;
        s_idx[warp_id] = my_idx;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        corr = (lane < num_warps) ? s_val[lane] : -1.0f;
        my_idx = (lane < num_warps) ? s_idx[lane] : -1;
        warpReduceArgMax(corr, my_idx);
        
        if (lane == 0) {
            block_max[blockIdx.x] = corr;
            block_idx[blockIdx.x] = my_idx;
        }
    }
}

// Explicit instantiation for D=4
template __global__ void correlate_and_reduce<4>(const float* __restrict__,
                                                  const int* __restrict__,
                                                  int, float*, int*);

/**
 * Fused kernel that also stores correlations (for incremental update mode).
 * Computes correlation, stores it, and performs block-level argmax.
 */
template<int D_VAL>
__global__ void correlate_store_and_reduce(const float* __restrict__ r,
                                           const int* __restrict__ idx,
                                           int n, float* corr_out,
                                           float* block_max, int* block_idx) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float corr = -1.0f;
    float corr_signed = 0.0f;
    int my_idx = -1;
    
    if (tid < n) {
        float sum = 0.0f;
        int base = tid * D_VAL;
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            sum += __ldg(&r[idx[base + d]]);
        }
        corr_signed = sum;
        corr = fabsf(sum);
        my_idx = tid;
        
        // Store correlation for incremental updates
        corr_out[tid] = corr_signed;
    }
    
    warpReduceArgMax(corr, my_idx);
    
    if (lane == 0) {
        s_val[warp_id] = corr;
        s_idx[warp_id] = my_idx;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        corr = (lane < num_warps) ? s_val[lane] : -1.0f;
        my_idx = (lane < num_warps) ? s_idx[lane] : -1;
        warpReduceArgMax(corr, my_idx);
        
        if (lane == 0) {
            block_max[blockIdx.x] = corr;
            block_idx[blockIdx.x] = my_idx;
        }
    }
}

template __global__ void correlate_store_and_reduce<4>(const float* __restrict__,
                                                        const int* __restrict__,
                                                        int, float*, float*, int*);

// =============================================================================
// Incremental Correlation Update
// =============================================================================

/**
 * Update correlations based on coefficient DELTAS for ALL selected flows.
 * This correctly handles the fact that OMP re-solves least squares, changing
 * all coefficients, not just the new one.
 * 
 * For each selected flow i with delta_x[i] = x_new[i] - x_old[i]:
 *   For each counter c in flow i's pattern:
 *     For each flow f that hashes to counter c:
 *       corr[f] -= delta_x[i]
 * 
 * @param inv_offset   CSR offset array (length M+1)
 * @param inv_flows    CSR flow indices
 * @param flow_idx     Flow hash positions (N * D)
 * @param support      Array of selected flow indices
 * @param x_new        New coefficients after Cholesky update
 * @param x_old        Old coefficients before Cholesky update  
 * @param k            Number of selected flows (including new one)
 * @param corr         Correlation array to update
 */
template<int D_VAL>
__global__ void update_correlations_all_deltas(
    const int* __restrict__ inv_offset,
    const int* __restrict__ inv_flows,
    const int* __restrict__ flow_idx,
    const int* __restrict__ support,
    const float* __restrict__ x_new,
    const float* __restrict__ x_old,
    int k,
    float* corr) {
    
    // Each block handles one (support_flow, d) pair
    // Grid: k * D_VAL blocks
    int support_idx = blockIdx.x / D_VAL;
    int d = blockIdx.x % D_VAL;
    
    if (support_idx >= k) return;
    
    // Get the selected flow and its coefficient delta
    int flow_i = support[support_idx];
    float delta = x_new[support_idx] - x_old[support_idx];
    
    // Skip if delta is negligible
    if (fabsf(delta) < 1e-10f) return;
    
    // Get the counter for this (flow, d) pair
    int counter = flow_idx[flow_i * D_VAL + d];
    int start = inv_offset[counter];
    int end = inv_offset[counter + 1];
    
    // Update correlations of all flows sharing this counter
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int flow = inv_flows[i];
        atomicAdd(&corr[flow], -delta);
    }
}

template __global__ void update_correlations_all_deltas<4>(
    const int* __restrict__, const int* __restrict__, const int* __restrict__,
    const int* __restrict__, const float* __restrict__, const float* __restrict__,
    int, float*);

/**
 * Zero out correlation for selected flows (they shouldn't be selected again).
 */
__global__ void zero_selected_correlation(const int* selected_flows, int k,
                                          float* corr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        corr[selected_flows[i]] = 0.0f;
    }
}

#endif // CORRELATION_CUH

