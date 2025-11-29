#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda_runtime.h>
#include <math.h>

// =============================================================================
// Warp-Level Reduction Primitives
// =============================================================================

/**
 * Warp-level argmax reduction using shuffle instructions.
 * Finds maximum value and its index within a warp (32 threads).
 * 
 * @param val  Value to compare (modified in-place to contain warp max)
 * @param idx  Index associated with value (modified to contain argmax index)
 */
__device__ __forceinline__ void warpReduceArgMax(float& val, int& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

/**
 * Warp-level sum reduction using shuffle instructions.
 * 
 * @param val  Value to sum (modified in-place to contain warp sum in lane 0)
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp-level max reduction using shuffle instructions.
 * 
 * @param val  Value to compare (modified in-place to contain warp max in lane 0)
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// =============================================================================
// Block-Level ArgMax Kernel (using warp shuffles)
// =============================================================================

/**
 * First-stage argmax reduction kernel.
 * Computes absolute value and finds block-local maximum.
 * Uses warp shuffles for intra-warp reduction, shared memory only for inter-warp.
 * 
 * @param vals     Input values (correlations)
 * @param n        Number of elements
 * @param out_val  Output: one max value per block
 * @param out_idx  Output: one argmax index per block
 */
__global__ void argmax_warp_first(const float* __restrict__ vals, int n,
                                  float* out_val, int* out_idx) {
    // Shared memory only needed for inter-warp communication (32 warps max)
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;        // threadIdx.x % 32
    int warp_id = threadIdx.x >> 5;     // threadIdx.x / 32
    int num_warps = blockDim.x >> 5;    // blockDim.x / 32
    
    // Load and take absolute value
    float val = -1.0f;
    int idx = -1;
    if (tid < n) {
        val = fabsf(vals[tid]);
        idx = tid;
    }
    
    // Warp-level reduction
    warpReduceArgMax(val, idx);
    
    // First thread of each warp writes to shared memory
    if (lane == 0) {
        s_val[warp_id] = val;
        s_idx[warp_id] = idx;
    }
    __syncthreads();
    
    // Final reduction in first warp only
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_val[lane] : -1.0f;
        idx = (lane < num_warps) ? s_idx[lane] : -1;
        warpReduceArgMax(val, idx);
        
        if (lane == 0) {
            out_val[blockIdx.x] = val;
            out_idx[blockIdx.x] = idx;
        }
    }
}

/**
 * Subsequent reduction stages for argmax.
 * Input values are already absolute, just need to reduce.
 */
__global__ void argmax_warp_reduce(const float* __restrict__ vals,
                                   const int* __restrict__ in_idx,
                                   int n,
                                   float* out_val, int* out_idx) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float val = -1.0f;
    int idx = -1;
    if (tid < n) {
        val = vals[tid];
        idx = in_idx[tid];
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
            out_val[blockIdx.x] = val;
            out_idx[blockIdx.x] = idx;
        }
    }
}

// =============================================================================
// Block-Level L2 Norm Kernels
// =============================================================================

/**
 * Compute partial sum of squares for L2 norm.
 */
__global__ void l2_norm_partial(const float* __restrict__ vals, int n, float* out) {
    __shared__ float s_data[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float val = 0.0f;
    if (tid < n) {
        float v = vals[tid];
        val = v * v;
    }
    
    // Warp-level sum
    val = warpReduceSum(val);
    
    if (lane == 0) {
        s_data[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_data[lane] : 0.0f;
        val = warpReduceSum(val);
        
        if (lane == 0) {
            out[blockIdx.x] = val;
        }
    }
}

/**
 * Final sum reduction for multiple values.
 */
__global__ void reduce_sum_final(const float* __restrict__ vals, int n, float* out) {
    __shared__ float s_data[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    
    float val = (tid < n) ? vals[tid] : 0.0f;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        s_data[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < num_warps) ? s_data[lane] : 0.0f;
        val = warpReduceSum(val);
        
        if (lane == 0) {
            out[blockIdx.x] = val;
        }
    }
}

#endif // REDUCTION_CUH

