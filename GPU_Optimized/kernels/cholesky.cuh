#ifndef CHOLESKY_CUH
#define CHOLESKY_CUH

#include <cuda_runtime.h>
#include "../server_gpu_opt.h"

// =============================================================================
// Precomputation Kernels
// =============================================================================

/**
 * Precompute flow overlap matrix.
 * overlap[i][j] = number of shared hash positions between flow i and flow j.
 * 
 * Uses int8 storage since overlap is in range [0, D].
 * 
 * @param idx     Flow index array (N * D entries)
 * @param n       Number of flows
 * @param overlap Output overlap matrix (N x N, stored as int8)
 */
template<int D_VAL>
__global__ void precompute_overlaps(const int* __restrict__ idx, int n,
                                    signed char* overlap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n && j <= i) {
        signed char count = 0;
        int base_i = i * D_VAL;
        int base_j = j * D_VAL;
        
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            if (idx[base_i + d] == idx[base_j + d]) {
                count++;
            }
        }
        
        // Store symmetric (only compute lower triangle, copy to upper)
        overlap[i * n + j] = count;
        if (i != j) {
            overlap[j * n + i] = count;
        }
    }
}

template __global__ void precompute_overlaps<4>(const int* __restrict__, int,
                                                 signed char*);

// =============================================================================
// Incremental Cholesky Update Kernel
// =============================================================================

/**
 * Combined kernel: Update support set, Cholesky factor, solve, and update residual.
 * Runs on a single thread block for simplicity (appropriate for moderate K).
 * 
 * For large K (> threshold), use cuSOLVER instead.
 * 
 * @param y          Original measurement vector (length M)
 * @param r          Residual vector (length M, updated in place)
 * @param idx        Flow index array (N * D)
 * @param lambda     Newly selected flow index
 * @param S          Support set (selected flow indices)
 * @param k          Current support size (before adding lambda)
 * @param L          Cholesky factor (K_max x K_max, lower triangular)
 * @param x          Flow coefficients (length K_max)
 * @param max_k      Maximum support size (for matrix stride)
 * @param overlap    Precomputed overlap matrix (N x N, int8) or NULL
 * @param n          Total number of flows (for overlap indexing)
 */
template<int D_VAL>
__global__ void update_cholesky_and_residual(
    const float* __restrict__ y,
    float* r,
    const int* __restrict__ idx,
    int lambda,
    int* S,
    int k,
    float* L,
    float* x,
    int max_k,
    const signed char* __restrict__ overlap,
    int n) {
    
    extern __shared__ float shmem[];
    float* v = shmem;                    // Overlaps with existing support (k elements)
    float* w = v + max_k;                // Forward solve scratch
    float* b = w + max_k;                // RHS: Φ_S^T y
    float* t = b + max_k;                // Forward solve result
    
    // Parallel copy: r = y (reset residual for recomputation)
    for (int pos = threadIdx.x; pos < GPU_OPT_M; pos += blockDim.x) {
        r[pos] = y[pos];
    }
    __syncthreads();
    
    // Single thread performs Cholesky update and solve
    if (threadIdx.x == 0) {
        // Add lambda to support set
        S[k] = lambda;
        
        // Get lambda's hash positions
        int lambda_base = lambda * D_VAL;
        int lambda_rows[D_VAL];
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            lambda_rows[d] = idx[lambda_base + d];
        }
        
        // Compute overlaps with existing support
        // v[p] = overlap(lambda, S[p])
        for (int p = 0; p < k; ++p) {
            int other = S[p];
            
            if (overlap != NULL) {
                // Use precomputed overlap matrix
                v[p] = (float)overlap[lambda * n + other];
            } else {
                // Compute overlap on the fly
                int other_base = other * D_VAL;
                int count = 0;
                #pragma unroll
                for (int d = 0; d < D_VAL; ++d) {
                    if (lambda_rows[d] == idx[other_base + d]) {
                        count++;
                    }
                }
                v[p] = (float)count;
            }
        }
        
        // Compute diagonal element (self-overlap = number of unique positions)
        int unique_rows = D_VAL;
        for (int a = 0; a < D_VAL; ++a) {
            for (int b_idx = a + 1; b_idx < D_VAL; ++b_idx) {
                if (lambda_rows[a] == lambda_rows[b_idx]) {
                    unique_rows--;
                }
            }
        }
        
        // Forward solve: L[0:k, 0:k] * w = v
        for (int i = 0; i < k; ++i) {
            float sum = v[i];
            for (int j = 0; j < i; ++j) {
                sum -= L[i * max_k + j] * w[j];
            }
            w[i] = sum / L[i * max_k + i];
        }
        
        // Compute new diagonal: L[k,k] = sqrt(diag - w^T w)
        float diag_sq = (float)unique_rows;
        for (int i = 0; i < k; ++i) {
            diag_sq -= w[i] * w[i];
        }
        if (diag_sq < GPU_OPT_CHOL_MIN_DIAG) {
            diag_sq = GPU_OPT_CHOL_MIN_DIAG;
        }
        float diag = sqrtf(diag_sq);
        
        // Store new row of L
        for (int j = 0; j < k; ++j) {
            L[k * max_k + j] = w[j];
        }
        L[k * max_k + k] = diag;
        
        // Build RHS: b[p] = Φ_{S[p]}^T y
        for (int p = 0; p <= k; ++p) {
            int flow_idx_p = S[p];
            int base = flow_idx_p * D_VAL;
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_VAL; ++d) {
                sum += y[idx[base + d]];
            }
            b[p] = sum;
        }
        
        // Forward solve: L * t = b
        for (int i = 0; i <= k; ++i) {
            float sum = b[i];
            for (int j = 0; j < i; ++j) {
                sum -= L[i * max_k + j] * t[j];
            }
            t[i] = sum / L[i * max_k + i];
        }
        
        // Backward solve: L^T * x = t
        for (int i = k; i >= 0; --i) {
            float sum = t[i];
            for (int j = i + 1; j <= k; ++j) {
                sum -= L[j * max_k + i] * x[j];
            }
            x[i] = sum / L[i * max_k + i];
        }
        
        // Update residual: r = y - Φ_S * x_S
        for (int p = 0; p <= k; ++p) {
            int flow_idx_p = S[p];
            int base = flow_idx_p * D_VAL;
            float coeff = x[p];
            #pragma unroll
            for (int d = 0; d < D_VAL; ++d) {
                int pos = idx[base + d];
                r[pos] -= coeff;
            }
        }
    }
}

template __global__ void update_cholesky_and_residual<4>(
    const float* __restrict__, float*, const int* __restrict__,
    int, int*, int, float*, float*, int, const signed char* __restrict__, int);

// =============================================================================
// Utility Kernels
// =============================================================================

/**
 * Initialize arrays to zero.
 */
__global__ void init_zero_float(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0.0f;
}

__global__ void init_zero_int(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0;
}

/**
 * Copy y to r (for residual initialization).
 */
__global__ void copy_array(const float* __restrict__ src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

#endif // CHOLESKY_CUH

