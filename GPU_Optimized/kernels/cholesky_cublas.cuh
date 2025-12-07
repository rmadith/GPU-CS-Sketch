#ifndef CHOLESKY_CUBLAS_CUH
#define CHOLESKY_CUBLAS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include "../server_gpu_opt.h"

// =============================================================================
// cuBLAS-Accelerated Cholesky Update and Solve
//
// Replaces the sequential triangular solves with parallel cuBLAS calls.
// This provides significant speedup for large K (K > 100).
// =============================================================================

// =============================================================================
// Kernel 1: Compute overlaps and update Cholesky row (incremental update)
// This is O(K) work - stays on single thread as it's fast
// =============================================================================

template<int D_VAL>
__global__ void update_cholesky_row_cublas(
    const int* __restrict__ idx,
    int lambda,
    int* S,
    int k,
    float* L,
    float* v,           // Output: overlap vector for cuBLAS solve
    float* diag_out,    // Output: base diagonal value (unique_rows)
    int max_k,
    const signed char* __restrict__ support_overlap = nullptr) {
    
    if (threadIdx.x != 0) return;
    
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
    int unique_rows = D_VAL;
    
    if (support_overlap != nullptr) {
        // Use precomputed support overlap cache
        for (int p = 0; p < k; ++p) {
            v[p] = (float)support_overlap[k * max_k + p];
        }
        unique_rows = (int)support_overlap[k * max_k + k];
    } else {
        // Compute overlap on the fly
        for (int p = 0; p < k; ++p) {
            int other = S[p];
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
        // Compute diagonal (unique positions)
        for (int a = 0; a < D_VAL; ++a) {
            for (int b = a + 1; b < D_VAL; ++b) {
                if (lambda_rows[a] == lambda_rows[b]) {
                    unique_rows--;
                }
            }
        }
    }
    
    // Store unique_rows for later diagonal computation
    *diag_out = (float)unique_rows;
}

template __global__ void update_cholesky_row_cublas<4>(
    const int* __restrict__, int, int*, int, float*, float*, float*, int,
    const signed char* __restrict__);

// =============================================================================
// Kernel 2: Complete Cholesky row after cuBLAS solve
// w contains the solved L[0:k-1, 0:k-1]^{-1} * v
// =============================================================================

__global__ void complete_cholesky_row_cublas(
    float* L,
    const float* w,      // Result from cuBLAS forward solve
    float diag_base,     // unique_rows value
    int k,
    int max_k) {
    
    if (threadIdx.x != 0) return;
    
    // Compute new diagonal: L[k,k] = sqrt(diag_base - w^T w)
    float diag_sq = diag_base;
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
}

// =============================================================================
// Kernel 3: Build RHS vector b = Φ_S^T y
// Parallelized across support elements
// =============================================================================

template<int D_VAL>
__global__ void build_rhs_vector_cublas(
    const float* __restrict__ y,
    const int* __restrict__ idx,
    const int* __restrict__ S,
    int k_plus_1,        // Support size (k+1)
    float* b) {
    
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p < k_plus_1) {
        int flow_idx = S[p];
        int base = flow_idx * D_VAL;
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            sum += y[idx[base + d]];
        }
        b[p] = sum;
    }
}

template __global__ void build_rhs_vector_cublas<4>(
    const float* __restrict__, const int* __restrict__,
    const int* __restrict__, int, float*);

// =============================================================================
// Kernel 4: Reset residual to y
// =============================================================================

__global__ void reset_residual_cublas(float* r, const float* __restrict__ y, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        r[i] = y[i];
    }
}

// =============================================================================
// Kernel 5: Subtract flow contributions from residual
// r[pos] -= x[p] for each position in flow S[p]
// =============================================================================

template<int D_VAL>
__global__ void subtract_from_residual_cublas(
    float* r,
    const int* __restrict__ idx,
    const int* __restrict__ S,
    const float* __restrict__ x,
    int k_plus_1) {
    
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p < k_plus_1) {
        int flow_idx = S[p];
        int base = flow_idx * D_VAL;
        float coeff = x[p];
        
        // Atomic subtract since multiple flows may hash to same bucket
        #pragma unroll
        for (int d = 0; d < D_VAL; ++d) {
            int pos = idx[base + d];
            atomicAdd(&r[pos], -coeff);
        }
    }
}

template __global__ void subtract_from_residual_cublas<4>(
    float*, const int* __restrict__, const int* __restrict__,
    const float* __restrict__, int);

// =============================================================================
// Host function: Perform Cholesky update and solve using cuBLAS
// 
// This replaces the monolithic update_cholesky_and_residual kernel with:
// 1. Small kernel to compute overlaps (O(K))
// 2. cuBLAS triangular solve for Cholesky row (O(K²) parallelized)
// 3. Small kernel to complete Cholesky row (O(K))
// 4. Parallel kernel to build RHS (O(K))
// 5. cuBLAS forward solve L*t=b (O(K²) parallelized)
// 6. cuBLAS backward solve L^T*x=t (O(K²) parallelized)
// 7. Parallel kernels to update residual (O(M + K*D))
// =============================================================================

inline bool cublas_cholesky_update_and_solve(
    cublasHandle_t handle,
    const float* d_y,
    float* d_r,
    const int* d_idx,
    int lambda,
    int* d_S,
    int k,                  // Current support size BEFORE adding lambda
    float* d_L,
    float* d_x,
    int max_k,
    float* d_v,             // Scratch: overlap vector (max_k floats)
    float* d_b,             // Scratch: RHS vector (max_k floats)
    float* d_diag,          // Scratch: single float for diagonal
    const signed char* d_support_overlap,
    cudaStream_t stream = 0) {
    
    cublasSetStream(handle, stream);
    
    // Step 1: Compute overlaps and prepare for Cholesky row update
    update_cholesky_row_cublas<GPU_OPT_D><<<1, 1, 0, stream>>>(
        d_idx, lambda, d_S, k, d_L, d_v, d_diag, max_k, d_support_overlap);
    
    if (k > 0) {
        // Step 2: Solve L[0:k, 0:k] * w = v using cuBLAS
        // This is the expensive O(K²) part - now parallelized!
        // Note: cuBLAS expects column-major, but our L is row-major.
        // For lower triangular row-major, we use upper triangular column-major with transpose.
        cublasStatus_t status = cublasStrsv(
            handle,
            CUBLAS_FILL_MODE_UPPER,     // Row-major lower = Column-major upper
            CUBLAS_OP_T,                // Transpose because of row/column major difference
            CUBLAS_DIAG_NON_UNIT,       // Has non-unit diagonal
            k,                          // Matrix dimension
            d_L,                        // L matrix (max_k x max_k, row-major)
            max_k,                      // Leading dimension
            d_v,                        // RHS vector (overwritten with solution)
            1                           // Increment
        );
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS Strsv (Cholesky row) failed: %d\n", status);
            return false;
        }
    }
    
    // Step 3: Complete Cholesky row (compute diagonal and store)
    // Need to sync to get diag value, but we can overlap with async copy
    float h_diag;
    cudaMemcpyAsync(&h_diag, d_diag, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    complete_cholesky_row_cublas<<<1, 1, 0, stream>>>(d_L, d_v, h_diag, k, max_k);
    
    // Step 4: Build RHS vector b = Φ_S^T y (parallelized!)
    int k_plus_1 = k + 1;
    int rhs_blocks = (k_plus_1 + 255) / 256;
    build_rhs_vector_cublas<GPU_OPT_D><<<rhs_blocks, 256, 0, stream>>>(
        d_y, d_idx, d_S, k_plus_1, d_b);
    
    // Step 5: Forward solve L * t = b (result in d_b)
    // Same row/column major handling as above
    cublasStatus_t status = cublasStrsv(
        handle,
        CUBLAS_FILL_MODE_UPPER,     // Row-major lower = Column-major upper
        CUBLAS_OP_T,                // Transpose
        CUBLAS_DIAG_NON_UNIT,
        k_plus_1,
        d_L,
        max_k,
        d_b,        // RHS, overwritten with solution t
        1
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Strsv (forward) failed: %d\n", status);
        return false;
    }
    
    // Step 6: Backward solve L^T * x = t (result in d_x)
    // Copy t to x first, then solve in place
    cudaMemcpyAsync(d_x, d_b, k_plus_1 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    // For L^T solve with row-major L, we need CUBLAS_OP_N (no transpose)
    status = cublasStrsv(
        handle,
        CUBLAS_FILL_MODE_UPPER,     // Row-major lower = Column-major upper
        CUBLAS_OP_N,                // No transpose (because we want L^T)
        CUBLAS_DIAG_NON_UNIT,
        k_plus_1,
        d_L,
        max_k,
        d_x,        // Solution vector
        1
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Strsv (backward) failed: %d\n", status);
        return false;
    }
    
    // Step 7: Update residual r = y - Φ_S * x
    int m_blocks = (GPU_OPT_M + 255) / 256;
    reset_residual_cublas<<<m_blocks, 256, 0, stream>>>(d_r, d_y, GPU_OPT_M);
    
    int res_blocks = (k_plus_1 + 255) / 256;
    subtract_from_residual_cublas<GPU_OPT_D><<<res_blocks, 256, 0, stream>>>(
        d_r, d_idx, d_S, d_x, k_plus_1);
    
    return true;
}

#endif // CHOLESKY_CUBLAS_CUH

