#ifndef SERVER_GPU_OPT_H
#define SERVER_GPU_OPT_H

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// GPU-Optimized OMP Configuration for Count-Min Sketch
// =============================================================================

// Sketch parameters (must match CPU/switch.h)
#define GPU_OPT_D             4
#define GPU_OPT_W             16384
#define GPU_OPT_M             (GPU_OPT_D * GPU_OPT_W)

// Capacity limits
#define GPU_OPT_MAX_FLOWS     100000
#define GPU_OPT_MAX_SUPPORT   10000

// =============================================================================
// Feature Flags - Enable/disable optimizations
// =============================================================================

// Use precomputed N x N overlap matrix (int8)
// Memory: N^2 bytes (10k flows = 100MB)
// Trades memory for O(1) overlap lookups during Cholesky
#define GPU_OPT_PRECOMPUTE_OVERLAPS  1

// Use incremental correlation updates via inverse index
// O(N*D/M * K) per iteration vs O(N*D) for full recompute
// Tracks coefficient DELTAS for all selected flows (not just new one)
#define GPU_OPT_USE_INCREMENTAL_CORR 1

// =============================================================================
// Kernel Configuration
// =============================================================================

// Block size for correlation kernel (should be multiple of 32)
#define GPU_OPT_CORR_BLOCK_SIZE      256

// Block size for reduction kernels
#define GPU_OPT_REDUCE_BLOCK_SIZE    256

// =============================================================================
// Numerical Parameters
// =============================================================================

// Minimum diagonal value to prevent numerical instability in Cholesky
#define GPU_OPT_CHOL_MIN_DIAG        1e-9f

// Convergence threshold - stop when residual norm below this
#define GPU_OPT_RESIDUAL_THRESHOLD   1e-6f

// Minimum correlation to consider a flow significant
#define GPU_OPT_CORR_THRESHOLD       1e-6f

// =============================================================================
// API - Same signature as GPU/server_gpu.h for drop-in replacement
// =============================================================================

/**
 * GPU-optimized OMP reconstruction for count-min sketch.
 * 
 * @param K_max      Maximum number of flows to select
 * @param flow_count Number of candidate flows (size of flow set Ω)
 */
void server_reconstruct_omp_gpu(int K_max, int flow_count);

#ifdef __cplusplus
}
#endif

#endif // SERVER_GPU_OPT_H

