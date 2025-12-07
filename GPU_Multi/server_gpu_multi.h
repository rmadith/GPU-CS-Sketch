#ifndef SERVER_GPU_MULTI_H
#define SERVER_GPU_MULTI_H

// NCCL is required for multi-GPU communication context (ncclComm_t)
#include <nccl.h> 
// stdint is included to ensure uint64_t is available for the packed value/index
#include <stdint.h>

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
// Memory: N^2 bytes (10k flows = 100MB, 100k flows = 10GB!)
// Trades memory for O(1) overlap lookups during Cholesky
// WARNING: Disable for N > 10000 to avoid GPU OOM
#define GPU_OPT_PRECOMPUTE_OVERLAPS  0

// Use incremental K x K support overlap cache (int8)
// Memory: K^2 bytes (K=100 = 10KB, K=1000 = 1MB)
// Builds overlap cache incrementally as flows are selected
// Much more memory-efficient than full N x N precomputation
#define GPU_OPT_USE_SUPPORT_OVERLAP_CACHE 1

// Use incremental correlation updates via inverse index
// O(N*D/M * K) per iteration vs O(N*D) for full recompute
// Tracks coefficient DELTAS for all selected flows (not just new one)
#define GPU_OPT_USE_INCREMENTAL_CORR 1

// Use restricted search space for argmax (only scan flows in high-energy buckets)
// Reduces argmax from O(N) to O(K * bucket_load) per iteration
// Requires GPU_OPT_USE_INCREMENTAL_CORR to be enabled
#define GPU_OPT_USE_RESTRICTED_SEARCH 1

// Number of heavy buckets to consider for restricted search
// Higher values increase accuracy but reduce speedup
#define GPU_OPT_HEAVY_BUCKET_COUNT   32

// Periodic full scan interval (do full argmax every N iterations for safety)
// Set to 0 to disable periodic full scans (not recommended)
#define GPU_OPT_FULL_SCAN_INTERVAL   10

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
 * 
 * Multi-GPU-optimized OMP reconstruction for count-min sketch.
 * 
 * @param K_max      Maximum number of flows to select
 * @param flow_count Number of candidate flows (size of flow set Ω)
 * @param device_id
 * @param comm
 */
void server_reconstruct_omp_multi_gpu(int K_max, int flow_count, int device_id, int num_gpus, ncclComm_t comm);

#endif // SERVER_GPU_MULTI_H

