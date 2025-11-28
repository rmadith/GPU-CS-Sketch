#include "server_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "../CPU/server.h"
#include "../CPU/switch.h"

#define CUDA_CHECK(expr)                                                     \
  do {                                                                      \
    cudaError_t _err = (expr);                                              \
    if (_err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(_err));                                   \
      goto cleanup;                                                         \
    }                                                                       \
  } while (0)

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

__global__ void correlate_gather(const float *r, const int *idx, int n,
                                 float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0f;
    int base = i * D;
#pragma unroll
    for (int j = 0; j < D; ++j) {
      int pos = idx[base + j];
      sum += r[pos];
    }
    c[i] = sum;
  }
}

// First reduction pass: takes |vals| and keeps the originating index (gid).
__global__ void argmax_abs_first(const float *vals, int n, float *out_val,
                                 int *out_idx) {
  extern __shared__ unsigned char smem[];
  float *s_val = reinterpret_cast<float *>(smem);
  int *s_idx = reinterpret_cast<int *>(s_val + blockDim.x);

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = -1.0f;
  int idx = -1;
  if (gid < n) {
    v = fabsf(vals[gid]);
    idx = gid;
  }

  s_val[threadIdx.x] = v;
  s_idx[threadIdx.x] = idx;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      float other = s_val[threadIdx.x + offset];
      int other_idx = s_idx[threadIdx.x + offset];
      if (other > s_val[threadIdx.x]) {
        s_val[threadIdx.x] = other;
        s_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_val[blockIdx.x] = s_val[0];
    out_idx[blockIdx.x] = s_idx[0];
  }
}

// Subsequent reduction passes: vals are already absolute, indices provided.
__global__ void argmax_reduce_next(const float *vals, const int *in_idx, int n,
                                   float *out_val, int *out_idx) {
  extern __shared__ unsigned char smem[];
  float *s_val = reinterpret_cast<float *>(smem);
  int *s_idx = reinterpret_cast<int *>(s_val + blockDim.x);

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = -1.0f;
  int idx = -1;
  if (gid < n) {
    v = vals[gid];
    idx = in_idx[gid];
  }

  s_val[threadIdx.x] = v;
  s_idx[threadIdx.x] = idx;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      float other = s_val[threadIdx.x + offset];
      int other_idx = s_idx[threadIdx.x + offset];
      if (other > s_val[threadIdx.x]) {
        s_val[threadIdx.x] = other;
        s_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_val[blockIdx.x] = s_val[0];
    out_idx[blockIdx.x] = s_idx[0];
  }
}

__global__ void l2_partial(const float *vals, int n, float *out) {
  extern __shared__ float sdata[];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 0.0f;
  if (gid < n) {
    float t = vals[gid];
    v = t * t;
  }
  sdata[threadIdx.x] = v;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

__global__ void reduce_sum(const float *vals, int n, float *out) {
  extern __shared__ float sdata[];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (gid < n) ? vals[gid] : 0.0f;
  sdata[threadIdx.x] = v;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

__global__ void update_active_and_residual(const float *y, float *r,
                                           const int *idx, int lambda, int *S,
                                           int k, float *L, float *x,
                                           int max_k) {
  extern __shared__ float shmem[];
  float *v = shmem;             // overlaps Φ_S^T φ_lambda
  float *w = v + max_k;         // forward-solve scratch
  float *b = w + max_k;         // Φ_S^T y
  float *t = b + max_k;         // forward solve result

  // r = y (parallel copy)
  for (int pos = threadIdx.x; pos < M; pos += blockDim.x) {
    r[pos] = y[pos];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    S[k] = lambda;
    int lambda_base = lambda * D;
    int lambda_rows[D];
#pragma unroll
    for (int j = 0; j < D; ++j) {
      lambda_rows[j] = idx[lambda_base + j];
    }

    for (int p = 0; p < k; ++p) {
      int other = S[p];
      int other_base = other * D;
      int overlap = 0;
#pragma unroll
      for (int j = 0; j < D; ++j) {
        if (lambda_rows[j] == idx[other_base + j]) {
          overlap++;
        }
      }
      v[p] = (float)overlap;
    }

    int unique_rows = D;
    for (int a = 0; a < D; ++a) {
      for (int b_idx = a + 1; b_idx < D; ++b_idx) {
        if (lambda_rows[a] == lambda_rows[b_idx]) {
          unique_rows--;
        }
      }
    }

    for (int i = 0; i < k; ++i) {
      float sum = v[i];
      for (int j = 0; j < i; ++j) {
        sum -= L[i * max_k + j] * w[j];
      }
      w[i] = sum / L[i * max_k + i];
    }

    float diag_sq = (float)unique_rows;
    for (int i = 0; i < k; ++i) {
      diag_sq -= w[i] * w[i];
    }
    if (diag_sq < 1e-9f) {
      diag_sq = 1e-9f;
    }
    float diag = sqrtf(diag_sq);

    for (int j = 0; j < k; ++j) {
      L[k * max_k + j] = w[j];
    }
    L[k * max_k + k] = diag;

    for (int p = 0; p <= k; ++p) {
      int flow_idx = S[p];
      int base = flow_idx * D;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < D; ++j) {
        sum += y[idx[base + j]];
      }
      b[p] = sum;
    }

    for (int i = 0; i <= k; ++i) {
      float sum = b[i];
      for (int j = 0; j < i; ++j) {
        sum -= L[i * max_k + j] * t[j];
      }
      t[i] = sum / L[i * max_k + i];
    }

    for (int i = k; i >= 0; --i) {
      float sum = t[i];
      for (int j = i + 1; j <= k; ++j) {
        sum -= L[j * max_k + i] * x[j];
      }
      x[i] = sum / L[i * max_k + i];
    }

    for (int p = 0; p <= k; ++p) {
      int flow_idx = S[p];
      int base = flow_idx * D;
      float coeff = x[p];
#pragma unroll
      for (int j = 0; j < D; ++j) {
        int pos = idx[base + j];
        r[pos] -= coeff;
      }
    }
  }
}

static bool device_argmax_abs(const float *d_vals, int n, float *scratch_val,
                              int *scratch_idx, int block_size, float *h_val,
                              int *h_idx) {
  const float *cur_vals = d_vals;
  const int *cur_idx = nullptr;
  int cur_n = n;
  bool first = true;

  while (true) {
    int blocks = div_up(cur_n, block_size);
    size_t shmem = (size_t)block_size * (sizeof(float) + sizeof(int));
    if (first) {
      argmax_abs_first<<<blocks, block_size, shmem>>>(cur_vals, cur_n,
                                                      scratch_val,
                                                      scratch_idx);
      first = false;
    } else {
      argmax_reduce_next<<<blocks, block_size, shmem>>>(cur_vals, cur_idx,
                                                        cur_n, scratch_val,
                                                        scratch_idx);
    }
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }
    if (blocks == 1) {
      break;
    }
    cur_vals = scratch_val;
    cur_idx = scratch_idx;
    cur_n = blocks;
  }

  if (cudaMemcpy(h_val, scratch_val, sizeof(float), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  if (cudaMemcpy(h_idx, scratch_idx, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  return true;
}

static bool device_l2_norm(const float *d_r, int m, float *scratch, int block,
                           float *h_out) {
  int blocks = div_up(m, block);
  size_t shmem = (size_t)block * sizeof(float);
  l2_partial<<<blocks, block, shmem>>>(d_r, m, scratch);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  int cur_n = blocks;
  const float *cur_vals = scratch;
  while (cur_n > 1) {
    int next_blocks = div_up(cur_n, block);
    reduce_sum<<<next_blocks, block, shmem>>>(cur_vals, cur_n, scratch);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }
    cur_vals = scratch;
    cur_n = next_blocks;
  }

  float sum = 0.0f;
  if (cudaMemcpy(&sum, cur_vals, sizeof(float), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  *h_out = sqrtf(sum);
  return true;
}

extern "C" void server_reconstruct_omp_gpu(int K_max, int flow_count) {
  if (flow_count <= 0) {
    return;
  }
  if (flow_count > N) {
    flow_count = N;
  }
  if (K_max > flow_count) {
    K_max = flow_count;
  }

  // host buffers
  std::vector<float> h_y(M);
  std::vector<int> h_idx(flow_count * D);
  std::vector<int> h_support(K_max, -1);
  std::vector<float> h_support_x(K_max, 0.0f);
  std::vector<float> h_r(M, 0.0f);

  for (int row = 0; row < D; ++row) {
    for (int col = 0; col < W; ++col) {
      int pos = row * W + col;
      h_y[pos] = (float)cms_sketch[row][col];
    }
  }

  for (int i = 0; i < M; ++i) {
    r[i] = (double)h_y[i];
  }

  for (int i = 0; i < flow_count; ++i) {
    for (int j = 0; j < D; ++j) {
      h_idx[i * D + j] = j * W + (int)flows[i].idx[j];
    }
  }

  // reset CPU-side outputs
  for (int i = 0; i < N; ++i) {
    selected[i] = 0;
    x[i] = 0.0;
  }

  // device buffers
  float *d_y = nullptr;
  float *d_r = nullptr;
  int *d_idx = nullptr;
  float *d_c = nullptr;
  int *d_S = nullptr;
  float *d_L = nullptr;
  float *d_x = nullptr;
  float *d_tmp_val = nullptr;
  int *d_tmp_idx = nullptr;
  float *d_tmp_norm = nullptr;

  size_t y_bytes = M * sizeof(float);
  size_t idx_bytes = (size_t)flow_count * D * sizeof(int);
  int corr_block = 256;
  int corr_grid = div_up(flow_count, corr_block);
  int reduce_block = 256;
  int tmp_len = div_up(flow_count, reduce_block);
  int norm_tmp_len = div_up(M, reduce_block);
  int support_size = 0;

  CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
  CUDA_CHECK(cudaMalloc(&d_r, y_bytes));
  CUDA_CHECK(cudaMalloc(&d_idx, idx_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, flow_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_S, K_max * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_L, (size_t)K_max * K_max * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, K_max * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp_val, tmp_len * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp_idx, tmp_len * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tmp_norm, norm_tmp_len * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), y_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r, h_y.data(), y_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), idx_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_L, 0, (size_t)K_max * K_max * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_x, 0, K_max * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_S, 0xff, K_max * sizeof(int)));

  while (support_size < K_max) {
    correlate_gather<<<corr_grid, corr_block>>>(d_r, d_idx, flow_count, d_c);
    CUDA_CHECK(cudaGetLastError());

    float best_val = 0.0f;
    int best_idx = -1;
    if (!device_argmax_abs(d_c, flow_count, d_tmp_val, d_tmp_idx,
                           reduce_block, &best_val, &best_idx)) {
      fprintf(stderr, "device_argmax_abs failed\n");
      break;
    }

    if (best_idx < 0 || best_val <= 1e-6f) {
      break;
    }

    size_t shmem = (size_t)K_max * 4 * sizeof(float);
    update_active_and_residual<<<1, 256, shmem>>>(
        d_y, d_r, d_idx, best_idx, d_S, support_size, d_L, d_x, K_max);
    CUDA_CHECK(cudaGetLastError());

    float norm_r = 0.0f;
    if (!device_l2_norm(d_r, M, d_tmp_norm, reduce_block, &norm_r)) {
      fprintf(stderr, "device_l2_norm failed\n");
      break;
    }

    support_size++;
    if (norm_r < 1e-6f) {
      break;
    }
  }

  if (support_size > 0) {
    CUDA_CHECK(cudaMemcpy(h_support.data(), d_S,
                          support_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_support_x.data(), d_x,
                          support_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(h_r.data(), d_r, y_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; ++i) {
      r[i] = (double)h_r[i];
    }

    for (int p = 0; p < support_size; ++p) {
      int fi = h_support[p];
      if (fi >= 0 && fi < N) {
        selected[fi] = 1;
        x[fi] = (double)h_support_x[p];
      }
    }
  }

cleanup:
  cudaFree(d_y);
  cudaFree(d_r);
  cudaFree(d_idx);
  cudaFree(d_c);
  cudaFree(d_S);
  cudaFree(d_L);
  cudaFree(d_x);
  cudaFree(d_tmp_val);
  cudaFree(d_tmp_idx);
  cudaFree(d_tmp_norm);
}
