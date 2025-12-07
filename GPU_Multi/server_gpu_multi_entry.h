#ifndef SERVER_GPU_MULTI_SHIM_H
#define SERVER_GPU_MULTI_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

void server_reconstruct_omp_gpu(int K_max, int flow_count);

#ifdef __cplusplus
}
#endif

#endif
