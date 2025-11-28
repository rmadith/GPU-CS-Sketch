#ifndef SERVER_GPU_H
#define SERVER_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

// GPU OMP reconstruction entry point. `flow_count` is the number of
// candidate flows (size of the flow-key set Ω) to consider on the device.
void server_reconstruct_omp_gpu(int K_max, int flow_count);

#ifdef __cplusplus
}
#endif

#endif
