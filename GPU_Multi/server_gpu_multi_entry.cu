#include <vector>
#include <thread>
#include <algorithm>
#include <cstdio>

#include <cuda_runtime.h>
#include <nccl.h>

#include "server_gpu_multi.h"

// Simple safety macros (or reuse your existing CUDA_SAFE/NCCL_SAFE)
#define CUDA_SAFE(call)                                                   \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            return;                                                       \
        }                                                                 \
    } while (0)

#define NCCL_SAFE(call)                                                   \
    do {                                                                  \
        ncclResult_t r = (call);                                          \
        if (r != ncclSuccess) {                                           \
            fprintf(stderr, "NCCL error %s:%d: %s\n",                     \
                    __FILE__, __LINE__, ncclGetErrorString(r));           \
            return;                                                       \
        }                                                                 \
    } while (0)

// This is the symbol that benchmark.c expects
extern "C"
void server_reconstruct_omp_gpu(int K_max, int flow_count)
{
    fprintf(stderr, "[multi-entry] K_max=%d, flow_count=%d\n", K_max, flow_count);

    int device_count = 0;
    CUDA_SAFE(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "[multi-entry] No CUDA devices found!\n");
        return;
    }

    // You can clamp to some max if you like
    int num_gpus = device_count;

    std::vector<int> devs(num_gpus);
    for (int i = 0; i < num_gpus; ++i) devs[i] = i;

    std::vector<ncclComm_t> comms(num_gpus);
    NCCL_SAFE(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    // Launch one CPU thread per GPU
    std::vector<std::thread> workers;
    workers.reserve(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        int dev_id = devs[i];
        ncclComm_t comm = comms[i];

        workers.emplace_back([=]() {
            fprintf(stderr, "[multi-entry] worker for device %d start\n", dev_id);
            CUDA_SAFE(cudaSetDevice(dev_id));
            CUDA_SAFE(cudaFree(0));  // force context creation

            server_reconstruct_omp_multi_gpu(
                K_max,
                flow_count,
                dev_id,
                num_gpus,
                comm
            );

            CUDA_SAFE(cudaDeviceSynchronize());
            fprintf(stderr, "[multi-entry] worker for device %d done\n", dev_id);
        });
    }

    for (auto &t : workers) {
        t.join();
    }

    for (auto &c : comms) {
        ncclCommDestroy(c);
    }

    fprintf(stderr, "[multi-entry] all workers done\n");
}
