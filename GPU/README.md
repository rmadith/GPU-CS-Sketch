# CUDA OMP Reconstruction

This directory holds a CUDA implementation of the OMP-based reconstruction. The
switch/CM-sketch stays on the CPU; only the sparse OMP steps run on the GPU.

## Build

Example standalone build from the repo root (needs CUDA toolkit and a GPU):

```
nvcc -O2 -std=c++14 -DUSE_CUDA \
  CPU/run.c CPU/server.c CPU/switch.c GPU/server_gpu.cu \
  -o gpu-run
```

Run `./gpu-run` to generate the sketch on CPU and reconstruct on GPU. The
existing CPU binaries keep working; define `USE_CUDA` to invoke the GPU path.
