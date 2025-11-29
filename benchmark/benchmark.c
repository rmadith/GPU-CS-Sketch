/**
 * Benchmark Harness for OMP Implementations
 * 
 * Compares CPU (single-threaded), CPU-OMP, GPU, and GPU-optimized implementations
 * across different flow distributions and sizes.
 * 
 * Build commands:
 *   CPU:     gcc -O3 -march=native -std=c11 -DN=<N> benchmark/benchmark.c CPU/switch.c CPU/server.c -o build/cpu-bench -lm
 *   CPU-OMP: gcc -O3 -march=native -fopenmp -std=c11 -DN=<N> -DUSE_OPENMP benchmark/benchmark.c CPU/switch.c CPU/server.c -o build/cpu-omp-bench -lm
 *   GPU:     nvcc -O3 --use_fast_math -std=c++14 -DUSE_CUDA -DN=<N> -gencode arch=compute_75,code=sm_75 benchmark/benchmark.c CPU/server.c CPU/switch.c GPU/server_gpu.cu -o build/gpu-bench
 *   GPU-Opt: nvcc -O3 --use_fast_math -std=c++14 -DUSE_CUDA -DN=<N> -gencode arch=compute_75,code=sm_75 benchmark/benchmark.c CPU/server.c CPU/switch.c GPU_Optimized/server_gpu_opt.cu -o build/gpu-opt-bench
 */

// Required for clock_gettime on Linux with -std=c11
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "../CPU/server.h"
#include "../CPU/switch.h"
#include "flow_generator.h"

#ifdef USE_CUDA
// Forward declarations for CUDA functions
extern void server_reconstruct_omp_gpu(int K_max, int flow_count);

// CUDA runtime for synchronization
#include <cuda_runtime.h>
#endif

// Number of benchmark iterations
#define NUM_ITERATIONS 10
#define WARMUP_ITERATIONS 2

// Global state
static int num_flows = 0;

/**
 * Generate a random 64-bit flow key
 */
static uint64_t gen_flow_key(void) {
    uint64_t a = (uint64_t)(rand() & 0xffff);
    uint64_t b = (uint64_t)rand();
    uint64_t c = (uint64_t)(rand() & 0xffff);
    return (a << 48) ^ (b << 16) ^ c;
}

/**
 * Find or add a flow to the flow table
 */
static int find_or_add_flow(uint64_t key) {
    for (int i = 0; i < num_flows; ++i) {
        if (flows[i].key == key) {
            return i;
        }
    }
    int idx = num_flows++;
    flows[idx].key = key;
    for (int j = 0; j < D; ++j) {
        flows[idx].idx[j] = cms_hash(j, key);
    }
    true_x[idx] = 0.0;
    return idx;
}

/**
 * Reset all state for a fresh benchmark run
 */
static void reset_benchmark_state(void) {
    num_flows = 0;
    memset(cms_sketch, 0, sizeof(cms_sketch));
    memset(bloom_filter_map, 0, sizeof(bloom_filter_map));
    memset(flows, 0, sizeof(flows));
    memset(true_x, 0, sizeof(true_x));
    memset(x, 0, sizeof(x));
    memset(selected, 0, sizeof(selected));
}

/**
 * Generate flows with specified distribution
 */
static void generate_flows(DistributionType dist, uint32_t seed) {
    srand(seed);
    
    for (int i = 0; i < N; ++i) {
        uint64_t key = gen_flow_key();
        uint32_t weight = generate_weight_for_distribution(dist, i, N);
        
        // Update CMS sketch
        for (int j = 0; j < D; ++j) {
            uint32_t col = cms_hash(j, key);
            cms_sketch[j][col] += weight;
        }
        
        // Track flow
        int idx = find_or_add_flow(key);
        true_x[idx] += (double)weight;
    }
}

/**
 * Get current time in milliseconds (high resolution)
 */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/**
 * Compute standard deviation from samples
 */
static double compute_stddev(const double* samples, int count, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < count; ++i) {
        double diff = samples[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (double)count);
}

/**
 * Run benchmark for a single configuration
 */
static void run_benchmark(
    DistributionType dist,
    int k_max,
    const char* impl_name
) {
    double times[NUM_ITERATIONS];
    double total_time = 0.0;
    
    // Warm-up runs (not timed, no output)
    for (int w = 0; w < WARMUP_ITERATIONS; ++w) {
        reset_benchmark_state();
        generate_flows(dist, 12345u + w);
        
#ifdef USE_CUDA
        cudaDeviceSynchronize();
        server_reconstruct_omp_gpu(k_max, num_flows);
        cudaDeviceSynchronize();
#else
        server_reconstruct_omp(k_max);
#endif
    }
    
    // Timed runs
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        reset_benchmark_state();
        generate_flows(dist, 12345u + WARMUP_ITERATIONS + iter);
        
#ifdef USE_CUDA
        // GPU: Sync before timing to ensure no pending work
        cudaDeviceSynchronize();
#endif
        
        double start = get_time_ms();
        
#ifdef USE_CUDA
        server_reconstruct_omp_gpu(k_max, num_flows);
        // Sync after to ensure kernel has completed
        cudaDeviceSynchronize();
#else
        server_reconstruct_omp(k_max);
#endif
        
        double end = get_time_ms();
        times[iter] = end - start;
        total_time += times[iter];
    }
    
    // Compute statistics
    double mean_time = total_time / NUM_ITERATIONS;
    double stddev = compute_stddev(times, NUM_ITERATIONS, mean_time);
    double throughput = (double)N / (mean_time / 1000.0); // flows/sec
    
    // Output in CSV-parseable format
    // RESULT,<dist>,<N>,<K>,<impl>,<time_ms>,<stddev_ms>,<throughput>
    printf("RESULT,%s,%d,%d,%s,%.3f,%.3f,%.0f\n",
           distribution_names[dist],
           N,
           k_max,
           impl_name,
           mean_time,
           stddev,
           throughput);
    
    fflush(stdout);
}

/**
 * Print usage information
 */
static void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", prog_name);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --dist <type>    Distribution type (default: all)\n");
    fprintf(stderr, "                   Types: uniform, zipf, heavy_elephant, heavy_mice, bimodal, all\n");
    fprintf(stderr, "  --k <value>      K_max value (default: 100)\n");
    fprintf(stderr, "  --impl <name>    Implementation name for output (default: cpu)\n");
    fprintf(stderr, "  --help           Show this help\n");
    fprintf(stderr, "\nOutput format:\n");
    fprintf(stderr, "  RESULT,<dist>,<N>,<K>,<impl>,<time_ms>,<stddev_ms>,<throughput>\n");
}

int main(int argc, char* argv[]) {
    // Default parameters
    DistributionType dist = DIST_UNIFORM;
    int run_all_dists = 1;
    int k_max = 100;
    const char* impl_name = "cpu";
    
#ifdef USE_OPENMP
    impl_name = "cpu_omp";
#endif
#ifdef USE_CUDA
    impl_name = "gpu";
#endif
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dist") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "all") == 0) {
                run_all_dists = 1;
            } else {
                run_all_dists = 0;
                dist = parse_distribution(argv[i]);
            }
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            i++;
            k_max = atoi(argv[i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            i++;
            impl_name = argv[i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Print header
    fprintf(stderr, "Benchmark: N=%d, K_max=%d, impl=%s\n", N, k_max, impl_name);
    fprintf(stderr, "Iterations: %d (+ %d warmup)\n", NUM_ITERATIONS, WARMUP_ITERATIONS);
    fprintf(stderr, "---\n");
    
    // Run benchmarks
    if (run_all_dists) {
        for (int d = 0; d < DIST_COUNT; ++d) {
            run_benchmark((DistributionType)d, k_max, impl_name);
        }
    } else {
        run_benchmark(dist, k_max, impl_name);
    }
    
    return 0;
}

