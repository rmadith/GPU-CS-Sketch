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

// Required for clock_gettime and snprintf with -std=c11
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

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

// Hash table for fast flow lookup during generation
#define HASH_TABLE_SIZE (N * 2)
static int flow_hash_table[HASH_TABLE_SIZE];  // -1 = empty, else flow index

// Cache directory for saved states
#define CACHE_DIR "benchmark/.flow_cache"

// Global state
static int num_flows = 0;
static int disable_cache = 0;  // Set to 1 to disable disk caching

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
 * Initialize hash table for flow lookup
 */
static void init_flow_hash_table(void) {
    memset(flow_hash_table, 0xFF, sizeof(flow_hash_table));  // Fill with -1
}

/**
 * Find or add a flow to the flow table (with hash table for O(1) lookup)
 */
static int find_or_add_flow(uint64_t key) {
    // Hash the key (mix high and low bits)
    uint32_t hash = (uint32_t)((key ^ (key >> 32)) % HASH_TABLE_SIZE);
    
    // Linear probing to handle collisions
    for (int probe = 0; probe < HASH_TABLE_SIZE; probe++) {
        uint32_t idx = (hash + probe) % HASH_TABLE_SIZE;
        int flow_idx = flow_hash_table[idx];
        
        if (flow_idx == -1) {
            // Empty slot, add new flow
            flow_idx = num_flows++;
            flow_hash_table[idx] = flow_idx;
            flows[flow_idx].key = key;
            for (int j = 0; j < D; ++j) {
                flows[flow_idx].idx[j] = cms_hash(j, key);
            }
            true_x[flow_idx] = 0.0;
            return flow_idx;
        } else if (flows[flow_idx].key == key) {
            // Found existing flow
            return flow_idx;
        }
        // Collision, continue probing
    }
    
    // Table full (should never happen if sized correctly)
    fprintf(stderr, "ERROR: Hash table full! Increase HASH_TABLE_SIZE\n");
    exit(1);
}

/**
 * Reset all state for a fresh benchmark run
 */
static void reset_benchmark_state(void) {
    fprintf(stderr, "  [State Reset] Clearing all data structures...\n");
    num_flows = 0;
    init_flow_hash_table();
    memset(cms_sketch, 0, sizeof(cms_sketch));
    memset(bloom_filter_map, 0, sizeof(bloom_filter_map));
    memset(flows, 0, sizeof(flows));
    memset(true_x, 0, sizeof(true_x));
    memset(x, 0, sizeof(x));
    memset(selected, 0, sizeof(selected));
    fprintf(stderr, "  [State Reset] Reset complete\n");
}

/**
 * Create cache directory if it doesn't exist
 */
static void ensure_cache_dir(void) {
    struct stat st = {0};
    if (stat(CACHE_DIR, &st) == -1) {
        #ifdef _WIN32
            mkdir(CACHE_DIR);
        #else
            mkdir(CACHE_DIR, 0755);
        #endif
    }
}

/**
 * Generate cache filename for a specific configuration
 */
static void get_cache_filename(char* buffer, size_t size, DistributionType dist, uint32_t seed) {
    snprintf(buffer, size, "%s/flows_n%d_%s_seed%u.bin", 
             CACHE_DIR, N, distribution_names[dist], seed);
}

/**
 * Save generated state to cache file
 */
static void save_state_to_cache(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "  [Cache] Warning: Could not create cache file: %s\n", filename);
        return;
    }
    
    // Write state
    fwrite(&num_flows, sizeof(num_flows), 1, f);
    fwrite(cms_sketch, sizeof(cms_sketch), 1, f);
    fwrite(flows, sizeof(FlowPattern) * num_flows, 1, f);  // Only write used flows
    fwrite(true_x, sizeof(double) * num_flows, 1, f);      // Only write used true_x
    
    fclose(f);
    fprintf(stderr, "  [Cache] Saved state to: %s\n", filename);
}

/**
 * Load generated state from cache file
 * Returns 1 on success, 0 if cache doesn't exist or is invalid
 */
static int load_state_from_cache(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        return 0;  // Cache doesn't exist
    }
    
    // Read state
    size_t read_count = 0;
    read_count += fread(&num_flows, sizeof(num_flows), 1, f);
    read_count += fread(cms_sketch, sizeof(cms_sketch), 1, f);
    read_count += fread(flows, sizeof(FlowPattern) * num_flows, 1, f);
    read_count += fread(true_x, sizeof(double) * num_flows, 1, f);
    
    fclose(f);
    
    if (read_count != 4) {
        fprintf(stderr, "  [Cache] Warning: Corrupted cache file, regenerating...\n");
        return 0;
    }
    
    fprintf(stderr, "  [Cache] ✓ Loaded from cache: %s (%d unique flows)\n", 
            filename, num_flows);
    return 1;
}

/**
 * Generate flows with specified distribution (with caching)
 */
static void generate_flows(DistributionType dist, uint32_t seed) {
    fprintf(stderr, "  [Flow Generation] Starting for %s distribution (seed=%u)...\n", 
            distribution_names[dist], seed);
    
    // Try to load from cache first (unless disabled)
    char cache_file[512];
    if (!disable_cache) {
        ensure_cache_dir();
        get_cache_filename(cache_file, sizeof(cache_file), dist, seed);
        
        if (load_state_from_cache(cache_file)) {
            return;  // Successfully loaded from cache
        }
        
        // Cache miss - generate flows
        fprintf(stderr, "  [Flow Generation] Cache miss, generating from scratch...\n");
    } else {
        fprintf(stderr, "  [Flow Generation] Cache disabled, generating from scratch...\n");
    }
    srand(seed);
    init_flow_hash_table();
    
    // Progress tracking
    int progress_interval = N / 10; // Report every 10%
    if (progress_interval < 1) progress_interval = 1;
    
    uint64_t total_weight = 0;
    
    for (int i = 0; i < N; ++i) {
        uint64_t key = gen_flow_key();
        uint32_t weight = generate_weight_for_distribution(dist, i, N);
        total_weight += weight;
        
        // Update CMS sketch
        for (int j = 0; j < D; ++j) {
            uint32_t col = cms_hash(j, key);
            cms_sketch[j][col] += weight;
        }
        
        // Track flow (now O(1) with hash table!)
        int idx = find_or_add_flow(key);
        true_x[idx] += (double)weight;
        
        // Progress reporting
        if ((i + 1) % progress_interval == 0 || i == N - 1) {
            double percent = ((double)(i + 1) / N) * 100.0;
            fprintf(stderr, "  [Flow Generation] Progress: %.0f%% (%d/%d packets, %d unique flows)\n", 
                    percent, i + 1, N, num_flows);
        }
    }
    
    // Summary statistics
    double avg_weight = (double)total_weight / N;
    fprintf(stderr, "  [Flow Generation] Complete: %d packets, %d unique flows, avg weight: %.2f\n",
            N, num_flows, avg_weight);
    
    // Save to cache for future runs (unless disabled)
    if (!disable_cache) {
        save_state_to_cache(cache_file);
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
    fprintf(stderr, "\n[Benchmark] Starting: %s distribution, K=%d, impl=%s\n",
            distribution_names[dist], k_max, impl_name);
    
    double times[NUM_ITERATIONS];
    double total_time = 0.0;
    
    // Warm-up runs (not timed, no output)
    if (WARMUP_ITERATIONS > 0) {
        fprintf(stderr, "[Warmup] Running %d warmup iteration(s)...\n", WARMUP_ITERATIONS);
        for (int w = 0; w < WARMUP_ITERATIONS; ++w) {
            fprintf(stderr, "[Warmup] Iteration %d/%d\n", w + 1, WARMUP_ITERATIONS);
            reset_benchmark_state();
            generate_flows(dist, 12345u + w);
            
            fprintf(stderr, "  [Warmup] Executing reconstruction algorithm...\n");
#ifdef USE_CUDA
            cudaDeviceSynchronize();
            server_reconstruct_omp_gpu(k_max, num_flows);
            cudaDeviceSynchronize();
#else
            server_reconstruct_omp(k_max);
#endif
            fprintf(stderr, "  [Warmup] Iteration %d complete\n", w + 1);
        }
    }
    
    // Timed runs
    fprintf(stderr, "[Timed Run] Starting %d timed iteration(s)...\n", NUM_ITERATIONS);
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        fprintf(stderr, "[Iteration %d/%d] Resetting state...\n", 
                iter + 1, NUM_ITERATIONS);
        reset_benchmark_state();
        generate_flows(dist, 12345u + WARMUP_ITERATIONS + iter);
        
#ifdef USE_CUDA
        // GPU: Sync before timing to ensure no pending work
        fprintf(stderr, "  [Iteration %d/%d] Synchronizing GPU...\n", 
                iter + 1, NUM_ITERATIONS);
        cudaDeviceSynchronize();
#endif
        
        fprintf(stderr, "  [Iteration %d/%d] Starting timed execution...\n", 
                iter + 1, NUM_ITERATIONS);
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
        
        fprintf(stderr, "  [Iteration %d/%d] Complete: %.3f ms\n", 
                iter + 1, NUM_ITERATIONS, times[iter]);
    }
    
    // Compute statistics
    double mean_time = total_time / NUM_ITERATIONS;
    double stddev = compute_stddev(times, NUM_ITERATIONS, mean_time);
    double throughput = (double)N / (mean_time / 1000.0); // flows/sec
    
    fprintf(stderr, "[Benchmark] Results: mean=%.3f ms, stddev=%.3f ms, throughput=%.2f M flows/sec\n",
            mean_time, stddev, throughput / 1e6);
    
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
    fprintf(stderr, "  --no-cache       Disable disk caching (force regeneration)\n");
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
        } else if (strcmp(argv[i], "--no-cache") == 0) {
            disable_cache = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Print header
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "OMP BENCHMARK HARNESS\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  N (packets):     %d\n", N);
    fprintf(stderr, "  K_max:           %d\n", k_max);
    fprintf(stderr, "  Implementation:  %s\n", impl_name);
    fprintf(stderr, "  Iterations:      %d\n", NUM_ITERATIONS);
    fprintf(stderr, "  Warmup:          %d\n", WARMUP_ITERATIONS);
    fprintf(stderr, "  CMS dimensions:  D=%d, W=%d\n", D, W);
    
#ifdef USE_CUDA
    fprintf(stderr, "  Mode:            GPU (CUDA)\n");
#elif USE_OPENMP
    fprintf(stderr, "  Mode:            CPU (OpenMP)\n");
#else
    fprintf(stderr, "  Mode:            CPU (Single-threaded)\n");
#endif
    
    fprintf(stderr, "  Flow cache:      %s\n", disable_cache ? "Disabled" : "Enabled");
    
    fprintf(stderr, "========================================\n");
    
    // Run benchmarks
    if (run_all_dists) {
        fprintf(stderr, "Running all %d distributions...\n\n", DIST_COUNT);
        for (int d = 0; d < DIST_COUNT; ++d) {
            fprintf(stderr, "\n>>> Distribution %d/%d: %s\n", 
                    d + 1, DIST_COUNT, distribution_names[d]);
            run_benchmark((DistributionType)d, k_max, impl_name);
        }
    } else {
        fprintf(stderr, "Running single distribution: %s\n\n", distribution_names[dist]);
        run_benchmark(dist, k_max, impl_name);
    }
    
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "BENCHMARK COMPLETE\n");
    fprintf(stderr, "========================================\n");
    
    return 0;
}

