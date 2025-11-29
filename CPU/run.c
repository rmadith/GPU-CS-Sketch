#include "run.h"
#include <string.h>

#ifdef USE_CUDA
#include "../GPU/server_gpu.h"
#endif

int num_flows = 0; // how many distinct flows we've seen

int find_or_add_flow(uint64_t key) {
  // 1. look for existing flow
  for (int i = 0; i < num_flows; ++i) {
    if (flows[i].key == key) {
      return i;
    }
  }
  // 2. not found: create new entry
  int idx = num_flows++;
  flows[idx].key = key;
  for (int j = 0; j < D; ++j) {
    flows[idx].idx[j] = cms_hash(j, key); // SAME hash as switch
  }
  true_x[idx] = 0.0;
  return idx;
}

uint64_t generate_flow_key(void) {
  // combine three rand() calls to fill 64 bits
  uint64_t a = (uint64_t)(rand() & 0xffff);
  uint64_t b = (uint64_t)rand();
  uint64_t c = (uint64_t)(rand() & 0xffff);
  return (a << 48) ^ (b << 16) ^ c;
}

uint32_t generate_flow_weight(void) {
  return (uint32_t)(1 + (rand() % 100)); // weights in [1,100]
}

void reset_state(void) {
  num_flows = 0;
  memset(cms_sketch, 0, sizeof(cms_sketch));
  memset(bloom_filter_map, 0, sizeof(bloom_filter_map));
  memset(flows, 0, sizeof(flows));
}

static void dump_arrays(void) {
  // Dump cms_sketch
  for (int row = 0; row < D; ++row) {
    printf("cms_sketch row %d:\n", row);
    for (int col = 0; col < W; ++col) {
      printf("%u%c", cms_sketch[row][col], (col == W - 1) ? '\n' : ' ');
    }
  }

  // Dump bloom filter map
  for (int i = 0; i < BLOOM_FILTER_HASH_COUNT; ++i) {
    printf("bloom_filter_map hash %d:\n", i);
    for (int j = 0; j < BLOOM_FILTER_SIZE; ++j) {
      printf("%u%c", bloom_filter_map[i][j],
             (j == BLOOM_FILTER_SIZE - 1) ? '\n' : ' ');
    }
  }
}

RunStats run_once(uint32_t seed, int dump_arrays_flag) {
  reset_state();
  srand(seed);

  int new_flows = 0;

  for (int i = 0; i < N; i++) {
    uint64_t key = generate_flow_key();
    uint32_t weight = generate_flow_weight();

    uint8_t seen = update_and_check_new(key, weight);
    int idx = find_or_add_flow(key);
    true_x[idx] += (double)weight;
    if (seen == 0) {
      new_flows++;
    }

    // keep a copy of what we sent for reference
    flows[i].key = key;
    for (int j = 0; j < D; ++j) {
      flows[i].idx[j] = cms_hash(j, key);
    }
  }

  if (dump_arrays_flag) {
    printf("Processed %d keys (%d new)\n", N, new_flows);
    dump_arrays();
  }

  // compute simple checksums for testing
  uint64_t cms_total = 0;
  for (int row = 0; row < D; ++row) {
    for (int col = 0; col < W; ++col) {
      cms_total += cms_sketch[row][col];
    }
  }

  uint64_t bloom_ones = 0;
  for (int i = 0; i < BLOOM_FILTER_HASH_COUNT; ++i) {
    for (int j = 0; j < BLOOM_FILTER_SIZE; ++j) {
      bloom_ones += bloom_filter_map[i][j] ? 1 : 0;
    }
  }

  RunStats stats = {.processed = N,
                    .new_flows = new_flows,
                    .cms_total = cms_total,
                    .bloom_ones = bloom_ones};
  return stats;
}

int main(void) {
  // Fixed seed for deterministic runs; set dump flag to 1 to mirror previous
  // output
  RunStats stats = run_once(12345u, 0);
  printf("Processed %d keys (%d new)\n", stats.processed, stats.new_flows);
  printf("cms_total=%llu bloom_ones=%llu\n",
         (unsigned long long)stats.cms_total,
         (unsigned long long)stats.bloom_ones);
#ifdef USE_CUDA
  server_reconstruct_omp_gpu(100, num_flows);
#else
  server_reconstruct_omp(100);
#endif
  for (int i = 0; i < num_flows; ++i) {
    if (!selected[i]) continue;
    double t = true_x[i];
    double e = x[i]; // reconstructed from OMP
    double abs_err = e - t;
    double rel_err = (t != 0.0) ? fabs(abs_err) / t : 0.0;

    printf("flow %d key=%llu  true=%8.2f  est=%8.2f  abs_err=%8.2f  "
           "rel_err=%6.2f%%\n",
           i, (unsigned long long)flows[i].key, t, e, abs_err, rel_err * 100.0);
  }
  double mse = 0.0;
  for (int i = 0; i < num_flows; ++i) {
    if (!selected[i]) continue;
    double d = x[i] - true_x[i];
    mse += d * d;
  }
  mse /= (double)num_flows;
  printf("MSE over flows = %f\n", mse);

  return stats.processed;
}
