#ifndef RUN_H
#define RUN_H

#include "server.h"
#include "switch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Global state
extern int num_flows;

// Flow generation utilities (exposed for benchmark)
uint64_t generate_flow_key(void);
uint32_t generate_flow_weight(void);
int find_or_add_flow(uint64_t key);
void reset_state(void);

typedef struct {
    int processed;
    int new_flows;
    uint64_t cms_total;
    uint64_t bloom_ones;
} RunStats;

RunStats run_once(uint32_t seed, int dump_arrays);

#endif
