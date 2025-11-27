#ifndef RUN_H
#define RUN_H

#include "server.h"
#include "switch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    int processed;
    int new_flows;
    uint64_t cms_total;
    uint64_t bloom_ones;
} RunStats;

RunStats run_once(uint32_t seed, int dump_arrays);

#endif
