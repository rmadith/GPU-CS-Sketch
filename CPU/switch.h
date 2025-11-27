#ifndef SWITCH_H
#define SWITCH_H

#include <stdint.h>

#define D 4
#define W 16384

#define BLOOM_FILTER_SIZE 1024
#define BLOOM_FILTER_HASH_COUNT 8

#if ((W & (W - 1)) != 0)
#error "W must be a power of two"
#endif

#if ((BLOOM_FILTER_SIZE & (BLOOM_FILTER_SIZE - 1)) != 0)
#error "BLOOM_FILTER_SIZE must be a power of two"
#endif

uint32_t cms_hash(int j, uint64_t key);
uint8_t update_and_check_new(uint64_t key, uint32_t weight);
extern uint32_t cms_sketch[D][W];
extern uint8_t bloom_filter_map[8][BLOOM_FILTER_SIZE];

#endif
