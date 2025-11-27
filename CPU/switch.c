#include "switch.h"
#include <stdint.h>

// Initialize the 2D sketch
uint32_t cms_sketch[D][W] = {0};

// Simple per-row seeded mix; W is a power of two so we can mask
static const uint64_t cms_seeds[D] = {
    0x9e3779b185ebca87ULL, 0xc2b2ae3d27d4eb4fULL,
    0x165667b19e3779f9ULL, 0xd6e8feb86659fd93ULL
};

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Deterministic row-specific hash mapped into [0, W)
uint32_t cms_hash(int j, uint64_t key) {
    uint64_t h = mix64(key ^ cms_seeds[j % D]);
    return (uint32_t)(h & (W - 1));
}

// Update hash every time a packet is received
void cms_update( uint64_t key, uint32_t weight) {
    for (int j = 0; j < D; j++) {
        uint32_t col = cms_hash(j, key);
        cms_sketch[j][col] += weight;
    }
}

// Use a bloom filter if it is a new flow. We will use a different map for this.
uint8_t bloom_filter_map[8][BLOOM_FILTER_SIZE] = {0};

static const uint64_t bloom_seeds[BLOOM_FILTER_HASH_COUNT] = {
    0x27d4eb2f165667c5ULL, 0x94d049bb133111ebULL, 0xbf58476d1ce4e5b9ULL,
    0x94d049bb133111ebULL ^ 0x9e3779b185ebca87ULL,
    0xd6e8feb86659fd93ULL, 0xa5b85c5e198ed849ULL,
    0x165667b19e3779f9ULL ^ 0xc2b2ae3d27d4eb4fULL,
    0xff51afd7ed558ccdULL
};

static inline uint32_t bloom_filter_hash(uint64_t key, int i) {
    uint64_t h = mix64(key ^ bloom_seeds[i % BLOOM_FILTER_HASH_COUNT]);
    return (uint32_t)(h & (BLOOM_FILTER_SIZE - 1));
}

void bloom_filter_update(uint64_t key) {
    for (int i = 0; i < BLOOM_FILTER_HASH_COUNT; i++) {
        uint32_t hash = bloom_filter_hash(key, i);
        bloom_filter_map[i][hash] = 1;
    }
}

uint8_t bloom_filter_query(uint64_t key) {
    for (int i = 0; i < BLOOM_FILTER_HASH_COUNT; i++) {
        uint32_t hash = bloom_filter_hash(key, i);
        if (bloom_filter_map[i][hash] == 0) {
            // Update the bloom filter
            bloom_filter_update(key);
            return 0; // Probably not in the flow
        }
    }
    return 1;
}

// Update sketch and check if packet is new
uint8_t update_and_check_new(uint64_t key, uint32_t weight) {
    cms_update(key, weight);
    return bloom_filter_query(key);
}