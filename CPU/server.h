#ifndef SERVER_H
#define SERVER_H

#include "switch.h"

#define M (D * W)
#define N 100 // number of flows


extern double true_x[N];   // true flow weights for testing

typedef struct {
  uint64_t key;    // flow key
  uint32_t idx[D]; // D positions in y/r
} FlowPattern;

double y[M];          // flattened sketch (from switch)
double r[M];          // residual
FlowPattern flows[N]; // built from keys
double x[N];          // flow sizes (output)
int selected[N];      // 0/1 flags


void server_reconstruct_omp(int K_max);

#endif