#include "server.h"
#include <math.h>    // fabs, sqrt

#ifdef USE_OPENMP
#include <omp.h>
#endif


// server.c
double true_x[N];
double y[M];          // flattened sketch (from switch)
double r[M];          // residual
FlowPattern flows[N]; // built from keys
double x[N];          // flow sizes (output)
int selected[N];      // 0/1 flags



// --------- helper: flatten cms_sketch into y ---------

void server_flatten_cms_to_y(void) {
    // cms_sketch[row][col] -> y[row * W + col]
#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int row = 0; row < D; ++row) {
        for (int col = 0; col < W; ++col) {
            int idx_flat = row * W + col;
            y[idx_flat] = (double)cms_sketch[row][col];
        }
    }
}

// --------- low-level helpers: indexing & dot products ---------

// φ_i^T r = sum of residual at the D counters where flow i lives.
static double flow_score_from_residual(const double *residual,
                                       const FlowPattern *f) {
    double score = 0.0;
    for (int row = 0; row < D; ++row) {
        uint32_t col = f->idx[row];          // column index in this row
        int pos = row * W + (int)col;        // flattened index
        score += residual[pos];
    }
    return score;
}

// φ_i^T y = sum of original sketch at the D counters for flow i.
static double flow_dot_y(const FlowPattern *f) {
    double sum = 0.0;
    for (int row = 0; row < D; ++row) {
        uint32_t col = f->idx[row];
        int pos = row * W + (int)col;
        sum += y[pos];
    }
    return sum;
}

// φ_i^T φ_j = number of shared counters between flows i and j.
// In this sketch, that's how many rows where they hash to the same column.
static double flow_overlap(const FlowPattern *fi, const FlowPattern *fj) {
    int overlap = 0;
    for (int row = 0; row < D; ++row) {
        if (fi->idx[row] == fj->idx[row]) {
            overlap++;
        }
    }
    return (double)overlap;  // integer in [0, D]
}

// --------- Cholesky factorization + solve ---------

// In-place Cholesky decomposition: A = L * L^T, L stored in lower triangle of A.
// Returns 1 on success, 0 if matrix is not positive definite.
static int cholesky_decompose(int n, double A[n][n]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];

            for (int k = 0; k < j; ++k) {
                sum -= A[i][k] * A[j][k];
            }

            if (i == j) {
                if (sum <= 1e-12) {  // not positive definite / numerical issue
                    return 0;
                }
                A[i][j] = sqrt(sum);
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
        // upper triangle not used; can optionally zero it out
        for (int j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    return 1;
}

// Solve A_orig * x = b using Cholesky factorization of A_orig.
// A will be overwritten by its Cholesky factor L in lower triangle.
// Returns 1 on success, 0 on failure.
static int cholesky_solve(int n, double A[n][n],
                          const double *b, double *x) {
    if (!cholesky_decompose(n, A)) {
        return 0;
    }

    double y_tmp[n];

    // Forward solve: L * y = b
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int k = 0; k < i; ++k) {
            sum -= A[i][k] * y_tmp[k];
        }
        y_tmp[i] = sum / A[i][i];
    }

    // Backward solve: L^T * x = y
    for (int i = n - 1; i >= 0; --i) {
        double sum = y_tmp[i];
        for (int k = i + 1; k < n; ++k) {
            sum -= A[k][i] * x[k];
        }
        x[i] = sum / A[i][i];
    }

    return 1;
}

// --------- OMP selection helpers ---------

// Select the index of the best next flow according to |φ_i^T r|.
// Returns -1 if no unselected flow has a significant score.
static int select_best_flow_index(void) {
    int best_i = -1;
    double best_score = 0.0;

#ifdef USE_OPENMP
    // Parallel argmax: each thread finds local best, then reduce
    #pragma omp parallel
    {
        int local_best_i = -1;
        double local_best_score = 0.0;
        
        #pragma omp for nowait
        for (int i = 0; i < N; ++i) {
            if (selected[i]) {
                continue;
            }
            double s = flow_score_from_residual(r, &flows[i]);
            double mag = fabs(s);
            if (mag > local_best_score) {
                local_best_score = mag;
                local_best_i = i;
            }
        }
        
        #pragma omp critical
        {
            if (local_best_score > best_score) {
                best_score = local_best_score;
                best_i = local_best_i;
            }
        }
    }
#else
    for (int i = 0; i < N; ++i) {
        if (selected[i]) {
            continue; // already in support set
        }

        double s = flow_score_from_residual(r, &flows[i]);
        double mag = fabs(s);

        if (mag > best_score) {
            best_score = mag;
            best_i = i;
        }
    }
#endif

    // Optional: if best score is extremely small, we consider nothing left to explain
    if (best_i >= 0 && best_score <= 1e-9) {
        return -1;
    }

    return best_i;
}

// Recompute residual: r = y - Φ_S x_S,
// where S is the current support of size k and support_indices[p] = flow index in S.
static void recompute_residual_from_support(const int *support_indices, int k) {
    // start with r = y
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int m = 0; m < M; ++m) {
        r[m] = y[m];
    }

    // subtract contributions of all selected flows
    // Note: This loop has potential race conditions on r[pos], so we keep it sequential
    // or use atomic updates. For small k, sequential is fine.
    for (int p = 0; p < k; ++p) {
        int fi = support_indices[p];
        double coeff = x[fi];
        const FlowPattern *f = &flows[fi];

        for (int row = 0; row < D; ++row) {
            uint32_t col = f->idx[row];
            int pos = row * W + (int)col;
            r[pos] -= coeff;
        }
    }
}

// Optional stopping rule: L2 norm of residual
static double residual_l2_norm(void) {
    double sum_sq = 0.0;
#ifdef USE_OPENMP
    #pragma omp parallel for reduction(+:sum_sq)
#endif
    for (int m = 0; m < M; ++m) {
        double v = r[m];
        sum_sq += v * v;
    }
    return sqrt(sum_sq);
}

// --------- main reconstruction (OMP + Cholesky) ---------

// Full OMP-style reconstruction using Cholesky to solve
// (Φ_S^T Φ_S) a = Φ_S^T y for the selected set S.
// K_max = maximum number of flows to select in the support.
void server_reconstruct_omp(int K_max) {
    if (K_max > N) {
        K_max = N;
    }

    // 1) Flatten sketch into y
    server_flatten_cms_to_y();

    // 2) Initialize x, selected, and residual
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; ++i) {
        x[i] = 0.0;
        selected[i] = 0;
    }
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int m = 0; m < M; ++m) {
        r[m] = y[m];
    }

    // Support set: indices of selected flows
    int support_indices[K_max];
    int k = 0; // current support size

    for (; k < K_max; ++k) {
        // ---- Step 1: pick best flow (largest |φ_i^T r|) ----
        int best_i = select_best_flow_index();
        if (best_i < 0) {
            // nothing useful left
            break;
        }

        selected[best_i] = 1;
        support_indices[k] = best_i;
        int cur_k = k + 1; // size of support AFTER this addition

        // ---- Step 2: build normal equations (Φ_S^T Φ_S) a = Φ_S^T y ----

        // G = Φ_S^T Φ_S   (cur_k x cur_k)
        double G[cur_k][cur_k];
        // b = Φ_S^T y     (cur_k)
        double b[cur_k];
        // a = solution coefficients for support flows
        double a[cur_k];

        // Build G: G[p][q] = φ_{i_p}^T φ_{i_q} = overlap of patterns
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (int p = 0; p < cur_k; ++p) {
            int ip = support_indices[p];
            const FlowPattern *fp = &flows[ip];
            for (int q = 0; q < cur_k; ++q) {
                int iq = support_indices[q];
                const FlowPattern *fq = &flows[iq];
                G[p][q] = flow_overlap(fp, fq);
            }
        }

        // Build b: b[p] = φ_{i_p}^T y = sum of y at its D indices
        for (int p = 0; p < cur_k; ++p) {
            int ip = support_indices[p];
            b[p] = flow_dot_y(&flows[ip]);
        }

        // ---- Step 3: solve G a = b via Cholesky ----
        if (!cholesky_solve(cur_k, G, b, a)) {
            // Matrix not positive definite (rare with good hashing),
            // stop to avoid blowing up.
            break;
        }

        // ---- Step 4: write coefficients back into x for all selected flows ----
        for (int p = 0; p < cur_k; ++p) {
            int fi = support_indices[p];
            x[fi] = a[p];
        }

        // ---- Step 5: recompute residual r = y - Φ_S x_S ----
        recompute_residual_from_support(support_indices, cur_k);

        // ---- Step 6: optional stopping: residual small enough? ----
        double norm_r = residual_l2_norm();
        if (norm_r < 1e-6) {
            break;
        }
    }

    // At this point:
    //  - x[i] holds the estimated size for flow i (0 for non-selected flows)
    //  - selected[i] marks which flows ended up in the support
    //  - r is the final residual
}
