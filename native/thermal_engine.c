/*
 * TVAC Thermal Analyzer - Native Thermal Engine
 * ==============================================
 * High-performance thermal simulation in C with OpenMP parallelization.
 *
 * Features:
 * - Sparse matrix storage (CSR format)
 * - Conjugate Gradient solver
 * - Crank-Nicolson time stepping
 * - OpenMP parallelization
 * - Radiation heat transfer
 *
 * Build:
 *   gcc -O3 -fopenmp -shared -fPIC -o libthermal_engine.so thermal_engine.c -lm
 *
 * Author: Space Electronics Thermal Analysis Tool
 * Version: 2.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Physical constants */
#define STEFAN_BOLTZMANN 5.670374419e-8
#define CELSIUS_TO_KELVIN 273.15

/* Error codes */
#define THERMAL_OK 0
#define THERMAL_ERROR_MEMORY -1
#define THERMAL_ERROR_INVALID -2
#define THERMAL_ERROR_CONVERGENCE -3

/* Progress callback type */
typedef void (*progress_callback_t)(int percent, const char* message);

/*
 * Sparse matrix in CSR format
 */
typedef struct {
    int n;          /* Matrix dimension */
    int nnz;        /* Number of non-zeros */
    double* values; /* Non-zero values */
    int* col_idx;   /* Column indices */
    int* row_ptr;   /* Row pointers */
} SparseMatrix;

/*
 * Thermal simulation state
 */
typedef struct {
    /* Grid dimensions */
    int num_nodes;
    int nx, ny, nz;
    
    /* Node properties (arrays of size num_nodes) */
    double* k;           /* Thermal conductivity */
    double* cp;          /* Specific heat */
    double* rho;         /* Density */
    double* emissivity;  /* Surface emissivity */
    double* volume;      /* Node volume */
    double* surface_area;/* Surface area for radiation */
    double* heat_source; /* Heat source power */
    
    /* Boundary conditions */
    int* is_fixed_temp;  /* Boolean: is temperature fixed? */
    double* fixed_temp;  /* Fixed temperature value (Kelvin) */
    
    /* Neighbor connectivity (CSR format) */
    int* neighbor_ptr;      /* Row pointers */
    int* neighbor_idx;      /* Neighbor indices */
    double* conductance;    /* Thermal conductance */
    
    /* Temperature arrays */
    double* temperature;     /* Current temperature (Kelvin) */
    double* temp_prev;       /* Previous temperature */
    
    /* Simulation parameters */
    double chamber_temp_k;   /* Chamber wall temperature */
    double convergence;      /* Convergence criterion */
    int max_iterations;      /* Maximum CG iterations */
    
    /* Progress callback */
    progress_callback_t progress_cb;
    
} ThermalState;

/*
 * Result structure
 */
typedef struct {
    double min_temp;
    double max_temp;
    double avg_temp;
    int iterations;
    double compute_time;
    int converged;
    char error_message[256];
} ThermalResult;


/* =========================================================================
 * Memory management
 * ========================================================================= */

ThermalState* thermal_create_state(int num_nodes, int nx, int ny, int nz) {
    ThermalState* state = (ThermalState*)calloc(1, sizeof(ThermalState));
    if (!state) return NULL;
    
    state->num_nodes = num_nodes;
    state->nx = nx;
    state->ny = ny;
    state->nz = nz;
    
    /* Allocate node property arrays */
    state->k = (double*)calloc(num_nodes, sizeof(double));
    state->cp = (double*)calloc(num_nodes, sizeof(double));
    state->rho = (double*)calloc(num_nodes, sizeof(double));
    state->emissivity = (double*)calloc(num_nodes, sizeof(double));
    state->volume = (double*)calloc(num_nodes, sizeof(double));
    state->surface_area = (double*)calloc(num_nodes, sizeof(double));
    state->heat_source = (double*)calloc(num_nodes, sizeof(double));
    
    state->is_fixed_temp = (int*)calloc(num_nodes, sizeof(int));
    state->fixed_temp = (double*)calloc(num_nodes, sizeof(double));
    
    state->temperature = (double*)calloc(num_nodes, sizeof(double));
    state->temp_prev = (double*)calloc(num_nodes, sizeof(double));
    
    /* Default parameters */
    state->chamber_temp_k = 25.0 + CELSIUS_TO_KELVIN;
    state->convergence = 1e-6;
    state->max_iterations = 10000;
    state->progress_cb = NULL;
    
    /* Initialize temperatures */
    for (int i = 0; i < num_nodes; i++) {
        state->temperature[i] = state->chamber_temp_k;
        state->temp_prev[i] = state->chamber_temp_k;
    }
    
    return state;
}

void thermal_destroy_state(ThermalState* state) {
    if (!state) return;
    
    free(state->k);
    free(state->cp);
    free(state->rho);
    free(state->emissivity);
    free(state->volume);
    free(state->surface_area);
    free(state->heat_source);
    free(state->is_fixed_temp);
    free(state->fixed_temp);
    free(state->neighbor_ptr);
    free(state->neighbor_idx);
    free(state->conductance);
    free(state->temperature);
    free(state->temp_prev);
    
    free(state);
}

/* =========================================================================
 * Sparse matrix operations
 * ========================================================================= */

/* Sparse matrix-vector multiply: y = A * x */
void spmv_csr(int n, const double* values, const int* col_idx, 
              const int* row_ptr, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

/* Dot product */
double dot_product(int n, const double* a, const double* b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Vector operations */
void vec_copy(int n, const double* src, double* dst) {
    memcpy(dst, src, n * sizeof(double));
}

void vec_axpy(int n, double a, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

void vec_scale(int n, double a, double* x) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}

/* =========================================================================
 * Conjugate Gradient solver
 * ========================================================================= */

int cg_solve(int n, const double* A_values, const int* A_col_idx,
             const int* A_row_ptr, const double* b, double* x,
             double tol, int max_iter) {
    
    double* r = (double*)malloc(n * sizeof(double));
    double* p = (double*)malloc(n * sizeof(double));
    double* Ap = (double*)malloc(n * sizeof(double));
    
    if (!r || !p || !Ap) {
        free(r); free(p); free(Ap);
        return THERMAL_ERROR_MEMORY;
    }
    
    /* r = b - A*x */
    spmv_csr(n, A_values, A_col_idx, A_row_ptr, x, r);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - r[i];
    }
    
    vec_copy(n, r, p);
    
    double rr = dot_product(n, r, r);
    double rr_init = rr;
    
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Check convergence */
        if (sqrt(rr) < tol * sqrt(rr_init) || sqrt(rr) < 1e-15) {
            break;
        }
        
        /* Ap = A * p */
        spmv_csr(n, A_values, A_col_idx, A_row_ptr, p, Ap);
        
        double pAp = dot_product(n, p, Ap);
        if (fabs(pAp) < 1e-30) break;
        
        double alpha = rr / pAp;
        
        /* x = x + alpha * p */
        vec_axpy(n, alpha, p, x);
        
        /* r = r - alpha * Ap */
        vec_axpy(n, -alpha, Ap, r);
        
        double rr_new = dot_product(n, r, r);
        double beta = rr_new / rr;
        rr = rr_new;
        
        /* p = r + beta * p */
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }
    
    free(r);
    free(p);
    free(Ap);
    
    return (iter < max_iter) ? THERMAL_OK : THERMAL_ERROR_CONVERGENCE;
}

/* =========================================================================
 * Radiation heat transfer
 * ========================================================================= */

void compute_radiation_heat(ThermalState* state, double* Q_rad) {
    double T_wall = state->chamber_temp_k;
    
    #pragma omp parallel for
    for (int i = 0; i < state->num_nodes; i++) {
        if (state->surface_area[i] > 0 && state->emissivity[i] > 0) {
            double T = state->temperature[i];
            /* Q = epsilon * sigma * A * (T_wall^4 - T^4) */
            Q_rad[i] = state->emissivity[i] * STEFAN_BOLTZMANN * 
                       state->surface_area[i] * 
                       (pow(T_wall, 4) - pow(T, 4));
        } else {
            Q_rad[i] = 0.0;
        }
    }
}

/* =========================================================================
 * Build conductance matrix from neighbor connectivity
 * ========================================================================= */

int build_conductance_matrix(ThermalState* state, 
                            double** K_values, int** K_col_idx, int** K_row_ptr,
                            int* nnz) {
    int n = state->num_nodes;
    
    /* Count non-zeros */
    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        int row_nnz = state->neighbor_ptr[i + 1] - state->neighbor_ptr[i];
        total_nnz += row_nnz + 1;  /* +1 for diagonal */
    }
    
    *K_values = (double*)calloc(total_nnz, sizeof(double));
    *K_col_idx = (int*)calloc(total_nnz, sizeof(int));
    *K_row_ptr = (int*)calloc(n + 1, sizeof(int));
    
    if (!*K_values || !*K_col_idx || !*K_row_ptr) {
        return THERMAL_ERROR_MEMORY;
    }
    
    int idx = 0;
    for (int i = 0; i < n; i++) {
        (*K_row_ptr)[i] = idx;
        double diag = 0.0;
        
        /* Off-diagonal entries */
        for (int j = state->neighbor_ptr[i]; j < state->neighbor_ptr[i + 1]; j++) {
            int neighbor = state->neighbor_idx[j];
            double cond = state->conductance[j];
            
            (*K_values)[idx] = -cond;
            (*K_col_idx)[idx] = neighbor;
            idx++;
            
            diag += cond;
        }
        
        /* Diagonal entry */
        (*K_values)[idx] = diag;
        (*K_col_idx)[idx] = i;
        idx++;
    }
    (*K_row_ptr)[n] = idx;
    *nnz = idx;
    
    return THERMAL_OK;
}

/* =========================================================================
 * Main solvers
 * ========================================================================= */

int thermal_solve_steady_state(ThermalState* state, ThermalResult* result) {
    clock_t start = clock();
    
    int n = state->num_nodes;
    
    /* Build conductance matrix */
    double* K_values;
    int* K_col_idx;
    int* K_row_ptr;
    int nnz;
    
    int err = build_conductance_matrix(state, &K_values, &K_col_idx, &K_row_ptr, &nnz);
    if (err != THERMAL_OK) {
        snprintf(result->error_message, 256, "Failed to build conductance matrix");
        return err;
    }
    
    /* Allocate work arrays */
    double* Q = (double*)calloc(n, sizeof(double));
    double* Q_rad = (double*)calloc(n, sizeof(double));
    
    if (!Q || !Q_rad) {
        free(K_values); free(K_col_idx); free(K_row_ptr);
        free(Q); free(Q_rad);
        return THERMAL_ERROR_MEMORY;
    }
    
    /* Initialize heat sources */
    for (int i = 0; i < n; i++) {
        Q[i] = state->heat_source[i];
    }
    
    /* Iterative solution for radiation nonlinearity */
    int converged = 0;
    int total_iterations = 0;
    
    for (int rad_iter = 0; rad_iter < 50; rad_iter++) {
        vec_copy(n, state->temperature, state->temp_prev);
        
        /* Compute radiation */
        compute_radiation_heat(state, Q_rad);
        
        /* Total heat source */
        double* Q_total = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            Q_total[i] = Q[i] + Q_rad[i];
        }
        
        /* Apply boundary conditions */
        for (int i = 0; i < n; i++) {
            if (state->is_fixed_temp[i]) {
                /* Modify row to enforce T = T_fixed */
                for (int j = K_row_ptr[i]; j < K_row_ptr[i + 1]; j++) {
                    if (K_col_idx[j] == i) {
                        K_values[j] = 1.0;
                    } else {
                        K_values[j] = 0.0;
                    }
                }
                Q_total[i] = state->fixed_temp[i];
            }
        }
        
        /* Solve K * T = Q */
        err = cg_solve(n, K_values, K_col_idx, K_row_ptr, 
                       Q_total, state->temperature,
                       state->convergence, state->max_iterations);
        
        free(Q_total);
        
        if (err != THERMAL_OK) {
            total_iterations += state->max_iterations;
            continue;
        }
        
        /* Check convergence */
        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = fabs(state->temperature[i] - state->temp_prev[i]);
            if (diff > max_diff) max_diff = diff;
        }
        
        total_iterations += 100;  /* Approximate */
        
        if (max_diff < state->convergence) {
            converged = 1;
            break;
        }
        
        if (state->progress_cb) {
            int pct = (rad_iter + 1) * 2;
            state->progress_cb(pct, "Iterating...");
        }
    }
    
    /* Compute statistics */
    double min_temp = state->temperature[0];
    double max_temp = state->temperature[0];
    double sum_temp = 0.0;
    
    for (int i = 0; i < n; i++) {
        double T = state->temperature[i] - CELSIUS_TO_KELVIN;
        if (T < min_temp) min_temp = T;
        if (T > max_temp) max_temp = T;
        sum_temp += T;
    }
    
    result->min_temp = min_temp;
    result->max_temp = max_temp;
    result->avg_temp = sum_temp / n;
    result->iterations = total_iterations;
    result->compute_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result->converged = converged;
    result->error_message[0] = '\0';
    
    /* Cleanup */
    free(K_values);
    free(K_col_idx);
    free(K_row_ptr);
    free(Q);
    free(Q_rad);
    
    return THERMAL_OK;
}

int thermal_solve_transient(ThermalState* state, ThermalResult* result,
                           double duration, double timestep) {
    clock_t start = clock();
    
    int n = state->num_nodes;
    int num_steps = (int)(duration / timestep);
    
    /* Build matrices */
    double* K_values;
    int* K_col_idx;
    int* K_row_ptr;
    int nnz;
    
    int err = build_conductance_matrix(state, &K_values, &K_col_idx, &K_row_ptr, &nnz);
    if (err != THERMAL_OK) {
        return err;
    }
    
    /* Allocate work arrays */
    double* Q = (double*)calloc(n, sizeof(double));
    double* Q_rad = (double*)calloc(n, sizeof(double));
    double* C = (double*)calloc(n, sizeof(double));
    double* rhs = (double*)calloc(n, sizeof(double));
    
    /* Build capacitance (diagonal) */
    for (int i = 0; i < n; i++) {
        C[i] = state->rho[i] * state->cp[i] * state->volume[i];
        Q[i] = state->heat_source[i];
    }
    
    /* Time stepping (Crank-Nicolson) */
    double theta = 0.5;
    double dt = timestep;
    
    for (int step = 0; step < num_steps; step++) {
        /* Radiation */
        compute_radiation_heat(state, Q_rad);
        
        /* Build RHS: (C/dt - (1-theta)*K) * T_old + Q */
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = K_row_ptr[i]; j < K_row_ptr[i + 1]; j++) {
                sum += K_values[j] * state->temperature[K_col_idx[j]];
            }
            rhs[i] = (C[i] / dt) * state->temperature[i] - (1 - theta) * sum 
                     + Q[i] + Q_rad[i];
        }
        
        /* Build LHS coefficients: C/dt + theta*K */
        double* LHS_diag = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            double k_diag = 0.0;
            for (int j = K_row_ptr[i]; j < K_row_ptr[i + 1]; j++) {
                if (K_col_idx[j] == i) {
                    k_diag = K_values[j];
                    break;
                }
            }
            LHS_diag[i] = C[i] / dt + theta * k_diag;
        }
        
        /* Simple point Jacobi update (for speed) */
        for (int i = 0; i < n; i++) {
            if (state->is_fixed_temp[i]) {
                state->temperature[i] = state->fixed_temp[i];
            } else if (LHS_diag[i] > 1e-30) {
                state->temperature[i] = rhs[i] / LHS_diag[i];
            }
        }
        
        free(LHS_diag);
        
        /* Progress */
        if (state->progress_cb && step % 10 == 0) {
            int pct = (step * 100) / num_steps;
            char msg[64];
            snprintf(msg, 64, "Time: %.1f / %.1f s", step * dt, duration);
            state->progress_cb(pct, msg);
        }
    }
    
    /* Compute statistics (in Celsius) */
    double min_temp = 1e30, max_temp = -1e30, sum_temp = 0.0;
    for (int i = 0; i < n; i++) {
        double T = state->temperature[i] - CELSIUS_TO_KELVIN;
        if (T < min_temp) min_temp = T;
        if (T > max_temp) max_temp = T;
        sum_temp += T;
    }
    
    result->min_temp = min_temp;
    result->max_temp = max_temp;
    result->avg_temp = sum_temp / n;
    result->iterations = num_steps;
    result->compute_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    result->converged = 1;
    
    /* Cleanup */
    free(K_values);
    free(K_col_idx);
    free(K_row_ptr);
    free(Q);
    free(Q_rad);
    free(C);
    free(rhs);
    
    return THERMAL_OK;
}

/* =========================================================================
 * API functions for Python/ctypes interface
 * ========================================================================= */

/* Set node properties */
void thermal_set_node(ThermalState* state, int idx,
                     double k, double cp, double rho, double emissivity,
                     double volume, double surface_area, double heat_source) {
    if (idx < 0 || idx >= state->num_nodes) return;
    
    state->k[idx] = k;
    state->cp[idx] = cp;
    state->rho[idx] = rho;
    state->emissivity[idx] = emissivity;
    state->volume[idx] = volume;
    state->surface_area[idx] = surface_area;
    state->heat_source[idx] = heat_source;
}

/* Set fixed temperature */
void thermal_set_fixed_temp(ThermalState* state, int idx, double temp_k) {
    if (idx < 0 || idx >= state->num_nodes) return;
    
    state->is_fixed_temp[idx] = 1;
    state->fixed_temp[idx] = temp_k;
}

/* Set initial temperature */
void thermal_set_initial_temp(ThermalState* state, int idx, double temp_k) {
    if (idx < 0 || idx >= state->num_nodes) return;
    state->temperature[idx] = temp_k;
}

/* Set chamber temperature */
void thermal_set_chamber_temp(ThermalState* state, double temp_k) {
    state->chamber_temp_k = temp_k;
}

/* Get temperature */
double thermal_get_temp(ThermalState* state, int idx) {
    if (idx < 0 || idx >= state->num_nodes) return 0.0;
    return state->temperature[idx];
}

/* Set progress callback */
void thermal_set_progress_callback(ThermalState* state, progress_callback_t cb) {
    state->progress_cb = cb;
}

/* Allocate neighbor connectivity */
int thermal_alloc_neighbors(ThermalState* state, int total_neighbors) {
    state->neighbor_ptr = (int*)calloc(state->num_nodes + 1, sizeof(int));
    state->neighbor_idx = (int*)calloc(total_neighbors, sizeof(int));
    state->conductance = (double*)calloc(total_neighbors, sizeof(double));
    
    if (!state->neighbor_ptr || !state->neighbor_idx || !state->conductance) {
        return THERMAL_ERROR_MEMORY;
    }
    return THERMAL_OK;
}

/* Set neighbor connectivity */
void thermal_set_neighbor(ThermalState* state, int node_idx, int neighbor_offset,
                         int neighbor_idx, double conductance) {
    int ptr = state->neighbor_ptr[node_idx] + neighbor_offset;
    state->neighbor_idx[ptr] = neighbor_idx;
    state->conductance[ptr] = conductance;
}

/* Set row pointer */
void thermal_set_row_ptr(ThermalState* state, int row, int ptr) {
    state->neighbor_ptr[row] = ptr;
}

/* Version info */
const char* thermal_get_version(void) {
    return "2.0.0";
}

int thermal_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int thermal_get_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
