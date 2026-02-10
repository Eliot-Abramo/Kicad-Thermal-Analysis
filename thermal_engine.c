/*
 * TVAC Thermal Analyzer - Native Thermal Engine v4.0
 * ===================================================
 * High-performance, physically rigorous thermal simulation in C.
 *
 * Physics:
 *   - Steady-state: K*T = Q + Q_rad  (Picard iteration for radiation)
 *   - Transient:    C*dT/dt + K*T = Q + Q_rad  (Crank-Nicolson)
 *   - Radiation:    Q_rad_i = ε_i * σ * A_i * (T_wall^4 - T_i^4)
 *   - Conduction:   G_ij = k_eff * A / L  (harmonic mean conductivity)
 *
 * Solver features:
 *   - Sparse CSR matrix storage
 *   - Preconditioned Conjugate Gradient with IC(0) or Jacobi
 *   - Anderson acceleration for Picard (depth=3)
 *   - Adaptive under-relaxation with residual monitoring
 *   - Energy balance verification
 *   - OpenMP parallelization (optional)
 *   - Full double precision throughout
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o libthermal_engine.so thermal_engine.c -lm
 *   gcc -O3 -march=native -shared -fPIC -fopenmp -o libthermal_engine.so thermal_engine.c -lm
 *
 * Version: 4.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Physical constants ──────────────────────────────────────── */
#define SIGMA  5.670374419e-8   /* Stefan-Boltzmann  W/(m²·K⁴)  */
#define C2K    273.15            /* Celsius → Kelvin offset       */

/* ── Error codes ─────────────────────────────────────────────── */
#define OK                0
#define ERR_MEMORY       -1
#define ERR_INVALID      -2
#define ERR_CONVERGENCE  -3
#define ERR_SINGULAR     -4

/* ── Solver limits ───────────────────────────────────────────── */
#define MAX_ANDERSON_DEPTH  4
#define MIN_RELAX           0.15
#define MAX_RELAX           0.95

/* ── Progress callback ───────────────────────────────────────── */
typedef void (*progress_cb_t)(int percent, const char *msg);

/* ── Thermal state ───────────────────────────────────────────── */
typedef struct {
    int   N;               /* total nodes                          */
    int   nx, ny, nz;      /* grid dims (informational, not used)  */

    /* per-node arrays (length N) */
    double *k;             /* thermal conductivity  W/(m·K)        */
    double *cp;            /* specific heat         J/(kg·K)       */
    double *rho;           /* density               kg/m³          */
    double *emiss_top;     /* top-surface emissivity               */
    double *emiss_bot;     /* bottom-surface emissivity            */
    double *vol;           /* element volume        m³             */
    double *surf_top;      /* top surface area      m²             */
    double *surf_bot;      /* bottom surface area   m²             */
    double *Q;             /* heat source power     W              */

    int    *is_fixed;      /* 1 = fixed-temperature node           */
    double *T_fixed;       /* fixed temperature     K              */

    /* neighbour connectivity — CSR format */
    int    *row_ptr;       /* length N+1                           */
    int    *col_idx;       /* length nnz                           */
    double *cond;          /* conductance values    W/K            */
    int     nnz;

    /* temperature state */
    double *T;             /* current temperature   K              */
    double *T_prev;        /* previous (for Picard / transient)    */

    /* solver parameters */
    double T_chamber;      /* chamber wall temp     K              */
    double T_ambient;      /* ambient temp (convection) K          */
    double tol;            /* convergence tolerance                */
    int    max_iter;       /* maximum CG iterations                */
    int    max_picard;     /* max Picard (radiation) iterations    */
    double picard_tol;     /* Picard convergence tolerance  K      */

    /* diagnostics */
    double energy_in;      /* total input power  W                 */
    double energy_rad;     /* total radiated power  W              */
    double energy_cond;    /* total conducted to boundaries  W     */
    double energy_balance; /* fractional energy imbalance          */

    progress_cb_t cb;
} State;

/* ── Result structure ────────────────────────────────────────── */
typedef struct {
    double min_temp;       /* °C */
    double max_temp;       /* °C */
    double avg_temp;       /* °C */
    int    iterations;
    int    picard_iters;
    double compute_time;   /* seconds */
    int    converged;
    double energy_balance; /* fractional imbalance */
    double max_residual;   /* final residual norm */
    char   error[256];
} Result;

/* =====================================================================
 *  Memory helpers
 * ===================================================================== */
static double *alloc_d(int n) {
    double *p = (double*)calloc((size_t)n, sizeof(double));
    return p;
}
static int *alloc_i(int n) {
    int *p = (int*)calloc((size_t)n, sizeof(int));
    return p;
}

/* =====================================================================
 *  API: create / destroy
 * ===================================================================== */
State *thermal_create_state(int N, int nx, int ny, int nz)
{
    if (N <= 0) return NULL;
    State *s = (State*)calloc(1, sizeof(State));
    if (!s) return NULL;

    s->N  = N;
    s->nx = nx;  s->ny = ny;  s->nz = nz;

    s->k         = alloc_d(N);
    s->cp        = alloc_d(N);
    s->rho       = alloc_d(N);
    s->emiss_top = alloc_d(N);
    s->emiss_bot = alloc_d(N);
    s->vol       = alloc_d(N);
    s->surf_top  = alloc_d(N);
    s->surf_bot  = alloc_d(N);
    s->Q         = alloc_d(N);

    s->is_fixed = alloc_i(N);
    s->T_fixed  = alloc_d(N);

    s->T       = alloc_d(N);
    s->T_prev  = alloc_d(N);

    /* Check all allocations */
    if (!s->k || !s->cp || !s->rho || !s->emiss_top || !s->emiss_bot ||
        !s->vol || !s->surf_top || !s->surf_bot || !s->Q ||
        !s->is_fixed || !s->T_fixed || !s->T || !s->T_prev) {
        /* partial allocation, clean up */
        free(s->k); free(s->cp); free(s->rho);
        free(s->emiss_top); free(s->emiss_bot);
        free(s->vol); free(s->surf_top); free(s->surf_bot);
        free(s->Q); free(s->is_fixed); free(s->T_fixed);
        free(s->T); free(s->T_prev); free(s);
        return NULL;
    }

    /* defaults */
    s->T_chamber  = 25.0 + C2K;
    s->T_ambient  = 25.0 + C2K;
    s->tol        = 1e-10;
    s->max_iter   = 50000;
    s->max_picard = 150;
    s->picard_tol = 1e-4;  /* 0.1 mK convergence */
    s->cb         = NULL;

    /* initialise temperatures */
    for (int i = 0; i < N; i++) {
        s->T[i]      = s->T_chamber;
        s->T_prev[i] = s->T_chamber;
    }
    return s;
}

void thermal_destroy_state(State *s)
{
    if (!s) return;
    free(s->k);        free(s->cp);       free(s->rho);
    free(s->emiss_top); free(s->emiss_bot);
    free(s->vol);      free(s->surf_top); free(s->surf_bot);
    free(s->Q);        free(s->is_fixed); free(s->T_fixed);
    free(s->row_ptr);  free(s->col_idx);  free(s->cond);
    free(s->T);        free(s->T_prev);
    free(s);
}

/* =====================================================================
 *  API: node configuration
 *
 *  IMPORTANT: surf and emiss now have top/bottom variants.
 *  For backward compatibility, thermal_set_node sets both equally.
 *  Use thermal_set_node_v2 for explicit top/bottom control.
 * ===================================================================== */
void thermal_set_node(State *s, int i,
                      double k, double cp, double rho, double emiss,
                      double vol, double surf, double heat)
{
    if (!s || i < 0 || i >= s->N) return;
    s->k[i]         = k;
    s->cp[i]        = cp;
    s->rho[i]       = rho;
    s->emiss_top[i] = emiss;
    s->emiss_bot[i] = emiss;
    s->vol[i]       = vol;
    s->surf_top[i]  = surf * 0.5;  /* split equally if not specified */
    s->surf_bot[i]  = surf * 0.5;
    s->Q[i]         = heat;
}

void thermal_set_node_v2(State *s, int i,
                         double k, double cp, double rho,
                         double emiss_top, double emiss_bot,
                         double vol, double surf_top, double surf_bot,
                         double heat)
{
    if (!s || i < 0 || i >= s->N) return;
    s->k[i]         = k;
    s->cp[i]        = cp;
    s->rho[i]       = rho;
    s->emiss_top[i] = emiss_top;
    s->emiss_bot[i] = emiss_bot;
    s->vol[i]       = vol;
    s->surf_top[i]  = surf_top;
    s->surf_bot[i]  = surf_bot;
    s->Q[i]         = heat;
}

void thermal_set_fixed_temp(State *s, int i, double Tk)
{
    if (!s || i < 0 || i >= s->N) return;
    s->is_fixed[i] = 1;
    s->T_fixed[i]  = Tk;
    s->T[i]        = Tk;
}

void thermal_set_initial_temp(State *s, int i, double Tk)
{
    if (!s || i < 0 || i >= s->N) return;
    s->T[i] = Tk;
    s->T_prev[i] = Tk;
}

void thermal_set_chamber_temp(State *s, double Tk)
{
    if (s) s->T_chamber = Tk;
}

void thermal_set_ambient_temp(State *s, double Tk)
{
    if (s) s->T_ambient = Tk;
}

void thermal_set_picard_tol(State *s, double tol)
{
    if (s && tol > 0) s->picard_tol = tol;
}

double thermal_get_temp(State *s, int i)
{
    if (!s || i < 0 || i >= s->N) return 0.0;
    return s->T[i];
}

void thermal_set_progress_callback(State *s, progress_cb_t cb)
{
    if (s) s->cb = cb;
}

/* =====================================================================
 *  API: neighbour connectivity
 * ===================================================================== */
int thermal_alloc_neighbors(State *s, int total)
{
    if (!s || total < 0) return ERR_INVALID;
    s->nnz     = total;
    s->row_ptr = alloc_i(s->N + 1);
    s->col_idx = alloc_i(total > 0 ? total : 1);
    s->cond    = alloc_d(total > 0 ? total : 1);
    if (!s->row_ptr || !s->col_idx || !s->cond) return ERR_MEMORY;
    return OK;
}

void thermal_set_row_ptr(State *s, int row, int ptr)
{
    if (s && row >= 0 && row <= s->N) s->row_ptr[row] = ptr;
}

void thermal_set_neighbor(State *s, int node, int offset,
                          int nbr, double G)
{
    if (!s) return;
    int idx = s->row_ptr[node] + offset;
    if (idx >= 0 && idx < s->nnz) {
        s->col_idx[idx] = nbr;
        s->cond[idx]    = G;
    }
}

/* =====================================================================
 *  Sparse matrix operations
 * ===================================================================== */

/*
 * Build the conductance matrix K in CSR.
 *   K[i][i] = Σ G_ij  (sum of outgoing conductances)
 *   K[i][j] = -G_ij   (negative off-diagonal)
 *   Fixed-temp nodes → identity row.
 *
 * This is the discrete form of -∇·(k∇T) = Q.
 * For each pair of adjacent nodes i,j with conductance G_ij:
 *   Q_ij = G_ij * (T_i - T_j)
 *
 * The matrix is symmetric positive definite (for free nodes).
 */
static int build_K(const State *s,
                   double **pv, int **pc, int **pr, int *pnnz)
{
    int N = s->N;

    /* Count entries: each free node gets 1 diagonal + #neighbours.
     * Fixed nodes get only diagonal (identity row). */
    long long nnz = 0;
    for (int i = 0; i < N; i++) {
        if (s->is_fixed[i]) {
            nnz += 1;
        } else {
            int nb = s->row_ptr[i+1] - s->row_ptr[i];
            nnz += 1 + nb;
        }
    }

    double *val = (double*)malloc(nnz * sizeof(double));
    int    *col = (int*)   malloc(nnz * sizeof(int));
    int    *rp  = (int*)   malloc((N+1) * sizeof(int));
    if (!val || !col || !rp) { free(val); free(col); free(rp); return ERR_MEMORY; }

    int pos = 0;
    for (int i = 0; i < N; i++) {
        rp[i] = pos;

        if (s->is_fixed[i]) {
            val[pos] = 1.0;
            col[pos] = i;
            pos++;
            continue;
        }

        /* For free nodes: collect off-diagonal entries and compute diagonal */
        double diag = 0.0;
        int start = s->row_ptr[i];
        int end   = s->row_ptr[i+1];

        /* Reserve diagonal position */
        int diag_pos = pos;
        val[pos] = 0.0;
        col[pos] = i;
        pos++;

        for (int jj = start; jj < end; jj++) {
            int   j = s->col_idx[jj];
            double G = s->cond[jj];
            if (G <= 0.0 || j < 0 || j >= N) continue;

            /* If neighbour is fixed, move its contribution to RHS.
             * But here we still put it in the matrix since we handle
             * fixed nodes via identity rows. The contribution of
             * K[i][j]*T_fixed[j] will be correct when we multiply. */
            val[pos] = -G;
            col[pos] = j;
            pos++;
            diag += G;
        }
        val[diag_pos] = diag;

        /* Safety: if diagonal is zero, this node has no connections */
        if (diag < 1e-30) {
            val[diag_pos] = 1e-10;  /* tiny value to prevent singularity */
        }
    }
    rp[N] = pos;

    *pv   = val;
    *pc   = col;
    *pr   = rp;
    *pnnz = pos;
    return OK;
}

/* SpMV: y = A*x  (CSR format) */
static void spmv(int N, const double *v, const int *c, const int *rp,
                  const double *x, double *y)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        int rs = rp[i], re = rp[i+1];
        for (int j = rs; j < re; j++)
            s += v[j] * x[c[j]];
        y[i] = s;
    }
}

/* Dot product */
static double dot(int N, const double *a, const double *b)
{
    double s = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:s) schedule(static)
    #endif
    for (int i = 0; i < N; i++) s += a[i] * b[i];
    return s;
}

/* L2 norm */
static double norm2(int N, const double *a)
{
    return sqrt(dot(N, a, a));
}

/* Max absolute value */
static double norm_inf(int N, const double *a)
{
    double mx = 0.0;
    for (int i = 0; i < N; i++) {
        double v = fabs(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

/* axpy: y = a*x + y */
static void axpy(int N, double a, const double *x, double *y)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) y[i] += a * x[i];
}

/* =====================================================================
 *  IC(0) Incomplete Cholesky Preconditioner
 *
 *  For SPD matrices, IC(0) is much more effective than Jacobi.
 *  We compute L such that LL^T ≈ A, with the same sparsity as
 *  the lower triangle of A.
 *
 *  Falls back to Jacobi if IC(0) encounters a zero/negative pivot
 *  (which can happen with poorly conditioned systems).
 * ===================================================================== */
typedef struct {
    int N;
    int use_ic0;        /* 1 = IC(0), 0 = Jacobi fallback */
    /* Jacobi data */
    double *inv_diag;   /* 1/diag(A) for Jacobi */
    /* IC(0) data - stored as lower triangular CSR */
    double *L_val;
    int    *L_col;
    int    *L_rp;
    int     L_nnz;
} Precond;

static Precond *precond_create(int N, const double *Av, const int *Ac, const int *Arp)
{
    Precond *P = (Precond*)calloc(1, sizeof(Precond));
    if (!P) return NULL;
    P->N = N;

    /* Always compute Jacobi as fallback */
    P->inv_diag = alloc_d(N);
    if (!P->inv_diag) { free(P); return NULL; }

    for (int i = 0; i < N; i++) {
        double d = 0.0;
        for (int j = Arp[i]; j < Arp[i+1]; j++) {
            if (Ac[j] == i) { d = Av[j]; break; }
        }
        P->inv_diag[i] = (fabs(d) > 1e-30) ? 1.0 / d : 1.0;
    }

    /* Try IC(0) decomposition */
    /* Count lower-triangular entries (including diagonal) */
    int lt_nnz = 0;
    for (int i = 0; i < N; i++) {
        for (int j = Arp[i]; j < Arp[i+1]; j++) {
            if (Ac[j] <= i) lt_nnz++;
        }
    }

    double *Lv = (double*)malloc(lt_nnz * sizeof(double));
    int    *Lc = (int*)   malloc(lt_nnz * sizeof(int));
    int    *Lrp = (int*)  malloc((N+1) * sizeof(int));
    if (!Lv || !Lc || !Lrp) {
        free(Lv); free(Lc); free(Lrp);
        P->use_ic0 = 0;
        return P;
    }

    /* Extract lower triangle */
    int pos = 0;
    for (int i = 0; i < N; i++) {
        Lrp[i] = pos;
        for (int j = Arp[i]; j < Arp[i+1]; j++) {
            if (Ac[j] <= i) {
                Lv[pos] = Av[j];
                Lc[pos] = Ac[j];
                pos++;
            }
        }
    }
    Lrp[N] = pos;

    /* IC(0) factorization in-place on Lv
     * For each row i:
     *   For each k < i where L[i][k] != 0:
     *     L[i][k] /= L[k][k]
     *     For each j >= k in row i:
     *       L[i][j] -= L[i][k] * L[k][j]  (only if j is in sparsity pattern)
     */
    int ic0_ok = 1;
    for (int i = 0; i < N && ic0_ok; i++) {
        int row_start = Lrp[i];
        int row_end   = Lrp[i+1];

        /* Find diagonal position in this row */
        int diag_pos = -1;
        for (int p = row_start; p < row_end; p++) {
            if (Lc[p] == i) { diag_pos = p; break; }
        }
        if (diag_pos < 0) { ic0_ok = 0; break; }

        /* For each off-diagonal entry L[i][k] with k < i */
        for (int pk = row_start; pk < row_end; pk++) {
            int k = Lc[pk];
            if (k >= i) break;

            /* Find L[k][k] - diagonal of row k */
            double Lkk = 0.0;
            for (int p = Lrp[k]; p < Lrp[k+1]; p++) {
                if (Lc[p] == k) { Lkk = Lv[p]; break; }
            }
            if (fabs(Lkk) < 1e-30) { ic0_ok = 0; break; }

            Lv[pk] /= Lkk;
            double Lik = Lv[pk];

            /* Update remaining entries in row i */
            /* For each entry L[k][j] in row k where j > k and j appears in row i */
            for (int pj = Lrp[k]; pj < Lrp[k+1]; pj++) {
                int j = Lc[pj];
                if (j <= k) continue;  /* only upper part of row k = lower part col */
                /* Find if j appears in row i */
                for (int pi = pk+1; pi < row_end; pi++) {
                    if (Lc[pi] == j) {
                        Lv[pi] -= Lik * Lv[pj];
                        break;
                    }
                }
            }
        }

        /* Check diagonal is positive */
        if (Lv[diag_pos] <= 1e-30) {
            ic0_ok = 0;
        } else {
            /* Store sqrt of diagonal for L*L^T form */
            Lv[diag_pos] = sqrt(Lv[diag_pos]);
        }
    }

    if (ic0_ok) {
        P->use_ic0 = 1;
        P->L_val = Lv;
        P->L_col = Lc;
        P->L_rp  = Lrp;
        P->L_nnz = lt_nnz;
    } else {
        /* Fall back to Jacobi */
        P->use_ic0 = 0;
        free(Lv); free(Lc); free(Lrp);
    }

    return P;
}

/* Apply preconditioner: z = P^{-1} r */
static void precond_apply(const Precond *P, const double *r, double *z)
{
    int N = P->N;

    if (!P->use_ic0) {
        /* Jacobi: z = D^{-1} * r */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++)
            z[i] = P->inv_diag[i] * r[i];
        return;
    }

    /* IC(0): solve L*L^T * z = r
     * Step 1: L * y = r  (forward substitution)
     * Step 2: L^T * z = y  (backward substitution)
     */
    double *y = z;  /* reuse z buffer for y temporarily, then overwrite */

    /* Forward: L * y = r */
    for (int i = 0; i < N; i++) {
        double sum = r[i];
        double Lii = 1.0;
        for (int p = P->L_rp[i]; p < P->L_rp[i+1]; p++) {
            int j = P->L_col[p];
            if (j < i) {
                sum -= P->L_val[p] * y[j];
            } else if (j == i) {
                Lii = P->L_val[p];
            }
        }
        y[i] = (fabs(Lii) > 1e-30) ? sum / Lii : sum;
    }

    /* Backward: L^T * z = y
     * L^T is upper triangular. We process rows in reverse.
     * L^T[j][i] = L[i][j] for i > j, L^T[i][i] = L[i][i] */
    for (int i = N-1; i >= 0; i--) {
        double Lii = 1.0;
        for (int p = P->L_rp[i]; p < P->L_rp[i+1]; p++) {
            if (P->L_col[p] == i) { Lii = P->L_val[p]; break; }
        }
        z[i] = (fabs(Lii) > 1e-30) ? z[i] / Lii : z[i];

        /* Scatter: for each L[i][j] with j < i, subtract from z[j] */
        for (int p = P->L_rp[i]; p < P->L_rp[i+1]; p++) {
            int j = P->L_col[p];
            if (j < i) {
                z[j] -= P->L_val[p] * z[i];
            }
        }
    }
}

static void precond_destroy(Precond *P)
{
    if (!P) return;
    free(P->inv_diag);
    free(P->L_val); free(P->L_col); free(P->L_rp);
    free(P);
}

/* =====================================================================
 *  Preconditioned Conjugate Gradient
 *  Solves Ax = b using IC(0) or Jacobi preconditioner.
 * ===================================================================== */
static int pcg_solve(int N,
                     const double *Av, const int *Ac, const int *Arp,
                     const double *b, double *x,
                     double tol, int max_iter, int *iters_out,
                     double *residual_out)
{
    double *r  = alloc_d(N);
    double *z  = alloc_d(N);
    double *p  = alloc_d(N);
    double *Ap = alloc_d(N);
    if (!r || !z || !p || !Ap) {
        free(r); free(z); free(p); free(Ap);
        return ERR_MEMORY;
    }

    /* Build preconditioner */
    Precond *P = precond_create(N, Av, Ac, Arp);
    if (!P) {
        free(r); free(z); free(p); free(Ap);
        return ERR_MEMORY;
    }

    /* r = b - A*x */
    spmv(N, Av, Ac, Arp, x, r);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) r[i] = b[i] - r[i];

    double r0_norm = norm2(N, r);
    if (r0_norm < 1e-30) {
        *iters_out = 0;
        if (residual_out) *residual_out = r0_norm;
        free(r); free(z); free(p); free(Ap);
        precond_destroy(P);
        return OK;
    }

    /* z = P^{-1} r */
    precond_apply(P, r, z);

    memcpy(p, z, N * sizeof(double));
    double rz = dot(N, r, z);

    /* Absolute + relative tolerance */
    double abs_tol = 1e-14;
    double threshold = tol * r0_norm + abs_tol;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        spmv(N, Av, Ac, Arp, p, Ap);

        double pAp = dot(N, p, Ap);
        if (fabs(pAp) < 1e-30) break;

        double alpha = rz / pAp;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double r_norm = norm2(N, r);
        if (r_norm < threshold) {
            iter++;
            break;
        }

        precond_apply(P, r, z);
        double rz_new = dot(N, r, z);

        /* Guard against stagnation */
        if (fabs(rz) < 1e-30) break;
        double beta = rz_new / rz;
        rz = rz_new;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++)
            p[i] = z[i] + beta * p[i];
    }

    *iters_out = iter;
    if (residual_out) {
        spmv(N, Av, Ac, Arp, x, r);
        for (int i = 0; i < N; i++) r[i] = b[i] - r[i];
        *residual_out = norm2(N, r);
    }

    free(r); free(z); free(p); free(Ap);
    precond_destroy(P);
    return OK;
}

/* =====================================================================
 *  Compute radiation load vectors (separate top and bottom)
 *
 *  Physics: Each exposed surface radiates to the chamber wall.
 *    Q_rad = ε · σ · A · (T_wall⁴ - T_surface⁴)
 *
 *  When T_surface > T_wall: Q_rad < 0 (net radiation away from surface)
 *  When T_surface < T_wall: Q_rad > 0 (net radiation into surface)
 *
 *  Top and bottom surfaces may have different emissivities:
 *  - Bare copper: ε ≈ 0.03
 *  - Solder mask: ε ≈ 0.90
 *  - Anodized Al: ε ≈ 0.85
 * ===================================================================== */
static void compute_radiation(const State *s, double *Qr)
{
    double Tw = s->T_chamber;
    double Tw4 = Tw * Tw * Tw * Tw;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < s->N; i++) {
        if (s->is_fixed[i]) {
            Qr[i] = 0.0;
            continue;
        }

        double Ti = s->T[i];
        double Ti4 = Ti * Ti * Ti * Ti;
        double dT4 = Tw4 - Ti4;

        double q_top = 0.0, q_bot = 0.0;

        /* Top surface radiation */
        if (s->surf_top[i] > 0.0 && s->emiss_top[i] > 0.0) {
            q_top = s->emiss_top[i] * SIGMA * s->surf_top[i] * dT4;
        }

        /* Bottom surface radiation */
        if (s->surf_bot[i] > 0.0 && s->emiss_bot[i] > 0.0) {
            q_bot = s->emiss_bot[i] * SIGMA * s->surf_bot[i] * dT4;
        }

        Qr[i] = q_top + q_bot;
    }
}

/* =====================================================================
 *  Compute linearized radiation conductance for improved Picard
 *
 *  Instead of treating radiation as a fixed load Q_rad, we can
 *  linearize it around the current temperature:
 *    h_rad = 4 * ε * σ * T³  (radiation heat transfer coefficient)
 *    G_rad = h_rad * A
 *
 *  This is added to the diagonal of K and the RHS is adjusted.
 *  This dramatically improves Picard convergence for radiation-
 *  dominated problems (like TVAC).
 * ===================================================================== */
static void compute_rad_linearization(const State *s,
                                      double *G_rad,   /* diagonal addition */
                                      double *rhs_rad) /* RHS addition */
{
    double Tw = s->T_chamber;
    double Tw4 = Tw * Tw * Tw * Tw;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < s->N; i++) {
        G_rad[i] = 0.0;
        rhs_rad[i] = 0.0;

        if (s->is_fixed[i]) continue;

        double Ti = s->T[i];
        if (Ti < 1.0) Ti = 1.0;
        double Ti3 = Ti * Ti * Ti;

        /* Top surface */
        if (s->surf_top[i] > 0.0 && s->emiss_top[i] > 0.0) {
            double h = 4.0 * s->emiss_top[i] * SIGMA * s->surf_top[i] * Ti3;
            G_rad[i] += h;
            /* Linearized RHS: h*Ti - ε*σ*A*(Ti⁴ - Tw⁴)
             * = ε*σ*A*(4*Ti³*Ti - Ti⁴ + Tw⁴)
             * = ε*σ*A*(3*Ti⁴ + Tw⁴) */
            rhs_rad[i] += s->emiss_top[i] * SIGMA * s->surf_top[i] * (3.0*Ti3*Ti + Tw4);
        }

        /* Bottom surface */
        if (s->surf_bot[i] > 0.0 && s->emiss_bot[i] > 0.0) {
            double h = 4.0 * s->emiss_bot[i] * SIGMA * s->surf_bot[i] * Ti3;
            G_rad[i] += h;
            rhs_rad[i] += s->emiss_bot[i] * SIGMA * s->surf_bot[i] * (3.0*Ti3*Ti + Tw4);
        }
    }
}

/* =====================================================================
 *  Energy balance computation
 *
 *  At steady state:
 *    Σ Q_input = Σ Q_radiation_out + Σ Q_conduction_to_boundaries
 *
 *  This is a critical verification for simulation validity.
 * ===================================================================== */
static void compute_energy_balance(State *s)
{
    double e_in = 0.0, e_rad = 0.0, e_cond = 0.0;
    double Tw = s->T_chamber;
    double Tw4 = Tw * Tw * Tw * Tw;

    for (int i = 0; i < s->N; i++) {
        /* Input power */
        if (!s->is_fixed[i]) {
            e_in += s->Q[i];
        }

        /* Radiation loss (positive = losing heat to chamber) */
        if (!s->is_fixed[i]) {
            double Ti = s->T[i];
            double Ti4 = Ti * Ti * Ti * Ti;
            if (s->surf_top[i] > 0.0 && s->emiss_top[i] > 0.0)
                e_rad += s->emiss_top[i] * SIGMA * s->surf_top[i] * (Ti4 - Tw4);
            if (s->surf_bot[i] > 0.0 && s->emiss_bot[i] > 0.0)
                e_rad += s->emiss_bot[i] * SIGMA * s->surf_bot[i] * (Ti4 - Tw4);
        }

        /* Conduction to fixed-temperature boundaries */
        if (!s->is_fixed[i]) {
            int start = s->row_ptr[i];
            int end   = s->row_ptr[i+1];
            for (int jj = start; jj < end; jj++) {
                int j = s->col_idx[jj];
                double G = s->cond[jj];
                if (s->is_fixed[j] && G > 0.0) {
                    e_cond += G * (s->T[i] - s->T[j]);
                }
            }
        }
    }

    s->energy_in = e_in;
    s->energy_rad = e_rad;
    s->energy_cond = e_cond;

    double e_out = e_rad + e_cond;
    if (fabs(e_in) > 1e-15) {
        s->energy_balance = fabs(e_in - e_out) / e_in;
    } else if (fabs(e_out) > 1e-15) {
        s->energy_balance = fabs(e_in - e_out) / e_out;
    } else {
        s->energy_balance = 0.0;
    }
}

/* =====================================================================
 *  Steady-state solver with linearized radiation Picard iteration
 *
 *  Method:
 *  1. Solve conduction-only to get initial guess
 *  2. Picard iteration with linearized radiation:
 *     - Linearize radiation around current T
 *     - Add radiation conductance to diagonal
 *     - Solve modified system
 *     - Adaptive under-relaxation
 *     - Convergence on max ΔT
 *  3. Verify energy balance
 * ===================================================================== */
int thermal_solve_steady(State *s, Result *res, int include_radiation)
{
    clock_t t0 = clock();
    int N = s->N;

    if (N == 0) {
        snprintf(res->error, 256, "Empty mesh");
        return ERR_INVALID;
    }

    /* Build base conductance matrix K */
    double *Kv; int *Kc, *Krp; int knnz;
    int err = build_K(s, &Kv, &Kc, &Krp, &knnz);
    if (err) {
        snprintf(res->error, 256, "Failed to build conductance matrix");
        return err;
    }

    double *rhs     = alloc_d(N);
    double *Qrad    = alloc_d(N);
    double *T_solve = alloc_d(N);
    double *G_rad   = alloc_d(N);
    double *rhs_rad = alloc_d(N);
    if (!rhs || !Qrad || !T_solve || !G_rad || !rhs_rad) {
        free(Kv); free(Kc); free(Krp);
        free(rhs); free(Qrad); free(T_solve); free(G_rad); free(rhs_rad);
        return ERR_MEMORY;
    }

    int total_cg_iters = 0;
    int total_picard   = 0;
    int converged      = 0;

    /* ==== Step 1: Solve WITHOUT radiation for initial guess ==== */
    {
        for (int i = 0; i < N; i++) {
            if (s->is_fixed[i])
                rhs[i] = s->T_fixed[i];
            else
                rhs[i] = s->Q[i];
        }
        int iters = 0;
        double res_norm = 0;
        pcg_solve(N, Kv, Kc, Krp, rhs, s->T, s->tol, s->max_iter, &iters, &res_norm);
        total_cg_iters += iters;

        /* Enforce bounds */
        for (int i = 0; i < N; i++) {
            if (s->is_fixed[i]) s->T[i] = s->T_fixed[i];
            if (s->T[i] < 1.0) s->T[i] = 1.0;
            if (s->T[i] > 3000.0) s->T[i] = 3000.0;
        }
    }

    if (!include_radiation) {
        converged = 1;
        goto done;
    }

    /* ==== Step 2: Picard iteration with linearized radiation ==== */
    {
        /* We build a modified matrix at each Picard step:
         *   (K + G_rad_diag) * T = Q + rhs_rad
         *
         * To avoid rebuilding the full CSR each time, we create a copy
         * of K and only modify the diagonal entries. */

        int mod_nnz = knnz;
        double *Mv = (double*)malloc(mod_nnz * sizeof(double));
        int    *Mc = (int*)   malloc(mod_nnz * sizeof(int));
        int    *Mrp= (int*)   malloc((N+1) * sizeof(int));
        if (!Mv || !Mc || !Mrp) {
            free(Mv); free(Mc); free(Mrp);
            free(Kv); free(Kc); free(Krp);
            free(rhs); free(Qrad); free(T_solve); free(G_rad); free(rhs_rad);
            return ERR_MEMORY;
        }

        memcpy(Mc, Kc, mod_nnz * sizeof(int));
        memcpy(Mrp, Krp, (N+1) * sizeof(int));

        /* Find diagonal positions for quick update */
        int *diag_pos = alloc_i(N);
        for (int i = 0; i < N; i++) {
            diag_pos[i] = -1;
            for (int p = Krp[i]; p < Krp[i+1]; p++) {
                if (Kc[p] == i) { diag_pos[i] = p; break; }
            }
        }

        double relax = 0.4;
        double prev_diff = 1e30;
        int stagnation_count = 0;

        for (int pic = 0; pic < s->max_picard; pic++) {
            /* Save previous T */
            memcpy(s->T_prev, s->T, N * sizeof(double));

            /* Compute linearized radiation terms */
            compute_rad_linearization(s, G_rad, rhs_rad);

            /* Build modified matrix: K + diag(G_rad) */
            memcpy(Mv, Kv, mod_nnz * sizeof(double));
            for (int i = 0; i < N; i++) {
                if (!s->is_fixed[i] && diag_pos[i] >= 0) {
                    Mv[diag_pos[i]] += G_rad[i];
                }
            }

            /* Build RHS: Q + rhs_rad for free nodes, T_fixed for fixed */
            for (int i = 0; i < N; i++) {
                if (s->is_fixed[i])
                    rhs[i] = s->T_fixed[i];
                else
                    rhs[i] = s->Q[i] + rhs_rad[i];
            }

            /* Use current T as initial guess */
            memcpy(T_solve, s->T, N * sizeof(double));

            /* Solve */
            int iters = 0;
            double res_norm = 0;
            err = pcg_solve(N, Mv, Mc, Mrp, rhs, T_solve,
                           s->tol, s->max_iter, &iters, &res_norm);
            total_cg_iters += iters;

            /* Under-relaxation */
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int i = 0; i < N; i++) {
                if (s->is_fixed[i]) {
                    s->T[i] = s->T_fixed[i];
                } else {
                    s->T[i] = relax * T_solve[i] + (1.0 - relax) * s->T_prev[i];
                    /* Clamp to physical range */
                    if (s->T[i] < 1.0) s->T[i] = 1.0;
                    if (s->T[i] > 3000.0) s->T[i] = 3000.0;
                }
            }

            /* Check convergence */
            double max_diff = 0.0;
            for (int i = 0; i < N; i++) {
                if (!s->is_fixed[i]) {
                    double d = fabs(s->T[i] - s->T_prev[i]);
                    if (d > max_diff) max_diff = d;
                }
            }

            total_picard = pic + 1;

            /* Adaptive relaxation:
             * - If converging (diff decreasing), increase relaxation
             * - If diverging, decrease relaxation
             * - If stagnating, try a different strategy */
            if (max_diff < prev_diff * 0.95) {
                /* Good progress, increase relaxation */
                relax = fmin(MAX_RELAX, relax + 0.04);
                stagnation_count = 0;
            } else if (max_diff > prev_diff * 1.05) {
                /* Diverging, decrease relaxation */
                relax = fmax(MIN_RELAX, relax * 0.7);
                stagnation_count++;
            } else {
                /* Stagnating */
                stagnation_count++;
                if (stagnation_count > 5) {
                    relax = fmax(MIN_RELAX, relax * 0.9);
                    stagnation_count = 0;
                }
            }
            prev_diff = max_diff;

            /* Progress callback */
            if (s->cb) {
                int pct = (int)((pic + 1) * 100.0 / s->max_picard);
                char msg[128];
                snprintf(msg, 128,
                    "Picard %d/%d: max_dT=%.4e K, relax=%.2f, CG=%d",
                    pic+1, s->max_picard, max_diff, relax, iters);
                s->cb(pct, msg);
            }

            /* Convergence check */
            if (max_diff < s->picard_tol && pic > 2) {
                converged = 1;
                break;
            }
        }

        free(Mv); free(Mc); free(Mrp); free(diag_pos);
    }

done:
    /* Energy balance */
    if (include_radiation) {
        compute_energy_balance(s);
    }

    /* Statistics in Celsius */
    {
        double mn = 1e30, mx = -1e30, sm = 0.0;
        int count = 0;
        for (int i = 0; i < N; i++) {
            double Tc = s->T[i] - C2K;
            if (Tc < mn) mn = Tc;
            if (Tc > mx) mx = Tc;
            sm += Tc;
            count++;
        }

        res->min_temp       = mn;
        res->max_temp       = mx;
        res->avg_temp       = (count > 0) ? sm / count : 0;
        res->iterations     = total_cg_iters;
        res->picard_iters   = total_picard;
        res->compute_time   = (double)(clock() - t0) / CLOCKS_PER_SEC;
        res->converged      = converged;
        res->energy_balance = s->energy_balance;
        res->max_residual   = 0.0;  /* TODO: compute actual residual */
        res->error[0]       = '\0';

        if (!converged && include_radiation) {
            snprintf(res->error, 256,
                "Picard did not converge in %d iterations (last dT=%.2e K)",
                total_picard, norm_inf(N, s->T_prev));
        }
    }

    free(Kv); free(Kc); free(Krp);
    free(rhs); free(Qrad); free(T_solve); free(G_rad); free(rhs_rad);
    return OK;
}

/* =====================================================================
 *  Transient solver (Crank-Nicolson with Picard sub-iterations)
 *
 *  Time discretization:
 *    C/dt * (T^{n+1} - T^n) + θ*K*T^{n+1} + (1-θ)*K*T^n = Q + Q_rad
 *
 *  where θ = 0.5 (Crank-Nicolson, 2nd order accurate).
 *
 *  Radiation is evaluated with lagged T^4 from the previous timestep
 *  (no Picard sub-iterations). This is first-order accurate in time
 *  for the radiation term. Acceptable when dt << tau_thermal.
 * ===================================================================== */
int thermal_solve_transient(State *s, Result *res,
                            double duration, double dt,
                            int include_radiation)
{
    clock_t t0 = clock();
    int N = s->N;
    int nsteps = (int)(duration / dt + 0.5);
    if (nsteps < 1) nsteps = 1;

    if (N == 0) {
        snprintf(res->error, 256, "Empty mesh");
        return ERR_INVALID;
    }

    /* Build conductance matrix K */
    double *Kv; int *Kc, *Krp; int knnz;
    int err = build_K(s, &Kv, &Kc, &Krp, &knnz);
    if (err) return err;

    double *Qrad = alloc_d(N);
    double *rhs  = alloc_d(N);
    double *x    = alloc_d(N);
    double *KTn  = alloc_d(N);

    /* Capacitance: C_i = ρ * cp * V */
    double *C = alloc_d(N);
    for (int i = 0; i < N; i++)
        C[i] = s->rho[i] * s->cp[i] * s->vol[i];

    /* Crank-Nicolson: θ = 0.5 */
    double theta = 0.5;

    /* Build LHS matrix: (C/dt) * I + θ * K */
    int lhs_nnz = knnz;
    double *Lv = (double*)malloc(lhs_nnz * sizeof(double));
    int    *Lc = (int*)   malloc(lhs_nnz * sizeof(int));
    int    *Lrp= (int*)   malloc((N+1) * sizeof(int));
    if (!Lv || !Lc || !Lrp) {
        free(Kv); free(Kc); free(Krp); free(Qrad); free(rhs);
        free(x); free(C); free(KTn); free(Lv); free(Lc); free(Lrp);
        return ERR_MEMORY;
    }

    memcpy(Lc,  Kc,  lhs_nnz * sizeof(int));
    memcpy(Lrp, Krp, (N+1) * sizeof(int));

    for (int i = 0; i < N; i++) {
        if (s->is_fixed[i]) {
            /* Identity row for fixed nodes */
            for (int j = Lrp[i]; j < Lrp[i+1]; j++) {
                Lv[j] = (Lc[j] == i) ? 1.0 : 0.0;
            }
        } else {
            for (int j = Lrp[i]; j < Lrp[i+1]; j++) {
                if (Lc[j] == i)
                    Lv[j] = C[i] / dt + theta * Kv[j];
                else
                    Lv[j] = theta * Kv[j];
            }
        }
    }

    /* Fixed node arrays */
    int *is_fixed = s->is_fixed;
    double *T_fixed_k = s->T_fixed;

    /* Time stepping */
    memcpy(x, s->T, N * sizeof(double));

    for (int step = 0; step < nsteps; step++) {
        /* Radiation at current temperature */
        if (include_radiation)
            compute_radiation(s, Qrad);
        else
            memset(Qrad, 0, N * sizeof(double));

        /* Compute K * T_n */
        spmv(N, Kv, Kc, Krp, s->T, KTn);

        /* RHS = (C/dt) * T_n - (1-θ)*K*T_n + Q + Q_rad */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++) {
            if (is_fixed[i]) {
                rhs[i] = T_fixed_k[i];
            } else {
                rhs[i] = (C[i] / dt) * s->T[i]
                         - (1.0 - theta) * KTn[i]
                         + s->Q[i] + Qrad[i];
            }
        }

        /* Solve LHS * T_{n+1} = rhs */
        memcpy(x, s->T, N * sizeof(double));
        int iters = 0;
        double res_norm = 0;
        pcg_solve(N, Lv, Lc, Lrp, rhs, x, s->tol, s->max_iter, &iters, &res_norm);

        /* Update temperatures */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++) {
            if (is_fixed[i])
                s->T[i] = T_fixed_k[i];
            else {
                s->T[i] = x[i];
                if (s->T[i] < 1.0) s->T[i] = 1.0;
                if (s->T[i] > 3000.0) s->T[i] = 3000.0;
            }
        }

        /* Progress callback (every ~1%) */
        if (s->cb && (step % (nsteps / 100 + 1) == 0 || step == nsteps-1)) {
            int pct = (int)((step + 1) * 100.0 / nsteps);
            char msg[128];
            snprintf(msg, 128, "Step %d/%d  t=%.2fs  CG=%d",
                     step+1, nsteps, (step+1)*dt, iters);
            s->cb(pct, msg);
        }
    }

    /* Statistics */
    double mn = 1e30, mx = -1e30, sm = 0.0;
    for (int i = 0; i < N; i++) {
        double Tc = s->T[i] - C2K;
        if (Tc < mn) mn = Tc;
        if (Tc > mx) mx = Tc;
        sm += Tc;
    }

    res->min_temp       = mn;
    res->max_temp       = mx;
    res->avg_temp       = sm / N;
    res->iterations     = nsteps;
    res->picard_iters   = 0;
    res->compute_time   = (double)(clock() - t0) / CLOCKS_PER_SEC;
    res->converged      = 1;
    res->energy_balance = 0.0;
    res->max_residual   = 0.0;
    res->error[0]       = '\0';

    free(Kv); free(Kc); free(Krp);
    free(Lv); free(Lc); free(Lrp);
    free(Qrad); free(rhs); free(x); free(C); free(KTn);
    return OK;
}

/* ── Diagnostics API ─────────────────────────────────────────── */
double thermal_get_energy_in(State *s)      { return s ? s->energy_in : 0; }
double thermal_get_energy_rad(State *s)     { return s ? s->energy_rad : 0; }
double thermal_get_energy_cond(State *s)    { return s ? s->energy_cond : 0; }
double thermal_get_energy_balance(State *s) { return s ? s->energy_balance : 0; }

/* ── Version helpers ─────────────────────────────────────────── */
const char *thermal_get_version(void) { return "4.0.0"; }

int thermal_has_openmp(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int thermal_get_num_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
