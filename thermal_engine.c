/*
 * TVAC Thermal Analyzer - Native Thermal Engine v3.0
 * ===================================================
 * High-performance thermal simulation in C.
 *
 * Features:
 *   - Sparse CSR matrix storage
 *   - Preconditioned Conjugate Gradient (Jacobi preconditioner)
 *   - Crank-Nicolson time integration for transient
 *   - Radiation heat transfer (Stefan-Boltzmann, top + bottom surfaces)
 *   - Picard iteration for radiation nonlinearity
 *   - Full double precision throughout
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o libthermal_engine.so thermal_engine.c -lm
 *
 * Author: TVAC Thermal Analysis Tool
 * Version: 3.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* ── Physical constants ──────────────────────────────────────────── */
#define SIGMA  5.670374419e-8   /* Stefan-Boltzmann  W/(m²·K⁴) */
#define C2K    273.15            /* Celsius → Kelvin offset       */

/* ── Error codes ─────────────────────────────────────────────────── */
#define OK                0
#define ERR_MEMORY       -1
#define ERR_INVALID      -2
#define ERR_CONVERGENCE  -3

/* ── Progress callback ───────────────────────────────────────────── */
typedef void (*progress_cb_t)(int percent, const char *msg);

/* ── Thermal state ───────────────────────────────────────────────── */
typedef struct {
    int   N;               /* total nodes                          */
    int   nx, ny, nz;      /* grid dimensions                      */

    /* per-node arrays (length N) */
    double *k;             /* thermal conductivity  W/(m·K)        */
    double *cp;            /* specific heat         J/(kg·K)       */
    double *rho;           /* density               kg/m³          */
    double *emiss;         /* surface emissivity                   */
    double *vol;           /* element volume        m³             */
    double *surf;          /* surface area exposed to radiation m² */
    double *Q;             /* heat source power     W              */

    int    *is_fixed;      /* 1 = fixed-temperature node           */
    double *T_fixed;       /* fixed temperature     K              */

    /* neighbour connectivity – CSR */
    int    *row_ptr;       /* length N+1                           */
    int    *col_idx;       /* length nnz                           */
    double *cond;          /* conductance values    W/K            */
    int     nnz;

    /* temperature state */
    double *T;             /* current temperature   K              */
    double *T_prev;        /* previous (for Picard / transient)    */

    /* solver parameters */
    double T_chamber;      /* chamber wall temp     K              */
    double tol;            /* convergence tolerance                */
    int    max_iter;       /* maximum CG iterations                */
    int    max_picard;     /* max Picard (radiation) iterations    */

    progress_cb_t cb;
} State;

/* ── Result structure ────────────────────────────────────────────── */
typedef struct {
    double min_temp;       /* °C */
    double max_temp;       /* °C */
    double avg_temp;       /* °C */
    int    iterations;
    double compute_time;   /* seconds */
    int    converged;
    char   error[256];
} Result;

/* =====================================================================
 *  Memory helpers
 * ===================================================================== */
static double *alloc_d(int n) { return (double*)calloc(n, sizeof(double)); }
static int    *alloc_i(int n) { return (int*)   calloc(n, sizeof(int));    }

/* =====================================================================
 *  API: create / destroy
 * ===================================================================== */
State *thermal_create_state(int N, int nx, int ny, int nz)
{
    State *s = (State*)calloc(1, sizeof(State));
    if (!s) return NULL;

    s->N  = N;
    s->nx = nx;  s->ny = ny;  s->nz = nz;

    s->k       = alloc_d(N);
    s->cp      = alloc_d(N);
    s->rho     = alloc_d(N);
    s->emiss   = alloc_d(N);
    s->vol     = alloc_d(N);
    s->surf    = alloc_d(N);
    s->Q       = alloc_d(N);

    s->is_fixed = alloc_i(N);
    s->T_fixed  = alloc_d(N);

    s->T       = alloc_d(N);
    s->T_prev  = alloc_d(N);

    /* defaults */
    s->T_chamber  = 25.0 + C2K;
    s->tol        = 1e-8;
    s->max_iter   = 20000;
    s->max_picard = 80;
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
    free(s->k);      free(s->cp);     free(s->rho);
    free(s->emiss);  free(s->vol);    free(s->surf);
    free(s->Q);      free(s->is_fixed); free(s->T_fixed);
    free(s->row_ptr); free(s->col_idx); free(s->cond);
    free(s->T);      free(s->T_prev);
    free(s);
}

/* =====================================================================
 *  API: node configuration
 * ===================================================================== */
void thermal_set_node(State *s, int i,
                      double k, double cp, double rho, double emiss,
                      double vol, double surf, double heat)
{
    if (i < 0 || i >= s->N) return;
    s->k[i]     = k;
    s->cp[i]    = cp;
    s->rho[i]   = rho;
    s->emiss[i] = emiss;
    s->vol[i]   = vol;
    s->surf[i]  = surf;
    s->Q[i]     = heat;
}

void thermal_set_fixed_temp(State *s, int i, double Tk)
{
    if (i < 0 || i >= s->N) return;
    s->is_fixed[i] = 1;
    s->T_fixed[i]  = Tk;
    s->T[i]        = Tk;
}

void thermal_set_initial_temp(State *s, int i, double Tk)
{
    if (i < 0 || i >= s->N) return;
    s->T[i] = Tk;
    s->T_prev[i] = Tk;
}

void thermal_set_chamber_temp(State *s, double Tk) { s->T_chamber = Tk; }

double thermal_get_temp(State *s, int i)
{
    if (i < 0 || i >= s->N) return 0.0;
    return s->T[i];
}

void thermal_set_progress_callback(State *s, progress_cb_t cb) { s->cb = cb; }

/* =====================================================================
 *  API: neighbour connectivity
 * ===================================================================== */
int thermal_alloc_neighbors(State *s, int total)
{
    s->nnz     = total;
    s->row_ptr = alloc_i(s->N + 1);
    s->col_idx = alloc_i(total);
    s->cond    = alloc_d(total);
    if (!s->row_ptr || !s->col_idx || !s->cond) return ERR_MEMORY;
    return OK;
}

void thermal_set_row_ptr(State *s, int row, int ptr)
{
    if (row >= 0 && row <= s->N) s->row_ptr[row] = ptr;
}

void thermal_set_neighbor(State *s, int node, int offset,
                          int nbr, double G)
{
    int idx = s->row_ptr[node] + offset;
    if (idx >= 0 && idx < s->nnz) {
        s->col_idx[idx] = nbr;
        s->cond[idx]    = G;
    }
}

/* =====================================================================
 *  Sparse matrix helpers
 * ===================================================================== */

/* Build the conductance matrix K in CSR.
 * K[i][i]  = Σ G_ij   (sum of outgoing conductances)
 * K[i][j]  = -G_ij
 * For fixed-temperature nodes we replace the row with identity.
 */
static int build_K(const State *s,
                   double **pv, int **pc, int **pr, int *pnnz)
{
    int N = s->N;

    /* count entries: each node gets 1 diagonal + #neighbours */
    int nnz = 0;
    for (int i = 0; i < N; i++) {
        if (s->is_fixed[i]) {
            nnz += 1;                          /* diagonal only */
        } else {
            int nb = s->row_ptr[i+1] - s->row_ptr[i];
            nnz += 1 + nb;                     /* diagonal + off-diag */
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
            /* identity row */
            val[pos] = 1.0;
            col[pos] = i;
            pos++;
            continue;
        }

        /* off-diagonal entries first, then diagonal */
        double diag = 0.0;
        int start = s->row_ptr[i];
        int end   = s->row_ptr[i+1];

        /* diagonal placeholder */
        int diag_pos = pos;
        val[pos] = 0.0;
        col[pos] = i;
        pos++;

        for (int jj = start; jj < end; jj++) {
            int   j = s->col_idx[jj];
            double G = s->cond[jj];
            if (G <= 0.0) continue;
            val[pos] = -G;
            col[pos] = j;
            pos++;
            diag += G;
        }
        val[diag_pos] = diag;
    }
    rp[N] = pos;

    *pv   = val;
    *pc   = col;
    *pr   = rp;
    *pnnz = pos;
    return OK;
}

/* SpMV: y = A*x */
static void spmv(int N, const double *v, const int *c, const int *rp,
                  const double *x, double *y)
{
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int j = rp[i]; j < rp[i+1]; j++)
            s += v[j] * x[c[j]];
        y[i] = s;
    }
}

/* Dot product */
static double dot(int N, const double *a, const double *b)
{
    double s = 0.0;
    for (int i = 0; i < N; i++) s += a[i] * b[i];
    return s;
}

/* =====================================================================
 *  Preconditioned Conjugate Gradient (Jacobi preconditioner)
 * ===================================================================== */
static int pcg_solve(int N,
                     const double *Av, const int *Ac, const int *Arp,
                     const double *b, double *x,
                     double tol, int max_iter, int *iters_out)
{
    double *r  = alloc_d(N);
    double *z  = alloc_d(N);
    double *p  = alloc_d(N);
    double *Ap = alloc_d(N);
    double *M  = alloc_d(N);  /* Jacobi preconditioner (inverse diag) */
    if (!r || !z || !p || !Ap || !M) {
        free(r); free(z); free(p); free(Ap); free(M);
        return ERR_MEMORY;
    }

    /* Build Jacobi preconditioner */
    for (int i = 0; i < N; i++) {
        double d = 0.0;
        for (int j = Arp[i]; j < Arp[i+1]; j++) {
            if (Ac[j] == i) { d = Av[j]; break; }
        }
        M[i] = (fabs(d) > 1e-30) ? 1.0 / d : 1.0;
    }

    /* r = b - A*x */
    spmv(N, Av, Ac, Arp, x, r);
    for (int i = 0; i < N; i++) r[i] = b[i] - r[i];

    /* z = M*r */
    for (int i = 0; i < N; i++) z[i] = M[i] * r[i];

    memcpy(p, z, N * sizeof(double));

    double rz = dot(N, r, z);
    double r0_norm = sqrt(dot(N, r, r));
    if (r0_norm < 1e-30) { /* already solved */
        *iters_out = 0;
        free(r); free(z); free(p); free(Ap); free(M);
        return OK;
    }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        spmv(N, Av, Ac, Arp, p, Ap);

        double pAp = dot(N, p, Ap);
        if (fabs(pAp) < 1e-30) break;

        double alpha = rz / pAp;

        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double r_norm = sqrt(dot(N, r, r));
        if (r_norm < tol * r0_norm + 1e-15) break;

        for (int i = 0; i < N; i++) z[i] = M[i] * r[i];

        double rz_new = dot(N, r, z);
        double beta = rz_new / (fabs(rz) > 1e-30 ? rz : 1e-30);
        rz = rz_new;

        for (int i = 0; i < N; i++)
            p[i] = z[i] + beta * p[i];
    }

    *iters_out = iter;
    free(r); free(z); free(p); free(Ap); free(M);
    return OK;
}

/* =====================================================================
 *  Compute radiation load vector
 *  Q_rad[i] = ε·σ·A·(T_chamber⁴ - T_i⁴)
 * ===================================================================== */
static void compute_radiation(const State *s, double *Qr)
{
    double Tw4 = s->T_chamber * s->T_chamber;
    Tw4 *= Tw4;

    for (int i = 0; i < s->N; i++) {
        if (s->is_fixed[i] || s->surf[i] <= 0.0 || s->emiss[i] <= 0.0) {
            Qr[i] = 0.0;
            continue;
        }
        double Ti4 = s->T[i] * s->T[i];
        Ti4 *= Ti4;
        Qr[i] = s->emiss[i] * SIGMA * s->surf[i] * (Tw4 - Ti4);
    }
}

/* =====================================================================
 *  Steady-state solver (Picard iteration for radiation)
 * ===================================================================== */
int thermal_solve_steady(State *s, Result *res, int include_radiation)
{
    clock_t t0 = clock();
    int N = s->N;

    /* Build conductance matrix */
    double *Kv; int *Kc, *Krp; int knnz;
    int err = build_K(s, &Kv, &Kc, &Krp, &knnz);
    if (err) {
        snprintf(res->error, 256, "Failed to build conductance matrix");
        return err;
    }

    double *rhs  = alloc_d(N);
    double *Qrad = alloc_d(N);
    double *T_solve = alloc_d(N);  /* temp buffer for CG output */
    if (!rhs || !Qrad || !T_solve) {
        free(Kv); free(Kc); free(Krp); free(rhs); free(Qrad); free(T_solve);
        return ERR_MEMORY;
    }

    int total_iters = 0;
    int converged = 0;

    /* Step 1: Solve WITHOUT radiation first to get a good initial guess */
    {
        for (int i = 0; i < N; i++) {
            if (s->is_fixed[i])
                rhs[i] = s->T_fixed[i];
            else
                rhs[i] = s->Q[i];
        }
        int iters = 0;
        pcg_solve(N, Kv, Kc, Krp, rhs, s->T, s->tol, s->max_iter, &iters);
        total_iters += iters;

        /* Enforce fixed temps and clamp */
        for (int i = 0; i < N; i++) {
            if (s->is_fixed[i]) s->T[i] = s->T_fixed[i];
            if (s->T[i] < 1.0)    s->T[i] = 1.0;
            if (s->T[i] > 2000.0) s->T[i] = 2000.0;
        }
    }

    if (!include_radiation) {
        converged = 1;
        goto done;
    }

    /* Step 2: Picard iteration for radiation nonlinearity */
    {
        int n_picard = s->max_picard;
        double relax = 0.5;  /* under-relaxation for stability */

        for (int pic = 0; pic < n_picard; pic++) {
            /* Save previous temperatures */
            memcpy(s->T_prev, s->T, N * sizeof(double));

            /* Radiation load using current temperatures */
            compute_radiation(s, Qrad);

            /* Build RHS: Q + Q_rad */
            for (int i = 0; i < N; i++) {
                if (s->is_fixed[i])
                    rhs[i] = s->T_fixed[i];
                else
                    rhs[i] = s->Q[i] + Qrad[i];
            }

            /* Copy current T as initial guess for CG */
            memcpy(T_solve, s->T, N * sizeof(double));

            /* Solve K * T = rhs */
            int iters = 0;
            err = pcg_solve(N, Kv, Kc, Krp, rhs, T_solve, s->tol, s->max_iter, &iters);
            total_iters += iters;

            /* Under-relaxation: T = relax * T_new + (1-relax) * T_prev */
            for (int i = 0; i < N; i++) {
                if (s->is_fixed[i]) {
                    s->T[i] = s->T_fixed[i];
                } else {
                    s->T[i] = relax * T_solve[i] + (1.0 - relax) * s->T_prev[i];
                    /* Clamp to physical range */
                    if (s->T[i] < 1.0)    s->T[i] = 1.0;
                    if (s->T[i] > 2000.0) s->T[i] = 2000.0;
                }
            }

            /* Check Picard convergence */
            double max_diff = 0.0;
            for (int i = 0; i < N; i++) {
                double d = fabs(s->T[i] - s->T_prev[i]);
                if (d > max_diff) max_diff = d;
            }

            /* Increase relaxation as we converge */
            if (max_diff < 1.0) {
                relax += 0.05;
                if (relax > 0.9) relax = 0.9;
            }

            if (s->cb) {
                int pct = (int)((pic + 1) * 100.0 / n_picard);
                char msg[64];
                snprintf(msg, 64, "Picard %d/%d, delta=%.4e K", pic+1, n_picard, max_diff);
                s->cb(pct, msg);
            }

            if (max_diff < s->tol * 10.0 && pic > 0) {
                converged = 1;
                break;
            }
        }
    }

done:

    /* Statistics in Celsius */
    {
        double mn = 1e30, mx = -1e30, sm = 0.0;
        for (int i = 0; i < N; i++) {
            double Tc = s->T[i] - C2K;
            if (Tc < mn) mn = Tc;
            if (Tc > mx) mx = Tc;
            sm += Tc;
        }

        res->min_temp     = mn;
        res->max_temp     = mx;
        res->avg_temp     = sm / N;
        res->iterations   = total_iters;
        res->compute_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
        res->converged    = converged;
        res->error[0]     = '\0';
    }

    free(Kv); free(Kc); free(Krp); free(rhs); free(Qrad); free(T_solve);
    return OK;
}

/* =====================================================================
 *  Transient solver (Crank-Nicolson)
 * ===================================================================== */
int thermal_solve_transient(State *s, Result *res,
                            double duration, double dt,
                            int include_radiation)
{
    clock_t t0 = clock();
    int N = s->N;
    int nsteps = (int)(duration / dt + 0.5);
    if (nsteps < 1) nsteps = 1;

    /* Build conductance matrix K */
    double *Kv; int *Kc, *Krp; int knnz;
    int err = build_K(s, &Kv, &Kc, &Krp, &knnz);
    if (err) return err;

    double *Qrad = alloc_d(N);
    double *rhs  = alloc_d(N);
    double *x    = alloc_d(N);

    /* Build LHS matrix: (C/dt)*I + 0.5*K  in CSR */
    /* We modify K in-place to become LHS, and compute RHS each step */
    /* Actually, build separate LHS CSR with same structure as K but
       with diagonal += C/dt, and off-diag *= 0.5 */

    /* First compute capacitance C_i = rho*cp*vol */
    double *C = alloc_d(N);
    for (int i = 0; i < N; i++)
        C[i] = s->rho[i] * s->cp[i] * s->vol[i];

    /* LHS = C/dt*I + theta*K  (theta=0.5 for Crank-Nicolson) */
    double theta = 0.5;
    int lhs_nnz = knnz;
    double *Lv = (double*)malloc(lhs_nnz * sizeof(double));
    int    *Lc = (int*)   malloc(lhs_nnz * sizeof(int));
    int    *Lrp= (int*)   malloc((N+1) * sizeof(int));
    if (!Lv || !Lc || !Lrp) {
        free(Kv); free(Kc); free(Krp); free(Qrad); free(rhs);
        free(x); free(C); free(Lv); free(Lc); free(Lrp);
        return ERR_MEMORY;
    }

    memcpy(Lc,  Kc,  lhs_nnz * sizeof(int));
    memcpy(Lrp, Krp, (N+1) * sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j = Lrp[i]; j < Lrp[i+1]; j++) {
            if (Lc[j] == i)
                Lv[j] = C[i] / dt + theta * Kv[j];  /* diagonal */
            else
                Lv[j] = theta * Kv[j];               /* off-diag */
        }
        /* Fixed temp nodes: identity row */
        if (s->is_fixed[i]) {
            for (int j = Lrp[i]; j < Lrp[i+1]; j++) {
                Lv[j] = (Lc[j] == i) ? 1.0 : 0.0;
            }
        }
    }

    /* Time stepping */
    memcpy(x, s->T, N * sizeof(double));

    for (int step = 0; step < nsteps; step++) {
        /* Radiation at current temperature */
        if (include_radiation)
            compute_radiation(s, Qrad);
        else
            memset(Qrad, 0, N * sizeof(double));

        /* RHS = (C/dt - (1-theta)*K) * T_n + Q + Q_rad */
        /* First compute K * T_n */
        double *KTn = alloc_d(N);
        spmv(N, Kv, Kc, Krp, s->T, KTn);

        for (int i = 0; i < N; i++) {
            if (s->is_fixed[i]) {
                rhs[i] = s->T_fixed[i];
            } else {
                rhs[i] = (C[i] / dt) * s->T[i] - (1.0 - theta) * KTn[i]
                         + s->Q[i] + Qrad[i];
            }
        }
        free(KTn);

        /* Solve LHS * T_{n+1} = rhs */
        memcpy(x, s->T, N * sizeof(double));
        int iters = 0;
        pcg_solve(N, Lv, Lc, Lrp, rhs, x, s->tol, s->max_iter, &iters);

        /* Enforce fixed temps & update state */
        for (int i = 0; i < N; i++) {
            if (s->is_fixed[i])
                s->T[i] = s->T_fixed[i];
            else
                s->T[i] = x[i];
        }

        if (s->cb && step % (nsteps / 100 + 1) == 0) {
            int pct = (int)(step * 100.0 / nsteps);
            char msg[64];
            snprintf(msg, 64, "Step %d/%d  t=%.2fs", step+1, nsteps, (step+1)*dt);
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

    res->min_temp     = mn;
    res->max_temp     = mx;
    res->avg_temp     = sm / N;
    res->iterations   = nsteps;
    res->compute_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
    res->converged    = 1;
    res->error[0]     = '\0';

    free(Kv); free(Kc); free(Krp);
    free(Lv); free(Lc); free(Lrp);
    free(Qrad); free(rhs); free(x); free(C);
    return OK;
}

/* ── Version helpers ─────────────────────────────────────────────── */
const char *thermal_get_version(void)  { return "3.0.0"; }
int  thermal_has_openmp(void)          { return 0; }
int  thermal_get_num_threads(void)     { return 1; }
