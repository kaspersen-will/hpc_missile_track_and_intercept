/*
 * test_resample.c - correctness check for the parallel resampling kernels.
 *
 * Compares, on random weights (uniform and very peaked):
 *   1. the parallel blocked prefix sum vs the serial running sum (max |diff|);
 *   2. the parallel systematic walk vs the serial walk on the SAME cdf
 *      (must pick identical source particles: 0 mismatches);
 *   3. end to end: each mode with its own cdf (mismatches only where the
 *      rounding difference moves a cdf entry across a sampling point).
 *
 * Build/run (from repo root):
 *   gcc -O3 -march=native -fopenmp -std=c11 src/test/test_resample.c -lm -o build/test_resample
 *   OMP_NUM_THREADS=32 ./build/test_resample
 */
#define main pf_track_main
#include "../pf_track.c"
#undef main

static long first_mismatch(const State *a, const State *b, long N, long *count) {
    long first = -1; *count = 0;
    for (long i = 0; i < N; ++i)
        if (memcmp(&a[i], &b[i], sizeof(State))) { if (first < 0) first = i; ++*count; }
    return first;
}

int main(void) {
    const long N = 1000000;
    int T = omp_get_max_threads(), fails = 0;
    State *p = malloc(sizeof(State) * N), *ps = malloc(sizeof(State) * N),
          *pp = malloc(sizeof(State) * N);
    double *w = malloc(sizeof(double) * N), *cs = malloc(sizeof(double) * N),
           *cp = malloc(sizeof(double) * N), *part = malloc(sizeof(double) * (T + 1));
    for (long i = 0; i < N; ++i) p[i].s[0] = (double)i;      /* tag = index */

    const char *name[2] = { "uniform-ish", "peaked (ESS ~ 1%)" };
    for (int kind = 0; kind < 2; ++kind) {
        double sum = 0;
        for (long i = 0; i < N; ++i) {
            double u = u01(rng_key(7, 99, kind, i), 0);
            w[i] = kind == 0 ? u : exp(-0.5 * pow((u - 0.5) / 0.004, 2));
            sum += w[i];
        }
        for (long i = 0; i < N; ++i) w[i] /= sum;
        double u0 = u01(rng_key(7, 98, kind, 0), 0) / N;

        build_cdf(w, cs, N, 0, part);
        build_cdf(w, cp, N, 1, part);
        double dmax = 0;
        for (long i = 0; i < N; ++i) { double d = fabs(cs[i] - cp[i]); if (d > dmax) dmax = d; }

        long n_same, n_e2e;
        resample_systematic(p, ps, cs, N, u0, 0);
        resample_systematic(p, pp, cs, N, u0, 1);            /* same cdf  */
        first_mismatch(ps, pp, N, &n_same);
        resample_systematic(p, pp, cp, N, u0, 1);            /* own cdf   */
        first_mismatch(ps, pp, N, &n_e2e);

        printf("[%s] threads=%d N=%ld\n", name[kind], T, N);
        printf("  cdf max |serial - parallel|      = %.3e\n", dmax);
        printf("  walk, same cdf: mismatches       = %ld   %s\n", n_same, n_same ? "FAIL" : "ok");
        printf("  end to end:     mismatches       = %ld of %ld (%.4f%%)\n",
               n_e2e, N, 100.0 * n_e2e / N);
        if (n_same || dmax > 1e-12) fails = 1;
    }
    puts(fails ? "RESULT: FAIL" : "RESULT: PASS");
    free(p); free(ps); free(pp); free(w); free(cs); free(cp); free(part);
    return fails;
}
