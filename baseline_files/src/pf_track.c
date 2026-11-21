/*
 * pf_track.c - Multi-radar ballistic target tracking + impact prediction.
 *
 * Baseline (serial / OpenMP) version of the group application.
 *
 *   1. Truth: a ballistic re-entry vehicle (gravity + exponential-atmosphere
 *      drag) flying in a flat-earth local frame (x east, y north, z up, metres).
 *   2. Sensors: R ground radars, each giving range / azimuth / elevation with
 *      Gaussian noise at 10 Hz.
 *   3. Tracker: bootstrap (SIR) particle filter, 6-D state [x y z vx vy vz]:
 *        predict  -> propagate every particle one step + process noise
 *        weight   -> log-likelihood of all radar measurements
 *        resample -> systematic resampling when ESS < N/2
 *   4. Impact prediction: every K updates, draw M particles and integrate each
 *      to the ground (RK4). Mean impact point + spread = predicted footprint.
 *
 * Every stage is timed separately so the per-stage cost (and Amdahl
 * behaviour) can be read straight from the CSV.
 *
 * PARALLELISM IN THIS VERSION
 *   predict, weight, ESS, impact ensemble -> OpenMP parallel for
 *   resample                               -> SERIAL on purpose: it is the
 *       planned optimisation target (parallel prefix-sum resampling).
 *
 * RANDOM NUMBERS
 *   Counter-based (hash of seed, step, particle, component), so each particle
 *   gets the same random numbers whatever the thread count. Results differ
 *   between thread counts only through floating-point reduction order.
 *
 * Build:  gcc -O3 -march=native -fopenmp -std=c11 pf_track.c -lm -o pf_track
 * Run:    OMP_NUM_THREADS=8 ./pf_track --particles 100000 --steps 200
 *         ./pf_track --header     (prints the CSV header only)
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
#define G0        9.80665     /* m/s^2                                  */
#define RHO0      1.225       /* kg/m^3 sea-level density               */
#define H_SCALE   7200.0      /* m, atmosphere scale height             */
#define BETA      5000.0      /* kg/m^2 ballistic coefficient           */
#define DT        0.1         /* s, radar update period (10 Hz)         */
#define DT_IMPACT 0.1         /* s, integrator step for impact ensemble */
#define SIG_R     50.0        /* m, range noise                         */
#define SIG_ANG   2.0e-3      /* rad, azimuth/elevation noise           */
#define SIG_ACC   3.0         /* m/s^2, process (unmodelled accel) noise*/
#define PRIOR_POS 200.0       /* m, initial position uncertainty        */
#define PRIOR_VEL 30.0        /* m/s, initial velocity uncertainty      */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MAX_RADARS 64
#define DEADLINE_MS 100.0

typedef struct { double s[6]; } State;          /* x y z vx vy vz */
typedef struct { double x, y, z; } Vec3;
typedef struct { double r, az, el; } Meas;

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */
static double wtime(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif
}

/* ------------------------------------------------------------------ */
/* Counter-based RNG (splitmix64 finaliser)                            */
/* ------------------------------------------------------------------ */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t rng_key(uint64_t seed, uint64_t stream,
                               uint64_t step, uint64_t idx) {
    return mix64(mix64(mix64(seed ^ (stream << 56)) ^ step) ^ idx);
}
/* uniform in (0,1) */
static inline double u01(uint64_t key, uint64_t c) {
    return ((mix64(key + c) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
/* standard normal, Box-Muller; c selects the pair */
static inline double gauss(uint64_t key, uint64_t c) {
    double u1 = u01(key, 2 * c), u2 = u01(key, 2 * c + 1);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
enum { S_TRUTHNOISE = 1, S_INIT, S_PROC, S_RESAMPLE, S_IMPACT, S_MEAS, S_ROUGH };

/* ------------------------------------------------------------------ */
/* Physics                                                             */
/* ------------------------------------------------------------------ */
static inline void deriv(const double *s, double *d) {
    double vx = s[3], vy = s[4], vz = s[5];
    double z = s[2] > 0.0 ? s[2] : 0.0;
    double rho = RHO0 * exp(-z / H_SCALE);
    double v = sqrt(vx * vx + vy * vy + vz * vz);
    double k = -0.5 * rho * v / BETA;
    d[0] = vx; d[1] = vy; d[2] = vz;
    d[3] = k * vx; d[4] = k * vy; d[5] = k * vz - G0;
}

static inline void rk4(double *s, double h) {
    double k1[6], k2[6], k3[6], k4[6], t[6];
    deriv(s, k1);
    for (int i = 0; i < 6; ++i) t[i] = s[i] + 0.5 * h * k1[i];
    deriv(t, k2);
    for (int i = 0; i < 6; ++i) t[i] = s[i] + 0.5 * h * k2[i];
    deriv(t, k3);
    for (int i = 0; i < 6; ++i) t[i] = s[i] + h * k3[i];
    deriv(t, k4);
    for (int i = 0; i < 6; ++i)
        s[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

/* Integrate to z = 0; returns impact (x,y) via linear interpolation. */
static int propagate_to_ground(double *s, double h, double *ix, double *iy) {
    for (int it = 0; it < 200000; ++it) {
        double prev[6];
        memcpy(prev, s, sizeof prev);
        rk4(s, h);
        if (s[2] <= 0.0) {
            double f = prev[2] / (prev[2] - s[2]);
            *ix = prev[0] + f * (s[0] - prev[0]);
            *iy = prev[1] + f * (s[1] - prev[1]);
            return 1;
        }
    }
    return 0;
}

static inline Meas measure(const double *s, Vec3 r) {
    double dx = s[0] - r.x, dy = s[1] - r.y, dz = s[2] - r.z;
    double h = sqrt(dx * dx + dy * dy);
    Meas m = { sqrt(h * h + dz * dz), atan2(dy, dx), atan2(dz, h) };
    return m;
}

static inline double wrap_pi(double a) {
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

/* ------------------------------------------------------------------ */
/* Options                                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    long n_particles;
    int steps, n_radars, impact_every;
    long ensemble;
    uint64_t seed;
    const char *label;
} Opts;

static void usage(const char *p) {
    fprintf(stderr,
        "usage: %s [--particles N] [--steps S] [--radars R] [--ensemble M]\n"
        "          [--impact-every K] [--seed X] [--label TEXT] [--header]\n", p);
}

static const char *CSV_HEADER =
    "label,threads,particles,steps,radars,ensemble,impact_every,seed,"
    "t_total,t_predict,t_weight,t_resample,t_impact,n_resample,"
    "cycle_mean_ms,cycle_max_ms,deadline_frac,"
    "rmse_pos_m,final_pos_err_m,impact_err_m,impact_spread_m";

int main(int argc, char **argv) {
    Opts o = { 100000, 200, 3, 10, 2000, 12345, "run" };
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "--header")) { puts(CSV_HEADER); return 0; }
        if (i + 1 >= argc) { usage(argv[0]); return 1; }
        if      (!strcmp(a, "--particles"))    o.n_particles  = atol(argv[++i]);
        else if (!strcmp(a, "--steps"))        o.steps        = atoi(argv[++i]);
        else if (!strcmp(a, "--radars"))       o.n_radars     = atoi(argv[++i]);
        else if (!strcmp(a, "--ensemble"))     o.ensemble     = atol(argv[++i]);
        else if (!strcmp(a, "--impact-every")) o.impact_every = atoi(argv[++i]);
        else if (!strcmp(a, "--seed"))         o.seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(a, "--label"))        o.label = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (o.n_radars < 1 || o.n_radars > MAX_RADARS || o.n_particles < 2 ||
        o.steps < 1 || o.ensemble < 1 || o.impact_every < 1) {
        usage(argv[0]); return 1;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const long N = o.n_particles;

    /* Radars on a 40 km ring around the defended point (origin). */
    Vec3 radar[MAX_RADARS];
    for (int k = 0; k < o.n_radars; ++k) {
        double a = 2 * M_PI * k / o.n_radars + 0.3;
        radar[k] = (Vec3){ 40e3 * cos(a), 40e3 * sin(a), 0.0 };
    }

    /* Truth trajectory: deterministic physics, fine RK4 sub-steps. */
    double truth0[6] = { -120e3, 10e3, 70e3, 2500.0, -150.0, -800.0 };
    double *truth = malloc(sizeof(double) * 6 * (o.steps + 1));
    memcpy(truth, truth0, sizeof truth0);
    for (int t = 1; t <= o.steps; ++t) {
        memcpy(truth + 6 * t, truth + 6 * (t - 1), sizeof truth0);
        for (int k = 0; k < 10; ++k) rk4(truth + 6 * t, DT / 10);
        if (truth[6 * t + 2] <= 0) {
            fprintf(stderr, "target hit the ground at step %d; reduce --steps\n", t);
            return 1;
        }
    }
    /* Noisy measurements for every step / radar (generated up front so the
       tracker's timing excludes sensor simulation). */
    Meas *meas = malloc(sizeof(Meas) * o.steps * o.n_radars);
    for (int t = 1; t <= o.steps; ++t)
        for (int k = 0; k < o.n_radars; ++k) {
            uint64_t key = rng_key(o.seed, S_MEAS, t, k);
            Meas m = measure(truth + 6 * t, radar[k]);
            m.r  += SIG_R   * gauss(key, 0);
            m.az += SIG_ANG * gauss(key, 1);
            m.el += SIG_ANG * gauss(key, 2);
            meas[(t - 1) * o.n_radars + k] = m;
        }
    /* True impact point (continue the truth from the last tracked state). */
    double tix = 0, tiy = 0, tmp[6];
    memcpy(tmp, truth + 6 * o.steps, sizeof tmp);
    propagate_to_ground(tmp, DT_IMPACT, &tix, &tiy);

    /* Particle buffers. */
    State  *p    = malloc(sizeof(State) * N);
    State  *pnew = malloc(sizeof(State) * N);
    double *logw = malloc(sizeof(double) * N);
    double *w    = malloc(sizeof(double) * N);
    double *cdf  = malloc(sizeof(double) * N);
    if (!p || !pnew || !logw || !w || !cdf) { fprintf(stderr, "out of memory\n"); return 1; }

    /* Prior: cueing track = truth + one random offset, then spread. */
    const double psig[6] = { PRIOR_POS, PRIOR_POS, PRIOR_POS,
                             PRIOR_VEL, PRIOR_VEL, PRIOR_VEL };
    double prior[6];
    { uint64_t key = rng_key(o.seed, S_TRUTHNOISE, 0, 0);
      for (int i = 0; i < 6; ++i) prior[i] = truth0[i] + psig[i] * gauss(key, i); }

    double t0 = wtime();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) {
        uint64_t key = rng_key(o.seed, S_INIT, 0, i);
        for (int c = 0; c < 6; ++c) p[i].s[c] = prior[c] + psig[c] * gauss(key, c);
        w[i] = 1.0 / N;
    }

    double t_pred = 0, t_wt = 0, t_res = 0, t_imp = 0;
    double cyc_sum = 0, cyc_max = 0, se_sum = 0, final_err = 0;
    double last_ix = 0, last_iy = 0, last_spread = 0;
    int n_res = 0, n_ok = 0;
    const double inv_r2 = 1.0 / (SIG_R * SIG_R), inv_a2 = 1.0 / (SIG_ANG * SIG_ANG);
    const double qv = SIG_ACC * DT, qp = 0.5 * SIG_ACC * DT * DT;
    /* Optimal Gaussian-kernel bandwidth for d = 6 dimensions. */
    const double h_opt = pow(4.0 / (N * (6 + 2.0)), 1.0 / (6 + 4.0));

    for (int t = 1; t <= o.steps; ++t) {
        double c0 = wtime();

        /* ---- predict -------------------------------------------- */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < N; ++i) {
            uint64_t key = rng_key(o.seed, S_PROC, t, i);
            double *s = p[i].s;
            rk4(s, DT);
            for (int c = 0; c < 3; ++c) {
                double n = gauss(key, c);
                s[c]     += qp * n;
                s[c + 3] += qv * n;
            }
        }
        double c1 = wtime();

        /* ---- weight (log domain, then normalise) ----------------- */
        const Meas *mt = meas + (t - 1) * o.n_radars;
        double maxlw = -INFINITY;
        #pragma omp parallel for schedule(static) reduction(max:maxlw)
        for (long i = 0; i < N; ++i) {
            double lw = log(w[i]);
            for (int k = 0; k < o.n_radars; ++k) {
                Meas h = measure(p[i].s, radar[k]);
                double dr = h.r - mt[k].r;
                double da = wrap_pi(h.az - mt[k].az);
                double de = h.el - mt[k].el;
                lw += -0.5 * (dr * dr * inv_r2 + (da * da + de * de) * inv_a2);
            }
            logw[i] = lw;
            if (lw > maxlw) maxlw = lw;
        }
        if (!isfinite(maxlw)) {
            fprintf(stderr, "filter collapsed at step %d (all weights zero)\n", t);
            return 2;
        }
        double wsum = 0;
        #pragma omp parallel for schedule(static) reduction(+:wsum)
        for (long i = 0; i < N; ++i) { w[i] = exp(logw[i] - maxlw); wsum += w[i]; }
        double w2 = 0, mx = 0, my = 0, mz = 0;
        #pragma omp parallel for schedule(static) reduction(+:w2,mx,my,mz)
        for (long i = 0; i < N; ++i) {
            w[i] /= wsum;
            w2 += w[i] * w[i];
            mx += w[i] * p[i].s[0]; my += w[i] * p[i].s[1]; mz += w[i] * p[i].s[2];
        }
        double ess = 1.0 / w2;
        double c2 = wtime();

        /* ---- resample (systematic, SERIAL baseline) --------------- */
        int resampled = ess < 0.5 * N;
        if (resampled) {
            cdf[0] = w[0];
            for (long i = 1; i < N; ++i) cdf[i] = cdf[i - 1] + w[i];
            cdf[N - 1] = 1.0;
            double u0 = u01(rng_key(o.seed, S_RESAMPLE, t, 0), 0) / N;
            long j = 0;
            for (long i = 0; i < N; ++i) {
                double u = u0 + (double)i / N;
                while (cdf[j] < u) ++j;
                pnew[i] = p[j];
            }
            State *sw = p; p = pnew; pnew = sw;
            for (long i = 0; i < N; ++i) w[i] = 1.0 / N;
            ++n_res;

            /* Regularisation (roughening): resampling duplicates particles,
               so add Gaussian kernel jitter scaled to the cloud's spread
               (Silverman bandwidth, diagonal covariance). Without this the
               velocity diversity collapses and the track diverges. */
            double m0=0,m1=0,m2=0,m3=0,m4=0,m5=0, q0=0,q1=0,q2=0,q3=0,q4=0,q5=0;
            #pragma omp parallel for schedule(static) \
                reduction(+:m0,m1,m2,m3,m4,m5,q0,q1,q2,q3,q4,q5)
            for (long i = 0; i < N; ++i) {
                const double *s = p[i].s;
                m0 += s[0]; m1 += s[1]; m2 += s[2]; m3 += s[3]; m4 += s[4]; m5 += s[5];
                q0 += s[0]*s[0]; q1 += s[1]*s[1]; q2 += s[2]*s[2];
                q3 += s[3]*s[3]; q4 += s[4]*s[4]; q5 += s[5]*s[5];
            }
            double mm[6] = {m0,m1,m2,m3,m4,m5}, qq[6] = {q0,q1,q2,q3,q4,q5}, sd[6];
            for (int c = 0; c < 6; ++c) {
                double mean = mm[c] / N, var = qq[c] / N - mean * mean;
                sd[c] = h_opt * sqrt(var > 0 ? var : 0);
            }
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < N; ++i) {
                uint64_t key = rng_key(o.seed, S_ROUGH, t, i);
                for (int c = 0; c < 6; ++c) p[i].s[c] += sd[c] * gauss(key, c);
            }
        }
        double c3 = wtime();

        /* ---- impact prediction (Monte Carlo ensemble) ------------ */
        double c4 = c3;
        if (t % o.impact_every == 0 || t == o.steps) {
            /* After resampling weights are uniform: pick any index.
               Otherwise draw by weight through the CDF. */
            int weights_uniform = resampled;
            if (!weights_uniform) {
                cdf[0] = w[0];
                for (long i = 1; i < N; ++i) cdf[i] = cdf[i - 1] + w[i];
                cdf[N - 1] = 1.0;
            }
            double sx = 0, sy = 0, sxx = 0, syy = 0; long hit = 0;
            #pragma omp parallel for schedule(dynamic, 64) reduction(+:sx,sy,sxx,syy,hit)
            for (long m = 0; m < o.ensemble; ++m) {
                double u = u01(rng_key(o.seed, S_IMPACT, t, m), 0);
                long idx;
                if (weights_uniform) {
                    idx = (long)(u * N); if (idx >= N) idx = N - 1;
                } else {               /* binary search in CDF */
                    long lo = 0, hi = N - 1;
                    while (lo < hi) { long mid = (lo + hi) / 2;
                                      if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
                    idx = lo;
                }
                double s[6], ix, iy;
                memcpy(s, p[idx].s, sizeof s);
                if (propagate_to_ground(s, DT_IMPACT, &ix, &iy)) {
                    sx += ix; sy += iy; sxx += ix * ix; syy += iy * iy; ++hit;
                }
            }
            if (hit > 0) {
                last_ix = sx / hit; last_iy = sy / hit;
                double vx = sxx / hit - last_ix * last_ix;
                double vy = syy / hit - last_iy * last_iy;
                last_spread = sqrt((vx > 0 ? vx : 0) + (vy > 0 ? vy : 0));
            }
            c4 = wtime();
        }

        t_pred += c1 - c0; t_wt += c2 - c1; t_res += c3 - c2; t_imp += c4 - c3;
        double cyc = 1e3 * (c4 - c0);
        cyc_sum += cyc; if (cyc > cyc_max) cyc_max = cyc;
        if (cyc <= DEADLINE_MS) ++n_ok;

        const double *tr = truth + 6 * t;
        double e2 = (mx - tr[0]) * (mx - tr[0]) + (my - tr[1]) * (my - tr[1]) +
                    (mz - tr[2]) * (mz - tr[2]);
        se_sum += e2;
        if (t == o.steps) final_err = sqrt(e2);
    }
    double t_total = wtime() - t0;

    double imp_err = hypot(last_ix - tix, last_iy - tiy);
    printf("%s,%d,%ld,%d,%d,%ld,%d,%llu,"
           "%.6f,%.6f,%.6f,%.6f,%.6f,%d,"
           "%.3f,%.3f,%.4f,%.2f,%.2f,%.2f,%.2f\n",
           o.label, threads, N, o.steps, o.n_radars, o.ensemble, o.impact_every,
           (unsigned long long)o.seed,
           t_total, t_pred, t_wt, t_res, t_imp, n_res,
           cyc_sum / o.steps, cyc_max, (double)n_ok / o.steps,
           sqrt(se_sum / o.steps), final_err, imp_err, last_spread);

    fprintf(stderr,
        "[pf_track] threads=%d N=%ld steps=%d radars=%d ensemble=%ld\n"
        "  total %.3f s | predict %.3f | weight %.3f | resample %.3f (%d x) | impact %.3f\n"
        "  cycle mean %.2f ms, max %.2f ms, %.1f%% within %.0f ms\n"
        "  pos RMSE %.1f m, final %.1f m | impact err %.1f m, spread %.1f m "
        "(true impact %.0f, %.0f)\n",
        threads, N, o.steps, o.n_radars, o.ensemble,
        t_total, t_pred, t_wt, t_res, n_res, t_imp,
        cyc_sum / o.steps, cyc_max, 100.0 * n_ok / o.steps, DEADLINE_MS,
        sqrt(se_sum / o.steps), final_err, imp_err, last_spread, tix, tiy);

    free(truth); free(meas); free(p); free(pnew); free(logw); free(w); free(cdf);
    return 0;
}
