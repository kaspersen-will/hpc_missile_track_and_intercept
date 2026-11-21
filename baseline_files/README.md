# Real-time multi-radar tracking and impact prediction

HPC group project: William Kaspersen, Georgy Lifshitz, Xander Chen, Elshan Ragimov.

A ballistic target is observed by several radars at 10 Hz. A particle filter
tracks it (predict / weight / resample), and a Monte Carlo ensemble propagates
the particle cloud to the ground to predict the impact footprint. The goal is
to keep every update inside a **100 ms deadline** as particle count grows.

## Status

| Stage | State |
|---|---|
| Serial / OpenMP baseline (`src/pf_track.c`) | done, verified tracking (RMSE ~40 m over 6 seeds) |
| OpenMP thread sweep (`slurm/baseline_omp.sh`) | ready to run |
| MPI (distributed particles) | next |
| GPU (CUDA/OpenACC) experiment | toolchain verified on haskell + julia |
| Optimisation target | resampling is serial: parallel prefix-sum resampling |

## Layout

```
src/pf_track.c          tracker + impact prediction, stage timers, CSV output
src/test/saxpy.cu       GPU toolchain smoke test
slurm/                  Slurm scripts (submit from repo root)
scripts/                plotting / analysis
results/                CSV, job logs, plots
```

See [REPRODUCE.md](REPRODUCE.md) for exact commands.
