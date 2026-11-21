# Reproducing the experiments

All commands run from the repository root on the SciTech cluster.
`results/` must exist before `sbatch`, otherwise Slurm cannot write the log.

## 1. Baseline + OpenMP thread sweep (one node)

```bash
mkdir -p results
sbatch slurm/baseline_omp.sh             # ~10-20 min of compute, 64 cores
squeue -u $USER                          # wait for it to finish
python3 scripts/plot_baseline.py results/baseline_<jobid>.csv
```

What the job runs (all on one node, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`):

| label | particles | threads | purpose |
|---|---|---|---|
| `strong_1e5` | 100 000 | 1..64 | strong scaling, reduced problem (baseline = 1 thread) |
| `strong_1e6` | 1 000 000 | 1..64 | same at 10x size (problem-size sensitivity) |
| `radars_3/6/12` | 100 000 | 16 | sensitivity to sensor count (weight-stage cost, accuracy) |

Fixed: 200 tracking steps (20 s at 10 Hz), impact ensemble 2000, seed 12345, 3 repeats.
The job log records the node, git commit, compiler, modules, CPU model and `CFLAGS`.

## 2. GPU smoke test (toolchain check only)

```bash
sbatch slurm/ncu_test.sh                       # any GPU node
sbatch --nodelist=julia slurm/ncu_test.sh      # specific node
```
`ncu` is expected to fail with `ERR_NVGPUCTRPERM`: GPU performance counters are
locked on this cluster (confirmed with admins). Use `nsys` timings instead.

## Single run by hand (compute node only)

```bash
gcc -O3 -march=native -fopenmp -std=c11 src/pf_track.c -lm -o build/pf_track
OMP_NUM_THREADS=8 ./build/pf_track --particles 100000 --steps 200
./build/pf_track --header      # CSV column names
```
Options: `--particles --steps --radars --ensemble --impact-every --seed --label`.
