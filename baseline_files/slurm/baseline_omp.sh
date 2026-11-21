#!/bin/bash
# Midline baseline: serial + OpenMP thread sweep on ONE node, reduced problem.
#
# Submit from the REPO ROOT (results/ must already exist, or Slurm cannot
# write the log and the job dies in seconds, as happened with job 25101):
#     mkdir -p results && sbatch slurm/baseline_omp.sh
#
# Produces:
#     results/baseline_<jobid>.csv   one row per run (see pf_track --header)
#     results/baseline_<jobid>.out   environment + human-readable summaries
#
#SBATCH --job-name=pf-baseline
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=results/baseline_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ---- environment ---------------------------------------------------------
module purge
# EESSI toolchain (same GCC the MPI version will use later).
module load OpenMPI/4.1.5-GCC-12.3.0 || echo "WARNING: module load failed, using system gcc"

echo "== Reproducibility record =="
echo "date:        $(date -Is)"
echo "node:        $(hostname)"
echo "job:         $SLURM_JOB_ID  cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "git commit:  $(git rev-parse --short HEAD 2>/dev/null || echo n/a)  $(git status --porcelain | wc -l) uncommitted file(s)"
echo "compiler:    $(gcc --version | head -1)"
module list 2>&1 | sed 's/^/  /'
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|L3' | sed 's/^/  /'

# ---- build (on the compute node, so -march=native matches it) -------------
CFLAGS="-O3 -march=native -fopenmp -std=c11 -Wall"
echo "CFLAGS:      $CFLAGS"
mkdir -p build
gcc $CFLAGS src/pf_track.c -lm -o build/pf_track

# ---- experiment -----------------------------------------------------------
export OMP_PROC_BIND=close
export OMP_PLACES=cores
CSV=results/baseline_${SLURM_JOB_ID}.csv
./build/pf_track --header > "$CSV"

REPEATS=3
THREADS="1 2 4 8 16 32 64"
STEPS=200          # 20 s of tracking at 10 Hz (reduced from a full flight)
ENSEMBLE=2000      # impact Monte Carlo samples per prediction
SEED=12345

run() {  # run <label> <threads> <extra args...>
    local label=$1 t=$2; shift 2
    OMP_NUM_THREADS=$t ./build/pf_track --label "$label" --steps $STEPS \
        --ensemble $ENSEMBLE --seed $SEED "$@" >> "$CSV"
}

# 1) Strong scaling, small problem (main baseline): 1e5 particles
for r in $(seq $REPEATS); do
  for t in $THREADS; do run strong_1e5 $t --particles 100000; done
done

# 2) Strong scaling, 10x problem (sensitivity: does scaling improve with size?)
for r in $(seq $REPEATS); do
  for t in $THREADS; do run strong_1e6 $t --particles 1000000; done
done

# 3) Sensitivity to radar count (weight-stage cost) at fixed 16 threads
for r in $(seq $REPEATS); do
  for R in 3 6 12; do run radars_$R 16 --particles 100000 --radars $R; done
done

echo "== done: $CSV ($(($(wc -l < "$CSV") - 1)) runs) =="
