#!/bin/bash
# Optimisation experiment: serial vs parallel (prefix-sum) resampling.
# Same binary, same node, same seeds; only --resample changes.
# Modes are interleaved inside every repeat so node drift hits both equally.
#
# Submit from the REPO ROOT:
#     mkdir -p results && sbatch slurm/opt_resample.sh
#
# Produces:
#     results/opt_resample_<jobid>.csv   one row per run (pf_track --header)
#     results/opt_resample_<jobid>.out   environment + correctness test + summaries
#
#SBATCH --job-name=pf-opt-resample
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=results/opt_resample_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load OpenMPI/4.1.5-GCC-12.3.0 || echo "WARNING: module load failed, using system gcc"

echo "== Reproducibility record =="
echo "date:        $(date -Is)"
echo "node:        $(hostname)"
echo "job:         $SLURM_JOB_ID  cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "git commit:  $(git rev-parse --short HEAD 2>/dev/null || echo n/a)  $(git status --porcelain --untracked-files=no | wc -l) uncommitted file(s)"
echo "compiler:    $(gcc --version | head -1)"
module list 2>&1 | sed 's/^/  /'
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA node\(s\)|L3' | sed 's/^/  /'

CFLAGS="-O3 -march=native -fopenmp -std=c11 -Wall"
echo "CFLAGS:      $CFLAGS"
mkdir -p build
gcc $CFLAGS src/pf_track.c -lm -o build/pf_track
gcc $CFLAGS src/test/test_resample.c -lm -o build/test_resample

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ---- correctness first: parallel kernels must match the serial ones ------
echo "== test_resample (32 threads) =="
OMP_NUM_THREADS=32 ./build/test_resample

# ---- experiment -----------------------------------------------------------
CSV=results/opt_resample_${SLURM_JOB_ID}.csv
./build/pf_track --header > "$CSV"

REPEATS=3
THREADS=""; for t in 1 2 4 8 16 32; do [ $t -le $SLURM_CPUS_PER_TASK ] && THREADS="$THREADS $t"; done
STEPS=200; ENSEMBLE=2000; SEED=12345

run() {  # run <label> <threads> <mode> <extra args...>
    local label=$1 t=$2 mode=$3; shift 3
    OMP_NUM_THREADS=$t ./build/pf_track --label "$label" --steps $STEPS \
        --ensemble $ENSEMBLE --seed $SEED --resample "$mode" "$@" >> "$CSV"
}

for r in $(seq $REPEATS); do
  # 1) 1e6 strong scaling, before vs after (main result)
  for t in $THREADS; do for m in serial parallel; do run ${m}_1e6 $t $m --particles 1000000; done; done
  # 2) 1e5 strong scaling (small problem: is the gain still visible?)
  for t in $THREADS; do for m in serial parallel; do run ${m}_1e5 $t $m --particles 100000; done; done
  # 3) 1e7 at 16/32 threads only (deadline stress test; serial 1-thread run would take ~10 min)
  for t in 16 32; do for m in serial parallel; do run ${m}_1e7 $t $m --particles 10000000; done; done
done

echo "== done: $CSV ($(($(wc -l < "$CSV") - 1)) runs) =="
