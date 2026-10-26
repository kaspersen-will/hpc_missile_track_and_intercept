#!/bin/bash
#SBATCH --job-name=ncu-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=ncu_test_%j.out

module purge
module load nvhpc/25.7

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "== Compile =="
# Compiled on the GPU node, so 'native' targets whichever GPU we landed on.
nvcc -O2 -arch=native saxpy.cu -o saxpy \
  || nvc++ -cuda -gpu=ccnative -O2 saxpy.cu -o saxpy   # fallback if nvcc isn't on PATH

echo "== Plain run =="
./saxpy

echo "== nsys (timeline: transfers vs kernel) =="
nsys profile --stats=true --force-overwrite=true -o nsys_test ./saxpy

echo "== ncu (hardware metrics: the real permission test) =="
ncu --section SpeedOfLight ./saxpy
