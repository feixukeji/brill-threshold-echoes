#!/bin/bash
#SBATCH --gres=gpu:1 -c 8 --mem=64G -t 4:00:00 -J ghgpu
# A python script on one GPU (gh_linmap.py, gh_cuda.py checks, ...).
# Usage: sbatch jobs/gh_gpu.sh <log> <script.py> <args...>
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"   # repository root
nt=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=src NUMBA_NUM_THREADS=$nt OMP_NUM_THREADS=$nt OPENBLAS_NUM_THREADS=$nt MKL_NUM_THREADS=$nt
: "${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit with libNVVM, for numba.cuda}"
log=$1; shift
${PYTHON:-python} "$@" > $log 2>&1
