#!/bin/bash
#SBATCH -c 8 --mem=64G -t 4:00:00 -J ghpy
# A python script on CPUs (gh_analysis.py curvature / horizons on pair checkpoints, ...).
# Usage: sbatch jobs/gh_py.sh <log> <script.py> <args...>
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"   # repository root
nt=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=src NUMBA_NUM_THREADS=$nt OMP_NUM_THREADS=$nt OPENBLAS_NUM_THREADS=$nt MKL_NUM_THREADS=$nt
log=$1; shift
${PYTHON:-python} "$@" > $log 2>&1
