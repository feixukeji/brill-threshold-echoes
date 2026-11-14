#!/bin/bash
#SBATCH --gres=gpu:1 -c 8 --mem=240G -t 2-00:00:00 -J ghbis
# Threshold bisection (src/vacuum/gh_bisect.py) on one B200 / H200.  A round
# from T = 0 keeps every member's checkpoints in CPU memory: 240 GB (64 GB is
# not enough).  Usage: sbatch -J <name> jobs/gh_bisect.sh <out-stem> <gh_bisect args...>
# writes <out-stem>_round<k>.json, <out-stem>_pair.npz and the log <out-stem>.txt.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"   # repository root
export PYTHONPATH=src NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
: "${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit with libNVVM, for numba.cuda}"
stem=$1; shift
${PYTHON:-python} src/vacuum/gh_bisect.py "$@" --out $stem > $stem.txt 2>&1
