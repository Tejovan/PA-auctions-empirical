#!/bin/bash -l
# Batch job of one run; submitted by scc/submit.sh (qsub sets project, slots, run time).
#$ -j y
#$ -o $HOME/audited-auctions-empirical/scc/logs/
set -euo pipefail
cd "$HOME/audited-auctions-empirical"
echo "job $JOB_ID on $(hostname), $NSLOTS slots, config $1, commit $(git rev-parse --short HEAD)"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python run.py "$1" --yes --workers "$NSLOTS"
