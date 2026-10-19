#!/bin/bash -l
# Submit one run as a batch job (on the SCC):
#   bash scc/submit.sh CONFIG.toml [SLOTS=16] [HOURS=12]
# Prints the plan and time estimate (run.py --dry-run) first. Log: scc/logs/<name>.o<job id>.
set -euo pipefail
cd "$HOME/audited-auctions-empirical"
CONFIG=${1:?usage: bash scc/submit.sh CONFIG.toml [SLOTS] [HOURS]}
SLOTS=${2:-16}
HOURS=${3:-12}
: "${SCC_PROJECT:?SCC_PROJECT not set; run scc/setup.sh PROJECT first}"
mkdir -p scc/logs
.venv/bin/python run.py "$CONFIG" --dry-run --workers "$SLOTS"
qsub -P "$SCC_PROJECT" -N "aa-$(basename "$CONFIG" .toml)" -pe omp "$SLOTS" \
     -l h_rt="$HOURS":00:00 scc/job.sh "$CONFIG"
