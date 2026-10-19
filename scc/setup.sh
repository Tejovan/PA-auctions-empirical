#!/bin/bash -l
# One-time setup on the BU SCC (run on a login node):
#   bash setup.sh PROJECT
# Clones the code into $HOME, installs uv and the locked Python environment, links runs/ to
# project storage (/projectnb/PROJECT/$USER/runs; home has a 10 GB quota), runs the tests.
# Safe to rerun: it updates the clone and the environment.
set -euo pipefail

PROJECT=${1:?usage: bash setup.sh PROJECT}
REPO=https://github.com/Tejovan/audited-auctions-empirical.git
CODE=$HOME/audited-auctions-empirical
RUNS=/projectnb/$PROJECT/$USER/runs

if [ -d "$CODE/.git" ]; then
    git -C "$CODE" pull --ff-only
else
    git clone --branch web27 "$REPO" "$CODE"
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH

# uv's caches and Python downloads go to project storage, not the home quota.
mkdir -p "/projectnb/$PROJECT/$USER/uv-cache" "$RUNS"
grep -q 'audited-auctions SCC' "$HOME/.bashrc" 2>/dev/null || cat >> "$HOME/.bashrc" <<EOF

# audited-auctions SCC
export PATH=\$HOME/.local/bin:\$PATH
export SCC_PROJECT=$PROJECT
export UV_CACHE_DIR=/projectnb/$PROJECT/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/projectnb/$PROJECT/$USER/uv-python
EOF
export UV_CACHE_DIR=/projectnb/$PROJECT/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/projectnb/$PROJECT/$USER/uv-python

cd "$CODE"
uv sync --frozen --python 3.13
[ -L runs ] || { [ -d runs ] && mv runs "$RUNS/from-home-$(date +%s)"; ln -s "$RUNS" runs; }

.venv/bin/python -m pytest -q tests
echo "Setup done: code $CODE, runs $RUNS, project $PROJECT."
