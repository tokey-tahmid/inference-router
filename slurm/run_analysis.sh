#!/bin/bash
# THE one documented command that regenerates every figure and table.
#
#   sbatch slurm/run_analysis.sh
#
# Reads results/raw/ only; writes results/figures/ and results/RESULTS.md.
# Runs on the gpu partition because the allocation it ran under had no CPU
# hours, not because it needs a GPU.
#
#SBATCH --job-name=ir-analysis
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=logs/ir_analysis_%j.out

set -euo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -eu
fi

find_repo() {
    local c
    for c in "${IR_REPO:-}" "${SLURM_SUBMIT_DIR:-}"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

echo "[analysis] host=$(hostname) job=${SLURM_JOB_ID} repo=${HERE}"
cd "$HERE"
"${IR_VENV}/bin/python" analysis/make_figures.py
echo "[analysis] ANALYSIS COMPLETE"
