#!/bin/bash
#SBATCH --job-name=alp_tune_bw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline/logs/%x-%A_%a.log
#SBATCH --array=0-11
set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 8. Hyperparameter selection of the backward model: 12 configurations, validation set
#   -> databases/neural_networks/tune_backward/cfg_*.json (table: python ml/tune_backward.py summary, written by step 12)
# As a SLURM array job each task runs one configuration; with bash the 12 configurations run one after the other.
export OMP_NUM_THREADS=$NCPU
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    $PY ml/tune_backward.py "$SLURM_ARRAY_TASK_ID"
else
    for k in $(seq 0 11); do $PY ml/tune_backward.py "$k"; done
    $PY ml/tune_backward.py summary
fi
