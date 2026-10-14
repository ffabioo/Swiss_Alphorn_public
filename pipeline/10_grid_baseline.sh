#!/bin/bash
#SBATCH --job-name=alp_gridbl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 10. Exhaustive reduced-model search of the grid (baseline of the forward search): adds 'rb_grid' to
# results/data/ml_best_mus.json and forward_search_summary.json (needs databases/resonance_datasets/resonances_grid26_rank50.tsv and step 7)
$PY analysis/grid_baseline.py
