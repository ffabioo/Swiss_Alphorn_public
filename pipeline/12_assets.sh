#!/bin/bash
#SBATCH --job-name=alp_assets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 12. Figures, tables and numbers of the chapter from the results of the previous steps (plain Python, no FEM):
# offset statistics over the grid (a few GB of memory), then analysis/build_assets.sh
if [ -f databases/resonance_datasets/resonances_ML_rank50.tsv ] && [ -f databases/resonance_datasets/resonances_grid26_rank50.tsv ]; then
    $PY analysis/offset_analysis.py
else
    echo "databases/resonance_datasets/*.tsv not found (published data, see README): analysis/output/offset_analysis.json is not recomputed"
fi
bash analysis/build_assets.sh
