#!/bin/bash
#SBATCH --job-name=alp_backward
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
# Step 9. Backward (tandem) model with the selected hyperparameters: training, test error, query with the scale
# transposed by 0, 5, ..., 150 cents -> databases/neural_networks/backward.pt, results/data/{backward_summary,ml_best_mus}.json
#   pipeline/09_backward.sh                 training + query (needs databases/neural_networks/forward.pt)
#   pipeline/09_backward.sh --select-only   query with the saved databases/neural_networks/backward.pt (exact results of the chapter)
# (small torch models run faster with few threads)
export OMP_NUM_THREADS=$NCPU
$PY ml/train_backward.py "$@"
