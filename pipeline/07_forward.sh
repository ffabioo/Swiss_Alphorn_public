#!/bin/bash
#SBATCH --job-name=alp_forward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 7. Forward model: training, test error, search over the grid (1000 candidates re-ranked with the reduced model),
# best geometry of the training data, exhaustive grid search if the grid dataset exists
#   -> databases/neural_networks/forward.pt, results/data/{ml_best_mus,forward_search_summary,forward_random_predictions}.json
#   pipeline/07_forward.sh                 training + search
#   pipeline/07_forward.sh --search-only   search with the saved databases/neural_networks/forward.pt (exact results of the chapter)
export OMP_NUM_THREADS=$NCPU
$PY ml/train_forward.py "$@"
