#!/bin/bash
#SBATCH --job-name=alp_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 6. Reduced-model resonances, numbered from the lowest one (plain Python, NCPU processes):
#   pipeline/06_ml_dataset.sh dataset   -> databases/resonance_datasets/resonances_ML_rank50.tsv     (1e5 sampled geometries, ~4 CPU h)
#   pipeline/06_ml_dataset.sh grid      -> databases/resonance_datasets/resonances_grid26_rank50.tsv (4.4 M grid configurations, ~140 CPU h)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SLURM_CPUS_PER_TASK=$NCPU
$PY ml/build_ml_dataset_modes.py "${1:-dataset}"
