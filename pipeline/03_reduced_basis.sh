#!/bin/bash
#SBATCH --job-name=alp_rb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 3. Frequency-localized reduced bases (randomized SVD with a fixed seed, 11 zones, ranks 50-500) from the merged
# snapshot database -> databases/reduced_basis/ML/rank*/zone*.npz (the chapter uses rank 50), then the FE quantities used
# by the pure-numpy reduced model ml/rb_fast.py -> databases/reduced_basis/rb_fast_data.npz
#   pipeline/03_reduced_basis.sh [database] [output folder]   (defaults: databases/solutions.sqlite,
#                                                             databases/reduced_basis)
# The database is the merged one of step 2 (see README, section Data); ~80 GB are read; 53 GB of memory used (128 GB requested).
need_container
DB=${1:-databases/solutions.sqlite}
OUT=${2:-databases/reduced_basis}
[ -f "$DB" ] || { echo "snapshot database not found: $DB (steps 1-2)" >&2; exit 1; }
mkdir -p "$OUT"
DB=$(realpath "$DB"); OUT=$(realpath "$OUT")
export BIND=$(dirname "$DB"),$OUT${BIND:+,$BIND}   # the database and the output may be outside the repository
export APPTAINERENV_OMP_NUM_THREADS=$NCPU APPTAINERENV_OPENBLAS_NUM_THREADS=$NCPU
run_fem "python3 rbm_by_zone/db_to_rb_by_zone.py --db $DB --out $OUT/ML"
run_fem "python3 ml/export_rb_fast_data.py $OUT/rb_fast_data.npz"
