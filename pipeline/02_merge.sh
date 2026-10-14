#!/bin/bash
#SBATCH --job-name=alp_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 2. Merge the snapshot databases databases/snapshots/solutions_*.sqlite -> databases/solutions.sqlite (83 GB,
# serial)
need_container
run_fem "python3 utils/merge_databases.py"
