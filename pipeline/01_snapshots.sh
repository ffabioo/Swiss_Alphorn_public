#!/bin/bash
#SBATCH --job-name=alp_snapshots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 1. High-fidelity snapshots: 1225 geometries x 200 frequencies (MPI) -> databases/snapshots/solutions_*.sqlite (83 GB)
need_container
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
run_fem "mpirun -n $NCPU python3 utils/compute_hf_solutions.py"
