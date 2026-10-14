#!/bin/bash
#SBATCH --job-name=alp_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 0. Validation of the FE implementation on a benchmark problem with an analytical solution (appendix):
# errors and convergence rates for degrees 1-3 -> analysis/output/benchmark/{errors.json,error_H10.png,error_L2.png}
need_container
export APPTAINERENV_OMP_NUM_THREADS=1
run_fem "python3 analysis/benchmark_convergence.py"
