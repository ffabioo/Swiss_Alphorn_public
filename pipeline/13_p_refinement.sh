#!/bin/bash
#SBATCH --job-name=alp_p_refinement
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 13. Discretization error of the FEM resonances (p-refinement): the reference alphorn recomputed on the same mesh
# with Lagrange elements of degree 2 instead of 1 (159 401 instead of 21 093 unknowns; about 1.5 h, 5 GB of memory)
# -> results/data/white_alphorn_Q2.csv, white_alphorn_Q2_resonances.json; then the comparison with the degree-1
# results of step 11 (results/data/white_alphorn_resonances.json) -> analysis/output/p_refinement.json
need_container
export APPTAINERENV_OMP_NUM_THREADS=1
run_fem "python3 analysis/fem_resonances.py white_alphorn_Q2 --degree 2 --mu $($PY -c 'from utils.valais import mu_glob_h4; print(" ".join(repr(float(x)) for x in mu_glob_h4))')"
$PY analysis/p_refinement.py
