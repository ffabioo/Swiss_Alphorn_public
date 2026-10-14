#!/bin/bash
#SBATCH --job-name=alp_rbval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 4. Accuracy of the reduced model on the solution: 20 test geometries x 100 frequencies (one process per
# geometry) -> analysis/output/rb_validation/
need_container
export APPTAINERENV_OMP_NUM_THREADS=1
for i in $(seq 0 19); do
    run_fem "python3 analysis/rb_validation.py $i" &
    while [ "$(jobs -rp | wc -l)" -ge "$NCPU" ]; do wait -n; done
done
wait
$PY analysis/rb_validation.py summary
