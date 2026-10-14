#!/bin/bash
#SBATCH --job-name=alp_rbfem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=170G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 5. Accuracy of the reduced model on the resonances: FEM resonances of the same 20 test geometries, then the
# comparison with the reduced model -> analysis/output/rb_vs_fem/, and the computing times -> analysis/output/timing.json
need_container
export APPTAINERENV_OMP_NUM_THREADS=1
while read -r name mu; do
    run_fem "python3 analysis/fem_resonances.py $name --mu $mu --out-dir analysis/output/rb_vs_fem" &
    while [ "$(jobs -rp | wc -l)" -ge "$NCPU" ]; do wait -n; done
done < <($PY analysis/rb_vs_fem_resonances.py mus)
wait
$PY analysis/rb_vs_fem_resonances.py summary
# computing times on one core (footnote of the chapter) -> analysis/output/timing.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY analysis/timing.py rb
APPTAINERENV_OMP_NUM_THREADS=1 run_fem "python3 analysis/timing.py fem"
