#!/bin/bash
#SBATCH --job-name=alp_fem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=pipeline/logs/%x-%j.log

set -euo pipefail
# repository root: the directory of this script, or (under sbatch, which runs a copy of it) the submission directory
ROOT=$(cd "$(dirname "$0")/.." && pwd); [ -f "$ROOT/pipeline/env.sh" ] || ROOT=${SLURM_SUBMIT_DIR:-$PWD}
source "$ROOT/pipeline/env.sh"
# Step 11. FEM impedance (1 Hz grid) and refined resonances -> results/data/<name>.csv and <name>_resonances.json:
# the real alphorns on their own meshes, the reference alphorn on the parametric domain, the optimized geometries of
# results/data/ml_best_mus.json and the random geometries of the forward-model figure (one process per geometry)
need_container
export APPTAINERENV_OMP_NUM_THREADS=1
mu_of() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); v=d[sys.argv[2]]; v=v[int(sys.argv[3])] if len(sys.argv)>3 else v; print(' '.join(repr(float(x)) for x in v))" "$@"; }
runs=()
for k in 1 3 4; do
    runs+=("python3 analysis/fem_resonances.py impedance_Alphorn_$k --mesh mesh/real_alphorns/mesh_mesh_data20252026.$k.msh")
done
runs+=("python3 analysis/fem_resonances.py white_alphorn --mu $($PY -c 'from utils.valais import mu_glob_h4; print(" ".join(repr(float(x)) for x in mu_glob_h4))')")
for name in forward backward lookup rb_grid; do
    runs+=("python3 analysis/fem_resonances.py $name --mu $(mu_of results/data/ml_best_mus.json $name)")
done
for i in 0 1 2; do
    runs+=("python3 analysis/fem_resonances.py valais_random_$i --mu $(mu_of results/data/forward_random_predictions.json X_RANDOM $i)")
done
for r in "${runs[@]}"; do
    run_fem "$r" &
    while [ "$(jobs -rp | wc -l)" -ge "$NCPU" ]; do wait -n; done
done
wait
