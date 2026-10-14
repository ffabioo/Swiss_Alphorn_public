#!/bin/bash
# Figures, tables and numbers of the chapter from the stored results (results/data, databases/, analysis/output):
# no FEM, no reduced-basis assembly, a few minutes on one core. Called by pipeline/12_assets.sh; can also be
# run directly from anywhere:   bash analysis/build_assets.sh
# The printed output of every script is kept in analysis/output/logs/<script>.txt (numbers quoted in the chapter).
# The scripts that need the published data (databases/reduced_basis/, databases/resonance_datasets/; see README) are
# skipped if it is missing.
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/pipeline/env.sh"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
LOGS=analysis/output/logs
mkdir -p "$LOGS"
py() {   # py <script> [args]: run and keep the printed output
    local name; name=$(basename "$1" .py)${2:+_$2}
    $PY "$@" 2>&1 | sed -e "/^$/d" -e "s#$REPO/##g" | tee "$LOGS/$name.txt"   # paths relative to the repository
}
have_dataset() { [ -f databases/resonance_datasets/resonances_ML_rank50.tsv ]; }
have_rb() { [ -f databases/reduced_basis/rb_fast_data.npz ]; }

echo "== figures (results/figures)"
py analysis/impedance_all_real_alphorns.py
py analysis/three_random_predictions_vs_impedance.py
py analysis/impedance_comparison_backward_forward_real.py

echo "== tables (results/tables)"
py analysis/make_tables.py

echo "== numbers (analysis/output)"
py analysis/geometry_comparison.py
py analysis/tmm1d.py
py analysis/tmm_shapes.py
py analysis/rb_validation.py summary
if have_rb; then
    py analysis/rb_vs_fem_resonances.py summary
else
    echo "databases/reduced_basis/ not found (published data, see README): reduced vs FEM resonances not recomputed"
fi
py ml/tune_backward.py summary
if have_dataset && have_rb; then
    py analysis/dataset_stats.py
    py analysis/average_geometry_demo.py
else
    echo "databases/resonance_datasets/ or databases/reduced_basis/ not found (published data, see README): dataset statistics skipped"
fi
