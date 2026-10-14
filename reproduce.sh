#!/bin/bash
# Reproduce the results of the chapter (see README.md, section "Reproducing the results").
#
#   ./reproduce.sh assets      figures, tables and numbers from the stored results               (minutes, Python only)
#   ./reproduce.sh saved       searches with the saved networks, FEM verification, then assets  (~1 h, + container)
#   ./reproduce.sh retrain     networks retrained (forward, hyperparameter study, backward), then 10-12  (~20 CPU h)
#   ./reproduce.sh all         every step from scratch: benchmark, snapshots, reduced bases, datasets, then retrain
#                                                                                  (~220 CPU h, ~160 GB of disk)
#   ./reproduce.sh benchmark   FE convergence test of the appendix (container, minutes)
#   ./reproduce.sh step NN [args]   one step of pipeline/ (e.g. ./reproduce.sh step 06 grid)
#
# Add --slurm after the target to submit the steps as SLURM jobs, each one starting when the previous one has
# finished (sbatch --dependency=afterok), instead of running them one after the other on this machine.
# Settings (container, Python, number of cores): see pipeline/env.sh.
set -euo pipefail
cd "$(dirname "$0")"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
target=${1:-}
[ -n "$target" ] || usage
shift
args=()
SLURM=0
for a in "$@"; do
    if [ "$a" = "--slurm" ]; then SLURM=1; else args+=("$a"); fi
done

dep=""
run() {   # run <step script> [args...]
    local s=pipeline/$1
    shift
    [ -f "$s" ] || { echo "no such step: $s" >&2; exit 1; }
    if [ $SLURM = 1 ]; then
        local jid
        jid=$(sbatch --parsable ${dep:+--dependency=afterok:$dep} "$s" "$@")
        echo "submitted $s $* -> job $jid (log in pipeline/logs/)"
        dep=${jid%%;*}
    else
        echo "=== $s $* ($(date '+%F %T'))"
        bash "$s" "$@"
    fi
}

saved()   { run 07_forward.sh --search-only; run 09_backward.sh --select-only; run 10_grid_baseline.sh
            run 11_fem_resonances.sh; run 12_assets.sh; }
retrain() { run 07_forward.sh; run 08_tune_backward.sh; run 09_backward.sh; run 10_grid_baseline.sh
            run 11_fem_resonances.sh; run 12_assets.sh; }

case $target in
    assets)    run 12_assets.sh ;;
    saved)     saved ;;
    retrain)   retrain ;;
    all)       run 00_benchmark.sh; run 01_snapshots.sh; run 02_merge.sh; run 03_reduced_basis.sh
               run 04_rb_validation.sh; run 05_rb_vs_fem.sh; run 06_ml_dataset.sh dataset; run 06_ml_dataset.sh grid
               retrain ;;
    benchmark) run 00_benchmark.sh ;;
    step)      [ ${#args[@]} -ge 1 ] || usage
               s=$(cd pipeline && ls "${args[0]}"_*.sh 2>/dev/null | head -1)
               [ -n "$s" ] || { echo "no step ${args[0]} in pipeline/" >&2; exit 1; }
               run "$s" "${args[@]:1}" ;;
    *)         usage ;;
esac
