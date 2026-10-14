# Sourced by every step of the pipeline. Each step runs from the repository root, either directly
#   bash pipeline/<step>.sh            (any Linux machine with Apptainer and the Python environment)
# or as a SLURM job (submit from the repository root; the #SBATCH lines are only read by sbatch)
#   sbatch pipeline/<step>.sh
# Settings (environment variables, all optional):
#   SIF   Apptainer image of dolfinx/lab, complex mode (default: <repo>/dolfinx_lab.sif), built with
#           apptainer build dolfinx_lab.sif docker://dolfinx/lab@sha256:e3235bb2355e7974b595cdff3b8f1c91491140b12e1118c88f30f9a245eac1e0
#   PY    Python with requirements-ml.txt (default: <repo>/.venv/bin/python if it exists, else python3)
#   NCPU  number of cores used by a step (default: $SLURM_CPUS_PER_TASK under SLURM, else all the cores)
#   BIND  extra folders visible in the container (comma-separated; set by step 3 for the database and its output)
export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SIF=${SIF:-$REPO/dolfinx_lab.sif}
if [ -z "${PY:-}" ]; then
    if [ -x "$REPO/.venv/bin/python" ]; then PY=$REPO/.venv/bin/python; else PY=python3; fi
fi
export PY
export NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
cd "$REPO"
export PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-0}   # fixed dates in the PDF figures: the same files at every run
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_HYDRA_LAUNCHER=fork

# run a command inside the dolfinx container (complex mode), from the repository root; the container sees the
# repository and the folders listed in BIND (comma-separated, e.g. the folder of the snapshot database)
run_fem() {
    apptainer exec --bind "$REPO${BIND:+,$BIND}" "$SIF" bash -c \
        "source /usr/local/bin/dolfinx-complex-mode; cd $REPO; export PYTHONPATH=$REPO:\$PYTHONPATH; $*"
}

# FEM steps need the container, the other steps only the Python environment
need_container() {
    [ -f "$SIF" ] || { echo "dolfinx container not found: SIF=$SIF (see pipeline/env.sh)" >&2; exit 1; }
}
