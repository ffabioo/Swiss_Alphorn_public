"""
Data folders of the pipeline, relative to the repository root (all the scripts run from it). The subfolders of
databases/ have the layout of the published dataset (README, section Data): download or regenerate the data there.
"""
from pathlib import Path

DATABASES = Path("databases")
SNAPSHOTS = DATABASES / "snapshots"                     # step 1: solutions_<mu>.sqlite, one database per geometry
MERGED_SNAPSHOTS = DATABASES / "solutions.sqlite"       # step 2: all the snapshots in one database (input of step 3)
SOLVER_CACHE = DATABASES                                # FE solutions cached by core/parametric_solver.py
REDUCED_BASIS = DATABASES / "reduced_basis"             # step 3: ML/rank<r>/zone<k>[fmin_fmax].npz, rb_fast_data.npz
RESONANCE_DATASETS = DATABASES / "resonance_datasets"   # step 6: resonances_ML_rank50.tsv, resonances_grid26_rank50.tsv
NEURAL_NETWORKS = DATABASES / "neural_networks"         # steps 7-9: forward.pt, backward.pt, tune_backward/
