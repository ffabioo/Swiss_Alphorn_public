"""
Resonance dataset labelled by resonance number (replaces the frequency-window labels of the earlier dataset builder,
legacy/ml/build_ml_dataset.py).

For each geometry, the resonances of the reduced model are found as the peaks of |Z_in| in [30, 615] Hz
(prominence of log|Z_in| >= 0.2), refined locally (ml/rb_fast.py), and numbered from the lowest one.
The notes of the alphorn scale correspond to the resonances n = 2..10 and 12. Geometries that do not have
exactly 12 resonances in the band are flagged (valid = 0).

Usage (from the repo root):
  python ml/build_ml_dataset_modes.py dataset   # 1e5 sampled geometries -> databases/resonance_datasets/resonances_ML_rank50.tsv
  python ml/build_ml_dataset_modes.py grid      # whole admissible search grid (num_l = num_R = 26, l4 >= l_min)
                                                #   -> databases/resonance_datasets/resonances_grid26_rank50.tsv
Plain Python (numpy/scipy/pandas); uses $SLURM_CPUS_PER_TASK processes (default: all cores).
"""
import sys
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ml.rb_fast import FastRB
from utils.mus_utils import mus_glob_grid
from utils.valais import L, L_MIN, R_MIN, R_MAX
from utils.paths import RESONANCE_DATASETS

N_RES = 12
_rb = None


def _init():
    global _rb
    _rb = FastRB()


def _resonances(mu):
    p = _rb.resonances(mu, fmin=30.0, fmax=615.0)
    out = np.full(N_RES, np.nan)
    out[:min(N_RES, len(p))] = p[:N_RES]
    return out, len(p)


def evaluate(mus, processes):
    t0 = time.time()
    with Pool(processes, initializer=_init) as pool:
        res = pool.map(_resonances, list(mus), chunksize=64)
    F = np.array([r[0] for r in res]); n = np.array([r[1] for r in res])
    print(f"{len(mus)} geometries in {time.time() - t0:.0f} s; valid (exactly {N_RES} resonances): {np.mean(n == N_RES):.4f}", flush=True)
    return F, n


def dataset_mus(num=25, seed=0, n_mu=100_000, n_ranks=72):
    """The 1e5 geometries of the training dataset: the admissible grid with 25 values per parameter, shuffled
    (seed 0), first 1e5. The rows are in the order of the original 72-process run (process r took the rows r::72),
    so that the training / validation / test split of ml/ml_common.split_indices is reproduced."""
    l_min = L_MIN / L
    mus = mus_glob_grid(l_mins=(l_min,) * 3, R_mins=(R_MIN,) * 3, R_max=R_MAX, num_l=num, num_R=num, l_min4=l_min)
    np.random.default_rng(seed).shuffle(mus)
    mus = mus[:n_mu]
    return np.vstack([mus[r::n_ranks] for r in range(n_ranks)])


def main(which):
    processes = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    out_dir = RESONANCE_DATASETS
    if which == "dataset":
        mus = dataset_mus()
    elif which == "grid":
        l_min = L_MIN / L
        mus = mus_glob_grid(l_mins=(l_min,) * 3, R_mins=(R_MIN,) * 3, R_max=R_MAX, num_l=26, num_R=26, l_min4=l_min)
    else:
        raise ValueError(which)
    F, n = evaluate(mus, processes)
    df = pd.DataFrame(mus, columns=["l1", "l2", "l3", "R2", "R3", "R4"])
    for k in range(N_RES):
        df[f"f_res_{k + 1}"] = F[:, k]
    df["n_res"] = n
    df["valid"] = (n == N_RES).astype(int)
    name = "resonances_ML_rank50.tsv" if which == "dataset" else "resonances_grid26_rank50.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, sep="\t", index=False)
    print("saved", out_dir / name)


if __name__ == "__main__":
    main(sys.argv[1])
