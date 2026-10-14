"""
Shared helpers for the forward and backward ML models (Section "Results of the optimization").

The original training notebooks are not part of this repository; the models below are
re-implemented from the description in the chapter (architecture and hyperparameters).

Dataset: databases/resonance_datasets/resonances_ML_rank50.tsv (ml/build_ml_dataset_modes.py). The resonances are numbered
from the lowest one; the notes of the alphorn scale are the resonances n = 2..10 and 12.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from utils.valais import L, L_MIN, R_MIN, R_MAX
from utils.frequencies import OFFICIAL_NOTES_FREQUENCIES
from ml.rb_fast import NOTE_MODES
from utils.paths import RESONANCE_DATASETS, NEURAL_NETWORKS

L_MIN_REL = L_MIN / L
DATASET = RESONANCE_DATASETS / "resonances_ML_rank50.tsv"
GRID_DATASET = RESONANCE_DATASETS / "resonances_grid26_rank50.tsv"
MODELS_DIR = NEURAL_NETWORKS

MU_COLS = ["l1", "l2", "l3", "R2", "R3", "R4"]
F_COLS = [f"f_res_{n}" for n in NOTE_MODES]

# Offset of the model (cents): mean deviation of the FEM resonances of the reference alphorn from the alphorn scale
# (results/data/white_alphorn_resonances.json, checked in analysis/make_tables.py). It is taken as the error of the model
# and subtracted from the deviations of all the geometries (tuning_error); the rest of the offset of a geometry is due to
# its shape and counts as an error.
COMMON_OFFSET = 86.37


def load_dataset(path=DATASET):
    """Geometries and the frequencies of the resonances 2..10 and 12 (valid rows only)."""
    df = pd.read_csv(path, sep="\t")
    df = df[df["valid"] == 1].dropna(subset=MU_COLS + F_COLS)
    return df[MU_COLS].to_numpy(dtype=np.float64), df[F_COLS].to_numpy(dtype=np.float64)


def split_indices(n, seed=0, frac_val=0.1, frac_test=0.1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test, n_val = int(frac_test * n), int(frac_val * n)
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def cents(f, f_ref):
    return 1200.0 * np.log2(np.asarray(f) / np.asarray(f_ref))


def tuning_error(f, f_ref=OFFICIAL_NOTES_FREQUENCIES, offset=COMMON_OFFSET):
    """Error E (cents): RMSE of the deviations after subtracting the offset of the model, the same for all the
    geometries. The criterion used to rank the geometries."""
    return np.sqrt(np.mean((cents(f, f_ref) - offset) ** 2, axis=-1))


def tuning(f, f_ref=OFFICIAL_NOTES_FREQUENCIES):
    """(error E, offset, RMSE) of the deviations in cents: E as in tuning_error, the offset of the geometry (mean
    deviation from the scale, reported) and the RMSE with nothing subtracted (reported)."""
    d = cents(f, f_ref)
    return tuning_error(f, f_ref), d.mean(axis=-1), np.sqrt(np.mean(d ** 2, axis=-1))


class Standardizer:
    def __init__(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0  # constant column: leave it centred only

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state):
        """Standardizer saved with state() (e.g. in a model checkpoint)."""
        s = cls.__new__(cls)
        s.mean, s.std = np.asarray(state["mean"], dtype=np.float64), np.asarray(state["std"], dtype=np.float64)
        return s
