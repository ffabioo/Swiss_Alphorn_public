"""
Statistics of the training dataset (databases/resonance_datasets/resonances_ML_rank50.tsv) quoted in the chapter:
  - number of resonances found between 30 and 615 Hz: geometries with 12 (kept) and with fewer (discarded);
  - shape of the discarded geometries: fraction with at least two thirds of the radius increase (R1 -> R5) in one cone;
  - training / validation / test split, best geometry of the training set (baseline 'lookup') and of the whole dataset.
Output: analysis/output/dataset_stats.json

Run from the repo root (plain Python):  python analysis/dataset_stats.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ml_common import DATASET, MU_COLS, F_COLS, R_MIN, R_MAX, load_dataset, split_indices, tuning

OUT = Path("analysis/output/dataset_stats.json")
df = pd.read_csv(DATASET, sep="\t")
valid = df["valid"] == 1
gaps = np.column_stack([df.R2 - R_MIN, df.R3 - df.R2, df.R4 - df.R3, R_MAX - df.R4])
one_cone = gaps.max(1) >= (2 / 3) * (R_MAX - R_MIN)

mu, f = load_dataset()
tr, va, te = split_indices(len(mu))
err, off, _ = tuning(f)
j_tr = int(tr[np.argmin(err[tr])])
j_all = int(np.argmin(err))
res = {
    "n_sampled": int(len(df)), "n_kept": int(valid.sum()), "n_discarded": int((~valid).sum()),
    "n_resonances": {int(k): int(v) for k, v in df["n_res"].value_counts().sort_index().items()},
    "one_cone_fraction_discarded": float(one_cone[~valid].mean()), "one_cone_fraction_kept": float(one_cone[valid].mean()),
    "first_resonance_range_kept_hz": [float(df.loc[valid, "f_res_1"].min()), float(df.loc[valid, "f_res_1"].max())],
    "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
    "best_train": {"mu": mu[j_tr].tolist(), "err": float(err[j_tr]), "offset": float(off[j_tr])},
    "best_all": {"mu": mu[j_all].tolist(), "err": float(err[j_all]), "offset": float(off[j_all]),
                 "in_train": bool(j_all in set(tr.tolist()))},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))
print(f"{res['n_sampled']} sampled, {res['n_kept']} kept ({100 * res['n_kept'] / res['n_sampled']:.1f}%), "
      f"{res['n_discarded']} discarded; resonances found: {res['n_resonances']}")
print(f"at least 2/3 of the radius increase in one cone: {100 * res['one_cone_fraction_discarded']:.0f}% of the discarded, "
      f"{100 * res['one_cone_fraction_kept']:.0f}% of the kept geometries")
print(f"split {res['split']}; best training geometry {np.round(mu[j_tr], 4)} (error {err[j_tr]:.2f} cents); "
      f"best of the whole dataset {err[j_all]:.2f} cents (in the training set: {res['best_all']['in_train']})")
