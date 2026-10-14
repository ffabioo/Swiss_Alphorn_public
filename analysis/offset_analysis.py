"""
Statistics of the offset (mean deviation of the notes from the alphorn scale, in cents) quoted in the chapter.
The geometries are ranked by the error E, the RMSE of the deviations after subtracting the offset of the model c, i.e.
the FEM offset of the reference alphorn (ml_common.COMMON_OFFSET, the same for all the geometries); the rest of the
offset of a geometry (offset - c) is due to its shape and counts in E.
  - range of the offset, and of its part due to the shape, over the sampled geometries (training dataset) and over
    the search grid;
  - offsets and errors of the 1000 grid configurations with the lowest error;
  - error and rank of the forward-model geometry on the grid;
  - sensitivity to c: optimum of the grid when another value of c is subtracted (c = 0, i.e. nothing subtracted,
    ... 140 cents), compared with the forward-model geometry.
All values from the reduced model (databases/resonance_datasets/*.tsv); the reference offset from results/data/white_alphorn_resonances.json.
Output: analysis/output/offset_analysis.json

Run from the repo root (plain Python, a few GB of memory for the grid):  python analysis/offset_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ml_common import cents, load_dataset, F_COLS, MU_COLS, GRID_DATASET, COMMON_OFFSET, OFFICIAL_NOTES_FREQUENCIES as FN
from ml.rb_fast import NOTE_MODES

OUT = Path("analysis/output/offset_analysis.json")
ref = np.array(json.loads(Path("results/data/white_alphorn_resonances.json").read_text())["resonances"])[NOTE_MODES - 1]
C_REF = float(cents(ref, FN).mean())
assert abs(C_REF - COMMON_OFFSET) < 0.01, (C_REF, COMMON_OFFSET)
mu_fwd = np.array(json.loads(Path("results/data/ml_best_mus.json").read_text())["forward"])


def error(d, c=COMMON_OFFSET):
    """RMSE of the deviations d (cents, one row per geometry) after subtracting c."""
    return np.sqrt(((d - c) ** 2).mean(1))


res = {"reference_offset_fem": C_REF, "common_offset": COMMON_OFFSET}

# training dataset (98 718 geometries with 12 resonances)
_, f = load_dataset()
off = cents(f, FN).mean(1)
res["dataset"] = {"n": int(len(f)), "offset_min": float(off.min()), "offset_max": float(off.max()),
                  "shape_offset_min": float(off.min() - COMMON_OFFSET), "shape_offset_max": float(off.max() - COMMON_OFFSET)}

# search grid
G = pd.read_csv(GRID_DATASET, sep="\t", usecols=MU_COLS + F_COLS + ["valid"])
G = G[G["valid"] == 1]
mu = G[MU_COLS].to_numpy()
d = cents(G[F_COLS].to_numpy(), FN)
del G
off, err = d.mean(1), error(d)
top = np.argsort(err)[:1000]
jf = int(np.argmin(np.abs(mu - mu_fwd).sum(1)))
res["grid"] = {
    "n": int(len(mu)), "offset_min": float(off.min()), "offset_max": float(off.max()),
    "shape_offset_min": float(off.min() - COMMON_OFFSET), "shape_offset_max": float(off.max() - COMMON_OFFSET),
    "offset_quantiles": {q: float(np.quantile(off, float(q))) for q in ("0.01", "0.5", "0.99")},
    "top1000_offset_min": float(off[top].min()), "top1000_offset_max": float(off[top].max()),
    "top1000_err_max": float(err[top].max()),
    "forward": {"err": float(err[jf]), "offset": float(off[jf]), "rank": int((err < err[jf] - 1e-9).sum()) + 1},
}
scan = {}
for c in (0.0, 40.0, 70.0, 80.0, C_REF - 5, C_REF, C_REF + 5, 90.0, 100.0, 120.0, 140.0):
    s = error(d, c)
    j = int(np.argmin(s))
    scan[f"{c:.2f}"] = {"best_mu": mu[j].tolist(), "best": float(s[j]), "best_offset": float(off[j]),
                        "best_err": float(err[j]), "forward": float(s[jf]), "forward_rank": int((s < s[jf] - 1e-9).sum()) + 1}
res["grid"]["other_c"] = scan

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))
g = res["grid"]
print(f"reference offset (FEM) {C_REF:+.2f} cents, c = {COMMON_OFFSET}")
print(f"offset range: dataset {res['dataset']['offset_min']:+.1f} .. {res['dataset']['offset_max']:+.1f}, "
      f"grid {g['offset_min']:+.1f} .. {g['offset_max']:+.1f}; due to the shape (offset - c): dataset "
      f"{res['dataset']['shape_offset_min']:+.1f} .. {res['dataset']['shape_offset_max']:+.1f}")
print(f"forward geometry: error {g['forward']['err']:.2f} (rank {g['forward']['rank']} of {g['n']}), offset {g['forward']['offset']:+.2f}")
print(f"1000 lowest-error grid configurations: offset {g['top1000_offset_min']:+.1f} .. {g['top1000_offset_max']:+.1f}, "
      f"error <= {g['top1000_err_max']:.2f}")
for c, v in scan.items():
    print(f"c = {c}: optimum {v['best']:.2f} (offset {v['best_offset']:+.1f}, error with c = {COMMON_OFFSET}: {v['best_err']:.2f}), "
          f"forward geometry {v['forward']:.2f}, rank {v['forward_rank']}")
