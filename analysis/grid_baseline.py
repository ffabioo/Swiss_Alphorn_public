"""
Exhaustive reduced-model search over the grid (baseline for the forward-model search of ml/train_forward.py).

Reads the reduced-model resonances of all the grid configurations (databases/resonance_datasets/resonances_grid26_rank50.tsv, written by
`python ml/build_ml_dataset_modes.py grid`), ranks the configurations with 12 resonances in the band by the error
(RMSE of the notes after subtracting the offset of the model, ml_common.tuning_error), and adds the optimum ('rb_grid') to results/data/ml_best_mus.json and
results/data/forward_search_summary.json (rank of the forward-model geometry, best values, quantiles).

Run from the repo root (plain Python, a few GB of memory for the 4.4 M rows):  python analysis/grid_baseline.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ml_common import tuning, F_COLS, MU_COLS, GRID_DATASET

out = Path("results/data")
summary = json.loads((out / "forward_search_summary.json").read_text())
G = pd.read_csv(GRID_DATASET, sep="\t", usecols=MU_COLS + F_COLS + ["valid"])
G = G[G["valid"] == 1]
err_g, off_g, rmse_g = tuning(G[F_COLS].to_numpy())
g = int(np.argmin(err_g))
mu_g = G[MU_COLS].to_numpy()[g]
rank_fwd = int(np.sum(err_g < summary["forward"]["err_rb"] - 1e-9)) + 1
order = np.sort(err_g)
summary["rb_grid"] = {"mu": mu_g.tolist(), "err_rb": float(err_g[g]), "offset_rb": float(off_g[g]), "rmse_rb": float(rmse_g[g]), "n_valid": int(len(G)), "rank_of_forward": rank_fwd,
                      "err_rb_quantiles": {"best10": order[:10].tolist(), "p0.01": float(np.quantile(err_g, 1e-4)),
                                           "p1": float(np.quantile(err_g, 0.01)), "median": float(np.median(err_g))}}
print(f"Exhaustive reduced-model search ({len(G):,} valid geometries): error {err_g[g]:.2f} cents "
      f"(offset {off_g[g]:+.2f}, RMSE {rmse_g[g]:.2f}), mu = {np.round(mu_g, 4)}; "
      f"the forward-model geometry ({summary['forward']['err_rb']:.2f} cents) ranks {rank_fwd}")
print("10 best:", np.round(order[:10], 2), " 1% quantile", round(float(np.quantile(err_g, 0.01)), 2),
      " median", round(float(np.median(err_g)), 2))
(out / "forward_search_summary.json").write_text(json.dumps(summary, indent=2))
best = json.loads((out / "ml_best_mus.json").read_text())
best["rb_grid"] = mu_g.tolist()
(out / "ml_best_mus.json").write_text(json.dumps(best, indent=2))
