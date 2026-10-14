"""
Why the backward model is not trained with a loss on the geometry (Section on the backward model): pairs of geometries
of the training dataset whose resonance frequencies differ by less than 3 cents (RMS over the notes) but whose geometries
are very different; a loss on mu would drive the prediction towards their average, and the average geometry is evaluated
with the reduced model against the frequencies of the pair.
Output: analysis/output/average_geometry_demo.json

Run from the repo root (plain Python):  python analysis/average_geometry_demo.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ml.ml_common import load_dataset, cents
from ml.rb_fast import FastRB, NOTE_MODES

N_PAIRS, MAX_CENTS, MIN_DMU = 3, 3.0, 0.15
OUT = Path("analysis/output/average_geometry_demo.json")

mu, f = load_dataset()
lc = 1200 * np.log2(f)                                       # frequencies in cents
dist, nn = cKDTree(lc).query(lc, k=2)
dist, nn = dist[:, 1] / np.sqrt(f.shape[1]), nn[:, 1]       # RMS distance (cents) to the nearest neighbour
scale = np.array([1, 1, 1, 1 / 45, 1 / 45, 1 / 45])          # lengths (fraction of L) and radii (fraction of R5 - R1)
dmu = np.abs((mu - mu[nn]) * scale).max(1)
cand = np.where((dist < MAX_CENTS) & (dmu > MIN_DMU))[0]
rb, rows = FastRB(), []
for i in cand[np.argsort(-dmu[cand])][:N_PAIRS]:
    a, b = mu[i], mu[nn[i]]
    r = rb.resonances((a + b) / 2, fmin=30.0, fmax=615.0)
    if len(r) != 12:
        continue
    e = cents(r[NOTE_MODES - 1], (f[i] + f[nn[i]]) / 2)
    rows.append({"mu_a": a.tolist(), "mu_b": b.tolist(), "pair_rms_cents": float(dist[i]),
                 "average_rms_cents": float(np.sqrt(np.mean(e ** 2))), "average_max_cents": float(np.abs(e).max())})
    print(f"A {np.round(a, 3)}  B {np.round(b, 3)}: A vs B {dist[i]:.1f} cents; "
          f"average geometry vs their frequencies {rows[-1]['average_rms_cents']:.1f} cents RMS (max {rows[-1]['average_max_cents']:.0f})")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"criteria": {"max_cents": MAX_CENTS, "min_dmu": MIN_DMU}, "pairs": rows}, indent=2))
