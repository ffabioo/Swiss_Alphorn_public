"""
Reduced model vs FEM on the resonance frequencies, for the 20 test geometries of analysis/rb_validation.py.

    python analysis/rb_vs_fem_resonances.py mus       # prints "name l1 l2 l3 R2 R3 R4", one line per test geometry
    # for each line, in the dolfinx container (see pipeline/05_rb_vs_fem.sh):
    #   python analysis/fem_resonances.py <name> --mu <mu> --out-dir analysis/output/rb_vs_fem
    python analysis/rb_vs_fem_resonances.py summary   # -> analysis/output/rb_vs_fem/summary.json

The summary compares the resonances of the notes (n = 2..10 and 12) computed with the reduced model (ml/rb_fast.py)
and with FEM (refined peaks of both). Plain Python for 'mus' and 'summary'.
"""
import json
import sys
from pathlib import Path

import numpy as np

from utils.mus_utils import mus_glob_grid
from utils.valais import L, L_MIN, R_MIN, R_MAX

OUT = Path("analysis/output/rb_vs_fem")


def test_mus(n=20, seed=0):
    """Same geometries as analysis/rb_validation.py:build_mu_test(n, np.random.default_rng(seed))."""
    lm = L_MIN / L
    grid = mus_glob_grid(l_mins=(lm,) * 3, R_mins=(R_MIN,) * 3, R_max=R_MAX, num_l=20, num_R=20, l_min4=lm)
    return grid[np.random.default_rng(seed).choice(grid.shape[0], size=n, replace=False)]


if sys.argv[1] == "mus":
    for i, mu in enumerate(test_mus()):
        print(f"rbtest_{i:02d} " + " ".join(repr(float(x)) for x in mu))
    sys.exit()

from ml.rb_fast import FastRB, NOTE_MODES  # noqa: E402

rb, E, bad = FastRB(), [], 0
for p in sorted(OUT.glob("rbtest_*_resonances.json")):
    d = json.loads(p.read_text())
    fem = np.array(d["resonances"])
    r = rb.resonances(np.array(d["mu"]), fmin=30.0, fmax=615.0)
    if len(fem) != 12 or len(r) != 12:
        bad += 1
        print("different number of resonances:", p.name, len(fem), len(r))
        continue
    E.append(1200 * np.log2(r / fem))
a = np.abs(np.array(E)[:, NOTE_MODES - 1])
s = {"n_geom": len(E), "n_mismatch": bad, "median": float(np.median(a)), "q90": float(np.quantile(a, 0.9)), "max": float(a.max())}
(OUT / "summary.json").write_text(json.dumps(s, indent=2))
print(f"{len(E)} geometries: |reduced - FEM| on the notes: median {s['median']:.2f}, 90% {s['q90']:.2f}, max {s['max']:.2f} cents")
