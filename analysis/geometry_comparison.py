"""
Comparison of the optimized geometries with the reference alphorn (paragraph on tab:geom_optimized): cone lengths and
junction radii in mm, and the difference of the bore radius along the conical part (piecewise-linear profile from
R1 = 6.5 mm to R5 = 51.5 mm through the junctions), measured from the beginning of the first cone.
Output: analysis/output/geometry_comparison.json

Run from the repo root (plain Python):  python analysis/geometry_comparison.py
"""
import json
from pathlib import Path

import numpy as np

from utils.valais import mu_glob_h4, L, R_MIN, R_MAX

OUT = Path("analysis/output/geometry_comparison.json")
best = json.loads(Path("results/data/ml_best_mus.json").read_text())
GEOMS = {"reference": mu_glob_h4, "forward": best["forward"], "lookup": best["lookup"], "backward": best["backward"]}


def profile(mu, x):
    mu = np.asarray(mu, float)
    joints = np.r_[0, np.cumsum(np.r_[mu[:3], 1 - mu[:3].sum()] * L)]
    radii = np.r_[R_MIN, mu[3:], R_MAX]
    return np.interp(x, joints, radii), joints, radii


x = np.linspace(0, L, 5001)
r_ref, _, _ = profile(GEOMS["reference"], x)
res = {}
for name, mu in GEOMS.items():
    r, joints, radii = profile(mu, x)
    diff = r - r_ref
    res[name] = {"L_mm": np.diff(joints).tolist(), "R_mm": radii[1:4].tolist(),
                 "radius_diff_min_mm": float(diff.min()), "at_mm": float(x[np.argmin(diff)]),
                 "radius_diff_max_mm": float(diff.max()), "at_max_mm": float(x[np.argmax(diff)])}
    if name != "reference":
        print(f"{name:9s}: L = {np.round(np.diff(joints), 1)} mm, R2..R4 = {np.round(radii[1:4], 2)} mm; radius - reference "
              f"from {diff.min():+.1f} mm (at {x[np.argmin(diff)] / 1000:.2f} m) to {diff.max():+.1f} mm (at {x[np.argmax(diff)] / 1000:.2f} m)")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))
