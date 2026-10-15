"""
Discretization error of the FEM resonances (step 13): the reference alphorn computed with Lagrange elements of degree 1
(step 11, results/data/white_alphorn_resonances.json, the discretization of all the results of the chapter) and of
degree 2 on the same mesh (step 13, results/data/white_alphorn_Q2_resonances.json).

Writes analysis/output/p_refinement.json: for each note (resonances n = 2..10 and 12) the two frequencies and their
difference in Hz and in cents; the offset phi (mean deviation of the notes from the scale, as in analysis/make_tables.py)
and the tuning error E (RMSE of the deviations after subtracting phi) of the reference alphorn for both degrees; the
numbers of unknowns. Plain Python.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "results" / "data", ROOT / "analysis" / "output"
NOTES = np.array([92, 139, 185, 234, 278, 331, 371, 417, 468, 556.])
NOTE_MODES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])


def cents(f, f0):
    return 1200 * np.log2(np.asarray(f, float) / np.asarray(f0, float))


def mesh_nodes(path):
    """Number of nodes of a gmsh 4.1 mesh (= unknowns with degree-1 Lagrange elements)."""
    lines = path.read_text().splitlines()
    return int(lines[lines.index("$Nodes") + 1].split()[1])


q1 = json.loads((DATA / "white_alphorn_resonances.json").read_text())
q2 = json.loads((DATA / "white_alphorn_Q2_resonances.json").read_text())
assert q1["mu"] == q2["mu"] and q1["n_resonances"] == q2["n_resonances"] == 12
f1, f2 = (np.array(q["resonances"])[NOTE_MODES - 1] for q in (q1, q2))
out = {"notes_Hz": NOTES.tolist(), "f_degree1_Hz": f1.tolist(), "f_degree2_Hz": f2.tolist(),
       "difference_Hz": (f1 - f2).tolist(), "difference_cents": cents(f1, f2).tolist(),
       "n_dofs_degree1": mesh_nodes(ROOT / "mesh" / "reference_domain.msh"), "n_dofs_degree2": q2["n_dofs"]}
for d, f in (("degree1", f1), ("degree2", f2)):
    dev = cents(f, NOTES)
    out[f"phi_{d}_cents"] = float(dev.mean())
    out[f"E_{d}_cents"] = float(np.sqrt(np.mean((dev - dev.mean()) ** 2)))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "p_refinement.json").write_text(json.dumps(out, indent=2) + "\n")

print(f"unknowns: degree 1 {out['n_dofs_degree1']}, degree 2 {out['n_dofs_degree2']}")
print("note (Hz)   degree 1 (Hz)   degree 2 (Hz)   difference (Hz)   difference (cents)")
for n, a, b in zip(NOTES, f1, f2):
    print(f"{n:9.0f}   {a:13.3f}   {b:13.3f}   {a - b:+15.3f}   {cents(a, b):+18.2f}")
print(f"offset phi: {out['phi_degree1_cents']:.2f} -> {out['phi_degree2_cents']:.2f} cents; "
      f"tuning error E of the reference alphorn: {out['E_degree1_cents']:.2f} -> {out['E_degree2_cents']:.2f} cents")
