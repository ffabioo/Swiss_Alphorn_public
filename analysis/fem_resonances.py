"""
FEM resonance frequencies of one geometry, numbered from the lowest one and refined (no grid quantisation).

  - parametric geometry (reference mesh + affine map):  python analysis/fem_resonances.py NAME --mu l1 l2 l3 R2 R3 R4
  - own mesh (real alphorns):                           python analysis/fem_resonances.py NAME --mesh path/to/mesh.msh
  - option --degree D: Lagrange elements of degree D (default 1, used for all the results of the chapter), with the
    quadrature degree 4 D; used by pipeline/13_p_refinement.sh (D = 2) to check the discretization error.

|Z_in| is computed on a 1 Hz grid in [30, 615] Hz (saved to results/data/NAME.csv, used for the figures); every peak
of log|Z_in| with prominence >= 0.2 is then refined by a golden-section search between its grid neighbours
(tolerance 0.01 Hz). The resonances are written to results/data/NAME_resonances.json.
Run inside the dolfinx container (complex mode), from the repo root.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from dolfinx import fem, io
import dolfinx.fem.petsc
from mpi4py import MPI
import ufl

from utils.settings import ProblemSetting
from core.parametric_solver import compute_constants_matrices, setup_solver

DATA = Path(__file__).resolve().parents[1] / "results" / "data"
GOLDEN = (np.sqrt(5) - 1) / 2


class FemImpedance:
    def __init__(self, mu=None, mesh=None, degree=1):
        s = self.s = ProblemSetting(function_space=("Lagrange", degree), quadrature_degree=4 * degree)
        if mu is not None:   # parametric: constant matrices + cone matrices for this mu
            from core.all_cones_problem import AllConesProblem
            from utils.read_mesh import read_mesh
            M0, K0, Bm, Lv = compute_constants_matrices(s)
            cones = AllConesProblem(problem_setting=s)
            K, M = K0.copy(), M0.copy()
            for Kc in cones.stiffness_matrix(np.asarray(mu, dtype=float)):
                K.axpy(1.0, Kc)
            for Mc in cones.mass_matrix(np.asarray(mu, dtype=float)):
                M.axpy(1.0, Mc)
            domain, _, facet_tags = read_mesh(s.mesh_path)
        else:                # own mesh: assemble the same forms on it
            domain, _, facet_tags, _, _, _ = io.gmshio.read_from_msh(str(mesh), MPI.COMM_SELF, 0, gdim=3)
            V = fem.functionspace(domain, s.function_space)
            dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": s.quadrature_degree})
            dsm = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata={"quadrature_degree": s.quadrature_degree})
            p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
            opts = {"scalar_type": "complex"}
            def mat(a):
                A = fem.petsc.assemble_matrix(fem.form(a, form_compiler_options=opts)); A.assemble(); return A
            K, M = mat(ufl.inner(ufl.grad(p), ufl.grad(v)) * dx), mat(ufl.inner(p, v) * dx)
            Bm = mat(ufl.inner(p, v) * dsm(s.surface_outlet_id))
            Lv = fem.petsc.assemble_vector(fem.form(ufl.inner(s.u_n, v) * dsm(s.surface_inlet_id), form_compiler_options=opts))
        self.K, self.M, self.Bm, self.L = K, M, Bm, Lv
        V = fem.functionspace(domain, s.function_space)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata={"quadrature_degree": s.quadrature_degree})
        v = ufl.TestFunction(V)
        self.ell = fem.petsc.assemble_vector(fem.form(ufl.inner(1.0, v) * ds(s.surface_inlet_id),
                                                      form_compiler_options={"scalar_type": "complex"})).getArray().real.copy()
        self.area = self.ell.sum()
        self.A = K.duplicate(copy=False); self.x = K.createVecRight(); self.F = K.createVecRight()
        self.solver = setup_solver(K.getComm())

    def zabs(self, f):
        s = self.s; w = 2 * np.pi * f; k = w / s.c
        self.A.zeroEntries(); self.A.axpy(1.0, self.K); self.A.axpy(-(k**2), self.M)
        self.A.axpy((1.0 - 1j * k * s.R) / s.R, self.Bm); self.A.assemble()
        self.F.zeroEntries(); self.F.axpy(1j * s.rho0 * w, self.L); self.F.assemble()
        self.solver.setOperators(self.A); self.solver.setUp(); self.solver.solve(self.F, self.x)
        return abs(self.ell @ self.x.getArray()) / (s.u_n * self.area)


def golden_max(fun, a, b, tol=0.01):
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = fun(c), fun(d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc; c = b - GOLDEN * (b - a); fc = fun(c)
        else:
            a, c, fc = c, d, fd; d = a + GOLDEN * (b - a); fd = fun(d)
    return 0.5 * (a + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--mu", type=float, nargs=6); ap.add_argument("--mesh")
    ap.add_argument("--out-dir", default=str(DATA))
    ap.add_argument("--degree", type=int, default=1, help="polynomial degree of the Lagrange elements (default 1)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Zf = FemImpedance(mu=a.mu, mesh=a.mesh, degree=a.degree)
    n_dofs = Zf.K.getSize()[0]
    print(f"{a.name}: Lagrange elements of degree {a.degree}, {n_dofs} unknowns", flush=True)
    f = np.arange(30.0, 615.0 + 0.5, 1.0)
    z = np.array([Zf.zabs(x) for x in f])
    np.savetxt(out_dir / f"{a.name}.csv", np.column_stack([f, z]), delimiter=",", header="frequency,impedance", comments="")
    idx, _ = find_peaks(np.log(z), prominence=0.2)
    res = [golden_max(Zf.zabs, f[i - 1], f[i + 1]) for i in idx]
    out = {"name": a.name, "mu": a.mu, "mesh": a.mesh, "resonances": res, "n_resonances": len(res)}
    if a.degree != 1:   # (degree 1: the format of the results of the chapter, unchanged)
        out.update(degree=a.degree, n_dofs=n_dofs)
    (out_dir / f"{a.name}_resonances.json").write_text(json.dumps(out, indent=2))
    print(a.name, "resonances:", np.round(res, 2).tolist(), flush=True)


if __name__ == "__main__":
    main()
