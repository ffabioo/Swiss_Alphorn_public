"""
Validation of the FE implementation on a benchmark problem with an analytical solution (appendix "Validation on a
benchmark problem", fig:error_convergence): Helmholtz equation on the square [0, pi]^2,

    Delta p + k^2 p = 0                in the square,   k^2 = 2,
    dp/dn = g_N                        on x = 0,        g_N(y) = (1 + i) sqrt(3)/2 cos(y),
    dp/dn = 0                          on y = 0, pi,
    dp/dn + eta p = 0                  on x = pi,       eta = sqrt(3),

with exact solution p_ex = (1 + i) cos(x + pi/3) cos(y). Lagrange elements of degree r = 1, 2, 3 on uniform
triangular meshes with n x n squares (each split in two triangles), n = 8, ..., 256. The figures use h = 1/n, as the
figures of the chapter (the largest cell diameter, pi sqrt(2)/n, is also stored; the rates are the same).
Errors in the L2 norm and in the H1 seminorm; expected rates h^(r+1) and h^r.
Output: analysis/output/benchmark/errors.json, error_H10.png, error_L2.png (same layout as the figures of the chapter,
which were made with an earlier version of the code).

Run from the repo root in the dolfinx container (complex mode):  python3 analysis/benchmark_convergence.py
"""
import json
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

assert np.issubdtype(PETSc.ScalarType, np.complexfloating), "run in complex mode (source dolfinx-complex-mode)"
OUT = Path("analysis/output/benchmark")
K2, ETA = 2.0, np.sqrt(3.0)
NS, DEGREES = [8, 16, 32, 64, 128, 256], [1, 2, 3]


def solve(n, r):
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([np.pi, np.pi])], [n, n],
                                mesh.CellType.triangle)
    tdim, fdim = msh.topology.dim, msh.topology.dim - 1
    left = mesh.locate_entities_boundary(msh, fdim, lambda X: np.isclose(X[0], 0.0))
    right = mesh.locate_entities_boundary(msh, fdim, lambda X: np.isclose(X[0], np.pi))
    facets = np.hstack([left, right])
    order = np.argsort(facets)
    tags = mesh.meshtags(msh, fdim, facets[order], np.hstack([np.full(len(left), 1), np.full(len(right), 2)])[order].astype(np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    q = 2 * r + 4   # quadrature degree of the error integrals (the exact solution is not polynomial)
    dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": q})

    V = fem.functionspace(msh, ("Lagrange", r))
    x = ufl.SpatialCoordinate(msh)
    p_ex = (1 + 1j) * ufl.cos(x[0] + np.pi / 3) * ufl.cos(x[1])
    g_n = (1 + 1j) * np.sqrt(3) / 2 * ufl.cos(x[1])
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx - K2 * ufl.inner(p, v) * ufl.dx + ETA * ufl.inner(p, v) * ds(2)
    L = ufl.inner(g_n, v) * ds(1)
    problem = LinearProblem(a, L, petsc_options_prefix=f"bench_{n}_{r}_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    ph = problem.solve()

    e = ph - p_ex
    err = lambda form: np.sqrt(abs(msh.comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)))
    cells = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    h = msh.comm.allreduce(float(msh.h(tdim, cells).max()), op=MPI.MAX)
    return h, err(ufl.inner(e, e) * dx), err(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors, ref_colors = ["blue", "red", "green"], ["orange", "purple", "magenta", "yellow"]
    for key, label, shift in [("H1", "Error H10", 0), ("L2", "Error L2", 1)]:
        fig, ax = plt.subplots()
        for r, c in zip(DEGREES, colors):
            h = np.array(res[str(r)]["h"])
            ax.loglog(h, res[str(r)][key], "o-", color=c, label=f"r={r}")
        h = np.array(res["1"]["h"])
        for k, c in zip(range(1, len(DEGREES) + 1 + shift), ref_colors):
            ax.loglog(h, h ** k, "--", color=c, label="h" if k == 1 else f"h^{k}")
        ax.set_xlabel("h")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()
        fig.savefig(OUT / f"error_{'H10' if key == 'H1' else 'L2'}.png", dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    res = {}
    for r in DEGREES:
        rows = [solve(n, r) for n in NS]
        h, l2, h1 = (np.array(c) for c in zip(*rows))
        res[str(r)] = {"n": NS, "h": (1.0 / np.array(NS)).tolist(), "h_max_diameter": h.tolist(), "L2": l2.tolist(), "H1": h1.tolist(),
                       "rate_L2": (np.diff(np.log(l2)) / np.diff(np.log(h))).tolist(),
                       "rate_H1": (np.diff(np.log(h1)) / np.diff(np.log(h))).tolist()}
        if MPI.COMM_WORLD.rank == 0:
            print(f"r = {r}: L2 errors {np.array2string(l2, precision=2)}, rates {np.round(res[str(r)]['rate_L2'], 2)}")
            print(f"       H1 errors {np.array2string(h1, precision=2)}, rates {np.round(res[str(r)]['rate_H1'], 2)}")
    if MPI.COMM_WORLD.rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "errors.json").write_text(json.dumps(res, indent=2))
        plot(res)
        print("written:", OUT / "errors.json", OUT / "error_H10.png", OUT / "error_L2.png")
