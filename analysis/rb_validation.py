"""
Accuracy of the frequency-localized reduced bases (rank 50) against the parametric FEM solver: relative error of the
solution (Euclidean norm of the nodal values) for 20 geometries not in the snapshot set (build_mu_test, seed 0) and
100 frequencies between SIMULATION_FMIN and SIMULATION_FMAX (69-579 Hz); the zone of each
frequency is the one whose centre is closest.

Run in the dolfinx container, from the repo root, one geometry per process, then the summary:
    python analysis/rb_validation.py <i_mu>      # i_mu = 0..19 -> analysis/output/rb_validation/mu<i_mu>.npz
    python analysis/rb_validation.py summary     # -> analysis/output/rb_validation/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("analysis/output/rb_validation")
OUT.mkdir(parents=True, exist_ok=True)
N_MU, N_F, RANK = 20, 100, 50

if sys.argv[1] == "summary":
    R = np.vstack([np.load(p)["res"] for p in sorted(OUT.glob("mu*.npz"))])   # columns: i_mu, f, zone, err
    e, z = R[:, 3], R[:, 2]
    s = {"samples": int(len(e)), "median": float(np.median(e)), "q90": float(np.quantile(e, 0.9)), "mean": float(e.mean()),
         "max": float(e.max()), "n_above_1e-1": int(np.sum(e > 0.1)), "fraction_above_1e-1": float(np.mean(e > 0.1)),
         "above_1e-1_per_zone": {int(k): int(np.sum((e > 0.1) & (z == k))) for k in np.unique(z)},
         "per_zone": {int(k): {"median": float(np.median(e[z == k])), "q90": float(np.quantile(e[z == k], 0.9)),
                               "max": float(e[z == k].max())} for k in np.unique(z)}}
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"samples {s['samples']} | median {s['median']:.2e} | 90% quantile {s['q90']:.2e} | max {s['max']:.2e} | "
          f"> 1e-1: {s['n_above_1e-1']} ({100 * s['fraction_above_1e-1']:.2f}%), per zone {s['above_1e-1_per_zone']}")
    sys.exit()

from rbm_by_zone.rbm_offline_by_zone import RBM_Offline  # noqa: E402
from core.parametric_solver import solve_system_parametric  # noqa: E402
from utils.settings import ProblemSetting  # noqa: E402
from utils.frequencies import SIMULATION_FMIN, SIMULATION_FMAX, FREQUENCY_ZONES_CENTERS  # noqa: E402
from utils.valais import L, L_MIN, R_MIN, R_MAX  # noqa: E402
from utils.mus_utils import mus_glob_grid  # noqa: E402
from utils.paths import REDUCED_BASIS  # noqa: E402


def solve_reduced(mu_glob, f, Ai_reduced, L_reduced, V, rbm: RBM_Offline):
    """
    Compute reduced solution u_N and lifted full field u_rb = V u_N
    using reduced operators and RBM_Offline.get_ai for parametric coefficients.
    """
    omega = 2 * np.pi * f

    # Coefficients a_i(mu, f) from RBM_Offline
    a = rbm.get_ai(mu_glob, f)

    # Build reduced matrix A_N(mu, f) = sum_i a_i A_i^N
    A_N = np.zeros_like(Ai_reduced[0], dtype=complex)
    for alpha, A_i_red in zip(a, Ai_reduced):
        A_N += alpha * A_i_red

    # Reduced RHS F_N = i rho0 omega L_reduced
    F_N = 1j * rbm.setting.rho0 * omega * L_reduced

    # Solve reduced system
    u_N = np.linalg.solve(A_N, F_N)

    # Lift to full space
    u_rb = V @ u_N
    return u_rb, u_N


def build_mu_test(n_mu_test=5, rng=None):
    """
    Build a small set of geometric parameters mu_glob for testing.

    Uses mus_glob_grid then selects a subset of size n_mu_test.
    """
    l_min = L_MIN / L

    mu_all = mus_glob_grid(
        l_mins=(l_min, l_min, l_min),
        R_mins=(R_MIN, R_MIN, R_MIN),
        R_max=R_MAX,
        num_l=20,
        num_R=20,
        l_min4=l_min,
    )

    n_total = mu_all.shape[0]
    if n_mu_test >= n_total:
        return mu_all

    # Select a subset of indices (deterministic sampling)
    # idx = np.linspace(0, n_total - 1, n_mu_test).astype(int)
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(mu_all.shape[0], size=n_mu_test, replace=False)
    mu_test = mu_all[idx]

    print(f"Number of test geometries: {mu_test.shape[0]}")
    return mu_test



i_mu = int(sys.argv[1])
mu = build_mu_test(n_mu_test=N_MU, rng=np.random.default_rng(0))[i_mu]
f_test = np.linspace(SIMULATION_FMIN, SIMULATION_FMAX, N_F)
setting = ProblemSetting()
base = REDUCED_BASIS / "ML" / f"rank{RANK}"
# the solver yields a view of one PETSc vector that is reused for every frequency: copy it
u_full = [np.array(u, copy=True) for u in solve_system_parametric(mu_glob=mu, frequencies=f_test, store_vectors=False)]
models, res = {}, []
for f, uf in zip(f_test, u_full):
    zone = int(np.argmin(np.abs(FREQUENCY_ZONES_CENTERS - f))) + 1
    if zone not in models:
        d = np.load(sorted(base.glob(f"zone{zone}[*.npz"))[0], allow_pickle=True)
        models[zone] = (d["Ai_reduced"], d["L_reduced"], d["V"],
                        RBM_Offline(setting=setting, mus_globs=[], frequencies=[], V=d["V"]))
    A, Lr, V, rbm = models[zone]
    u_rb, _ = solve_reduced(mu_glob=mu, f=f, Ai_reduced=A, L_reduced=Lr, V=V, rbm=rbm)
    res.append([i_mu, f, zone, np.linalg.norm(uf - u_rb) / np.linalg.norm(uf)])
np.savez(OUT / f"mu{i_mu:02d}.npz", res=np.array(res), mu=mu)
print(f"mu {i_mu} = {np.round(mu, 4)}: median relative error {np.median(np.array(res)[:, 3]):.2e}")
