"""
Export the FEM quantities needed by ml/rb_fast.py (pure numpy reduced-basis evaluator):
  - ell: vector with ell_i = int_{Gamma_in} phi_i, so that int_{Gamma_in} p = ell . p
  - inlet_area
  - cone_muh: reference parameters (x1, x2, r1, r2) of the four cones, as computed by AllConesProblem
Run (in the dolfinx container) from the repo root:  python ml/export_rb_fast_data.py [output, default databases/reduced_basis/rb_fast_data.npz]
"""
import sys
import numpy as np
from pathlib import Path
from dolfinx import fem
import dolfinx.fem.petsc
import ufl

from utils.settings import ProblemSetting
from utils.read_mesh import read_mesh
from core.all_cones_problem import AllConesProblem
from utils.paths import REDUCED_BASIS

setting = ProblemSetting()
domain, _, facet_tags = read_mesh(setting.mesh_path)
V = fem.functionspace(domain, setting.function_space)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags,
                 metadata={"quadrature_degree": setting.quadrature_degree})
v = ufl.TestFunction(V)
ell = fem.petsc.assemble_vector(fem.form(ufl.inner(1.0, v) * ds(setting.surface_inlet_id),
                                         form_compiler_options={"scalar_type": "complex"}))
ell = ell.getArray().real.copy()
area = fem.assemble_scalar(fem.form(1.0 * ds(setting.surface_inlet_id))).real
cones = AllConesProblem(problem_setting=setting)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else REDUCED_BASIS / "rb_fast_data.npz"
np.savez(out, ell=ell, inlet_area=area, cone_muh=np.array(cones.local_mus),
         c=setting.c, R=setting.R, rho0=setting.rho0, u_n=setting.u_n)
print(f"saved {out}: ell sum = {ell.sum():.6f} (inlet area {area:.6f}), cone_muh =\n{np.array(cones.local_mus)}")
