import numpy as np
from dolfinx import fem, io
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
from functools import lru_cache

from utils.read_mesh import read_mesh
from utils.settings import ProblemSetting
from utils.mus_utils import compute_mu_glob_from_mesh
from core.all_cones_problem import AllConesProblem
from utils.cache_solutions import CacheVectors
from utils.paths import SOLVER_CACHE


@lru_cache(maxsize=1)
def compute_constants_matrices(setting):
    domain, cell_tags, facet_tags = read_mesh(setting.mesh_path)

    # Function space
    V = fem.functionspace(domain, setting.function_space)

    dx = ufl.Measure(
        "dx", 
        domain=domain,
        subdomain_data=cell_tags,
        metadata={"quadrature_degree": setting.quadrature_degree}
    )
    ds = ufl.Measure(
        "ds", 
        domain=domain,
        subdomain_data=facet_tags,
        metadata={"quadrature_degree": setting.quadrature_degree}
    )

    dx_non_conic = dx(setting.vol_mouthpiece_id) + dx(setting.vol_bell_id)
    
    # Trail and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ----------------------------------
    # --- Assemble constant matrices ---
    # ----------------------------------
    def assemble_mat(biform: ufl.form.Form) -> PETSc.Mat:
        A = fem.petsc.assemble_matrix(
            fem.form(biform, form_compiler_options={"scalar_type": "complex"})
        )
        A.assemble()
        return A

    # Mass matrix (on the non conic parts)
    m_non_conic = ufl.inner(p, v) * dx_non_conic
    M_non_conic = assemble_mat(m_non_conic)

    # Stiffness matrix (on the non conic parts)
    k_non_conic = ufl.inner(ufl.grad(p), ufl.grad(v)) * dx_non_conic
    K_non_conic = assemble_mat(k_non_conic)

    # Boundary mass matrix
    bm = ufl.inner(p, v) * ds(setting.surface_outlet_id)
    Bm = assemble_mat(bm)

    # RHS vector
    u_n = setting.u_n
    l = ufl.inner(u_n, v) * ds(setting.surface_inlet_id)
    L = fem.petsc.assemble_vector(
        fem.form(l, form_compiler_options={"scalar_type": "complex"})
    )

    return M_non_conic, K_non_conic, Bm, L


def setup_solver(comm):
    opts = PETSc.Options()
    # Automatic reordering
    opts["mat_mumps_icntl_7"]  = 7
    # Detection of zero pivots during factorization
    opts["mat_mumps_icntl_24"] = 1
    ksp = PETSc.KSP().create(comm)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setFromOptions()
    return ksp


def solve_system_parametric(
        mu_glob=None, 
        frequencies = [400], 
        store_vectors=False, 
        verbose=False, 
        debug_matrices=False,
        database_name=None,
        database_folder=SOLVER_CACHE
    ):
    # Problem setting
    setting = ProblemSetting()

    if mu_glob is None:
        if verbose: print("mu_glob is None -> setting mu_glob to mu_glob_h")
        mu_glob = compute_mu_glob_from_mesh(
            setting.mesh_path,
            setting.cones_ids
        )

    # To store the vector solutions
    if database_name is None:
        database_name = "par_" + setting.mesh_path.stem
    cache = CacheVectors(database_name, database_folder)


    # Initialize the problem on the 4 reference cones
    all_cones = AllConesProblem(problem_setting=setting)

    # Precompute constant matrices
    M_non_conic, K_non_conic, Bm, L = compute_constants_matrices(setting)

    # Physical constants shorthand
    c = setting.c
    R = setting.R
    rho0 = setting.rho0

    # Linear system objects (Ax = F)
    A = M_non_conic.duplicate(copy=False)
    x = M_non_conic.createVecRight()
    F = M_non_conic.createVecRight()

    # Setup the solver
    solver = setup_solver(M_non_conic.getComm())

    # frequency loop
    for f in frequencies:
        key = (mu_glob, f)
        if key in cache:
            if verbose:
                print(f"Using cached solution for f = {f}")
            yield cache[key]
            continue

        if verbose: 
            print(f"Solving for frequency f = {f}")

        omega = 2 * np.pi * f
        k = omega / c

        # ----------------------------------
        # --- Assemble the system matrix ---
        # ----------------------------------
        # A <- 0
        A.zeroEntries()
        
        # A <- 1*K
        A.axpy(1.0, K_non_conic)
        for K_cone in all_cones.stiffness_matrix(mu_glob):
            A.axpy(1.0, K_cone)
        
        # A <- A - k^2 M
        A.axpy(-(k**2), M_non_conic)
        for M_cone in all_cones.mass_matrix(mu_glob):
            A.axpy(-(k**2), M_cone)
        
        # Boundary coefficient
        coef_Bm = (1.0 - 1j * k * R) / R
        # A <- A + coef * Bm
        A.axpy(coef_Bm, Bm)
        
        A.assemble()

        # ------------------------
        # --- Assemble the RHS ---
        # ------------------------
        F.zeroEntries()
        coef_F = 1j * rho0 * omega
        F.axpy(coef_F, L)
        F.assemble()

        # ---------------------------------
        # ----- solve A x = F_current -----
        # ---------------------------------
        solver.setOperators(A)
        solver.setUp()
        solver.solve(F, x)

        solution = x.array

        if debug_matrices:
            M = M_non_conic.copy()
            for Mi in all_cones.mass_matrix(mu_glob):
                M.axpy(1.0, Mi)
            M.assemble()
            K = K_non_conic.copy()
            for Ki in all_cones.stiffness_matrix(mu_glob):
                K.axpy(1.0, Ki)
            K.assemble()            
            yield solution, A, F, Bm, M, K
            return

        if store_vectors:
            cache[key] = solution
        
        yield solution
