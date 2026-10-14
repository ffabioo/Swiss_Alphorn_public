"""
Affine decomposition of the FE operators, A(mu, f) = sum_i a_i(mu, f) A_i, and its projection on a reduced basis V
(A_i^N = V^H A_i V, L^N = V^H L), written to the zone files of the reduced model (step 3, rbm_by_zone/db_to_rb_by_zone.py).
"""
import numpy as np
from dataclasses import dataclass
from petsc4py import PETSc
from mpi4py import MPI

from utils.settings import ProblemSetting
from core.parametric_solver import compute_constants_matrices
from core.all_cones_problem import AllConesProblem


def project_operators_on_basis(Ai, L, V):
    """ Compute reduced operators A_i_N = V^H A_i V and reduced source vector L_N = V^H L for the reduced basis V """
    D, N = V.shape

    # Make sure V is Fortran contiguous because PETSc expects column-major layout
    V_f = np.asfortranarray(V)
    V_conj_f = np.asfortranarray(np.conjugate(V_f))

    comm = Ai[0].getComm()

    # PETSc dense matrices for V and conj(V)
    V_petsc = PETSc.Mat().createDense([D, N], array=V_f, comm=comm)
    V_conj_petsc = PETSc.Mat().createDense([D, N], array=V_conj_f, comm=comm)

    # List that will store A_i_N = V^H A_i V
    Ai_reduced = []

    for A in Ai:
        # Z = A * V
        Z = A.matMult(V_petsc)

        # V^H A V = (conj(V))^T * Z
        VhAV_petsc = V_conj_petsc.transposeMatMult(Z)

        # Store as NumPy array (N x N)
        Ai_reduced.append(VhAV_petsc.getDenseArray().copy())

        # Destroy temporaries to free PETSc memory
        Z.destroy()
        VhAV_petsc.destroy()

    # Precompute reduced right-hand side L_N = V^H L
    y = PETSc.Vec().createSeq(N)
    V_conj_petsc.multTranspose(L, y)
    L_reduced = y.getArray().copy()

    # Destroy PETSc dense matrices for V
    V_petsc.destroy()
    V_conj_petsc.destroy()

    return Ai_reduced, L_reduced


@dataclass
class RBM_Offline:
    setting: ProblemSetting
    mus_globs: list
    frequencies: list
    V: np.ndarray
    verbose: bool = False
    comm: MPI.Comm | None = None

    def __post_init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.all_cones = AllConesProblem(problem_setting=self.setting)
        self.Ai, self.L = self.get_Ai_L()

        # The published reduced operators were projected from A_i^H: the first version of this class transposed the
        # A_i in place (a side effect of a greedy-algorithm step, since removed). The A_i are symmetric up to rounding,
        # so A_i^H = A_i to ~1e-16; the transposition is kept so that step 3 reproduces the published files bit for bit.
        for A in self.Ai:
            A.hermitianTranspose()   # in place

    def get_Ai_L(self):
        M_non_conic, K_non_conic, Bm, L = compute_constants_matrices(self.setting)

        Ki, Mj = [], []
        for cone in self.all_cones.cones:
            Ki.extend(cone.get_Ki())
            Mj.extend(cone.get_Mj())

        Ai = [K_non_conic, *Ki, M_non_conic, *Mj, Bm]
        return Ai, L

    def get_ai(self, mu_glob, f):
        omega = 2 * np.pi * f
        k = omega / self.setting.c

        K_non_conic = 1.0
        M_non_conic = -(k**2)

        # Cone-dependent coefficients: omegas for Ki, thetas for Mj
        omegas, thetas = self.all_cones.omegas_thetas(mu_glob)

        Ki = omegas
        Mj = -(k**2) * thetas

        Bm = (1 - 1j * k * self.setting.R) / self.setting.R

        return np.concatenate([
            np.array([K_non_conic], dtype=complex),
            Ki,
            np.array([M_non_conic], dtype=complex),
            Mj,
            np.array([Bm], dtype=complex)
        ])

    def compute_and_save_reduced_matrices(self, path: str):
        """
        Build the reduced operators associated with the current basis self.V
        and save them to a .npz file.

        File contents
            Ai_reduced : array (nA, N, N) with A_i^N = V^H A_i V
            L_reduced  : vector (N,)
            V          : reduced basis (D, N)
            frequencies
            mus_globs
        """

        if self.rank != 0:
            return

        if self.V is None:
            raise ValueError(
                "Reduced basis self.V is not defined. "
                "Provide V first."
            )

        # Make sure self.Ai and self.L exist
        if not hasattr(self, "Ai") or self.Ai is None or not hasattr(self, "L"):
            Ai, L = self.get_Ai_L()
            self.Ai = Ai
            self.L = L

        # Projection V^H A_i V and V^H L
        Ai_reduced_list, L_reduced = project_operators_on_basis(
            self.Ai,
            self.L,
            self.V,
        )

        # Stack reduced matrices into a single array
        # Ai_reduced_array[i] = A_i^N
        Ai_reduced_array = np.stack(Ai_reduced_list, axis=0)

        # Save everything
        np.savez(
            path,
            Ai_reduced=Ai_reduced_array,
            L_reduced=L_reduced,
            V=self.V,
            frequencies=np.array(self.frequencies),
            mus_globs=np.array(self.mus_globs, dtype=object),
        )

        if self.verbose:
            print(f"Reduced operators saved to {path}")
            print(f"  reduced size N      : {Ai_reduced_array.shape[1]}")

        # ============================
        #  RBM NUMERICAL DIAGNOSTICS
        # ============================

        print("\n=== RBM DIAGNOSTICS ===")

        def norm_mat(A):
            return A.norm(PETSc.NormType.FROBENIUS)

        # --- 2) Orthonormality of the reduced basis V ---
        V = self.V
        H = V.conj().T @ V
        N_red = H.shape[0]
        ortho_err = np.linalg.norm(H - np.eye(N_red))
        print("\nOrthonormality error =", ortho_err)

        # --- 3) Identity projection test ---
        # For the identity matrix, V^H I V = V^H V = H
        I_red = H
        print("\nIdentity projection test")
        print("||I_red - I|| =", np.linalg.norm(I_red - np.eye(N_red)))

        # --- 4) Group diagnostics: K_non_conic, Ki, M_non_conic, Mj, Bm ---
        # Determine counts using self.all_cones
        n_Ki_total = sum(len(cone.get_Ki()) for cone in self.all_cones.cones)
        n_Mj_total = sum(len(cone.get_Mj()) for cone in self.all_cones.cones)

        # Same ordering as in get_Ai_L :
        # Ai = [K_non_conic, *Ki, M_non_conic, *Mj, Bm]
        idx = 0
        K_non_conic_full = self.Ai[idx]
        K_non_conic_red  = Ai_reduced_array[idx]
        idx += 1

        Ki_full = self.Ai[idx : idx + n_Ki_total]
        Ki_red  = Ai_reduced_array[idx : idx + n_Ki_total]
        idx += n_Ki_total

        M_non_conic_full = self.Ai[idx]
        M_non_conic_red  = Ai_reduced_array[idx]
        idx += 1

        Mj_full = self.Ai[idx : idx + n_Mj_total]
        Mj_red  = Ai_reduced_array[idx : idx + n_Mj_total]
        idx += n_Mj_total

        Bm_full = self.Ai[idx]
        Bm_red  = Ai_reduced_array[idx]

        print("\nGrouped operator diagnostics")

        print("||K_non_conic_full|| =", norm_mat(K_non_conic_full))
        print("||K_non_conic_red||  =", np.linalg.norm(K_non_conic_red))

        for k, (F, R) in enumerate(zip(Ki_full, Ki_red)):
            print(f"||Ki[{k}]_full|| =", norm_mat(F))
            print(f"||Ki[{k}]_red||  =", np.linalg.norm(R))

        print("||M_non_conic_full|| =", norm_mat(M_non_conic_full))
        print("||M_non_conic_red||  =", np.linalg.norm(M_non_conic_red))

        for k, (F, R) in enumerate(zip(Mj_full, Mj_red)):
            print(f"||Mj[{k}]_full|| =", norm_mat(F))
            print(f"||Mj[{k}]_red||  =", np.linalg.norm(R))

        print("||Bm_full|| =", norm_mat(Bm_full))
        print("||Bm_red||  =", np.linalg.norm(Bm_red))

        print("\n=== END OF RBM DIAGNOSTICS ===\n")


