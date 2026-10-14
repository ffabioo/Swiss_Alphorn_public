import numpy as np

from mpi4py import MPI

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

from utils.read_mesh import read_mesh


class ConeProblem:
    def __init__(
        self, 
        setting,
        verbose = False
    ):    
        self.setting = setting
        self.verbose = verbose

        self.domain, self.cell_tags, self.facet_tags = read_mesh(setting.mesh_path)

        # Function space
        self.V = fem.functionspace(self.domain, setting.function_space)
        
        # Define dx for integration
        self._dx = ufl.Measure(
            "dx", 
            domain=self.domain,
            subdomain_data=self.cell_tags,
            metadata={"quadrature_degree": setting.quadrature_degree}
        )
        # Restrained on the volume defined by volume_id
        self.dx = self._dx(self.setting.volume_id)

        x1h, x2h, r1h, r2h = self.setting.muh
        self.alphah = (r2h-r1h) / (x2h-x1h)

    def a(self, mu):
        x1h, x2h, _, _ = self.setting.muh
        x1, x2, _, _ =  mu
        return (x2 - x1) / (x2h - x1h)

    def alpha(self, mu):
        x1h, x2h, _, _ = self.setting.muh
        _, _, r1, r2 =  mu
        return (r2 - r1) / (x2h - x1h)

    def N(self, x, mu):
        x1h, _, _, _ = self.setting.muh
        _, _, r1, _ =  mu
        return r1 + self.alpha(mu) * (x - x1h)

    def D(self, x):
        x1h, _, r1h, _ = self.setting.muh
        return r1h + self.alphah * (x - x1h)

    def l(self, x, mu):
        return self.N(x, mu) / self.D(x)


    def _assemble_Ki(self):
        x1h, _, _, _ = self.setting.muh

        x, y, z = ufl.SpatialCoordinate(self.domain)
        D = self.D(x)

        A1 = ufl.as_matrix([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        
        A234 = ufl.as_matrix([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        A2 = (1 / D**2) * A234
        A3 = (2 * (x - x1h) / D**2) * A234
        A4 = ((x - x1h)**2 / D**2) * A234

        A56  = ufl.as_matrix([
                [0, y, z],
                [y, 0, 0],
                [z, 0, 0]
            ])
        A5 = (1 / D**3) * A56
        A6 = ((x - x1h) / D**3) * A56

        A7 = (1 / D**4) * ufl.as_matrix([
                [0, 0, 0],
                [0, y**2, y*z],
                [0, y*z, z**2]
            ])

        p = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Preassemble Ki matrices
        self.Ki = []
        for Ai in [A1, A2, A3, A4, A5, A6, A7]:
            form_i = ufl.inner(ufl.grad(p), Ai * ufl.grad(v)) * self.dx
            K_i = fem.petsc.assemble_matrix(
                    fem.form(form_i , form_compiler_options={"scalar_type": "complex"})
                )
            K_i.assemble()
            self.Ki.append(K_i)

        if self.verbose: print(f"{self.setting.name()}: Stiffness components assembled")


    def _assemble_Mj(self):
        x1h, _, _, _ = self.setting.muh

        x, y, z = ufl.SpatialCoordinate(self.domain)
        D = self.D(x)

        # Define Bj(x)
        B1 = (1 / D**2)
        B2 = (2 * (x - x1h) / D**2)
        B3 = ((x - x1h)**2 / D**2)

        p = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Preassemble Mj matrices
        self.Mj = []
        for Bj in [B1, B2, B3]:
            form_j = Bj * ufl.inner(p, v) * self.dx
            M_j = fem.petsc.assemble_matrix(
                    fem.form(form_j , form_compiler_options={"scalar_type": "complex"})
                )
            M_j.assemble()
            self.Mj.append(M_j)

        if self.verbose: print(f"{self.setting.name()}: Mass components assembled")

    def omegas(self, mu):
        _, _, r1h, _ = self.setting.muh
        _, _, r1, _ = mu
        a   = self.a(mu)
        alpha = self.alpha(mu)
        alphah = self.alphah

        w1 = a
        w2 = (r1**2)      / a
        w3 = (r1*alpha)   / a
        w4 = (alpha**2)   / a
        
        w5 = -(alpha*r1h*r1 - alphah*(r1**2)) / a
        w6 = -(alpha**2 * r1h - alpha*alphah*r1) / a

        w7 = ((alpha*r1h - alphah*r1)**2) / a
        return np.array([w1, w2, w3, w4, w5, w6, w7])

    def thetas(self, mu):
        _, _, r1, _ = mu
        a   = self.a(mu)
        alpha = self.alpha(mu)
        
        t1 = a * (r1**2)
        t2 = a * alpha * r1
        t3 = a * (alpha**2)
        return np.array([t1, t2, t3])
    
    def _lincomb_into(self, target: PETSc.Mat, mats, coeffs):
        target.zeroEntries()
        for c, M in zip(coeffs, mats):
            if c != 0.0:
                target.axpy(c, M)
        target.assemble()

    def _build_K(self, mu):
        if not hasattr(self, "Ki"):
            self._assemble_Ki()
        if not hasattr(self, "K"):
            self.K = self.Ki[0].duplicate(copy=False)
        omegas = self.omegas(mu)
        self._lincomb_into(self.K, self.Ki, omegas)

    def _build_M(self, mu):
        if not hasattr(self, "Mj"):
            self._assemble_Mj()
        if not hasattr(self, "M"):
            self.M = self.Mj[0].duplicate(copy=False)
        thetas = self.thetas(mu)
        self._lincomb_into(self.M, self.Mj, thetas)

    # # Deprecated: assemble full system matrix
    # def assemble_matrix(self, f, mu):
    #     self._build_K(mu)
    #     self._build_M(mu)

    #     A = self.K.duplicate(copy=False)

    #     omega = 2 * np.pi * f
    #     k = omega / self.setting.c

    #     # Assemble system matrix
    #     A.zeroEntries()
    #     # A <- 1*K
    #     A.axpy(1.0, self.K)
    #     # A <- A - k^2 M
    #     A.axpy(-(k**2), self.M)
    #     A.assemble()

    #     if self.verbose: print(f"{self.setting.name()}: System matrix assembled for f = {f}")
        
    #     return A
    

    def mass_matrix(self, mu):    
        self._build_M(mu)
        return self.M
    
    def stiffness_matrix(self, mu):
        self._build_K(mu)
        return self.K
    
    def get_Mj(self):
        if not hasattr(self, "Mj"):
            self._assemble_Mj()
        return self.Mj

    def get_Ki(self):
        if not hasattr(self, "Ki"):
            self._assemble_Ki()
        return self.Ki