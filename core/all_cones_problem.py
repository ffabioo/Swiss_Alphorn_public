import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

from utils.settings import ConeProblemSetting
from core.cone_problem import ConeProblem
from utils.mus_utils import compute_local_mus_from_mesh, to_local


class AllConesProblem:
    def __init__(self, problem_setting, verbose=False):
        self.problem_setting = problem_setting
        self.cone_volume_ids = problem_setting.cones_ids

        self.local_mus = compute_local_mus_from_mesh(
            self.problem_setting.mesh_path,
            self.cone_volume_ids
        )
        
        self.cone_settings = [
            ConeProblemSetting(muh=muh, volume_id=id)
            for muh, id in zip(self.local_mus, self.cone_volume_ids)
        ]

        self.cones = [
            ConeProblem(setting, verbose=verbose)
            for setting in self.cone_settings
        ]

        self.x1 = self.cone_settings[0].muh[0]
        self.x5 = self.cone_settings[-1].muh[1]
        
        self.L = self.x5 - self.x1

        self.R1 = self.cone_settings[0].muh[2]
        self.R5 = self.cone_settings[-1].muh[3]


    def __call__(self, f, mu_glob):
        """Iterator on the assembled matrices on the cones"""
        local_mus = to_local(
            x1=self.x1, 
            R1=self.R1, 
            R5=self.R5, 
            L=self.L, 
            mu_glob=mu_glob
        )
        for cone, mu_local in zip(self.cones, local_mus):
            yield cone.assemble_matrix(f=f, mu=mu_local)

    def _local_mus_from_global(self, mu_glob):
        """
        Map a global parameter mu_glob to the list of local mus
        """
        return to_local(
            x1=self.x1,
            R1=self.R1,
            R5=self.R5,
            L=self.L,
            mu_glob=mu_glob
        )

    def omegas_thetas(self, mu_glob):
        """
        Compute all stiffness and mass coefficients for all cones
        given the global parameter mu_glob.

        Returns
        -------
        omegas_all : Concatenated omegas for all cones
        thetas_all : np.ndarray Concatenated thetas for all cones
        """
        local_mus = self._local_mus_from_global(mu_glob)

        omegas_list = []
        thetas_list = []

        for cone, mu_local in zip(self.cones, local_mus):
            omegas_list.append(cone.omegas(mu_local))
            thetas_list.append(cone.thetas(mu_local))

        omegas_all = np.concatenate(omegas_list)
        thetas_all = np.concatenate(thetas_list)

        return omegas_all, thetas_all


    def mass_matrix(self, mu_glob):
        """Iterator on the mass matrices on the cones"""
        local_mus = self._local_mus_from_global(mu_glob)
        for cone, mu_local in zip(self.cones, local_mus):
            yield cone.mass_matrix(mu=mu_local)

    def stiffness_matrix(self, mu_glob):
        """Iterator on the stiffness matrices on the cones"""
        local_mus = self._local_mus_from_global(mu_glob)
        for cone, mu_local in zip(self.cones, local_mus):
            yield cone.stiffness_matrix(mu=mu_local)
