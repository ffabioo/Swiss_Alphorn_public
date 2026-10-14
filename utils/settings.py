import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProblemSetting:
    mesh_path: Path = Path("mesh/reference_domain.msh")
    cones_ids = [1, 2, 3, 4]
    vol_mouthpiece_id: int = 0
    vol_bell_id: int = 5
    surface_inlet_id: int = 2
    surface_outlet_id: int = 3

    # Physical constants
    R:    float = 447.09            # mm
    u_n:  float = 1.0*1000          # mm/s
    rho0: float = 1.225*1000/10**9  # g/mm^3
    c:    float = 340.0*1000        # mm/s

    function_space: tuple = ("Lagrange", 1)

    quadrature_degree: int = 4

    def __hash__(self):
        # Used for caching
        return hash((
            self.mesh_path,
            tuple(self.cones_ids),
            self.function_space,
            self.quadrature_degree,
            self.vol_mouthpiece_id,
            self.vol_bell_id,
            self.surface_inlet_id,
            self.surface_outlet_id,
        ))


@dataclass(kw_only=True)
class ConeProblemSetting(ProblemSetting):
    # Dimensions of the reference cone
    # Example: np.array([0, 1, 2, 3]) means the cone goes from x=0 to x=1 
    # with first radius = 2 and second radius = 3
    muh: np.ndarray

    # Volume index in the global mesh
    volume_id: int

    def name(self):
        return f"Cone {self.volume_id}"
