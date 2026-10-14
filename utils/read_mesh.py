from functools import lru_cache
from mpi4py import MPI
from dolfinx import io
import sys

# To read the mesh only once
@lru_cache(maxsize=None)
def read_mesh(mesh_path: str):
    import gmsh
    # To mute gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    
    msg_start = f"Reading {mesh_path}: in progress..."
    msg_done = f"Reading {mesh_path}: done"
    print(msg_start, end="", flush=True)

    domain, cell_tags, facet_tags, _, _, _ = io.gmshio.read_from_msh(
        mesh_path, comm=MPI.COMM_SELF, rank=0, gdim=3
    )

    sys.stdout.write("\r" + " " * len(msg_start) + "\r")
    sys.stdout.write(msg_done + "\n")
    sys.stdout.flush()

    return domain, cell_tags, facet_tags
