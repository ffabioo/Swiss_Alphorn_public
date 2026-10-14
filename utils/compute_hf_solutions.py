from mpi4py import MPI
from core.parametric_solver import solve_system_parametric
from utils.settings import ProblemSetting

from utils.valais import L, L_MIN, R_MIN, R_MAX
from utils.mus_utils import mus_glob_grid
from utils.frequencies import SNAPSHOTS_FREQUENCIES
from utils.paths import SNAPSHOTS


def distribute_tasks(mu_list, size, rank):
    """
    Split mu_glob_list into size chunks.
    Each rank gets one chunk, possibly empty.
    """
    n = len(mu_list)
    chunk = n // size
    rest = n % size

    start = rank * chunk + min(rank, rest)
    end = start + chunk + (1 if rank < rest else 0)

    return list(range(start, end))  # return indices instead of values


def mu_to_str(mu):
    return "_".join(f"{x:.6f}" for x in mu)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"MPI started with {size} processes")

    # -----------------------------
    # Generate geometric parameters
    # -----------------------------
    setting = ProblemSetting()

    l_min = L_MIN / L

    mu_glob_list = mus_glob_grid(
        l_mins=(l_min, l_min, l_min),
        R_mins=(R_MIN, R_MIN, R_MIN),
        R_max=R_MAX,
        # num_l=6 with l4 >= l_min gives 35 x 35 = 1225 geometries
        # (num_l=5 gave 1225 too, but 525 of them had l4 < l_min)
        num_l=6,
        num_R=5,
        l_min4=l_min,
    )

    total_mu = len(mu_glob_list)

    if rank == 0:
        print(f"Total number of mu_glob = {total_mu}")

    # -----------------------------
    # Distribute workload
    # -----------------------------
    local_mu_indices = distribute_tasks(mu_glob_list, size, rank)
    print(f"[Rank {rank}] received {len(local_mu_indices)} mu_glob")

    # -----------------------------
    # Frequencies
    # -----------------------------
    freqs = SNAPSHOTS_FREQUENCIES

    if rank == 0:
        print(f"Total number of frequencies = {len(freqs)}")

    # -----------------------------
    # Solve and store in separate DBs
    # -----------------------------
    n_local = len(local_mu_indices)
    
    total_count = 0
    for mu_counter, idx in enumerate(local_mu_indices, start=1):
        mu_glob = mu_glob_list[idx]

        # Unique database for each mu
        mu_str = mu_to_str(mu_glob)
        db_name = f"solutions_{mu_str}"


        count = 0
        for _ in solve_system_parametric(
            mu_glob=mu_glob,
            frequencies=freqs,
            store_vectors=True,
            verbose=False,
            database_name=db_name,
            database_folder=SNAPSHOTS,
        ):
            count += 1

        total_count += count

        print(f"[Rank {rank}] computed {mu_counter} / {n_local} mus", flush=True)

    print(f"[Rank {rank}] total solutions = {total_count}")

    comm.Barrier()

    if rank == 0:
        print("MPI run complete")


if __name__ == "__main__":
    main()
