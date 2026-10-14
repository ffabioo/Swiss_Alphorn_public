import numpy as np
from typing import Tuple


def to_local(x1, R1, R5, L, mu_glob):
    """
    Converts a global mu mu_glob = (l1, l2, l3, R2, R3, R4) 
    to the 4 local mus (x1, x2, r1, r2)
    """
    l1, l2, l3, R2, R3, R4 = mu_glob
    l4 = 1.0 - (l1 + l2 + l3)

    R = [R1, R2, R3, R4, R5]

    x = [x1]
    for l in [l1, l2, l3, l4]:
        x.append(x[-1] + l * L)

    return [
        np.array([x[i], x[i+1], R[i], R[i+1]]) 
        for i in range(4)
    ]


def to_global(local_mus):
    """
    Converts 4 local mus (x1, x2, r1, r2) to a global mu mu_glob = (l1, l2, l3, R2, R3, R4) 
    """
    x1 = local_mus[0][0]
    x5 = local_mus[-1][1]
    L = x5 - x1

    l_list = []
    R_list = []

    for mu in local_mus:
        x_start, x_end, r_start, r_end = mu
        l = (x_end - x_start) / L
        l_list.append(l)
        R_list.append(r_end)

    l1, l2, l3 = l_list[:3]
    R2, R3, R4 = R_list[:3]

    return np.array([l1, l2, l3, R2, R3, R4])


def compute_local_mus_from_mesh(mesh_path, cone_volume_ids):
        from utils.read_mesh import read_mesh
        domain, cell_tags, _ = read_mesh(mesh_path)

        def extract_one(volume_id):
            # Select volume cells
            mask = (cell_tags.values == volume_id)
            cells = cell_tags.indices[mask]

            # DOF -> coords
            gdof = domain.geometry.dofmap
            pts = np.unique(gdof[cells].reshape(-1))

            X = domain.geometry.x[pts]
            x = X[:, 0]
            y = X[:, 1]
            z = X[:, 2]

            r = np.sqrt(y**2 + z**2)

            x1 = x.min()
            x2 = x.max()

            x1_slice = np.where(x == x1)
            x2_slice = np.where(x == x2)

            r1 = np.max(r[x1_slice])
            r2 = np.max(r[x2_slice])


            # analytical_radii = lambda x: ( (r2 - r1)/(x2 - x1) * (x - x1) + r1 )
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter(x, r)
            # plt.plot(
            #     np.linspace(x1, x2, 100),
            #     analytical_radii( np.linspace(x1, x2, 100) ),
            #     color="red",
            #     label="Analytical radius"
            # )
            # plt.xlabel("x")
            # plt.ylabel("r")
            # plt.title(f"Volume ID {volume_id}")
            # plt.grid()
            # plt.savefig(f"volume_{volume_id}_xr.png", dpi=300)

            return np.array([x1, x2, r1, r2])

        mus = [extract_one(v) for v in cone_volume_ids]
        return np.vstack(mus)


def compute_mu_glob_from_mesh(mesh_path, cone_volume_ids):
    local_mus = compute_local_mus_from_mesh(mesh_path, cone_volume_ids)
    mu_glob = to_global(local_mus)
    return mu_glob


def mus_glob_grid(
    l_mins: Tuple[float, float, float],
    R_mins: Tuple[float, float, float],
    R_max: float,
    num_l: int = 10,
    num_R: int = 10,
    l_min4: float = 0.0,
) -> np.ndarray:
    """
    Build a grid of vectors (l1, l2, l3, R2, R3, R4) for grid search

    Constraints:
        - l1 in [l_min1, 1)
        - l2 in [l_min2, 1)
        - l3 in [l_min3, 1)
        - l1 + l2 + l3 < 1
        - l4 = 1 - (l1 + l2 + l3) >= l_min4 (l4 > 0 if l_min4 = 0)

        - R2 in [R_min2, R_max]
        - R3 in [R_min3, R_max]
        - R4 in [R_min4, R_max]
        - R2 <= R3 <= R4

    Parameters
    ----------
    l_mins : tuple of 3 floats
        (l_min1, l_min2, l_min3) lower bounds for l1, l2, l3.
    R_mins : tuple of 3 floats
        (R_min2, R_min3, R_min4) lower bounds for R2, R3, R4.
    R_max : float
        Common upper bound for R2, R3, R4.
    num_l : int
        Number of grid points per l_i dimension.
    num_R : int
        Number of grid points per R_i dimension.
    l_min4 : float
        Lower bound for the implicit fourth length l4 = 1 - (l1 + l2 + l3).

    Returns
    -------
    np.ndarray
        Array of shape (N, 6) where each row is (l1, l2, l3, R2, R3, R4).
    """

    l_min1, l_min2, l_min3 = l_mins
    R_min2, R_min3, R_min4 = R_mins

    # Basic sanity checks on bounds
    if num_l <= 0 or num_R <= 0:
        raise ValueError("num_l and num_R must be positive integers")

    if not (l_min1 < 1.0 and l_min2 < 1.0 and l_min3 < 1.0):
        raise ValueError("All l_min values must be strictly less than 1")

    if not (R_min2 <= R_max and R_min3 <= R_max and R_min4 <= R_max):
        raise ValueError("Each R_min must be less than or equal to R_max")

    # l_i: interval [l_min_i, 1)
    # endpoint=False means the upper bound (1.0) is excluded
    l1_values = np.linspace(l_min1, 1.0, num_l, endpoint=False)
    l2_values = np.linspace(l_min2, 1.0, num_l, endpoint=False)
    l3_values = np.linspace(l_min3, 1.0, num_l, endpoint=False)

    # R_i: interval [R_min_i, R_max] inclusive
    R2_values = np.linspace(R_min2, R_max, num_R, endpoint=True)
    R3_values = np.linspace(R_min3, R_max, num_R, endpoint=True)
    R4_values = np.linspace(R_min4, R_max, num_R, endpoint=True)

    # Create the 3D grid for l1, l2, l3
    # Shapes: each is (num_l, num_l, num_l)
    L1, L2, L3 = np.meshgrid(l1_values, l2_values, l3_values, indexing="ij")

    # Apply constraint l1 + l2 + l3 < 1 and l4 = 1 - (l1 + l2 + l3) >= l_min4
    mask = ((L1 + L2 + L3) < 1.0) & ((1.0 - (L1 + L2 + L3)) >= l_min4)

    # Extract only valid (l1, l2, l3) as a 2D array of shape (N_l, 3)
    valid_l = np.stack([L1[mask], L2[mask], L3[mask]], axis=1)

    # Now build the grid for R2, R3, R4
    # Shapes: each is (num_R, num_R, num_R)
    R2, R3, R4 = np.meshgrid(R2_values, R3_values, R4_values, indexing="ij")

    # Flatten to a 2D array of shape (N_R, 3)
    R_grid = np.stack([R2.ravel(), R3.ravel(), R4.ravel()], axis=1)


    # Filter R2 <= R3 <= R4
    R_grid = R_grid[R_grid[:, 0] <= R_grid[:, 1]]
    R_grid = R_grid[R_grid[:, 1] <= R_grid[:, 2]]

    # Cartesian product between valid_l (N_l, 3) and R_grid (N_R, 3)
    N_l = valid_l.shape[0]
    N_R = R_grid.shape[0]

    # Repeat each l row for all R combinations
    l_block = np.repeat(valid_l, N_R, axis=0)  # shape (N_l * N_R, 3)

    # Tile the R grid to match the number of l rows
    R_block = np.tile(R_grid, (N_l, 1))        # shape (N_l * N_R, 3)

    # Concatenate into final (N, 6) array
    grid = np.concatenate([l_block, R_block], axis=1)

    return grid


def random_admissible_mu_glob(l_min, R_min, R_max, seed):
    """Random mu_glob with l_i >= l_min (i = 1..4, l4 = 1 - l1 - l2 - l3) and R_min <= R2 <= R3 <= R4 <= R_max."""
    rng = np.random.default_rng(seed)
    l = l_min + (1.0 - 4.0 * l_min) * rng.dirichlet(np.ones(4))
    R = np.sort(rng.uniform(R_min, R_max, size=3))
    return np.array([l[0], l[1], l[2], R[0], R[1], R[2]])


def figure_random_mus(l_min, R_min, R_max):
    """Three random geometries of the forward-model figure. The first two are the ones used before
    (they satisfy l_i >= l_min); the third one is redrawn because the previous one had l3 = 0.065 < l_min."""
    return np.array([
        [0.5986585, 0.15601864, 0.15599452, 9.113763, 33.550175, 45.47793],
        [0.13949387, 0.29214466, 0.36636186, 15.48532, 27.02315, 41.83292],
        random_admissible_mu_glob(l_min, R_min, R_max, seed=2),
    ])
