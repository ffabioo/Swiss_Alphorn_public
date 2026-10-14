import numpy as np


# The notes of the alphorn scale (Hz)
OFFICIAL_NOTES = [
    {"freq": 92,  "name": "Do"},
    {"freq": 139, "name": "Sol"},
    {"freq": 185, "name": "Do"},
    {"freq": 234, "name": "Mi"},
    {"freq": 278, "name": "Sol"},
    {"freq": 331, "name": "Sib"},
    {"freq": 371, "name": "Do"},
    {"freq": 417, "name": "Ré"},
    {"freq": 468, "name": "Mi"},
    {"freq": 556, "name": "Sol"},
]

OFFICIAL_NOTES_FREQUENCIES = np.sort(np.array([
    note["freq"] for note in OFFICIAL_NOTES
]))


# To handle the big gap between the last two notes we add 512
FREQUENCY_ZONES_CENTERS = np.sort(np.append(OFFICIAL_NOTES_FREQUENCIES, 512))

# ADDED BEFORE AND AFTER THE SMALLEST AND BIGGEST NOTES OT PRODUCE THE SNAPSHOTS
SNAPSHOTS_MARGIN = 46 # ~= means of the distances between notes
SNAPSHOTS_FMIN = FREQUENCY_ZONES_CENTERS[0] - SNAPSHOTS_MARGIN
SNAPSHOTS_FMAX = FREQUENCY_ZONES_CENTERS[-1] +  SNAPSHOTS_MARGIN

SIMULATION_MARGIN = SNAPSHOTS_MARGIN/2
SIMULATION_FMIN = FREQUENCY_ZONES_CENTERS[0] - SIMULATION_MARGIN
SIMULATION_FMAX = FREQUENCY_ZONES_CENTERS[-1] +  SIMULATION_MARGIN


def dense_around(
        xs,                # points to be dense around
        xmin,              # lower bound of the interval
        xmax,              # upper bound of the interval
        N=200,             # Number of points
        sigma=0.05,        # Variance of the gaussians
        alpha_uniform=0.2, # Proportion of uniform density in the sample (alpha = 1 -> linspace, alpha = 0 -> mixture of gaussians)
        add_xs=True,       # Whether to force the xs to be in the interval 
    ):
    xs = np.asarray(xs)

    assert np.all((xs > xmin) & (xs < xmax))

    # Base grid where the density is evaluated
    base = np.linspace(xmin, xmax, N)

    # Sum of Gaussian bumps centered on each x
    w = np.zeros_like(base, dtype=float)
    for x in xs:
        w += np.exp(-(base - x) ** 2 / sigma ** 2)

    # Add a uniform component to enforce a minimal density
    if alpha_uniform > 0.0:
        w = (1.0 - alpha_uniform) * w + alpha_uniform

    # Normalize weights
    s = w.sum()
    w /= s

    # Cumulative distribution function
    cdf = np.cumsum(w)

    # Sample N points from this distribution
    u = np.linspace(0.0, 1.0, N)
    new_points = np.interp(u, cdf, base)

    # Optionally include original xs
    if add_xs:
        new_points = np.hstack([new_points, xs])
    
    return np.sort(np.unique(new_points))


SNAPSHOTS_FREQUENCIES = dense_around(
    FREQUENCY_ZONES_CENTERS,
    N=200,
    xmin=SNAPSHOTS_FMIN,
    xmax=SNAPSHOTS_FMAX,
    sigma=10,
    alpha_uniform=1,
    add_xs=False
)


def build_frequency_zones(overlap, xs=FREQUENCY_ZONES_CENTERS):
    """
    overlap: extra margin added to the radius
    xs: 1D ndarray of floats, positions of the freqs

    Returns:
      - intervals: ndarray (n, 2) containing [left, right] for each xs
    """
    assert xs.size >= 1

    # Sort values for safety
    xs_sorted = np.sort(xs)

    # Compute midpoints between consecutive points
    midpoints = 0.5 * (xs_sorted[:-1] + xs_sorted[1:])

    # Distances from points to their neighboring midpoints
    dist_left = midpoints - xs_sorted[:-1]
    dist_right = xs_sorted[1:] - midpoints

    # Maximum distance from any point to a midpoint
    max_point_to_midpoint_dist = np.max(np.maximum(dist_left, dist_right))

    # Final radius including overlap
    final_radius = max_point_to_midpoint_dist + overlap

    # Safety check against global margin
    assert final_radius < SNAPSHOTS_MARGIN, "overlap too big: the boundary notes do not have enough snapshots"

    # Build intervals centered on each original xs value
    intervals = np.column_stack((xs - final_radius, xs + final_radius))
    
    return intervals
