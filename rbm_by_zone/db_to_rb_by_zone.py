import sqlite3
import io
import numpy as np
from pathlib import Path

from utils.settings import ProblemSetting
from rbm_by_zone.rbm_offline_by_zone import RBM_Offline
from utils.frequencies import build_frequency_zones
from utils.paths import MERGED_SNAPSHOTS, REDUCED_BASIS


def blob_to_array(blob):
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)

def load_snapshots(db_path, fmin=None, fmax=None):
    from utils.cache_solutions import _blob_to_array
    import sqlite3
    import numpy as np

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = "SELECT f, vector FROM results"
    params = []

    if fmin is not None and fmax is not None:
        query += " WHERE f BETWEEN ? AND ?"
        params.extend([fmin, fmax])

    cur.execute(query, params)
    rows = cur.fetchall()

    vectors = []
    freqs = []

    for f_val, vec_blob in rows:
        try:
            vec = _blob_to_array(vec_blob)
            if vec is not None:
                vectors.append(vec)
                freqs.append(f_val)
        except Exception:
            continue

    conn.close()

    if not vectors:
        raise ValueError("No snapshot found in this range")

    snapshot_matrix = np.column_stack(vectors)
    return snapshot_matrix


# -------------------------------------------------------------
# Compute reduced basis from SVD
# -------------------------------------------------------------


# -------------------------------------------------------------
# Compute reduced basis from randomized SVD
# -------------------------------------------------------------
def compute_reduced_basis_SVD_randomized(
        snapshots,
        energy=0.9999,
        max_rank=500,
        n_iter=2,
        random_state=0,
    ):
    """
    SVD randomisée pour construire une base réduite.

    snapshots : matrice (n_dofs, n_snapshots)
                chaque colonne correspond à un snapshot
    energy : fraction d'énergie à capturer (0 < energy < 1)
    max_rank : rang maximal autorisé pour la base réduite
    n_iter : nombre d'itérations de puissance pour mieux séparer les modes
    random_state : graine pour la reproductibilité
    """
    print(f"Snapshot matrix shape: {snapshots.shape}")
    m, n = snapshots.shape

    # Rang cible brut
    target_rank = min(max_rank, m, n)
    l = min(n, target_rank)

    print(f"Randomized SVD with target_rank={target_rank}")
    print(f"Subspace dimension l={l}")

    rng = np.random.default_rng(random_state)
    # Matrice gaussienne pour la projection
    Omega = rng.standard_normal(size=(n, l))

    # Sous-espace image
    Y = snapshots @ Omega

    # Itérations de puissance pour accentuer les valeurs singulières dominantes
    for it in range(n_iter):
        print(f"Power iteration {it + 1}/{n_iter}")
        Y = snapshots @ (snapshots.conj().T @ Y)

    # Orthonormalisation
    Q, _ = np.linalg.qr(Y, mode="reduced")
    print(f"Q shape after QR: {Q.shape}")

    # Matrice compressée
    B = Q.conj().T @ snapshots
    print(f"Compressed matrix B shape: {B.shape}")

    # SVD sur la petite matrice
    print("Computing SVD on compressed matrix B...")
    Ub, S, VT = np.linalg.svd(B, full_matrices=False)

    # Choix du rang par critère d'énergie
    total = np.sum(S**2)
    cumulative = np.cumsum(S**2)
    r = np.searchsorted(cumulative, energy * total) + 1
    r = min(r, target_rank)

    print(f"Energy threshold = {energy}")
    print(f"Selected r (approximate) = {r} out of {len(S)}")

    if r == target_rank and cumulative[-1] < energy * total:
        print(
            "Warning: reached max_rank before hitting requested energy. "
            "Consider increasing max_rank."
        )

    # Vecteurs singuliers gauches approchés
    U_approx = Q @ Ub[:, :r]

    # Diagnostic d’orthonormalité
    H = U_approx.conj().T @ U_approx
    ortho_err = np.linalg.norm(H - np.eye(r))
    print(f"Orthonormality error = {ortho_err}")

    # Base réduite
    return U_approx


def compute_reduced_bases_SVD_randomized_rank(
        snapshots,
        ranks=(50, 100, 200, 300, 400, 500),
        max_rank=500,
        n_iter=2,
        random_state=0,
    ):
    """
    SVD randomisée pour construire plusieurs bases réduites.

    snapshots : matrice (n_dofs, n_snapshots)
    ranks     : tailles de bases souhaitées
    max_rank  : rang maximal autorisé
    """

    print(f"Snapshot matrix shape: {snapshots.shape}")
    m, n = snapshots.shape

    # Rang cible brut
    target_rank = min(max_rank, m, n)
    l = min(n, target_rank)

    print(f"Randomized SVD with target_rank={target_rank}")
    print(f"Subspace dimension l={l}")

    rng = np.random.default_rng(random_state)
    Omega = rng.standard_normal(size=(n, l))

    Y = snapshots @ Omega

    for it in range(n_iter):
        print(f"Power iteration {it + 1}/{n_iter}")
        Y = snapshots @ (snapshots.conj().T @ Y)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    print(f"Q shape after QR: {Q.shape}")

    B = Q.conj().T @ snapshots
    print(f"Compressed matrix B shape: {B.shape}")

    print("Computing SVD on compressed matrix B...")
    Ub, S, VT = np.linalg.svd(B, full_matrices=False)

    # Nombre maximal de vecteurs singuliers exploitables
    max_r = min(target_rank, len(S))
    print(f"Max available rank from SVD = {max_r}")

    # Base complète jusqu’à max_r
    U_all = Q @ Ub[:, :max_r]

    # Diagnostic d’orthonormalité sur la base la plus grande
    H = U_all.conj().T @ U_all
    ortho_err = np.linalg.norm(H - np.eye(max_r))
    print(f"Orthonormality error (rank {max_r}) = {ortho_err}")

    # Construire les bases demandées
    bases = {}
    for r in ranks:
        if r > max_r:
            print(
                f"Requested basis size r={r} exceeds available rank {max_r}, "
                "it will be ignored."
            )
            continue
        bases[r] = U_all[:, :r]
        print(f"Returned basis of size r={r}")

    return bases


def main(db_path=MERGED_SNAPSHOTS, out_dir=REDUCED_BASIS / "ML"):
    """
    Frequency-localized reduced bases from the snapshot database (step 3 of pipeline/):
    one basis per zone and per rank in requested_ranks, written to <out_dir>/rank<r>/zone<k>[fmin_fmax].npz
    (reduced operators and basis). The chapter uses rank 50 (ml/rb_fast.py).
    """
    # ============================
    #  Configuration of this run
    # ============================
    base_output_dir = Path(out_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Ranks for which we want reduced operators
    requested_ranks = (50, 100, 200, 300, 400, 500)

    # Randomized SVD parameters
    max_rank = 500
    n_iter = 2
    random_state = 0

    # Minimal overlap for the frequency zones
    overlap = 10.0

    # ============================
    #  Build zones in frequency
    # ============================
    zones = build_frequency_zones(overlap=overlap)

    print("Frequency zones (without overlap):")
    for i, (fmin, fmax) in enumerate(zones, start=1):
        print(f"  Zone {i}: [{fmin:.2f}, {fmax:.2f}] Hz")

    # ============================
    #  Create ProblemSetting
    # ============================
    setting = ProblemSetting()

    # ============================
    #  Loop over zones
    # ============================
    for zone_index, (fmin, fmax) in enumerate(zones, start=1):
        print("\n================================")
        print(f"          ZONE {zone_index}")
        print("================================")

        # Effective frequency range with overlap for snapshot selection
        fmin_eff = fmin - overlap
        fmax_eff = fmax + overlap
        print(f"Using snapshots in [{fmin_eff:.2f}, {fmax_eff:.2f}] Hz")

        # -------------------------------------------
        # Load snapshots for this zone from database
        # -------------------------------------------
        snapshots = load_snapshots(
            db_path,
            fmin=fmin_eff,
            fmax=fmax_eff,
        )
        print("Snapshots loaded for this zone:", snapshots.shape, flush=True)

        # -------------------------------------------
        # Build reduced bases (one per requested rank)
        # using randomized SVD
        # -------------------------------------------
        bases = compute_reduced_bases_SVD_randomized_rank(
            snapshots,
            ranks=requested_ranks,
            max_rank=max_rank,
            n_iter=n_iter,
            random_state=random_state,
        )

        # -------------------------------------------
        # For each rank: project operators and store
        # -------------------------------------------
        for r in sorted(bases.keys()):
            print(f"\n=== Building reduced basis for zone {zone_index} with {r} basis vectors ===")

            V = bases[r]  # reduced basis of size r

            # We only use RBM_Offline here as a convenient container
            # to project global operators Ai and L onto V
            rbm = RBM_Offline(
                setting=setting,
                mus_globs=[],     # not used for projection here
                frequencies=[],   # not used for projection here
                V=V,
                verbose=False,
            )

            # Creating the folders and file name
            rank_dir = base_output_dir / f"rank{r}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            output_path = rank_dir / f"zone{zone_index}[{fmin:.0f}_{fmax:.0f}].npz"

            # Compute and save the reduced operators
            print(f"Computing and saving reduced matrices in {output_path}...")
            rbm.compute_and_save_reduced_matrices(str(output_path))
            print(f"Done for zone {zone_index} rank {r}. Reduced model saved in {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="frequency-localized reduced bases from the snapshot database")
    ap.add_argument("--db", default=str(MERGED_SNAPSHOTS), help="merged snapshot database (step 2)")
    ap.add_argument("--out", default=str(REDUCED_BASIS / "ML"), help="output folder of the bases")
    a = ap.parse_args()
    main(a.db, a.out)
