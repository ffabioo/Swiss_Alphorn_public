"""
Pure numpy evaluation of the frequency-zoned reduced model (same operators and coefficients as
analysis/rb_validation.py:solve_reduced), batched over frequencies, with Z_in computed from a reduced output vector
(int_{Gamma_in} p = ell . V u_N) instead of lifting the solution to the full mesh.

Needs databases/reduced_basis/rb_fast_data.npz (written by ml/export_rb_fast_data.py) and the zone files
databases/reduced_basis/<experiment>/rank<N>/zone*.npz (written by rbm_by_zone/db_to_rb_by_zone.py).

Resonances are identified by their number n (1st, 2nd, ... peak of |Z_in|), not by frequency windows,
and each peak is refined with local reduced solves.
"""
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

from utils.mus_utils import to_local
from utils.frequencies import FREQUENCY_ZONES_CENTERS
from utils.paths import REDUCED_BASIS

# Harmonic numbers of the notes of the alphorn scale (92, 139, 185, 234, 278, 331, 371, 417, 468, 556 Hz)
NOTE_MODES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])


def cone_coefficients(mu_local, muh):
    """omegas (7) and thetas (3) of one cone, as in core/cone_problem.py."""
    x1h, x2h, r1h, r2h = muh
    x1, x2, r1, r2 = mu_local
    a = (x2 - x1) / (x2h - x1h)
    alpha = (r2 - r1) / (x2h - x1h)
    alphah = (r2h - r1h) / (x2h - x1h)
    omegas = [a, r1**2 / a, r1 * alpha / a, alpha**2 / a,
              -(alpha * r1h * r1 - alphah * r1**2) / a,
              -(alpha**2 * r1h - alpha * alphah * r1) / a,
              (alpha * r1h - alphah * r1)**2 / a]
    thetas = [a * r1**2, a * alpha * r1, a * alpha**2]
    return np.array(omegas), np.array(thetas)


class FastRB:
    def __init__(self, experiment="ML", rank=50, base=REDUCED_BASIS):
        base = Path(base)
        d = np.load(base / "rb_fast_data.npz")
        self.ell, self.area, self.muh = d["ell"], float(d["inlet_area"]), d["cone_muh"]
        self.c, self.R, self.rho0, self.u_n = float(d["c"]), float(d["R"]), float(d["rho0"]), float(d["u_n"])
        self.x1, self.L = self.muh[0, 0], self.muh[-1, 1] - self.muh[0, 0]
        self.R1, self.R5 = self.muh[0, 2], self.muh[-1, 3]
        self.zones = []
        for zone_id in range(1, len(FREQUENCY_ZONES_CENTERS) + 1):
            f = sorted((base / experiment / f"rank{rank}").glob(f"zone{zone_id}[[]*.npz"))
            if len(f) != 1:
                raise FileNotFoundError(f"zone {zone_id}: expected one file, found {f}")
            z = np.load(f[0], allow_pickle=True)
            A = z["Ai_reduced"]
            self.zones.append({"A": A, "L": z["L_reduced"], "ell": self.ell @ z["V"]})

    def operators(self, mu_glob):
        """Per zone: K(mu), M(mu), B such that A_N(mu, f) = K - k^2 M + (1 - i k R)/R B."""
        locs = to_local(x1=self.x1, R1=self.R1, R5=self.R5, L=self.L, mu_glob=np.asarray(mu_glob, dtype=float))
        om, th = zip(*(cone_coefficients(m, mh) for m, mh in zip(locs, self.muh)))
        om, th = np.concatenate(om), np.concatenate(th)
        ops = []
        for z in self.zones:
            A = z["A"]
            K = A[0] + np.tensordot(om, A[1:29], axes=1)
            M = A[29] + np.tensordot(th, A[30:42], axes=1)
            ops.append((K, M, A[42]))
        return ops

    def impedance(self, mu_glob, freqs, ops=None):
        """Complex Z_in(f) for an array of frequencies (zone = the one whose centre is closest to f)."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        ops = self.operators(mu_glob) if ops is None else ops
        zone = np.argmin(np.abs(FREQUENCY_ZONES_CENTERS[None, :] - freqs[:, None]), axis=1)
        Z = np.empty(len(freqs), dtype=complex)
        for iz in np.unique(zone):
            sel = zone == iz
            f = freqs[sel]; w = 2 * np.pi * f; k = w / self.c
            K, M, B = ops[iz]; z = self.zones[iz]
            A = K[None] - (k**2)[:, None, None] * M[None] + ((1 - 1j * k * self.R) / self.R)[:, None, None] * B[None]
            F = 1j * self.rho0 * w[:, None] * z["L"][None, :]
            u = np.linalg.solve(A, F[..., None])[..., 0]
            Z[sel] = (u @ z["ell"]) / (self.u_n * self.area)
        return Z

    def resonances(self, mu_glob, fmin=30.0, fmax=615.0, df=1.0, prominence=0.2, n_fine=41):
        """
        Resonance frequencies (ascending) = peaks of |Z_in|, found on a grid of step df and refined on a
        local grid of n_fine points over +-df followed by a parabola on log|Z| through the 3 best points.
        Only peaks with a prominence of log|Z| >= `prominence` are kept (removes zone-switch artefacts).
        """
        ops = self.operators(mu_glob)
        f = np.arange(fmin, fmax + 0.5 * df, df)
        y = np.log(np.abs(self.impedance(mu_glob, f, ops)))
        idx, _ = find_peaks(y, prominence=prominence)
        out = []
        for i in idx:
            ff = np.linspace(f[i] - df, f[i] + df, n_fine)
            yy = np.log(np.abs(self.impedance(mu_glob, ff, ops)))
            j = int(np.clip(np.argmax(yy), 1, n_fine - 2))
            y0, y1, y2 = yy[j - 1:j + 2]; h = ff[1] - ff[0]
            den = y0 - 2 * y1 + y2
            out.append(ff[j] + (0.5 * h * (y0 - y2) / den if den < 0 else 0.0))
        return np.array(out)
