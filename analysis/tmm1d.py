"""
1D transmission-line model (stepped cylinders along the centreline) of the third alphorn, independent of the FEM code:
lossless vs visco-thermal losses (Keefe), flanged-piston radiation at the bell. The resonances are compared with the
refined 3D FEM ones (results/data/impedance_Alphorn_4_resonances.json): agreement within about 7 cents, and losses lower the resonances by
only about 6-11 cents, so neither the discretization nor the neglected losses explain the offset of the FEM resonances.

Run from the repo root (plain Python):  python analysis/tmm1d.py
"""
import numpy as np
from scipy.signal import find_peaks
D = np.loadtxt("mesh/real_alphorns/mesh_data20252026.4.csv", delimiter=",", skiprows=1)
X, r = D[:, :3] * 1e-3, D[:, 3] * 1e-3
ds = np.linalg.norm(np.diff(X, axis=0), axis=1); s = np.r_[0, np.cumsum(ds)]
rm = 0.5 * (r[1:] + r[:-1])
rho, mu, gam, Pr = 1.225, 1.81e-5, 1.4, 0.71
print(f"centreline length {s[-1]:.3f} m (inlet r = {r[0]*1e3:.1f} mm, min r = {r.min()*1e3:.2f} mm at s = {s[np.argmin(r)]*1e3:.0f} mm, bell r = {r[-1]*1e3:.1f} mm)")

def zin(f, c=340.0, losses=False):
    """|Z_in| for an array of frequencies f (loop over segments, vectorised over f)."""
    w = 2 * np.pi * f; k0 = w / c
    a = r[-1]; Sb = np.pi * a**2; ka = k0 * a
    Z = (rho * c / Sb) * (0.5 * ka**2 + 1j * 0.8216 * ka) / (1 + 0.5 * ka**2 + 1j * 0.8216 * ka)   # flanged piston (approx.)
    for i in range(len(rm) - 1, -1, -1):
        S = np.pi * rm[i]**2
        if losses:
            rv = rm[i] * np.sqrt(w * rho / mu); e = (1 + (gam - 1) / np.sqrt(Pr)) / (np.sqrt(2) * rv)
            k = k0 * (1 + e) - 1j * k0 * e; Zc = rho * c / S * (1 + e - 1j * e)
        else:
            k, Zc = k0, rho * c / S
        cl, sl = np.cos(k * ds[i]), np.sin(k * ds[i])
        Z = (Z * cl + 1j * Zc * sl) / (1j * Z / Zc * sl + cl)
    return np.abs(Z)

f = np.arange(40, 620, 0.05)
def peaks(**kw):
    z = zin(f, **kw); i, _ = find_peaks(z); return f[i]
p0, p1 = peaks(), peaks(losses=True)
# refined FEM resonances (analysis/fem_resonances.py), numbered from the lowest one
import json
pf = np.array(json.load(open("results/data/impedance_Alphorn_4_resonances.json"))["resonances"])
n = min(len(p0), len(pf), 12)
c = lambda a, b: 1200 * np.log2(a / b)
print("mode n          :", np.arange(1, n + 1))
print("3D FEM (Hz)     :", pf[:n].round(1))
print("1D lossless (Hz):", p0[:n].round(1), "  vs FEM (cents):", c(p0[:n], pf[:n]).round(0))
print("1D lossy (Hz)   :", p1[:n].round(1), "  shift due to visco-thermal losses (cents):", c(p1[:n], p0[:n]).round(1))
notes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12]) - 1
print(f"notes (n = 2..10, 12): |1D lossless - FEM| <= {np.abs(c(p0[notes], pf[notes])).max():.1f} cents; "
      f"losses shift {c(p1[notes], p0[notes]).min():.1f} to {c(p1[notes], p0[notes]).max():.1f} cents")
