"""
Is the offset of the model the same for all the geometries? 1D transmission-line model (as in analysis/tmm1d.py) of the
parametric geometries: mouthpiece and bell of the third alphorn (mesh/real_alphorns/mesh_data20252026.4.csv), conical
part replaced by the four cones of each geometry (R1 = 6.5 mm -> R5 = 51.5 mm over the length L).
For the reference, forward, best-training and backward geometries (and two extreme shapes) it reports
  - the offset of the lossless resonances (to compare with the FEM offsets of analysis/output/tables_summary.json);
  - the shift of the resonances due to the visco-thermal losses (neglected by the FEM), mean and range over the notes;
  - the error E of the FEM resonances (RMSE of the deviations after subtracting the offset of the model c) without and
    with these shifts added note by note (with the losses, c becomes the offset of the reference alphorn with losses).
A shift of the speed of sound (temperature) changes all the resonances of all the geometries by the same number of cents.
Output: analysis/output/tmm_shapes.json

Run from the repo root (plain Python, a few minutes):  python analysis/tmm_shapes.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from utils.valais import L, mu_glob_h4, R_MIN, R_MAX
from ml.ml_common import COMMON_OFFSET

OUT = Path("analysis/output/tmm_shapes.json")
NOTES = np.array([92, 139, 185, 234, 278, 331, 371, 417, 468, 556.])
MODES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
rho, mu_air, gam, Pr = 1.225, 1.81e-5, 1.4, 0.71

D = np.loadtxt("mesh/real_alphorns/mesh_data20252026.4.csv", delimiter=",", skiprows=1)
s0 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(D[:, :3], axis=0), axis=1))]
r0 = D[:, 3]
i0 = np.where((s0 > 80) & (r0 >= R_MIN))[0][0]   # end of the mouthpiece: the radius reaches R1 again after the throat
a = s0[i0]                                       # beginning of the conical part (mm)
rest = s0 > a + L                                # bell (and the short cylinder before it)


def profile(mu):
    """Centreline abscissa and radius (m) of the parametric geometry mu = (l1, l2, l3, R2, R3, R4)."""
    mu = np.asarray(mu, float)
    joints = a + np.r_[0, np.cumsum(np.r_[mu[:3], 1 - mu[:3].sum()] * L)]
    sc = np.linspace(a, a + L, 2001)[1:]
    s = np.r_[s0[:i0 + 1], sc, s0[rest] - s0[rest][0] + a + L + 1.0]
    r = np.r_[r0[:i0 + 1], np.interp(sc, joints, np.r_[R_MIN, mu[3:], R_MAX]), r0[rest]]
    return s * 1e-3, r * 1e-3


def zin(f, s, r, c=340.0, losses=False):
    """|Z_in| (stepped cylinders, flanged-piston radiation at the bell; visco-thermal losses after Keefe)."""
    ds, rm = np.diff(s), 0.5 * (r[1:] + r[:-1])
    w = 2 * np.pi * f; k0 = w / c
    ka = k0 * r[-1]
    Z = (rho * c / (np.pi * r[-1] ** 2)) * (0.5 * ka**2 + 1j * 0.8216 * ka) / (1 + 0.5 * ka**2 + 1j * 0.8216 * ka)
    for i in range(len(rm) - 1, -1, -1):
        S = np.pi * rm[i] ** 2
        if losses:
            e = (1 + (gam - 1) / np.sqrt(Pr)) / (np.sqrt(2) * rm[i] * np.sqrt(w * rho / mu_air))
            k, Zc = k0 * (1 + e) - 1j * k0 * e, rho * c / S * (1 + e - 1j * e)
        else:
            k, Zc = k0, rho * c / S
        cl, sl = np.cos(k * ds[i]), np.sin(k * ds[i])
        Z = (Z * cl + 1j * Zc * sl) / (1j * Z / Zc * sl + cl)
    return np.abs(Z)


F, DF = np.arange(20, 620, 0.05), 0.05   # from 20 Hz: the first resonance of the extreme shapes is close to 30 Hz


def notes(mu, losses):
    """Resonances n = 2..10, 12 (peaks of |Z_in| refined with a parabola through log|Z|)."""
    s, r = profile(mu)
    lz = np.log(zin(F, s, r, losses=losses))
    i, _ = find_peaks(lz)
    assert len(i) >= 12 and F[i[0]] > F[0] + 1, f"{len(i)} peaks, the first at {F[i[0]]:.1f} Hz"   # resonances 1..12 in the band
    y0, y1, y2 = lz[i - 1], lz[i], lz[i + 1]
    return (F[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * DF)[MODES - 1]


best = json.loads(Path("results/data/ml_best_mus.json").read_text())
GEOMS = {"reference": list(mu_glob_h4), "forward": best["forward"], "lookup": best["lookup"], "backward": best["backward"],
         "narrow bore": [0.25, 0.25, 0.25, 7.0, 9.0, 12.0], "wide bore": [0.1, 0.1, 0.2, 40.0, 48.0, 51.0]}
FEM_KEY = {"reference": "white_alphorn", "forward": "forward", "lookup": "lookup", "backward": "backward"}
fem = json.loads(Path("analysis/output/tables_summary.json").read_text())["final"]

res = {}
for name, mu in GEOMS.items():
    p0, p1 = notes(mu, False), notes(mu, True)
    shift = 1200 * np.log2(p1 / p0)
    res[name] = {"mu": [float(x) for x in mu], "tmm_offset": float(1200 * np.log2(p0 / NOTES).mean()),
                 "loss_shift": shift.tolist(), "loss_shift_mean": float(shift.mean()),
                 "fem_offset": fem[FEM_KEY[name]]["offset"] if name in FEM_KEY else None}
ref = np.array(res["reference"]["loss_shift"])
c_lossy = COMMON_OFFSET + ref.mean()
for name in FEM_KEY:
    dev = np.array(fem[FEM_KEY[name]]["dev"])
    res[name]["E_fem"] = float(np.sqrt(np.mean((dev - COMMON_OFFSET) ** 2)))
    res[name]["E_fem_with_losses"] = float(np.sqrt(np.mean((dev + np.array(res[name]["loss_shift"]) - c_lossy) ** 2)))
    print(f"{name:12s}: E = {res[name]['E_fem']:.2f} cents, with the losses {res[name]['E_fem_with_losses']:.2f} cents")
for name, v in res.items():
    dd = np.array(v["loss_shift"]) - ref
    v["loss_shift_minus_reference"] = {"mean": float(dd.mean()), "min": float(dd.min()), "max": float(dd.max())}
    fo = f", FEM offset {v['fem_offset']:+6.1f}" if v["fem_offset"] is not None else ""
    print(f"{name:12s}: 1D offset {v['tmm_offset']:+6.1f}{fo}; losses shift the resonances by {v['loss_shift_mean']:+5.1f} cents "
          f"(notes {min(v['loss_shift']):+5.1f} .. {max(v['loss_shift']):+5.1f}; minus reference: mean {dd.mean():+5.2f}, "
          f"notes {dd.min():+5.2f} .. {dd.max():+5.2f})")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))
