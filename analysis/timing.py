"""
Computing times quoted in the chapter (footnote of the section on the construction of the reduced basis), on one core:
  - reduced model (ml/rb_fast.py): solution of the reduced system and Z_in, per frequency (frequencies evaluated in
    batches, as in the resonance search), and the refined resonances of one geometry (grid of 1 Hz in [30, 615] Hz +
    local refinement), median over 20 geometries;
  - full FE solve (analysis/fem_resonances.py, parametric reference domain): assembly of the system for one frequency,
    direct solve (MUMPS) and Z_in, median over 20 frequencies.
The times depend on the machine (chapter: one core of an Intel Xeon Platinum 8360Y).

  python analysis/timing.py rb     # plain Python  -> analysis/output/timing.json ("rb")
  python3 analysis/timing.py fem   # dolfinx container, complex mode -> analysis/output/timing.json ("fem")
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from utils.valais import mu_glob_h4 as MU_REF   # reference alphorn

OUT = Path("analysis/output/timing.json")


def save(key, value):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[key] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))


def time_rb():
    from ml.rb_fast import FastRB
    from ml.build_ml_dataset_modes import dataset_mus
    rb = FastRB()
    ops = rb.operators(MU_REF)
    grid = np.arange(30.0, 615.5, 1.0)   # the 1 Hz grid of the resonance search, evaluated in one call as in rb.resonances
    rb.impedance(MU_REF, grid, ops)      # warm-up
    t = []
    for _ in range(20):
        t0 = time.perf_counter(); rb.impedance(MU_REF, grid, ops); t.append((time.perf_counter() - t0) / len(grid))
    per_freq = min(t)
    mus = dataset_mus()[:20]
    rb.resonances(mus[0])
    tg = []
    for m in mus:
        t0 = time.perf_counter(); rb.resonances(m, fmin=30.0, fmax=615.0); tg.append(time.perf_counter() - t0)
    per_geom = float(np.median(tg))
    save("rb", {"seconds_per_frequency": per_freq, "seconds_per_geometry_resonances": per_geom,
                "method": "per frequency: best of 20 calls on the 1 Hz grid (586 frequencies, operators of one geometry); "
                          "per geometry: median over 20 geometries of rb.resonances (operators, grid and refinement)",
                "threads": os.environ.get("OMP_NUM_THREADS")})
    print(f"reduced model: {1e3 * per_freq:.3f} ms per frequency, {per_geom:.3f} s per geometry (refined resonances)")


def time_fem():
    from analysis.fem_resonances import FemImpedance
    Zf = FemImpedance(mu=MU_REF)
    Zf.zabs(100.0)   # warm-up (solver set-up)
    t = []
    for f in np.linspace(40.0, 600.0, 20):
        t0 = time.perf_counter(); Zf.zabs(f); t.append(time.perf_counter() - t0)
    per_freq = float(np.median(t))
    save("fem", {"seconds_per_frequency": per_freq, "method": "median over 20 frequencies of FemImpedance.zabs "
                 "(assembly of the system, direct solve with MUMPS, Z_in)", "threads": os.environ.get("OMP_NUM_THREADS")})
    print(f"FEM: {per_freq:.3f} s per frequency (assembly + direct solve + Z_in)")


if __name__ == "__main__":
    {"rb": time_rb, "fem": time_fem}[sys.argv[1]]()
