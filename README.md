# Shape optimization of the Swiss alphorn: finite elements, reduced basis method and machine learning

Code to replicate the results of chapter 6 of my PhD thesis and of an upcoming publication.

**Data.** The database of the simulations (FE snapshots, reduced bases, resonance datasets, trained networks) can be
downloaded from Harvard Dataverse: https://doi.org/10.7910/DVN/C1FTYA (section [Data](#data)).

The input impedance of the alphorn is computed from the Helmholtz equation with the finite element method (FEM), on
a geometry described by the lengths and the radii of four conical pieces between a fixed mouthpiece and a fixed bell.
A frequency-localized reduced basis method (RBM) gives the resonance frequencies of $10^5$ geometries. A forward neural
network (geometry → resonances) and a tandem backward network (resonances → geometry) then search for the geometry
whose resonances are best tuned to the alphorn scale.

**Every computation, figure, table and number of the thesis chapter can be regenerated with this repository**,
from the stored results in a few minutes or from scratch on a cluster (section
[Reproducing the results](#reproducing-the-results)).

**Main result.** With 1000 evaluations of the reduced model, the forward model finds the best geometry of a grid with 26
values per parameter (4 357 080 admissible geometries, of which 4 311 053 have the 12 resonances in the band).
Verified with FEM, this geometry reduces the error $E$ (root mean squared deviation of the ten notes from the alphorn
scale, in cents, after subtracting the offset of the model, the same for all the geometries) from 19.3 cents for the
reference alphorn to 7.2 cents.

## Contents

1. [Quick start](#quick-start)
2. [Requirements](#requirements)
3. [Data](#data)
4. [Reproducing the results](#reproducing-the-results)
5. [Pipeline](#pipeline)
6. [Where every figure, table and number comes from](#where-every-figure-table-and-number-comes-from)
7. [Modelling choices and conventions](#modelling-choices-and-conventions)
8. [Repository layout](#repository-layout)
9. [Known limitations](#known-limitations)
10. [Citation, license, contact](#citation-license-contact)

## Quick start

```bash
git clone https://github.com/ffabioo/Swiss_Alphorn_public.git && cd Swiss_Alphorn_public
python3.9 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt
./reproduce.sh assets
```

This regenerates all the figures and tables of the chapter from the results stored in the repository, in a few
minutes on a laptop:
- the figures go to `results/figures/`;
- the tables go to `results/tables/*.tex`;
- the numbers quoted in the text go to `analysis/output/`, with the printed output of every script in
  `analysis/output/logs/`.

Two additions extend this:
- **The data** (section [Data](#data)): the same command also recomputes the numbers that need the reduced
  bases or the resonance datasets (statistics of the datasets, offset statistics, reduced vs FEM resonances). Without
  them, these files are left as they are in git.
- **The dolfinx container**: `./reproduce.sh saved` also repeats the searches and the FEM verification.

## Requirements

Two software environments are used:

| Environment | Used for | Version |
|---|---|---|
| **dolfinx container** | FEM, snapshots, assembly of the reduced bases, FEM resonances, benchmark | `dolfinx/lab@sha256:e3235bb2355e7974b595cdff3b8f1c91491140b12e1118c88f30f9a245eac1e0` (`Dockerfile`), in complex mode: dolfinx 0.10.0, PETSc 3.23.5 (complex), Python 3.12.3, gmsh 4.14.0 |
| **Python environment** | reduced model (pure numpy, `ml/rb_fast.py`), datasets, neural networks (PyTorch, CPU), analyses, figures, tables | Python 3.9 with `requirements-ml.txt`; exact versions of all packages in `requirements-ml.lock.txt` |

**Container.** With Apptainer (as used for the chapter), build the image at the repository root, or set `SIF` to its
path:
```bash
apptainer build dolfinx_lab.sif docker://dolfinx/lab@sha256:e3235bb2355e7974b595cdff3b8f1c91491140b12e1118c88f30f9a245eac1e0
```
The steps of the pipeline call the container through `apptainer exec` (function `run_fem` in `pipeline/env.sh`).
With Docker only, build the image with `docker build -t alphorn-sim .` and run the same commands in it, from the
repository root, for example:
```bash
docker run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work alphorn-sim bash -c \
    "source /usr/local/bin/dolfinx-complex-mode; python3 analysis/fem_resonances.py white_alphorn --mu 0.2089 0.3269 0.2921 16 28.5 44"
```

**Python.** Create the environment as in the quick start. The pipeline uses `.venv/bin/python` if it exists;
otherwise set `PY` to the interpreter.

**Hardware.**

| Target | Hardware | Time |
|---|---|---|
| `./reproduce.sh assets` | any computer | a few minutes |
| `./reproduce.sh saved` | a workstation, few GB of memory | about one hour on 8 cores |
| `./reproduce.sh all` | a cluster: 72-core nodes for steps 1 and 6, 128 GB of memory for step 3, about 180 GB of disk | about 220 CPU hours |

All times in this README were measured on cluster nodes with 2 × Intel Xeon Platinum 8360Y (72 cores, 512 GB per
node).

## Data

The code reads and writes its large data in `databases/` (folders set in `utils/paths.py`). The subfolders have the
same names and contents as the folders of the published dataset, https://doi.org/10.7910/DVN/C1FTYA:

| Folder | Content | Size | Written by | In git |
|---|---|---|---|---|
| `databases/snapshots/` | FE snapshots, 1225 geometries × 200 frequencies: `solutions_<mu>.sqlite`, one database per geometry | 83 GB | step 1 (about 45 CPU hours) | no |
| `databases/reduced_basis/` | reduced bases (`ML/rank50/`, 11 frequency zones) and FE data of the numpy reduced model (`rb_fast_data.npz`) | 196 MB | step 3 | no |
| `databases/resonance_datasets/` | reduced-model resonances of the $10^5$ sampled geometries and of the 4.4 million grid configurations | 29 MB + 1.3 GB | step 6 | no |
| `databases/neural_networks/` | trained forward and backward networks, hyperparameter study of the backward network | 4 MB | steps 7-9 | yes |

The small inputs and results are in git:

| Folder | Content | Size | Written by |
|---|---|---|---|
| `mesh/` | reference mesh; meshes and centreline data of the three measured alphorns | 12 MB | input |
| `results/data/` | FEM impedance and resonances of the real, reference, optimized and random geometries; results of the searches | 0.4 MB | steps 7, 9, 10, 11 |
| `analysis/output/` | outputs of the analyses: accuracy of the reduced model, statistics, timings, benchmark, logs | 1 MB | steps 0, 4, 5, 12 |

**Downloading the data.** `./reproduce.sh saved` and `retrain` need `databases/reduced_basis/` and
`databases/resonance_datasets/` besides git. From the repository root:
```bash
python3 utils/download_data.py              # 14 files, 1.5 GB, into databases/; checked against the checksums of Dataverse
```
The script uses the public API of Dataverse (no account needed); `python3 utils/download_data.py --list` lists the
files without downloading them. Alternatively, download from the dataset page either the folders `reduced_basis/`
and `resonance_datasets/` into `databases/`, or the single file `data_archive/alphorn_rbm_data.tar.gz` (681 MB), which
holds the same files (its folder `datasets/` is called `resonance_datasets/` here):
```bash
tar -xzf alphorn_rbm_data.tar.gz -C databases
mv databases/datasets databases/resonance_datasets
```

Then check the files, from the repository root:
```bash
sha256sum -c pipeline/data_release.sha256
```

**Snapshot database.** Needed only to rebuild the reduced bases (`./reproduce.sh all`, or steps 2-3). In the dataset,
the 1225 databases of `snapshots/` are packed in 35 tar files (83 GB). Download them into `databases/snapshots/` and
unpack them there (the tar files can be deleted afterwards):
```bash
python3 utils/download_data.py snapshots
cd databases/snapshots && for t in snapshots_*.tar; do tar -xf "$t"; done && cd -
```
Step 2 merges them into `databases/solutions.sqlite` (83 GB, not in the dataset), the only input of step 3, which
rebuilds all the reduced bases:
```bash
bash pipeline/02_merge.sh                                       # -> databases/solutions.sqlite
bash pipeline/03_reduced_basis.sh                               # or: sbatch pipeline/03_reduced_basis.sh
bash pipeline/03_reduced_basis.sh /path/to/solutions.sqlite     # merged database elsewhere
```
- The database can be anywhere: the container is given access to its folder.
- Copy it to a local disk first, since SQLite should not be read over a network share.

## Reproducing the results

`reproduce.sh` runs the steps of `pipeline/` (section [Pipeline](#pipeline)) in order. Each step is a bash
script that can run on any machine or be submitted to SLURM as it is.

| Command | Steps | Needs | Time |
|---|---|---|---|
| `./reproduce.sh assets` | 12 | Python; the data (section [Data](#data)) for the numbers that need the reduced bases or the resonance datasets | minutes |
| `./reproduce.sh saved` | 7 `--search-only`, 9 `--select-only`, 10, 11, 12 | Python, data, container | about 1 hour |
| `./reproduce.sh retrain` | 7, 8, 9, 10, 11, 12 | Python, data, container | about 20 CPU hours |
| `./reproduce.sh all` | 0-13 | Python, container, cluster | about 220 CPU hours |
| `./reproduce.sh benchmark` | 0 | container | minutes |
| `./reproduce.sh p_refinement` | 13 | container, Python | about 1.5 hours |
| `./reproduce.sh step NN [args]` | one step, e.g. `./reproduce.sh step 06 grid` | | |

- **`saved`** repeats the searches with the trained networks of the chapter (`databases/neural_networks/`). It therefore
  reproduces the published numbers exactly.
- **`retrain`** trains the networks again (see [Determinism](#determinism)).
- **`all`** also recomputes the snapshots, the reduced bases and the resonance datasets.
- **`p_refinement`** estimates the discretization error of the FEM resonances (step 13). Its results are not in
  git: run it to obtain them.

**SLURM.** Add `--slurm` after the target, e.g. `./reproduce.sh all --slurm`, to submit each step as a SLURM job that
starts when the previous one has finished. Each job asks for the resources in its `#SBATCH` lines, written for nodes
with 72 cores and 512 GB of memory; adapt them to your cluster. Without `--slurm`, the steps run one after the other
on the current machine.

**Settings.** Optional environment variables, read by `pipeline/env.sh`:

| Variable | Meaning | Default |
|---|---|---|
| `SIF` | the container | `<repo>/dolfinx_lab.sif` |
| `PY` | the Python interpreter | `.venv/bin/python` or `python3` |
| `NCPU` | the number of cores | the SLURM allocation, or all the cores |

### Checking the results

After `assets` or `saved`, the regenerated files are **identical to the files in git**:
- **Tables and figures:** byte by byte (`results/tables/*.tex`, `results/figures/*.pdf`; the PDF files are written
  without dates).
- **Results in `results/data/` and `analysis/output/`:** identical up to the last digits (relative differences around
  1e-13) of some values computed with the reduced model. These depend on the floating-point arithmetic of the linear
  algebra library (processor, number of threads) and do not affect any number of the chapter.
- **Computing times of step 5** (`analysis/output/timing.json`): depend on the machine.

The main numbers:

| Quantity | Value | File |
|---|---|---|
| offset of the model $\phi$ (FEM offset of the reference alphorn) | 86.37 cents | `analysis/output/offset_analysis.json` |
| forward model, test set | MSE on $\log_{10} f$: 3.42e-7; RMSE: 2.33 cents | `results/data/forward_search_summary.json` |
| forward search: first candidate of the network / retained candidate | $E$ = 8.98 / 7.27 cents (reduced model), rank 1 / 13 | same |
| forward geometry $(l_1, l_2, l_3, R_2, R_3, R_4)$ | (0.3666, 0.1907, 0.1203, 17.3, 33.5, 47.9), rank 1 of 4 311 053 in the exhaustive search | `results/data/ml_best_mus.json` |
| backward geometry | (0.1865, 0.2499, 0.2821, 12.91, 25.07, 34.89), δ = 85 cents, $E$ = 14.28 cents (reduced model) | `results/data/backward_summary.json` |
| $E$ with FEM: reference, forward, best training geometry, backward | 19.27, 7.18, 9.80, 14.26 cents | `analysis/output/tables_summary.json` |
| dataset | 98 718 geometries kept, 1 282 discarded | `analysis/output/dataset_stats.json` |
| reduced vs FEM resonances (20 test geometries) | median 0.41, 90 % below 1.11, max 6.60 cents | `analysis/output/rb_vs_fem/summary.json` |

### Determinism

- **Fully deterministic:** the FEM, the reduced model and the exhaustive search of the grid. Steps 1-2, rerun on two
  geometries, give byte-identical snapshots.
- **Resonance datasets:** rebuilt with step 6, `resonances_ML_rank50.tsv` has the same geometries, numbers of
  resonances and valid geometries as the published file; the frequencies agree to 3e-10 Hz (the published file was
  written by an earlier run and differs in the last digits, so the files are not byte-identical).
- **Exact with the saved networks:** `saved` uses the trained networks in git and reproduces the chapter exactly.
- **Reduced bases:** rebuilt from the snapshot database, all the files of `databases/reduced_basis/` (6 ranks × 11 zones)
  are bit-identical to the published ones.
- **Retraining the forward network** (step 7) with the environment of `requirements-ml.lock.txt` gives a bit-identical
  network and therefore the same search results. With other software versions or other hardware, the weights can
  differ slightly, and the search can then retain a different configuration among those with almost the same error
  (the 10 best configurations of the grid lie within 1.1 cents). The exhaustive search (step 10) does not depend on
  the networks.
- **Retraining the backward network** (step 9). With several threads, Intel MKL (the matrix products of PyTorch on
  the CPU) may add the terms of a sum in a different order from one run to the next. The training amplifies these
  differences in the last digit into a different network, because the inverse problem has many equally good solutions.
  - `ml/train_backward.py` therefore sets `MKL_CBWR=COMPATIBLE`, which fixes the order: two retrainings give the
    same network (δ = 80 cents, E = 14.9 cents with the reduced model).
  - The network of the chapter was trained before this setting and cannot be recovered by retraining. Two earlier
    retrainings without it gave E = 13.3 and 23.6 cents, each with a different geometry. The backward geometry of the
    chapter is one training among this spread; the saved network reproduces it exactly (`saved`).
  - The setting is not used for the forward network: it would change it, and the forward retraining already gives
    the published network.

## Pipeline

Steps 1-12 produce, in order, everything the chapter uses; steps 0 and 13 check the FE discretization. Each step reads the results of the
previous ones. CPU times were measured on the nodes described in [Requirements](#requirements).

| Step | Script | What it does | Output | Cost |
|---|---|---|---|---|
| 0 | `00_benchmark.sh` | FE convergence test on a problem with analytical solution, degrees 1-3 (`analysis/benchmark_convergence.py`) | `analysis/output/benchmark/` | 1 min |
| 1 | `01_snapshots.sh` | FE snapshots: 1225 geometries × 200 frequencies, MPI (`utils/compute_hf_solutions.py`) | `databases/snapshots/solutions_*.sqlite` (83 GB) | 72 cores × 36 min |
| 2 | `02_merge.sh` | merge of the snapshot databases (`utils/merge_databases.py`) | `databases/solutions.sqlite` (83 GB) | 8 min |
| 3 | `03_reduced_basis.sh [database] [output]` | reduced bases of the 11 frequency zones, ranks 50-500, randomized SVD with a fixed seed (`rbm_by_zone/db_to_rb_by_zone.py`); FE data of the numpy reduced model (`ml/export_rb_fast_data.py`). Defaults: `databases/solutions.sqlite`, `databases/reduced_basis` | `databases/reduced_basis/` | 32 cores × 25 min, 128 GB of memory (53 GB used) |
| 4 | `04_rb_validation.sh` | reduced vs FE solution: 20 test geometries × 100 frequencies (`analysis/rb_validation.py`) | `analysis/output/rb_validation/` | 20 cores × 1 min |
| 5 | `05_rb_vs_fem.sh` | reduced vs FEM resonances, same 20 geometries (`analysis/fem_resonances.py`, `analysis/rb_vs_fem_resonances.py`); computing times (`analysis/timing.py`) | `analysis/output/rb_vs_fem/`, `analysis/output/timing.json` | 25 cores × 5 min |
| 6 | `06_ml_dataset.sh dataset` / `grid` | reduced-model resonances of the $10^5$ sampled geometries / of the 4.4 million grid configurations (`ml/build_ml_dataset_modes.py`) | `databases/resonance_datasets/resonances_ML_rank50.tsv` / `resonances_grid26_rank50.tsv` | 3.6 / 137 CPU h (72 cores × 3 min / 1 h 54 min) |
| 7 | `07_forward.sh` | forward network: training, test error, search over the grid (1000 candidates re-ranked with the reduced model), best geometry of the training set, exhaustive search (`ml/train_forward.py`; `--search-only`: search with the saved network) | `databases/neural_networks/forward.pt`, `results/data/{ml_best_mus,forward_search_summary,forward_random_predictions}.json` | 8 cores × 14 min (`--search-only`: 4 min) |
| 8 | `08_tune_backward.sh` | hyperparameters of the backward network, 12 configurations (`ml/tune_backward.py`) | `databases/neural_networks/tune_backward/` | 18 CPU h |
| 9 | `09_backward.sh` | backward (tandem) network: training, test error, query with the scale transposed by 0-150 cents (`ml/train_backward.py`; `--select-only`: query with the saved network) | `databases/neural_networks/backward.pt`, `results/data/backward_summary.json` | 4 cores × 25 min (`--select-only`: 1 min) |
| 10 | `10_grid_baseline.sh` | exhaustive reduced-model search of the grid (`analysis/grid_baseline.py`) | `rb_grid` in `results/data/*.json` | 1 min |
| 11 | `11_fem_resonances.sh` | FEM impedance and refined resonances of 11 geometries: real alphorns (own meshes), reference, optimized and random geometries (`analysis/fem_resonances.py`) | `results/data/<name>.csv`, `<name>_resonances.json` | 11 × 4 min |
| 12 | `12_assets.sh` | offset statistics (`analysis/offset_analysis.py`), then all the figures, tables and numbers (`analysis/build_assets.sh`) | `results/figures/`, `results/tables/`, `analysis/output/` | 2 min |
| 13 | `13_p_refinement.sh` | discretization error: the reference alphorn recomputed on the same mesh with Lagrange elements of degree 2 instead of 1 (`analysis/fem_resonances.py --degree 2`), compared with the degree-1 results of step 11 (`analysis/p_refinement.py`) | `results/data/white_alphorn_Q2*`, `analysis/output/p_refinement.json` (not in git) | 2 cores × 1.5 h, 5 GB of memory |

Logs of the SLURM jobs go to `pipeline/logs/`. The names of the FEM results in `results/data/` are explained in
`results/README.md`.

## Where every figure, table and number comes from

The figures and tables of the chapter that show computed results, and the scripts that write them. The photographs,
the drawings of the geometry processing and the diagram of the pipeline are not results and are not part of this
repository.

**Figures**

| Figure | File | Source |
|---|---|---|
| FEM impedance of the three measured alphorns | `results/figures/impedance_all_real_alphorns.pdf` | `analysis/impedance_all_real_alphorns.py` (step 12, data of step 11) |
| forward-model predictions vs FEM for three random geometries | `results/figures/three_random_predictions_vs_impedance.pdf` | `analysis/three_random_predictions_vs_impedance.py` (steps 7, 11, 12) |
| reference alphorn vs forward and backward geometries | `results/figures/impedance_comparison_backward_forward_real.pdf` | `analysis/impedance_comparison_backward_forward_real.py` (steps 7, 9, 11, 12): FEM spectra with the offset of the model removed (frequencies × 2^(−φ/1200)) and, below, the deviations Δf − φ of the notes in cents; `--with-offset` gives the spectra as computed, in `results/figures/impedance_comparison_backward_forward_real_with_offset.pdf` |
| convergence of the FE solver on a benchmark problem (appendix) | `analysis/output/benchmark/error_H10.png`, `error_L2.png` | `analysis/benchmark_convergence.py` (step 0). The figures of the chapter were made with an earlier version of the code; this script solves the same problem and gives the same errors |

The mesh of the reference alphorn is `mesh/reference_domain.msh`.

**Tables**

| Table | Source |
|---|---|
| measured notes and FEM resonances of the three measured alphorns | `analysis/make_tables.py` → `results/tables/real_alphorns_table.tex`. Measured notes in `utils/valais.py`; FEM resonances from step 11; measured deviations marked in green by the authors (`GREEN_MEASURED` in `make_tables.py`) |
| FEM resonances of the reference and optimized geometries | `analysis/make_tables.py` → `results/tables/final_table.tex`. In green: the best value of each note and the lowest error |
| parameters of the optimized geometries | `analysis/make_tables.py` → `results/tables/geometry_table.tex` |
| measurements of the three alphorns (appendix) | measured values; alphorn 3, the reference, in `utils/valais.py` (`D4`, `M4`) |

**Numbers in the text**

| Section of the chapter | Numbers | Source |
|---|---|---|
| Offset between the FEM resonances and the measured notes | FEM offset +86 cents, played notes 12 cents below the scale, FEM above the played notes by 91-123 cents | `analysis/output/tables_summary.json` |
| | 1D model vs FEM within 7 cents | `analysis/output/logs/tmm1d.txt` (`analysis/tmm1d.py`) |
| | offset range +27 to +146 cents | `analysis/output/offset_analysis.json` |
| Construction of the reduced basis | zones and their overlap | `utils/frequencies.py`, `rbm_by_zone/db_to_rb_by_zone.py` |
| | accuracy of the reduced model: median 8.4e-3, ...; resonances 0.4 / 1.1 / 6.6 cents | `analysis/output/rb_validation/summary.json`, `analysis/output/rb_vs_fem/summary.json` |
| | computing times (footnote) | `analysis/output/timing.json` (`analysis/timing.py`, step 5) |
| | dataset: 98 718 kept, 1 282 discarded, shape of the discarded geometries | `analysis/output/dataset_stats.json` |
| Numerical results (mesh) | $\mathbb{Q}_1$ vs $\mathbb{Q}_2$ elements: 21 093 / 159 401 unknowns, offset $\phi$ 86.4 → 83.5 cents, $E$ of the reference alphorn 19.3 → 19.4 cents | `analysis/output/p_refinement.json` (`analysis/p_refinement.py`, step 13; not in git) |
| Forward model | test error; search (first candidate, 13th-ranked optimum); best training geometry | `results/data/forward_search_summary.json` |
| | exhaustive search (same optimum, 137 CPU hours) | `results/data/forward_search_summary.json` (`rb_grid`); CPU hours of step 6 (grid) |
| Backward model | test error, δ = 85 cents, $E$ = 14.3 cents | `results/data/backward_summary.json` |
| Discussion | errors, offsets, deviations of the notes | `analysis/output/tables_summary.json` |
| | geometries and bore profiles | `analysis/output/geometry_comparison.json` |
| Appendix, backward hyperparameters | errors of the 12 configurations | `databases/neural_networks/tune_backward/cfg_*.json`, `analysis/output/logs/tune_backward_summary.txt` |
| Appendix, forward hyperparameters | | not in this repository (see [Known limitations](#known-limitations)) |

Three outputs support the analysis but are not quoted in the chapter: the effect of visco-thermal losses on $E$
(`analysis/output/tmm_shapes.json`), the sensitivity of the optimum to $\phi$ (`other_c` in
`analysis/output/offset_analysis.json`) and the numbers of the averaging argument for the tandem loss
(`analysis/output/average_geometry_demo.json`).

## Modelling choices and conventions

- **Equations.**
  - Helmholtz equation for the acoustic pressure, $c_0 = 340$ m/s, $\rho = 1.225$ kg/m³, time convention
    $e^{-i\omega t}$.
  - Inlet (mouthpiece): $\partial_n p = i\rho\omega u_n$.
  - Outlet (hemisphere of radius $R = 0.447$ m around the bell): outgoing spherical wave, $\partial_n p = (ik - 1/R)\,p$,
    i.e. the term $+\frac{1-ikR}{R}\int_{\Gamma_{out}} p\,v$ in the bilinear form.
  - $Z_{in}$ is the mean pressure on the inlet divided by $u_n$.
- **Geometry.**
  - $\mu = (l_1, l_2, l_3, R_2, R_3, R_4)$, with $l_i = L_i/L$ the relative lengths of the cones and
    $l_4 = 1 - l_1 - l_2 - l_3$.
  - $R_2, R_3, R_4$ are the radii (mm) at the junctions; $R_1 = 6.5$ mm and $R_5 = 51.5$ mm are fixed.
  - Constraints: $l_i \ge l_{\min} = 220\,\mathrm{mm}/L$ for $i = 1, \dots, 4$, and $R_1 \le R_2 \le R_3 \le R_4 \le R_5$.
  - The affine map of the four cones onto the reference domain is in `core/cone_problem.py` and
    `core/all_cones_problem.py`.
- **Discretization.** $\mathbb{Q}_1$ Lagrange finite elements (trilinear on the hexahedra of
  `mesh/reference_domain.msh`, 21 093 unknowns), quadrature degree 4 (`utils/settings.py`). Step 13 compares the
  resonances of the reference alphorn with $\mathbb{Q}_2$ elements on the same mesh.
- **Reduced model.**
  - 11 frequency zones: one per note, plus one at 512 Hz.
  - Each basis (rank 50) is built from the snapshots within ±46.5 Hz of its centre.
  - Each frequency is solved with the basis whose centre is the closest.
- **Resonances.**
  - Peaks of $|Z_{in}|$ on a 1 Hz grid in [30, 615] Hz, with a prominence of $\ln|Z_{in}| \ge 0.2$.
  - Refined locally: with the reduced model, 41 points over ±1 Hz and a parabola; with FEM, golden-section search to
    0.01 Hz.
  - **Numbered from the lowest one.** The notes of the scale (92, 139, 185, 234, 278, 331, 371, 417, 468, 556 Hz) are
    the resonances $n = 2, \dots, 10$ and $12$.
  - Geometries without exactly 12 resonances in the band are discarded.
- **Tuning error.**
  - $E$ is the RMSE, in cents, of the deviations of the ten notes from the scale after subtracting
    $\phi$ = 86.37 cents (`COMMON_OFFSET` in `ml/ml_common.py`). This value is the FEM
    offset of the reference alphorn, taken as the error of the model and the same for all the geometries.
  - The part of the offset of a geometry due to its shape counts in $E$.
  - Implemented in `ml_common.tuning_error`. `ml_common.tuning` also returns the offset and the plain RMSE.
- **Search grid.** 26 values per parameter; 4 357 080 admissible configurations, of which 4 311 053 have the 12
  resonances in the band.

## Repository layout

```text
reproduce.sh       entry point: runs the steps of pipeline/ (section "Reproducing the results")
pipeline/          the steps 00-13 (bash scripts, also SLURM jobs), env.sh (settings), data_release.sha256 (checksums
                   of the downloaded data), logs/
core/              parametric FE solver on the reference domain (parametric_solver.py; affine decomposition of the
                   cones in cone_problem.py, all_cones_problem.py)
utils/             data folders (paths.py), download of the published data (download_data.py), settings (mesh, FE
                   space, boundary ids), frequencies (scale, zones), geometric parameters and grids (mus_utils.py), the
                   real alphorns (valais.py), computation, storage and merging of the snapshots
mesh/              reference mesh; real_alphorns/: meshes and centreline/radius data of the three measured alphorns
rbm_by_zone/       frequency-localized reduced bases: randomized SVD of the snapshots of each zone
                   (db_to_rb_by_zone.py), projection of the affine operators on the bases (rbm_offline_by_zone.py)
ml/                pure-numpy reduced model (rb_fast.py), resonance datasets, forward network, backward network and its
                   hyperparameter study
analysis/          FEM resonances of given geometries (fem_resonances.py), figures, tables, benchmark, accuracy of the
                   reduced model, timings, statistics, 1D model; build_assets.sh; output/ (results of the analyses)
results/           data/ (FEM results and results of the searches), figures/ and tables/ (written by step 12)
databases/         large data (section "Data"); neural_networks/ is in git
Dockerfile         the container
requirements-ml.txt, requirements-ml.lock.txt   the Python environment
```

All the scripts run from the repository root, with the repository root on `PYTHONPATH`; `pipeline/env.sh` sets both.

## Known limitations

- **Offset of the model.**
  - The FEM resonances of all the measured and optimized geometries lie 80-90 cents above the alphorn scale, while the
    notes played on the reference alphorn are close to it. The cause is not known; the bell, reconstructed from a
    photograph of the mould, is the least certain part of the geometry.
  - The FEM offset of the reference alphorn is therefore subtracted from all the geometries. This assumes that the
    error of the model does not depend on the shape. `analysis/tmm_shapes.py` supports this for the visco-thermal
    losses; it cannot be checked for the unknown main cause.
  - The optimum depends on the value subtracted: the forward geometry ranks 1st, 2nd and 3rd in the grid for 86.37, 90
    and 91.37 cents, but 67th for 80 cents and 1437th for 100 cents (`analysis/output/offset_analysis.json`,
    `other_c`).
- **Forward network.** Its hyperparameters were selected in the first version of the study (4-fold cross-validation
  on an earlier dataset) and were not selected again on the present dataset.
- **Not repeated after the correction of the outlet condition:**
  - the comparison with a single global reduced basis (its code is not part of this repository);
  - the mesh-convergence study (refinement of the mesh). The discretization error is estimated instead by
    p-refinement of the reference alphorn (step 13).
  - the benchmark figure of the appendix. Its test problem does not involve the outlet condition, and
    `analysis/benchmark_convergence.py` gives the same errors.

Changes with respect to the first version of the code:
- **Outlet boundary condition:** the sign of the Robin coefficient now matches the time convention.
- **Constraint** $l_4 \ge l_{\min}$: enforced everywhere.
- **Resonances:** numbered from the lowest one, instead of the highest peak in a frequency window around each note.
- **Forward search:** re-ranked with the reduced model.
- **Backward network:** trained as a tandem network.
- **Ranking:** the geometries are compared after subtracting the common offset of the model.

All the snapshots, bases, resonance datasets and results were recomputed after these changes.

## Citation, license, contact

- **Citation:** if you use this code or its data, please cite the thesis and the dataset
  (https://doi.org/10.7910/DVN/C1FTYA).
- **License:** TODO.
- **Contact:** TODO.
