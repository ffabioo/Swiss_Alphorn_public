# Results

Written by the pipeline (see the main README); the scripts are in `analysis/` and run from the repository root.

- `data/`: FEM impedance and resonances of given geometries, and the results of the searches.
  - `analysis/fem_resonances.py` (dolfinx container): FEM |Z_in| of one geometry on a 1 Hz grid in [30, 615] Hz and
    its resonances, numbered from the lowest one and refined (golden-section search, 0.01 Hz)
    -> `data/<name>.csv`, `data/<name>_resonances.json`.
    - parametric geometry: `python analysis/fem_resonances.py NAME --mu l1 l2 l3 R2 R3 R4`
    - own mesh (real alphorns): `python analysis/fem_resonances.py NAME --mesh mesh/real_alphorns/mesh_mesh_data20252026.<k>.msh`

    All the runs of the chapter are in `pipeline/11_fem_resonances.sh`.
  - `ml_best_mus.json`, `forward_search_summary.json`, `forward_random_predictions.json`, `backward_summary.json`:
    written by `ml/train_forward.py`, `ml/train_backward.py` and `analysis/grid_baseline.py` (steps 7, 9, 10).
- `figures/` (plain Python, run by `analysis/build_assets.sh`):
  - `analysis/impedance_all_real_alphorns.py`: the three real alphorns (`data/impedance_Alphorn_{1,3,4}*`)
  - `analysis/three_random_predictions_vs_impedance.py`: forward-model predictions vs FEM
    (`data/valais_random_{0,1,2}*`, `data/forward_random_predictions.json`)
  - `analysis/impedance_comparison_backward_forward_real.py`: reference alphorn and optimized geometries
    (`data/{white_alphorn,forward,backward}*`)
- `tables/`: LaTeX tables written by `analysis/make_tables.py`.

Names in `data/`: `impedance_Alphorn_1/3/4` = first/second/third real alphorn (own meshes); `white_alphorn` = third
alphorn on the parametric reference domain (reference instrument); `forward`, `backward`, `lookup` (best geometry of
the training data), `rb_grid` (exhaustive grid search, same geometry as `forward`): optimized geometries, with their
parameters in `ml_best_mus.json`; `valais_random_0/1/2`: random geometries of the forward-model figure.
