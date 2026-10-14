"""
Forward model: geometry mu -> frequencies of the resonances n = 2..10, 12 (the notes of the alphorn scale),
and search for the best geometry.

As in the chapter: MLP with 5 hidden layers of 256 neurons, ReLU, Adam (lr 1e-3), mini-batches of 256,
l2 regularization 1e-5, 200 epochs; standardized inputs and standardized log10 frequencies as targets.

Search: the model is evaluated on the admissible grid (l_i >= l_min for i = 1..4, num_l = num_R = 26); the
objective is the RMSE of the deviations in cents after subtracting the offset of the model, the same for all the
geometries (ml_common.tuning_error). The K = 1000 best predicted geometries are re-evaluated with the reduced model
(ml/rb_fast.py) and the best one by the reduced model is retained. Two baselines are reported: the best geometry of
the training data (lookup) and, if databases/resonance_datasets/resonances_grid26_rank50.tsv exists, the exhaustive search of the grid
with the reduced model.

Run from the repo root:  python ml/train_forward.py                 (training + search)
                         python ml/train_forward.py --search-only   (search with the saved databases/neural_networks/forward.pt)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import os
import sys
import pandas as pd

from ml.ml_common import (load_dataset, split_indices, Standardizer, cents, tuning, tuning_error, F_COLS, MU_COLS,
                          L_MIN_REL, MODELS_DIR, GRID_DATASET, R_MIN, R_MAX, OFFICIAL_NOTES_FREQUENCIES)
from ml.rb_fast import FastRB, NOTE_MODES
from utils.mus_utils import mus_glob_grid, figure_random_mus

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
TOP_K = 1000
HIDDEN, DEPTH, LR, BATCH, WEIGHT_DECAY, EPOCHS = 256, 5, 1e-3, 256, 1e-5, 200


def make_mlp(n_in, n_out, hidden, depth, dropout=0.0):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def train(model, X, Y, tr, va):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best, best_state = np.inf, None
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), BATCH):
            i = torch.as_tensor(tr)[perm[b:b + BATCH]]
            opt.zero_grad()
            loss = loss_fn(model(X[i]), Y[i])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X[va]), Y[va]).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  val MSE (standardized log10 f) = {val:.3e}  [{time.time() - t0:.0f} s]", flush=True)
    model.load_state_dict(best_state)


def main(search_only=False):
    mu, f = load_dataset()
    tr, va, te = split_indices(len(mu))
    sx, sy = Standardizer(mu[tr]), Standardizer(np.log10(f[tr]))
    model = make_mlp(6, 10, HIDDEN, DEPTH)
    if search_only:   # saved network with its own standardizers
        ck = torch.load(MODELS_DIR / "forward.pt", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        sx, sy = Standardizer.from_state(ck["sx"]), Standardizer.from_state(ck["sy"])
    else:
        train(model, torch.tensor(sx(mu), dtype=torch.float32), torch.tensor(sy(np.log10(f)), dtype=torch.float32), tr, va)
    model.eval()

    def predict(mu_np):
        with torch.no_grad():
            z = model(torch.tensor(sx(mu_np), dtype=torch.float32)).numpy().astype(np.float64)
        return 10.0 ** sy.inverse(z)

    f_te = predict(mu[te])
    mse_log10 = np.mean((np.log10(f_te) - np.log10(f[te])) ** 2)
    mse_std = np.mean((sy(np.log10(f_te)) - sy(np.log10(f[te]))) ** 2)
    rmse_c = np.sqrt(np.mean(cents(f_te, f[te]) ** 2))
    per_note = np.sqrt(np.mean(cents(f_te, f[te]) ** 2, axis=0))
    print(f"TEST: MSE on log10(f) = {mse_log10:.3e}, MSE on standardized log10(f) = {mse_std:.3e}, RMSE = {rmse_c:.2f} cents")
    print("TEST RMSE per note (cents):", np.round(per_note, 1))

    if not search_only:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "sx": sx.state(), "sy": sy.state()}, MODELS_DIR / "forward.pt")

    # ---- search over the admissible grid ----
    grid = mus_glob_grid(l_mins=(L_MIN_REL,) * 3, R_mins=(R_MIN,) * 3, R_max=R_MAX,
                         num_l=26, num_R=26, l_min4=L_MIN_REL)
    pred = np.empty((len(grid), 3))   # predicted (error, offset, RMSE) of each configuration
    for b in range(0, len(grid), 200_000):
        pred[b:b + 200_000] = np.column_stack(tuning(predict(grid[b:b + 200_000])))
    top = np.argsort(pred[:, 0])[:TOP_K]
    rb = FastRB()
    def rb_tuning(m):
        r = rb.resonances(m, fmin=30.0, fmax=615.0)
        if len(r) != 12:
            return np.nan, np.nan, np.nan
        return tuning(r[NOTE_MODES - 1])
    rb_top = np.array([rb_tuning(grid[i]) for i in top])
    j = int(np.nanargmin(rb_top[:, 0]))
    mu_best = grid[top[j]]

    def entry(i, t):
        return {"mu": grid[i].tolist(), "err_pred": float(pred[i, 0]), "offset_pred": float(pred[i, 1]),
                "err_rb": float(t[0]), "offset_rb": float(t[1]), "rmse_rb": float(t[2])}
    summary = {
        "grid_size": int(len(grid)), "top_k": TOP_K,
        "test_mse_log10": mse_log10, "test_mse_standardized": mse_std, "test_rmse_cents": rmse_c,
        "test_rmse_per_note_cents": per_note.tolist(),
        "top1_predicted": entry(top[0], rb_top[0]),
        "forward": {**entry(top[j], rb_top[j]), "rank_in_prediction": j + 1},
    }
    print(f"Search on {len(grid):,} geometries: best predicted error = {pred[top[0], 0]:.2f} cents "
          f"(reduced model: {rb_top[0, 0]:.2f}); after re-ranking the {TOP_K} best with the reduced model: "
          f"{rb_top[j, 0]:.2f} cents (prediction {pred[top[j], 0]:.2f}, rank {j + 1})")
    print("mu_best_forward =", np.array2string(mu_best, precision=4), " l4 =", round(1 - mu_best[:3].sum(), 4))

    # ---- baseline 1: best geometry of the training data ----
    err_data = tuning_error(f[tr])
    k = int(np.argmin(err_data))
    t = tuning(f[tr][k])
    summary["lookup"] = {"mu": mu[tr][k].tolist(), "err_rb": float(t[0]), "offset_rb": float(t[1]), "rmse_rb": float(t[2])}
    print(f"Lookup baseline (best of {len(tr)} training geometries): error {err_data[k]:.2f} cents, mu = {np.round(mu[tr][k], 4)}")

    # ---- baseline 2: exhaustive search of the grid with the reduced model ----
    if GRID_DATASET.exists():
        G = pd.read_csv(GRID_DATASET, sep="\t")
        G = G[G["valid"] == 1]
        err_g, off_g, rmse_g = tuning(G[F_COLS].to_numpy())
        g = int(np.argmin(err_g))
        mu_g = G[MU_COLS].to_numpy()[g]
        rank_fwd = int(np.sum(err_g < summary["forward"]["err_rb"] - 1e-9)) + 1
        summary["rb_grid"] = {"mu": mu_g.tolist(), "err_rb": float(err_g[g]), "offset_rb": float(off_g[g]),
                              "rmse_rb": float(rmse_g[g]), "n_valid": int(len(G)), "rank_of_forward": rank_fwd}
        print(f"Exhaustive reduced-model search ({len(G):,} valid geometries): error {err_g[g]:.2f} cents, "
              f"mu = {np.round(mu_g, 4)}; the forward-model geometry ranks {rank_fwd}")

    # ---- predictions for the three random geometries of the figure ----
    rnd = figure_random_mus(L_MIN_REL, R_MIN, R_MAX)

    out = Path("results/data")
    best_file = out / "ml_best_mus.json"
    best = json.loads(best_file.read_text()) if best_file.exists() else {}
    best["forward"] = mu_best.tolist()
    best["lookup"] = summary["lookup"]["mu"]
    if "rb_grid" in summary:
        best["rb_grid"] = summary["rb_grid"]["mu"]
    best_file.write_text(json.dumps(best, indent=2))
    (out / "forward_search_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "forward_random_predictions.json").write_text(json.dumps(
        {"X_RANDOM": rnd.tolist(), "Y_PRED": predict(rnd).tolist(), "modes": NOTE_MODES.tolist(),
         "test_mse_log10": mse_log10, "test_mse_standardized": mse_std, "test_rmse_cents": rmse_c}, indent=2))
    print("saved", best_file, out / "forward_search_summary.json", out / "forward_random_predictions.json")


if __name__ == "__main__":
    main(search_only="--search-only" in sys.argv[1:])
