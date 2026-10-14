"""
Backward model: frequencies of the resonances n = 2..10, 12 -> geometry mu, with the constraints built into
the output layer, trained as a tandem network.

Architecture: MLP with ReLU (optional dropout after each hidden layer, none in the selected model), Adam, mini-batches of 256, l2 regularization;
inputs log10(f) standardized on the training set. The hyperparameters (HIDDEN, DEPTH, DROPOUT, LR, WEIGHT_DECAY)
are selected with ml/tune_backward.py. Output layer:
  - lengths: l = l_min + (1 - 4 l_min) * softmax(z_l)  -> l_i >= l_min (i = 1..4) and sum = 1; l4 is discarded
  - radii:   gaps = (R5 - R1) * softmax(z_R),  R_{i+1} = R1 + cumsum(gaps)  -> R1 < R2 < R3 < R4 < R5
Tandem training: since several geometries give almost the same frequencies, the loss is not computed on mu
(which would average different geometries) but on the frequencies: the predicted geometry is passed through
the trained forward model (ml/train_forward.py, frozen) and compared with the input frequencies.

Query: the alphorn scale transposed by delta cents, delta = 0, 5, ..., 150; each predicted geometry is evaluated
with the reduced model and the one with the lowest error (RMSE after subtracting the offset of the model,
ml_common.tuning_error) is retained.

Run from the repo root (after ml/train_forward.py):  python ml/train_backward.py                 (training + query)
                                                     python ml/train_backward.py --select-only   (query with the saved databases/neural_networks/backward.pt)
"""
import json
import os
import sys
import time
from pathlib import Path

# Reproducible training. With several threads, Intel MKL (the matrix products of PyTorch on the CPU) may add the terms
# of a sum in a different order from one run to the next; the training amplifies these differences in the last digit
# into a different network (the inverse problem has many equally good solutions). MKL_CBWR=COMPATIBLE fixes the order:
# the same network at every run. It must be set before torch is imported. The network of the chapter
# (databases/neural_networks/backward.pt) was trained without it and cannot be recovered by retraining (README, Determinism);
# its query (--select-only) therefore runs without it, as when the results of the chapter were made.
if "--select-only" not in sys.argv[1:]:
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import torch
from torch import nn

from ml.ml_common import (load_dataset, split_indices, Standardizer, cents, tuning, L_MIN_REL, MODELS_DIR,
                          R_MIN, R_MAX, OFFICIAL_NOTES_FREQUENCIES)
from ml.train_forward import make_mlp
from ml.rb_fast import FastRB, NOTE_MODES

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
torch.set_flush_denormal(True)  # tiny gradients (saturated softmax) are otherwise very slow on CPU
HIDDEN, DEPTH, DROPOUT, LR, BATCH, WEIGHT_DECAY, EPOCHS = 512, 4, 0.0, 1e-3, 256, 1e-5, 200   # selected with ml/tune_backward.py
DELTAS = np.arange(0, 151, 5)


class BackwardModel(nn.Module):
    def __init__(self, hidden=HIDDEN, depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        self.body = make_mlp(10, 8, hidden, depth, dropout=dropout)

    def forward(self, x):
        z = self.body(x)
        lengths = L_MIN_REL + (1.0 - 4.0 * L_MIN_REL) * torch.softmax(z[:, :4], dim=1)
        gaps = (R_MAX - R_MIN) * torch.softmax(z[:, 4:], dim=1)
        radii = R_MIN + torch.cumsum(gaps, dim=1)[:, :3]
        return torch.cat([lengths[:, :3], radii], dim=1)


def prepare():
    """Dataset, split, frozen forward model. Returns the data, the input standardizer and the tandem map."""
    mu, f = load_dataset()
    tr, va, te = split_indices(len(mu))
    ck = torch.load(MODELS_DIR / "forward.pt", weights_only=False)
    fwd = make_mlp(6, 10, 256, 5); fwd.load_state_dict(ck["state_dict"]); fwd.eval()
    for p in fwd.parameters():
        p.requires_grad_(False)
    fx_m, fx_s = (torch.tensor(ck["sx"][k], dtype=torch.float32) for k in ("mean", "std"))
    fy_m, fy_s = (torch.tensor(ck["sy"][k], dtype=torch.float32) for k in ("mean", "std"))

    sx = Standardizer(np.log10(f[tr]))
    X = torch.tensor(sx(np.log10(f)), dtype=torch.float32)
    Yf = (torch.tensor(np.log10(f), dtype=torch.float32) - fy_m) / fy_s   # targets in the forward model's output space

    def cycle(mu_pred):   # geometry -> frequencies (forward model output space)
        return fwd((mu_pred - fx_m) / fx_s)

    def to_hz(y):
        return 10.0 ** (y * fy_s + fy_m).numpy().astype(np.float64)

    return dict(mu=mu, f=f, tr=tr, va=va, te=te, sx=sx, X=X, Yf=Yf, cycle=cycle, to_hz=to_hz)


def train(d, hidden=HIDDEN, depth=DEPTH, dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS, seed=0):
    """Tandem training; returns the model at the epoch with the lowest validation loss, and that loss."""
    torch.manual_seed(seed)
    X, Yf, tr, va, cycle = d["X"], d["Yf"], d["tr"], d["va"], d["cycle"]
    model = BackwardModel(hidden, depth, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state = np.inf, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), BATCH):
            i = torch.as_tensor(tr)[perm[b:b + BATCH]]
            opt.zero_grad()
            loss = torch.mean((cycle(model(X[i])) - Yf[i]) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = torch.mean((cycle(model(X[va])) - Yf[va]) ** 2).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch < 10 or epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  val MSE (frequencies of the predicted geometry, forward-model space) = {val:.3e}  [{time.time() - t0:.0f} s]", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def predict_mu(model, sx, freqs):
    with torch.no_grad():
        return model(torch.tensor(sx(np.log10(np.atleast_2d(freqs))), dtype=torch.float32)).numpy().astype(np.float64)


def cycle_errors(model, d, idx, rb, n_rb=200, seed=0):
    """RMSE (cents) between requested and obtained frequencies on the samples idx:
    obtained with the forward model (all samples) and with the reduced model (n_rb random samples)."""
    with torch.no_grad():
        f_cycle = d["to_hz"](d["cycle"](model(d["X"][idx])))
    rmse_fwd = float(np.sqrt(np.mean(cents(f_cycle, d["f"][idx]) ** 2)))
    sub = np.random.default_rng(seed).choice(idx, n_rb, replace=False)
    errs = []
    for m, ft in zip(predict_mu(model, d["sx"], d["f"][sub]), d["f"][sub]):
        r = rb.resonances(m, fmin=30.0, fmax=615.0)
        if len(r) == 12:
            errs.append(cents(r[NOTE_MODES - 1], ft))
    return rmse_fwd, float(np.sqrt(np.mean(np.array(errs) ** 2))), len(errs)


def main(select_only=False):
    d = prepare()
    hyper = {"hidden": HIDDEN, "depth": DEPTH, "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY}
    if select_only:   # saved network with its own input standardizer and hyperparameters
        ck = torch.load(MODELS_DIR / "backward.pt", weights_only=False)
        hyper = ck["hyper"]
        model = BackwardModel(hyper["hidden"], hyper["depth"], hyper["dropout"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        d["sx"] = Standardizer.from_state(ck["sx"])
        d["X"] = torch.tensor(d["sx"](np.log10(d["f"])), dtype=torch.float32)
    else:
        model, _ = train(d)
    rb = FastRB()
    rmse_cycle, rmse_rb, _ = cycle_errors(model, d, d["te"], rb)
    print(f"TEST: requested vs obtained frequencies: {rmse_cycle:.2f} cents (forward model), {rmse_rb:.2f} cents (reduced model, 200 cases)")

    # query: transposed scale, evaluated with the reduced model
    rows = []
    for dl in DELTAS:
        m = predict_mu(model, d["sx"], OFFICIAL_NOTES_FREQUENCIES * 2 ** (dl / 1200))[0]
        r = rb.resonances(m, fmin=30.0, fmax=615.0)
        t = tuning(r[NOTE_MODES - 1]) if len(r) == 12 else (np.nan,) * 3
        rows.append((dl, m, *t))
        print(f"  delta = {dl:3d} cents: mu = {np.round(m, 4)}, reduced model: error {t[0]:.2f}, offset {t[1]:+.2f}, RMSE {t[2]:.2f}")
    j = int(np.nanargmin([r[2] for r in rows]))
    d_best, mu_best = rows[j][0], rows[j][1]
    print("mu_best_backward =", np.array2string(mu_best, precision=4), f" (delta = {d_best} cents), l4 = {1 - mu_best[:3].sum():.4f}")

    if not select_only:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "sx": d["sx"].state(), "hyper": hyper}, MODELS_DIR / "backward.pt")
    out = Path("results/data")
    best_file = out / "ml_best_mus.json"
    best_d = json.loads(best_file.read_text()) if best_file.exists() else {}
    best_d["backward"] = mu_best.tolist()
    best_file.write_text(json.dumps(best_d, indent=2))
    (out / "backward_summary.json").write_text(json.dumps({
        "hyper": hyper, "test_rmse_cycle_forward_cents": rmse_cycle, "test_rmse_cycle_rb_cents": rmse_rb,
        "delta_best": int(d_best), "mu": mu_best.tolist(),
        "err_rb": rows[j][2], "offset_rb": rows[j][3], "rmse_rb": rows[j][4],
        "scan": [{"delta": int(r[0]), "mu": r[1].tolist(), "err_rb": r[2], "offset_rb": r[3], "rmse_rb": r[4]} for r in rows]},
        indent=2))
    print("saved", best_file, out / "backward_summary.json")


if __name__ == "__main__":
    main(select_only="--select-only" in sys.argv[1:])
