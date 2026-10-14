"""
Hyperparameter selection for the backward (tandem) model.

Grid: width {256, 512, 1024} x l2 regularization {1e-3, 1e-5} x dropout {0, 0.1}; depth 4, learning rate 1e-3,
200 epochs. Each configuration is trained on the training set (checkpoint with the lowest validation loss) and
scored on the validation set by the RMSE (cents) between requested and obtained frequencies, the obtained
frequencies being computed with the reduced model (200 validation samples); the test set is not used.

Usage (from the repo root):
  python ml/tune_backward.py K        # train configuration K (0..11), e.g. K = $SLURM_ARRAY_TASK_ID
  python ml/tune_backward.py summary  # table of the results
"""
import itertools
import json
import sys

from ml.ml_common import MODELS_DIR
from ml.rb_fast import FastRB
from ml.train_backward import prepare, train, cycle_errors, DEPTH, LR, EPOCHS

CONFIGS = [dict(hidden=h, weight_decay=w, dropout=p, depth=DEPTH, lr=LR)
           for h, w, p in itertools.product((256, 512, 1024), (1e-3, 1e-5), (0.0, 0.1))]
OUT = MODELS_DIR / "tune_backward"


def run(k):
    cfg = CONFIGS[k]
    print("configuration", k, cfg, flush=True)
    d = prepare()
    model, val_mse = train(d, epochs=EPOCHS, **cfg)
    rmse_fwd, rmse_rb, n = cycle_errors(model, d, d["va"], FastRB(), seed=1)
    res = {**cfg, "k": k, "val_mse_forward_space": val_mse, "val_rmse_cycle_forward_cents": rmse_fwd,
           "val_rmse_cycle_rb_cents": rmse_rb, "n_rb": n}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cfg_{k:02d}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


def summary():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("cfg_*.json"))]
    rows.sort(key=lambda r: r["val_rmse_cycle_rb_cents"])
    print(f"{'k':>2} {'width':>5} {'l2':>7} {'dropout':>7} {'val MSE':>9} {'fwd [c]':>8} {'RB [c]':>7}")
    for r in rows:
        print(f"{r['k']:2d} {r['hidden']:5d} {r['weight_decay']:7.0e} {r['dropout']:7.1f} {r['val_mse_forward_space']:9.2e} "
              f"{r['val_rmse_cycle_forward_cents']:8.2f} {r['val_rmse_cycle_rb_cents']:7.2f}")
    print(f"{len(rows)} of {len(CONFIGS)} configurations done")


if __name__ == "__main__":
    summary() if sys.argv[1] == "summary" else run(int(sys.argv[1]))
