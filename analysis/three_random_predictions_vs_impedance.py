from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "results" / "figures"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from scipy.signal import find_peaks

# ============================================================
# VARIABLES À MODIFIER
# ============================================================
CSV_FILES = [
    str(DATA_DIR / "valais_random_0.csv"),
    str(DATA_DIR / "valais_random_1.csv"),
    str(DATA_DIR / "valais_random_2.csv"),
]

X_RANDOM = np.array([
    [0.5986585,  0.15601864, 0.15599452,  9.113763, 33.550175, 45.47793],
    [0.13949387, 0.29214466, 0.36636186, 15.48532,  27.02315,  41.83292],
    [0.60754484, 0.17052412, 0.06505159, 42.87788,  49.19985,  49.95344],
], dtype=np.float32)

Y_PRED = np.array([
    [94.94175, 144.69936, 196.58524, 247.48817, 292.49454, 338.72583, 382.18497, 432.08328, 488.7137, 551.98016],
    [94.90093, 144.71173, 196.54736, 247.53426, 292.63525, 338.61523, 382.28503, 432.1074,  488.70813, 551.8443 ],
    [94.94573, 144.59262, 196.64618, 247.41217, 292.7556,  338.90015, 382.26382, 432.2816,  488.72098, 551.4027 ],
], dtype=np.float32)

# Geometries and forward-model predictions written by ml/train_forward.py (override the arrays above)
_pred_file = DATA_DIR / "forward_random_predictions.json"
if _pred_file.exists():
    import json
    _pred = json.loads(_pred_file.read_text())
    X_RANDOM = np.array(_pred["X_RANDOM"], dtype=np.float32)
    Y_PRED = np.array(_pred["Y_PRED"], dtype=np.float32)

OUT_FILE = str(FIGURES_DIR / "three_random_predictions_vs_impedance.pdf")

FIGSIZE = (13, 12)

# paramètres find_peaks à ajuster si besoin
PEAK_HEIGHT = None
PEAK_DISTANCE = None
PEAK_PROMINENCE = None


CURVE_LW = 1.5
PEAK_LW = 1.0
PRED_LW = 1.2
BAR_LW = 1.0

PEAK_STYLE = "--"
PRED_STYLE = "--"


ALPHA_CURVE = 0.95
ALPHA_PEAK = 0.8
ALPHA_PRED = 0.9
ALPHA_BAR = 0.9

# ============================================================
# OUTILS
# ============================================================
def load_impedance_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    frequencies = data[:, 0]
    impedances = data[:, 1]
    return frequencies, impedances


def detect_peaks(frequencies, impedances, height=None, distance=None, prominence=None):
    peak_indices, _ = find_peaks(
        impedances,
        height=height,
        distance=distance,
        prominence=prominence
    )
    peak_freqs = frequencies[peak_indices]
    peak_vals = impedances[peak_indices]
    return peak_indices, peak_freqs, peak_vals


def match_pred_to_detected(pred_freqs, detected_peak_freqs):
    pred_freqs = np.asarray(pred_freqs, dtype=float)
    detected_peak_freqs = np.asarray(detected_peak_freqs, dtype=float)

    matched = []
    for f_pred in pred_freqs:
        best_f = detected_peak_freqs[np.argmin(np.abs(detected_peak_freqs - f_pred))]
        matched.append(best_f)
    return np.asarray(matched, dtype=float)


def compute_deviation_cents(freqs, true_freqs):
    freqs = np.asarray(freqs, dtype=float)
    true_freqs = np.asarray(true_freqs, dtype=float)
    return 1200 * np.log2(freqs / true_freqs)


class HandlerBarBar(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):
        y = ydescent + 0.5 * height
        x0 = xdescent
        x1 = xdescent + width
        tick = 0.35 * height

        color = orig_handle.get_color()
        lw = orig_handle.get_linewidth()
        alpha = orig_handle.get_alpha()

        h = Line2D([x0, x1], [y, y], color=color, linewidth=lw, alpha=alpha)
        v0 = Line2D([x0, x0], [y - tick, y + tick], color=color, linewidth=lw, alpha=alpha)
        v1 = Line2D([x1, x1], [y - tick, y + tick], color=color, linewidth=lw, alpha=alpha)

        for a in (h, v0, v1):
            a.set_transform(trans)

        return [h, v0, v1]


# ============================================================
# PLOT
# ============================================================
n_cases = len(CSV_FILES)
fig, axes = plt.subplots(n_cases, 1, figsize=FIGSIZE, sharex=False)

if n_cases == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    csv_file = CSV_FILES[i]
    x_case = X_RANDOM[i]
    pred_freqs = Y_PRED[i]

    frequencies, impedances = load_impedance_csv(csv_file)

    peak_indices, detected_peak_freqs, detected_peak_vals = detect_peaks(
        frequencies,
        impedances,
        height=PEAK_HEIGHT,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
    )

    # FEM resonances with the same numbers as the predicted ones (2..10 and 12), refined (analysis/fem_resonances.py)
    _res = np.array(json.loads(Path(csv_file).with_name(Path(csv_file).stem + "_resonances.json").read_text())["resonances"])
    matched_detected = _res[np.array(_pred["modes"]) - 1]
    dev_pred_vs_detected = compute_deviation_cents(pred_freqs, matched_detected)
    rmse_pred_vs_detected = float(np.sqrt(np.mean(dev_pred_vs_detected ** 2)))
    print(f"case {i}: mu = {np.round(x_case, 4)}, RMSE predicted vs FEM resonances = {rmse_pred_vs_detected:.2f} cents")

    # courbe d'impédance
    # ax.plot(
    #     frequencies,
    #     impedances,
    #     linewidth=CURVE_LW,
    #     color="blue",
    #     alpha=ALPHA_CURVE,
    # )
    ax.semilogy(
        frequencies,
        impedances,
        linewidth=CURVE_LW,
        color="blue",
        alpha=ALPHA_CURVE,
    )

    y_max = float(np.max(impedances)) if len(impedances) else 1.0

    # pics détectés
    for f_peak in matched_detected:
        ax.axvline(
            x=float(f_peak),
            color="blue",
            linestyle=PEAK_STYLE,
            linewidth=PEAK_LW,
            alpha=ALPHA_PEAK,
        )

    # fréquences prédites
    # y_level_pred = y_max * 0.88
    y_level_pred = y_max * (10 ** 0.5)
    for j, f_pred in enumerate(pred_freqs):
        ax.axvline(
            x=float(f_pred),
            color="green",
            linestyle=PRED_STYLE,
            linewidth=PRED_LW,
            alpha=ALPHA_PRED,
        )

        ax.annotate(
            "",
            xy=(matched_detected[j], y_level_pred),
            xytext=(float(f_pred), y_level_pred),
            arrowprops=dict(
                arrowstyle="|-|",
                color="green",
                linewidth=BAR_LW,
                mutation_scale=4,
                alpha=ALPHA_BAR,
            ),
        )

    l1, l2, l3, R1, R2, R3 = x_case
    ax.set_title(
        rf"$\mathbf{{\mu=({l1:.3f},\, {l2:.3f},\, {l3:.3f},\, {R1:.2f},\, {R2:.2f},\, {R3:.2f})}}$"
        + " " * 5
        + rf"$\mathbf{{RMSE: {rmse_pred_vs_detected:.2f}\ \mathbf{{cents}}}}$",
        fontsize=14,
    )

    ax.set_ylabel("Impedance", fontsize=14)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    # ax.grid(True, axis="x")
    ax.set_ylim(bottom=y_max * (10 ** -4.6), top=y_max * (10 ** 1.0))

    print(f"\nCase {i+1}")
    print(f"CSV file: {csv_file}")
    print(f"Detected peaks: {detected_peak_freqs}")
    print(f"Predicted freqs: {pred_freqs}")
    print(f"Matched detected peaks: {matched_detected}")
    print(f"RMSE pred vs detected: {rmse_pred_vs_detected:.4f} cents")

# ============================================================
# LÉGENDE
# ============================================================
barbar_diff = Line2D([0], [0], color="green", linewidth=BAR_LW)

legend_elements = [
    Line2D([0], [0], color="blue", linewidth=CURVE_LW, label="Impedance"),
    Line2D([0], [0], color="blue", linestyle=PEAK_STYLE, linewidth=PEAK_LW, label=r"$f_{\mathrm{FEM}}$"),
    Line2D([0], [0], color="green", linestyle=PRED_STYLE, linewidth=PRED_LW, label=r"$f_{\mathrm{pred}}$"),
    barbar_diff,
]

legend_labels = [
    "Impedance",
    r"$f_{\mathrm{FEM}}$",
    r"$f_{\mathrm{pred}}$",
    r"$f_{\mathrm{pred}} - f_{\mathrm{FEM}}$",
]

fig.legend(
    legend_elements,
    legend_labels,
    handler_map={barbar_diff: HandlerBarBar()},
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=4,
    frameon=True,
    handlelength=2.2,
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_FILE)
plt.show()