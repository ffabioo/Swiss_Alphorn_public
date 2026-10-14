"""
Figure fig:comparaison-back-forw-real: FEM impedance of the reference alphorn and of the geometries of the forward and
backward models, with their resonances (n = 2..10, 12) and the alphorn scale.
By default the offset of the model, phi (ml_common.COMMON_OFFSET, the same for all the geometries), is removed: the FEM
frequencies are multiplied by 2^(-phi/1200), so that the bars show the deviations Delta f - phi of the final table.
  python analysis/impedance_comparison_backward_forward_real.py               offset of the model removed (figure of the chapter)
  python analysis/impedance_comparison_backward_forward_real.py --with-offset  FEM frequencies as computed
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "results" / "figures"

sys.path.insert(0, str(SCRIPT_DIR.parent))  # make repo root importable (utils/) when run as `python analysis/impedance_comparison_backward_forward_real.py`

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

# ============================================================
# VARIABLES À MODIFIER
# ============================================================
CSV_BACKWARD = str(DATA_DIR / "backward.csv")
CSV_FORWARD = str(DATA_DIR / "forward.csv")
CSV_REAL = str(DATA_DIR / "white_alphorn.csv")

OUT_FILE = str(FIGURES_DIR / "impedance_comparison_backward_forward_real.pdf")

from utils.frequencies import OFFICIAL_NOTES_FREQUENCIES, OFFICIAL_NOTES
from ml.ml_common import COMMON_OFFSET
WITH_OFFSET = "--with-offset" in sys.argv[1:]
SHIFT = 1.0 if WITH_OFFSET else 2.0 ** (-COMMON_OFFSET / 1200.0)   # FEM frequencies x SHIFT: offset of the model removed
TARGET_NOTE_FREQS = OFFICIAL_NOTES_FREQUENCIES
TARGET_NOTE_NAMES = [note["name"] for note in OFFICIAL_NOTES]
# Notes of the alphorn scale = resonances n = 2..10 and 12, numbered from the lowest one (analysis/fem_resonances.py)
NOTE_MODES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])

PEAK_HEIGHT = None
PEAK_DISTANCE = None
PEAK_PROMINENCE = None

# ============================================================
# STYLE (harmonisé avec le plot précédent)
# ============================================================
FIGSIZE = (12, 6)

CURVE_LW = 1.5
PEAK_LW = 1.0
REF_LW = 1.0
BAR_LW = 1.0

PEAK_STYLE = "--"

ALPHA_CURVE = 0.95
ALPHA_PEAK = 0.8

COLOR_BACKWARD = "tab:red"
COLOR_FORWARD = "tab:green"
COLOR_REAL = "tab:blue"
COLOR_REF = "black"

LABEL_BACKWARD = "Backward model"
LABEL_FORWARD = "Forward model"
LABEL_REAL = "Reference alphorn"

BAR_Y_START = 0.88
BAR_Y_STEP = 0.08
BAR_MUTATION_SCALE = 5


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
        prominence=prominence,
    )
    peak_freqs = frequencies[peak_indices]
    peak_vals = impedances[peak_indices]
    return peak_indices, peak_freqs, peak_vals


def plot_curve_with_peaks(
    ax,
    csv_path,
    curve_color,
    curve_label,
    height=None,
    distance=None,
    prominence=None,
):
    frequencies, impedances = load_impedance_csv(csv_path)
    frequencies = frequencies * SHIFT

    # ax.plot(
    #     frequencies,
    #     impedances,
    #     color=curve_color,
    #     linewidth=CURVE_LW,
    #     alpha=ALPHA_CURVE,
    #     label=curve_label,
    # )
    ax.semilogy(
        frequencies,
        impedances,
        color=curve_color,
        linewidth=CURVE_LW,
        alpha=ALPHA_CURVE,
        label=curve_label,
    )

    # refined FEM resonances of the notes (resonances 2..10 and 12)
    res_file = Path(csv_path).with_name(Path(csv_path).stem + "_resonances.json")
    peak_freqs = np.array(json.loads(res_file.read_text())["resonances"])[NOTE_MODES - 1] * SHIFT

    for f_peak in peak_freqs:
        ax.axvline(
            x=float(f_peak),
            color=curve_color,
            linestyle=PEAK_STYLE,
            linewidth=PEAK_LW,
            alpha=ALPHA_PEAK,
        )

    return frequencies, impedances, peak_freqs


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
# CHARGEMENT + PLOT
# ============================================================
if WITH_OFFSET:
    fig, ax = plt.subplots(figsize=FIGSIZE)
else:   # spectra on top, deviations of the notes (cents, offset of the model removed) below
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.7]})

freq_b, imp_b, peaks_b = plot_curve_with_peaks(
    ax=ax,
    csv_path=CSV_BACKWARD,
    curve_color=COLOR_BACKWARD,
    curve_label=LABEL_BACKWARD,
    height=PEAK_HEIGHT,
    distance=PEAK_DISTANCE,
    prominence=PEAK_PROMINENCE,
)

freq_f, imp_f, peaks_f = plot_curve_with_peaks(
    ax=ax,
    csv_path=CSV_FORWARD,
    curve_color=COLOR_FORWARD,
    curve_label=LABEL_FORWARD,
    height=PEAK_HEIGHT,
    distance=PEAK_DISTANCE,
    prominence=PEAK_PROMINENCE,
)

freq_r, imp_r, peaks_r = plot_curve_with_peaks(
    ax=ax,
    csv_path=CSV_REAL,
    curve_color=COLOR_REAL,
    curve_label=LABEL_REAL,
    height=PEAK_HEIGHT,
    distance=PEAK_DISTANCE,
    prominence=PEAK_PROMINENCE,
)

# ============================================================
# f_ref
# ============================================================
y_max = float(max(np.max(imp_b), np.max(imp_f), np.max(imp_r)))
# y_text = y_max * 1.03
y_text = y_max * (10 ** 1.5)

# y_bar_backward = y_max * BAR_Y_START
# y_bar_forward = y_max * (BAR_Y_START - BAR_Y_STEP)
# y_bar_real = y_max * (BAR_Y_START - 2 * BAR_Y_STEP)
y_bar_backward = y_max * (10 ** 1.0)
y_bar_forward = y_max * (10 ** 0.65)
y_bar_real = y_max * (10 ** 0.3)

for f, name in zip(TARGET_NOTE_FREQS, TARGET_NOTE_NAMES):
    ax.axvline(
        x=float(f),
        color=COLOR_REF,
        linestyle="-",
        linewidth=REF_LW,
        alpha=0.9,
    )
    ax.text(
        float(f),
        y_text,
        name,
        ha="center",
        va="bottom",
        fontsize=11,
        color=COLOR_REF,
        clip_on=False,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="square,pad=0.25",
        ),
    )

# one resonance per note, by resonance number
matched_b, ref_b = peaks_b, np.asarray(TARGET_NOTE_FREQS, dtype=float)
matched_f, ref_f = peaks_f, np.asarray(TARGET_NOTE_FREQS, dtype=float)
matched_r, ref_r = peaks_r, np.asarray(TARGET_NOTE_FREQS, dtype=float)

# ============================================================
# BARRES |-| : f_peak - f_ref (figure with the offset only; without it the deviations are shown in cents below)
# ============================================================
if WITH_OFFSET:
    for matched, ref, y_bar, color in [(matched_b, ref_b, y_bar_backward, COLOR_BACKWARD),
                                       (matched_f, ref_f, y_bar_forward, COLOR_FORWARD),
                                       (matched_r, ref_r, y_bar_real, COLOR_REAL)]:
        for f_peak, f_ref in zip(matched, ref):
            ax.annotate("", xy=(float(f_ref), y_bar), xytext=(float(f_peak), y_bar),
                        arrowprops=dict(arrowstyle="|-|", color=color, linewidth=BAR_LW, mutation_scale=BAR_MUTATION_SCALE))

# ============================================================
# MISE EN FORME
# ============================================================
ax.set_ylabel("Impedance", fontsize=13)
ax.grid(True, axis="x")
ax.set_ylim(bottom=y_max * (10 ** -4.6), top=y_max * (10 ** 1.90))

curves = [(COLOR_BACKWARD, LABEL_BACKWARD, "Backward", peaks_b), (COLOR_FORWARD, LABEL_FORWARD, "Forward", peaks_f),
          (COLOR_REAL, LABEL_REAL, "Reference", peaks_r)]
notes = np.asarray(TARGET_NOTE_FREQS, dtype=float)
if WITH_OFFSET:
    ax.set_xlabel("Frequency (Hz)", fontsize=13)
    bars = [Line2D([0], [0], color=c, linewidth=BAR_LW) for c, *_ in curves]
    legend_elements = ([Line2D([0], [0], color=c, linewidth=CURVE_LW) for c, *_ in curves]
                       + [Line2D([0], [0], color=c, linestyle=PEAK_STYLE, linewidth=PEAK_LW) for c, *_ in curves] + bars)
    legend_labels = ([lab for _, lab, _, _ in curves] + [rf"{short} $f_{{\mathrm{{FEM}}}}$" for _, _, short, _ in curves]
                     + [rf"{short}: $f_{{\mathrm{{FEM}}}} - f_{{\mathrm{{note}}}}$" for _, _, short, _ in curves])
    handler_map = {b: HandlerBarBar() for b in bars}
    ncol = 3
else:
    # deviations of the notes in cents after removing the offset of the model (rows Delta f - phi of the final table)
    axd.axhline(0.0, color=COLOR_REF, linewidth=REF_LW)
    for f in notes:
        axd.axvline(float(f), color=COLOR_REF, linewidth=REF_LW, alpha=0.9)
    legend_elements, legend_labels = [], []
    for k, (color, label, short, peaks) in enumerate(curves):
        dev = 1200.0 * np.log2(peaks / SHIFT / notes) - COMMON_OFFSET     # peaks are already shifted: undo for cents
        x = notes + (k - 1) * 4.0                                          # small horizontal offset per geometry
        axd.vlines(x, 0.0, dev, color=color, linewidth=1.6)
        axd.plot(x, dev, "o", color=color, markersize=5)
        E = float(np.sqrt(np.mean(dev ** 2)))
        legend_elements += [Line2D([0], [0], color=color, linewidth=CURVE_LW)]
        legend_labels += [rf"{label} ($E = {E:.1f}$ cents)"]
    legend_elements += [Line2D([0], [0], color="0.4", linestyle=PEAK_STYLE, linewidth=PEAK_LW)]
    legend_labels += [r"$f_{\mathrm{FEM}}$ (resonances $n = 2, \dots, 10, 12$)"]
    axd.set_ylabel(r"$\Delta f - \phi$ (cents)", fontsize=13)
    axd.set_xlabel(rf"Frequency (Hz); FEM frequencies shifted by $-\phi = -{COMMON_OFFSET:.1f}$ cents", fontsize=13)
    axd.grid(True, axis="y", alpha=0.4)
    lim = 5 * np.ceil(max(abs(axd.get_ylim()[0]), abs(axd.get_ylim()[1])) / 5)
    axd.set_ylim(-lim, lim)
    handler_map = {}
    ncol = 2

ax.legend(legend_elements, legend_labels, handler_map=handler_map, loc="lower center", fontsize=12, frameon=True,
          bbox_to_anchor=(0.5, 1.02), ncol=ncol)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(OUT_FILE)

print("Backward peaks (Hz):", peaks_b / SHIFT)
print("Forward peaks (Hz):", peaks_f / SHIFT)
print("Real peaks (Hz):", peaks_r / SHIFT)
