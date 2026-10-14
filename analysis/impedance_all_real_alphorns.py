import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "results" / "figures"

sys.path.insert(0, str(SCRIPT_DIR.parent))  # make repo root importable (utils/) when run as `python analysis/impedance_all_real_alphorns.py`

import json
import numpy as np
from scipy.signal import find_peaks

# Notes of the alphorn scale = resonances n = 2..10 and 12, numbered from the lowest one (analysis/fem_resonances.py)
NOTE_MODES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])

# the three real alphorns (meshes in mesh/real_alphorns; FEM data in results/data/impedance_Alphorn_<k>*)
meshes = {"Alphorn 1": None, "Alphorn 3": None, "Alphorn 4": None}
from utils.frequencies import OFFICIAL_NOTES_FREQUENCIES, OFFICIAL_NOTES
from utils.valais import measured_freqs1, measured_freqs3, measured_freqs4

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

measured_map = {
    "Alphorn 1": measured_freqs1,
    "Alphorn 3": measured_freqs3,
    "Alphorn 4": measured_freqs4,
}

publication_names = {
    "Alphorn 1": "First alphorn",
    "Alphorn 3": "Second alphorn",
    "Alphorn 4": "Third alphorn",
}


CURVE_LW = 1.5
PEAK_LW = 1.0
MEAS_LW = 1.0
REF_LW = 1.0
BAR_LW = 1.0

PEAK_STYLE = "--"
MEAS_STYLE = "--"

ALPHA_CURVE = 0.95
ALPHA_PEAK = 0.8
ALPHA_MEAS = 0.9
ALPHA_BAR = 0.9

# DY = 0.1
DY = 0.4

def print_latex_table(name, publication_name, measured_used, official_for_measured, matched_peaks, dev_meas_vs_off, k=2):
    n = len(measured_used)

    print(f"% LaTeX table for {name}")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\hline")
    print(r"Note & Freq. (Hz) & Measured freq. (Hz) & Measured dev. (cents) & Simulated freq. (Hz) & Simulated dev. (cents) \\")
    print(r"\hline")

    for i in range(n):
        note_name = OFFICIAL_NOTES[k + i]["name"]
        true_freq = float(official_for_measured[i])
        measured_freq = float(measured_used[i])
        measured_dev = float(dev_meas_vs_off[i])

        simulated_freq = float(matched_peaks[k + i])
        simulated_dev = float(1200 * np.log2(simulated_freq / true_freq))

        print(
            f"{note_name} & "
            f"{true_freq:.2f} & "
            f"{measured_freq:.2f} & "
            f"{measured_dev:+.2f} & "
            f"{simulated_freq:.2f} & "
            f"{simulated_dev:+.2f} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(rf"\caption{{Measured and simulated frequencies compared to the true frequencies for {publication_name}.}}")
    print(rf"\label{{tab:{name.lower().replace(' ', '_')}}}")
    print(r"\end{table}")
    print()


def print_latex_table_transpose(name, publication_name, measured_used, official_for_measured, matched_peaks, dev_meas_vs_off, k=2):
    n = 7  # Nombre de notes

    official_for_measured = official_for_measured[:n]
    measured_used = measured_used[:n]
    dev_meas_vs_off = dev_meas_vs_off[:n]

    print("official:", official_for_measured)
    print("measured:", measured_used)

    notes = [OFFICIAL_NOTES[k + i]["name"] for i in range(n)]
    simulated_freqs = [float(matched_peaks[k + i]) for i in range(n)]
    simulated_dev = [
        float(1200 * np.log2(simulated_freqs[i] / float(official_for_measured[i])))
        for i in range(n)
    ]

    print(f"% LaTeX table for {name}")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{l" + "c" * n + "}")
    print(r"\hline")

    print(" & " + " & ".join(notes) + r" \\")
    print(r"\hline")

    print(r"Reference frequency $f_{\mathrm{ref}}$ (Hz) & " + " & ".join(f"{float(f):.2f}" for f in official_for_measured) + r" \\")
    print(r"Measured frequency $f_{\mathrm{meas}}$ (Hz) & " + " & ".join(f"{float(f):.2f}" for f in measured_used) + r" \\")
    print(r"Measured deviation $\Delta f_{\mathrm{meas}}$ (cents) & " + " & ".join(f"{float(d):+.2f}" for d in dev_meas_vs_off) + r" \\")
    print(r"Simulated frequency $f_{\mathrm{sim}}$ (Hz) & " + " & ".join(f"{float(f):.2f}" for f in simulated_freqs) + r" \\")
    print(r"Simulated deviation $\Delta f_{\mathrm{sim}}$ (cents) & " + " & ".join(f"{float(d):+.2f}" for d in simulated_dev) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(rf"\caption{{Measured and simulated frequencies compared to the true frequencies for {publication_name}.}}")
    print(rf"\label{{tab:{name.lower().replace(' ', '_')}}}")
    print(r"\end{table}")
    print()



def compute_deviation_cents(freqs, true_freqs=OFFICIAL_NOTES_FREQUENCIES):
    if len(freqs) != len(true_freqs):
        raise ValueError("The number of spikes must match the number of official notes frequencies.")
    freqs = np.array(freqs)
    return 1200 * np.log2(freqs / true_freqs)



names = list(meshes.keys())
fig, axes = plt.subplots(len(names), 1, figsize=(13, 11), sharex=True)

for ax, name in zip(axes, names):
    print(name)
    # Charger le fichier
    data = np.loadtxt(
        str(DATA_DIR / f"impedance_{name.replace(' ', '_')}.csv"),
        delimiter=",",
        skiprows=1  # on ignore la ligne d'en-tête
    )

    # Séparer fréquences et impédances
    frequencies = data[:, 0]
    impedances = data[:, 1]

    # Resonances (refined, numbered from the lowest one): the n-th note of the scale is resonance NOTE_MODES[n]
    res = np.array(json.loads((DATA_DIR / f"impedance_{name.replace(' ', '_')}_resonances.json").read_text())["resonances"])
    note_res = res[NOTE_MODES - 1]
    spikes = [(float(f), float(np.interp(f, frequencies, impedances))) for f in note_res]
    sim_peaks_freqs = [s[0] for s in spikes]

    measured_freqs = measured_map.get(name)


    # --- Simulation vs officiel ---
    deviations = compute_deviation_cents(sim_peaks_freqs)

    print("RMSE sim vs official:", np.sqrt(np.mean(deviations**2)))
    # print("MAE sim vs official:", np.mean(np.abs(deviations)))
    # print("")



    # --- Jouée vs simulation : déviation (pic simulé) par rapport à la note jouée ---
    def match_peaks_to_official_notes(spike_freqs, official_freqs):
        spike_freqs = np.asarray(spike_freqs, dtype=float)
        official_freqs = np.asarray(official_freqs, dtype=float)

        matched = []
        for f_off in official_freqs:
            best_f = spike_freqs[0]
            best_d = abs(best_f - f_off)
            for f in spike_freqs[1:]:
                d = abs(f - f_off)
                if d < best_d:
                    best_d = d
                    best_f = f
            matched.append(best_f)

        return np.asarray(matched, dtype=float)

    # one resonance per note, by resonance number (no nearest-peak matching)
    matched_peaks = np.asarray(sim_peaks_freqs, dtype=float)

    dev_meas_vs_sim = compute_deviation_cents(
                        measured_freqs, 
                        true_freqs=matched_peaks[2:len(measured_freqs)+2]
                    )

    print("RMSE measured vs sim:", float(np.sqrt(np.mean(dev_meas_vs_sim**2))))
    # print("MAE measured vs sim:", float(np.mean(np.abs(dev_meas_vs_sim))))
    # print()



    # --- Mesuré vs officiel ---
    k = 2  # on saute les 2 premières notes officielles
    
    n = min(len(measured_freqs), len(OFFICIAL_NOTES_FREQUENCIES) - k)

    official_for_measured = OFFICIAL_NOTES_FREQUENCIES[k:k+n]
    measured_used = measured_freqs[:n]

    dev_meas_vs_off = compute_deviation_cents(measured_used, true_freqs=official_for_measured)

    print("RMSE measured vs official:", float(np.sqrt(np.mean(dev_meas_vs_off**2))))
    # print("MAE measured vs official:", float(np.mean(np.abs(dev_meas_vs_off))))
    # print()


    print_latex_table(
        name=name,
        publication_name=publication_names[name],
        measured_used=measured_used,
        official_for_measured=official_for_measured,
        matched_peaks=matched_peaks,
        dev_meas_vs_off=dev_meas_vs_off,
        k=k,
    )

    print_latex_table_transpose(
        name=name,
        publication_name=publication_names[name],
        measured_used=measured_used,
        official_for_measured=official_for_measured,
        matched_peaks=matched_peaks,
        dev_meas_vs_off=dev_meas_vs_off,
        k=k,
    )

    
    def closest_official_freq(f):
        official = np.array([n["freq"] for n in OFFICIAL_NOTES], dtype=float)
        return float(official[np.argmin(np.abs(official - f))])

    # ---------- PLOT ----------
    # plt.figure(figsize=(10, 6))
    # plt.plot(frequencies, impedances, linewidth=1, color="blue")
    # ax.plot(frequencies, impedances, linewidth=CURVE_LW, color="blue", alpha=ALPHA_CURVE)
    ax.semilogy(frequencies, impedances, linewidth=CURVE_LW, color="blue", alpha=ALPHA_CURVE)

    y_max = float(impedances.max()) if len(impedances) else 1.0
    
    k = 2  # on saute les 2 premières notes officielles

    # --- Notes officielles ---
    # y_text_off = y_max * 0.98 # position des labels au dessus
    # y_text_off = y_max * 1.5 # position des labels au dessus
    # y_text_off = y_max * (10 ** (2 * DY)) # position des labels au dessus, indépendant de y_level/y_level_peaks
    # y_text_off = y_max * (10 ** 1.5) # position des labels au dessus (1.5 décade au-dessus du pic)
    y_text_off = y_max * (10 ** 1.6) # position des labels au dessus (1.6 décade au-dessus du pic)

    for note in OFFICIAL_NOTES:
        f = note["freq"]
        ax.axvline(x=f, color="black", linestyle="-", linewidth=REF_LW)
        ax.text(
            f,
            y_text_off,
            note["name"],
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            clip_on=False,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="square,pad=0.4"
            )
        )

    # --- Measured freqs avec flèches horizontales ---
    # y_level = y_text_off - DY * y_max  # niveau des flèches horizontales (un peu en dessous des labels)
    # y_level = y_text_off / (10 ** DY)  # niveau des flèches horizontales (un peu en dessous des labels)
    # y_level = y_max * (10 ** DY)  # niveau des flèches horizontales, indépendant de y_text_off
    y_level = y_max * (10 ** 0.9)  # niveau des flèches horizontales (0.9 décade au-dessus du pic)

    if measured_freqs is not None:

        # the seven notes reported in the table (185-468 Hz)
        n = min(len(measured_freqs), 7, len(OFFICIAL_NOTES) - k)

        for i in range(n):

            f_meas = float(measured_freqs[i])
            f_off = float(OFFICIAL_NOTES[k + i]["freq"])

            # ligne mesurée
            ax.axvline(
                x=f_meas,
                color="green",
                linestyle=MEAS_STYLE,
                linewidth=MEAS_LW,
                alpha=ALPHA_MEAS,
            )

            # flèche horizontale du mesuré vers l'officiel
            ax.annotate(
                "",
                xy=(f_off, y_level),       # pointe vers officiel
                xytext=(f_meas, y_level),  # part du mesuré
                arrowprops=dict(
                    arrowstyle="|-|",
                    color="green",
                    linewidth=BAR_LW,
                    mutation_scale=4,
                    alpha=ALPHA_BAR,
                ),
            )

    # --- Pics détectés ---
    # y_level_peaks = y_level - DY * y_max  # un peu plus bas que les flèches vertes
    # y_level_peaks = y_level / (10 ** DY)  # un peu plus bas que les flèches vertes
    # y_level_peaks = y_max * (10 ** (0.3 * DY))  # juste au-dessus du pic de la courbe, indépendant de y_level
    # y_level_peaks = y_max * (10 ** (0.6 * DY))  # juste au-dessus du pic de la courbe, indépendant de y_level
    y_level_peaks = y_max * (10 ** 0.4)  # juste au-dessus du pic de la courbe (0.4 décade)

    for (f_peak, _), f_note in zip(spikes, OFFICIAL_NOTES_FREQUENCIES):
        # ligne du pic
        ax.axvline(
            x=f_peak,
            color="blue",
            linestyle=PEAK_STYLE,
            linewidth=PEAK_LW,
            alpha=ALPHA_PEAK,
        )

        # note of the scale corresponding to this resonance
        f_off = float(f_note)

        # flèche horizontale pic -> officiel
        ax.annotate(
            "",
            xy=(f_off, y_level_peaks),
            xytext=(float(f_peak), y_level_peaks),
            arrowprops=dict(
                arrowstyle="|-|",
                color="blue",
                linewidth=BAR_LW,
                mutation_scale=4,
                alpha=ALPHA_BAR,
            ),
        )



    # --- Légendes ---
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D

    class HandlerBarBar(HandlerBase):
        def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height,
                        fontsize, trans):

            y = ydescent + 0.5 * height
            x0 = xdescent
            x1 = xdescent + width
            tick = 0.35 * height  # hauteur des barres verticales

            color = orig_handle.get_color()
            lw = orig_handle.get_linewidth()
            alpha = orig_handle.get_alpha()

            h = Line2D([x0, x1], [y, y], color=color, linewidth=lw, alpha=alpha)
            v0 = Line2D([x0, x0], [y - tick, y + tick], color=color, linewidth=lw, alpha=alpha)
            v1 = Line2D([x1, x1], [y - tick, y + tick], color=color, linewidth=lw, alpha=alpha)

            for a in (h, v0, v1):
                a.set_transform(trans)

            return [h, v0, v1]

    # handles "|-|" pour la légende
    barbar_green = Line2D([0], [0], color="green", linewidth=BAR_LW, alpha=ALPHA_BAR)
    barbar_blue = Line2D([0], [0], color="blue", linewidth=BAR_LW, alpha=ALPHA_BAR)


    legend_elements = [
        Line2D([0], [0], color="blue", linewidth=CURVE_LW),
        Line2D([0], [0], color="black", linestyle="-", linewidth=REF_LW),
        Line2D([0], [0], color="green", linestyle=MEAS_STYLE, linewidth=MEAS_LW, alpha=ALPHA_MEAS),
        Line2D([0], [0], color="blue", linestyle=PEAK_STYLE, linewidth=PEAK_LW, alpha=ALPHA_PEAK),
        barbar_green,
        barbar_blue,
    ]

    # legend_labels = [
    #     "Impedance",
    #     "Official notes",
    #     "Measured frequencies",
    #     "Detected peaks",
    #     "Measured - official",
    #     "Peak - nearest official",
    # ]

    legend_labels = [
        "Impedance",
        r"$f_{\mathrm{note}}$",
        r"$f_{\mathrm{meas}}$",
        r"$f_{\mathrm{FEM}}$",
        r"$f_{\mathrm{meas}} - f_{\mathrm{note}}$",
        r"$f_{\mathrm{FEM}} - f_{\mathrm{note}}$",
    ]

    # ax.legend(
    #     legend_elements,
    #     legend_labels,
    #     handler_map={barbar_green: HandlerBarBar(), barbar_blue: HandlerBarBar()},
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.14),
    #     ncol=3,
    #     frameon=True,
    #     handlelength=2.2
    # )
    ax.set_title(publication_names[name], fontsize=14, fontweight="bold")

    # ax.set_ylim(bottom=0, top=y_max * 1.1)
    # ax.set_ylim(top=y_max * 1.1)
    # ax.set_ylim(top=y_max * 3)
    # ax.set_ylim(top=y_max * (10 ** (2 * DY + 0.2)))
    # ax.set_ylim(top=y_max * (10 ** (2 * DY + 0.4)))
    # ax.set_ylim(bottom=y_max * (10 ** -3), top=y_max * (10 ** 2))
    # ax.set_ylim(bottom=y_max * (10 ** -4.5), top=y_max * (10 ** 2))
    ax.set_ylim(bottom=y_max * (10 ** -4.5), top=y_max * (10 ** 2.3))
    ax.set_ylabel("Impedance", fontsize=14)
    
axes[-1].set_xlabel("Frequency (Hz)", fontsize=14)

fig.legend(
    legend_elements,
    legend_labels,
    handler_map={barbar_green: HandlerBarBar(), barbar_blue: HandlerBarBar()},
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=3,
    frameon=True,
    handlelength=2.2,
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(str(FIGURES_DIR / "impedance_all_real_alphorns.pdf"))

