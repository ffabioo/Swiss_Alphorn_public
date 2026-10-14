r"""
LaTeX tables of the chapter, written to results/tables/:
  - real_alphorns_table.tex  (tab:real_alphorns)   measured notes and FEM resonances of the three real alphorns
  - final_table.tex          (tab:final_comparison_models)   FEM resonances of the optimized geometries
  - geometry_table.tex       (tab:geom_optimized)   cone lengths and junction radii, in mm
The FEM resonances are read from results/data/<name>_resonances.json (written by analysis/fem_resonances.py): refined
peaks of |Z_in| numbered from the lowest one; the notes of the scale are the resonances n = 2..10 and 12.
The optimized geometries are compared through the RMSE of the deviations after subtracting the offset of the model
c = ml_common.COMMON_OFFSET (FEM offset of the reference alphorn, the same for all the geometries).
A summary of the numbers is written to analysis/output/tables_summary.json.

Run from the repo root (plain Python, no dolfinx):  python analysis/make_tables.py
"""
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.valais import measured_freqs1, measured_freqs3, measured_freqs4, mu_glob_h4, L, R_MIN, R_MAX  # noqa: E402
from ml.ml_common import COMMON_OFFSET as C  # noqa: E402

DATA = REPO / "results" / "data"
OUT = REPO / "results" / "tables"
SUMMARY = REPO / "analysis" / "output" / "tables_summary.json"

NOTES = np.array([92, 139, 185, 234, 278, 331, 371, 417, 468, 556.])
NOTE_MODES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
NAMES = ["Do$_1$", "Sol$_1$", "Do$_2$", "Mi$_1$", "Sol$_2$", "Sib", "Do$_3$", "Ré", "Mi$_2$", "Sol$_3$"]
LETTERS = ["C", "G", "C", "E", "G", r"B$\flat$", "C", "D", "E", "G"]

# real alphorns: FEM on their own meshes (analysis/fem_resonances.py --mesh mesh/real_alphorns/...)
REAL = {"First alphorn": ("impedance_Alphorn_1", measured_freqs1),
        "Second alphorn": ("impedance_Alphorn_3", measured_freqs3),
        "Third alphorn": ("impedance_Alphorn_4", measured_freqs4)}
# optimized geometries: parametric FEM (analysis/fem_resonances.py --mu), mu from results/data/ml_best_mus.json
BLOCKS = [("Backward model", "backward"), ("Forward model", "forward"),
          ("Best geometry of the training data", "lookup"), ("Reference alphorn", "white_alphorn")]
GEOMETRIES = [("Reference", None), ("Forward", "forward"), ("Best training", "lookup"), ("Backward", "backward")]


def cents(f, f0):
    return 1200 * np.log2(np.asarray(f, float) / np.asarray(f0, float))


def on_scale(values, first=2):
    """Measured notes listed from NOTES[first] upwards -> array over the 10 notes (NaN = not measured)."""
    out = np.full(len(NOTES), np.nan)
    v = np.asarray(values, float)
    out[first:first + len(v)] = v
    return out


def fem_notes(name):
    r = np.array(json.loads((DATA / f"{name}_resonances.json").read_text())["resonances"])
    return r[NOTE_MODES - 1]


REF = r"\phi"   # offset of the model (offset of the reference alphorn), notation of the chapter


def cell(x, spec):
    return "N/A" if np.isnan(x) else format(x, spec)


BEST = r"\providecommand{\best}[1]{\textcolor[rgb]{0,0.55,0}{#1}}"   # best value of each note and lowest error, in green


def row(label, values, spec, math=False, best=None, green=None):
    fmt = (lambda x: f"${cell(x, spec)}$") if math else (lambda x: cell(x, spec))   # math: proper minus sign
    best = np.zeros(len(values), bool) if best is None else best
    green = np.zeros(len(values), bool) if green is None else green
    def one(x, b, g):
        t = fmt(x)
        t = rf"\best{{{t}}}" if b else t
        return rf"\textcolor{{green}}{{{t}}}" if g else t
    return label + " & " + " & ".join(one(x, b, g) for x, b, g in zip(values, best, green)) + r" \\"


# Table of the real alphorns: measured deviations marked in green by the authors (indices in NAMES)
GREEN_MEASURED = {"First alphorn": [4, 6, 7, 8], "Second alphorn": [4, 7], "Third alphorn": [2, 3, 4, 5, 7]}


def real_table():
    L_ = [r"\begin{table}[H]", r"\renewcommand{\arraystretch}{1.3}", r"\centering",
          r"\caption{Measured and FEM-derived resonance frequencies compared to the alphorn scale frequencies for the three alphorns. The resonance frequencies $f_{\mathrm{FEM}}$ are extracted from FEM-computed impedance spectra (resonances $n = 2,\dots,10$ and $12$, numbered from the lowest one. The deviations $\Delta f_{\mathrm{meas}}$ and $\Delta f_{\mathrm{FEM}}$ are expressed in cents relative to $f_{\mathrm{note}}$. N/A: not available (note not measured). In green, the best performances relative to a specific note are highlighted.}",
          r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccccc}", ""]
    for k, (title, (name, measured)) in enumerate(REAL.items()):
        fm, ff = on_scale(measured), fem_notes(name)
        header = [f"{n} ({c})" if k == 0 else n for n, c in zip(NAMES, LETTERS)]
        L_ += [rf"\multicolumn{{11}}{{c}}{{\textbf{{{title}}}}} \\", r"\hline", "Notes & " + " & ".join(header) + r" \\", r"\hline",
               row(r"Target frequency $f_{\mathrm{note}}$ (Hz)", NOTES, ".2f"),
               row(r"Measured frequency $f_{\mathrm{meas}}$ (Hz)", fm, ".2f"),
               row(r"Measured deviation $\Delta f_{\mathrm{meas}}$ (cents)", cents(fm, NOTES), "+.2f",
                   green=np.isin(np.arange(len(NOTES)), GREEN_MEASURED[title])),
               row(r"FEM frequency $f_{\mathrm{FEM}}$ (Hz)", ff, ".2f"),
               row(r"Deviation $\Delta f_{\mathrm{FEM}}$ (cents)", cents(ff, NOTES), "+.2f"),
               r"\hline \\", ""]
    L_ += [r"\end{tabular}", "}", r"\label{tab:real_alphorns}", "", r"\end{table}"]
    return "\n".join(L_) + "\n"


def final_table():
    cap_dc = rf"$\Delta f - {REF}$, the deviations after subtracting the offset of the model, ${REF} = {C:.1f}$~cents (the offset of the reference alphorn, the same for all the geometries)"
    cap_mean, cap_err = r"(mean of $\Delta f$)", rf"$\Delta f - {REF}$"
    L_ = [BEST, r"\begin{table}", r"\centering", "",
          rf"\caption{{Comparison of the reference alphorn geometry and the geometries obtained from the forward and backward models and the best geometry of the training data (the exhaustive search of the grid with the reduced model gives the same geometry as the forward model). For each case, the alphorn scale frequencies $f_{{\mathrm{{note}}}}$, the resonance frequencies $f_{{\mathrm{{FEM}}}}$ extracted from FEM-computed impedance spectra (resonances $n = 2,\dots,10$ and $12$, refined), and the corresponding deviations $\Delta f$, expressed in cents relative to $f_{{\mathrm{{note}}}}$, are reported, together with {cap_dc}. Below each block: the offset $\overline{{\Delta f}}$ {cap_mean}, the root mean squared error (RMSE) of $\Delta f$, and the RMSE of {cap_err}, used to compare the geometries. In green: for each note, the value of $\Delta f - {REF}$ closest to zero, and the lowest RMSE of $\Delta f - {REF}$.}}",
          "", r"\renewcommand{\arraystretch}{1.3}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccccc}", "", "% -----------------", ""]
    summary = {}
    devc = {name: cents(fem_notes(name), NOTES) - C for _, name in BLOCKS}
    best_note = np.argmin(np.abs(np.array([devc[n] for _, n in BLOCKS])), axis=0)   # block with the best value of each note
    errs = {n: np.sqrt(np.mean(v**2)) for n, v in devc.items()}
    best_err = min(errs, key=errs.get)
    for k, (title, name) in enumerate(BLOCKS):
        f = fem_notes(name)
        d = cents(f, NOTES)
        rmse, off, err = np.sqrt(np.mean(d**2)), d.mean(), np.sqrt(np.mean((d - C)**2))
        off_s, rmse_s = f"{off:+.2f}", f"{rmse:.2f}"
        err_v = rf"\textbf{{{err:.2f}}}"
        err_s = rf"RMSE of $\Delta f - {REF}$: " + (rf"\best{{{err_v}}}" if name == best_err else err_v)
        summary[name] = {"f_fem": f.tolist(), "dev": d.tolist(), "rmse": float(rmse), "offset": float(off), "err": float(err), "dev_minus_c": (d - C).tolist()}
        L_ += [rf"\multicolumn{{11}}{{c}}{{\textbf{{{title}}}}} \\", r"\hline", "Notes & " + " & ".join(NAMES) + r" \\", r"\hline",
               row(r"$f_{\mathrm{note}}$ (Hz)", NOTES, ".2f"),
               row(r"$f_{\mathrm{FEM}}$ (Hz)", f, ".2f"),
               row(r"$\Delta f$ (cents)", d, "+.2f"),
               row(rf"$\Delta f - {REF}$ (cents)", d - C, "+.2f", math=True, best=best_note == k),
               rf"\multicolumn{{11}}{{r}}{{\textit{{Offset $\overline{{\Delta f}}$: {off_s} cents; RMSE of $\Delta f$: {rmse_s} cents; {err_s} cents}}}} \\",
               r"\hline \\"]
    L_ += ["% ------------------", r"\end{tabular}", "}", r"\label{tab:final_comparison_models}", r"\end{table}"]
    return "\n".join(L_) + "\n", summary


def half_up(x, nd):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def geometry_mm(mu):
    """mu = (l1, l2, l3, R2, R3, R4) -> cone lengths L1..L4 (mm) and radii R2..R4 (mm)."""
    mu = np.asarray(mu, float)
    lengths = np.r_[mu[:3], 1 - mu[:3].sum()] * L
    return lengths, mu[3:]


def geometry_table():
    best = json.loads((DATA / "ml_best_mus.json").read_text())
    rows, summary = [], {}
    for label, key in GEOMETRIES:
        lengths, radii = geometry_mm(mu_glob_h4 if key is None else best[key])
        summary[label] = {"L_mm": lengths.tolist(), "R_mm": radii.tolist()}
        rows.append(f"{label:<13} & " + " & ".join(half_up(x, 1) for x in lengths) + " & "
                    + " & ".join(half_up(x, 2) for x in radii) + r" \\")
    L_ = [r"\begin{table}[H]", r"\centering",
          rf"\caption{{Geometrical parameters (in mm) of the reference alphorn, of the geometries obtained with the forward and backward models and of the best geometry of the training data: lengths $L_1, \dots, L_4$ of the four cones and radii $R_2$, $R_3$, $R_4$ at the junctions between them. The radii $R_1 = {R_MIN:g}$~mm and $R_5 = {R_MAX:g}$~mm and the total length are the same for all the geometries.}}",
          r"\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lccccccc}", r"\hline",
          r" & $L_1$ & $L_2$ & $L_3$ & $L_4$ & $R_2$ & $R_3$ & $R_4$ \\", r"\hline", *rows, r"\hline",
          r"\end{tabular}", r"\label{tab:geom_optimized}", r"\end{table}"]
    return "\n".join(L_) + "\n", summary


if __name__ == "__main__":
    c_ref = cents(fem_notes("white_alphorn"), NOTES).mean()
    assert abs(c_ref - C) < 0.01, f"COMMON_OFFSET = {C} differs from the FEM offset of the reference alphorn {c_ref:.3f}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_alphorns_table.tex").write_text(real_table())
    tex, summary = final_table()
    (OUT / "final_table.tex").write_text(tex)
    gtex, gsummary = geometry_table()
    (OUT / "geometry_table.tex").write_text(gtex)
    real = {}
    for title, (name, measured) in REAL.items():
        fm, ff = on_scale(measured), fem_notes(name)
        d, dm = cents(ff, NOTES), cents(fm, NOTES)
        k = ~np.isnan(dm)
        real[title] = {"fem_dev": d.tolist(), "meas_dev": dm.tolist(), "offset": float(d.mean()),
                       "fem_minus_played": (d - dm)[k].tolist(), "fem_minus_played_mean": float((d - dm)[k].mean())}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({"final": summary, "real": real, "geometry": gsummary}, indent=2))
    for k, v in summary.items():
        print(f"{k:14s}: error (RMSE of dev - {C}) {v['err']:5.2f}, offset {v['offset']:+6.2f}, RMSE {v['rmse']:6.2f}")
    for t, v in real.items():
        print(f"{t}: FEM offset {v['offset']:+.1f} cents; FEM - played: {np.round(v['fem_minus_played'], 1)} (mean {v['fem_minus_played_mean']:.1f})")
    print("written:", *(OUT / n for n in ("real_alphorns_table.tex", "final_table.tex", "geometry_table.tex")))
