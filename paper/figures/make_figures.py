"""Regenerate every figure in qiopt4qnet_npj_final.tex from SOURCE_DATA/.

    fig1_workflow            schematic (no data)
    fig2_solver_benchmark    SOURCE_DATA/journal_benchmark/regime_summary_all.csv
    fig3_tn_diagnostics      SOURCE_DATA/tn/tn_ablation.csv, true_mps_ablation.csv
    fig4_robustness          SOURCE_DATA/robustness/robustness_runs.csv
    fig5_coefficient_ratios  SOURCE_DATA/calibration/coefficient_scan.csv
    fig6_solver_paired_effects
                             SOURCE_DATA/calibration/paired_effects.csv,
                             global_changed_effects.csv,
                             sampler_check_effects.csv

The visual style follows the figures of Shirakawa et al., arXiv:2511.00224:
Computer Modern text, closed frames with inward ticks on all four sides,
"(a)" labels inside the panels, frameless legends with the marker after the
text, dashed light-blue reference lines, and muted maroon/orange/olive/blue
data colours.

Run from paper/figures/:  python make_figures.py [fig6 ...]

With no arguments every figure is rebuilt. The repository keeps only the
calibration source data, so there run ``python make_figures.py fig6``.
SOURCE_DATA/calibration/sampler_check_effects.csv is a copy of
QNet_Sim/results/penalty_calibration/sampler_check_summary.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "SOURCE_DATA"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus glyph
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

C = {
    "maroon": "#8B3A3A",
    "orange": "#F08A1C",
    "olive": "#8A9A2B",
    "blue": "#1C3FB8",
    "lightblue": "#5B9BD5",
    "purple": "#A51FD6",
    "grey": "#7A7A7A",
    "black": "#000000",
}
REF_LINE = dict(color=C["lightblue"], linestyle=(0, (3, 2)), linewidth=1.1)


def boxed(ax, minor_x=True, minor_y=True):
    """Closed frame with inward major/minor ticks on all four sides."""
    if minor_x:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    if minor_y:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def panel_label(ax, letter, x=0.03, y=0.96):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=13,
            va="top", ha="left", zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 1: workflow
# ---------------------------------------------------------------------------
BOX = {  # pastel fill, darker edge of the same hue
    "grey": ("#F2F2F2", "#9A9A9A"),
    "blue": ("#D3E3F1", "#7EA3C6"),
    "yellow": ("#FFF3C9", "#D9A93A"),
    "green": ("#DDF1E7", "#3E9C7A"),
    "pink": ("#F6DEDA", "#C58B82"),
    "lavender": ("#E7E3F3", "#8A83B8"),
}
ARROW = {"green": "#3E9C7A", "yellow": "#C9982A", "lavender": "#7E77AE"}


def make_fig1():
    boxes, arrows, frames = [], [], []

    def box(cx, cy, w, h, title, body, style):
        boxes.append((cx, cy, w, h, title, body, style))
        return dict(cx=cx, cy=cy, w=w, h=h, l=cx - w / 2, r=cx + w / 2,
                    t=cy + h / 2, b=cy - h / 2)

    def arrow(points, color):
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1)
        arrows.append((MPath(points, codes), color))

    # Row 1: physical screening pipeline.
    y1, h1 = 5.35, 1.05
    w1, gap = 2.75, 0.5
    xs = [w1 / 2 + i * (w1 + gap) for i in range(4)]
    inst = box(xs[0], y1, w1, h1, "Network instance",
               "loss, raw fidelity,\nedge capacity, memory", "grey")
    gen = box(xs[1], y1, w1, h1, "Bundle generation",
              "path + per-link\npurification depth $q_e$", "blue")
    scr = box(xs[2], y1, w1, h1, "Physical screening",
              "swap success, storage\ndecoherence, SLA", "yellow")
    tab = box(xs[3], y1, w1, h1, "Bundle table",
              "$u_b,\\ d_{eb},\\ m_{vb}$", "green")
    for a, b in ((inst, gen), (gen, scr), (scr, tab)):
        arrow([(a["r"], y1), (b["l"], y1)], ARROW["green"])
    right = tab["r"]
    frames.append((-0.25, y1 - h1 / 2 - 0.25, right + 0.25, y1 + h1 / 2 + 0.5,
                   "Physics layer", "#8FC7B0", "#F6FBF8", "#2F7A5F"))

    # Row 2: the allocation problem.
    y2 = 3.55
    alloc = box(right / 2, y2, 4.4, 1.0, "Constrained allocation",
                "at most one bundle per request;\nedge and memory capacities", "green")
    arrow([(tab["cx"], tab["b"] - 0.25), (tab["cx"], y2), (alloc["r"], y2)],
          ARROW["green"])

    # Row 3: the two solution routes.
    y3 = 1.75
    solve = box(2.15, y3, 4.3, 1.2, "Constrained solvers",
                "CP-SAT, greedy, parallel tempering,\nALNS, Boundary-TN, TN+ALNS", "green")
    qubo = box(7.05, y3, 2.5, 1.2, "QUBO encoding",
               "$A$: request conflicts\n$B_e,\\,D_v$: capacities", "yellow")
    cal = box(10.6, y3, 3.7, 1.2, "Per-resource calibration",
              "$B_e=\\mathrm{max}_b\\, u_b^+/\\Delta_{e,b}+\\varepsilon$\n$D_v$ likewise", "pink")
    arrow([(alloc["cx"] - 1.2, alloc["b"]), (alloc["cx"] - 1.2, y2 - 0.9),
           (solve["cx"], y2 - 0.9), (solve["cx"], solve["t"])], ARROW["green"])
    arrow([(alloc["cx"] + 1.2, alloc["b"]), (alloc["cx"] + 1.2, y2 - 0.9),
           (qubo["cx"], y2 - 0.9), (qubo["cx"], qubo["t"])], ARROW["yellow"])
    arrow([(qubo["r"], y3), (cal["l"], y3)], ARROW["yellow"])

    # Row 4: sampling and evaluation.
    y4 = 0.0
    samp = box(cal["cx"], y4, 2.7, 1.0, "OpenJij sampling", "SA and SQA", "lavender")
    ev = box(2.7, y4, 5.4, 1.0, "Evaluation",
             "reference gap, feasibility, runtime,\nout-of-sample SLA violations", "blue")
    arrow([(cal["cx"], cal["b"]), (cal["cx"], samp["t"])], ARROW["lavender"])
    arrow([(samp["l"], y4), (ev["r"], y4)], ARROW["lavender"])
    arrow([(solve["cx"], solve["b"]), (solve["cx"], ev["t"])], ARROW["green"])
    frames.append((qubo["l"] - 0.2, samp["b"] - 0.2, cal["r"] + 0.2, y3 + 1.2 / 2 + 0.5,
                   "Annealing route", "#B5AED6", "#FAF9FD", "#5E5794"))

    x0, x1 = -0.35, max(right, cal["r"]) + 0.45
    yb0, yb1 = y4 - 0.75, y1 + h1 / 2 + 0.6
    unit = 0.555
    fig, ax = plt.subplots(figsize=((x1 - x0) * unit, (yb1 - yb0) * unit))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(yb0, yb1)
    ax.axis("off")

    for fx0, fy0, fx1, fy1, title, ec, fc, tc in frames:
        ax.add_patch(FancyBboxPatch((fx0, fy0), fx1 - fx0, fy1 - fy0,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=0))
        ax.text(fx1 - 0.15, fy1 - 0.08, title, ha="right", va="top", fontsize=8.5,
                style="italic", color=tc, zorder=1)
    for cx, cy, w, h, title, body, st in boxes:
        fc, ec = BOX[st]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(cx, cy + h / 2 - 0.13, title, ha="center", va="top", fontsize=9,
                fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.13, body, ha="center", va="center", fontsize=8,
                linespacing=1.3, zorder=4)
    for path, color in arrows:
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.3, color=color, zorder=2,
                                     shrinkA=0, shrinkB=0.5))
    save(fig, "fig1_workflow")


# ---------------------------------------------------------------------------
# Fig. 2: solver benchmark
# ---------------------------------------------------------------------------
REGIMES = ["exact_bottleneck", "exact_geometric", "medium_bottleneck",
           "medium_waxman", "large_bottleneck", "large_geometric"]
REGIME_LABEL = {r: r.replace("_", " ").capitalize().replace("waxman", "Waxman")
                for r in REGIMES}
SOLVERS = [  # csv name, label, colour, marker, filled
    ("CP-SAT", "CP-SAT", C["black"], "o", False),
    ("AdaptiveLNS-CP-SAT", "ALNS", C["olive"], "P", True),
    ("HybridBoundaryTN+ALNS", "TN+ALNS hybrid", C["lightblue"], "v", True),
    ("ParallelTempering", "Parallel tempering", C["blue"], "^", True),
    ("Greedy-fidelity", "Greedy (fidelity)", C["purple"], "s", False),
    ("Greedy-scarcity", "Greedy (scarcity)", C["grey"], "x", True),
    ("StochasticBoundaryTN", "Boundary-TN", C["orange"], "o", True),
    ("OpenJij-SQA-calibrated-ensemble", "OpenJij SQA", C["maroon"], "D", True),
]


def solver_marker_kw(color, marker, filled, size=5.0):
    kw = dict(marker=marker, markersize=size, linestyle="none", color=color)
    if marker == "x":
        kw.update(mew=1.2)
    elif filled:
        kw.update(mec="black", mew=0.4)
    else:
        kw.update(mfc="white", mec=color, mew=1.1)
    return kw


def make_fig2():
    df = pd.read_csv(DATA / "journal_benchmark" / "regime_summary_all.csv")
    cell = {(r.regime, r.solver): r for r in df.itertuples()}
    n = len(SOLVERS)
    offsets = (np.arange(n) - (n - 1) / 2) * 0.1

    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.2], hspace=0.06, wspace=0.27,
                          top=0.975, bottom=0.305, left=0.085, right=0.985)
    axt = fig.add_subplot(gs[0, 0])
    axg = fig.add_subplot(gs[1, 0], sharex=axt)
    axw = fig.add_subplot(gs[:, 1])

    for i, (name, label, color, marker, filled) in enumerate(SOLVERS):
        xs, gaps, times = [], [], []
        for ri, regime in enumerate(REGIMES):
            row = cell.get((regime, name))
            if row is None:
                continue
            xs.append(ri + offsets[i])
            gaps.append(row.mean_gap_to_reference_pct)
            times.append(row.median_wall_time_s)
        kw = solver_marker_kw(color, marker, filled)
        for ax in (axt, axg):
            ax.plot(xs, gaps, zorder=4, clip_on=False, **kw)
        axw.plot(xs, times, color=color, linewidth=0.6, alpha=0.6, zorder=2)
        axw.plot(xs, times, zorder=4, **kw)

    for ri in range(1, len(REGIMES)):
        for ax in (axt, axg, axw):
            ax.axvline(ri - 0.5, color="0.85", linewidth=0.5, zorder=0)

    # Broken y axis: SQA sits at 39-50 %, everything else below 12 %.
    axt.set_ylim(34, 54)
    axg.set_ylim(-0.4, 13.5)
    axt.set_yticks([40, 50])
    axg.set_yticks([0, 5, 10])
    for ax in (axt, axg):
        boxed(ax, minor_x=False)
        ax.set_xlim(-0.5, len(REGIMES) - 0.5)
    axt.spines["bottom"].set_visible(False)
    axg.spines["top"].set_visible(False)
    axt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    axg.tick_params(axis="x", which="both", top=False, bottom=False)
    d = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
             color="black", mec="black", mew=0.8, clip_on=False)
    axt.plot([0, 1], [0, 0], transform=axt.transAxes, **d)
    axg.plot([0, 1], [1, 1], transform=axg.transAxes, **d)
    fig.text(0.012, 0.64, "Mean gap to reference (%)", rotation=90, va="center",
             fontsize=10)
    panel_label(axt, "a", y=0.9)

    axw.set_yscale("log")
    axw.set_ylim(2e-4, 1e5)
    axw.set_xlim(-0.5, len(REGIMES) - 0.5)
    axw.set_ylabel("Median wall time (s)")
    axw.tick_params(which="both", direction="in", top=False, right=True)
    axw.tick_params(axis="x", which="both", bottom=False)
    panel_label(axw, "b")

    labels = [REGIME_LABEL[r] for r in REGIMES]
    for ax in (axg, axw):
        ax.set_xticks(range(len(REGIMES)))
        ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor",
                           fontsize=8.5)

    handles = [mlines.Line2D([], [], label=label,
                             **solver_marker_kw(color, marker, filled, 5.5))
               for _, label, color, marker, filled in SOLVERS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.53, 0.0),
               markerfirst=False, columnspacing=2.2, fontsize=8.5)
    save(fig, "fig2_solver_benchmark")


# ---------------------------------------------------------------------------
# Fig. 3: tensor-network diagnostics
# ---------------------------------------------------------------------------
def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def make_fig3():
    tn = pd.read_csv(DATA / "tn" / "tn_ablation.csv")
    mps = pd.read_csv(DATA / "tn" / "true_mps_ablation.csv")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.16, wspace=0.42)

    # (a) Boundary-TN width sweep: per-seed gaps (faint), seed mean, median runtime.
    agg = tn.groupby("chi").agg(gap=("gap_to_lns_pct", "mean"),
                                time=("wall_time_s", "median")).reset_index()
    chis = agg.chi.tolist()
    for _, s in tn.groupby("seed"):
        s = s.sort_values("chi")
        axa.plot(s.chi, s.gap_to_lns_pct, color=C["maroon"], alpha=0.18,
                 linewidth=0.7, zorder=1)
    axa.plot(agg.chi, agg.gap, marker="D", color=C["maroon"], mec="black", mew=0.4,
             zorder=3, label="mean gap to ALNS")
    axa2 = axa.twinx()
    axa2.plot(agg.chi, agg.time, marker="s", color=C["orange"], mec="black", mew=0.4,
              linestyle="--", zorder=3, label="median runtime")
    log2_axis(axa, chis)
    axa.set_xlabel(r"Retained boundary width $\chi$")
    axa.set_ylabel("Gap to ALNS (%)", color=C["maroon"])
    axa.set_ylim(0, 42)
    axa.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    axa.tick_params(which="both", direction="in", top=True, right=False)
    axa.tick_params(axis="y", which="both", colors=C["maroon"])
    axa2.set_ylabel("Median runtime (s)", color=C["orange"])
    axa2.set_ylim(0, 16.5)
    axa2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    axa2.tick_params(axis="y", which="both", direction="in", colors=C["orange"])
    axa2.spines["left"].set_color(C["maroon"])
    axa2.spines["right"].set_color(C["orange"])
    panel_label(axa, "a")
    h1, l1 = axa.get_legend_handles_labels()
    h2, l2 = axa2.get_legend_handles_labels()
    axa.legend(h1 + h2, l1 + l2, loc="upper right", markerfirst=False, handlelength=2.0)

    # (b) true-MPS ablation: mean +/- s.e.m. over seeds, greedy baseline dashed.
    colors = {1: C["orange"], 2: C["maroon"], 4: C["blue"]}
    markers = {1: "s", 2: "D", 4: "^"}
    shift = {1: 0.93, 2: 1.0, 4: 1.075}
    for sweeps, g in mps.groupby("sweeps"):
        a = g.groupby("chi").repaired_gap_pct.agg(["mean", "sem"]).reset_index()
        axb.errorbar(a.chi * shift[sweeps], a["mean"], yerr=a["sem"], color=colors[sweeps],
                     marker=markers[sweeps], mec="black", mew=0.4, capsize=2, elinewidth=0.8,
                     label=f"{sweeps} sweep" + ("s" if sweeps > 1 else ""))
    greedy_gap = (100 * (1 - mps.greedy_utility / mps.reference_utility)).mean()
    axb.axhline(greedy_gap, **REF_LINE)
    axb.text(0.97, greedy_gap + 0.5, f"greedy ({greedy_gap:.2f}\\%)".replace("\\", ""),
             color=C["lightblue"], ha="right", va="bottom", fontsize=9,
             transform=axb.get_yaxis_transform())
    log2_axis(axb, sorted(mps.chi.unique()))
    axb.set_xlabel(r"Bond dimension $\chi$")
    axb.set_ylabel("Repaired MPS gap (%)")
    axb.set_ylim(-1, 37)
    boxed(axb, minor_x=False)
    panel_label(axb, "b")
    axb.legend(loc="upper right", ncol=3, markerfirst=False, handlelength=1.6,
               columnspacing=1.0)
    save(fig, "fig3_tn_diagnostics")


# ---------------------------------------------------------------------------
# Fig. 4: reliability study
# ---------------------------------------------------------------------------
POLICIES = [("nominal", "Nominal", C["orange"], "s"),
            ("robust", "Risk-\nweighted", C["blue"], "^"),
            ("chance", "Chance-\nconstrained", C["maroon"], "D")]


def make_fig4():
    df = pd.read_csv(DATA / "robustness" / "robustness_runs.csv")
    df["sla_violation_pct"] = 100 * df.sla_violation_rate
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.975, bottom=0.15, wspace=0.3)
    x = np.arange(len(POLICIES))
    for ax, col, ylabel, fmt in (
        (axes[0], "optimization_utility", "Optimization utility", "{:.1f}"),
        (axes[1], "sla_violation_pct", "Out-of-sample SLA violation (%)", "{:.1f}%"),
    ):
        wide = df.pivot(index="seed", columns="policy", values=col)[[p[0] for p in POLICIES]]
        for _, row in wide.iterrows():  # one grey line per seed: the design is paired
            ax.plot(x, row.values, color="0.8", linewidth=0.6, zorder=1)
        for xi, (key, label, color, marker) in enumerate(POLICIES):
            ax.plot(np.full(len(wide), xi), wide[key], marker=marker, linestyle="none",
                    color=color, alpha=0.45, markersize=3.5, mew=0, zorder=2)
            m, se = wide[key].mean(), wide[key].sem()
            ax.errorbar(xi + 0.16, m, yerr=se, marker=marker, color=color, mec="black",
                        mew=0.4, markersize=6.5, capsize=2.5, elinewidth=1.0, zorder=3)
            ax.text(xi + 0.27, m, fmt.format(m), color=color, va="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([p[1] for p in POLICIES], fontsize=8.5)
        ax.set_xlim(-0.4, len(POLICIES) - 0.25)
        ax.set_ylabel(ylabel)
        boxed(ax, minor_x=False)
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
    axes[0].set_ylim(0, 250)
    axes[1].set_ylim(0, 37)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "fig4_reliability")


# ---------------------------------------------------------------------------
# Fig. 5: coefficient reductions
# ---------------------------------------------------------------------------
def ecdf(values):
    v = np.sort(np.asarray(values))
    y = 100 * np.arange(1, len(v) + 1) / len(v)
    return np.r_[v[0], v], np.r_[0, y]


def make_fig5():
    df = pd.read_csv(DATA / "calibration" / "coefficient_scan.csv")
    load_colors = {8: "#F6B26B", 16: C["orange"], 24: "#9C4A00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.975, bottom=0.16, wspace=0.08)
    for ax, fam, letter, title in zip(axes, "BD", "ab", ["Edge penalty $B$", "Memory penalty $D$"]):
        x, y = ecdf(100 * (1 - df[f"{fam}_ratio"]))
        ax.step(x, y, where="post", color=C["maroon"], linewidth=1.5, label="Global, all loads")
        x, y = ecdf(100 * (1 - df[f"{fam}_ratio_per"]))
        ax.step(x, y, where="post", color="black", linewidth=1.5, label="Per resource, all loads")
        for n_req, g in df.groupby("n_requests"):
            x, y = ecdf(100 * (1 - g[f"{fam}_ratio_per"]))
            ax.step(x, y, where="post", color=load_colors[n_req], linewidth=0.9,
                    linestyle=(0, (4, 1.5)), label=f"Per resource, {n_req} requests")
        ax.axvline(0, **REF_LINE)
        ax.set_xlim(-3, 102)
        ax.set_ylim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Mean coefficient reduction (%)")
        boxed(ax)
        panel_label(ax, letter)
        ax.text(0.15, 0.945, title, transform=ax.transAxes, va="top", fontsize=9.5)
    axes[0].set_ylabel("Cumulative instances (%)")
    axes[1].legend(loc="lower right", markerfirst=False, handlelength=2.2, fontsize=7.8)
    save(fig, "fig5_coefficient_ratios")


# ---------------------------------------------------------------------------
# Fig. 6: paired calibration effects
# ---------------------------------------------------------------------------
CONTRASTS = [  # (file, treatment, baseline), label, colour, marker, filled, y offset
    (("paired", "resource_aware", "conventional"), "Global vs conventional",
     C["maroon"], "D", True, 0.3),
    (("changed", "resource_aware", "conventional"), "Global vs conventional (127 changed)",
     C["maroon"], "D", False, 0.1),
    (("paired", "resource_aware_per", "conventional"), "Per resource vs conventional",
     C["orange"], "s", True, -0.1),
    (("paired", "resource_aware_per", "resource_aware"), "Per resource vs global",
     C["blue"], "^", True, -0.3),
]


SAMPLER_CHECK_CONTRASTS = [  # (treatment, baseline), label, colour, marker, y offset
    (("resource_aware", "conventional"), "Global vs conventional", C["maroon"], "D", 0.32),
    (("resource_aware_per", "conventional"), "Per resource vs conventional",
     C["orange"], "s", 0.16),
    (("resource_aware_per", "resource_aware"), "Per resource vs global", C["blue"], "^", 0.0),
    (("local_utility", "conventional"), "Local utility vs conventional",
     C["olive"], "o", -0.16),
    (("resource_aware_per", "local_utility"), "Per resource vs local utility",
     C["purple"], "v", -0.32),
]


def effect_point(ax, row, mult, y, color, marker, filled, label):
    est = mult * row.mean_paired_difference
    lo, hi = sorted([mult * row.bootstrap_95_ci_low, mult * row.bootstrap_95_ci_high])
    ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="none",
                ecolor=color, elinewidth=1.1, capsize=2.5, capthick=1.0)
    kw = dict(mec="black", mew=0.4) if filled else dict(mfc="white", mec=color, mew=1.1)
    ax.plot(est, y, marker=marker, color=color, markersize=5.5, linestyle="none",
            label=label, **kw)


def effects_axis(ax, letter, title, xlabel, xlim, step):
    ax.axvline(0, **REF_LINE)
    ax.axhline(0.5, color="0.8", linewidth=0.5)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SQA", "SA"])
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.set_xlabel(xlabel)
    boxed(ax, minor_y=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    panel_label(ax, letter)
    ax.text(0.15, 0.945, title, transform=ax.transAxes, va="top", fontsize=9.5,
            zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))


def make_fig6():
    calib = DATA / "calibration"
    files = {"paired": pd.read_csv(calib / "paired_effects.csv"),
             "changed": pd.read_csv(calib / "global_changed_effects.csv")}
    check = pd.read_csv(calib / "sampler_check_effects.csv")
    xlabels = {"feas": "Increase in feasible reads (pp)",
               "gap": "Reduction in repaired gap (pp)"}
    ypos = {"sa": 1.0, "sqa": 0.0}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.985, bottom=0.165,
                        wspace=0.12, hspace=0.42)

    # (a, b): fixed-seed OpenJij study, five solver seeds per instance.
    top = [("raw_feasible_rate", 100.0, "feas", "Fixed-seed study: feasibility"),
           ("repaired_reference_gap_pct", -1.0, "gap", "Fixed-seed study: repaired gap")]
    for ax, letter, (metric, mult, xl, title) in zip(axes[0], "ab", top):
        for (src, trt, base), label, color, marker, filled, off in CONTRASTS:
            d = files[src]
            for sampler, y0 in ypos.items():
                r = d[(d.calibration == trt) & (d.baseline == base) & (d.sampler == sampler)
                      & (d.metric == metric)].iloc[0]
                effect_point(ax, r, mult, y0 + off, color, marker, filled,
                             label if sampler == "sa" else None)
        effects_axis(ax, letter, title, xlabels[xl], (-1.5, 15), 2)

    # (c, d): sampler-configuration check with the scope ablation.
    bottom = [("raw_feasible_rate", 100.0, "feas", "Sampler check: feasibility",
               (-1.0, 10.5), 1),
              ("repaired_optimality_gap_pct", -1.0, "gap", "Sampler check: repaired gap",
               (-0.6, 2.6), 0.5)]
    for ax, letter, (metric, mult, xl, title, xlim, step) in zip(axes[1], "cd", bottom):
        for (trt, base), label, color, marker, off in SAMPLER_CHECK_CONTRASTS:
            for sampler, y0 in ypos.items():
                r = check[(check.treatment == trt) & (check.baseline == base)
                          & (check.sampler == sampler) & (check.metric == metric)].iloc[0]
                effect_point(ax, r, mult, y0 + off, color, marker, True,
                             label if sampler == "sa" else None)
        effects_axis(ax, letter, title, xlabels[xl], xlim, step)

    for row in axes:
        row[1].set_yticklabels([])
    handles, labels = {}, []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    fig.legend([handles[lab] for lab in labels], labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.53, 0.0), markerfirst=False, columnspacing=2.0,
               fontsize=8.5)
    save(fig, "fig6_solver_paired_effects")


if __name__ == "__main__":
    import sys

    makers = {f"fig{i}": f for i, f in enumerate(
        [make_fig1, make_fig2, make_fig3, make_fig4, make_fig5, make_fig6], start=1)}
    names = sys.argv[1:] or list(makers)
    for name in names:
        makers[name]()
    print(f"wrote {', '.join(names)} (.pdf and .png) to", HERE)
