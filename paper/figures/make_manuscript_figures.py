"""Generate paper/figures/fig1_workflow.pdf, fig4_solver_benchmark.pdf,
fig5_robustness.pdf and fig6_tn_diagnostics.pdf for manuscript.tex.

Data provenance
----------------
fig1 (workflow) is a schematic of the pipeline described in Methods; it does
not plot numerical data.

fig4, fig5 and fig6 plot the aggregate numbers already transcribed, verbatim,
from the collaborator's draft into manuscript.tex (Table~tab:benchmark) and
supplementary.tex (hardness sweep, Boundary-TN width ablation, true-MPS
ablation, reliability-study aggregates). The per-instance CSVs behind those
aggregates are not in this repository, so these numbers are NOT independently
reproduced here -- only re-plotted. See the PR description / chat handoff for
that caveat. The penalty-calibration figures (figure1_mechanism_summary,
figure2_coefficient_ratios, figure3_solver_paired_effects) are unaffected:
they are generated separately by
QNet_Sim/src/experiments/plot_penalty_calibration_heldout.py from data this
repository does regenerate.

Run from paper/figures/:  python make_manuscript_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import numpy as np

# ---------------------------------------------------------------------------
# Shared academic style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

# Okabe-Ito colorblind-safe categorical palette.
C = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}


def clean_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color="#CCCCCC", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.16, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
             va="bottom", ha="left")


# ---------------------------------------------------------------------------
# fig1: workflow schematic
# ---------------------------------------------------------------------------
def make_fig1():
    UNIT = 0.66  # inches per data unit; keeps box/text proportions constant
    boxes = []   # (cx, cy, w, h, text, fc, fontsize)
    arrows = []  # (p0, p1, connectionstyle)

    def add_box(cx, cy, w, h, text, fc, fontsize=7.4):
        boxes.append((cx, cy, w, h, text, fc, fontsize))
        return cx, cy, w, h

    def add_arrow(p0, p1, connectionstyle="arc3,rad=0.0"):
        arrows.append((p0, p1, connectionstyle))

    def add_elbow(p0, p1, bus_y):
        """Down-across-down connector: avoids arc3's wide bulge on long spans."""
        verts = [p0, (p0[0], bus_y), (p1[0], bus_y), p1]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        arrows.append((Path(verts, codes), None, "elbow"))

    def row_layout(y, items, gap=0.4, start=0.0, h=1.0):
        """items: list of (width, text, facecolor[, fontsize]). Returns centers."""
        centers = []
        x = start
        for it in items:
            w = it[0]
            centers.append(x + w / 2)
            x += w + gap
        for cx, it in zip(centers, items):
            w, txt, fc = it[0], it[1], it[2]
            fs = it[3] if len(it) > 3 else 7.4
            add_box(cx, y, w, h, txt, fc, fs)
        return centers, x - gap  # centers, total width used

    top_y = 5.3
    top_items = [
        (1.85, "Link and memory\nparameters", C["grey"]),
        (1.85, "Candidate paths\n($k$ per request)", "#EAF3FB"),
        (2.05, "Per-link purification\nprofiles ($q=0,1,2$)", "#EAF3FB"),
        (2.55, "Physical screening\nswap, sync., decoherence,\nstochastic fidelity/gen.", "#FDF1DE"),
        (1.95, "Bundle\n$u_b,\\,d_{eb},\\,m_{vb}$", "#E7F5EE"),
    ]
    top_centers, total_w = row_layout(top_y, top_items, gap=0.4)
    for i in range(len(top_items) - 1):
        x0 = top_centers[i] + top_items[i][0] / 2
        x1 = top_centers[i + 1] - top_items[i + 1][0] / 2
        add_arrow((x0, top_y), (x1, top_y))

    bundle_x = top_centers[-1]

    mid_y = 3.5
    solve_w, encode_w = 3.6, 2.5
    solve_x = 2.1
    add_box(solve_x, mid_y, solve_w, 1.0,
            "Constrained solve\nCP-SAT, greedy, parallel tempering,\nALNS, Boundary-TN", "#EAF3FB")
    add_box(bundle_x, mid_y, encode_w, 1.0, "QUBO / Ising\nencoding", "#FDF1DE")
    add_elbow((bundle_x, top_y - 0.49), (solve_x, mid_y + 0.5),
              bus_y=(top_y - 0.49 + mid_y + 0.5) / 2)
    add_arrow((bundle_x, top_y - 0.49), (bundle_x, mid_y + 0.5))

    cal_y = 1.75
    cal_w = 3.55
    add_box(bundle_x, cal_y, cal_w, 1.25,
            "Reachable-load penalty\ncalibration:\nglobal ($B,D$) vs.\nper-resource ($B_e,D_v$)",
            "#FBEAF2", fontsize=7.1)
    add_arrow((bundle_x, mid_y - 0.5), (bundle_x, cal_y + 0.63))

    sampler_x = bundle_x - cal_w / 2 - 0.4 - 1.15
    add_box(sampler_x, cal_y, 2.3, 1.0, "OpenJij\nSA / SQA", "#F3E9F5")
    add_arrow((bundle_x - cal_w / 2 - 0.05, cal_y), (sampler_x + 1.2, cal_y))

    eval_y = 0.5
    eval_x = 5.35
    eval_w = 7.9
    add_box(eval_x, eval_y, eval_w, 1.0,
            "Evaluation: feasibility, repaired reference gap, runtime,\nout-of-sample SLA reliability",
            "#F1F1F1")
    add_arrow((solve_x, mid_y - 0.5), (solve_x + 0.9, eval_y + 0.5), "arc3,rad=-0.15")
    add_arrow((sampler_x, cal_y - 0.5), (eval_x + 1.0, eval_y + 0.5), "arc3,rad=0.12")

    # Bounding box across every box (arrows stay within box extents).
    xs = [cx - w / 2 for cx, cy, w, h, *_ in boxes] + [cx + w / 2 for cx, cy, w, h, *_ in boxes]
    ys = [cy - h / 2 for cx, cy, w, h, *_ in boxes] + [cy + h / 2 for cx, cy, w, h, *_ in boxes]
    margin = 0.3
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin

    fig, ax = plt.subplots(figsize=((x1 - x0) * UNIT, (y1 - y0) * UNIT))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")

    for cx, cy, w, h, text, fc, fs in boxes:
        b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.0, edgecolor="#333333", facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                 color="#111111", zorder=4, linespacing=1.35)
    for p0, p1, cs in arrows:
        if cs == "elbow":
            a = FancyArrowPatch(path=p0, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color="#333333", zorder=2,
                                 shrinkA=2, shrinkB=2)
        else:
            a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color="#333333", zorder=2,
                                 connectionstyle=cs, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    fig.tight_layout(pad=0.25)
    fig.savefig("fig1_workflow.pdf")
    fig.savefig("fig1_workflow.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig4: solver benchmark (Table tab:benchmark in manuscript.tex)
# ---------------------------------------------------------------------------
BENCH = {
    "Exact\nbottleneck": {
        "Exact / near-exact": (0.00, (0.07, 8.0)),
        "Greedy-fidelity": (0.00, 0.008),
        "Boundary-TN": (5.89, None),
        "OpenJij SQA": (38.80, 7877),
    },
    "Exact\ngeometric": {
        "Exact / near-exact": (0.00, None),
        "Greedy-fidelity": (0.012, 0.008),
        "Boundary-TN": (0.00, None),
        "OpenJij SQA": (49.99, 9191),
    },
    "Medium\nbottleneck": {
        "Exact / near-exact": (0.00, None),
        "Greedy-fidelity": (0.53, 0.008),
        "OpenJij SQA": (48.93, 11544),
    },
    "Medium\nWaxman": {
        "Exact / near-exact": (0.00, None),
        "Greedy-fidelity": (0.15, 0.008),
        "OpenJij SQA": (47.03, 14978),
    },
    "Large\nbottleneck": {
        "Exact / near-exact": (0.00, None),
        "Boundary-TN": (11.87, None),
    },
    "Large\ngeometric": {
        "Exact / near-exact": (0.00, None),
        "Parallel tempering": (0.0003, None),
        "Boundary-TN": (0.60, None),
    },
}
# "Exact / near-exact" merges the table's CP-SAT/LNS/Hybrid/PT (exact and
# medium regimes) and LNS/Hybrid(/PT) (large regimes) entries, which are
# reported as tied at the same gap; parallel tempering is broken out
# separately only where its own gap differs (large-geometric).
SOLVER_COLOR = {
    "Exact / near-exact": C["green"],
    "Parallel tempering": C["sky"],
    "Greedy-fidelity": C["blue"],
    "Boundary-TN": C["vermillion"],
    "OpenJij SQA": C["purple"],
}
SOLVER_ORDER = ["Exact / near-exact", "Parallel tempering", "Greedy-fidelity",
                "Boundary-TN", "OpenJij SQA"]


def make_fig4():
    regimes = list(BENCH.keys())
    n_regimes = len(regimes)
    max_solvers = max(len(v) for v in BENCH.values())
    bar_w = 0.8 / max_solvers

    fig = plt.figure(figsize=(7.3, 3.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 2.5],
                            hspace=0.08, wspace=0.36, top=0.91, bottom=0.25,
                            left=0.11, right=0.965)
    axtop = fig.add_subplot(gs[0, 0])
    axbot = fig.add_subplot(gs[1, 0], sharex=axtop)
    axb = fig.add_subplot(gs[:, 1])

    def bars(ax, skip=()):
        for ri, regime in enumerate(regimes):
            solvers = [s for s in SOLVER_ORDER if s in BENCH[regime] and s not in skip]
            n = len(solvers)
            offsets = (np.arange(n) - (n - 1) / 2) * bar_w
            for off, solver in zip(offsets, solvers):
                gap, _ = BENCH[regime][solver]
                ax.bar(ri + off, gap, width=bar_w * 0.92, color=SOLVER_COLOR[solver],
                       edgecolor="white", linewidth=0.3, zorder=3)

    # Broken y-axis: axtop shows the OpenJij SQA outliers, axbot the near-zero
    # classical/greedy/Boundary-TN gaps, at a scale where they are legible.
    bars(axtop)
    bars(axbot)
    axtop.set_ylim(34, 55)
    axbot.set_ylim(0, 13.5)
    axtop.set_yticks([35, 45, 55])
    axbot.set_yticks([0, 5, 10])
    clean_axes(axtop)
    clean_axes(axbot)
    axtop.spines["bottom"].set_visible(False)
    axbot.spines["top"].set_visible(False)
    axtop.tick_params(axis="x", bottom=False, labelbottom=False)
    axbot.set_xticks(range(n_regimes))
    axbot.set_xticklabels(regimes, fontsize=6.7)
    fig.text(0.010, 0.60, "Mean gap to reference (%)", rotation=90,
              va="center", ha="left", fontsize=9)
    panel_label(axtop, "a", x=-0.16, y=1.12)

    # Diagonal break marks between the two stacked axes.
    d = 0.55
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
                   color="#333333", mec="#333333", mew=0.9, clip_on=False)
    axtop.plot([0], [0], transform=axtop.transAxes, **kwargs)
    axbot.plot([0], [1], transform=axbot.transAxes, **kwargs)

    # Panel b: median wall time (log scale), only where reported. The one
    # range (CP-SAT/LNS/Hybrid/PT, exact-bottleneck) is a plain range bar,
    # not a bar rooted at zero, so it cannot be mistaken for a point value.
    for ri, regime in enumerate(regimes):
        solvers = [s for s in SOLVER_ORDER if s in BENCH[regime]]
        n = len(solvers)
        offsets = (np.arange(n) - (n - 1) / 2) * bar_w
        for off, solver in zip(offsets, solvers):
            _, t = BENCH[regime][solver]
            if t is None:
                continue
            color = SOLVER_COLOR[solver]
            if isinstance(t, tuple):
                lo, hi = t
                axb.plot([ri + off, ri + off], [lo, hi], color=color, linewidth=4,
                          solid_capstyle="butt", zorder=3)
                axb.plot([ri + off] * 2, [lo, hi], marker="_", color=color,
                          markersize=9, markeredgewidth=1.6, zorder=4)
            else:
                axb.bar(ri + off, t, width=bar_w * 0.92, color=color,
                         edgecolor="white", linewidth=0.3, zorder=3)
    axb.set_yscale("log")
    axb.set_xticks(range(n_regimes))
    axb.set_xticklabels(regimes, fontsize=6.7)
    axb.set_ylabel("Median wall time (s)")
    axb.set_ylim(3e-3, 3e4)
    clean_axes(axb)
    panel_label(axb, "b")

    handles = [mpatches.Patch(color=SOLVER_COLOR[s], label=s) for s in SOLVER_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.53, 0.015),
               handlelength=1.1, columnspacing=1.1)

    fig.savefig("fig4_solver_benchmark.pdf")
    fig.savefig("fig4_solver_benchmark.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig5: robustness / chance-constrained policy comparison
# (supplementary.tex, "Reliability-study aggregates")
# ---------------------------------------------------------------------------
POLICIES = ["Nominal", "Robust", "Chance"]
POLICY_COLOR = {"Nominal": C["orange"], "Robust": C["sky"], "Chance": C["green"]}
UTILITY = {"Nominal": 124.254, "Robust": 114.514, "Chance": 115.643}
VIOLATION = {"Nominal": 0.1627, "Robust": 0.1212, "Chance": 0.0602}
FIDELITY = {"Nominal": 0.8600, "Robust": 0.8626, "Chance": 0.8634}


def make_fig5():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.4, 2.9))

    x = np.arange(len(POLICIES))
    colors = [POLICY_COLOR[p] for p in POLICIES]
    bars = axa.bar(x, [UTILITY[p] for p in POLICIES], width=0.56, color=colors,
                     edgecolor="white", linewidth=0.4, zorder=3)
    for xi, p in zip(x, POLICIES):
        axa.text(xi, UTILITY[p] + 1.8, f"{UTILITY[p]:.1f}", ha="center", fontsize=7.5)
    axa.set_xticks(x)
    axa.set_xticklabels(POLICIES)
    axa.set_ylabel("Mean optimization utility")
    axa.set_ylim(0, 140)
    clean_axes(axa)
    panel_label(axa, "a")

    bars2 = axb.bar(x, [100 * VIOLATION[p] for p in POLICIES], width=0.56, color=colors,
                      edgecolor="white", linewidth=0.4, zorder=3)
    for xi, p in zip(x, POLICIES):
        axb.text(xi, 100 * VIOLATION[p] + 0.4, f"{100*VIOLATION[p]:.1f}%", ha="center",
                   fontsize=7.5)
    axb.set_xticks(x)
    axb.set_xticklabels(POLICIES)
    axb.set_ylabel("Out-of-sample SLA violation (%)")
    axb.set_ylim(0, 20)
    clean_axes(axb)
    panel_label(axb, "b")

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig("fig5_robustness.pdf")
    fig.savefig("fig5_robustness.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig6: tensor-network diagnostics
# (supplementary.tex, Boundary-TN width ablation & true-MPS ablation)
# ---------------------------------------------------------------------------
CHI_TN = [8, 16, 32, 64, 128, 256]
GAP_TN = [13.188, 13.130, 9.114, 8.128, 7.793, 4.998]
TIME_TN = [0.326, 0.848, 1.717, 2.906, 5.380, 11.270]

CHI_MPS = [2, 4, 8, 16, 32, 64]
MPS_GAP = {
    1: [19.65, 21.32, 21.53, 17.64, 21.44, 21.37],
    2: [14.78, 18.72, 21.72, 17.64, 21.44, 21.37],
    4: [11.63, 16.58, 21.47, 17.64, 21.37, 21.68],
}
SWEEP_COLOR = {1: C["sky"], 2: C["blue"], 4: C["black"]}


def make_fig6():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.55))
    fig.subplots_adjust(left=0.09, right=0.90, top=0.93, bottom=0.30, wspace=0.55)

    axa.plot(CHI_TN, GAP_TN, marker="o", color=C["vermillion"], linewidth=1.4,
              markersize=4.5, zorder=3, label="mean gap to LNS")
    axa.set_xscale("log", base=2)
    axa.set_xticks(CHI_TN)
    axa.set_xticklabels([str(c) for c in CHI_TN])
    axa.set_xlabel("Retained boundary width $\\chi$")
    axa.set_ylabel("Mean gap to LNS (%)", color=C["vermillion"])
    axa.tick_params(axis="y", labelcolor=C["vermillion"])
    axa.set_ylim(0, 16)
    clean_axes(axa)
    panel_label(axa, "a")

    axa2 = axa.twinx()
    axa2.plot(CHI_TN, TIME_TN, marker="s", color=C["blue"], linewidth=1.2,
               markersize=4, linestyle="--", zorder=3, label="median runtime")
    axa2.set_ylabel("Median runtime (s)", color=C["blue"])
    axa2.tick_params(axis="y", labelcolor=C["blue"])
    axa2.spines["top"].set_visible(False)
    axa2.set_ylim(0, 13)

    lines1, labels1 = axa.get_legend_handles_labels()
    lines2, labels2 = axa2.get_legend_handles_labels()
    axa.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
                bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.6)

    for sweeps in (1, 2, 4):
        axb.plot(CHI_MPS, MPS_GAP[sweeps], marker="o", markersize=4, linewidth=1.3,
                  color=SWEEP_COLOR[sweeps], label=f"{sweeps} sweep" + ("s" if sweeps > 1 else ""))
    axb.set_xscale("log", base=2)
    axb.set_xticks(CHI_MPS)
    axb.set_xticklabels([str(c) for c in CHI_MPS])
    axb.set_xlabel("Bond dimension $\\chi$")
    axb.set_ylabel("True-MPS repaired gap (%)")
    axb.set_ylim(0, 24)
    clean_axes(axb)
    panel_label(axb, "b")
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.6)

    fig.savefig("fig6_tn_diagnostics.pdf")
    fig.savefig("fig6_tn_diagnostics.png")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig4()
    make_fig5()
    make_fig6()
    print("Wrote fig1_workflow, fig4_solver_benchmark, fig5_robustness, "
          "fig6_tn_diagnostics (.pdf + .png) into", ".")
