from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "penalty_calibration"
)
FIGURES_DIR = RESULTS_DIR / "figures"

COEFFICIENT_SCAN_CSV = RESULTS_DIR / "heldout_coefficient_scan.csv"
PAIRED_ANALYSIS_CSV = RESULTS_DIR / "heldout_paired_analysis.csv"

CALIBRATION_STYLES = {
    "resource_aware": {
        "label": "Global",
        "color": "#4477AA",
        "marker": "o",
    },
    "resource_aware_per": {
        "label": "Per resource",
        "color": "#EE8833",
        "marker": "s",
    },
}

FIG_WIDTH = 7.2
GRID_COLOR = "#E6E6E6"
SPINE_COLOR = "#333333"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require(mapping: dict, key, description: str):
    """Fetch a grouped row, failing with a message that names what is missing."""
    if key not in mapping:
        raise KeyError(
            f"Missing {description} for {key!r}. "
            f"Available keys: {sorted(mapping)}"
        )
    return mapping[key]


def apply_common_axis_style(ax, *, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.45)
    ax.tick_params(
        axis="both",
        color=SPINE_COLOR,
        width=0.7,
        length=2.8,
        labelsize=7.5,
    )
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.7)


def apply_legend_style(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D8D8D8")
    frame.set_linewidth(0.6)
    frame.set_alpha(0.95)


def save_figure(fig, stem: str, *, tight_rect=None) -> None:
    fig.tight_layout(pad=0.45, w_pad=1.0, rect=tight_rect)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_figure_1(scan_rows: list[dict[str, str]]) -> None:
    request_counts = sorted(
        {int(row["n_requests"]) for row in scan_rows}
    )
    fig, axes = plt.subplots(1,2,figsize=(FIG_WIDTH, 2.8))

    for ax, family, title in zip(
        axes, 
        ["B", "D"],
        ["(a) Edge penalty $B$", "(b) Memory penalty $D$"],
    ):
        for calibration, style in CALIBRATION_STYLES.items():
            frequencies = []

            for n_requests in request_counts:
                group=[
                    row for row in scan_rows
                    if int(row["n_requests"]) == n_requests
                ]
                tightened = [
                    as_bool(row[f"{family}_tighter"])
                    if calibration == "resource_aware"
                    else int(row[f"{family}_n_tightened"]) >0
                    for row in group
                ]
                frequencies.append(
                    100.0 * sum(tightened) / len(group)
                )
            ax.plot(
                request_counts,
                frequencies,
                color=style["color"],
                marker=style["marker"],
                markersize=4,
                linewidth=1.1,
                label=style["label"],
            )
        ax.axhline(
            0,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            label="Conventional reference",
        )
        ax.set(
            title=title,
            xlabel="Number of requests",
            ylabel="instances tightened (%)",
        )
        ax.set_xticks(request_counts)
        ax.set_ylim(-3,105)
        ax.set_yticks([0, 25, 50, 75, 100])
        apply_common_axis_style(ax)

    apply_legend_style(axes[0].legend(loc="center right"))
    save_figure(fig, "figure1_mechanism_summary")
    plt.close(fig)

def make_figure_2(scan_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1,2,figsize=(FIG_WIDTH, 2.8))

    for ax, family, title in zip(
        axes,
        ["B", "D"],
        ["(a) Edge penalty $B$", "(b) Memory penalty $D$"],
    ):
        for calibration, style in CALIBRATION_STYLES.items():
            column = f"{family}_ratio"
            if calibration == "resource_aware_per":
                column += "_per"
            reductions = sorted(
                100.0 * (1.0 - float(row[column]))
                for row in scan_rows
            )
            cumulative = [
                100.0 * (i+1) / len(reductions)
                for i in range(len(reductions))
            ]

            ax.step(
                [reductions[0]] + reductions,
                [0.0] + cumulative,
                where="post",
                color=style["color"],
                linewidth=1.2,
                label=style["label"],
            )

        ax.axvline(
            0,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            label="Conventional reference",
        )
        ax.set(
            title=title,
            xlabel="Mean coefficient reduction (%)",
            ylabel="Cumulative instances (%)",
            xlim=(-2, 102),
            ylim=(0,102),
        )
        ax.set_xticks([0,25,50,75,100])
        ax.set_yticks([0,25,50,75,100])
        apply_common_axis_style(ax)
    apply_legend_style(axes[0].legend(loc="lower right"))
    save_figure(fig, "figure2_coefficient_ratios")
    plt.close(fig)


def make_figure_3(paired_rows: list[dict[str, str]]) -> None:
    grouped = {
        (row["calibration"], row["sampler"], row["metric"]): row
        for row in paired_rows
    }

    metrics = [
        (
            "raw_feasible_rate",
            "(a) Raw feasibility",
            "Increase in feasible calls (pp)",
            100.0,
        ),
        (
            "repaired_reference_gap_pct",
            "(b) Repaired solution quality",
            "Reduction in reference gap (pp)",
            -1.0,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.8))
    y_positions = {"sa": 1.0, "sqa": 0.0}
    offsets = {
        "resource_aware": 0.13,
        "resource_aware_per": -0.13,
    }

    for ax, (metric,title,xlabel,multiplier) in zip(axes, metrics):
        plotted_values = [0.0]

        for sampler in ["sa", "sqa"]:
            for calibration, style in CALIBRATION_STYLES.items():
                row=require(
                    grouped,
                    (calibration, sampler, metric),
                    "paired-analysis row",
                )
                estimate = (
                    multiplier * float(row["mean_paired_difference"])
                )
                ci_low, ci_high = sorted(
                    [
                        multiplier * float(row["bootstrap_95_ci_low"]),
                        multiplier * float(row["bootstrap_95_ci_high"]),
                    ]
                )
                y=y_positions[sampler] + offsets[calibration]

                ax.hlines(
                    y,
                    ci_low,
                    ci_high,
                    color=style["color"],
                    linewidth=1.2,
                )
                ax.plot(
                    estimate,
                    y,
                    marker=style["marker"],
                    color=style["color"],
                    markersize=4.5,
                    linestyle="none",
                    label=style["label"] if sampler == "sa" else None,
                )
                plotted_values.extend([ci_low, estimate, ci_high])

        padding = max(
            0.08 * (max(plotted_values) - min(plotted_values)),
            0.2,
        )
        ax.set_xlim(
            min(plotted_values)-padding,
            max(plotted_values)+padding,
        )
        ax.axvline(
            0,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            label="Conventional reference",
        )
        ax.set(title=title, xlabel=xlabel, ylim=(-0.42, 1.48))
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["SQA", "SA"])
        apply_common_axis_style(ax)
    apply_legend_style(axes[0].legend(loc="center right"))
    save_figure(fig, "figure3_solver_paired_effects")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    make_figure_1(load_csv_rows(COEFFICIENT_SCAN_CSV))
    make_figure_2(load_csv_rows(COEFFICIENT_SCAN_CSV))
    make_figure_3(load_csv_rows(PAIRED_ANALYSIS_CSV))


if __name__ == "__main__":
    main()
