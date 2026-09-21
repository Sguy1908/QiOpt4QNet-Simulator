"""Paired analysis for the held-out penalty-calibration experiment.

This script compares conventional and resource-aware calibration on matched
(instance, solver-seed) runs from the completed held-out experiment.

Run from QNet_Sim:

    PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_heldout.py
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import statistics
from collections import defaultdict
from experiments.run_penalty_calibration_heldout import NUM_READS, SOLVER_SEEDS, TIGHTENING_TOL, summarize_group

BOOTSTRAP_SEED = 20260811
BOOTSTRAP_SAMPLES = 10000
TIE_TOL = 1e-12
CALIBRATIONS = ["conventional", "resource_aware", "resource_aware_per"]
SAMPLERS = ["sa", "sqa"]
# (treatment, baseline). The third contrast isolates the per-resource
# contribution relative to the global reachable-load rule, which is the
# comparison the per-resource proposal actually needs to justify itself.
CONTRASTS = [
    ("resource_aware", "conventional"),
    ("resource_aware_per", "conventional"),
    ("resource_aware_per", "resource_aware"),
]
METRICS = [("raw_feasible_rate", "raw_feasible_rate", False), ("repaired_reference_gap_pct", "repaired_optimality_gap_pct", True), ("raw_mean_overload_units", "raw_mean_overload_units", True)]


def contrast_seed(treatment, baseline, sampler, metric):
    """Bootstrap seed fixed by the comparison itself, not by loop order.

    Using an explicit digest rather than ``hash()`` keeps the seed stable
    across processes: Python randomises string hashing per interpreter run.
    """
    key = "|".join([treatment, baseline, sampler, metric]).encode()
    return BOOTSTRAP_SEED + int(hashlib.sha256(key).hexdigest()[:8], 16)

def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(blocks, seed):
    """Percentile bootstrap CI for the mean paired difference.

    Resampling is over instance-seed blocks rather than individual instances,
    so instances generated from the same held-out seed stay together and the
    interval does not assume more independence than the design provides. The
    statistic is the ratio-of-sums across resampled blocks, which equals the
    overall mean paired difference. 10,000 resamples, 2.5/97.5 percentiles.
    """
    rng = random.Random(seed)
    totals=[(sum(values), len(values)) for values in blocks]
    bootstrap_means = []

    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            totals[rng.randrange(len(totals))]
            for _ in totals
        ]
        bootstrap_means.append(sum(total for total, _ in sample) / sum(n for _, n in sample))

    bootstrap_means.sort()

    return (
        bootstrap_means[int(0.025 * BOOTSTRAP_SAMPLES)],
        bootstrap_means[int(0.975 * BOOTSTRAP_SAMPLES) - 1],
    )


def analyze_rows(rows, contrasts=None, restrict_to=None):
    """Paired summary rows, one per (contrast, sampler, metric).

    ``restrict_to`` is an optional set of ``(topology, instance_seed,
    n_requests)`` keys; when given, only those instances are analysed.
    """
    contrasts = CONTRASTS if contrasts is None else contrasts
    matched = defaultdict(dict)

    for row in rows:
        key=(
            row["topology"],
            int(row["instance_seed"]),
            int(row["n_requests"]),
            row["sampler"],
            int(row["solver_seed"]),
        )
        calibration = row["calibration"]
        if calibration in matched[key]:
            raise ValueError(
                f"Duplicate calibration row: {key}, {calibration}"
            )
        matched[key][calibration] = row
    instances = defaultdict(dict)
    for key,group in sorted(matched.items()):
        if set(group) != set(CALIBRATIONS):
            raise ValueError(f"Incomplete calibration group: {key}")

        if key[-1] not in SOLVER_SEEDS or key[-2] not in {"sa", "sqa"}:
            raise ValueError(f"Unexpected sampler or solver seed: {key}")

        for column in [
            "n_bundles",
            "bqm_variables",
            "oracle_utility",
            "A",
            "C",
            "E",
            "coefficient_scale",
        ]:
            if len({float(row[column]) for row in group.values()}) != 1:
                raise ValueError(f"Unmatched {column}: {key}")
        for row in group.values():
            if int(row["n_reads"]) != NUM_READS:
                raise ValueError(f"Unexpected requested read count: {key}")
            if(row["oracle_status"] != "OPTIMAL" or float(row["oracle_utility"]) <=0):
                raise ValueError(f"Unavailable positive CP-SAT reference: {key}")
            if float(row["raw_feasible_rate"]) not in {0.0, 1.0}:
                raise ValueError(f"Archived call is not uniformly feasible/infeasible: {key}")

            for _, column, _ in METRICS:
                if not math.isfinite(float(row[column])):
                    raise ValueError(f"Nonfinite {column}: {key}")
        instances[key[:-1]][key[-1]] = group
    samples = defaultdict(list)
    for instance_key, seeded_groups in sorted(instances.items()):
        if restrict_to is not None and instance_key[:3] not in restrict_to:
            continue
        if set(seeded_groups) != set(SOLVER_SEEDS):
            raise ValueError(f"Incomplete solver-seed: {instance_key}")

        sampler = instance_key[-1]

        for treatment, baseline_arm in contrasts:
            for metric, column, _ in METRICS:
                baseline = statistics.fmean(
                    float(group[baseline_arm][column])
                    for group in seeded_groups.values()
                )
                calibrated = statistics.fmean(
                    float(group[treatment][column])
                    for group in seeded_groups.values()
                )
                samples[(treatment, baseline_arm, sampler, metric)].append(
                    (instance_key[1], baseline, calibrated)
                )
    summary_rows = []
    for treatment, baseline_arm in contrasts:
        for sampler in SAMPLERS:
            for metric, _, lower_is_better in METRICS:
                records = samples[(treatment, baseline_arm, sampler, metric)]
                if not records:
                    raise ValueError(
                        f"No paired results: {treatment} vs {baseline_arm}, {sampler}"
                    )
                values=[calibrated - baseline for _, baseline, calibrated in records]
                blocks = defaultdict(list)
                for (seed, _, _), value in zip(records, values):
                    blocks[seed].append(value)
                ci_low, ci_high = bootstrap_mean_ci(
                    list(blocks.values()),
                    contrast_seed(treatment, baseline_arm, sampler, metric),
                )
                direction = -1 if lower_is_better else 1
                better = sum(direction*value > TIE_TOL for value in values)
                worse = sum(direction*value < -TIE_TOL for value in values)

                summary_rows.append(
                    {
                        "calibration": treatment,
                        "baseline": baseline_arm,
                        "sampler": sampler,
                        "metric": metric,
                        "difference_definition":
                            f"{treatment} - {baseline_arm}",
                        "lower_is_better": lower_is_better,
                        "n_instances": len(values),
                        "n_seed_blocks": len(blocks),
                        "solver_seeds_per_instance": len(SOLVER_SEEDS),
                        "mean_baseline":
                            statistics.fmean(r[1] for r in records),
                        "mean_treatment":
                            statistics.fmean(r[2] for r in records),
                        "mean_paired_difference":
                            statistics.fmean(values),
                        "median_paired_difference":
                            statistics.median(values),
                        "bootstrap_95_ci_low": ci_low,
                        "bootstrap_95_ci_high": ci_high,
                        "treatment_better_fraction": better / len(values),
                        "baseline_better_fraction": worse / len(values),
                        "tie_fraction": (len(values) - better - worse) / len(values),
                    }
                )
    return summary_rows


def main():
    results_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            "penalty_calibration",
        )
    )

    rows = read_csv(os.path.join(results_dir, "heldout_tightening_solver_runs.csv"))
    scan = read_csv(os.path.join(results_dir, "heldout_coefficient_scan.csv"))

    selected = {
        (
            row["topology"],
            int(row["instance_seed"]),
            int(row["n_requests"]),
        )
        for row in scan
        if any(float(row[column]) < TIGHTENING_TOL for column in ["B_ratio", "D_ratio", "B_ratio_per", "D_ratio_per"])
    }

    expected = {
        case + (sampler,seed,calibration)
        for case in selected
        for sampler in SAMPLERS
        for seed in SOLVER_SEEDS
        for calibration in CALIBRATIONS
    }
    actual = {
        (
            row["topology"],
            int(row["instance_seed"]),
            int(row["n_requests"]),
            row["sampler"],
            int(row["solver_seed"]),
            row["calibration"],
        )
        for row in rows
    }

    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            "solver CSV does not contain the complete archived selection."
        )

    summary = analyze_rows(rows)
    output_path = os.path.join(results_dir, "heldout_paired_analysis.csv")
    write_csv(output_path, summary)

    # Global rule versus conventional, restricted to the instances where the
    # global rule actually changed a coefficient. Over all selected instances
    # most differences are exact zeros (those instances were selected on the
    # per-resource criterion), which dilutes the mean.
    global_changed = {
        (row["topology"], int(row["instance_seed"]), int(row["n_requests"]))
        for row in scan
        if float(row["B_ratio"]) < TIGHTENING_TOL
        or float(row["D_ratio"]) < TIGHTENING_TOL
    }
    subset_summary = analyze_rows(
        rows, contrasts=[CONTRASTS[0]], restrict_to=global_changed
    )
    subset_path = os.path.join(
        results_dir, "heldout_paired_analysis_global_changed.csv"
    )
    write_csv(subset_path, subset_summary)

    reference = {
        (row["topology"], row["instance_seed"], row["n_requests"]): row
        for row in scan
    }

    corrected = []
    for row in rows:
        baseline = reference[
            (row["topology"], row["instance_seed"], row["n_requests"])
        ]
        updated = dict(row)

        for family in ["B", "D"]:
            ratio = float(row[family]) / float(baseline[f"{family}_conventional"])
            updated[f"{family}_ratio"] = ratio
            updated[f"{family}_tighter"] = ratio < TIGHTENING_TOL

        updated["found_feasible"] = row["found_feasible"].lower() =="true"
        corrected.append(updated)

    for columns, filename in [
        (
            ["sampler", "calibration"],
            "heldout_tightening_solver_summary_global.csv",
        ),
        (
            ["topology", "n_requests", "sampler", "calibration"],
            "heldout_tightening_solver_summary_by_regime.csv",
        ),
    ]:
        groups = defaultdict(list)
        for row in corrected:
            groups[tuple(row[column] for column in columns)].append(row)
        write_csv(
            os.path.join(results_dir, filename),
            [
                {
                    **dict(zip(columns, key)),
                    **summarize_group(group),
                }
                for key,group in sorted(groups.items())
            ],
        )
    
    for row in summary:
        print(
            f"{row['difference_definition']:42s} "
            f"{row['sampler'].upper():3s} "
            f"{row['metric']:28s} "
            f"mean={row['mean_paired_difference']:+.6f} "
            f"95% CI=[{row['bootstrap_95_ci_low']:+.6f}, "
            f"{row['bootstrap_95_ci_high']:+.6f}]"
        )
    print()
    for row in subset_summary:
        print(
            f"[global-changed subset, n={row['n_instances']}] "
            f"{row['difference_definition']:32s} "
            f"{row['sampler'].upper():3s} "
            f"{row['metric']:28s} "
            f"mean={row['mean_paired_difference']:+.6f} "
            f"95% CI=[{row['bootstrap_95_ci_low']:+.6f}, "
            f"{row['bootstrap_95_ci_high']:+.6f}]"
        )
    print(
        f"\n{len(summary)} paired comparisons "
        f"({len(CONTRASTS)} contrasts x {len(SAMPLERS)} samplers "
        f"x {len(METRICS)} metrics) plus {len(subset_summary)} for the "
        f"global-changed subset = {len(summary) + len(subset_summary)} in "
        f"total; no multiplicity correction applied."
    )
    print(f"Wrote {len(summary)} comparisons to {output_path}")


if __name__ == "__main__":
    main()
