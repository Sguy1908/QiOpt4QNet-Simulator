"""Paired analysis for the held-out penalty-calibration experiment.

This script compares conventional and resource-aware calibration on matched
(instance, solver-seed) runs from the completed held-out experiment.

Run from QNet_Sim:

    PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_heldout.py
"""

from __future__ import annotations

import csv
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
METRICS = [("raw_feasible_rate", "raw_feasible_rate", False), ("repaired_reference_gap_pct", "repaired_optimality_gap_pct", True), ("raw_mean_overload_units", "raw_mean_overload_units", True)]

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


def analyze_rows(rows):
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
        if set(seeded_groups) != set(SOLVER_SEEDS):
            raise ValueError(f"Incomplete solver-seed: {instance_key}")

        sampler = instance_key[-1]

        for calibration in CALIBRATIONS[1:]:
            for metric, column, _ in METRICS:
                baseline = statistics.fmean(
                    float(group["conventional"][column])
                    for group in seeded_groups.values()
                )
                calibrated = statistics.fmean(
                    float(group[calibration][column])
                    for group in seeded_groups.values()
                )
                samples[(calibration, sampler, metric)].append((instance_key[1], baseline, calibrated))
    summary_rows = []
    for calibration in CALIBRATIONS[1:]:
        for sampler in {"sa", "sqa"}:
            for metric, _, lower_is_better in METRICS:
                records = samples[(calibration, sampler, metric)]
                if not records:
                    raise ValueError(f"No paired results: {calibration}, {sampler}")
                values=[calibrated - baseline for _, baseline, calibrated in records]
                blocks = defaultdict(list)
                for (seed, _, _), value in zip(records, values):
                    blocks[seed].append(value)
                ci_low, ci_high = bootstrap_mean_ci(list(blocks.values()), BOOTSTRAP_SEED + len(summary_rows))
                direction = -1 if lower_is_better else 1
                better = sum(direction*value > TIE_TOL for value in values)
                worse = sum(direction*value < -TIE_TOL for value in values)

                summary_rows.append(
                    {
                        "calibration": calibration,
                        "sampler": sampler,
                        "metric": metric,
                        "difference_definition":
                            f"{calibration} - conventional",
                        "lower_is_better": lower_is_better,
                        "n_instances": len(values),
                        "n_seed_blocks": len(blocks),
                        "solver_seeds_per_instance": len(SOLVER_SEEDS),
                        "mean_conventional":
                            statistics.fmean(r[1] for r in records),
                        "mean_calibrated":
                            statistics.fmean(r[2] for r in records),
                        "mean_paired_difference":
                            statistics.fmean(values),
                        "median_paired_difference":
                            statistics.median(values),
                        "bootstrap_95_ci_low": ci_low,
                        "bootstrap_95_ci_high": ci_high,
                        "ra_better_fraction": better / len(values),
                        "conventional_better_fraction": worse / len(values),
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
        for sampler in {"sa", "sqa"}
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
            f"{row['calibration']:20s} "
            f"{row['sampler'].upper():3s} "
            f"{row['metric']:28s} "
            f"mean={row['mean_paired_difference']:.6f} "
            f"95% CI=[{row['bootstrap_95_ci_low']:.6f}, "
            f"{row['bootstrap_95_ci_high']:.6f}]"
        )
    print(f"Wrote {len(summary)} comparisons to {output_path}")


if __name__ == "__main__":
    main()
