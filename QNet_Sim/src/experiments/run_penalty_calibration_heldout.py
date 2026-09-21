"""Held-out validation of penalty-coefficient calibration.

The experiment uses a fixed set of held-out instance seeds. It first scans
all instances for cases where resource-aware calibration tightens B or D.
Those changed instances are then evaluated with matched SA/SQA runs and a
CP-SAT oracle.

Run the full experiment:

    PYTHONPATH=src python3 src/experiments/run_penalty_calibration_heldout.py

Run only the coefficient scan:

    PYTHONPATH=src python3 src/experiments/run_penalty_calibration_heldout.py --scan-only
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from collections import defaultdict

from experiments.instances import contention_sweep_instances
from experiments.run_penalty_calibration import (
    _gap_pct,
    coefficient_stats,
    _oracle_result,
    _topology_cases,
    _write_csv,
    summarize_response,
)
from optimization.openjij_solver import (
    build_calibrated_bqm,
    calibrated_coefficients,
    solve_sa,
    solve_sqa,
)
from optimization.qubo_optimizer import QUBOOptimizer


INSTANCE_SEEDS = list(range(10, 100))
REQUEST_COUNTS = [8, 16, 24]
SOLVER_SEEDS = [101, 202, 303, 404, 505]

NUM_READS = 100
TIGHTENING_TOL = 0.999999


def mean_finite(rows, key):
    values = []

    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            values.append(value)

    return statistics.fmean(values) if values else math.nan

def summarize_group(rows):
    instances = {
        (
            row["topology"],
            row["instance_seed"],
            row["n_requests"],
        )
        for row in rows
    }

    return {
        "n_solver_runs": len(rows),
        "n_instances": len(instances),
        "mean_raw_feasible_read_rate":
            mean_finite(rows, "raw_feasible_rate"),
        "fraction_runs_with_any_feasible":
            statistics.fmean(
                1.0 if row["found_feasible"] else 0.0
                for row in rows
            ),
        "mean_raw_request_conflicts":
            mean_finite(rows, "raw_mean_request_conflicts"),
        "mean_raw_edge_violations":
            mean_finite(rows, "raw_mean_edge_violations"),
        "mean_raw_memory_violations":
            mean_finite(rows, "raw_mean_memory_violations"),
        "mean_raw_violation_count":
            mean_finite(rows, "raw_mean_violation_count"),
        "mean_raw_overload_units":
            mean_finite(rows, "raw_mean_overload_units"),
        "mean_raw_gap_on_feasible_runs_pct":
            mean_finite(rows, "raw_optimality_gap_pct"),
        "mean_repaired_gap_pct":
            mean_finite(rows, "repaired_optimality_gap_pct"),
        "mean_B": mean_finite(rows, "B"),
        "mean_D": mean_finite(rows, "D"),
        "mean_B_ratio": mean_finite(rows, "B_ratio"),
        "mean_D_ratio": mean_finite(rows, "D_ratio"),
        "mean_sample_time_s":
            mean_finite(rows, "sample_time_s"),
    }


def main():
    scan_only = "--scan-only" in sys.argv

    out_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            "penalty_calibration",
        )
    )
    os.makedirs(out_dir, exist_ok=True)

    scan_path = os.path.join(
        out_dir,
        "heldout_coefficient_scan.csv",
    )
    runs_path = os.path.join(
        out_dir,
        "heldout_tightening_solver_runs.csv",
    )
    global_summary_path = os.path.join(
        out_dir,
        "heldout_tightening_solver_summary_global.csv",
    )
    regime_summary_path = os.path.join(
        out_dir,
        "heldout_tightening_solver_summary_by_regime.csv",
    )

    topologies = dict(_topology_cases(True))

    scan_rows = []
    selected = []

    print("Scanning held-out instances...")

    for topology_name, topology_fn in topologies.items():
        for instance_seed in INSTANCE_SEEDS:

            # Generate all request counts together so the random-number
            # sequence matches the benchmark construction.
            instances = contention_sweep_instances(
                topology_fn,
                REQUEST_COUNTS,
                seed=instance_seed,
            )

            for n_requests in REQUEST_COUNTS:
                instance_name = f"req{n_requests}"
                inst = instances[instance_name]

                bundles = inst["bundles"]

                if not bundles:
                    continue

                optimizer = QUBOOptimizer(
                    bundles,
                    inst["edge_capacities"],
                    inst["memory_capacities"],
                )

                conventional = calibrated_coefficients(
                    optimizer,
                    "conventional",
                    coefficient_scale=1.0,
                )

                resource_aware = calibrated_coefficients(
                    optimizer,
                    "resource_aware",
                    coefficient_scale=1.0,
                )

                per_resource = calibrated_coefficients(
                    optimizer,
                    "resource_aware_per",
                    coefficient_scale=1.0,
                )
                b_stats  = coefficient_stats(per_resource["B"])
                d_stats  = coefficient_stats(per_resource["D"])
                b_ratio = (
                    resource_aware["B"]
                    / conventional["B"]
                )
                d_ratio = (
                    resource_aware["D"]
                    / conventional["D"]
                )
                b_ratio_per = b_stats["mean"]/conventional["B"]
                d_ratio_per = d_stats["mean"]/conventional["D"]

                b_tighter = b_ratio < TIGHTENING_TOL
                d_tighter = d_ratio < TIGHTENING_TOL
                b_tighter_per = b_ratio_per < TIGHTENING_TOL
                d_tighter_per = d_ratio_per < TIGHTENING_TOL
                b_n_tightened = sum(
                    1 for v in per_resource["B"].values()
                    if v < conventional["B"] * TIGHTENING_TOL
                )
                d_n_tightened = sum(
                    1 for v in per_resource["D"].values()
                    if v < conventional["D"] * TIGHTENING_TOL
                )

                scan_rows.append(
                    {
                        "topology": topology_name,
                        "instance": instance_name,
                        "instance_seed": instance_seed,
                        "n_requests": n_requests,
                        "n_bundles": len(bundles),
                        "A_conventional":
                            conventional["A"],
                        "B_conventional":
                            conventional["B"],
                        "D_conventional":
                            conventional["D"],
                        "A_resource_aware":
                            resource_aware["A"],
                        "B_resource_aware":
                            resource_aware["B"],
                        "D_resource_aware":
                            resource_aware["D"],
                        "B_ratio": b_ratio,
                        "D_ratio": d_ratio,
                        "B_per_resource_mean": b_stats["mean"],
                        "B_per_resource_min": b_stats["min"],
                        "B_per_resource_max": b_stats["max"],
                        "B_per_resource_n": b_stats["n"],
                        "D_per_resource_mean": d_stats["mean"],
                        "D_per_resource_min": d_stats["min"],
                        "D_per_resource_max": d_stats["max"],
                        "D_per_resource_n": d_stats["n"],
                        "B_ratio_per": b_ratio_per,
                        "D_ratio_per": d_ratio_per,
                        "B_tighter": b_tighter,
                        "D_tighter": d_tighter,
                        "B_n_tightened": b_n_tightened,
                        "D_n_tightened": d_n_tightened,
                    }
                )

                if b_tighter or d_tighter or b_tighter_per or d_tighter_per:
                    selected.append(
                        (
                            topology_name,
                            n_requests,
                            instance_seed,
                        )
                    )

    _write_csv(scan_path, scan_rows)

    b_tight_rows = [
        row for row in scan_rows
        if row["B_tighter"]
    ]
    d_tight_rows = [
        row for row in scan_rows
        if row["D_tighter"]
    ]
    both_tight_rows = [
        row for row in scan_rows
        if row["B_tighter"] and row["D_tighter"]
    ]

    mean_b_reduction = statistics.fmean(
        1.0 - row["B_ratio"]
        for row in b_tight_rows
    )

    mean_d_reduction = statistics.fmean(
        1.0 - row["D_ratio"]
        for row in d_tight_rows
    )

    print()
    print("Held-out coefficient scan")
    print(f"Instances checked: {len(scan_rows)}")
    print(
        f"B tighter: {len(b_tight_rows)}/{len(scan_rows)} "
        f"({100 * len(b_tight_rows) / len(scan_rows):.2f}%)"
    )
    print(
        f"D tighter: {len(d_tight_rows)}/{len(scan_rows)} "
        f"({100 * len(d_tight_rows) / len(scan_rows):.2f}%)"
    )
    print(
        f"Both tighter: {len(both_tight_rows)}/{len(scan_rows)} "
        f"({100 * len(both_tight_rows) / len(scan_rows):.2f}%)"
    )
    print(
        "Mean B reduction when tightened: "
        f"{100 * mean_b_reduction:.2f}%"
    )
    print(
        "Mean D reduction when tightened: "
        f"{100 * mean_d_reduction:.2f}%"
    )
    print(
        "Instances with B or D tightened: "
        f"{len(selected)}"
    )

    if scan_only:
        return

    rows = []

    for index, (
        topology_name,
        n_requests,
        instance_seed,
    ) in enumerate(selected, start=1):

        topology_fn = topologies[topology_name]

        instances = contention_sweep_instances(
            topology_fn,
            REQUEST_COUNTS,
            seed=instance_seed,
        )

        instance_name = f"req{n_requests}"
        inst = instances[instance_name]

        bundles = inst["bundles"]
        edge_caps = inst["edge_capacities"]
        mem_caps = inst["memory_capacities"]

        optimizer = QUBOOptimizer(
            bundles,
            edge_caps,
            mem_caps,
        )

        conventional = calibrated_coefficients(
            optimizer,
            "conventional",
            coefficient_scale=1.0,
        )

        resource_aware = calibrated_coefficients(
            optimizer,
            "resource_aware",
            coefficient_scale=1.0,
        )
        per_resource = calibrated_coefficients(
            optimizer,
            "resource_aware_per",
            coefficient_scale=1.0,
        )

        b_ratio = (
            resource_aware["B"]
            / conventional["B"]
        )
        d_ratio = (
            resource_aware["D"]
            / conventional["D"]
        )
        b_ratio_per = (
            coefficient_stats(per_resource["B"])["mean"]
            / conventional["B"]
        )
        d_ratio_per = (
            coefficient_stats(per_resource["D"])["mean"]
            / conventional["D"]
        )
        oracle_utility, oracle_optimal, oracle_status = (
            _oracle_result(
                bundles,
                edge_caps,
                mem_caps,
            )
        )

        print(
            f"[{index}/{len(selected)}] "
            f"{topology_name} {instance_name} "
            f"seed={instance_seed} "
            f"B-ratio={b_ratio:.3f} "
            f"D-ratio={d_ratio:.3f} "
            f"B-per={b_ratio_per:.3f} "
            f"D-per={d_ratio_per:.3f}"
        )

        for calibration in [
            "conventional",
            "resource_aware",
            "resource_aware_per",
        ]:
            compile_start = time.perf_counter()

            bqm, coeffs = build_calibrated_bqm(
                optimizer,
                calibration,
                coefficient_scale=1.0,
                congestion_penalty=0.0,
                memory_congestion_penalty=0.0,
            )

            compile_time = (
                time.perf_counter() - compile_start
            )

            for sampler_name in ["sa", "sqa"]:
                for solver_seed in SOLVER_SEEDS:

                    sample_start = time.perf_counter()

                    if sampler_name == "sa":
                        response = solve_sa(
                            bqm,
                            num_reads=NUM_READS,
                            seed=solver_seed,
                        )
                    else:
                        response = solve_sqa(
                            bqm,
                            num_reads=NUM_READS,
                            seed=solver_seed,
                        )

                    sample_time = (
                        time.perf_counter() - sample_start
                    )

                    stats = summarize_response(
                        optimizer,
                        response,
                        bundles,
                        edge_caps,
                        mem_caps,
                    )

                    if oracle_optimal:
                        raw_gap = _gap_pct(
                            oracle_utility,
                            stats[
                                "raw_best_feasible_utility"
                            ],
                        )
                        repaired_gap = _gap_pct(
                            oracle_utility,
                            stats[
                                "repaired_best_utility"
                            ],
                        )
                    else:
                        raw_gap = math.nan
                        repaired_gap = math.nan

                    found_feasible = math.isfinite(
                        stats[
                            "raw_best_feasible_utility"
                        ]
                    )

                    rows.append(
                        {
                            "topology": topology_name,
                            "instance": instance_name,
                            "instance_seed":
                                instance_seed,
                            "solver_seed": solver_seed,
                            "n_requests": n_requests,
                            "n_bundles": len(bundles),
                            "bqm_variables":
                                bqm.num_variables,
                            "sampler": sampler_name,
                            "calibration": calibration,
                            "coefficient_scale": 1.0,
                            "A": coeffs["A"],
                            "B": coefficient_stats(coeffs["B"])["mean"],
                            "B_min": coefficient_stats(coeffs["B"])["min"],
                            "B_max": coefficient_stats(coeffs["B"])["max"],
                            "B_n": coefficient_stats(coeffs["B"])["n"],
                            "C": coeffs["C"],
                            "D": coefficient_stats(coeffs["D"])["mean"],
                            "D_min": coefficient_stats(coeffs["D"])["min"],
                            "D_max": coefficient_stats(coeffs["D"])["max"],
                            "D_n": coefficient_stats(coeffs["D"])["n"],
                            "E": coeffs["E"],
                            # Instance-level ratios describing the GLOBAL rule.
                            # They characterise the instance, not this arm.
                            "B_ratio": b_ratio,
                            "D_ratio": d_ratio,
                            "B_ratio_per": b_ratio_per,
                            "D_ratio_per": d_ratio_per,
                            # This arm's own mean coefficient vs conventional.
                            "B_ratio_arm":
                                coefficient_stats(coeffs["B"])["mean"]
                                / conventional["B"],
                            "D_ratio_arm":
                                coefficient_stats(coeffs["D"])["mean"]
                                / conventional["D"],
                            "B_tighter":
                                b_ratio
                                < TIGHTENING_TOL,
                            "D_tighter":
                                d_ratio
                                < TIGHTENING_TOL,
                            "oracle_utility":
                                oracle_utility,
                            "oracle_optimal":
                                oracle_optimal,
                            "oracle_status":
                                oracle_status,
                            "compile_time_s":
                                compile_time,
                            "sample_time_s":
                                sample_time,
                            **stats,
                            "found_feasible":
                                found_feasible,
                            "raw_optimality_gap_pct":
                                raw_gap,
                            "repaired_optimality_gap_pct":
                                repaired_gap,
                        }
                    )

        # Checkpoint after every instance.
        _write_csv(runs_path, rows)

    global_groups = defaultdict(list)

    for row in rows:
        global_groups[
            (
                row["sampler"],
                row["calibration"],
            )
        ].append(row)

    global_summary = []

    for (
        sampler,
        calibration,
    ), group in sorted(global_groups.items()):
        global_summary.append(
            {
                "sampler": sampler,
                "calibration": calibration,
                **summarize_group(group),
            }
        )

    _write_csv(
        global_summary_path,
        global_summary,
    )

    regime_groups = defaultdict(list)

    for row in rows:
        regime_groups[
            (
                row["topology"],
                row["n_requests"],
                row["sampler"],
                row["calibration"],
            )
        ].append(row)

    regime_summary = []

    for (
        topology,
        n_requests,
        sampler,
        calibration,
    ), group in sorted(regime_groups.items()):
        regime_summary.append(
            {
                "topology": topology,
                "n_requests": n_requests,
                "sampler": sampler,
                "calibration": calibration,
                **summarize_group(group),
            }
        )

    _write_csv(
        regime_summary_path,
        regime_summary,
    )

    print()
    print("Held-out solver summary")

    for row in global_summary:
        print(
            f"{row['sampler'].upper():3s} "
            f"{row['calibration']:14s} "
            f"instances={row['n_instances']:3d} "
            f"read-feas="
            f"{row['mean_raw_feasible_read_rate']:.3f} "
            f"any-feasible="
            f"{row['fraction_runs_with_any_feasible']:.3f} "
            f"raw-gap|feasible="
            f"{row['mean_raw_gap_on_feasible_runs_pct']:.2f}% "
            f"repair-gap="
            f"{row['mean_repaired_gap_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()