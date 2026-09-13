"""Mechanism analysis for held-out penalty calibration.

Tests the prediction that increasing request count makes reachable resource
loads denser, causing the minimum realizable penalty drop delta to equal 1
more often and reducing the opportunity for resource-aware tightening.

This analysis regenerates the same deterministic held-out instances.
It does not run SA or SQA.

Run from QNet_Sim:

    PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_mechanism.py
"""

from __future__ import annotations

import csv
import math
import os
import statistics

from experiments.instances import contention_sweep_instances
from experiments.run_penalty_calibration import _topology_cases
from experiments.run_penalty_calibration_heldout import (
    INSTANCE_SEEDS,
    REQUEST_COUNTS,
    TIGHTENING_TOL,
)
from optimization.conventional_calibrator import (
    conventional_coefficients,
    positive_utility_scale,
)
from optimization.proposed_calibrator import (
    possible_loads,
    proposed_global_coefficients,
)
from optimization.qubo_optimizer import QUBOOptimizer


FLOAT_TOL = 1e-12


def penalty_drop(other_load, demand, capacity):
    before = max(
        0,
        other_load + demand - capacity,
    ) ** 2

    after = max(
        0,
        other_load - capacity,
    ) ** 2

    return float(before - after)


def analyze_family(
    grouped_demands,
    capacities,
    utilities,
    p0,
):
    pair_rows = []

    for resource, key_demand_pairs in grouped_demands.items():
        capacity = int(capacities[resource])

        loads_cache = {}

        for key, demand_raw in key_demand_pairs:
            demand = int(demand_raw)

            if demand <= 0:
                continue

            request_id = key[0]
            cache_key = (request_id, demand)

            if cache_key not in loads_cache:
                loads_cache[cache_key] = possible_loads(
                    key_demand_pairs,
                    request_id,
                    capacity=capacity,
                    candidate_demand=demand,
                )

            loads = loads_cache[cache_key]

            violating = [
                load
                for load in loads
                if load + demand > capacity
            ]

            if not violating:
                continue

            other_load = min(violating)

            delta = penalty_drop(
                other_load,
                demand,
                capacity,
            )

            if delta <= 0:
                continue

            utility = max(
                0.0,
                float(utilities.get(key, 0.0)),
            )

            pair_rows.append(
                {
                    "delta": delta,
                    "utility": utility,
                    "ratio": utility / delta,
                }
            )

    if not pair_rows:
        return {
            "n_pairs": 0,
            "delta1_fraction": math.nan,
            "bound_set_by_delta1": False,
            "p0_delta1_pin": False,
        }

    delta1_count = sum(
        1
        for row in pair_rows
        if math.isclose(
            row["delta"],
            1.0,
            rel_tol=0.0,
            abs_tol=FLOAT_TOL,
        )
    )

    bound = max(
        row["ratio"]
        for row in pair_rows
    )

    binding_rows = [
        row
        for row in pair_rows
        if math.isclose(
            row["ratio"],
            bound,
            rel_tol=FLOAT_TOL,
            abs_tol=FLOAT_TOL,
        )
    ]

    bound_set_by_delta1 = any(
        math.isclose(
            row["delta"],
            1.0,
            rel_tol=0.0,
            abs_tol=FLOAT_TOL,
        )
        for row in binding_rows
    )

    p0_delta1_pin = (
        p0 > 0
        and any(
            math.isclose(
                row["utility"],
                p0,
                rel_tol=FLOAT_TOL,
                abs_tol=FLOAT_TOL,
            )
            and math.isclose(
                row["delta"],
                1.0,
                rel_tol=0.0,
                abs_tol=FLOAT_TOL,
            )
            for row in pair_rows
        )
    )

    return {
        "n_pairs": len(pair_rows),
        "delta1_fraction":
            delta1_count / len(pair_rows),
        "bound_set_by_delta1":
            bound_set_by_delta1,
        "p0_delta1_pin":
            p0_delta1_pin,
    }


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    topologies = dict(_topology_cases(True))

    instance_rows = []

    print("Analyzing held-out delta mechanism...")

    for topology_name, topology_fn in topologies.items():
        for instance_seed in INSTANCE_SEEDS:

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

                utilities = {}

                for bundle in optimizer.bundles:
                    key = optimizer._bundle_key(bundle)
                    utilities[key] = max(
                        0.0,
                        float(bundle["utility"]),
                    )

                p0 = positive_utility_scale(optimizer)

                conventional = conventional_coefficients(
                    optimizer,
                )

                resource_aware = proposed_global_coefficients(
                    optimizer,
                )

                families = [
                    (
                        "edge",
                        optimizer.edge_demands,
                        optimizer.edge_capacities,
                        "B",
                    ),
                    (
                        "memory",
                        optimizer.memory_demands,
                        optimizer.memory_capacities,
                        "D",
                    ),
                ]

                for (
                    family,
                    grouped_demands,
                    capacities,
                    coefficient_name,
                ) in families:

                    mechanism = analyze_family(
                        grouped_demands,
                        capacities,
                        utilities,
                        p0,
                    )

                    coefficient_ratio = (
                        resource_aware[coefficient_name]
                        / conventional[coefficient_name]
                    )

                    instance_rows.append(
                        {
                            "topology": topology_name,
                            "n_requests": n_requests,
                            "instance_seed":
                                instance_seed,
                            "family": family,
                            "n_pairs":
                                mechanism["n_pairs"],
                            "delta1_fraction":
                                mechanism[
                                    "delta1_fraction"
                                ],
                            "bound_set_by_delta1":
                                mechanism[
                                    "bound_set_by_delta1"
                                ],
                            "p0_delta1_pin":
                                mechanism[
                                    "p0_delta1_pin"
                                ],
                            "coefficient_ratio":
                                coefficient_ratio,
                            "coefficient_tightened":
                                coefficient_ratio
                                < TIGHTENING_TOL,
                        }
                    )

    summary_rows = []

    for n_requests in REQUEST_COUNTS:
        for family in ["edge", "memory"]:

            group = [
                row
                for row in instance_rows
                if (
                    row["n_requests"] == n_requests
                    and row["family"] == family
                )
            ]

            with_pairs = [
                row
                for row in group
                if row["n_pairs"] > 0
            ]

            total_pairs = sum(
                row["n_pairs"]
                for row in with_pairs
            )

            total_delta1_pairs = sum(
                row["delta1_fraction"]
                * row["n_pairs"]
                for row in with_pairs
            )

            pair_weighted_delta1_fraction = (
                total_delta1_pairs / total_pairs
                if total_pairs
                else math.nan
            )

            mean_instance_delta1_fraction = (
                statistics.fmean(
                    row["delta1_fraction"]
                    for row in with_pairs
                )
                if with_pairs
                else math.nan
            )

            bound_set_by_delta1_fraction = (
                statistics.fmean(
                    1.0
                    if row["bound_set_by_delta1"]
                    else 0.0
                    for row in with_pairs
                )
                if with_pairs
                else math.nan
            )

            p0_delta1_pin_fraction = (
                statistics.fmean(
                    1.0
                    if row["p0_delta1_pin"]
                    else 0.0
                    for row in group
                )
                if group
                else math.nan
            )

            tightening_fraction = (
                statistics.fmean(
                    1.0
                    if row["coefficient_tightened"]
                    else 0.0
                    for row in group
                )
                if group
                else math.nan
            )

            mean_coefficient_ratio = (
                statistics.fmean(
                    row["coefficient_ratio"]
                    for row in group
                )
                if group
                else math.nan
            )

            summary_rows.append(
                {
                    "n_requests": n_requests,
                    "family": family,
                    "n_instances": len(group),
                    "n_instances_with_pairs":
                        len(with_pairs),
                    "n_violating_pairs":
                        total_pairs,
                    "mean_instance_delta1_fraction":
                        mean_instance_delta1_fraction,
                    "pair_weighted_delta1_fraction":
                        pair_weighted_delta1_fraction,
                    "bound_set_by_delta1_fraction":
                        bound_set_by_delta1_fraction,
                    "p0_delta1_pin_fraction":
                        p0_delta1_pin_fraction,
                    "coefficient_tightened_fraction":
                        tightening_fraction,
                    "mean_coefficient_ratio":
                        mean_coefficient_ratio,
                }
            )

    results_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            "penalty_calibration",
        )
    )

    output_path = os.path.join(
        results_dir,
        "heldout_mechanism_analysis.csv",
    )

    write_csv(
        output_path,
        summary_rows,
    )

    print()
    print("Held-out delta mechanism")
    print()

    for row in summary_rows:
        print(
            f"n={row['n_requests']:2d} "
            f"{row['family']:6s} "
            f"delta1="
            f"{100 * row['mean_instance_delta1_fraction']:6.2f}% "
            f"binding-delta1="
            f"{100 * row['bound_set_by_delta1_fraction']:6.2f}% "
            f"p0-pin="
            f"{100 * row['p0_delta1_pin_fraction']:6.2f}% "
            f"tightened="
            f"{100 * row['coefficient_tightened_fraction']:6.2f}% "
            f"mean-ratio="
            f"{row['mean_coefficient_ratio']:.3f}"
        )

    print()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()