"""Benchmark conventional vs resource-aware OpenJij QUBO calibration.

This experiment is intentionally designed to test *calibration*, not merely the
post-repair solution. For each SA/SQA run it records:

- the actual A/B/C/D/E coefficients;
- fraction of raw OpenJij reads that are already feasible;
- raw request/capacity violation statistics before any repair;
- best raw-feasible utility;
- best utility after a deterministic repair pass;
- CP-SAT optimality gap where OR-Tools is available;
- QUBO compilation/calibration time and OpenJij sampling time.

Run from ``QNet_Sim``:

    PYTHONPATH=src python src/experiments/run_penalty_calibration.py

For a larger publication-style sweep:

    PYTHONPATH=src python src/experiments/run_penalty_calibration.py --full
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.instances import (  # noqa: E402
    contention_sweep_instances,
    generate_chain_topology,
    generate_grid_topology,
)
from optimization.openjij_solver import (  # noqa: E402
    build_calibrated_bqm,
    solve_sa,
    solve_sqa,
)
from optimization.qubo_optimizer import QUBOOptimizer  # noqa: E402

try:  # noqa: E402
    from baselines.classical_baselines import CPSATAllocator, CPSAT_AVAILABLE
except Exception:  # pragma: no cover - optional dependency/import path
    CPSATAllocator = None
    CPSAT_AVAILABLE = False


BundleKey = Tuple[str, str]


def _bundle_map(bundles: Sequence[dict]) -> Dict[BundleKey, dict]:
    return {(b["request_id"], b["bundle_id"]): b for b in bundles}


def selection_metrics(
    selected: Sequence[BundleKey],
    bundles: Sequence[dict],
    edge_capacities: dict,
    memory_capacities: dict,
) -> dict:
    """Evaluate a raw selection without silently collapsing conflicts."""
    by_key = _bundle_map(bundles)
    req_counts = Counter(rid for rid, _ in selected)
    request_conflicts = sum(max(0, count - 1) for count in req_counts.values())

    edge_load = defaultdict(int)
    mem_load = defaultdict(int)
    utility = 0.0
    for key in selected:
        bundle = by_key.get(key)
        if bundle is None:
            continue
        utility += float(bundle["utility"])
        for edge, demand in bundle["edge_demands"].items():
            edge_load[tuple(sorted(edge))] += int(demand)
        for node, demand in bundle["memory_demands"].items():
            mem_load[node] += int(demand)

    edge_violations = 0
    edge_overload = 0
    for edge, load in edge_load.items():
        cap = int(edge_capacities.get(tuple(sorted(edge)), 0))
        if load > cap:
            edge_violations += 1
            edge_overload += load - cap

    memory_violations = 0
    memory_overload = 0
    for node, load in mem_load.items():
        cap = int(memory_capacities.get(node, 0))
        if load > cap:
            memory_violations += 1
            memory_overload += load - cap

    violation_count = request_conflicts + edge_violations + memory_violations
    overload_units = edge_overload + memory_overload

    return {
        "utility": utility,
        "request_conflicts": request_conflicts,
        "edge_violations": edge_violations,
        "memory_violations": memory_violations,
        "violation_count": violation_count,
        "overload_units": overload_units,
        "feasible": violation_count == 0,
    }


def deterministic_repair(
    selected: Sequence[BundleKey],
    bundles: Sequence[dict],
    edge_capacities: dict,
    memory_capacities: dict,
) -> List[BundleKey]:
    """Repair only the bundles present in one raw OpenJij sample.

    If a sample selects multiple bundles for one request, retain that request's
    highest-utility selected candidate. Requests are then processed from high
    to low utility and admitted only when edge and memory capacities permit.
    No unselected bundle is introduced by the repair.
    """
    by_key = _bundle_map(bundles)
    by_request: Dict[str, List[BundleKey]] = defaultdict(list)
    for key in selected:
        if key in by_key:
            by_request[key[0]].append(key)

    candidates: List[BundleKey] = []
    for rid, keys in by_request.items():
        best = max(
            keys,
            key=lambda key: (
                float(by_key[key]["utility"]),
                str(key[1]),
            ),
        )
        candidates.append(best)

    candidates.sort(
        key=lambda key: (
            float(by_key[key]["utility"]),
            str(key[0]),
            str(key[1]),
        ),
        reverse=True,
    )

    edge_load = defaultdict(int)
    mem_load = defaultdict(int)
    repaired: List[BundleKey] = []

    for key in candidates:
        bundle = by_key[key]
        fits = True
        for edge, demand in bundle["edge_demands"].items():
            e = tuple(sorted(edge))
            if edge_load[e] + int(demand) > int(edge_capacities.get(e, 0)):
                fits = False
                break
        if fits:
            for node, demand in bundle["memory_demands"].items():
                if mem_load[node] + int(demand) > int(memory_capacities.get(node, 0)):
                    fits = False
                    break
        if not fits:
            continue

        repaired.append(key)
        for edge, demand in bundle["edge_demands"].items():
            edge_load[tuple(sorted(edge))] += int(demand)
        for node, demand in bundle["memory_demands"].items():
            mem_load[node] += int(demand)

    return repaired


def summarize_response(
    optimizer: QUBOOptimizer,
    response,
    bundles: Sequence[dict],
    edge_capacities: dict,
    memory_capacities: dict,
) -> dict:
    """Aggregate raw and repaired statistics across all OpenJij reads."""
    n_reads = 0
    raw_feasible_reads = 0
    raw_request_conflicts: List[float] = []
    raw_edge_violations: List[float] = []
    raw_memory_violations: List[float] = []
    raw_violation_counts: List[float] = []
    raw_overloads: List[float] = []
    raw_best_feasible_utility = float("-inf")
    repaired_best_utility = float("-inf")

    for sample in response.samples():
        n_reads += 1
        raw = optimizer.decode_sample(sample, repair=False)
        raw_metrics = selection_metrics(raw, bundles, edge_capacities, memory_capacities)
        raw_request_conflicts.append(float(raw_metrics["request_conflicts"]))
        raw_edge_violations.append(float(raw_metrics["edge_violations"]))
        raw_memory_violations.append(float(raw_metrics["memory_violations"]))
        raw_violation_counts.append(float(raw_metrics["violation_count"]))
        raw_overloads.append(float(raw_metrics["overload_units"]))

        if raw_metrics["feasible"]:
            raw_feasible_reads += 1
            raw_best_feasible_utility = max(
                raw_best_feasible_utility,
                float(raw_metrics["utility"]),
            )

        repaired = deterministic_repair(
            raw,
            bundles,
            edge_capacities,
            memory_capacities,
        )
        repaired_metrics = selection_metrics(
            repaired,
            bundles,
            edge_capacities,
            memory_capacities,
        )
        repaired_best_utility = max(
            repaired_best_utility,
            float(repaired_metrics["utility"]),
        )

    if n_reads == 0:
        return {
            "n_reads": 0,
            "raw_feasible_rate": 0.0,
            "raw_mean_request_conflicts": math.nan,
            "raw_mean_edge_violations": math.nan,
            "raw_mean_memory_violations": math.nan,
            "raw_mean_violation_count": math.nan,
            "raw_mean_overload_units": math.nan,
            "raw_best_feasible_utility": math.nan,
            "repaired_best_utility": math.nan,
        }

    return {
        "n_reads": n_reads,
        "raw_feasible_rate": raw_feasible_reads / n_reads,
        "raw_mean_request_conflicts": statistics.fmean(raw_request_conflicts),
        "raw_mean_edge_violations": statistics.fmean(raw_edge_violations),
        "raw_mean_memory_violations": statistics.fmean(raw_memory_violations),
        "raw_mean_violation_count": statistics.fmean(raw_violation_counts),
        "raw_mean_overload_units": statistics.fmean(raw_overloads),
        "raw_best_feasible_utility": (
            raw_best_feasible_utility
            if raw_best_feasible_utility != float("-inf")
            else math.nan
        ),
        "repaired_best_utility": (
            repaired_best_utility
            if repaired_best_utility != float("-inf")
            else math.nan
        ),
    }


def _oracle_result(bundles, edge_caps, mem_caps, time_limit_s: float = 20.0):
    if not CPSAT_AVAILABLE or CPSATAllocator is None:
        return math.nan, False, "UNAVAILABLE"
    result = CPSATAllocator(
        list(bundles),
        edge_caps,
        mem_caps,
        time_limit_s=time_limit_s,
    ).solve()
    return (
        float(result.get("total_utility", math.nan)),
        bool(result.get("is_optimal", False)),
        str(result.get("status", "UNKNOWN")),
    )


def _gap_pct(reference: float, value: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(value):
        return math.nan
    denom = max(abs(reference), 1e-12)
    return 100.0 * (reference - value) / denom


def _topology_cases(full: bool):
    cases = [
        (
            "chain8_c4",
            lambda: generate_chain_topology(
                n_nodes=8, edge_capacity=4, memory_capacity=8,
                raw_fidelity=0.85, generation_prob=0.95,
            ),
        ),
        (
            "grid3x3_c4",
            lambda: generate_grid_topology(
                rows=3, cols=3, edge_capacity=4, memory_capacity=8,
                raw_fidelity=0.85, generation_prob=0.95,
            ),
        ),
    ]
    if full:
        cases.extend(
            [
                (
                    "chain10_c6",
                    lambda: generate_chain_topology(
                        n_nodes=10, edge_capacity=6, memory_capacity=10,
                        raw_fidelity=0.88, generation_prob=0.90,
                    ),
                ),
                (
                    "grid4x4_c5",
                    lambda: generate_grid_topology(
                        rows=4, cols=4, edge_capacity=5, memory_capacity=10,
                        raw_fidelity=0.88, generation_prob=0.90,
                    ),
                ),
            ]
        )
    return cases


def run_benchmark(
    *,
    out_dir: str,
    full: bool = False,
    num_reads: int = 50,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
    strategies: Sequence[str] = ("conventional", "resource_aware"),
) -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)

    instance_seeds = list(range(10)) if full else [0, 1]
    solver_seeds = [101, 202, 303] if full else [101]
    request_counts = [8, 16, 24] if full else [8, 16]
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    strategies = list(strategies)
    samplers = ["sa", "sqa"]

    rows: List[dict] = []

    for topology_name, topology_fn in _topology_cases(full):
        for instance_seed in instance_seeds:
            instances = contention_sweep_instances(
                topology_fn,
                request_counts,
                seed=instance_seed,
            )

            for instance_name, inst in instances.items():
                bundles = inst["bundles"]
                edge_caps = inst["edge_capacities"]
                mem_caps = inst["memory_capacities"]
                n_requests = int(inst["n_requests"])

                if not bundles:
                    continue

                oracle_utility, oracle_optimal, oracle_status = _oracle_result(
                    bundles,
                    edge_caps,
                    mem_caps,
                )

                optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)

                for strategy in strategies:
                    for scale in scales:
                        t0 = time.perf_counter()
                        bqm, coeffs = build_calibrated_bqm(
                            optimizer,
                            strategy,
                            coefficient_scale=scale,
                            congestion_penalty=congestion_penalty,
                            memory_congestion_penalty=memory_congestion_penalty,
                        )
                        compile_time = time.perf_counter() - t0

                        for sampler_name in samplers:
                            for solver_seed in solver_seeds:
                                t0 = time.perf_counter()
                                if sampler_name == "sa":
                                    response = solve_sa(
                                        bqm,
                                        num_reads=num_reads,
                                        seed=solver_seed,
                                    )
                                else:
                                    response = solve_sqa(
                                        bqm,
                                        num_reads=num_reads,
                                        seed=solver_seed,
                                    )
                                sample_time = time.perf_counter() - t0

                                stats = summarize_response(
                                    optimizer,
                                    response,
                                    bundles,
                                    edge_caps,
                                    mem_caps,
                                )

                                b_stats = coefficient_stats(coeffs["B"])
                                d_stats = coefficient_stats(coeffs["D"])

                                row = {
                                    "topology": topology_name,
                                    "instance": instance_name,
                                    "instance_seed": instance_seed,
                                    "solver_seed": solver_seed,
                                    "n_requests": n_requests,
                                    "n_bundles": len(bundles),
                                    "bqm_variables": bqm.num_variables,
                                    "sampler": sampler_name,
                                    "calibration": strategy,
                                    "coefficient_scale": scale,
                                    "A": coeffs["A"],
                                    # B/D are mappings under resource_aware_per;
                                    # "B"/"D" hold the mean over resources.
                                    "B": b_stats["mean"],
                                    "B_min": b_stats["min"],
                                    "B_max": b_stats["max"],
                                    "B_n": b_stats["n"],
                                    "C": coeffs["C"],
                                    "D": d_stats["mean"],
                                    "D_min": d_stats["min"],
                                    "D_max": d_stats["max"],
                                    "D_n": d_stats["n"],
                                    "E": coeffs["E"],
                                    "oracle_utility": oracle_utility,
                                    "oracle_optimal": oracle_optimal,
                                    "oracle_status": oracle_status,
                                    "compile_time_s": compile_time,
                                    "sample_time_s": sample_time,
                                    **stats,
                                }
                                row["raw_optimality_gap_pct"] = (
                                    _gap_pct(oracle_utility, row["raw_best_feasible_utility"])
                                    if oracle_optimal else math.nan
                                )
                                row["repaired_optimality_gap_pct"] = (
                                    _gap_pct(oracle_utility, row["repaired_best_utility"])
                                    if oracle_optimal else math.nan
                                )
                                rows.append(row)

                                print(
                                    f"{topology_name:12s} {instance_name:4s} "
                                    f"{sampler_name.upper():3s} {strategy:14s} "
                                    f"x{scale:<4g} raw-feas={row['raw_feasible_rate']:.2f} "
                                    f"raw-U={row['raw_best_feasible_utility']:.2f} "
                                    f"repair-U={row['repaired_best_utility']:.2f}"
                                )

    run_path = os.path.join(out_dir, "calibration_runs.csv")
    _write_csv(run_path, rows)

    summary = aggregate_rows(rows)
    summary_path = os.path.join(out_dir, "calibration_summary.csv")
    _write_csv(summary_path, summary)

    print(f"\nWrote {len(rows)} run rows to {run_path}")
    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    return rows


def coefficient_stats(value) -> dict:
    """Summarise a scalar or per-resource coefficient for CSV output."""
    if isinstance(value, dict):
        values = list(value.values()) or [0.0]
        return {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return {"mean": value, "min": value, "max": value, "n": 1}


def _write_csv(path: str, rows: Sequence[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[dict]) -> List[dict]:
    """Aggregate by sampler/calibration/scale for quick paper-table inspection."""
    groups: Dict[Tuple[str, str, float], List[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["sampler"], row["calibration"], row["coefficient_scale"])].append(row)

    summary = []
    for (sampler, calibration, scale), group in sorted(groups.items()):
        def mean_finite(key):
            vals = [float(r[key]) for r in group if math.isfinite(float(r[key]))]
            return statistics.fmean(vals) if vals else math.nan

        summary.append(
            {
                "sampler": sampler,
                "calibration": calibration,
                "coefficient_scale": scale,
                "n_runs": len(group),
                "mean_raw_feasible_rate": mean_finite("raw_feasible_rate"),
                "mean_raw_request_conflicts": mean_finite("raw_mean_request_conflicts"),
                "mean_raw_edge_violations": mean_finite("raw_mean_edge_violations"),
                "mean_raw_memory_violations": mean_finite("raw_mean_memory_violations"),
                "mean_raw_violation_count": mean_finite("raw_mean_violation_count"),
                "mean_raw_overload_units": mean_finite("raw_mean_overload_units"),
                "mean_raw_optimality_gap_pct": mean_finite("raw_optimality_gap_pct"),
                "mean_repaired_optimality_gap_pct": mean_finite("repaired_optimality_gap_pct"),
                "mean_compile_time_s": mean_finite("compile_time_s"),
                "mean_sample_time_s": mean_finite("sample_time_s"),
                "mean_A": mean_finite("A"),
                "mean_B": mean_finite("B"),
                "mean_D": mean_finite("D"),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "results",
                "penalty_calibration",
            )
        ),
    )
    parser.add_argument("--reads", type=int, default=50)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run 10 instance seeds, 3 solver seeds and larger topologies.",
    )
    parser.add_argument(
        "--congestion-penalty",
        type=float,
        default=0.0,
        help="Soft edge-congestion C. Keep at zero for the clean calibration ablation.",
    )
    parser.add_argument(
        "--memory-congestion-penalty",
        type=float,
        default=0.0,
        help="Soft memory-congestion E. Keep at zero for the clean calibration ablation.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["conventional", "resource_aware"],
        help="Calibration strategies to run, e.g. add resource_aware_per.",
    )
    args = parser.parse_args()

    run_benchmark(
        out_dir=args.out_dir,
        full=args.full,
        num_reads=args.reads,
        congestion_penalty=args.congestion_penalty,
        memory_congestion_penalty=args.memory_congestion_penalty,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()