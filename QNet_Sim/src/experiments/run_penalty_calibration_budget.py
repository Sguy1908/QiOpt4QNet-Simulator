"""Solver-budget sensitivity of penalty calibration.

Checks that the per-resource advantage over the conventional rule is not an
artifact of an under-budgeted annealer. For each budget (number of reads and,
for SA, number of sweeps) it runs matched conventional and per-resource QUBOs
on held-out instances and reports paired effects with bootstrap intervals.

It also records how many samples each OpenJij call actually returned
(``n_samples`` must equal the requested reads) and how many of them are
*distinct* (``n_distinct_samples``). With a fixed ``seed`` OpenJij returns the
same state for every read in a call, so extra reads change nothing; pass
``--independent-reads`` to draw each read as its own call with its own seed.

Run a small check (about 40 instances):

    PYTHONPATH=src python3 src/experiments/run_penalty_calibration_budget.py

Run the full 1,078-instance cohort (hours):

    PYTHONPATH=src python3 src/experiments/run_penalty_calibration_budget.py --full
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.instances import contention_sweep_instances  # noqa: E402
from experiments.run_penalty_calibration import (  # noqa: E402
    _gap_pct,
    _oracle_result,
    _topology_cases,
    _write_csv,
    summarize_response,
)
from experiments.run_penalty_calibration_heldout import (  # noqa: E402
    INSTANCE_SEEDS,
    REQUEST_COUNTS,
    TIGHTENING_TOL,
)
from optimization.openjij_solver import (  # noqa: E402
    build_calibrated_bqm,
    calibrated_coefficients,
    solve_sa,
    solve_sqa,
)
from optimization.qubo_optimizer import QUBOOptimizer  # noqa: E402

CALIBRATIONS = ("conventional", "resource_aware_per")
DEFAULT_SEEDS = (101, 202, 303)


class _MergedReads:
    """Minimal stand-in for an OpenJij response made of several 1-read calls."""

    def __init__(self, responses):
        self._responses = responses

    def samples(self):
        for response in self._responses:
            yield from response.samples()


def draw_reads(sampler, bqm, reads, seed, sweeps=None, independent=False):
    """Draw ``reads`` samples.

    With a fixed seed OpenJij yields identical reads within one call, so
    ``independent=True`` makes ``reads`` single-read calls with seeds
    ``seed, seed + 1, ...`` instead. Still reproducible, but the reads differ.
    """
    def call(n, call_seed):
        if sampler == "sa":
            return solve_sa(bqm, num_reads=n, seed=call_seed, num_sweeps=sweeps)
        return solve_sqa(bqm, num_reads=n, seed=call_seed)

    if not independent:
        return call(reads, seed)
    return _MergedReads([call(1, seed + i) for i in range(reads)])


def count_distinct(response):
    return len({tuple(sorted(sample.items())) for sample in response.samples()})


def sample_instances(topologies, instance_seeds, request_counts, n_instances, seed=0):
    """(topology, instance_seed, n_requests) triples; all of them if n is None."""
    grid = [
        (topology, instance_seed, n_requests)
        for topology in topologies
        for instance_seed in instance_seeds
        for n_requests in request_counts
    ]
    if n_instances is None or n_instances >= len(grid):
        return grid
    return random.Random(seed).sample(grid, n_instances)


def _per_resource_tightened(optimizer):
    conventional = calibrated_coefficients(optimizer, "conventional")
    per = calibrated_coefficients(optimizer, "resource_aware_per")
    for family in ("B", "D"):
        mean = statistics.fmean(per[family].values()) if per[family] else math.nan
        if mean / conventional[family] < TIGHTENING_TOL:
            return True
    return False


def run_budget_study(
    topologies,
    triples,
    budgets,
    samplers=("sa",),
    solver_seeds=DEFAULT_SEEDS,
    request_counts=REQUEST_COUNTS,
    independent_reads=False,
    log=print,
):
    """Return one row per (instance, calibration, sampler, budget, seed).

    ``budgets`` is a sequence of ``(num_reads, num_sweeps_or_None)``. Sweeps
    only apply to SA.
    """
    rows = []
    for index, (topology_name, instance_seed, n_requests) in enumerate(triples, 1):
        instances = contention_sweep_instances(
            topologies[topology_name], request_counts, seed=instance_seed
        )
        inst = instances[f"req{n_requests}"]
        bundles = inst["bundles"]
        if not bundles:
            continue
        edge_caps, mem_caps = inst["edge_capacities"], inst["memory_capacities"]
        optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)
        if not _per_resource_tightened(optimizer):
            continue

        oracle_utility, oracle_optimal, _ = _oracle_result(bundles, edge_caps, mem_caps)
        if not oracle_optimal:
            continue
        log(f"[{index}/{len(triples)}] {topology_name} n={n_requests} seed={instance_seed}")

        for calibration in CALIBRATIONS:
            bqm, _ = build_calibrated_bqm(optimizer, calibration)
            for sampler in samplers:
                for reads, sweeps in budgets:
                    if sampler == "sqa" and sweeps is not None:
                        continue
                    for solver_seed in solver_seeds:
                        response = draw_reads(
                            sampler, bqm, reads, solver_seed, sweeps, independent_reads
                        )
                        n_distinct = count_distinct(response)
                        stats = summarize_response(
                            optimizer, response, bundles, edge_caps, mem_caps
                        )
                        rows.append(
                            {
                                "topology": topology_name,
                                "instance_seed": instance_seed,
                                "n_requests": n_requests,
                                "calibration": calibration,
                                "sampler": sampler,
                                "reads": reads,
                                "sweeps": "default" if sweeps is None else sweeps,
                                "solver_seed": solver_seed,
                                "independent_reads": independent_reads,
                                "n_samples": stats["n_reads"],
                                "n_distinct_samples": n_distinct,
                                "raw_feasible_rate": stats["raw_feasible_rate"],
                                "any_raw_feasible": float(
                                    math.isfinite(stats["raw_best_feasible_utility"])
                                ),
                                "repaired_gap_pct": _gap_pct(
                                    oracle_utility, stats["repaired_best_utility"]
                                ),
                            }
                        )
    return rows


def summarize(rows, n_boot=2000, seed=0):
    """Paired per-resource minus conventional effect per budget, with bootstrap CIs."""
    per_key = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["sampler"], row["reads"], row["sweeps"])
        inst = (row["topology"], row["instance_seed"], row["n_requests"])
        per_key[key][(inst, row["calibration"])].append(row)

    rng = np.random.default_rng(seed)
    out = []
    for key, cells in sorted(per_key.items(), key=lambda kv: str(kv[0])):
        instances = sorted({inst for inst, _ in cells})
        diffs = {"raw_feasible_rate": [], "repaired_gap_pct": []}
        for inst in instances:
            conv = cells.get((inst, "conventional"))
            per = cells.get((inst, "resource_aware_per"))
            if not conv or not per:
                continue
            for metric in diffs:
                c = statistics.fmean(r[metric] for r in conv)
                p = statistics.fmean(r[metric] for r in per)
                diffs[metric].append(p - c)
        n = len(diffs["raw_feasible_rate"])
        if n == 0:
            continue
        record = {
            "sampler": key[0], "reads": key[1], "sweeps": key[2], "n_instances": n,
            "samples_match_reads": all(
                r["n_samples"] == r["reads"] for c in cells.values() for r in c
            ),
            "mean_distinct_fraction": statistics.fmean(
                r["n_distinct_samples"] / r["reads"] for c in cells.values() for r in c
            ),
        }
        for metric, values in diffs.items():
            arr = np.asarray(values, dtype=float)
            boots = rng.choice(arr, size=(n_boot, n)).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            record[f"{metric}_mean_diff"] = float(arr.mean())
            record[f"{metric}_ci_low"] = float(lo)
            record[f"{metric}_ci_high"] = float(hi)
        record["per_resource_worse_gap_fraction"] = float(
            np.mean(np.asarray(diffs["repaired_gap_pct"]) > 0)
        )
        out.append(record)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-instances", type=int, default=40)
    parser.add_argument("--full", action="store_true", help="use every held-out instance")
    parser.add_argument("--reads", type=int, nargs="+", default=[100, 1000])
    parser.add_argument(
        "--sweeps", type=int, nargs="*", default=[10000],
        help="extra SA budgets: 100 reads at each of these sweep counts",
    )
    parser.add_argument("--samplers", nargs="+", default=["sa"], choices=["sa", "sqa"])
    parser.add_argument("--solver-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--independent-reads", action="store_true",
        help="draw every read as its own seeded call (fixed-seed reads are identical)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "results", "penalty_calibration")
        ),
    )
    args = parser.parse_args(argv)

    topologies = dict(_topology_cases(True))
    triples = sample_instances(
        topologies, INSTANCE_SEEDS, REQUEST_COUNTS,
        None if args.full else args.n_instances, args.sample_seed,
    )
    budgets = [(reads, None) for reads in args.reads]
    budgets += [(min(args.reads), sweeps) for sweeps in args.sweeps]

    rows = run_budget_study(
        topologies, triples, budgets, tuple(args.samplers), tuple(args.solver_seeds),
        independent_reads=args.independent_reads,
    )
    summary = summarize(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "budget_sensitivity_runs.csv"), rows)
    _write_csv(os.path.join(args.out_dir, "budget_sensitivity_summary.csv"), summary)
    for record in summary:
        print(
            f"{record['sampler']:3s} reads={record['reads']:5d} sweeps={str(record['sweeps']):8s} "
            f"n={record['n_instances']:4d} "
            f"d_feas={100 * record['raw_feasible_rate_mean_diff']:+6.2f}pp "
            f"d_gap={record['repaired_gap_pct_mean_diff']:+6.2f}pp "
            f"CI=[{record['repaired_gap_pct_ci_low']:+.2f},{record['repaired_gap_pct_ci_high']:+.2f}] "
            f"samples_match_reads={record['samples_match_reads']} "
            f"distinct_fraction={record['mean_distinct_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
