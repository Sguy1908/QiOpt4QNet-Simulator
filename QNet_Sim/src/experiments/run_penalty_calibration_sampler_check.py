"""Penalty calibration under well-configured samplers, with scope ablations.

The held-out study (``run_penalty_calibration_heldout.py``) calls OpenJij with a
fixed seed, which returns one distinct state per call, and uses OpenJij's
default SA schedule. Under that protocol SA returns near-empty allocations
(about 86% repaired gap under conventional calibration). This runner asks
whether the calibration effects survive samplers that actually explore:

* ``sa``  -- dwave-samplers ``SimulatedAnnealingSampler``, 1000 sweeps, reads
  independent within a call;
* ``sqa`` -- OpenJij SQA with every read drawn as its own seeded call
  (``draw_reads(..., independent=True)``).

and adds two ablations next to the conventional, global and per-resource arms:

* ``local_utility`` -- per-resource scope without the reachable-load bound
  (``B_e = max u_b^+`` over bundles using ``e``), which separates coefficient
  scope from the dynamic program;
* ``resource_aware_per_pairA`` -- per-resource B/D plus per-pair request
  penalties ``A_bc = min(u_b^+, u_c^+) + eps``.

Every arm keeps the ground-state guarantee for ``C = E = 0``.

Run from ``QNet_Sim`` (about 1.5 h on 8 processes for all 1,078 instances):

    PYTHONPATH=src python3 src/experiments/run_penalty_calibration_sampler_check.py --full --workers 8

Outputs in ``results/penalty_calibration/``: ``sampler_check_runs.csv`` (one
row per instance, arm and sampler), ``sampler_check_levels.csv`` (mean of each
metric per arm) and ``sampler_check_summary.csv`` (paired contrasts with
generation-seed-block bootstrap intervals). ``--from-runs`` rebuilds the two
summaries from an existing ``sampler_check_runs.csv`` without re-solving.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.instances import contention_sweep_instances, instance_key  # noqa: E402
from experiments.run_penalty_calibration import (  # noqa: E402
    _gap_pct,
    _oracle_result,
    _topology_cases,
    _write_csv,
    summarize_response,
)
from experiments.run_penalty_calibration_budget import (  # noqa: E402
    _per_resource_tightened,
    block_bootstrap_ci,
    draw_reads,
    sample_instances,
)
from experiments.run_penalty_calibration_heldout import (  # noqa: E402
    INSTANCE_SEEDS,
    REQUEST_COUNTS,
)
from optimization.openjij_solver import build_calibrated_bqm  # noqa: E402
from optimization.qubo_optimizer import QUBOOptimizer  # noqa: E402

# arm name -> (calibration strategy, pairwise request penalties)
ARMS = {
    "conventional": ("conventional", False),
    "resource_aware": ("resource_aware", False),
    "resource_aware_per": ("resource_aware_per", False),
    "local_utility": ("local_utility", False),
    "resource_aware_per_pairA": ("resource_aware_per", True),
}
SAMPLERS = ("sa", "sqa")
READS = 100
SA_SWEEPS = 1000
SOLVER_SEED = 101
# (treatment, baseline)
CONTRASTS = [
    ("resource_aware_per", "conventional"),
    ("resource_aware_per", "resource_aware"),
    ("resource_aware", "conventional"),
    ("local_utility", "conventional"),
    ("resource_aware_per", "local_utility"),
    ("resource_aware_per_pairA", "resource_aware_per"),
]
# (metric, lower_is_better)
METRICS = [
    ("raw_feasible_rate", False),
    ("raw_nontrivial_feasible_rate", False),
    ("repaired_optimality_gap_pct", True),
    ("raw_mean_overload_units", True),
    ("raw_mean_request_conflicts", True),
]


def sample(sampler, bqm, seed):
    if sampler == "sa":
        from dwave.samplers import SimulatedAnnealingSampler

        return SimulatedAnnealingSampler().sample(
            bqm, num_reads=READS, num_sweeps=SA_SWEEPS, seed=seed
        )
    return draw_reads("sqa", bqm, READS, seed, independent=True)


def run_instances(topologies, triples, log=print):
    rows = []
    for index, (topology, instance_seed, n_requests) in enumerate(triples, start=1):
        inst = contention_sweep_instances(
            topologies[topology], REQUEST_COUNTS, seed=instance_seed
        )[instance_key(n_requests)]
        bundles = inst["bundles"]
        if not bundles:
            continue
        edge_caps, mem_caps = inst["edge_capacities"], inst["memory_capacities"]
        optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)
        if not _per_resource_tightened(optimizer):
            continue
        oracle, optimal, status = _oracle_result(bundles, edge_caps, mem_caps)
        if not optimal:
            raise RuntimeError(
                f"CP-SAT did not certify {topology} {instance_seed} {n_requests}: {status}"
            )
        for arm, (strategy, pairwise) in ARMS.items():
            bqm, _ = build_calibrated_bqm(
                optimizer, strategy, pairwise_request_penalty=pairwise
            )
            for sampler in SAMPLERS:
                stats = summarize_response(
                    optimizer,
                    sample(sampler, bqm, SOLVER_SEED),
                    bundles,
                    edge_caps,
                    mem_caps,
                )
                rows.append(
                    {
                        "topology": topology,
                        "instance_seed": instance_seed,
                        "n_requests": n_requests,
                        "arm": arm,
                        "sampler": sampler,
                        "solver_seed": SOLVER_SEED,
                        "reads": READS,
                        "sa_sweeps": SA_SWEEPS if sampler == "sa" else "",
                        "bqm_variables": bqm.num_variables,
                        "oracle_utility": oracle,
                        **stats,
                        "repaired_optimality_gap_pct": _gap_pct(
                            oracle, stats["repaired_best_utility"]
                        ),
                    }
                )
        log(
            f"[{index}/{len(triples)}] {topology} seed={instance_seed} req={n_requests}"
        )
    return rows


def _run_shard(triples):
    topologies = dict(_topology_cases(True))
    return run_instances(topologies, triples, log=lambda m: print(m, flush=True))


def _contrast_seed(*parts):
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def summarize(rows, n_boot=10000):
    """Absolute levels per arm and paired contrasts with seed-block bootstrap CIs."""
    by_key = defaultdict(dict)
    for row in rows:
        key = (
            row["topology"],
            int(row["instance_seed"]),
            int(row["n_requests"]),
            row["sampler"],
        )
        by_key[key][row["arm"]] = row
    for key, arms in by_key.items():
        if set(arms) != set(ARMS):
            raise ValueError(f"incomplete arms for {key}")

    out = []
    for sampler in SAMPLERS:
        cells = [arms for key, arms in by_key.items() if key[3] == sampler]
        for arm in ARMS:
            out.append(
                {
                    "kind": "level",
                    "sampler": sampler,
                    "treatment": arm,
                    "baseline": "",
                    "n_instances": len(cells),
                    **{
                        metric: statistics.fmean(float(c[arm][metric]) for c in cells)
                        for metric, _ in METRICS
                    },
                }
            )
        for treatment, baseline in CONTRASTS:
            for metric, lower_is_better in METRICS:
                blocks = defaultdict(list)
                for key, arms in by_key.items():
                    if key[3] != sampler:
                        continue
                    diff = float(arms[treatment][metric]) - float(
                        arms[baseline][metric]
                    )
                    blocks[key[1]].append(diff)
                values = [v for block in blocks.values() for v in block]
                rng = np.random.default_rng(
                    _contrast_seed(sampler, treatment, baseline, metric)
                )
                low, high = block_bootstrap_ci(blocks, rng, n_boot)
                direction = -1.0 if lower_is_better else 1.0
                out.append(
                    {
                        "kind": "contrast",
                        "sampler": sampler,
                        "treatment": treatment,
                        "baseline": baseline,
                        "metric": metric,
                        "n_instances": len(values),
                        "n_seed_blocks": len(blocks),
                        "mean_paired_difference": statistics.fmean(values),
                        "bootstrap_95_ci_low": low,
                        "bootstrap_95_ci_high": high,
                        "treatment_better_fraction": sum(
                            direction * v > 1e-12 for v in values
                        )
                        / len(values),
                        "baseline_better_fraction": sum(
                            direction * v < -1e-12 for v in values
                        )
                        / len(values),
                    }
                )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", action="store_true", help="every held-out instance")
    parser.add_argument("--n-instances", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "results", "penalty_calibration"
            )
        ),
    )
    parser.add_argument(
        "--from-runs",
        action="store_true",
        help="re-summarize <out-dir>/sampler_check_runs.csv instead of solving",
    )
    args = parser.parse_args(argv)
    runs_path = os.path.join(args.out_dir, "sampler_check_runs.csv")
    if args.from_runs:
        with open(runs_path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = _solve(args)
        os.makedirs(args.out_dir, exist_ok=True)
        _write_csv(runs_path, rows)
    summary = summarize(rows)
    _write_csv(
        os.path.join(args.out_dir, "sampler_check_levels.csv"),
        [r for r in summary if r["kind"] == "level"],
    )
    _write_csv(
        os.path.join(args.out_dir, "sampler_check_summary.csv"),
        [r for r in summary if r["kind"] == "contrast"],
    )
    for row in summary:
        if row["kind"] == "level":
            print(
                f"{row['sampler']:3s} {row['treatment']:26s} feas={row['raw_feasible_rate']:.3f} "
                f"nontrivial={row['raw_nontrivial_feasible_rate']:.3f} "
                f"gap={row['repaired_optimality_gap_pct']:6.2f}%"
            )
        elif row["metric"] in (
            "raw_nontrivial_feasible_rate",
            "repaired_optimality_gap_pct",
        ):
            print(
                f"    {row['treatment']} - {row['baseline']} {row['metric']}: "
                f"{row['mean_paired_difference']:+.4f} "
                f"[{row['bootstrap_95_ci_low']:+.4f}, {row['bootstrap_95_ci_high']:+.4f}]"
            )
    if any(math.isnan(float(r["repaired_optimality_gap_pct"])) for r in rows):
        print("warning: some repaired gaps are NaN")


def _solve(args):
    topology_names = [name for name, _ in _topology_cases(True)]
    triples = sample_instances(
        topology_names,
        INSTANCE_SEEDS,
        REQUEST_COUNTS,
        None if args.full else args.n_instances,
    )
    if args.workers > 1:
        shards = [triples[i :: args.workers] for i in range(args.workers)]
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            rows = [
                r for shard in pool.map(_run_shard, shards, chunksize=1) for r in shard
            ]
    else:
        rows = run_instances(dict(_topology_cases(True)), triples)
    rows.sort(
        key=lambda r: (
            r["topology"],
            r["instance_seed"],
            r["n_requests"],
            r["arm"],
            r["sampler"],
        )
    )
    return rows


if __name__ == "__main__":
    main()
