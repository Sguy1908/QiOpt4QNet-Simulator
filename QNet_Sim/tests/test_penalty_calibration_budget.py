"""Tests for the solver-budget sensitivity runner on a tiny configuration."""

import pytest

from experiments.instances import generate_chain_topology
from experiments.run_penalty_calibration_budget import (
    count_distinct,
    draw_reads,
    run_budget_study,
    sample_instances,
    summarize,
)
from optimization.openjij_solver import build_calibrated_bqm, solve_sa
from optimization.qubo_optimizer import QUBOOptimizer


def _tiny_topology():
    return generate_chain_topology(
        n_nodes=6, edge_capacity=3, memory_capacity=6,
        raw_fidelity=0.85, generation_prob=0.95,
    )


def test_sample_instances_is_seeded_and_bounded():
    topologies = {"a": None, "b": None}
    grid = sample_instances(topologies, [1, 2, 3], [8, 16], None)
    assert len(grid) == 2 * 3 * 2
    first = sample_instances(topologies, [1, 2, 3], [8, 16], 5, seed=3)
    again = sample_instances(topologies, [1, 2, 3], [8, 16], 5, seed=3)
    assert first == again and len(first) == 5 and set(first) <= set(grid)


def test_solve_sa_num_sweeps_is_optional_and_reaches_solver():
    from experiments.instances import contention_sweep_instances

    inst = contention_sweep_instances(_tiny_topology, [3], seed=10)["req3"]
    opt = QUBOOptimizer(inst["bundles"], inst["edge_capacities"], inst["memory_capacities"])
    bqm, _ = build_calibrated_bqm(opt, "conventional")

    for reads in (4, 9):
        assert len(solve_sa(bqm, num_reads=reads, seed=1).record) == reads
    # Extra sweeps are accepted and still return one sample per read.
    assert len(solve_sa(bqm, num_reads=4, seed=1, num_sweeps=50).record) == 4


def test_budget_study_reports_matching_sample_counts():
    triples = [("chain6", seed, 3) for seed in (10, 11, 12, 13)]
    rows = run_budget_study(
        {"chain6": _tiny_topology},
        triples,
        budgets=[(4, None), (8, None), (4, 40)],
        solver_seeds=(101,),
        request_counts=[3],
        log=lambda *_: None,
    )
    if not rows:
        pytest.skip("no tiny instance was tightened by the per-resource rule")

    assert all(row["n_samples"] == row["reads"] for row in rows)
    assert {row["reads"] for row in rows} == {4, 8}
    assert {row["sweeps"] for row in rows} == {"default", 40}

    summary = summarize(rows, n_boot=50)
    assert summary and all(record["samples_match_reads"] for record in summary)
    for record in summary:
        assert record["repaired_gap_pct_ci_low"] <= record["repaired_gap_pct_ci_high"]


def test_independent_reads_return_requested_count_and_diagnostic_is_bounded():
    from experiments.instances import contention_sweep_instances

    inst = contention_sweep_instances(_tiny_topology, [3], seed=10)["req3"]
    opt = QUBOOptimizer(inst["bundles"], inst["edge_capacities"], inst["memory_capacities"])
    bqm, _ = build_calibrated_bqm(opt, "conventional")

    for independent in (False, True):
        response = draw_reads("sa", bqm, 6, seed=101, independent=independent)
        assert len(list(response.samples())) == 6
        assert 1 <= count_distinct(response) <= 6

    # Reproducible for a fixed seed in both modes.
    a = draw_reads("sa", bqm, 6, seed=101, independent=True)
    b = draw_reads("sa", bqm, 6, seed=101, independent=True)
    assert list(a.samples()) == list(b.samples())


def test_budget_study_records_distinct_samples():
    rows = run_budget_study(
        {"chain6": _tiny_topology},
        [("chain6", seed, 3) for seed in (10, 11)],
        budgets=[(4, None)],
        solver_seeds=(101,),
        request_counts=[3],
        independent_reads=True,
        log=lambda *_: None,
    )
    if not rows:
        pytest.skip("no tiny instance was tightened by the per-resource rule")
    assert all(1 <= row["n_distinct_samples"] <= row["reads"] for row in rows)
    assert all(row["independent_reads"] for row in rows)



def test_parallel_run_matches_sequential():
    from experiments.run_penalty_calibration import _topology_cases
    from experiments.run_penalty_calibration_budget import run_budget_study_parallel

    triples = [("chain8_c4", 10, 8), ("chain8_c4", 10, 16), ("chain8_c4", 10, 24)]
    budgets = [(3, None)]
    sequential = run_budget_study(
        dict(_topology_cases(True)), triples, budgets, ("sa",), (101,), log=lambda _: None
    )
    parallel = run_budget_study_parallel(triples, budgets, ("sa",), (101,), False, 2)

    def key(row):
        return (row["n_requests"], row["calibration"])

    assert sequential, "expected at least one tightened instance"
    assert sorted(sequential, key=key) == sorted(parallel, key=key)


@pytest.mark.parametrize("sampler", ["sa", "sqa"])
def test_fixed_seed_reads_are_identical_but_reach_the_solver(sampler):
    """Documents why extra reads change nothing under a fixed seed.

    docs/PENALTY_CALIBRATION.md relies on this: ``num_reads`` reaches OpenJij
    (the requested number of samples comes back) but every read is the same
    state. If this test starts failing, revisit the read-budget discussion.
    """
    from experiments.instances import contention_sweep_instances

    inst = contention_sweep_instances(_tiny_topology, [3], seed=10)["req3"]
    opt = QUBOOptimizer(inst["bundles"], inst["edge_capacities"], inst["memory_capacities"])
    bqm, _ = build_calibrated_bqm(opt, "conventional")

    for reads in (1, 20, 100):
        response = draw_reads(sampler, bqm, reads, seed=101)
        assert len(list(response.samples())) == reads
        assert count_distinct(response) == 1
