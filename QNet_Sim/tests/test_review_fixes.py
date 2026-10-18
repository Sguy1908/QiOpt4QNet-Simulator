"""Regression tests for the fixes in the review of the npj manuscript branch."""

import itertools
import math
import random

import dimod
import pytest

from experiments.analyze_penalty_calibration_heldout import add_nontrivial_feasibility
from experiments.run_penalty_calibration import summarize_response
from optimization.openjij_solver import build_calibrated_bqm, calibrated_coefficients
from optimization.proposed_calibrator import pairwise_request_penalties
from optimization.qubo_optimizer import QUBOOptimizer
from optimization.time_dependent_optimizer import fidelity_after_hold
from routing.memory_scheduler import EntangledPair


def _bundle(bid, rid, utility, edges, memory):
    return {
        "bundle_id": bid,
        "request_id": rid,
        "path": [],
        "edge_demands": edges,
        "memory_demands": memory,
        "utility": utility,
    }


# --- Werner memory decay -------------------------------------------------


@pytest.mark.parametrize("f0", [0.6, 0.85, 1.0])
def test_hold_decay_matches_manuscript_and_floors_at_quarter(f0):
    t, t2 = 3.0, 5.0
    eta = math.exp(-t / t2) * (1.0 + math.exp(-t / (2 * t2))) / 2.0
    assert fidelity_after_hold(f0, t, t2) == pytest.approx(0.25 + (f0 - 0.25) * eta)
    assert fidelity_after_hold(f0, 1e9, t2) == pytest.approx(0.25)


def test_stored_pair_decays_to_mixed_state_not_below():
    pair = EntangledPair("p", "N0", generation_time=0.0, initial_fidelity=0.8, t2=2.0)
    assert pair.fidelity_at(0.0) == pytest.approx(0.8)
    assert pair.fidelity_at(1e9) == pytest.approx(0.25)
    assert pair.fidelity_at(1.0) == pytest.approx(
        0.25 + 0.55 * math.exp(-0.5) * (1 + math.exp(-0.25)) / 2
    )


# --- QUBO defaults and repair --------------------------------------------


def _two_request_optimizer():
    bundles = [
        _bundle("b0", "r1", 1.0, {("A", "B"): 2}, {"A": 1}),
        _bundle("b1", "r1", 5.0, {("A", "B"): 1}, {"A": 1}),
        _bundle("b0", "r2", 3.0, {("A", "B"): 2}, {"B": 1}),
    ]
    return QUBOOptimizer(bundles, {("A", "B"): 3}, {"A": 2, "B": 2})


def test_to_bqm_default_has_no_soft_congestion():
    opt = _two_request_optimizer()
    default = opt.to_bqm()
    explicit = opt.to_bqm(congestion_penalty=0.0, memory_congestion_penalty=0.0)
    assert default == explicit


def test_repair_keeps_highest_utility_bundle_of_a_request_regardless_of_order():
    opt = _two_request_optimizer()
    raw = [("r1", "b1"), ("r1", "b0")]
    assert opt.repair_selection(raw) == [("r1", "b1")]
    assert opt.repair_selection(list(reversed(raw))) == [("r1", "b1")]


def test_solution_energy_defaults_score_the_sampled_hamiltonian():
    opt = _two_request_optimizer()
    bqm = opt.to_bqm()
    names = {key: name for name, key in opt.variable_map.items()}
    selection = [("r1", "b1"), ("r2", "b0")]  # edge load 3 == capacity
    chosen = {names[key] for key in selection}
    # minimise over slack bits for this assignment of the bundle variables
    slack = [v for v in bqm.variables if v not in opt.variable_map]
    best = min(
        bqm.energy(
            {
                **{v: int(v in chosen) for v in opt.variable_map},
                **dict(zip(slack, bits)),
            }
        )
        for bits in itertools.product((0, 1), repeat=len(slack))
    )
    assert opt.solution_energy(selection) == pytest.approx(best)


# --- calibration rules keep the ground state ------------------------------


def _random_instance(rng):
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    nodes = ["a", "b", "c", "d"]
    bundles = []
    for r in range(rng.randint(2, 3)):
        for k in range(rng.randint(1, 3)):
            bundles.append(
                _bundle(
                    f"b{k}",
                    f"r{r}",
                    round(rng.uniform(-0.5, 3.0), 3),
                    {
                        e: rng.randint(1, 3)
                        for e in rng.sample(edges, rng.randint(1, 2))
                    },
                    {
                        v: rng.randint(1, 2)
                        for v in rng.sample(nodes, rng.randint(1, 2))
                    },
                )
            )
    return (
        bundles,
        {e: rng.randint(1, 4) for e in edges},
        {v: rng.randint(1, 3) for v in nodes},
    )


def _best_feasible_utility(bundles, ecap, mcap):
    by_request = {}
    for b in bundles:
        by_request.setdefault(b["request_id"], [None]).append(b)
    best = 0.0
    for choice in itertools.product(*by_request.values()):
        load_e, load_m, utility = {}, {}, 0.0
        for b in choice:
            if b is None:
                continue
            utility += b["utility"]
            for e, d in b["edge_demands"].items():
                load_e[e] = load_e.get(e, 0) + d
            for v, d in b["memory_demands"].items():
                load_m[v] = load_m.get(v, 0) + d
        if all(load_e[e] <= ecap[e] for e in load_e) and all(
            load_m[v] <= mcap[v] for v in load_m
        ):
            best = max(best, utility)
    return best


@pytest.mark.parametrize(
    "strategy,pairwise",
    [("local_utility", False), ("resource_aware_per", True)],
)
def test_new_calibration_rules_keep_feasible_optimal_ground_state(strategy, pairwise):
    rng = random.Random(5)
    checked = 0
    while checked < 25:
        bundles, ecap, mcap = _random_instance(rng)
        opt = QUBOOptimizer(bundles, ecap, mcap)
        bqm, _ = build_calibrated_bqm(opt, strategy, pairwise_request_penalty=pairwise)
        if bqm.num_variables > 16:
            continue
        checked += 1
        # rtol=0: the epsilon margin (1e-6 * P0) is below dimod's default rtol.
        ground = dimod.ExactSolver().sample(bqm).lowest(rtol=0.0, atol=1e-9)
        target = _best_feasible_utility(bundles, ecap, mcap)
        for sample in ground.samples():
            selected = opt.decode_sample(sample)
            assert len({r for r, _ in selected}) == len(selected)
            utility = sum(
                b["utility"]
                for b in bundles
                if (b["request_id"], b["bundle_id"]) in selected
            )
            assert utility == pytest.approx(target)


def test_local_utility_rule_is_per_resource_and_at_most_conventional():
    opt = _two_request_optimizer()
    local = calibrated_coefficients(opt, "local_utility")
    conventional = calibrated_coefficients(opt, "conventional")
    assert isinstance(local["B"], dict) and isinstance(local["D"], dict)
    assert local["D"]["B"] == pytest.approx(3.0, rel=1e-5)  # only r2 uses memory B
    assert all(v <= conventional["B"] + 1e-12 for v in local["B"].values())


def test_pairwise_request_penalty_uses_smaller_utility_of_the_pair():
    opt = _two_request_optimizer()
    pairs = pairwise_request_penalties(opt)
    assert list(pairs.values()) == [pytest.approx(1.0, abs=1e-5)]


# --- non-trivial feasibility ----------------------------------------------


class _Response:
    def __init__(self, samples):
        self._samples = samples

    def samples(self):
        return iter(self._samples)


def test_nontrivial_feasibility_excludes_the_empty_selection():
    opt = _two_request_optimizer()
    empty = {name: 0 for name in opt.variable_map}
    useful = dict(empty)
    useful[next(n for n, k in opt.variable_map.items() if k == ("r1", "b1"))] = 1
    stats = summarize_response(
        opt,
        _Response([empty, useful]),
        opt.bundles,
        opt.edge_capacities,
        opt.memory_capacities,
    )
    assert stats["raw_feasible_rate"] == 1.0
    assert stats["raw_nontrivial_feasible_rate"] == 0.5


def test_archived_rows_get_nontrivial_feasibility():
    rows = [
        {"raw_feasible_rate": "1.0", "raw_best_feasible_utility": "0.0"},
        {"raw_feasible_rate": "1.0", "raw_best_feasible_utility": "2.5"},
        {"raw_feasible_rate": "0.0", "raw_best_feasible_utility": ""},
    ]
    out = [r["raw_nontrivial_feasible_rate"] for r in add_nontrivial_feasibility(rows)]
    assert out == [0.0, 1.0, 0.0]
