"""Randomized and validation tests for the per-resource penalty calibration.

The case counts can be raised for a heavier local run, for example
``QIOPT_RANDOM_CASES=15000 pytest tests/test_penalty_calibration_followups.py``.
"""

import itertools
import math
import os
import random

import numpy as np
import pytest

from experiments.run_penalty_calibration import coefficient_stats, run_benchmark
from optimization.openjij_solver import calibrated_coefficients
from optimization.proposed_calibrator import (
    _demands_by_request,
    _penalty_drop,
    possible_loads,
    resource_bounds,
)
from optimization.qubo_optimizer import QUBOOptimizer

N_CASES = int(os.environ.get("QIOPT_RANDOM_CASES", "300"))
N_GROUND_STATE = 50
MAX_QUBO_VARIABLES = 14  # bundles plus slack bits; keeps 2**n enumeration cheap


def _random_pairs(rng):
    """Random (key, demand) pairs for one resource: several bundles per request."""
    pairs = []
    for r in range(rng.randint(1, 5)):
        for b in range(rng.randint(1, 3)):
            pairs.append(((f"r{r}", f"b{r}_{b}"), rng.randint(0, 5)))
    return pairs


def _uncached_bounds(pairs, capacity, utilities):
    """Reference bound: recompute the loads for every candidate, uncapped."""
    bound = 0.0
    for key, demand in pairs:
        if demand <= 0:
            continue
        loads = possible_loads(pairs, key[0])  # full enumeration, no cap
        violating = [load for load in loads if load + demand > capacity]
        if not violating:
            continue
        delta = _penalty_drop(min(violating), demand, capacity)
        if delta > 0:
            bound = max(bound, max(0.0, utilities[key]) / delta)
    return bound


def test_possible_loads_cache_key_is_request_only():
    """resource_bounds caches loads per excluded request (and caps the DP).

    That must agree with recomputing the uncapped loads for every candidate.
    """
    rng = random.Random(20240921)
    for _ in range(N_CASES):
        pairs = _random_pairs(rng)
        capacity = rng.randint(0, 8)
        utilities = {key: rng.uniform(0.0, 20.0) for key, _ in pairs}

        cached = resource_bounds({"res": pairs}, {"res": capacity}, utilities)["res"]
        reference = _uncached_bounds(pairs, capacity, utilities)
        assert cached == pytest.approx(reference), (pairs, capacity)


def test_possible_loads_capped_result_is_candidate_independent():
    """The capped set must contain the smallest violating load for any candidate."""
    rng = random.Random(7)
    for _ in range(N_CASES):
        pairs = _random_pairs(rng)
        capacity = rng.randint(0, 8)
        for request_id in {key[0] for key, _ in pairs}:
            full = possible_loads(pairs, request_id)
            capped = possible_loads(pairs, request_id, capacity=capacity)
            assert capped <= full
            for demand in range(1, 6):
                full_min = min((v for v in full if v + demand > capacity), default=None)
                capped_min = min((v for v in capped if v + demand > capacity), default=None)
                assert capped_min == full_min


def test_possible_loads_rejects_negative_capacity():
    with pytest.raises(ValueError, match="capacity must be nonnegative"):
        possible_loads([(("r0", "b0"), 1)], "r1", capacity=-1)


def test_candidate_demand_argument_is_gone():
    with pytest.raises(TypeError):
        possible_loads([(("r0", "b0"), 1)], "r1", capacity=1, candidate_demand=1)
    assert _demands_by_request([(("r0", "b0"), 1)], "r1") == {"r0": {0, 1}}


# --------------------------------------------------------------------------
# Randomized ground-state correspondence on the compiled QUBO
# --------------------------------------------------------------------------

def _random_instance(rng):
    # Small capacities keep the number of slack bits (and 2**n) low.
    caps = {("A", "B"): rng.randint(2, 3), ("B", "C"): rng.randint(2, 3)}
    mems = {"B": rng.randint(2, 3)}
    bundles = []
    for r in range(rng.randint(2, 3)):
        for b in range(rng.randint(1, 2)):
            if len(bundles) >= 5:
                break
            edge = {e: d for e in caps if (d := rng.randint(0, 3))}
            mem = {n: d for n in mems if (d := rng.randint(0, 3))}
            bundles.append({
                "bundle_id": f"b{r}_{b}",
                "request_id": f"r{r}",
                "path": ["A", "B", "C"],
                "edge_demands": edge,
                "memory_demands": mem,
                "utility": float(rng.randint(1, 15)),
            })
    return bundles, caps, mems


def _capacity_ok(group, caps, mems):
    loads, mem = {}, {}
    for b in group:
        for e, d in b["edge_demands"].items():
            loads[e] = loads.get(e, 0) + d
        for n, d in b["memory_demands"].items():
            mem[n] = mem.get(n, 0) + d
    return (all(v <= caps[e] for e, v in loads.items())
            and all(v <= mems[n] for n, v in mem.items()))


def _ground_state(opt, coeffs):
    """Brute-force the ground state of ``opt.to_qubo`` with the given coefficients."""
    qubo, offset = opt.to_qubo(
        penalty=coeffs["A"],
        edge_penalty=coeffs["B"],
        memory_penalty=coeffs["D"],
        congestion_penalty=coeffs["C"],
        memory_congestion_penalty=coeffs["E"],
    )
    names = sorted({v for pair in qubo for v in pair})
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    if n > MAX_QUBO_VARIABLES:
        return None

    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    energy = np.full(2 ** n, float(offset))
    for (i, j), weight in qubo.items():
        energy += weight * bits[:, index[i]] * bits[:, index[j]]
    best = int(np.argmin(energy))
    assignment = {name: int(bits[best, index[name]]) for name in names}
    sample = {name: assignment.get(name, 0) for name in opt.variable_map}
    return opt.decode_sample(sample)


def _best_feasible_utility(bundles, caps, mems):
    best = 0.0
    for size in range(len(bundles) + 1):
        for group in itertools.combinations(bundles, size):
            distinct = len({b["request_id"] for b in group}) == len(group)
            if distinct and _capacity_ok(group, caps, mems):
                best = max(best, sum(b["utility"] for b in group))
    return best


@pytest.mark.parametrize("strategy", ["resource_aware", "resource_aware_per"])
def test_randomized_ground_state_is_optimal_without_congestion(strategy):
    rng = random.Random(1234)
    tested = 0
    for _ in range(N_GROUND_STATE):
        bundles, caps, mems = _random_instance(rng)
        opt = QUBOOptimizer(bundles, caps, mems)
        coeffs = calibrated_coefficients(opt, strategy)  # C = E = 0
        selected = _ground_state(opt, coeffs)
        if selected is None:
            continue
        tested += 1

        by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
        chosen = [by_key[key] for key in selected]
        assert len(selected) == len({rid for rid, _ in selected})
        assert _capacity_ok(chosen, caps, mems)
        assert sum(b["utility"] for b in chosen) == pytest.approx(
            _best_feasible_utility(bundles, caps, mems)
        )
    assert tested >= N_GROUND_STATE // 2


@pytest.mark.parametrize("strategy", ["resource_aware", "resource_aware_per"])
def test_randomized_ground_state_is_feasible_with_congestion(strategy):
    """With C, E > 0 the ground state need not be optimal, only feasible.

    Congestion penalises feasible loads, so a smaller selection can win; the
    hard-constraint guarantee (no overload, at most one bundle per request)
    is what must survive.
    """
    rng = random.Random(4321)
    tested = 0
    for _ in range(N_GROUND_STATE):
        bundles, caps, mems = _random_instance(rng)
        opt = QUBOOptimizer(bundles, caps, mems)
        coeffs = calibrated_coefficients(
            opt, strategy, congestion_penalty=0.05, memory_congestion_penalty=0.05
        )
        selected = _ground_state(opt, coeffs)
        if selected is None:
            continue
        tested += 1

        by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
        chosen = [by_key[key] for key in selected]
        assert len(selected) == len({rid for rid, _ in selected})
        assert _capacity_ok(chosen, caps, mems)
    assert tested >= N_GROUND_STATE // 2


# --------------------------------------------------------------------------
# Per-resource input validation
# --------------------------------------------------------------------------

def _tiny_optimizer():
    bundles = [
        {"bundle_id": "b0", "request_id": "r0", "path": ["A", "B"],
         "edge_demands": {("A", "B"): 3}, "memory_demands": {"A": 1}, "utility": 5.0},
        {"bundle_id": "b1", "request_id": "r1", "path": ["A", "B"],
         "edge_demands": {("A", "B"): 3}, "memory_demands": {"A": 1}, "utility": 4.0},
    ]
    return QUBOOptimizer(bundles, {("A", "B"): 4}, {"A": 4})


def test_unknown_edge_key_is_rejected():
    opt = _tiny_optimizer()
    with pytest.raises(ValueError, match="unknown resources"):
        opt.to_qubo(
            penalty=1.0,
            edge_penalty={("A", "B"): 1.0, ("A", "Z"): 1.0},
            memory_penalty=1.0,
        )


def test_unknown_memory_key_is_rejected():
    opt = _tiny_optimizer()
    with pytest.raises(ValueError, match="memory_penalty has coefficients for 1 unknown"):
        opt.to_qubo(
            penalty=1.0, edge_penalty=1.0, memory_penalty={"A": 1.0, "typo": 2.0}
        )


def test_reversed_edge_orientation_is_still_accepted():
    opt = _tiny_optimizer()
    forward = opt.to_qubo(penalty=1.0, edge_penalty={("A", "B"): 2.0}, memory_penalty=1.0)
    reverse = opt.to_qubo(penalty=1.0, edge_penalty={("B", "A"): 2.0}, memory_penalty=1.0)
    assert forward == reverse


def test_duplicate_edge_orientation_is_rejected():
    opt = _tiny_optimizer()
    duplicated = {("A", "B"): 1.0, ("B", "A"): 9.0}
    with pytest.raises(ValueError, match="more than one entry"):
        opt.to_qubo(penalty=1.0, edge_penalty=duplicated, memory_penalty=1.0)
    with pytest.raises(ValueError, match="more than one entry"):
        opt.solution_energy([], edge_penalty=duplicated)


def test_feed_is_usable_immediately_after_construction():
    opt = _tiny_optimizer()
    feed = opt._feed(1.0, 2.0, 3.0, 0.0, 0.0)
    assert feed["B_0"] == 2.0 and feed["D_0"] == 3.0


# --------------------------------------------------------------------------
# Driver support for mapping-valued coefficients
# --------------------------------------------------------------------------

def test_coefficient_stats_handles_scalar_and_mapping():
    assert coefficient_stats(2.5) == {"mean": 2.5, "min": 2.5, "max": 2.5, "n": 1}
    stats = coefficient_stats({"a": 1.0, "b": 3.0})
    assert stats == {"mean": 2.0, "min": 1.0, "max": 3.0, "n": 2}
    assert coefficient_stats({})["n"] == 0 or coefficient_stats({})["mean"] == 0.0


def test_driver_runs_per_resource_strategy(tmp_path, monkeypatch):
    from experiments import run_penalty_calibration as driver
    from experiments.instances import generate_chain_topology

    monkeypatch.setattr(
        driver, "_topology_cases",
        lambda full: [("chain5", lambda: generate_chain_topology(
            n_nodes=5, edge_capacity=3, memory_capacity=6,
            raw_fidelity=0.85, generation_prob=0.95))],
    )
    # This test exercises the CSV/row plumbing, not the samplers: SQA on the
    # driver's fixed 8/16-request instances is slow, so reuse SA for it.
    monkeypatch.setattr(driver, "solve_sqa", driver.solve_sa)
    rows = run_benchmark(
        out_dir=str(tmp_path),
        num_reads=2,
        strategies=["conventional", "resource_aware_per"],
    )
    per = [r for r in rows if r["calibration"] == "resource_aware_per"]
    assert per, "per-resource strategy produced no rows"
    for row in per:
        assert all(math.isfinite(float(row[k])) for k in ("B", "B_min", "B_max", "D"))
        assert row["B_min"] <= row["B"] <= row["B_max"]
        assert row["B_n"] >= 1
    conventional = [r for r in rows if r["calibration"] == "conventional"]
    assert all(r["B_n"] == 1 and r["B_min"] == r["B_max"] == r["B"] for r in conventional)
    assert (tmp_path / "calibration_runs.csv").exists()
