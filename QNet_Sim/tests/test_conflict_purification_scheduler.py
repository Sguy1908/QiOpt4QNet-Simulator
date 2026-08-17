"""Tests for network-level, conflict-aware purification scheduling (PSC)."""

import random

import pytest

from experiments.instances import generate_chain_topology
from optimization.conflict_purification_scheduler import (
    ConflictAwarePurificationScheduler,
    PSCRequest,
    greedy_baseline,
    purification_need,
    route_requests,
    run_psc_comparison,
    threshold_baseline,
)


def _topology(n_nodes=6, edge_capacity=2, raw=0.75):
    return generate_chain_topology(n_nodes=n_nodes, edge_capacity=edge_capacity,
                                    memory_capacity=10, raw_fidelity=raw,
                                    generation_prob=1.0)


def _shared_bottleneck_pairs():
    # N0 - N1 - N2 - N3 on a chain: req0 (N0,N2) and req1 (N1,N3) both cross
    # (N1, N2) but each also has a private edge that alone suffices.
    return [("N0", "N2", 10.0, 0.60), ("N1", "N3", 10.0, 0.60)]


def test_route_requests_finds_shortest_path():
    topo = _topology()
    reqs = route_requests(topo, [("N0", "N3", 10.0, 0.6)])
    assert len(reqs) == 1
    assert reqs[0].path == ["N0", "N1", "N2", "N3"]


def test_route_requests_drops_unreachable_pairs():
    topo = _topology()
    reqs = route_requests(topo, [("N0", "N9", 10.0, 0.6)])
    assert reqs == []


def test_purification_need_zero_off_path():
    topo = _topology()
    req = PSCRequest("r0", ["N0", "N1", "N2"], min_fidelity=0.6)
    assert purification_need(topo, req, ("N3", "N4")) == 0.0


def test_purification_need_zero_when_already_satisfied():
    topo = _topology(raw=0.9)
    req = PSCRequest("r0", ["N0", "N1", "N2"], min_fidelity=0.5)
    assert purification_need(topo, req, ("N0", "N1")) == 0.0


def test_purification_need_positive_when_helpful():
    topo = _topology(raw=0.75)
    req = PSCRequest("r0", ["N0", "N1", "N2"], min_fidelity=0.60)
    need = purification_need(topo, req, ("N0", "N1"))
    assert 0.0 < need <= 1.0


def test_psc_matches_baseline_throughput_with_fewer_resources():
    """On a shared-bottleneck instance, PSC should serve every request the
    naive baselines serve, but at strictly lower Bell-pair cost: purifying
    the link both requests share once is cheaper than each request
    independently purifying (and, for the threshold baseline, sometimes
    redundantly re-attempting) its own link.
    """
    topo = _topology(n_nodes=6, edge_capacity=2, raw=0.75)
    pairs = _shared_bottleneck_pairs()
    result = run_psc_comparison(topo, pairs, bell_pairs_per_purification=2, seed=0)

    psc, threshold, greedy = result["psc"], result["threshold"], result["greedy"]
    assert psc["throughput"] == threshold["throughput"] == greedy["throughput"] == 2
    assert psc["purification_cost"] < threshold["purification_cost"]
    assert psc["purification_cost"] < greedy["purification_cost"]
    assert psc["resource_consumption_ratio"] < threshold["resource_consumption_ratio"]
    assert psc["resource_consumption_ratio"] < greedy["resource_consumption_ratio"]

    # the coordinated scheduler resolves the shared bottleneck with a single
    # purification event that both requests benefit from
    shared_edge = ("N1", "N2")
    assert shared_edge in psc["per_request"]["req_0"]["purified_links"]
    assert shared_edge in psc["per_request"]["req_1"]["purified_links"]


def test_psc_deterministic_given_seed():
    topo = _topology()
    reqs = route_requests(topo, _shared_bottleneck_pairs())
    r1 = ConflictAwarePurificationScheduler(topo, reqs, seed=7).schedule()
    r2 = ConflictAwarePurificationScheduler(topo, reqs, seed=7).schedule()
    assert r1["throughput"] == r2["throughput"]
    assert r1["purification_cost"] == r2["purification_cost"]
    for rid in r1["per_request"]:
        assert r1["per_request"][rid]["success"] == r2["per_request"][rid]["success"]


def test_psc_never_exceeds_available_edge_budget():
    """Sanity check: the scheduler must not purify more Bell pairs on any
    link than that link's edge capacity allows."""
    topo = _topology(n_nodes=8, edge_capacity=2, raw=0.7)
    rng = random.Random(3)
    nodes = topo["nodes"]
    pairs = []
    for _ in range(5):
        s, d = rng.sample(nodes, 2)
        if nodes.index(s) > nodes.index(d):
            s, d = d, s
        pairs.append((s, d, 10.0, rng.uniform(0.55, 0.75)))
    reqs = route_requests(topo, pairs)
    scheduler = ConflictAwarePurificationScheduler(topo, reqs, seed=3)
    result = scheduler.schedule()
    for edge, budget in scheduler.budgets.items():
        assert budget.remaining >= 0
        assert budget.remaining <= budget.capacity


def test_psc_beats_naive_baselines_in_aggregate():
    """Statistical claim (mirrors the PSC literature's own methodology of
    averaging over many randomly generated networks): across many random
    scenarios, PSC's aggregate throughput should be at least as good as
    each naive baseline's, and its aggregate cost-per-success should be
    lower. Individual random instances may go either way (PSC is a
    heuristic, not an exact solver), so the claim is checked in aggregate
    rather than per-instance.
    """
    totals = {"psc": [0, 0.0], "threshold": [0, 0.0], "greedy": [0, 0.0]}
    n_trials = 60
    for trial in range(n_trials):
        rng = random.Random(trial)
        n_nodes = rng.randint(5, 9)
        cap = rng.choice([2, 4])
        raw = rng.uniform(0.65, 0.85)
        topo = _topology(n_nodes=n_nodes, edge_capacity=cap, raw=raw)
        nodes = topo["nodes"]
        pairs = []
        for _ in range(rng.randint(2, 5)):
            s, d = rng.sample(nodes, 2)
            if nodes.index(s) > nodes.index(d):
                s, d = d, s
            pairs.append((s, d, 10.0, rng.uniform(0.55, 0.75)))
        result = run_psc_comparison(topo, pairs, bell_pairs_per_purification=2, seed=trial)
        for name in totals:
            totals[name][0] += result[name]["throughput"]
            totals[name][1] += result[name]["purification_cost"]

    psc_throughput, psc_cost = totals["psc"]
    for name in ("threshold", "greedy"):
        base_throughput, base_cost = totals[name]
        assert psc_throughput >= base_throughput
        assert (psc_cost / psc_throughput) < (base_cost / base_throughput)


def test_baseline_functions_agree_with_comparison_runner():
    topo = _topology()
    pairs = _shared_bottleneck_pairs()
    reqs = route_requests(topo, pairs)
    combined = run_psc_comparison(topo, pairs, seed=1)
    assert threshold_baseline(topo, reqs)["throughput"] == combined["threshold"]["throughput"]
    assert greedy_baseline(topo, reqs)["throughput"] == combined["greedy"]["throughput"]
