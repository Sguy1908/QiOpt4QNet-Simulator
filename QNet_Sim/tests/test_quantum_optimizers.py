from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
from optimization.sequential_branch_optimizer import SequentialBranchOptimizer
from optimization.qubo_optimizer import QUBOOptimizer
from optimization.refine_pipeline import TensorAnnealerPipeline
from optimization.baselines import utility_density_greedy, fidelity_aware_greedy


def _make_bundle(bundle_id, request_id, utility, edge_demands, memory_demands, path=None):
    return {
        "bundle_id": bundle_id,
        "request_id": request_id,
        "path": path or ["A", "R", "B"],
        "edge_demands": edge_demands,
        "memory_demands": memory_demands,
        "utility": utility,
    }


def test_metropolis_single_request_selects_best():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
        _make_bundle("b2", "r1", 10.0, {("A", "R"): 4, ("R", "B"): 4}, {"A": 4, "R": 8, "B": 4}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=1000)
    selected = {rid for rid, bid in result["selected"]}
    assert "r1" in selected
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b0"


def test_metropolis_two_requests():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=1000)
    assert len(result["selected"]) == 2


def test_metropolis_default_penalties_anchor_to_utility_scale():
    # With utilities above 100, the old fixed defaults (B=10) would silently
    # prefer a capacity-violating bundle; the utility-scale default must not.
    bundles = [
        _make_bundle("b0", "r1", 150.0, {("A", "R"): 6, ("R", "B"): 6}, {"A": 6, "R": 6, "B": 6}),
        _make_bundle("b1", "r1", 100.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=2000)  # no explicit penalties
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"  # feasible bundle, not the violating one


def test_metropolis_respects_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=2000, penalty=100.0, edge_penalty=50.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_metropolis_greedy_seed():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    seed = opt._greedy_seed()
    assert seed["r1"] == "b1"


def test_tensor_network_single_request():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b0"


def test_tensor_network_two_requests():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r2", 10.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    assert len(result["selected"]) == 2


def test_tensor_network_respects_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(edge_penalty=50.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_feasibility_repair():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    selections = {"r1": "b0"}
    repaired = opt._feasibility_repair(selections)
    assert repaired["r1"] == "b1"

    selections2 = {"r1": "b1"}
    repaired2 = opt._feasibility_repair(selections2)
    assert repaired2["r1"] == "b1"

    selections3 = {"r1": None}
    repaired3 = opt._feasibility_repair(selections3)
    assert repaired3["r1"] is None


def test_validates_missing_fields():
    import pytest
    with pytest.raises(ValueError, match="missing"):
        MetropolisAnnealer([{"bundle_id": "b0"}], {}, {})


def test_tensor_network_validates_missing_fields():
    import pytest
    with pytest.raises(ValueError, match="missing"):
        TensorNetworkOptimizer([{"bundle_id": "b0"}], {}, {})


def _congested_bundles():
    # Two requests whose high-utility bundles concentrate load on a shared
    # edge (E,R); low-utility alternatives use less capacity.
    return [
        _make_bundle("b_hi", "r1", 60.0, {("E", "R"): 2}, {"M": 1}),
        _make_bundle("b_lo", "r1", 30.0, {("E", "R"): 1}, {"M": 1}),
        _make_bundle("b_hi", "r2", 50.0, {("E", "R"): 2}, {"M": 1}),
        _make_bundle("b_lo", "r2", 20.0, {("E", "R"): 1}, {"M": 1}),
    ]


def _load_from_selections(opt, selections):
    edge_load = {e: 0 for e in opt.edge_capacities}
    mem_load = {n: 0 for n in opt.memory_capacities}
    for b in opt.bundles:
        if selections.get(b["request_id"]) == b["bundle_id"]:
            for e, d in b["edge_demands"].items():
                edge_load[tuple(sorted(e))] = edge_load.get(tuple(sorted(e)), 0) + d
            for n, d in b["memory_demands"].items():
                mem_load[n] = mem_load.get(n, 0) + d
    return edge_load, mem_load


def test_metropolis_congestion_penalty_discourages_concentration():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=1)
    opt._B, opt._C = 100.0, 10.0
    opt._D, opt._E = 100.0, 0.0

    concentrated = {"r1": "b_hi", "r2": "b_hi"}   # load 4 == cap; congestion C*16
    spread = {"r1": "b_hi", "r2": "b_lo"}          # load 3 < cap; congestion C*9

    # With the congestion term the quadratic penalty flips the ordering
    assert opt._energy(spread) < opt._energy(concentrated)
    # without it the higher-utility concentrated choice wins.
    opt._C = 0.0
    assert opt._energy(concentrated) < opt._energy(spread)


def test_metropolis_delta_energy_matches_full_energy():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=7)
    selections = opt._greedy_seed()
    load, mem_load = _load_from_selections(opt, selections)

    rid = "r2"
    prev = selections[rid]
    new = "b_lo" if prev != "b_lo" else "b_hi"
    delta = opt._energy_delta(rid, prev, new, load, mem_load)

    old_state = dict(selections)
    new_state = dict(selections)
    new_state[rid] = new
    full_delta = opt._energy(new_state) - opt._energy(old_state)

    assert abs(delta - full_delta) < 1e-9


def test_metropolis_delta_apply_consistent():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=7)
    selections = opt._greedy_seed()
    load, mem_load = _load_from_selections(opt, selections)

    rid = "r2"
    prev = selections[rid]
    new = "b_lo" if prev != "b_lo" else "b_hi"
    opt._apply_delta(rid, prev, new, load, mem_load)
    selections[rid] = new

    expect_load, expect_mem = _load_from_selections(opt, selections)
    assert load == expect_load
    assert mem_load == expect_mem


def test_metropolis_restarts_find_feasible():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=500, penalty=100.0, edge_penalty=50.0,
                       memory_penalty=50.0, n_restarts=5)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_branch_expander_single_request():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
        _make_bundle("b2", "r1", 10.0, {("A", "R"): 4, ("R", "B"): 4}, {"A": 4, "R": 8, "B": 4}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b0"


def test_branch_expander_two_requests():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r2", 10.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    assert len(result["selected"]) == 2


def test_branch_expander_respects_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(edge_penalty=50.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_tensor_network_greedy_seed_respects_combined_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 2}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    seed = opt._greedy_seed()
    assert seed["r1"] == "b0"
    assert seed["r2"] is None


def test_metropolis_greedy_seed_respects_combined_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 2}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    seed = opt._greedy_seed()
    assert seed["r1"] == "b0"
    assert seed["r2"] is None


def test_branch_expander_deterministic():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    r1 = SequentialBranchOptimizer(bundles, edge_caps, mem_caps).solve()
    r2 = SequentialBranchOptimizer(bundles, edge_caps, mem_caps).solve()
    assert r1["selected"] == r2["selected"]


def _memory_congested_bundles():
    # Two requests whose high-utility bundles concentrate load on shared node M.
    return [
        _make_bundle("b_hi", "r1", 60.0, {("E", "R"): 1}, {"M": 2}),
        _make_bundle("b_lo", "r1", 30.0, {("E", "R"): 1}, {"M": 1}),
        _make_bundle("b_hi", "r2", 50.0, {("E", "R"): 1}, {"M": 2}),
        _make_bundle("b_lo", "r2", 20.0, {("E", "R"): 1}, {"M": 1}),
    ]


def test_metropolis_memory_congestion_penalty_flips_ordering():
    bundles = _memory_congested_bundles()
    edge_caps = {("E", "R"): 100}
    mem_caps = {"M": 4}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=1)
    opt._B, opt._C = 100.0, 0.0
    opt._D, opt._E = 100.0, 10.0

    concentrated = {"r1": "b_hi", "r2": "b_hi"}   # mem load 4 == cap; cong E*16
    spread = {"r1": "b_hi", "r2": "b_lo"}          # mem load 3; cong E*9

    assert opt._energy(spread) < opt._energy(concentrated)
    opt._E = 0.0
    assert opt._energy(concentrated) < opt._energy(spread)


def test_metropolis_respects_memory_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 100, ("R", "B"): 100}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=1000, penalty=100.0, edge_penalty=10.0,
                       memory_penalty=50.0, n_restarts=5)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_tensor_network_congestion_decomposition_matches_direct():
    # Local C*d^2 + pairwise 2*C*dl*dr must reconstruct C*(dl+dr)^2 exactly.
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 10}
    mem_caps = {"M": 10}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    opt._B, opt._D = 0.0, 0.0
    opt._C, opt._E = 1.5, 0.0

    local = opt._local_penalty("r1", "b0") + opt._local_penalty("r2", "b0")
    pair = opt._pairwise_penalty("r1", "b0", "r2", "b0")
    direct = opt._C * (2 + 3) ** 2
    assert abs(local + pair - direct) < 1e-9


def test_branch_expander_respects_memory_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 100, ("R", "B"): 100}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(edge_penalty=10.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_branch_expander_memory_congestion_flips_ordering():
    bundles = _memory_congested_bundles()
    edge_caps = {("E", "R"): 100}
    mem_caps = {"M": 4}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(memory_congestion_penalty=10.0, congestion_penalty=0.0)
    bids = {rid: bid for rid, bid in result["selected"]}
    assert not (bids.get("r1") == "b_hi" and bids.get("r2") == "b_hi")


def test_branch_expander_more_orderings_never_worse():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    r1 = opt.solve(n_orderings=1)
    r3 = opt.solve(n_orderings=3)
    assert r3["energy"] <= r1["energy"] + 1e-12


def _qval(qubo, a, b):
    return qubo.get((a, b), qubo.get((b, a), 0.0))


def test_qubo_congestion_terms_in_bqm():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 1}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b1", "r2", 20.0, {("A", "R"): 1}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 4}
    mem_caps = {"M": 100}

    opt = QUBOOptimizer(bundles, edge_caps, mem_caps)
    qubo_c10, _ = opt.to_qubo(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                              congestion_penalty=10.0, memory_congestion_penalty=0.0)
    qubo_c0, _ = opt.to_qubo(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                             congestion_penalty=0.0, memory_congestion_penalty=0.0)
    qubo_e10, _ = opt.to_qubo(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                              congestion_penalty=0.0, memory_congestion_penalty=10.0)

    by_key = {k: v for v, k in opt.variable_map.items()}
    va, vb = by_key[("r1", "b0")], by_key[("r2", "b0")]

    # Edge congestion: linear C*d^2 and cross-term 2*C*d1*d2.
    assert abs((_qval(qubo_c10, va, va) - _qval(qubo_c0, va, va)) - 10.0 * 2 * 2) < 1e-6
    assert abs((_qval(qubo_c10, va, vb) - _qval(qubo_c0, va, vb)) - 2 * 10.0 * 2 * 2) < 1e-6

    # Memory congestion: same structure with the memory demands.
    assert abs((_qval(qubo_e10, va, va) - _qval(qubo_c0, va, va)) - 10.0 * 1 * 1) < 1e-6
    assert abs((_qval(qubo_e10, va, vb) - _qval(qubo_c0, va, vb)) - 2 * 10.0 * 1 * 1) < 1e-6


def test_branch_expander_prune_infeasible_avoids_overload():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b1", "r1", 10.0, {("A", "R"): 1}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b1", "r2", 5.0, {("A", "R"): 1}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    bids = {rid: bid for rid, bid in result["selected"]}
    assert not (bids.get("r1") == "b0" and bids.get("r2") == "b0")


def test_branch_expander_prune_disabled_runs():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(prune_infeasible=False)
    assert "selected" in result


def _is_feasible(selected, bundles, edge_caps, mem_caps):
    edge_load = {}
    mem_load = {}
    for rid, bid in selected:
        for b in bundles:
            if b["request_id"] == rid and b["bundle_id"] == bid:
                for e, d in b["edge_demands"].items():
                    edge_load[tuple(sorted(e))] = edge_load.get(tuple(sorted(e)), 0) + d
                for n, d in b["memory_demands"].items():
                    mem_load[n] = mem_load.get(n, 0) + d
                break
    for e, load in edge_load.items():
        if load > edge_caps.get(e, 0):
            return False
    for n, load in mem_load.items():
        if load > mem_caps.get(n, 0):
            return False
    return True


def test_qubo_decode_repair_enforces_capacity():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = QUBOOptimizer(bundles, edge_caps, mem_caps)
    infeasible = [("r1", "b0"), ("r2", "b0")]
    repaired = opt.repair_selection(infeasible)
    assert _is_feasible(repaired, bundles, edge_caps, mem_caps)
    repaired_map = {rid: bid for rid, bid in repaired}
    assert len(repaired_map) == 1


def test_qubo_decode_repair_keeps_best_feasible_deterministically():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 1}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = QUBOOptimizer(bundles, edge_caps, mem_caps)
    raw = [("r1", "b0"), ("r2", "b0")]
    r1 = opt.repair_selection(raw)
    r2 = opt.repair_selection(list(reversed(raw)))
    assert r1 == r2
    repaired_map = {rid: bid for rid, bid in r1}
    assert repaired_map["r1"] == "b0"
    assert "r2" not in repaired_map


def test_qubo_decode_repair_passthrough_parameter():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = QUBOOptimizer(bundles, edge_caps, mem_caps)
    sample = {k: 1 for k in opt.variable_map}
    assert opt.decode_sample(sample, repair=False) == [("r1", "b0")]
    assert opt.decode_sample(sample, repair=True) == [("r1", "b0")]


def test_metropolis_feasibility_repair():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    infeasible = {"r1": "b0", "r2": "b0"}
    repaired = opt._feasibility_repair(infeasible)
    assert _is_feasible([(r, b) for r, b in repaired.items() if b is not None],
                        bundles, edge_caps, mem_caps)
    assert len([b for b in repaired.values() if b is not None]) == 1


def test_metropolis_solve_returns_feasible():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r3", 40.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=500)
    assert _is_feasible(result["selected"], bundles, edge_caps, mem_caps)


def test_qubo_solution_energy_matches_hamiltonian():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = QUBOOptimizer(bundles, edge_caps, mem_caps)
    en = opt.solution_energy([("r1", "b0")], congestion_penalty=0.05,
                             memory_congestion_penalty=0.05)
    assert abs(en - (-60.0 + 0.05 * 9 + 0.05 * 1)) < 1e-9
    assert opt.solution_energy([("r1", "b0")]) == -60.0
    en_both = opt.solution_energy([("r1", "b0"), ("r2", "b0")])
    assert en_both > en


def test_qubo_sqa_decode_repair_feasible():
    from optimization.openjij_solver import solve_sqa

    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = QUBOOptimizer(bundles, edge_caps, mem_caps)
    bqm = opt.to_bqm(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0)
    response = solve_sqa(bqm, num_reads=10, seed=1)
    selected = opt.decode_sample(response.first.sample, repair=True)
    assert _is_feasible(selected, bundles, edge_caps, mem_caps)


def test_greedy_baselines_respect_combined_capacity():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    for baseline in (utility_density_greedy, fidelity_aware_greedy):
        result = baseline(bundles, edge_caps, mem_caps)
        assert _is_feasible(result["selected"], bundles, edge_caps, mem_caps)
        assert len(result["selected"]) == 1


def test_pipeline_produces_feasible_selection():
    bundles = [
        _make_bundle("b0", "r1", 60.0, {("A", "R"): 3}, {"M": 1}),
        _make_bundle("b0", "r2", 50.0, {("A", "R"): 3}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    pipe = TensorAnnealerPipeline(bundles, edge_caps, mem_caps, seed=42)
    result = pipe.solve(tn_bond_dim=4, tn_beta=3.0, tn_sweeps=5, anneal_max_iterations=500)
    assert _is_feasible(result["selected"], bundles, edge_caps, mem_caps)
    assert len(result["selected"]) == 1
