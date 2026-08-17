"""Tests for the quantum-annealing backend (Wave 3, item 6)."""

import math

import dimod
import pytest

from optimization.quantum_annealing_backend import (
    build_hardware_graph, hardware_qubits, default_chain_strength,
    minor_embed, embed_bqm, unembed_by_majority, quantum_anneal_solve,
    chain_strength_sweep, run_quantum_annealing_sweep,
    build_hardware_graph_pegasus, build_hardware_graph_zephyr,
    hardware_qubits_pegasus, hardware_qubits_zephyr,
    _fit_hardware_pegasus, _fit_hardware_zephyr,
    run_quantum_annealing_topology_sweep,
)
from experiments.instances import (
    generate_chain_topology, contention_sweep_instances,
)


def _topology(n_nodes=6, cap=6, mem=8, raw=0.85):
    return generate_chain_topology(n_nodes=n_nodes, edge_capacity=cap,
                                   memory_capacity=mem, raw_fidelity=raw,
                                   generation_prob=0.8)


def _instance(n_req=3, n_nodes=6, seed=42):
    topo = _topology(n_nodes)
    return topo, contention_sweep_instances(lambda: topo, [n_req], seed=seed)[
        f"req{n_req}"]


def test_hardware_graph_structure():
    hw = build_hardware_graph(rows=4, cols=4, t=4)
    assert len(hw) == hardware_qubits(4, 4, t=4) == 128
    # bipartite unit cells: no intra-cell same-side edges; degree bounded
    for u, nbrs in hw.items():
        assert 1 <= len(nbrs) <= 2 * 4 + 2
        for v in nbrs:
            assert u in hw[v]  # undirected


def test_embed_preserves_coherent_energies():
    bqm = dimod.BQM({0: -1.0, 1: -2.0, 2: -1.5},
                    {(0, 1): -0.5, (1, 2): -0.25}, 0.0, dimod.BINARY)
    la = {v: set() for v in bqm.variables}
    for u, v in bqm.quadratic:
        la.setdefault(u, set()).add(v)
        la.setdefault(v, set()).add(u)
    hw = build_hardware_graph(3, 3, t=4)
    emb = minor_embed(la, hw)
    assert emb
    cs = default_chain_strength(bqm)
    assert cs >= max(abs(v) for v in bqm.linear.values())
    eb = embed_bqm(bqm, emb, cs)
    for value in (0, 1):
        raw = {q: value for q in eb.variables}
        lg = unembed_by_majority(raw, emb, bqm.num_variables, __import__("random").Random(0))
        assert math.isclose(eb.energy(raw), bqm.energy(lg), rel_tol=1e-9)


def test_unembed_by_majority():
    emb = {"a": [0, 1, 2], "b": [3, 4], "c": [5]}
    import random
    rng = random.Random(0)
    s = unembed_by_majority({0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1},
                            emb, 3, rng)
    assert s == {"a": 1, "b": 0, "c": 1}  # 2/3 and tie->0


def test_chain_strength_default_rule():
    bqm = dimod.BQM({0: -3.0}, {(0, 1): -2.0}, 0.0, dimod.BINARY)
    assert default_chain_strength(bqm) >= 3.0 + 2 * 2.0


def test_quantum_anneal_solve_runs():
    topo, inst = _instance(n_req=3)
    res = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                               inst["memory_capacities"], k=8, num_reads=10,
                               num_sweeps=100, t=8, seed=42)
    assert res["embedded"] is True
    assert res["utility"] >= 0.0
    assert 0.0 <= res["chain_break_fraction"] <= 1.0
    assert res["n_qubits"] > res["n_qubo_variables"]
    assert all((rid, bid) in {(b["request_id"], b["bundle_id"])
                              for b in inst["bundles"]}
               for rid, bid in res["selected"])


def test_quantum_anneal_solve_deterministic():
    topo, inst = _instance(n_req=3)
    a = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                             inst["memory_capacities"], k=8, num_reads=8,
                             num_sweeps=80, t=8, seed=1)
    b = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                             inst["memory_capacities"], k=8, num_reads=8,
                             num_sweeps=80, t=8, seed=1)
    assert a["utility"] == b["utility"]
    assert a["selected"] == b["selected"]


def test_chain_strength_sweep_schema():
    topo, inst = _instance(n_req=3)
    from optimization.adaptive_qubo import AdaptiveCandidateSelector
    from optimization.qubo_optimizer import QUBOOptimizer
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    reduced = AdaptiveCandidateSelector(ec, mc, seed=42).select(b, k=8)
    bqm = QUBOOptimizer(reduced, ec, mc).to_bqm(
        penalty=100.0, edge_penalty=10.0, memory_penalty=10.0)
    la = {v: set() for v in bqm.variables}
    for u, v in bqm.quadratic:
        la.setdefault(u, set()).add(v)
        la.setdefault(v, set()).add(u)
    hw = build_hardware_graph(6, 6, t=8)
    emb = minor_embed(la, hw)
    assert emb
    cs = chain_strength_sweep(bqm, emb, hw, factors=[0.5, 1.0, 2.0],
                              num_reads=6, num_sweeps=60, seed=42)
    assert len(cs["rows"]) == 3
    for r in cs["rows"]:
        assert 0.0 <= r["chain_break_fraction"] <= 1.0
        assert r["chain_strength"] > 0.0
    assert cs["default_chain_strength"] == cs["rows"][1]["chain_strength"]


def test_quantum_annealing_sweep_schema():
    topo_fn = lambda: _topology(n_nodes=6)
    res = run_quantum_annealing_sweep(topo_fn, n_requests_list=[4],
                                      num_reads=8, num_sweeps=80, k=8, seed=42)
    assert len(res["rows"]) == 1
    r = res["rows"][0]
    assert r["qa_utility"] >= 0.0
    assert r["sa_utility"] >= 0.0
    assert r["sqa_utility"] >= 0.0
    assert r["qa_n_qubits"] > r["n_qubo_variables"]


# ---------------------------------------------------------------------------
# Pegasus / Zephyr hardware topologies
# ---------------------------------------------------------------------------
def test_pegasus_graph_structure():
    hw = build_hardware_graph_pegasus(3)
    assert len(hw) == hardware_qubits_pegasus(3) == 128
    for u, nbrs in hw.items():
        assert len(nbrs) > 0
        for v in nbrs:
            assert u in hw[v]  # undirected


def test_zephyr_graph_structure():
    hw = build_hardware_graph_zephyr(2)
    assert len(hw) == hardware_qubits_zephyr(2) == 160
    for u, nbrs in hw.items():
        assert len(nbrs) > 0
        for v in nbrs:
            assert u in hw[v]  # undirected


def test_fit_hardware_pegasus_monotonic_and_sufficient():
    m = _fit_hardware_pegasus(100)
    assert hardware_qubits_pegasus(m) >= 100
    assert hardware_qubits_pegasus(m - 1) < 100 if m > 2 else True
    # larger requirement never yields a smaller lattice
    assert _fit_hardware_pegasus(500) >= _fit_hardware_pegasus(100)


def test_fit_hardware_zephyr_monotonic_and_sufficient():
    m = _fit_hardware_zephyr(100)
    assert hardware_qubits_zephyr(m) >= 100
    assert hardware_qubits_zephyr(m - 1) < 100 if m > 1 else True
    assert _fit_hardware_zephyr(500) >= _fit_hardware_zephyr(100)


def test_quantum_anneal_solve_runs_on_pegasus():
    topo, inst = _instance(n_req=3)
    res = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                               inst["memory_capacities"], k=8, num_reads=10,
                               num_sweeps=100, topology="pegasus", seed=42)
    assert res["embedded"] is True
    assert res["topology"] == "pegasus"
    assert res["lattice_size"] >= 2
    assert res["utility"] >= 0.0
    assert 0.0 <= res["chain_break_fraction"] <= 1.0
    assert res["n_qubits"] > res["n_qubo_variables"]


def test_quantum_anneal_solve_runs_on_zephyr():
    topo, inst = _instance(n_req=3)
    res = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                               inst["memory_capacities"], k=8, num_reads=10,
                               num_sweeps=100, topology="zephyr", seed=42)
    assert res["embedded"] is True
    assert res["topology"] == "zephyr"
    assert res["lattice_size"] >= 1
    assert res["utility"] >= 0.0
    assert 0.0 <= res["chain_break_fraction"] <= 1.0
    assert res["n_qubits"] > res["n_qubo_variables"]


def test_quantum_anneal_solve_unknown_topology_raises():
    topo, inst = _instance(n_req=3)
    with pytest.raises(ValueError):
        quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                             inst["memory_capacities"], k=8,
                             topology="not_a_real_topology")


def test_chimera_default_backward_compatible():
    topo, inst = _instance(n_req=3)
    a = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                             inst["memory_capacities"], k=8, num_reads=8,
                             num_sweeps=80, t=8, seed=1)
    b = quantum_anneal_solve(inst["bundles"], inst["edge_capacities"],
                             inst["memory_capacities"], k=8, num_reads=8,
                             num_sweeps=80, t=8, topology="chimera", seed=1)
    assert a["utility"] == b["utility"]
    assert a["selected"] == b["selected"]
    assert a["n_qubits"] == b["n_qubits"]
    assert "lattice_size" not in a and "lattice_size" not in b


def test_quantum_annealing_topology_sweep_schema():
    topo_fn = lambda: _topology(n_nodes=6)
    res = run_quantum_annealing_topology_sweep(
        topo_fn, n_requests_list=[4], topologies=("chimera", "pegasus", "zephyr"),
        num_reads=8, num_sweeps=80, k=8, seed=42)
    assert len(res["rows"]) == 3
    seen_topologies = {r["topology"] for r in res["rows"]}
    assert seen_topologies == {"chimera", "pegasus", "zephyr"}
    for r in res["rows"]:
        assert r["utility"] >= 0.0
        assert r["n_qubits"] > r["n_qubo_variables"]
        assert 0.0 <= r["chain_break_fraction"] <= 1.0
        if r["topology"] != "chimera":
            assert r["lattice_size"] is not None
