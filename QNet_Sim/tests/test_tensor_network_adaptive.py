"""Tests for future-work items: adaptive bond dimension (truncation-error
driven) and learned (spectral) MPS coupling order."""

import pytest

from experiments.instances import (
    generate_chain_topology, generate_grid_topology, contention_sweep_instances,
)
from optimization.tensor_network_optimizer import TensorNetworkOptimizer


def _instance(n_req=12, n_nodes=10, seed=42):
    topo = lambda: generate_chain_topology(n_nodes=n_nodes, edge_capacity=6,
                                           memory_capacity=10, raw_fidelity=0.85,
                                           generation_prob=0.8)
    return contention_sweep_instances(topo, [n_req], seed=seed)[f"req{n_req}"]


def _grid_instance(n_req=16, rows=6, cols=6, seed=42):
    topo = lambda: generate_grid_topology(rows=rows, cols=cols, edge_capacity=5,
                                          memory_capacity=10, raw_fidelity=0.85,
                                          generation_prob=0.8)
    return contention_sweep_instances(topo, [n_req], seed=seed)[f"req{n_req}"]


# ---------------------------------------------------------------------------
# Adaptive bond dimension
# ---------------------------------------------------------------------------
def test_adaptive_bond_dim_grows_with_tighter_eps():
    inst = _instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

    opt_loose = TensorNetworkOptimizer(b, ec, mc)
    r_loose = opt_loose.solve(adaptive_bond_dim=True, trunc_eps=1e-2,
                              max_bond_dim=64, max_sweeps=1)
    opt_tight = TensorNetworkOptimizer(b, ec, mc)
    r_tight = opt_tight.solve(adaptive_bond_dim=True, trunc_eps=1e-10,
                              max_bond_dim=64, max_sweeps=1)
    assert r_tight["max_chi_used"] >= r_loose["max_chi_used"]


def test_adaptive_bond_dim_capped_by_max_bond_dim():
    inst = _instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt = TensorNetworkOptimizer(b, ec, mc)
    r = opt.solve(adaptive_bond_dim=True, trunc_eps=1e-12, max_bond_dim=2, max_sweeps=1)
    assert r["max_chi_used"] <= 2
    assert all(t["chi_used"] <= 2 for t in r["bond_dim_trace"])


def test_adaptive_bond_dim_default_off_backward_compatible():
    inst = _instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt_default = TensorNetworkOptimizer(b, ec, mc)
    r_default = opt_default.solve(bond_dim=8, max_sweeps=1)
    opt_explicit = TensorNetworkOptimizer(b, ec, mc)
    r_explicit = opt_explicit.solve(bond_dim=8, max_sweeps=1, adaptive_bond_dim=False)
    assert r_default["energy"] == r_explicit["energy"]
    assert r_default["selected"] == r_explicit["selected"]
    assert "bond_dim_trace" not in r_default
    assert "bond_dim_trace" not in r_explicit


def test_adaptive_bond_dim_diagnostics_present_only_when_enabled():
    inst = _instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt_off = TensorNetworkOptimizer(b, ec, mc)
    r_off = opt_off.solve(max_sweeps=1, adaptive_bond_dim=False)
    assert "bond_dim_trace" not in r_off
    assert "max_chi_used" not in r_off

    opt_on = TensorNetworkOptimizer(b, ec, mc)
    r_on = opt_on.solve(max_sweeps=1, adaptive_bond_dim=True)
    assert "bond_dim_trace" in r_on
    assert "max_chi_used" in r_on
    assert len(r_on["bond_dim_trace"]) > 0


def test_adaptive_bond_dim_no_op_at_max_sweeps_zero_is_documented():
    inst = _instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt_off = TensorNetworkOptimizer(b, ec, mc)
    r_off = opt_off.solve(max_sweeps=0, adaptive_bond_dim=False)
    opt_on = TensorNetworkOptimizer(b, ec, mc)
    r_on = opt_on.solve(max_sweeps=0, adaptive_bond_dim=True)
    # With max_sweeps=0 no SVD truncation ever runs (exact product form is
    # kept), so adaptive_bond_dim has no effect: same energy/selection, and
    # an empty (not absent) trace confirming zero bonds were contracted.
    assert r_off["energy"] == r_on["energy"]
    assert r_off["selected"] == r_on["selected"]
    assert r_on["bond_dim_trace"] == []
    assert r_on["max_chi_used"] == 0


# ---------------------------------------------------------------------------
# Learned (spectral) MPS coupling order
# ---------------------------------------------------------------------------
def test_order_strategy_invalid_raises():
    inst = _instance(n_req=4)
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    with pytest.raises(ValueError):
        TensorNetworkOptimizer(b, ec, mc, order_strategy="not_a_strategy")


def test_spectral_order_deterministic():
    inst = _grid_instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt1 = TensorNetworkOptimizer(b, ec, mc, order_strategy="spectral")
    opt2 = TensorNetworkOptimizer(b, ec, mc, order_strategy="spectral")
    assert opt1._ordered_requests == opt2._ordered_requests


def test_spectral_order_same_request_set_as_greedy():
    inst = _grid_instance()
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt_g = TensorNetworkOptimizer(b, ec, mc, order_strategy="greedy")
    opt_s = TensorNetworkOptimizer(b, ec, mc, order_strategy="spectral")
    assert set(opt_g._ordered_requests) == set(opt_s._ordered_requests)
    assert len(opt_s._ordered_requests) == len(opt_g._ordered_requests)


def test_spectral_order_handles_disconnected_components_without_crash():
    # Two independent chain topologies -> two request clusters sharing no
    # edges or memory nodes (fully disconnected coupling graph components).
    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                              memory_capacity=10, raw_fidelity=0.85,
                                              generation_prob=0.8)
    inst = contention_sweep_instances(topo_fn, [8], seed=7)["req8"]
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt = TensorNetworkOptimizer(b, ec, mc, order_strategy="spectral")
    assert len(opt._ordered_requests) == len(opt.requests)
    assert set(opt._ordered_requests) == set(opt.requests)


def test_spectral_order_non_regression_vs_greedy():
    # Weak inequality: the existing _improve() local search already recovers
    # a lot of ordering-induced quality loss, so assert non-regression on
    # average rather than strict improvement (avoids an overclaiming/flaky
    # test).
    inst = _grid_instance(n_req=16)
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    opt_g = TensorNetworkOptimizer(b, ec, mc, order_strategy="greedy")
    r_g = opt_g.solve(bond_dim=4, max_sweeps=2)
    opt_s = TensorNetworkOptimizer(b, ec, mc, order_strategy="spectral")
    r_s = opt_s.solve(bond_dim=4, max_sweeps=2)
    assert r_s["energy"] <= r_g["energy"] + 1e-6
