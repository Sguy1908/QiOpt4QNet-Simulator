"""Tests for the function-approximation RL router: function-approximation RL that generalizes across topologies)."""

import pytest

from experiments.instances import generate_chain_topology, generate_grid_topology
from optimization.time_dependent_optimizer import _poisson_trace
from baselines.qlearning_router import LinearQRouter, run_topology_generalization_study

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _chain_topo():
    return generate_chain_topology(n_nodes=8, edge_capacity=6, memory_capacity=10,
                                   raw_fidelity=0.85, generation_prob=0.8)


def _grid_topo():
    return generate_grid_topology(rows=4, cols=4, edge_capacity=5, memory_capacity=10,
                                  raw_fidelity=0.85, generation_prob=0.8)


def test_linear_router_features_fixed_length_across_topologies():
    topo_a, topo_b = _chain_topo(), _grid_topo()
    r_a = LinearQRouter(topo_a["edge_capacities"], topo_a["memory_capacities"], seed=1)
    r_b = LinearQRouter(topo_b["edge_capacities"], topo_b["memory_capacities"], seed=1)
    f_a = r_a._features("r1", None, {})
    f_b = r_b._features("r1", None, {})
    assert len(f_a) == len(f_b) == LinearQRouter.N_FEATURES


def test_linear_router_choose_action_prefers_feasible():
    topo = _chain_topo()
    r = LinearQRouter(topo["edge_capacities"], topo["memory_capacities"], seed=1, epsilon=0.0)
    trace = _poisson_trace(lambda: topo, 5, 1.0, seed=1)
    slot0 = trace[0]
    for rid, bundles in slot0.items():
        action = r.choose_action(rid, bundles, {})
        if action is not None:
            assert action in {b["bundle_id"] for b in bundles}
        break


def test_linear_router_reject_when_infeasible():
    topo = _chain_topo()
    r = LinearQRouter({}, {}, seed=1, epsilon=0.0)
    bundles = [{"bundle_id": "b0", "request_id": "r1", "utility": 5.0,
               "edge_demands": {("N0", "N1"): 100}, "memory_demands": {}}]
    action = r.choose_action("r1", bundles, {})
    assert action is None


def test_linear_router_evaluate_is_deterministic_and_matches_training_count():
    topo = _chain_topo()
    r = LinearQRouter(topo["edge_capacities"], topo["memory_capacities"], seed=2)
    trace = _poisson_trace(lambda: topo, 8, 1.0, seed=2)
    for _ in range(5):
        r.train_episode(trace)
    served1, util1, eps1 = r.evaluate(trace)
    served2, util2, eps2 = r.evaluate(trace)
    assert served1 == served2
    assert util1 == pytest.approx(util2)
    assert eps1 == eps2 == 5


def test_linear_router_weights_transfer_across_bundle_id_namespaces():
    """Centerpiece test: train on one topology's ids, zero-shot evaluate
    (epsilon=0) on a structurally different topology with completely
    disjoint node/edge/bundle ids; the router should still serve a
    meaningful fraction of requests using only topology-invariant
    features -- the actual claim behind "generalizes across topologies"."""
    train_topo_fn = _chain_topo
    r = LinearQRouter(train_topo_fn()["edge_capacities"],
                      train_topo_fn()["memory_capacities"], seed=7, epsilon=0.2)
    import random
    rng = random.Random(7)
    for _ in range(25):
        trace = _poisson_trace(train_topo_fn, 10, 1.2, seed=rng.randint(0, 2 ** 31 - 1))
        topo = train_topo_fn()
        r.edge_capacities = {tuple(sorted(k)): v for k, v in topo["edge_capacities"].items()}
        r.memory_capacities = topo["memory_capacities"]
        r.train_episode(trace)

    eval_topo = _grid_topo()
    r.edge_capacities = {tuple(sorted(k)): v for k, v in eval_topo["edge_capacities"].items()}
    r.memory_capacities = eval_topo["memory_capacities"]
    test_trace = _poisson_trace(lambda: eval_topo, 10, 1.2, seed=rng.randint(0, 2 ** 31 - 1))
    served, utility, _ = r.evaluate(test_trace)
    n_requests = sum(len(slot) for slot in test_trace)
    assert served > 0, "zero-shot router should serve at least some requests on an unseen topology"
    assert served / max(n_requests, 1) > 0.1


def test_run_topology_generalization_study_schema():
    res = run_topology_generalization_study(
        train_topology_fns=[_chain_topo],
        eval_topology_fns={"grid_4x4": _grid_topo},
        n_slots=6, mean_rate=1.0, episodes=6, seed=11)
    rows = res["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["eval_topology"] == "grid_4x4"
    for key in ["linear_zero_shot_served_ratio", "tabular_in_distribution_served_ratio",
               "streaming_annealer_served_ratio"]:
        assert 0.0 <= row[key] <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_mlp_router_runs_when_torch_available():
    topo = _chain_topo()
    r = LinearQRouter(topo["edge_capacities"], topo["memory_capacities"], seed=1,
                      use_torch_mlp=True)
    trace = _poisson_trace(lambda: topo, 5, 1.0, seed=1)
    served, reward = r.train_episode(trace)
    ev_served, ev_util, eps = r.evaluate(trace)
    assert eps == 1


def test_torch_mlp_router_requires_torch_extra_when_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    topo = _chain_topo()
    with pytest.raises(ImportError):
        LinearQRouter(topo["edge_capacities"], topo["memory_capacities"], use_torch_mlp=True)
