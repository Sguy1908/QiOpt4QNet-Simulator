"""
experiments/instances.py

Deterministic (fixed-parameter, non-randomized) topology generators and a
request-count "contention sweep" instance builder, used by
baselines/gnn_baseline.py to produce training/val/test data.

Unlike network.topology's generate_chain/generate_grid (which draw link
fidelity/probability/latency from random ranges by default), the
generators here take fixed values so that calling the same topology_fn()
twice yields an identical network -- required because gnn_baseline.py
calls the topology function once for the raw topology dict (for GNN
tensorization) and again inside contention_sweep_instances (to build
bundles), and the two need to agree.
"""

from __future__ import annotations

import random
from itertools import islice
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from network.network import QuantumNetwork
from network.node import QuantumNode
from network.link import QuantumLink
from network.request import Request

from baselines.request_generator import generate_requests
from baselines.path_strategies import CandidatePathGenerator
from routing.bundle_generation import BundleGenerator


def _add_link(net: QuantumNetwork, u: str, v: str, edge_capacity: int, raw_fidelity: float,
              generation_prob: float, latency: float, distance: float,
              edges: list, link_params: dict) -> None:
    net.add_link(QuantumLink(u, v, distance, generation_prob, raw_fidelity, latency, edge_capacity))
    e = tuple(sorted((u, v)))
    edges.append(e)
    link_params[e] = {
        "raw_fidelity": raw_fidelity,
        "generation_probability": generation_prob,
        "latency": latency,
    }


def generate_chain_topology(n_nodes: int, edge_capacity: int = 6, memory_capacity: int = 10,
                             raw_fidelity: float = 0.85, generation_prob: float = 1.0,
                             latency: float = 5.0, distance: float = 25.0) -> dict:
    net = QuantumNetwork()
    nodes = [f"N{i}" for i in range(n_nodes)]
    for nid in nodes:
        net.add_node(QuantumNode(nid, memory_capacity))

    edges: list = []
    link_params: dict = {}
    for i in range(n_nodes - 1):
        _add_link(net, nodes[i], nodes[i + 1], edge_capacity, raw_fidelity,
                   generation_prob, latency, distance, edges, link_params)

    return {
        "nodes": nodes,
        "edges": edges,
        "link_params": link_params,
        "edge_capacities": {e: edge_capacity for e in edges},
        "memory_capacities": {n: memory_capacity for n in nodes},
        "network": net,
    }


def generate_grid_topology(rows: int, cols: int, edge_capacity: int = 6, memory_capacity: int = 10,
                            raw_fidelity: float = 0.85, generation_prob: float = 1.0,
                            latency: float = 5.0, distance: float = 25.0) -> dict:
    net = QuantumNetwork()
    nodes = [f"G{i}_{j}" for i in range(rows) for j in range(cols)]
    for nid in nodes:
        net.add_node(QuantumNode(nid, memory_capacity))

    edges: list = []
    link_params: dict = {}
    for i in range(rows):
        for j in range(cols - 1):
            _add_link(net, f"G{i}_{j}", f"G{i}_{j + 1}", edge_capacity, raw_fidelity,
                       generation_prob, latency, distance, edges, link_params)
    for i in range(rows - 1):
        for j in range(cols):
            _add_link(net, f"G{i}_{j}", f"G{i + 1}_{j}", edge_capacity, raw_fidelity,
                       generation_prob, latency, distance, edges, link_params)

    return {
        "nodes": nodes,
        "edges": edges,
        "link_params": link_params,
        "edge_capacities": {e: edge_capacity for e in edges},
        "memory_capacities": {n: memory_capacity for n in nodes},
        "network": net,
    }


def instance_key(n_requests: int) -> str:
    """Key under which contention_sweep_instances() stores the workload with
    ``n_requests`` requests. Use this instead of formatting the key by hand."""
    return f"req{n_requests}"


def contention_sweep_instances(topology_fn: Callable[[], dict], request_counts: Sequence[int],
                                seed: int, k_paths: int = 3) -> Dict[str, dict]:
    """For each request count, builds a *fresh* network from topology_fn(),
    generates that many requests, generates candidate paths + bundles, and
    returns {workload_name: {"bundles": [...], "edge_capacities": {...},
    "memory_capacities": {...}}}.

    topology_fn must be deterministic (fixed link parameters, no randomness)
    so this network matches the one gnn_baseline.py separately builds for
    GNN node/edge feature tensorization.
    """
    out: Dict[str, dict] = {}
    for n_requests in request_counts:
        topo = topology_fn()
        net = topo.get("network") or _network_from_topology(topo)
        requests = generate_requests(net, load=n_requests, seed=seed)

        path_gen = CandidatePathGenerator(net)
        bundle_gen = BundleGenerator(net)

        bundles = []
        for req in requests:
            candidate_paths = path_gen.generate_candidates(req, k=k_paths)
            bundles.extend(bundle_gen.generate_bundles(req, candidate_paths))

        out[instance_key(n_requests)] = {
            "bundles": [b.to_dict() for b in bundles],
            "edge_capacities": dict(topo["edge_capacities"]),
            "memory_capacities": dict(topo["memory_capacities"]),
            "n_requests": n_requests,
        }
    return out


def _network_from_topology(topology: dict) -> QuantumNetwork:
    """Rebuild a :class:`QuantumNetwork` from a bare topology dict.

    The new-style topology generators embed a ``"network"`` key; legacy
    callers (e.g. the extension topology families in
    ``extensions.topologies``) only carry ``nodes``/``edges``/``link_params``
    plus capacity dicts.  Reconstructing the network from those lets the
    standard ``BundleGenerator`` pipeline serve both shapes.
    """
    net = QuantumNetwork()
    for nid in topology["nodes"]:
        mem = topology["memory_capacities"].get(nid, 10)
        net.add_node(QuantumNode(nid, int(mem)))
    seen = set()
    for u, v in topology["edges"]:
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        lp = topology["link_params"].get(key, topology["link_params"].get((v, u), {}))
        cap = topology["edge_capacities"].get(key, topology["edge_capacities"].get((v, u), 1))
        net.add_link(QuantumLink(
            u, v,
            lp.get("distance", 25.0),
            lp.get("generation_probability", 1.0),
            lp.get("raw_fidelity", 0.85),
            lp.get("latency", 5.0),
            int(cap),
        ))
    return net


def _shortest_paths(topology: dict, source: str, target: str, k: int = 3) -> List[List[str]]:
    """The k shortest simple paths between two nodes (legacy helper)."""
    G = nx.Graph()
    G.add_nodes_from(topology["nodes"])
    G.add_edges_from(topology["edges"])
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _pareto_prune(bundles: List[dict], fidelity_key: str = "fidelity",
                  cost_key: str = "bell_pair_cost",
                  latency_key: str = "latency") -> List[dict]:
    """Drop bundles dominated on (fidelity, cost, latency) -- legacy helper."""
    pruned = []
    for i, bc in enumerate(bundles):
        dominated = False
        for j, bo in enumerate(bundles):
            if i == j:
                continue
            no_worse = (
                bo[fidelity_key] >= bc[fidelity_key]
                and bo[cost_key] <= bc[cost_key]
                and bo[latency_key] <= bc[latency_key]
            )
            strict = (
                bo[fidelity_key] > bc[fidelity_key]
                or bo[cost_key] < bc[cost_key]
                or bo[latency_key] < bc[latency_key]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            pruned.append(bc)
    return pruned


def _evaluate_bundle(topology: dict, path: List[str], q: int, source: str, target: str,
                     weight: float, min_fidelity: float) -> Optional[dict]:
    """Score one path at purification level ``q`` (legacy helper).

    Delegates to the standard ``BundleGenerator`` pipeline and mirrors the
    legacy semantics: returns ``None`` for degenerate paths or when the
    purified end-to-end fidelity is below ``min_fidelity``.
    """
    if len(path) < 2:
        return None
    net = topology.get("network") or _network_from_topology(topology)
    req = Request(source=source, destination=target,
                  minimum_fidelity=min_fidelity, weight=weight)
    bundle = BundleGenerator(net)._evaluate_bundle(req, path, q)
    if bundle is None or bundle.fidelity < min_fidelity:
        return None
    d = bundle.to_dict()
    d["request_id"] = f"req_{source}_{target}"
    d["bundle_id"] = f"b_{source}_{target}_q{q}"
    return d


def generate_request_bundles(topology: dict, source: str, target: str,
                             weight: float, min_fidelity: float,
                             q_values: Optional[List[int]] = None) -> List[dict]:
    """All candidate bundles for one request (legacy helper).

    Uses the k shortest paths and the standard per-path evaluator, then
    Pareto-prunes the (fidelity, cost, latency) trade-off.
    """
    if q_values is None:
        q_values = [0, 1, 2]
    paths = _shortest_paths(topology, source, target, k=3)
    bundles = []
    for path in paths:
        for q in q_values:
            bundle = _evaluate_bundle(topology, path, q, source, target,
                                      weight, min_fidelity)
            if bundle:
                bundles.append(bundle)
    return _pareto_prune(bundles)


def generate_benchmark_instance(topology: dict,
                                request_pairs: List[Tuple[str, str, float, float]],
                                rng: random.Random) -> Tuple[List[dict], dict, dict]:
    """Build bundles for explicit ``(src, dst, weight, min_fidelity)`` pairs.

    Mirrors the legacy layout: per request ``generate_request_bundles`` on
    the standard pipeline, with ``req_{i}`` / ``req_{i}_b{j}`` ids.  Returns
    ``(bundles, edge_capacities, memory_capacities)``.
    """
    all_bundles: List[dict] = []
    for i, (src, dst, weight, min_fid) in enumerate(request_pairs):
        bundles = generate_request_bundles(topology, src, dst, weight, min_fid)
        for j, b in enumerate(bundles):
            b["request_id"] = f"req_{i}"
            b["bundle_id"] = f"req_{i}_b{j}"
        all_bundles.extend(bundles)
    return all_bundles, topology["edge_capacities"], topology["memory_capacities"]
 