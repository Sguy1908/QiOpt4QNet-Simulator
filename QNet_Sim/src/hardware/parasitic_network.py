"""Network-level impact of VOA parasitic emission.

Couples :mod:`hardware.parasitic_emission` (a per-link device model) to the
routing/bundle pipeline so that the solvers in ``baselines/`` and
``optimization/`` can be evaluated on networks whose links are driven by
transmitters that carry a leaky VOA.

Modelling assumptions (all explicit -- the paper studies a single point-to-point
QKD link, not a network):

* Every link is a fibre span of ``QuantumLink.distance`` km terminated by a
  detector of the kind in Table 1 of the paper.
* **Fidelity coupling.**  Unmodulated EL photons add clicks with error
  probability 1/2 (dual-source flaw).  The extra QBER ``dE`` (Eq. 24 minus its
  ``mu_EL = 0`` value) maps to Werner fidelity through ``QBER = 2(1-F)/3``, so
  ``F_eff = F_raw - 1.5 dE``.  The ``mu_EL = 0`` value is subtracted so that a
  link without parasitic light is left exactly unchanged.
* **Security coupling.**  A link is *insecure* when its passive-THA secure key
  rate (Eq. 16) is zero at the link length, i.e. an eavesdropper reading the
  VOA leakage could learn the key.  A *security-aware* allocator refuses
  bundles that traverse an insecure link; a *security-blind* one does not.
* An optical band-pass filter after the VOA (the paper's proposed mitigation)
  is modelled as a constant attenuation ``filter_db`` of the EL photons, which
  scales both ``mu_EL`` and ``mu_Eve`` by ``10**(-filter_db/10)``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from hardware.parasitic_emission import (
    DEFAULT_PARAMS, QKDParams, dual_source_gain_qber, key_rate_tha,
    mean_photon_number,
)

Edge = Tuple[str, str]


def filtered_mu(voltage: float, pulse_width_s: float, filter_db: float = 0.0) -> float:
    """Mean EL photon number per pulse after an optional ``filter_db`` filter."""
    mu = mean_photon_number(voltage, pulse_width_s)
    return mu * 10.0 ** (-filter_db / 10.0)


def parasitic_link_fidelity(raw_fidelity: float, length_km: float, mu_el: float,
                            params: QKDParams = DEFAULT_PARAMS) -> float:
    """Link fidelity after parasitic noise (see module docstring).

    Never below 0.25 (the fully mixed two-qubit state) and never above
    ``raw_fidelity``; equals ``raw_fidelity`` exactly when ``mu_el == 0``.
    """
    if mu_el <= 0.0:
        return raw_fidelity
    e_d = max(2.0 * (1.0 - raw_fidelity) / 3.0, 0.0)
    eta = params.eta_signal(length_km)
    eta_p = params.eta_parasitic(length_km)
    _, e_clean = dual_source_gain_qber(params.signal, 0.0, eta, eta_p, params.y0, e_d)
    _, e_noisy = dual_source_gain_qber(params.signal, mu_el, eta, eta_p, params.y0, e_d)
    f = raw_fidelity - 1.5 * max(e_noisy - e_clean, 0.0)
    return min(max(f, 0.25), raw_fidelity)


def link_is_secure(length_km: float, mu_eve: float,
                   params: QKDParams = DEFAULT_PARAMS) -> bool:
    """True if the passive-THA secure key rate is positive at ``length_km``."""
    if mu_eve <= 0.0:
        return True
    return key_rate_tha(length_km, mu_eve, params) > 0.0


def _links(topology: dict):
    """Yield ``(edge, QuantumLink)`` for the network embedded in a topology."""
    net = topology["network"]
    for u, v, data in net.graph.edges(data=True):
        yield tuple(sorted((u, v))), data["data"]


def apply_parasitic_emission(topology: dict, voltage: float, pulse_width_s: float,
                             filter_db: float = 0.0,
                             params: QKDParams = DEFAULT_PARAMS) -> Dict[str, object]:
    """Degrade, in place, every link of ``topology`` for a VOA operating point.

    Returns ``{"mu": mu_el, "insecure_edges": set(edges), "mean_delta_f": x}``.
    ``topology`` must carry a ``"network"`` (as built by
    ``experiments.instances.generate_*_topology``); pass a *fresh* topology
    because the links are mutated.
    """
    mu = filtered_mu(voltage, pulse_width_s, filter_db)
    insecure: Set[Edge] = set()
    deltas: List[float] = []
    for edge, link in _links(topology):
        before = link.raw_fidelity
        link.raw_fidelity = parasitic_link_fidelity(before, link.distance, mu, params)
        deltas.append(before - link.raw_fidelity)
        lp = topology.get("link_params", {}).get(edge)
        if lp is not None:
            lp["raw_fidelity"] = link.raw_fidelity
        if not link_is_secure(link.distance, mu, params):
            insecure.add(edge)
    return {"mu": mu, "insecure_edges": insecure,
            "mean_delta_f": sum(deltas) / len(deltas) if deltas else 0.0}


def bundle_edges(bundle: dict) -> Set[Edge]:
    """Undirected edge set used by a bundle."""
    return {tuple(sorted(e)) for e in bundle["edge_demands"]}


def secure_bundles(bundles: Iterable[dict], insecure_edges: Set[Edge]) -> List[dict]:
    """Bundles that avoid every insecure edge (security-aware admission)."""
    return [b for b in bundles if not (bundle_edges(b) & insecure_edges)]


def count_insecure_selected(selected, bundles: List[dict], insecure_edges: Set[Edge]) -> int:
    """How many selected ``(request_id, bundle_id)`` pairs traverse an insecure edge."""
    by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    return sum(1 for k in selected if bundle_edges(by_key[k]) & insecure_edges)


def assign_link_lengths(topology: dict, length_range: Tuple[float, float],
                        seed: int) -> None:
    """Give every link a deterministic pseudo-random length (km) in
    ``length_range`` (sorted-edge order, so the draw is topology-stable)."""
    import random
    rng = random.Random(seed)
    lo, hi = length_range
    for edge, link in sorted(_links(topology), key=lambda t: t[0]):
        link.distance = round(rng.uniform(lo, hi), 2)
        lp = topology.get("link_params", {}).get(edge)
        if lp is not None:
            lp["distance"] = link.distance


def _instance(topology_fn: Callable[[], dict], n_requests: int, seed: int,
              voltage: float, pulse_width_s: float, filter_db: float,
              params: QKDParams, length_range: Optional[Tuple[float, float]] = None):
    """Build one degraded instance; returns ``(bundles, edge_caps, mem_caps, info)``."""
    from experiments.instances import contention_sweep_instances

    holder: Dict[str, object] = {}

    def degraded_topology() -> dict:
        topo = topology_fn()
        if length_range is not None:
            assign_link_lengths(topo, length_range, seed)
        holder["info"] = apply_parasitic_emission(topo, voltage, pulse_width_s,
                                                  filter_db, params)
        return topo

    inst = contention_sweep_instances(degraded_topology, [n_requests], seed=seed)
    d = inst[f"req{n_requests}"]
    return d["bundles"], d["edge_capacities"], d["memory_capacities"], holder["info"]


def run_parasitic_network_study(topology_fn: Callable[[], dict],
                                voltage: float, pulse_width_s: float,
                                n_requests: int = 8, seed: int = 0,
                                filter_db: float = 0.0,
                                allocators: Optional[List[str]] = None,
                                params: QKDParams = DEFAULT_PARAMS,
                                length_range: Optional[Tuple[float, float]] = None
                                ) -> List[dict]:
    """Run allocators on a clean and a parasitic-degraded copy of one instance.

    For each allocator three rows are produced -- ``clean`` (no EL),
    ``blind`` (degraded links, security-blind) and ``aware`` (degraded links,
    bundles over insecure links removed) -- with acceptance, total utility,
    mean delivered fidelity of admitted bundles and the number of admitted
    bundles that traverse an insecure link.  Solvers are all fed the same
    bundle list, so differences are due to the device effect alone.  If
    ``length_range`` is given, link lengths are redrawn per link (same draw in
    every scenario) instead of using the topology's uniform ``distance``.
    """
    from baselines.classical_baselines import ALL_BASELINES
    from baselines.feasibility import compute_metrics

    names = allocators or ["utility_per_resource_greedy", "congestion_aware_greedy",
                           "greedy_local_search"]
    clean_b, ec, mc, _ = _instance(topology_fn, n_requests, seed, 0.0, 0.0, 0.0, params, length_range)
    deg_b, _, _, info = _instance(topology_fn, n_requests, seed, voltage,
                                  pulse_width_s, filter_db, params, length_range)
    insecure = info["insecure_edges"]
    aware_b = secure_bundles(deg_b, insecure)
    all_ids = sorted({b["request_id"] for b in clean_b} | {b["request_id"] for b in deg_b})

    rows: List[dict] = []
    for scenario, bundles in (("clean", clean_b), ("blind", deg_b), ("aware", aware_b)):
        for name in names:
            res = ALL_BASELINES[name](bundles, ec, mc, seed=seed).solve()
            m = compute_metrics(res, bundles, ec, mc, all_request_ids=all_ids)
            rows.append({
                "scenario": scenario, "allocator": name,
                "mu_el": info["mu"], "filter_db": filter_db,
                "n_bundles": len(bundles),
                "accepted": m["accepted_requests"], "n_requests": m["total_requests"],
                "acceptance_rate": m["acceptance_rate"],
                "total_utility": m["total_utility"],
                "avg_fidelity": m["avg_fidelity"],
                "insecure_admitted": count_insecure_selected(res["selected"], bundles,
                                                             insecure),
                "n_insecure_edges": len(insecure),
                "mean_delta_f": info["mean_delta_f"],
                "feasible": m["feasible"],
            })
    return rows
