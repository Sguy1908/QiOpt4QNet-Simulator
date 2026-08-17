"""Chance-constrained fidelity routing (research extension).

The baseline pipeline enforces the fidelity requirement *deterministically*:
a bundle whose nominal delivered fidelity ``F`` falls below the request
requirement ``F_min`` is discarded at generation time
(``instances._evaluate_bundle``).  Real delivered fidelity is stochastic,
however -- decoherence, imperfect measurement and state-estimation error
make the delivered fidelity a random variable :math:`F_r \\sim N(\\mu,\\sigma^2)`
truncated to :math:`[0,1]` around the nominal value.

This module replaces the hard rule with a *chance constraint*

.. math::

    P(F_r \\ge F_{\\min}) \\ge 1 - \\varepsilon,

and studies the resulting frontier: as :math:`\\varepsilon` is relaxed,
the optimizer may admit near-miss bundles (nominal fidelity just below the
requirement) whose violation probability is still bounded by
:math:`\\varepsilon`, gaining utility at a controlled SLA risk.

Because the QUBO pipeline can only express *deterministic* candidate
constraints, the chance constraint is enforced as a candidate-set filter
--- exactly the mechanism the baseline uses for the hard fidelity
constraint --- so every existing solver (exact ILP, full QUBO, adaptive
top-k QUBO, Metropolis) solves the chance-constrained problem unchanged.

Model notes
-----------

* ``violation_probability`` uses the analytic CDF of the truncated normal
  (:math:`\\Phi` via ``math.erf``); no SciPy required.
* At :math:`\\sigma \\to 0`` the chance constraint degenerates to the hard
  rule (:math:`F \\ge F_{\\min}`), which is used as a consistency check.
* The hard policy has an *implicit, unbounded* SLA risk: for a bundle with
  :math:`\\mu = F_{\\min}` and symmetric noise, :math:`P(F_r < F_{\\min}) = 1/2`.
  The chance policy caps this risk at :math:`\\varepsilon` under the model,
  and the DES evaluation confirms it empirically.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from experiments.instances import _evaluate_bundle, _shortest_paths


# ----------------------------------------------------------------------
# truncated-normal fidelity model
# ----------------------------------------------------------------------
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def truncnorm_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of N(mu, sigma^2) truncated to [0, 1]."""
    if sigma <= 0:
        return 1.0 if x > mu else 0.0
    a, b = -mu / sigma, (1.0 - mu) / sigma
    za, zb = _phi(a), _phi(b)
    if zb <= za:
        return 0.0
    return (_phi((x - mu) / sigma) - za) / (zb - za)


def violation_probability(nominal_f: float, min_fidelity: float,
                          sigma: float) -> float:
    """P(F_r < F_min) under the truncated-normal delivery model."""
    return truncnorm_cdf(min_fidelity, nominal_f, sigma)


def chance_feasible(nominal_f: float, min_fidelity: float,
                    eps: float, sigma: float) -> bool:
    """P(F_r >= F_min) >= 1 - eps."""
    if eps is None or eps >= 1.0:
        return True
    return violation_probability(nominal_f, min_fidelity, sigma) <= eps


def hard_feasible(nominal_f: float, min_fidelity: float) -> bool:
    """Deterministic (baseline) fidelity constraint."""
    return nominal_f >= min_fidelity


def sample_bounded_normal(mu: float, sigma: float, rng: random.Random,
                          lo: Optional[float] = None,
                          hi: Optional[float] = None) -> float:
    """Rejection-sample N(mu, sigma^2), optionally truncated to [lo, hi]
    (either bound may be omitted for an unbounded/half-bounded draw).

    Generalizes the fidelity-specific ``sample_truncnorm`` (below, now a
    thin wrapper) for reuse by future-work item vi's stochastic-window
    joint scheduling, where the bounded quantity is a completion time, not
    a fidelity in [0, 1]."""
    if sigma <= 0:
        x = mu
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        return x
    for _ in range(10000):
        x = rng.gauss(mu, sigma)
        if (lo is None or x >= lo) and (hi is None or x <= hi):
            return x
    x = mu
    if lo is not None:
        x = max(x, lo)
    if hi is not None:
        x = min(x, hi)
    return x


def sample_truncnorm(mu: float, sigma: float, rng: random.Random,
                     lo: float = 0.0, hi: float = 1.0) -> float:
    """Rejection-sample the truncated normal used by the fidelity model.
    Thin wrapper over ``sample_bounded_normal`` -- kept for backward
    compatibility with existing callers/tests."""
    return sample_bounded_normal(mu, sigma, rng, lo=lo, hi=hi)


# ----------------------------------------------------------------------
# candidate expansion
# ----------------------------------------------------------------------
def expand_instance(topology: dict,
                    pairs: List[Tuple[str, str, float, float]],
                    sigma: float,
                    floor_fidelity: float = 0.3,
                    q_values: Optional[List[int]] = None) -> List[dict]:
    """All (path, q) candidates with *no* hard-fidelity pruning.

    Each bundle is stamped with the true request ``min_fidelity`` and
    ``weight``, its utility is recomputed against that requirement (so the
    objective is identical across policies), and the model violation
    probability ``p_viol`` is attached for the chosen ``sigma``.
    """
    if q_values is None:
        q_values = [0, 1, 2]
    bundles: List[dict] = []
    for i, (src, dst, weight, mf) in enumerate(pairs):
        for path in _shortest_paths(topology, src, dst, k=3):
            for q in q_values:
                b = _evaluate_bundle(topology, path, q, src, dst, weight, 0.0)
                if b is None:
                    continue
                if b["fidelity"] < floor_fidelity:
                    continue
                b["request_id"] = f"req_{i}"
                b["bundle_id"] = f"req_{i}_p{len(path)}_q{q}"
                b["min_fidelity"] = mf
                b["weight"] = weight
                margin = max(0.0, b["fidelity"] - mf)
                b["utility"] = (weight * b["success_probability"]
                                * (1.0 + margin)
                                - 0.01 * b["latency"]
                                - 0.05 * b["bell_pair_cost"])
                b["p_viol"] = violation_probability(b["fidelity"], mf, sigma)
                bundles.append(b)
    return bundles


# ----------------------------------------------------------------------
# policy filters
# ----------------------------------------------------------------------
def filter_policy(bundles: List[dict], policy: str,
                  eps: Optional[float] = None) -> List[dict]:
    """Restrict the expanded candidate set to a policy's feasible bundles.

    ``hard``   -> deterministic ``F >= F_min`` (the baseline).
    ``chance`` -> ``P(F_r >= F_min) >= 1 - eps``.
    ``nominal``-> everything (upper bound; unbounded SLA risk).
    """
    if policy == "hard":
        return [b for b in bundles if hard_feasible(b["fidelity"],
                                                    b["min_fidelity"])]
    if policy == "chance":
        return [b for b in bundles if b["p_viol"] <= eps]
    if policy == "nominal":
        return list(bundles)
    raise ValueError(f"unknown policy {policy!r}")


# ----------------------------------------------------------------------
# solvers
# ----------------------------------------------------------------------
def exact_solve(bundles: List[dict], edge_capacities: dict,
                memory_capacities: dict,
                time_limit: float = 60.0) -> Dict:
    from experiments.optimality_benchmark import exact_ilp_solution
    return exact_ilp_solution(bundles, edge_capacities, memory_capacities,
                              time_limit=time_limit)


def qubo_solve(bundles: List[dict], edge_capacities: dict,
               memory_capacities: dict, k: Optional[int],
               num_reads: int, seed: int) -> Dict:
    from optimization.adaptive_qubo import adaptive_qubo_solve, reference_solution
    if k is not None and k > 0:
        return adaptive_qubo_solve(bundles, edge_capacities, memory_capacities,
                                   k=k, num_reads=num_reads, seed=seed)
    return reference_solution(bundles, edge_capacities, memory_capacities,
                              num_reads=num_reads, seed=seed)


# ----------------------------------------------------------------------
# study
# ----------------------------------------------------------------------
def run_chance_constrained_study(topology_fn,
                                 n_requests_list: Optional[List[int]] = None,
                                 eps_list: Optional[List[float]] = None,
                                 sigma_list: Optional[List[float]] = None,
                                 n_instances: int = 4,
                                 n_realizations: int = 60,
                                 time_limit: float = 60.0,
                                 qubo_k: int = 8,
                                 num_reads: int = 20,
                                 tau_mem: float = 500.0,
                                 seed: int = 42) -> Dict:
    """SLA-calibration frontier: hard vs chance(eps) vs nominal policies.

    Under symmetric fidelity noise a bundle whose *mean* fidelity clears
    the requirement can still violate it up to ~1/2 of the time.  The
    deterministic hard rule ``F >= F_min`` therefore has an *implicit,
    uncalibrated* SLA risk, while ``chance(eps)`` -- ``F >= F_min + z_eps*sigma``
    for ``eps < 1/2`` -- delivers exactly the requested reliability (at a
    utility price), and ``chance(eps)`` with ``eps > 1/2`` trades reliability
    for capacity by admitting near-miss bundles.

    For each (n_requests, sigma): expand the candidate set, solve the
    selection ILP exactly under each policy, execute the plan in the
    discrete-event engine with matching fidelity noise, and report
    parametric utility, sampled expected utility, served ratio, and the
    empirical SLA-violation probability.  The full-QUBO solver is run on
    the chance-constrained set for one eps value per size.
    """
    if n_requests_list is None:
        n_requests_list = [6, 8]
    if eps_list is None:
        eps_list = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
    if sigma_list is None:
        sigma_list = [0.03, 0.05]

    from simulation.discrete_event_engine import StochasticEventSimulator

    rows: List[dict] = []
    for n_req in n_requests_list:
        for sigma in sigma_list:
            for inst in range(n_instances):
                iseed = seed * 100000 + n_req * 1000 + inst
                topo = topology_fn()
                nodes = topo["nodes"]
                rng = random.Random(iseed)
                pairs = []
                for _ in range(n_req):
                    src, dst = rng.sample(nodes, 2)
                    pairs.append((src, dst, rng.uniform(10.0, 100.0),
                                  rng.uniform(0.5, 0.7)))
                bundles = expand_instance(topo, pairs, sigma)
                ec, mc = topo["edge_capacities"], topo["memory_capacities"]
                sla = {b["request_id"]: b["min_fidelity"] for b in bundles}
                sim = StochasticEventSimulator(topo, tau_mem=tau_mem,
                                               swap_success=0.95, seed=iseed,
                                               fidelity_noise_sigma=sigma)

                policy_specs = ([("hard", 0.0)]
                                + [("chance", e) for e in eps_list]
                                + [("nominal", 1.0)])
                for policy, eps in policy_specs:
                    feasible = filter_policy(bundles, policy, eps=eps)
                    if not feasible:
                        rows.append({
                            "n_requests": n_req,
                            "sigma": sigma,
                            "policy": policy,
                            "eps": eps,
                            "n_variables": 0,
                            "n_bundles_in": 0,
                            "n_selected": 0,
                            "param_utility": 0.0,
                            "e_utility": 0.0,
                            "reliability_gap": 0.0,
                            "served_ratio": 0.0,
                            "sla_violation_prob": 0.0,
                            "e_delivered_fidelity": float("nan"),
                            "exact_time_s": 0.0,
                            "gap_vs_exact": float("nan"),
                        })
                        continue
                    exact = exact_solve(feasible, ec, mc,
                                        time_limit=time_limit)
                    if exact["u_star"] is None:
                        continue
                    sel = exact["selected"]
                    res = sim.simulate_plan(feasible, sel,
                                            n_realizations=n_realizations,
                                            sla_thresholds=sla)
                    rows.append({
                        "n_requests": n_req,
                        "sigma": sigma,
                        "policy": policy,
                        "eps": eps,
                        "n_variables": exact["n_variables"],
                        "n_bundles_in": len(feasible),
                        "n_selected": len(sel),
                        "param_utility": exact["u_star"],
                        "e_utility": res["e_utility"],
                        "reliability_gap": res["utility_gap"],
                        "served_ratio": res["served_ratio"],
                        "sla_violation_prob": res["sla_violation_prob"],
                        "e_delivered_fidelity": res["e_delivered_fidelity"],
                        "exact_time_s": exact["wall_time_s"],
                        "gap_vs_exact": float("nan"),
                    })

                # QUBO on the chance-constrained set (one eps) per size/sigma
                qeps = 0.2
                feasible = filter_policy(bundles, "chance", eps=qeps)
                if feasible:
                    qr = qubo_solve(feasible, ec, mc, k=qubo_k,
                                    num_reads=num_reads, seed=iseed)
                    exact_q = exact_solve(feasible, ec, mc, time_limit=time_limit)
                    u_star_q = exact_q["u_star"] if exact_q["u_star"] is not None else 0.0
                    qsel = qr.get("selected", [])
                    qutil = sum(next((b["utility"] for b in feasible
                                      if (b["request_id"], b["bundle_id"]) == sel2),
                                     0.0) for sel2 in qsel)
                    resq = sim.simulate_plan(feasible, qsel,
                                             n_realizations=n_realizations,
                                             sla_thresholds=sla)
                    rows.append({
                        "n_requests": n_req,
                        "sigma": sigma,
                        "policy": "chance_qubo",
                        "eps": qeps,
                        "n_variables": exact_q["n_variables"],
                        "n_bundles_in": len(feasible),
                        "n_selected": len(qsel),
                        "param_utility": qutil,
                        "e_utility": resq["e_utility"],
                        "reliability_gap": resq["utility_gap"],
                        "served_ratio": resq["served_ratio"],
                        "sla_violation_prob": resq["sla_violation_prob"],
                        "e_delivered_fidelity": resq["e_delivered_fidelity"],
                        "exact_time_s": exact_q["wall_time_s"],
                        "gap_vs_exact": (u_star_q - qutil) / max(abs(u_star_q), 1e-12),
                    })
                else:
                    rows.append({
                        "n_requests": n_req,
                        "sigma": sigma,
                        "policy": "chance_qubo",
                        "eps": qeps,
                        "n_variables": 0,
                        "n_bundles_in": 0,
                        "n_selected": 0,
                        "param_utility": 0.0,
                        "e_utility": 0.0,
                        "reliability_gap": 0.0,
                        "served_ratio": 0.0,
                        "sla_violation_prob": 0.0,
                        "e_delivered_fidelity": float("nan"),
                        "exact_time_s": 0.0,
                        "gap_vs_exact": 0.0,
                    })
    return {"rows": rows}
