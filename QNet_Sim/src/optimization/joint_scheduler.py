"""Joint entanglement-routing and temporal memory scheduling (Extension 1).

The core claim of the paper title is that routing and *memory scheduling*
are optimized together.  Static solvers treat memory as an aggregate slot
budget; this module enforces the temporal constraint instead:

    M_i(t) <= C_i   for every node i and every time t,

where a selected bundle ``b`` for a request that arrives at ``t_arr`` occupies
``memory_demands[b][i]`` slots on each intermediate node during the window
``[t_arr, t_arr + latency_b)``.  Stored pairs also decohere over that window
(Extension 4): the delivered fidelity is

    F_del(b) = F_end(b) * (1/4)[1 + 3 e^{-hold/T2}(1 + e^{-hold/T1})/2],

and a bundle whose hold time exceeds the coherence lifetime ``tau_mem`` is
expired and excluded.  Deadlines (Extension 18) and priorities enter through
the request profile.

The solver is a Metropolis-style local search over (request -> bundle)
assignments.  The temporal memory state is kept exactly in sync with the
current selection, so every proposed move is either admitted (all windows fit
at every instant) or rejected --- the returned schedule is *feasible at every
time step by construction*.
"""

import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from routing.memory_scheduler import (
    MemoryScheduler, EntangledPair, EntanglementAgeTracker,
    temporal_memory_windows,
)


def _decay(hold_time: float, tau_mem: float, t1: Optional[float] = None) -> float:
    if hold_time <= 0 or tau_mem <= 0:
        return 1.0
    t2 = tau_mem
    t1v = t1 if t1 and t1 > 0 else 2.0 * t2
    return 0.25 * (1.0 + 3.0 * math.exp(-hold_time / t2)
                   * (1.0 + math.exp(-hold_time / t1v)) / 2.0)


def delivered_fidelity(bundle: dict, hold_time: float, tau_mem: float,
                       t1: Optional[float] = None) -> float:
    """End-to-end fidelity after ``hold_time`` of memory storage."""
    return bundle.get("fidelity", 0.0) * _decay(hold_time, tau_mem, t1)


def generate_temporal_bundles(topology: dict, requests: List,
                              q_values: Optional[List[int]] = None,
                              k_paths: int = 3,
                              hold_time_std_frac: float = 0.0) -> Dict[str, List[dict]]:
    """Build per-request candidate bundles tagged with their temporal profile.

    Reuses the paper's bundle evaluator (fidelity after BBPSSW purification,
    swapping, success probability, utility) and attaches the fields the joint
    scheduler needs: ``t_gen``, ``t_consume``, ``hold_time``, ``memory_windows``
    and the request's deadline/priority/n_pairs.

    ``hold_time_std_frac`` (future work item vi, "stochastic-window joint
    scheduling"): fraction of ``hold_time`` used as the standard deviation of
    the bundle's *completion*-time uncertainty, stored as ``hold_time_std``.
    Default ``0.0`` reproduces today's fully deterministic windows exactly
    (no new key value beyond ``0.0``, byte-identical to before this
    parameter existed).
    """
    from experiments.instances import generate_request_bundles
    if q_values is None:
        q_values = [0, 1, 2]
    out: Dict[str, List[dict]] = {}
    for req in requests:
        bundles = generate_request_bundles(
            topology, req.source, req.destination, req.weight,
            req.minimum_fidelity, q_values=q_values)
        tagged = []
        for b in bundles:
            b["request_id"] = req.request_id
            b["n_pairs"] = req.n_pairs
            b["priority"] = req.priority
            b["deadline"] = req.deadline
            b["arrival"] = req.arrival
            b["max_latency"] = req.max_latency
            b["t_gen"] = req.arrival
            b["t_consume"] = req.arrival + b.get("latency", 0.0)
            b["hold_time"] = b["t_consume"] - b["t_gen"]
            b["hold_time_std"] = hold_time_std_frac * b["hold_time"]
            b["memory_windows"] = temporal_memory_windows(
                b, req.arrival, req.max_latency)
            tagged.append(b)
        out[req.request_id] = tagged
    return out


class JointRoutingScheduler:
    """Annealing over (route, purification, memory-window) assignments.

    Parameters
    ----------
    edge_capacities : dict of (u, v) -> int
    memory_capacities : dict of node -> int
    tau_mem : coherence lifetime; bundles with hold > tau_mem are expired
    t1 : optional explicit T1 (defaults to 2*tau_mem)
    risk_weight : penalty weight on (1 - delivered fidelity)
    deadline_weight : soft penalty on missing a deadline (per priority)
    hard_deadline : exclude bundles that cannot finish before the deadline
    seed : RNG seed
    """

    def __init__(self, edge_capacities: dict, memory_capacities: dict,
                 tau_mem: float = 5.0, t1: Optional[float] = None,
                 risk_weight: float = 2.0, deadline_weight: float = 0.0,
                 hard_deadline: bool = True, seed: Optional[int] = None,
                 stochastic: bool = False, hold_time_std: float = 0.0,
                 eps_choices: Optional[List[float]] = None,
                 eps_risk_weight: float = 1.0, mc_samples: int = 100,
                 stochastic_others: bool = False):
        """
        stochastic : bool
            Future work item vi ("chance-constrained joint scheduling that
            co-optimizes epsilon with the temporal admission decision").
            Default False -- every parameter below is inert and the
            scheduler is byte-identical to before this feature existed.
            When True, each candidate bundle is additionally offered at
            every epsilon in ``eps_choices``; admission uses
            ``MemoryScheduler.can_reserve_prob`` instead of the exact
            ``can_reserve``, and ``_energy`` prices the chosen epsilon via
            ``eps_risk_weight`` -- so the anneal must "earn" a riskier
            (looser) epsilon through extra served utility, which is what
            jointly optimizing epsilon with the admission decision means.
        hold_time_std : float
            Fallback completion-time std (in the same units as hold_time)
            used when a bundle doesn't carry its own ``hold_time_std`` (see
            ``generate_temporal_bundles``'s ``hold_time_std_frac``).
        eps_choices : list of float, optional
            Discrete epsilon menu offered per bundle when stochastic=True.
            Defaults to [0.01, 0.05, 0.1, 0.2, 0.4].
        eps_risk_weight : float
            Linear price per unit of epsilon in the energy (only used when
            stochastic=True).
        mc_samples : int
            Monte-Carlo sample budget for each admission check.
        stochastic_others : bool
            Passed through to ``can_reserve_prob`` -- see its docstring.
        """
        self.edge_capacities = {tuple(sorted(k)): v for k, v in edge_capacities.items()}
        self.memory_capacities = dict(memory_capacities)
        self.tau_mem = tau_mem
        self.t1 = t1
        self.risk_weight = risk_weight
        self.deadline_weight = deadline_weight
        self.hard_deadline = hard_deadline
        self.stochastic = stochastic
        self.hold_time_std = hold_time_std
        self.eps_choices = eps_choices if eps_choices is not None else [0.01, 0.05, 0.1, 0.2, 0.4]
        self.eps_risk_weight = eps_risk_weight
        self.mc_samples = mc_samples
        self.stochastic_others = stochastic_others
        self._rng = random.Random(seed)
        self._mc_rng = random.Random(seed)
        # per (request_id, bid) window-fit probability estimated at the
        # moment of reservation, exposed in schedule() as
        # "window_fit_prob_est" (stochastic mode only; avoids re-running
        # the Monte-Carlo estimate a second time just for reporting).
        self._fit_prob_of: Dict[Tuple[str, str], float] = {}

        self._bundles: Dict[str, List[dict]] = {}
        self._bundle_map: Dict[Tuple[str, str], dict] = {}
        self._requests: List[str] = []
        self._edge_of: Dict[Tuple[str, str], dict] = {}
        self._mem_of: Dict[Tuple[str, str], dict] = {}
        self._windows: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
        self._latency_of: Dict[Tuple[str, str], float] = {}
        self._fidelity_of: Dict[Tuple[str, str], float] = {}
        self._priority_of: Dict[str, float] = {}
        self._deadline_of: Dict[str, float] = {}
        self._arrival_of: Dict[str, float] = {}

        self._memory = MemoryScheduler(memory_capacities)
        self._mem_reservations: Dict[Tuple[str, str], List[str]] = {}
        self._edge_load: Dict[tuple, int] = defaultdict(int)
        self.selections: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def add_request(self, request_id: str, bundles: List[dict]):
        self._bundles[request_id] = []
        self._requests.append(request_id)
        self.selections[request_id] = None
        self._mem_reservations[request_id] = []
        for b in bundles:
            b["request_id"] = request_id
            key = (request_id, b["bundle_id"])
            self._bundle_map[key] = b
            self._edge_of[key] = {tuple(sorted(e)): d for e, d in b["edge_demands"].items()}
            self._mem_of[key] = dict(b["memory_demands"])
            self._windows[key] = b.get("memory_windows", {})
            self._latency_of[key] = b.get("latency", 0.0)
            self._fidelity_of[key] = b.get("fidelity", 0.0)
            self._bundles[request_id].append(b)
            self._priority_of[request_id] = b.get("priority", 1.0)
            self._deadline_of[request_id] = b.get("deadline", float("inf"))
            self._arrival_of[request_id] = b.get("arrival", 0.0)

    # ------------------------------------------------------------------
    # stochastic-mode candidate encoding: a "candidate" is either a plain
    # bundle_id (deterministic mode) or a (bundle_id, eps) tuple
    # (stochastic mode); None means reject in both modes.
    # ------------------------------------------------------------------
    def _bid_of(self, candidate):
        if candidate is None:
            return None
        return candidate[0] if self.stochastic else candidate

    def _eps_of(self, candidate) -> Optional[float]:
        if candidate is None or not self.stochastic:
            return None
        return candidate[1]

    def _candidates(self, request_id: str) -> List:
        pool = [b["bundle_id"] for b in self._bundles.get(request_id, [])]
        if request_id in getattr(self, "_frozen", set()):
            cur = self.selections.get(request_id)
            return [cur] if cur is not None else [None]
        if self.hard_deadline:
            d = self._deadline_of.get(request_id, float("inf"))
            filtered = []
            for bid in pool:
                b = self._bundle_map[(request_id, bid)]
                lat = b.get("latency", 0.0)
                if lat > b.get("max_latency", float("inf")):
                    continue
                if self._arrival_of.get(request_id, 0.0) + lat > d:
                    continue
                filtered.append(bid)
            pool = filtered
        if self.stochastic:
            return [(bid, eps) for bid in pool for eps in self.eps_choices] + [None]
        return pool + [None]

    # ------------------------------------------------------------------
    # feasibility bookkeeping (temporal memory is exact)
    # ------------------------------------------------------------------
    def _release(self, request_id: str, candidate):
        bid = self._bid_of(candidate)
        for rid in self._mem_reservations[request_id]:
            self._memory.release(rid)
        self._mem_reservations[request_id] = []
        key = (request_id, bid)
        for edge, d in self._edge_of.get(key, {}).items():
            self._edge_load[edge] -= d
            if self._edge_load[edge] <= 0:
                self._edge_load.pop(edge, None)

    def _reserve(self, request_id: str, candidate) -> bool:
        bid = self._bid_of(candidate)
        eps = self._eps_of(candidate)
        key = (request_id, bid)
        b = self._bundle_map[key]
        # edge feasibility first (aggregate over the horizon) -- unchanged
        # by stochastic mode: only memory/timing is made probabilistic.
        for edge, d in self._edge_of.get(key, {}).items():
            if self._edge_load.get(edge, 0) + d > self.edge_capacities.get(edge, 0):
                return False
        # temporal memory feasibility at every instant
        windows = self._windows.get(key, {})
        demand = self._mem_of.get(key, {})
        hold_std = b.get("hold_time_std", self.hold_time_std) if self.stochastic else 0.0
        fit_probs = []
        for node, (start, end) in windows.items():
            if start >= end:
                continue
            amount = demand.get(node, 0)
            if self.stochastic:
                prob = self._memory.estimate_fit_probability(
                    node, amount, start, end, start_std=0.0, end_std=hold_std,
                    n_samples=self.mc_samples, rng=self._mc_rng,
                    stochastic_others=self.stochastic_others)
                fit_probs.append(prob)
                if prob < (1.0 - eps):
                    return False
            else:
                if not self._memory.can_reserve(node, amount, start, end):
                    return False
        for edge, d in self._edge_of.get(key, {}).items():
            self._edge_load[edge] = self._edge_load.get(edge, 0) + d
        for node, (start, end) in windows.items():
            if start >= end:
                continue
            rid = self._memory.reserve(node, demand.get(node, 0), start, end,
                                       std=hold_std)
            if rid is not None:
                self._mem_reservations[request_id].append(rid)
        if self.stochastic:
            self._fit_prob_of[key] = min(fit_probs) if fit_probs else 1.0
        return True

    def _apply(self, request_id: str, new_candidate):
        old_candidate = self.selections.get(request_id)
        if old_candidate is not None:
            self._release(request_id, old_candidate)
        self.selections[request_id] = new_candidate
        if new_candidate is not None:
            ok = self._reserve(request_id, new_candidate)
            if not ok:  # pragma: no cover - reserve() pre-checked by the caller
                self.selections[request_id] = None

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------
    def _energy(self) -> float:
        total = 0.0
        for request_id, candidate in self.selections.items():
            if candidate is None:
                continue
            bid = self._bid_of(candidate)
            eps = self._eps_of(candidate)
            key = (request_id, bid)
            b = self._bundle_map[key]
            priority = self._priority_of.get(request_id, 1.0)
            w = b.get("n_pairs", 1) * priority
            total -= w * b["utility"]
            if self.risk_weight > 0:
                f_del = delivered_fidelity(b, b.get("hold_time", 0.0),
                                           self.tau_mem, self.t1)
                total += self.risk_weight * (1.0 - f_del)
            if self.deadline_weight > 0:
                completion = self._arrival_of.get(request_id, 0.0) + b.get("latency", 0.0)
                d = self._deadline_of.get(request_id, float("inf"))
                if completion > d:
                    total += self.deadline_weight * priority * (completion - d)
            if self.stochastic and eps is not None:
                # Lagrangian-style price on risk tolerance: a looser
                # (larger) epsilon must be "earned" by extra served
                # utility elsewhere -- this is what jointly optimizing
                # epsilon with the admission decision means.
                total += self.eps_risk_weight * eps
        for edge, load in self._edge_load.items():
            cap = self.edge_capacities.get(edge, 0)
            if load > cap:
                total += 10.0 * (load - cap) ** 2
        return total

    # ------------------------------------------------------------------
    # greedy seed
    # ------------------------------------------------------------------
    def _greedy_seed(self):
        order = sorted(self._requests,
                       key=lambda rid: (-self._priority_of.get(rid, 1.0), rid))
        for request_id in order:
            scored = []
            for candidate in self._candidates(request_id):
                if candidate is None:
                    continue
                bid = self._bid_of(candidate)
                eps = self._eps_of(candidate)
                b = self._bundle_map[(request_id, bid)]
                hold = b.get("hold_time", 0.0)
                if self.tau_mem > 0 and hold > self.tau_mem:
                    continue
                f_del = delivered_fidelity(b, hold, self.tau_mem, self.t1)
                score = b["utility"] * self._priority_of.get(request_id, 1.0) \
                    - self.risk_weight * (1.0 - f_del)
                if eps is not None:
                    score -= self.eps_risk_weight * eps
                    # prefer the tightest (smallest) eps among equal-score
                    # ties, so the greedy seed doesn't gratuitously buy risk
                    scored.append((score, -eps, candidate))
                else:
                    scored.append((score, 0.0, candidate))
            scored.sort(reverse=True)
            for _, _, candidate in scored:
                if self._reserve(request_id, candidate):
                    self.selections[request_id] = candidate
                    break

    # ------------------------------------------------------------------
    # anneal
    # ------------------------------------------------------------------
    def solve(self, max_iterations: int = 4000, initial_temperature: float = 10.0,
              cooling_rate: float = 0.98, steps_per_temperature: int = 20,
              min_temperature: float = 1e-3, n_restarts: int = 3,
              movable: Optional[List[str]] = None) -> Dict:
        """Anneal over the request->bundle assignment.

        ``movable`` restricts the anneal to a subset of request ids (the rest
        are frozen at their current assignment); this is what the online /
        receding-horizon wrappers use to commit decisions irrevocably.
        """
        if movable is not None:
            movable_ids = list(movable)
            frozen = {rid for rid in self._requests if rid not in set(movable_ids)}
        else:
            movable_ids = list(self._requests)
            frozen = set()
        self._frozen = frozen
        if not movable_ids:
            # no movable requests: nothing to anneal
            return self.schedule()
        best = None
        best_energy = float("inf")
        for restart in range(n_restarts):
            if restart == 0:
                self._greedy_seed()
            else:
                for rid in self._requests:
                    if rid not in frozen:
                        self.selections[rid] = None
            current_energy = self._energy()
            temperature = initial_temperature
            for _ in range(max_iterations):
                if temperature < min_temperature:
                    break
                for _ in range(steps_per_temperature):
                    request_id = self._rng.choice(movable_ids) if movable_ids else self._rng.choice(self._requests)
                    pool = self._candidates(request_id)
                    old_bid = self.selections.get(request_id)
                    new_bid = self._rng.choice(pool)
                    if new_bid == old_bid:
                        continue
                    # compute the energy delta without mutating state
                    delta = self._energy_delta(request_id, old_bid, new_bid)
                    if delta < 0 or (temperature > 0 and self._rng.random()
                                     < math.exp(-delta / temperature)):
                        self._apply(request_id, new_bid)
                        current_energy += delta
                if current_energy < best_energy:
                    best = dict(self.selections)
                    best_energy = current_energy
                temperature *= cooling_rate
            if current_energy < best_energy:
                best = dict(self.selections)
                best_energy = current_energy
            self.selections = {rid: None for rid in self._requests}

        self.selections = best
        self._frozen = set()
        return self.schedule()

    def _energy_delta(self, request_id: str, old_candidate,
                      new_candidate) -> float:
        """Energy difference between the current state and one where
        ``request_id``'s assignment changes from ``old_candidate`` to
        ``new_candidate`` (no mutation). Candidates are bundle ids
        (deterministic mode) or (bundle_id, eps) tuples (stochastic mode)."""
        old_e = self._energy()
        old_candidate_now = self.selections.get(request_id)
        # temporarily switch (mutating edge/memory state), measure, revert
        if old_candidate_now is not None:
            self._release(request_id, old_candidate_now)
        self.selections[request_id] = None
        if new_candidate is not None and self._reserve(request_id, new_candidate):
            self.selections[request_id] = new_candidate
            new_e = self._energy()
        else:
            new_e = self._energy()
        # revert
        if new_candidate is not None and self.selections.get(request_id) == new_candidate:
            self._release(request_id, new_candidate)
        self.selections[request_id] = old_candidate_now
        if old_candidate_now is not None:
            self._reserve(request_id, old_candidate_now)
        return new_e - old_e

    # ------------------------------------------------------------------
    # schedule output
    # ------------------------------------------------------------------
    def schedule(self) -> Dict:
        """Return the joint schedule with per-request decisions, temporal
        reservations and aggregate metrics."""
        decisions = []
        delivered_fids = []
        deadlines_met = 0
        total_pairs = 0
        latency_sum = 0.0
        served = 0
        for request_id, candidate in self.selections.items():
            if candidate is None:
                continue
            bid = self._bid_of(candidate)
            eps = self._eps_of(candidate)
            b = self._bundle_map[(request_id, bid)]
            hold = b.get("hold_time", 0.0)
            f_del = delivered_fidelity(b, hold, self.tau_mem, self.t1)
            completion = self._arrival_of.get(request_id, 0.0) + b.get("latency", 0.0)
            on_time = completion <= self._deadline_of.get(request_id, float("inf"))
            decision = {
                "request_id": request_id,
                "bundle_id": bid,
                "path": b.get("path", []),
                "fidelity": b.get("fidelity", 0.0),
                "delivered_fidelity": f_del,
                "completion_time": completion,
                "deadline": self._deadline_of.get(request_id, float("inf")),
                "on_time": on_time,
                "hold_time": hold,
                "n_pairs": b.get("n_pairs", 1),
                "latency": b.get("latency", 0.0),
            }
            if self.stochastic:
                decision["eps"] = eps
                decision["window_fit_prob_est"] = self._fit_prob_of.get(
                    (request_id, bid), 1.0)
            decisions.append(decision)
            delivered_fids.append(f_del)
            deadlines_met += int(on_time)
            total_pairs += b.get("n_pairs", 1)
            latency_sum += b.get("latency", 0.0)
            served += 1

        n = len(self._requests)
        t_max = max((self._deadline_of.get(r, 0.0) for r in self._requests), default=1.0)
        return {
            "decisions": decisions,
            "served": served,
            "n_requests": n,
            "served_ratio": served / max(n, 1),
            "on_time": deadlines_met,
            "on_time_ratio": deadlines_met / max(served, 1),
            "mean_delivered_fidelity": (sum(delivered_fids) / len(delivered_fids)
                                        if delivered_fids else 0.0),
            "total_pairs_delivered": total_pairs,
            "mean_latency": latency_sum / max(served, 1),
            "mean_memory_utilization": self._memory.utilization(0.0, t_max, max(t_max / 50.0, 1e-3)),
            "energy": self._energy(),
        }


def run_stochastic_chance_joint_study(topology_fn: Callable, n_slots: int = 20,
                                      mean_rate: float = 1.5, tau_mem: float = 5.0,
                                      hold_time_std_frac: float = 0.15,
                                      eps_choices: Optional[List[float]] = None,
                                      eps_risk_weight_list: Optional[List[float]] = None,
                                      n_instances: int = 3, mc_samples: int = 100,
                                      seed: int = 42) -> Dict:
    """Future work items iv + vi: stochastic-window joint scheduling and
    chance-constrained joint scheduling that co-optimizes epsilon with the
    temporal admission decision.

    Compares three regimes on IDENTICAL stochastic-hold traces:
      - "deterministic": the unchanged legacy path (stochastic=False).
      - "stochastic_fixed_eps(eps=..)": epsilon is externally fixed (a
        single-choice eps_choices list), not optimized -- mirrors the old
        chance-constrained study's external eps sweep, now applied to the
        temporal admission decision instead of a static candidate filter.
      - "stochastic_coopt_eps": epsilon is a per-request decision variable
        the anneal jointly optimizes alongside (bundle, admission-time)
        via the full eps_choices menu.

    Reports served_ratio, utility, mean chosen eps, and a POST-HOC empirical
    SLA-violation estimate from an INDEPENDENT, larger Monte-Carlo replay of
    the committed windows (a separate sample budget from the one used inside
    the anneal, to avoid optimistic self-validation).
    """
    from routing.temporal_request import generate_dynamic_trace
    import random as _random

    if eps_choices is None:
        eps_choices = [0.01, 0.05, 0.1, 0.2, 0.4]
    if eps_risk_weight_list is None:
        eps_risk_weight_list = [1.0]

    rows = []
    for inst in range(n_instances):
        inst_seed = seed + inst
        topo = topology_fn()
        ec, mc = topo["edge_capacities"], topo["memory_capacities"]
        trace = generate_dynamic_trace(topo, n_slots, mean_rate,
                                       deadline_horizon=6.0, seed=inst_seed)
        bundles = generate_temporal_bundles(topo, trace,
                                            hold_time_std_frac=hold_time_std_frac)

        regimes = [("deterministic", False, [0.0], 0.0)]
        for eps in eps_choices:
            regimes.append((f"stochastic_fixed_eps({eps})", True, [eps], 1.0))
        for w in eps_risk_weight_list:
            regimes.append((f"stochastic_coopt_eps(w={w})", True, eps_choices, w))

        for label, stochastic, this_eps_choices, eps_w in regimes:
            sched = JointRoutingScheduler(
                ec, mc, tau_mem=tau_mem, hard_deadline=True, risk_weight=2.0,
                seed=inst_seed, stochastic=stochastic,
                eps_choices=this_eps_choices, eps_risk_weight=eps_w,
                mc_samples=mc_samples)
            for rid, bs in bundles.items():
                sched.add_request(rid, bs)
            result = sched.solve(max_iterations=1500, n_restarts=2)

            mean_eps = None
            if stochastic and result["decisions"]:
                mean_eps = sum(d["eps"] for d in result["decisions"]) / len(result["decisions"])

            # Post-hoc empirical SLA-violation estimate: independent replay
            # of each committed window's actual fit probability, using a
            # LARGER, separate sample budget than the one used during the
            # anneal (avoids optimistic self-validation).
            violations = 0
            checked = 0
            if stochastic:
                validation_memory = MemoryScheduler(mc)
                validation_rng = random.Random(inst_seed + 1000)
                for d in result["decisions"]:
                    key = (d["request_id"], d["bundle_id"])
                    windows = sched._windows.get(key, {})
                    demand = sched._mem_of.get(key, {})
                    b = sched._bundle_map.get(key, {})
                    hold_std = b.get("hold_time_std", 0.0)
                    for node, (start, end) in windows.items():
                        if start >= end:
                            continue
                        checked += 1
                        prob = validation_memory.estimate_fit_probability(
                            node, demand.get(node, 0), start, end,
                            end_std=hold_std, n_samples=mc_samples * 3,
                            rng=validation_rng)
                        if prob < (1.0 - d["eps"]):
                            violations += 1
                        validation_memory.reserve(node, demand.get(node, 0),
                                                  start, end, std=hold_std)

            rows.append({
                "instance": inst,
                "regime": label,
                "served": result["served"],
                "n_requests": result["n_requests"],
                "served_ratio": result["served_ratio"],
                "mean_delivered_fidelity": result["mean_delivered_fidelity"],
                "utility": sum(d["n_pairs"] * b.get("utility", 0.0)
                               for d, b in ((d, sched._bundle_map.get((d["request_id"], d["bundle_id"]), {}))
                                            for d in result["decisions"])),
                "mean_eps": mean_eps,
                "empirical_violation_rate": (violations / checked) if checked else None,
            })

    return {"rows": rows}


def run_joint_comparison(topology_fn: Callable, n_slots: int = 20,
                         mean_rate: float = 1.5, tau_mem: float = 5.0,
                         deadline_horizon: float = 6.0, seed: int = 42,
                         hard_deadline: bool = True) -> Dict:
    """Run the joint router on a dynamic trace and compare against a
    memory-agnostic (aggregate-slot) router on the same requests.

    Returns aggregate metrics for both, exposing how often temporal memory
    conflicts force the joint router to drop or re-route requests.
    """
    from experiments.instances import generate_chain_topology
    from routing.temporal_request import generate_dynamic_trace
    from optimization.streaming_annealer import StreamingAnnealer

    topo = topology_fn()
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    trace = generate_dynamic_trace(topo, n_slots, mean_rate,
                                   deadline_horizon=deadline_horizon, seed=seed)
    bundles = generate_temporal_bundles(topo, trace)

    joint = JointRoutingScheduler(ec, mc, tau_mem=tau_mem, hard_deadline=hard_deadline,
                                  risk_weight=2.0, seed=seed)
    for rid, bs in bundles.items():
        joint.add_request(rid, bs)
    joint_schedule = joint.solve(max_iterations=2000, n_restarts=2)

    # memory-agnostic baseline: streaming annealer with aggregate memory demand
    sa = StreamingAnnealer(ec, mc, seed=seed, risk_weight=2.0, risk_tau=tau_mem,
                           use_fidelity_risk=True)
    flat = []
    for req in trace:
        for b in bundles[req.request_id]:
            b["request_id"] = req.request_id
            flat.append(b)
    sa_bundles_by_rid = {}
    for rid, bs in bundles.items():
        sa_bundles_by_rid[rid] = bs
        sa.add_request(rid, bs)
        sa.local_sweep(n_steps=30, temperature=2.0)
    sa_sel = sa.get_selected()

    util_of = {(b["request_id"], b["bundle_id"]): b["utility"] for b in flat}
    sa_util = sum(util_of.get(k, 0.0) for k in sa_sel)

    return {
        "trace": {"n_requests": len(trace), "n_slots": n_slots,
                  "mean_rate": mean_rate, "tau_mem": tau_mem},
        "joint": {
            "served": joint_schedule["served"],
            "served_ratio": joint_schedule["served_ratio"],
            "on_time_ratio": joint_schedule["on_time_ratio"],
            "mean_delivered_fidelity": joint_schedule["mean_delivered_fidelity"],
            "mean_memory_utilization": joint_schedule["mean_memory_utilization"],
            "utility": sum(d["n_pairs"] * 1.0 * util_of.get((d["request_id"], d["bundle_id"]), 0.0)
                           for d in joint_schedule["decisions"]),
        },
        "memory_agnostic": {
            "served": len(sa_sel),
            "served_ratio": len(sa_sel) / max(len(trace), 1),
            "utility": sa_util,
        },
    }
