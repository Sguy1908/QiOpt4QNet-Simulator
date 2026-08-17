"""Temporal quantum-memory model (paper Extensions 1 & 4).

The static memory model in the bundle layer treats memory as a fixed slot
count: a request "uses M slots at node i" for the whole solve.  Real quantum
memories are *temporal* resources:

* each entangled pair occupies a slot only between its generation time and
  its consumption (swap / purification / delivery) time,
* pairs decay while stored: F(a) = F0 * (1/4)[1 + 3 e^{-a/T2}(1 + e^{-a/T1})/2]
  where ``a = t - t_gen`` is the pair's age,
* a pair that outlives its coherence window (age > tau_mem) is *expired* and
  must be discarded or regenerated.

This module provides the two building blocks used by the joint router:

* :class:`EntangledPair` -- an individual Bell pair with an age-dependent
  fidelity and an expiry test,
* :class:`MemoryScheduler` -- a per-node, per-time-slot occupancy tracker that
  answers ``M_i(t) <= C_i`` for *every* t, instead of only as an aggregate.

``temporal_memory_windows`` converts a static bundle's node demand into the
per-node occupation intervals a router must reserve.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple


@dataclass
class EntangledPair:
    """A single stored Bell pair with an age-dependent fidelity.

    Fidelity decays from ``initial_fidelity`` under the standard storage model
    F(t) = (1/4)[1 + 3 e^{-t/T2} (1 + e^{-t/T1}) / 2], the same convention used
    by ``optimization.time_dependent_optimizer.fidelity_after_hold``.
    """

    pair_id: str
    node: str
    generation_time: float
    initial_fidelity: float
    t2: float = 1.0
    t1: Optional[float] = None

    def age(self, t: float) -> float:
        return max(0.0, t - self.generation_time)

    def fidelity_at(self, t: float) -> float:
        a = self.age(t)
        if a <= 0 or self.t2 <= 0:
            return self.initial_fidelity
        t1 = self.t1 if self.t1 and self.t1 > 0 else 2.0 * self.t2
        decay = 0.25 * (1.0 + 3.0 * math.exp(-a / self.t2)
                        * (1.0 + math.exp(-a / t1)) / 2.0)
        return self.initial_fidelity * decay

    def is_expired(self, t: float, tau_mem: float) -> bool:
        """A pair is expired once its stored fidelity drops below the fully
        mixed-state floor (age >> tau_mem) --- here modelled as age exceeding
        the coherence lifetime ``tau_mem``."""
        return self.age(t) >= tau_mem


class EntanglementAgeTracker:
    """Tracks stored pairs and recommends keep/swap/purify/discard actions.

    This turns the scheduler-level decision (Extension 4) into an explicit,
    queryable policy: at any time ``t`` a node can ask what to do with each of
    its stored pairs given its current fidelity threshold.
    """

    def __init__(self, discard_floor: float = 0.5, purify_threshold: float = 0.9,
                 tau_mem: float = 1.0):
        self.pairs: Dict[str, EntangledPair] = {}
        self.discard_floor = discard_floor
        self.purify_threshold = purify_threshold
        self.tau_mem = tau_mem

    def add(self, pair: EntangledPair):
        self.pairs[pair.pair_id] = pair

    def remove(self, pair_id: str):
        self.pairs.pop(pair_id, None)

    def decide(self, pair_id: str, t: float) -> str:
        """Return one of ``keep`` / ``swap`` / ``purify`` / ``discard`` /
        ``replace`` for a stored pair at time ``t``."""
        pair = self.pairs.get(pair_id)
        if pair is None:
            return "replace"
        if pair.is_expired(t, self.tau_mem):
            return "discard"
        f = pair.fidelity_at(t)
        if f < self.discard_floor:
            return "discard"
        if f < self.purify_threshold:
            return "purify"
        return "keep" if t - pair.generation_time < self.tau_mem / 2.0 else "swap"

    def report(self, t: float) -> Dict[str, str]:
        return {pid: self.decide(pid, t) for pid in self.pairs}


class MemoryScheduler:
    """Time-indexed memory occupancy tracker for one node or a whole network.

    Reservations are half-open intervals ``[start, end)`` on a per-node
    timeline.  ``M_i(t)`` is the number of pairs occupying node ``i`` at time
    ``t``; a reservation is admitted only if ``M_i(t) + amount <= C_i`` for
    every ``t`` in the interval.
    """

    def __init__(self, memory_capacities: Dict[str, int]):
        self.capacities = dict(memory_capacities)
        self._reservations: Dict[str, List[Dict]] = {}
        for node in self.capacities:
            self._reservations[node] = []
        self._next_id = 0

    def _occupancy_at(self, node: str, t: float) -> int:
        return sum(r["amount"] for r in self._reservations[node]
                   if r["start"] <= t < r["end"])

    def occupancy(self, node: str, t: float) -> int:
        return self._occupancy_at(node, t)

    def peak_occupancy(self, node: str) -> int:
        events = []
        for r in self._reservations[node]:
            events.append((r["start"], r["amount"]))
            events.append((r["end"], -r["amount"]))
        events.sort()
        cur, peak = 0, 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        return peak

    def can_reserve(self, node: str, amount: int, start: float, end: float) -> bool:
        """True if ``amount`` slots can be occupied on ``node`` throughout
        ``[start, end)`` without exceeding capacity at any instant.

        Load is piecewise constant and only changes at reservation event
        times, so it suffices to check ``start`` and every event time strictly
        inside the window.
        """
        cap = self.capacities.get(node, 0)
        if amount <= 0 or end <= start:
            return True
        check_times = {start}
        for r in self._reservations[node]:
            if start < r["start"] < end:
                check_times.add(r["start"])
        for t in check_times:
            if self._occupancy_at(node, t) + amount > cap:
                return False
        return True

    def reserve(self, node: str, amount: int, start: float, end: float,
               std: float = 0.0) -> Optional[str]:
        """Reserve ``amount`` slots on ``node`` for ``[start, end)``.  Returns a
        reservation id, or ``None`` if infeasible at any instant.

        ``std`` (default 0.0, fully backward compatible) optionally records
        this reservation's completion-time uncertainty, used only by
        ``can_reserve_prob``'s ``stochastic_others=True`` mode -- the
        deterministic ``can_reserve``/``reserve`` path never reads it."""
        if not self.can_reserve(node, amount, start, end):
            return None
        rid = f"mem_{self._next_id}"
        self._next_id += 1
        self._reservations[node].append({
            "id": rid, "node": node, "amount": amount,
            "start": start, "end": end, "std": std,
        })
        return rid

    def _fits_against(self, reservations: List[Dict], cap: int, amount: int,
                      start: float, end: float) -> bool:
        if amount <= 0 or end <= start:
            return True
        check_times = {start}
        for r in reservations:
            if start < r["start"] < end:
                check_times.add(r["start"])
        for t in check_times:
            occ = sum(r["amount"] for r in reservations if r["start"] <= t < r["end"])
            if occ + amount > cap:
                return False
        return True

    def estimate_fit_probability(self, node: str, amount: int, start_mean: float,
                                 end_mean: float, start_std: float = 0.0,
                                 end_std: float = 0.0, n_samples: int = 200,
                                 rng: Optional[random.Random] = None,
                                 stochastic_others: bool = False) -> float:
        """Monte-Carlo estimate of P(the window fits at every instant),
        i.e. the fraction of resampled draws that pass ``can_reserve``.

        The candidate reservation's own ``[start, end)`` window is resampled
        from independent normals ``N(start_mean, start_std)`` /
        ``N(end_mean, end_std)`` each draw. ``stochastic_others=False``
        (default) checks each draw against the already-committed
        reservations at their *nominal* (mean) times -- cheaper, and
        mirrors the scheduler's existing sequential/online commitment
        style, where only the request being admitted right now is
        uncertain. ``stochastic_others=True`` additionally resamples every
        currently-committed reservation that carries a nonzero ``std`` (a
        fully joint chance constraint; more expensive since it resamples
        the whole node's reservation list on every draw)."""
        cap = self.capacities.get(node, 0)
        rng = rng or random.Random()
        successes = 0
        for _ in range(n_samples):
            s = rng.gauss(start_mean, start_std) if start_std > 0 else start_mean
            e = rng.gauss(end_mean, end_std) if end_std > 0 else end_mean
            if e < s:
                e = s
            if stochastic_others:
                sampled = []
                for r in self._reservations.get(node, []):
                    std = r.get("std", 0.0)
                    r_start = rng.gauss(r["start"], std) if std > 0 else r["start"]
                    duration = r["end"] - r["start"]
                    sampled.append({"start": r_start, "end": r_start + duration,
                                    "amount": r["amount"]})
                fits = self._fits_against(sampled, cap, amount, s, e)
            else:
                fits = self.can_reserve(node, amount, s, e)
            if fits:
                successes += 1
        return successes / n_samples

    def can_reserve_prob(self, node: str, amount: int, start_mean: float,
                         end_mean: float, start_std: float = 0.0,
                         end_std: float = 0.0, eps: float = 0.0,
                         n_samples: int = 200,
                         rng: Optional[random.Random] = None,
                         stochastic_others: bool = False) -> bool:
        """True iff ``estimate_fit_probability(...) >= 1 - eps``. When
        ``start_std == end_std == 0`` (and ``stochastic_others`` isn't
        asking for extra randomness) this reduces exactly to the
        deterministic ``can_reserve`` -- the ``std=0`` degenerate case."""
        has_other_uncertainty = stochastic_others and any(
            r.get("std", 0.0) > 0 for r in self._reservations.get(node, []))
        if start_std <= 0.0 and end_std <= 0.0 and not has_other_uncertainty:
            return self.can_reserve(node, amount, start_mean, end_mean)
        prob = self.estimate_fit_probability(
            node, amount, start_mean, end_mean, start_std=start_std,
            end_std=end_std, n_samples=n_samples, rng=rng,
            stochastic_others=stochastic_others)
        return prob >= (1.0 - eps)

    def release(self, reservation_id: str):
        for node, reservations in self._reservations.items():
            for i, r in enumerate(reservations):
                if r["id"] == reservation_id:
                    reservations.pop(i)
                    return

    def timeline(self, t_start: float, t_end: float, dt: float) -> Dict[str, List[int]]:
        """Per-node occupancy sampled on a uniform grid (for plotting/audit)."""
        out = {}
        t = t_start
        while t < t_end:
            for node in self.capacities:
                out.setdefault(node, []).append(self._occupancy_at(node, t))
            t += dt
        return out

    def utilization(self, t_start: float, t_end: float, dt: float) -> float:
        """Mean memory utilization across nodes and time (in [0, 1])."""
        tl = self.timeline(t_start, t_end, dt)
        totals, counts = 0.0, 0
        for node, series in tl.items():
            if not series:
                continue
            cap = max(self.capacities[node], 1)
            totals += sum(series) / (len(series) * cap)
            counts += 1
        return totals / max(counts, 1)


def temporal_memory_windows(bundle: dict, arrival_time: float,
                            tau_mem: float) -> Dict[str, Tuple[float, float]]:
    """Convert a bundle's static per-node memory demand into occupation
    windows ``{node: (t_occupy, t_release)}``.

    Pairs are generated at ``arrival_time`` (the request's start) and held on
    every intermediate node until the end-to-end link is established at
    ``arrival_time + latency``.  Storage beyond ``tau_mem`` is clipped: the
    pair would expire there, so the router should prefer a lower-latency
    bundle.
    """
    latency = bundle.get("latency", 0.0)
    release = arrival_time + min(latency, tau_mem) if latency > 0 else arrival_time
    windows = {}
    for node, _demand in (bundle.get("memory_demands") or {}).items():
        windows[node] = (arrival_time, release)
    return windows
