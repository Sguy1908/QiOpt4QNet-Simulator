"""Decoherence-aware, time-dependent routing (paper Future Work item i).

Static epochs assume the memory model is time-independent: memory capacity is a
fixed slot count and the risk term uses a static latency proxy.  Real quantum
memories are decaying: a Bell pair held at a repeater for ``Delta t`` loses
fidelity as exp(-Delta t / tau_mem), and the longer the network has been
running the more degraded stored pairs become.

This module implements a time-slotted router built on top of the streaming
annealer.  The decoherence-aware variant keeps a warm-start persistent
annealer across slots and grows the memory-risk weight with the epoch age, so
later-arriving requests are biased toward low-latency bundles; delivered
fidelity is then recomputed with hold-time decay.  ``run_time_dependent_
comparison`` benchmarks it against a decoherence-agnostic (static) baseline on
the same synthetic Poisson arrival trace.
"""

import csv
import math
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def fidelity_after_hold(fidelity: float, hold_time: float, tau_mem: float,
                        t1: Optional[float] = None) -> float:
    """T1/T2 decoherence of a stored Werner-like Bell pair.

    Uses the storage model of the manuscript (Eq. ``memory``)
        F(t) = 1/4 + (F0 - 1/4) eta(t),  eta(t) = e^{-t/T2} (1 + e^{-t/T1}) / 2
    with T2 = tau_mem and T1 = 2*T2 by default. F(0) = F0 and F(t) -> 1/4
    (the completely mixed state) as t -> inf for every F0.
    """
    if hold_time <= 0 or tau_mem <= 0:
        return fidelity
    t2 = tau_mem
    t1v = t1 if t1 and t1 > 0 else 2.0 * t2
    eta = math.exp(-hold_time / t2) * (1.0 + math.exp(-hold_time / t1v)) / 2.0
    return 0.25 + (fidelity - 0.25) * eta


def _utility_of(bundles, selected):
    util_of = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return sum(util_of.get(k, 0.0) for k in selected)


def _fidelity_of(bundles, selected):
    fid_of = {(b["request_id"], b["bundle_id"]): b.get("fidelity", 0.0) for b in bundles}
    vals = [fid_of.get(k, 0.0) for k in selected if k in fid_of]
    return (sum(vals) / len(vals)) if vals else 0.0


def _delivered_fidelity(bundles, selected, hold_time_fn,
                        tau_mem: float = 5.0, t1: Optional[float] = None):
    """Mean end-to-end fidelity after applying hold-time decoherence."""
    fid_of = {(b["request_id"], b["bundle_id"]): b.get("fidelity", 0.0) for b in bundles}
    vals = []
    for rid, bid in selected:
        if (rid, bid) not in fid_of:
            continue
        hold = hold_time_fn(rid, bid)
        vals.append(fidelity_after_hold(fid_of[(rid, bid)], hold, tau_mem, t1))
    return (sum(vals) / len(vals)) if vals else 0.0


class TimeDependentAnnealer:
    """Streaming annealer whose memory-risk weight grows with epoch age.

    Warm-starts a single persistent StreamingAnnealer across slots.  At each
    slot the risk weight is ``base_risk_weight * (1 + slot/n_slots)``: requests
    that arrive late in the epoch hold pairs longer (until the next full
    re-optimization) and are therefore penalised more for high-latency bundles.
    """

    def __init__(self, edge_capacities: dict, memory_capacities: dict,
                 tau_mem: float = 5.0, base_risk_weight: float = 2.0,
                 n_slots: int = 30, seed: Optional[int] = None):
        from optimization.streaming_annealer import StreamingAnnealer
        self.edge_capacities = edge_capacities
        self.memory_capacities = memory_capacities
        self.tau_mem = tau_mem
        self.base_risk_weight = base_risk_weight
        self.n_slots = n_slots
        self._rng = random.Random(seed)
        self._sa = None

    def step(self, slot_idx: int, arrivals: Dict[str, List[dict]],
             n_local_steps: int = 50) -> List[Tuple[str, str]]:
        from optimization.streaming_annealer import StreamingAnnealer
        if self._sa is None:
            self._sa = StreamingAnnealer(
                self.edge_capacities, self.memory_capacities,
                seed=self._rng.randint(0, 2 ** 31 - 1),
                risk_weight=self.base_risk_weight, risk_tau=self.tau_mem,
                use_fidelity_risk=True,
                hold_scale=1.0 + self.n_slots / 2.0)
        for rid, bundles in arrivals.items():
            if not bundles:
                continue
            self._sa.add_request(rid, bundles)
        self._sa.risk_weight = self.base_risk_weight * (1.0 + slot_idx / max(self.n_slots, 1))
        self._sa.local_sweep(n_steps=n_local_steps, temperature=2.0)
        return self._sa.get_selected()


class StaticBaselineRouter:
    """Decoherence-agnostic baseline: risk term disabled, fixed memory model."""

    def __init__(self, edge_capacities: dict, memory_capacities: dict,
                 seed: Optional[int] = None):
        from optimization.streaming_annealer import StreamingAnnealer
        self.edge_capacities = edge_capacities
        self.memory_capacities = memory_capacities
        self._rng = random.Random(seed)
        self._sa = StreamingAnnealer(edge_capacities, memory_capacities, seed=seed)

    def step(self, arrivals: Dict[str, List[dict]], n_local_steps: int = 50):
        for rid, bundles in arrivals.items():
            if not bundles:
                continue
            self._sa.add_request(rid, bundles)
        self._sa.local_sweep(n_steps=n_local_steps, temperature=2.0)
        return self._sa.get_selected()


def _poisson(rng: random.Random, lam: float) -> int:
    """Poisson(lam) sample from a seeded ``random.Random`` (no numpy needed)."""
    if lam <= 0:
        return 0
    count = 0
    remaining = 1.0
    while True:
        u = rng.random()
        if u <= 0.0:
            u = 1e-12
        remaining -= -math.log(u) / lam
        if remaining < 0.0:
            break
        count += 1
    return count


def _poisson_trace(topology_fn: Callable, n_slots: int, mean_rate: float,
                   request_weight_bounds: Tuple[float, float] = (10.0, 60.0),
                   seed: int = 42) -> List[Dict[str, List[dict]]]:
    """Synthetic Poisson arrival trace: list of per-slot arrival dictionaries."""
    from experiments.instances import generate_request_bundles
    rng = random.Random(seed)
    topo = topology_fn()
    nodes = topo["nodes"]
    trace = []
    for slot in range(n_slots):
        slot_arrivals = {}
        for k in range(_poisson(rng, mean_rate)):
            src, dst = rng.sample(nodes, 2)
            rid = f"req_{slot}_{k}"
            w = rng.uniform(*request_weight_bounds)
            mf = rng.uniform(0.5, 0.8)
            bs = generate_request_bundles(topo, src, dst, w, mf)
            for i, b in enumerate(bs):
                b["request_id"] = rid
                b["bundle_id"] = f"{rid}_b{i}"
            slot_arrivals[rid] = bs
        trace.append(slot_arrivals)
    return trace


def run_time_dependent_comparison(topology_fn: Callable, n_slots: int = 30,
                                  mean_rate: float = 1.5, tau_mem: float = 5.0,
                                  t1_us: Optional[float] = None,
                                  n_local_steps: int = 30, seed: int = 42,
                                  risk_gain: float = 1.0,
                                  out_dir: Optional[str] = None) -> Dict:
    """Run the decoherence-aware and static routers on the same Poisson trace.

    Returns aggregate metrics (served ratio, utility, mean delivered fidelity
    after hold-time decay) and writes a per-slot CSV to ``out_dir``.
    ``risk_gain`` scales the decoherence-aware router's risk weight so the
    fidelity-versus-utility tradeoff can be swept.
    """
    topo = topology_fn()
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    trace = _poisson_trace(topology_fn, n_slots, mean_rate, seed=seed)

    dec = TimeDependentAnnealer(ec, mc, tau_mem=tau_mem,
                                base_risk_weight=2.0 * risk_gain,
                                n_slots=n_slots, seed=seed)
    static = StaticBaselineRouter(ec, mc, seed=seed)

    all_bundles = [b for slot in trace for reqs in slot.values() for b in reqs]
    all_selected = {"decoherence_aware": [], "static": []}
    rows = []

    for slot_idx, slot in enumerate(trace):
        t0 = time.perf_counter()
        sel_dec = dec.step(slot_idx, slot, n_local_steps=n_local_steps)
        t_dec = time.perf_counter() - t0
        t0 = time.perf_counter()
        sel_static = static.step(slot, n_local_steps=n_local_steps)
        t_static = time.perf_counter() - t0
        all_selected["decoherence_aware"].extend(sel_dec)
        all_selected["static"].extend(sel_static)
        rows.append({
            "slot": slot_idx,
            "arrivals": len(slot),
            "served_dec": len(sel_dec),
            "served_static": len(sel_static),
            "time_dec_s": t_dec,
            "time_static_s": t_static,
        })

    result = {}
    n_requests = len(all_bundles)
    lat_of = {(b["request_id"], b["bundle_id"]): b.get("latency", 0.0) for b in all_bundles}
    for name, sel in all_selected.items():
        # the streaming router accumulates active requests across slots, so a
        # request may appear in several per-slot selections; count it once.
        unique = [(rid, bid) for rid, bid in dict(sel).items()]
        util = _utility_of(all_bundles, unique)
        fid = _fidelity_of(all_bundles, unique)
        def hold_fn(rid, bid):
            # both routers hold stored pairs until the end-to-end link is
            # established; pairs stored via high-latency bundles decay more.
            return lat_of.get((rid, bid), 0.0) * (1.0 + n_slots / 2.0)
        delivered = _delivered_fidelity(all_bundles, unique, hold_fn,
                                        tau_mem=tau_mem, t1=t1_us)
        result[name] = {
            "served": len(unique),
            "n_requests": n_requests,
            "served_ratio": len(unique) / max(n_requests, 1),
            "utility": util,
            "mean_fidelity": fid,
            "mean_delivered_fidelity": delivered,
        }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "time_dependent_slots.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")

    result["trace"] = {"n_slots": n_slots, "mean_rate": mean_rate, "tau_mem": tau_mem}
    result["slots"] = rows
    return result


RISK_GAINS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def run_risk_gain_sweep(topology_fn: Callable, risk_gains=RISK_GAINS,
                        out_dir: Optional[str] = None, **kwargs) -> List[Dict]:
    """Sweep ``risk_gain`` on one Poisson trace; writes ``risk_gain_tradeoff.csv``.

    The static router does not use ``risk_gain``, so its columns are constant
    across rows. ``kwargs`` go to ``run_time_dependent_comparison``.
    """
    rows = []
    for gain in risk_gains:
        res = run_time_dependent_comparison(topology_fn, risk_gain=gain, **kwargs)
        rows.append({
            "risk_gain": gain,
            "served_dec": res["decoherence_aware"]["served"],
            "served_static": res["static"]["served"],
            "utility_dec": round(res["decoherence_aware"]["utility"], 2),
            "utility_static": round(res["static"]["utility"], 2),
            "fid_dec": round(res["decoherence_aware"]["mean_delivered_fidelity"], 4),
            "fid_static": round(res["static"]["mean_delivered_fidelity"], 4),
        })
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "risk_gain_tradeoff.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")
    return rows


if __name__ == "__main__":
    from experiments.instances import generate_chain_topology
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "results", "experiments"))
    topo = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                           memory_capacity=10, raw_fidelity=0.85,
                                           latency=5.0)
    print("Running decoherence-aware vs static routing on a Poisson trace...")
    res = run_time_dependent_comparison(topo, n_slots=20, mean_rate=1.5,
                                        tau_mem=5.0, out_dir=out_dir)
    for name in ["decoherence_aware", "static"]:
        r = res[name]
        print(f"{name:>18}: served {r['served']}/{r['n_requests']} "
              f"(ratio {r['served_ratio']:.3f}), utility {r['utility']:.1f}, "
              f"delivered fidelity {r['mean_delivered_fidelity']:.3f}")
    run_risk_gain_sweep(topo, n_slots=20, mean_rate=1.5, tau_mem=5.0,
                        out_dir=out_dir)
