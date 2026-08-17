"""Conflict-aware, network-level purification scheduling (PSC-inspired).

``purification_scheduler.py`` already scores the ``(k, F(k), P(k), ...)``
trade-off for purification on *one* path in isolation. It has no notion of
*concurrent* requests competing for the same link's Bell-pair budget, which
is exactly the failure mode network-level purification-scheduling research
identifies: deciding purification per link/per request independently wastes
resources on links that are already "good enough" for other requests sharing
them, and starves genuinely contended links.

This module treats purification as a network resource-allocation problem:

* ``route_requests`` -- fixed Dijkstra routing per request (paths are frozen
  before purification is decided, matching the "routing first, purification
  second" pipeline).
* ``purification_need`` -- a continuous relaxation of "how much would
  purifying this one link help this one request close its fidelity gap",
  clipped to ``[0, 1]``.
* ``ConflictAwarePurificationScheduler`` -- a *link conflict metric*
  aggregates purification need across every request still pending on a link,
  scaled by how scarce that link's remaining Bell-pair budget is. The metric
  is then used, each round, as a purification *probability*: the most
  contended links purify first, purification is a network-wide decision
  shared by every request that uses the link (a single purified link can
  satisfy several concurrent requests at once), and the loop repeats against
  the updated resource state until the budget is exhausted or every request
  is resolved.
* ``threshold_baseline`` / ``greedy_baseline`` -- naive, per-request
  purification strategies (purify-when-below-threshold / purify-everywhere)
  that decide in isolation with no cross-request coordination, used as
  comparison points for ``run_psc_comparison``.

Two performance metrics mirror the network-level literature this extends:
``throughput`` (number of requests whose end-to-end fidelity meets their own
threshold) and ``resource_consumption_ratio`` (Bell pairs spent on
purification per successful request -- lower is better).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx

from fidelity.fidelity_model import FidelityModel

Edge = Tuple[str, str]


def _edge_key(u: str, v: str) -> Edge:
    return tuple(sorted((u, v)))


def _path_edges(path: Sequence[str]) -> List[Edge]:
    return [_edge_key(path[i], path[i + 1]) for i in range(len(path) - 1)]


@dataclass
class PSCRequest:
    request_id: str
    path: List[str]
    min_fidelity: float
    weight: float = 1.0


def route_requests(topology: dict,
                    pairs: Sequence[Tuple[str, str, float, float]]) -> List[PSCRequest]:
    """Step 1 of PSC: fixed Dijkstra routing, frozen before purification.

    ``pairs`` is ``(source, destination, weight, min_fidelity)``. Unreachable
    pairs are silently dropped (mirrors ``_shortest_paths`` elsewhere in this
    repo, which returns ``[]`` rather than raising on no path).
    """
    G = nx.Graph()
    G.add_nodes_from(topology["nodes"])
    for (u, v) in topology["edges"]:
        lp = topology["link_params"].get(_edge_key(u, v), {})
        G.add_edge(u, v, weight=lp.get("latency", 1.0))

    out = []
    for i, (src, dst, weight, min_fid) in enumerate(pairs):
        try:
            path = nx.shortest_path(G, src, dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        out.append(PSCRequest(f"req_{i}", path, min_fid, weight))
    return out


def _path_fidelity(topology: dict, path: Sequence[str], purified: Set[Edge]) -> float:
    """End-to-end fidelity swapping raw or one-round-purified link fidelities.

    One-round purification only (matches the repo-wide BBPSSW model): a
    purified link's fidelity is ``purification_bbpssw(raw)`` when
    ``raw > 0.5`` (below that the map is contracting, so purification is
    skipped and the raw fidelity is kept).
    """
    fids = []
    for e in _path_edges(path):
        raw = topology["link_params"][e]["raw_fidelity"]
        if e in purified and raw > 0.5:
            f = FidelityModel.purification_bbpssw(raw)
        else:
            f = raw
        fids.append(f)
    return FidelityModel.end_to_end_fidelity(fids)


def purification_need(topology: dict, request: PSCRequest, link: Edge) -> float:
    """Continuous relaxation of "how much does purifying `link` alone help".

    Fraction of ``request``'s fidelity gap (``min_fidelity - current``) that
    purifying just this one link would close, clipped to ``[0, 1]``. Zero if
    the link isn't on the request's path, the request already meets its
    threshold, or purifying the link doesn't help (raw fidelity <= 0.5).
    """
    path_edges = _path_edges(request.path)
    if link not in path_edges:
        return 0.0
    base = _path_fidelity(topology, request.path, purified=set())
    gap = request.min_fidelity - base
    if gap <= 0:
        return 0.0
    with_link = _path_fidelity(topology, request.path, purified={link})
    gain = with_link - base
    if gain <= 0:
        return 0.0
    return max(0.0, min(1.0, gap / gain))


@dataclass
class _LinkBudget:
    capacity: int
    remaining: int


def _finalize(topology: dict, all_requests: List[PSCRequest],
              purified: Set[Edge], bell_pairs_per_purification: int) -> dict:
    per_request = {}
    for r in all_requests:
        fid = _path_fidelity(topology, r.path, purified)
        per_request[r.request_id] = {
            "success": fid >= r.min_fidelity,
            "final_fidelity": fid,
            "purified_links": [l for l in _path_edges(r.path) if l in purified],
        }
    n_success = sum(1 for v in per_request.values() if v["success"])
    return {
        "per_request": per_request,
        "throughput": n_success,
        "n_requests": len(all_requests),
        # unique links actually purified -- a shared link benefits every
        # request that uses it, so cost is not summed per-request.
        "purification_cost": len(purified) * bell_pairs_per_purification,
        "resource_consumption_ratio": (
            (len(purified) * bell_pairs_per_purification) / n_success
            if n_success else float("inf")
        ),
    }


class ConflictAwarePurificationScheduler:
    """PSC-inspired scheduler: probabilistic, conflict-metric-driven
    purification across concurrent requests sharing a Bell-pair-limited
    network.

    Each round: drop requests that already meet their threshold given the
    links purified so far, compute a link *conflict metric* over the
    remaining candidate links (aggregated purification need across still-
    pending requests, scaled up as the link's remaining budget shrinks), and
    probabilistically purify links -- higher conflict means higher
    probability -- subject to the shared edge Bell-pair budget. Repeats
    until every request resolves, the budget is exhausted, or no round makes
    progress.
    """

    def __init__(self, topology: dict, requests: List[PSCRequest],
                 bell_pairs_per_purification: int = 2,
                 seed: int = 0, max_iterations: int = 20):
        self.topology = topology
        self.requests = requests
        self.bell_pairs_per_purification = bell_pairs_per_purification
        self.rng = random.Random(seed)
        self.max_iterations = max_iterations
        self.budgets: Dict[Edge, _LinkBudget] = {
            e: _LinkBudget(capacity=cap, remaining=cap)
            for e, cap in topology["edge_capacities"].items()
        }

    def _conflict_metric(self, link: Edge, pending: List[PSCRequest]) -> float:
        """``C_link``: aggregated purification need across pending requests,
        amplified by how scarce the link's remaining budget already is.
        """
        needs = [purification_need(self.topology, r, link) for r in pending]
        demand = sum(n for n in needs if n > 0)
        if demand <= 0:
            return 0.0
        budget = self.budgets[link]
        scarcity = 1.0 - (budget.remaining / max(budget.capacity, 1))
        return demand * (1.0 + scarcity)

    def schedule(self) -> dict:
        purified: Set[Edge] = set()
        pending = list(self.requests)

        for _ in range(self.max_iterations):
            pending = [r for r in pending
                       if _path_fidelity(self.topology, r.path, purified) < r.min_fidelity]
            if not pending:
                break

            candidate_links = set()
            for r in pending:
                candidate_links.update(_path_edges(r.path))
            candidate_links -= purified

            conflict = {l: self._conflict_metric(l, pending) for l in candidate_links}
            conflict = {l: c for l, c in conflict.items() if c > 0}
            if not conflict:
                break  # nothing left to purify that helps anyone pending

            # Purify (probabilistically) the single most-contended link, then
            # re-derive pending/conflict from scratch: a link serving several
            # requests at once can resolve all of them in one shot, so
            # re-checking after every purification (rather than working down
            # a stale, pre-computed ranking) is what lets the scheduler stop
            # spending Bell pairs the instant the network no longer needs
            # them.
            link, score = max(conflict.items(), key=lambda kv: kv[1])
            budget = self.budgets[link]
            if budget.remaining < self.bell_pairs_per_purification:
                break  # most-contended link is out of budget -- nothing else helps more
            p = min(1.0, score / len(pending))
            if self.rng.random() < p:
                purified.add(link)
                budget.remaining -= self.bell_pairs_per_purification

        return _finalize(self.topology, self.requests, purified,
                          self.bell_pairs_per_purification)


def threshold_baseline(topology: dict, requests: List[PSCRequest],
                        bell_pairs_per_purification: int = 2) -> dict:
    """Per-link "purify whenever below threshold" (PU baseline).

    Each request is processed independently, in order: it purifies links on
    its own path (highest-need first) from a *shared* capacity pool until it
    meets its own threshold or runs out of budget. Requests do not see each
    other's purification state -- a link already improved by an earlier
    request is not recognised as improved by a later one -- so contended
    links get redundantly re-attempted and capacity is wasted exactly as the
    "link-level purification can't provide an effective solution for
    concurrent requests" critique describes.
    """
    remaining = {e: cap for e, cap in topology["edge_capacities"].items()}
    purified_union: Set[Edge] = set()
    per_request = {}
    for r in requests:
        purified: Set[Edge] = set()
        edges_by_need = sorted(_path_edges(r.path),
                                key=lambda l: -purification_need(topology, r, l))
        fid = _path_fidelity(topology, r.path, purified)
        for link in edges_by_need:
            if fid >= r.min_fidelity:
                break
            if remaining.get(link, 0) < bell_pairs_per_purification:
                continue
            purified.add(link)
            remaining[link] -= bell_pairs_per_purification
            fid = _path_fidelity(topology, r.path, purified)
        purified_union |= purified
        per_request[r.request_id] = {
            "success": fid >= r.min_fidelity,
            "final_fidelity": fid,
            "purified_links": list(purified),
        }
    n_success = sum(1 for v in per_request.values() if v["success"])
    cost = sum(len(v["purified_links"]) for v in per_request.values()) * bell_pairs_per_purification
    return {
        "per_request": per_request,
        "throughput": n_success,
        "n_requests": len(requests),
        "purification_cost": cost,
        "resource_consumption_ratio": cost / n_success if n_success else float("inf"),
    }


def greedy_baseline(topology: dict, requests: List[PSCRequest],
                     bell_pairs_per_purification: int = 2) -> dict:
    """Purify almost everywhere: every link on every path, budget
    permitting, processed request-by-request with no regard for whether the
    request actually needs it -- maximises fidelity margin at the cost of
    resources (the "Greedy" baseline in the PSC literature).
    """
    remaining = {e: cap for e, cap in topology["edge_capacities"].items()}
    per_request = {}
    for r in requests:
        purified: Set[Edge] = set()
        for link in _path_edges(r.path):
            if remaining.get(link, 0) >= bell_pairs_per_purification:
                purified.add(link)
                remaining[link] -= bell_pairs_per_purification
        fid = _path_fidelity(topology, r.path, purified)
        per_request[r.request_id] = {
            "success": fid >= r.min_fidelity,
            "final_fidelity": fid,
            "purified_links": list(purified),
        }
    n_success = sum(1 for v in per_request.values() if v["success"])
    cost = sum(len(v["purified_links"]) for v in per_request.values()) * bell_pairs_per_purification
    return {
        "per_request": per_request,
        "throughput": n_success,
        "n_requests": len(requests),
        "purification_cost": cost,
        "resource_consumption_ratio": cost / n_success if n_success else float("inf"),
    }


def run_psc_comparison(topology: dict,
                        pairs: Sequence[Tuple[str, str, float, float]],
                        bell_pairs_per_purification: int = 2,
                        seed: int = 0) -> Dict[str, dict]:
    """PSC vs. the naive threshold/greedy baselines on identical routing.

    All three strategies act on the same frozen paths (``route_requests`` is
    run once), so differences in throughput and resource consumption come
    purely from the purification-scheduling strategy.
    """
    requests = route_requests(topology, pairs)
    return {
        "threshold": threshold_baseline(topology, requests, bell_pairs_per_purification),
        "greedy": greedy_baseline(topology, requests, bell_pairs_per_purification),
        "psc": ConflictAwarePurificationScheduler(
            topology, requests, bell_pairs_per_purification, seed=seed
        ).schedule(),
    }
