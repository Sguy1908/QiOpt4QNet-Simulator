"""Q-learning adaptive router (paper Future Work item iii).

A reinforcement-learning baseline for the streaming/adaptive routing setting:
requests arrive over time and a router must decide each request's bundle (or
reject it) given the current network occupancy.  We implement a tabular
Q-learning agent whose state is the quantized global resource pressure
(edge-occupancy and memory-occupancy bins) and whose action is the bundle
choice for the arriving request.

The policy is trained on episodes built from a synthetic Poisson arrival trace
and evaluated against the streaming annealer on the same trace.  This answers
"when should one prefer RL over Hamiltonian-based solvers for online routing"
by measuring the utility gap and per-decision latency.
"""

import os
import random
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _edge_load_of(selections, edge_of):
    load = defaultdict(int)
    for rid, bid in selections.items():
        if bid is None:
            continue
        for e, d in edge_of.get((rid, bid), {}).items():
            load[e] += d
    return load


def _mem_load_of(selections, mem_of):
    load = defaultdict(int)
    for rid, bid in selections.items():
        if bid is None:
            continue
        for n, d in mem_of.get((rid, bid), {}).items():
            load[n] += d
    return load


class QLearningRouter:
    """Tabular Q-learning router over quantized resource-pressure states."""

    def __init__(self, edge_capacities, memory_capacities, n_bins: int = 3,
                 lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                 seed: Optional[int] = None):
        self.edge_capacities = {tuple(sorted(k)): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.q: Dict[Tuple, float] = defaultdict(float)
        self.edge_of: Dict[Tuple[str, str], dict] = {}
        self.mem_of: Dict[Tuple[str, str], dict] = {}
        self.util_of: Dict[Tuple[str, str], float] = {}
        self.training_episodes = 0

    # -- feature plumbing ------------------------------------------------
    def _register_bundles(self, request_id: str, bundles: List[dict]):
        for b in bundles:
            key = (request_id, b["bundle_id"])
            self.util_of[key] = b["utility"]
            self.edge_of[key] = {tuple(sorted(e)): d for e, d in b["edge_demands"].items()}
            self.mem_of[key] = dict(b["memory_demands"])

    def _state(self, selections: dict) -> Tuple[int, int]:
        """Quantize global resource pressure into (edge_bin, memory_bin)."""
        edge_load = _edge_load_of(selections, self.edge_of)
        mem_load = _mem_load_of(selections, self.mem_of)
        ratios = [load / self.edge_capacities.get(e, 1)
                  for e, load in edge_load.items() if self.edge_capacities.get(e, 1) > 0]
        mem_ratios = [load / self.memory_capacities.get(n, 1)
                      for n, load in mem_load.items() if self.memory_capacities.get(n, 1) > 0]
        avg_edge = (sum(ratios) / len(ratios)) if ratios else 0.0
        max_mem = max(mem_ratios) if mem_ratios else 0.0
        b_e = min(self.n_bins - 1, int(avg_edge * self.n_bins))
        b_m = min(self.n_bins - 1, int(max_mem * self.n_bins))
        return (b_e, b_m)

    def _fits(self, rid, bid, selections) -> bool:
        trial = dict(selections)
        trial[rid] = bid
        edge_load = _edge_load_of(trial, self.edge_of)
        for e, load in edge_load.items():
            if load > self.edge_capacities.get(e, 0):
                return False
        mem_load = _mem_load_of(trial, self.mem_of)
        for n, load in mem_load.items():
            if load > self.memory_capacities.get(n, 0):
                return False
        return True

    def choose_action(self, rid: str, bundles: List[dict], selections: dict):
        """Epsilon-greedy choice over feasible bundles (or reject)."""
        self._register_bundles(rid, bundles)
        state = self._state(selections)
        feasible = [b["bundle_id"] for b in bundles
                    if self._fits(rid, b["bundle_id"], selections)]
        actions = feasible + [None]
        if not feasible:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        best = max(actions, key=lambda a: self.q.get((state, a), 0.0))
        return best

    def _update(self, state, action, reward, next_state):
        best_next = max((self.q[(next_state, a)] for a in self._all_actions(next_state)),
                        default=0.0)
        self.q[(state, action)] += self.lr * (
            reward + self.gamma * best_next - self.q[(state, action)])

    def _all_actions(self, state):
        # actions seen in this state so far (bundle ids or None)
        return {a for (s, a) in self.q if s == state}

    def train_episode(self, trace: List[Dict[str, List[dict]]],
                      full_capacity_penalty: float = 50.0) -> Tuple[float, float]:
        """One RL episode: process the arrival trace, return (served, utility)."""
        selections: Dict[str, Optional[str]] = {}
        total_reward = 0.0
        served = 0
        for slot in trace:
            for rid, bundles in slot.items():
                self._register_bundles(rid, bundles)
                state = self._state(selections)
                action = self.choose_action(rid, bundles, selections)
                reward = self.util_of.get((rid, action), 0.0) if action is not None else 0.0
                if action is not None and not self._fits(rid, action, selections):
                    reward = -full_capacity_penalty
                    action = None
                selections[rid] = action
                if action is not None:
                    served += 1
                next_state = self._state(selections)
                self._update(state, action, reward, next_state)
                total_reward += reward
        self.training_episodes += 1
        return served, total_reward

    def evaluate(self, trace: List[Dict[str, List[dict]]]) -> Tuple[int, float, int]:
        """Pure greedy replay of the trace (no exploration, no learning)."""
        selections: Dict[str, Optional[str]] = {}
        total_utility = 0.0
        served = 0
        for slot in trace:
            for rid, bundles in slot.items():
                self._register_bundles(rid, bundles)
                feasible = [b["bundle_id"] for b in bundles
                            if self._fits(rid, b["bundle_id"], selections)]
                best = None
                if feasible:
                    state = self._state(selections)
                    best = max(feasible, key=lambda a: self.q.get((state, a), 0.0))
                selections[rid] = best
                if best is not None:
                    served += 1
                    total_utility += self.util_of.get((rid, best), 0.0)
        return served, total_utility, self.training_episodes


class LinearQRouter:
    """Linear function-approximation Q-learning router (paper Future Work
    item ii: "function-approximation RL that generalizes across
    topologies").

    Unlike ``QLearningRouter``'s tabular ``(quantized_state, bundle_id)``
    lookup -- which cannot transfer across instances because ``bundle_id``
    strings and the network's node/edge names are instance-specific --
    ``LinearQRouter`` scores every (state, candidate) pair through a fixed
    set of DIMENSIONLESS, topology-invariant features (utility density,
    post-admission edge/memory pressure ratios, hop count, demand fraction
    of mean capacity, and continuous global resource pressure). Because the
    feature vector never references a raw node id, edge id, or bundle id,
    weights trained on one topology family are meaningful -- and can be
    evaluated zero-shot -- on a structurally different one.

    ``Q(s,a) = w . phi(s,a)`` by default (a linear approximator, always
    available). Set ``use_torch_mlp=True`` for a small 2-layer MLP instead
    (requires the optional ``torch`` dependency; raises a clear
    ``ImportError`` if unavailable).
    """

    N_FEATURES = 10

    def __init__(self, edge_capacities, memory_capacities, lr: float = 0.01,
                 gamma: float = 0.9, epsilon: float = 0.1, l2: float = 1e-4,
                 seed: Optional[int] = None, use_torch_mlp: bool = False,
                 hidden_dim: int = 16):
        self.edge_capacities = {tuple(sorted(k)): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.l2 = l2
        self._rng = random.Random(seed)
        self.edge_of: Dict[Tuple[str, str], dict] = {}
        self.mem_of: Dict[Tuple[str, str], dict] = {}
        self.util_of: Dict[Tuple[str, str], float] = {}
        self.training_episodes = 0
        self._mean_cap = (sum(self.edge_capacities.values()) / len(self.edge_capacities)
                          if self.edge_capacities else 1.0)

        self.use_torch_mlp = use_torch_mlp
        if use_torch_mlp:
            try:
                import torch
                import torch.nn as nn
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "use_torch_mlp=True requires the optional 'torch' "
                    "dependency (pip install -e '.[baselines]')") from e
            self._torch = torch
            if seed is not None:
                torch.manual_seed(seed)
            self._net = nn.Sequential(
                nn.Linear(self.N_FEATURES, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1))
            self._optimizer = torch.optim.SGD(self._net.parameters(), lr=lr,
                                              weight_decay=l2)
        else:
            self.w = np.zeros(self.N_FEATURES)

    # -- feature plumbing ------------------------------------------------
    def _register_bundles(self, request_id: str, bundles: List[dict]):
        for b in bundles:
            key = (request_id, b["bundle_id"])
            self.util_of[key] = b["utility"]
            self.edge_of[key] = {tuple(sorted(e)): d for e, d in b["edge_demands"].items()}
            self.mem_of[key] = dict(b["memory_demands"])

    def _global_pressure(self, selections: dict) -> Tuple[float, float]:
        edge_load = _edge_load_of(selections, self.edge_of)
        mem_load = _mem_load_of(selections, self.mem_of)
        ratios = [load / self.edge_capacities.get(e, 1)
                  for e, load in edge_load.items() if self.edge_capacities.get(e, 1) > 0]
        mem_ratios = [load / self.memory_capacities.get(n, 1)
                      for n, load in mem_load.items() if self.memory_capacities.get(n, 1) > 0]
        avg_edge = (sum(ratios) / len(ratios)) if ratios else 0.0
        max_mem = max(mem_ratios) if mem_ratios else 0.0
        return avg_edge, max_mem

    def _features(self, rid: str, bid: Optional[str], selections: dict) -> np.ndarray:
        """Fixed-length, dimensionless feature vector for (state, action).
        Carries no raw node/edge/bundle identifiers -- see class docstring."""
        avg_edge, max_mem = self._global_pressure(selections)
        f = np.zeros(self.N_FEATURES)
        f[0] = 1.0  # bias
        f[8] = avg_edge
        f[9] = max_mem
        if bid is None:
            return f  # reject sentinel: bias + global pressure only

        key = (rid, bid)
        edge_of = self.edge_of.get(key, {})
        mem_of = self.mem_of.get(key, {})
        util = self.util_of.get(key, 0.0)
        edge_load = _edge_load_of(selections, self.edge_of)
        mem_load = _mem_load_of(selections, self.mem_of)

        total_demand = sum(edge_of.values()) + sum(mem_of.values())
        # Utility density is unbounded as demand -> 0, which would otherwise
        # let a handful of tiny-demand bundles dominate the linear model's
        # gradient and destabilize training; clip to a fixed range (a
        # bounded, dimensionless feature is the point of this design).
        f[1] = float(np.clip(util / (total_demand + 1.0), -10.0, 10.0))

        edge_pressures = [(edge_load.get(e, 0) + d) / self.edge_capacities.get(e, 1)
                          for e, d in edge_of.items() if self.edge_capacities.get(e, 1) > 0]
        f[2] = min(sum(edge_pressures) / len(edge_pressures), 5.0) if edge_pressures else 0.0
        f[3] = min(max(edge_pressures), 5.0) if edge_pressures else 0.0

        mem_pressures = [(mem_load.get(n, 0) + d) / self.memory_capacities.get(n, 1)
                         for n, d in mem_of.items() if self.memory_capacities.get(n, 1) > 0]
        f[4] = min(sum(mem_pressures) / len(mem_pressures), 5.0) if mem_pressures else 0.0
        f[5] = min(max(mem_pressures), 5.0) if mem_pressures else 0.0

        f[6] = min(float(len(edge_of)), 20.0)  # hop count
        f[7] = min(total_demand / (self._mean_cap + 1e-9), 5.0)  # demand fraction of mean capacity
        return f

    def q_value(self, features: np.ndarray) -> float:
        if self.use_torch_mlp:
            with self._torch.no_grad():
                x = self._torch.as_tensor(features, dtype=self._torch.float32)
                return float(self._net(x).item())
        return float(self.w @ features)

    def _fits(self, rid, bid, selections) -> bool:
        trial = dict(selections)
        trial[rid] = bid
        edge_load = _edge_load_of(trial, self.edge_of)
        for e, load in edge_load.items():
            if load > self.edge_capacities.get(e, 0):
                return False
        mem_load = _mem_load_of(trial, self.mem_of)
        for n, load in mem_load.items():
            if load > self.memory_capacities.get(n, 0):
                return False
        return True

    def _best(self, rid: str, bundles: List[dict], selections: dict):
        """Returns (best_action, best_features, best_q) over feasible
        actions (bundle ids) plus the reject action."""
        self._register_bundles(rid, bundles)
        feasible = [b["bundle_id"] for b in bundles
                    if self._fits(rid, b["bundle_id"], selections)]
        candidates = feasible + [None]
        best_action, best_features, best_q = None, self._features(rid, None, selections), -float("inf")
        for a in candidates:
            feats = self._features(rid, a, selections)
            q = self.q_value(feats)
            if q > best_q:
                best_action, best_features, best_q = a, feats, q
        return best_action, best_features, best_q

    def choose_action(self, rid: str, bundles: List[dict], selections: dict):
        """Epsilon-greedy choice over feasible bundles (or reject); same
        contract as ``QLearningRouter.choose_action``."""
        self._register_bundles(rid, bundles)
        feasible = [b["bundle_id"] for b in bundles
                    if self._fits(rid, b["bundle_id"], selections)]
        if not feasible:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(feasible + [None])
        best_action, _, _ = self._best(rid, bundles, selections)
        return best_action

    def _sgd_update(self, features: np.ndarray, td_target: float):
        if self.use_torch_mlp:
            x = self._torch.as_tensor(features, dtype=self._torch.float32)
            pred = self._net(x)
            loss = (pred - self._torch.as_tensor([td_target], dtype=self._torch.float32)) ** 2
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        else:
            pred = float(self.w @ features)
            td_error = td_target - pred
            self.w += self.lr * td_error * features - self.lr * self.l2 * self.w
            # Safety net against linear semi-gradient TD divergence
            # (a known failure mode with bootstrapping): clip weight
            # magnitude rather than let a bad step compound unboundedly.
            np.clip(self.w, -50.0, 50.0, out=self.w)

    def train_episode(self, trace: List[Dict[str, List[dict]]],
                      full_capacity_penalty: float = 50.0) -> Tuple[float, float]:
        """One RL episode: process the arrival trace with a semi-gradient
        TD(0) update per decision, return (served, total_reward)."""
        selections: Dict[str, Optional[str]] = {}
        total_reward = 0.0
        served = 0
        seq = [(rid, bundles) for slot in trace for rid, bundles in slot.items()]
        for i, (rid, bundles) in enumerate(seq):
            self._register_bundles(rid, bundles)
            action = self.choose_action(rid, bundles, selections)
            reward = self.util_of.get((rid, action), 0.0) if action is not None else 0.0
            if action is not None and not self._fits(rid, action, selections):
                reward = -full_capacity_penalty
                action = None
            action_features = self._features(rid, action, selections)
            selections[rid] = action
            if action is not None:
                served += 1

            if i + 1 < len(seq):
                next_rid, next_bundles = seq[i + 1]
                _, _, next_best_q = self._best(next_rid, next_bundles, selections)
            else:
                next_best_q = 0.0

            td_target = reward + self.gamma * next_best_q
            self._sgd_update(action_features, td_target)
            total_reward += reward
        self.training_episodes += 1
        return served, total_reward

    def evaluate(self, trace: List[Dict[str, List[dict]]]) -> Tuple[int, float, int]:
        """Pure greedy replay of the trace (no exploration, no learning) --
        same contract as ``QLearningRouter.evaluate``."""
        selections: Dict[str, Optional[str]] = {}
        total_utility = 0.0
        served = 0
        for slot in trace:
            for rid, bundles in slot.items():
                self._register_bundles(rid, bundles)
                best, _, _ = self._best(rid, bundles, selections)
                selections[rid] = best
                if best is not None:
                    served += 1
                    total_utility += self.util_of.get((rid, best), 0.0)
        return served, total_utility, self.training_episodes


def run_topology_generalization_study(train_topology_fns: List[Callable],
                                      eval_topology_fns: Dict[str, Callable],
                                      n_slots: int = 20, mean_rate: float = 1.5,
                                      episodes: int = 40, seed: int = 42,
                                      use_torch_mlp: bool = False,
                                      out_dir: Optional[str] = None) -> Dict:
    """Trains one ``LinearQRouter`` on traces interleaved across
    ``train_topology_fns`` (one or more topology families/sizes), then
    zero-shot evaluates the SAME frozen weights on each topology in
    ``eval_topology_fns`` (which may be unseen families) -- this is the
    actual "generalizes across topologies" claim. Compares against (a) a
    tabular ``QLearningRouter`` trained in-distribution on each eval
    topology (an upper-bound control, since tabular state/action cannot
    transfer at all) and (b) the streaming annealer baseline.
    """
    from optimization.streaming_annealer import StreamingAnnealer
    from optimization.time_dependent_optimizer import _poisson_trace
    import csv
    import time as _time

    rng = random.Random(seed)

    # Train the linear router on interleaved traces from the training
    # topology family/families (one router, many topologies seen).
    linear = LinearQRouter(*_merged_capacities(train_topology_fns), seed=seed,
                           use_torch_mlp=use_torch_mlp)
    for ep in range(episodes):
        topo_fn = train_topology_fns[ep % len(train_topology_fns)]
        trace = _poisson_trace(topo_fn, n_slots, mean_rate,
                               seed=rng.randint(0, 2 ** 31 - 1))
        # re-point capacities at this episode's topology (the router's
        # feature extraction is topology-invariant, but feasibility checks
        # need the current instance's actual capacities)
        topo = topo_fn()
        linear.edge_capacities = {tuple(sorted(k)): v for k, v in topo["edge_capacities"].items()}
        linear.memory_capacities = topo["memory_capacities"]
        linear.train_episode(trace)

    rows = []
    for eval_name, eval_topo_fn in eval_topology_fns.items():
        topo = eval_topo_fn()
        ec, mc = topo["edge_capacities"], topo["memory_capacities"]
        test_trace = _poisson_trace(eval_topo_fn, n_slots, mean_rate,
                                    seed=rng.randint(0, 2 ** 31 - 1))
        n_requests = sum(len(slot) for slot in test_trace)

        # zero-shot: reuse the already-trained linear router's weights,
        # just re-point capacities at the eval topology.
        linear.edge_capacities = {tuple(sorted(k)): v for k, v in ec.items()}
        linear.memory_capacities = mc
        t0 = _time.perf_counter()
        lin_served, lin_utility, _ = linear.evaluate(test_trace)
        lin_time = _time.perf_counter() - t0

        # in-distribution control: a fresh tabular router trained and
        # evaluated on this exact eval topology.
        tabular = QLearningRouter(ec, mc, seed=seed)
        for _ in range(episodes):
            tab_train_trace = _poisson_trace(eval_topo_fn, n_slots, mean_rate,
                                             seed=rng.randint(0, 2 ** 31 - 1))
            tabular.train_episode(tab_train_trace)
        tab_served, tab_utility, _ = tabular.evaluate(test_trace)

        # streaming-annealer baseline
        sa = StreamingAnnealer(ec, mc, seed=seed)
        bundles_by_rid = {}
        for slot in test_trace:
            for rid, bundles in slot.items():
                bundles_by_rid[rid] = bundles
                sa.add_request(rid, bundles)
                sa.local_sweep(n_steps=50, temperature=2.0)
        sa_sel = sa.get_selected()

        rows.append({
            "eval_topology": eval_name,
            "n_requests": n_requests,
            "linear_zero_shot_served_ratio": lin_served / max(n_requests, 1),
            "linear_zero_shot_utility": lin_utility,
            "tabular_in_distribution_served_ratio": tab_served / max(n_requests, 1),
            "tabular_in_distribution_utility": tab_utility,
            "streaming_annealer_served_ratio": len(sa_sel) / max(n_requests, 1),
            "streaming_annealer_utility": _utility_sum(bundles_by_rid, sa_sel),
        })

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "rl_topology_generalization.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")

    return {"rows": rows}


def _merged_capacities(topology_fns: List[Callable]):
    """Capacities from the first training topology (LinearQRouter's
    edge_capacities/memory_capacities are re-pointed per-episode during
    training in run_topology_generalization_study; this just seeds the
    constructor with something valid)."""
    topo = topology_fns[0]()
    return topo["edge_capacities"], topo["memory_capacities"]


def _utility_sum(bundles_by_rid, selected):
    util = 0.0
    for rid, bid in selected:
        if bid is None:
            continue
        for b in bundles_by_rid.get(rid, []):
            if b["bundle_id"] == bid:
                util += b["utility"]
                break
    return util


def run_rl_comparison(topology_fn: Callable, n_slots: int = 20,
                      mean_rate: float = 1.5, episodes: int = 30,
                      seed: int = 42, out_dir: Optional[str] = None) -> Dict:
    """Train a tabular Q-learning router and compare it against the streaming
    annealer on a held-out Poisson trace.

    Returns aggregate served ratio / utility / per-decision latency for both
    routers and writes per-episode learning progress to ``out_dir``."""
    from optimization.streaming_annealer import StreamingAnnealer
    from optimization.time_dependent_optimizer import _poisson_trace
    import csv, time

    topo = topology_fn()
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    rng = random.Random(seed)
    all_traces = []
    for _ in range(episodes + 1):
        all_traces.append(_poisson_trace(topology_fn, n_slots, mean_rate, seed=rng.randint(0, 2 ** 31 - 1)))
    train_traces = all_traces[:episodes]
    test_trace = all_traces[-1]

    # Train the Q-learning router
    router = QLearningRouter(ec, mc, seed=seed)
    progress = []
    for ep in range(episodes):
        served, reward = router.train_episode(train_traces[ep])
        progress.append({"episode": ep, "served": served, "reward": reward})

    # Evaluate RL greedily
    t0 = time.perf_counter()
    rl_served, rl_utility, _ = router.evaluate(test_trace)
    rl_time = time.perf_counter() - t0

    # Streaming annealer baseline on the same test trace
    sa = StreamingAnnealer(ec, mc, seed=seed)
    bundles_by_rid = {}
    t0 = time.perf_counter()
    for slot in test_trace:
        for rid, bundles in slot.items():
            bundles_by_rid[rid] = bundles
            sa.add_request(rid, bundles)
            sa.local_sweep(n_steps=50, temperature=2.0)
    sa_time = time.perf_counter() - t0
    sa_sel = sa.get_selected()

    n_requests = sum(len(slot) for slot in test_trace)

    result = {
        "n_requests": n_requests,
        "qlearning": {
            "served": rl_served,
            "served_ratio": rl_served / max(n_requests, 1),
            "utility": rl_utility,
            "wall_time_s": rl_time,
        },
        "streaming_annealer": {
            "served": len(sa_sel),
            "served_ratio": len(sa_sel) / max(n_requests, 1),
            "utility": _utility_sum(bundles_by_rid, sa_sel),
            "wall_time_s": sa_time,
        },
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "qlearning_comparison.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(progress[0].keys()))
            w.writeheader()
            w.writerows(progress)
        print(f"Wrote {path}")
    result["progress"] = progress
    return result


if __name__ == "__main__":
    from experiments.instances import generate_chain_topology
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "results", "experiments"))
    topo = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
    print("Training Q-learning router and comparing against streaming annealer...")
    res = run_rl_comparison(topo, n_slots=10, mean_rate=1.0, episodes=10, out_dir=out_dir)
    for name in ["qlearning", "streaming_annealer"]:
        r = res[name]
        print(f"{name:>18}: served {r['served']}/{res['n_requests']} "
              f"({r['served_ratio']:.3f}), utility {r['utility']:.1f}, "
              f"time {r['wall_time_s']:.4f}s")
