"""GraphSAGE-style candidate ranker feeding the QUBO (Extension 8).

The ML ranking baseline (``baselines.ranking_model``) scores bundles from
hand-crafted per-bundle features.  This module replaces that encoder with a
real *graph* model: a one-layer GraphSAGE-style message-passing network that
produces node embeddings from the network topology, pools them along each
candidate path, and predicts a candidate score.  The scores are then used to
select the top-k bundles that enter the QUBO:

    quantum network -> node/edge embeddings -> path pooling -> score -> top-k -> QUBO

The implementation is dependency-light (numpy only): node embeddings are
h_v = ReLU(W @ mean_{u in N(v)} [x_u || e_uv]), the path embedding is the
mean of its nodes' embeddings concatenated with the mean edge features, and
a small MLP head regresses the target (bundle utility, or CP-SAT selection
labels).  Gradients are computed by hand, so training works with plain SGD.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _undirected(edge):
    return tuple(sorted(edge))


class GraphFeatureBuilder:
    """Build node and edge feature matrices from a topology dict."""

    def __init__(self, topology: dict):
        self.topology = topology
        self.nodes = list(topology["nodes"])
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.edges = list(topology["edges"])
        self.link_params = topology.get("link_params", {})

        self.node_feats = self._node_features()
        self.edge_feats = self._edge_features()

    def _memory_capacity(self, node: str) -> int:
        return int(self.topology.get("memory_capacities", {}).get(node, 10))

    def _node_features(self) -> np.ndarray:
        n = len(self.nodes)
        X = np.zeros((n, 4))
        degree = defaultdict(int)
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        for i, node in enumerate(self.nodes):
            X[i, 0] = degree.get(node, 0)
            X[i, 1] = self._memory_capacity(node)
            X[i, 2] = self._mean_incident(node, "raw_fidelity", default=0.85)
            X[i, 3] = self._mean_incident(node, "latency", default=5.0)
        return X

    def _edge_features(self) -> Dict[Tuple[str, str], np.ndarray]:
        feats: Dict[Tuple[str, str], np.ndarray] = {}
        for u, v in self.edges:
            lp = (self.link_params.get(_undirected((u, v)))
                  or self.link_params.get((v, u)) or {})
            feats[_undirected((u, v))] = np.array([
                lp.get("raw_fidelity", 0.85),
                lp.get("latency", 5.0),
                lp.get("generation_probability", 1.0),
                lp.get("capacity", 6.0),
            ], dtype=float)
        return feats

    def _mean_incident(self, node: str, key: str, default: float) -> float:
        vals = []
        for u, v in self.edges:
            if u == node or v == node:
                lp = (self.link_params.get(_undirected((u, v)))
                      or self.link_params.get((v, u)) or {})
                vals.append(lp.get(key, default))
        return float(np.mean(vals)) if vals else default

    def neighbors(self, node: str) -> List[Tuple[str, np.ndarray]]:
        """(neighbor, edge-feature-vector) pairs for a node."""
        out = []
        for u, v in self.edges:
            if u == node:
                out.append((v, self.edge_feats[_undirected((u, v))]))
            elif v == node:
                out.append((u, self.edge_feats[_undirected((u, v))]))
        return out

    def path_node_indices(self, path: List[str]) -> List[int]:
        return [self.index[n] for n in path if n in self.index]

    def path_edge_mean(self, path: List[str]) -> np.ndarray:
        feats = []
        for u, v in zip(path, path[1:]):
            key = _undirected((u, v))
            if key in self.edge_feats:
                feats.append(self.edge_feats[key])
        if not feats:
            return np.zeros(self.edge_feat_dim())
        return np.mean(feats, axis=0)

    def edge_feat_dim(self) -> int:
        return 4


N_BUNDLE_FEATS = 2  # purification_rounds, fidelity


def _bundle_feats(bundle: Optional[dict]) -> np.ndarray:
    """Per-bundle scalar features the path/graph structure cannot express.

    Two bundles can share an identical path (differing only in purification
    rounds) or traverse topologically symmetric paths (e.g. on a ring); the
    structural node/edge features are then identical and a path-only score
    cannot distinguish them (see the cross-topology generalization pilot).
    Appending these breaks that tie without changing the model for callers
    that only have a bare path (``bundle=None`` -> zeros, unchanged score).
    """
    if bundle is None:
        return np.zeros(N_BUNDLE_FEATS)
    return np.array([
        float(bundle.get("purification_rounds", 0)),
        float(bundle.get("fidelity", 0.0)),
    ])


class GraphSAGERanker:
    """Trainable GraphSAGE-style ranker with a numpy MLP head."""

    def __init__(self, topology: dict, hidden: int = 16, seed: int = 42,
                 lr: float = 0.01):
        self.graph = GraphFeatureBuilder(topology)
        self.topology = topology
        self.hidden = hidden
        self.seed = seed
        self.lr = lr
        rng = np.random.default_rng(seed)

        d0 = self.graph.node_feats.shape[1]
        d1 = self.graph.edge_feat_dim()
        self.W = rng.standard_normal((d0 + d1, hidden)) * 0.1
        self.W1 = rng.standard_normal((hidden + d1 + N_BUNDLE_FEATS, 16)) * 0.05
        self.b1 = np.zeros(16)
        self.W2 = rng.standard_normal((16, 1)) * 0.1
        self.b2 = np.zeros(1)
        self.node_mean = None
        self.node_std = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _node_embeddings(self) -> np.ndarray:
        X = self.graph.node_feats
        V = X.shape[0]
        emb = np.zeros((V, self.hidden))
        for i, node in enumerate(self.graph.nodes):
            msgs = [np.concatenate([X[self.graph.index[u]], e])
                    for u, e in self.graph.neighbors(node)]
            if not msgs:
                continue
            A = np.mean(msgs, axis=0)
            emb[i] = np.maximum(0.0, self.W.T @ A)
        return emb

    def _path_embedding(self, path: List[str],
                        bundle: Optional[dict] = None) -> np.ndarray:
        emb = self._node_embeddings()
        idx = self.graph.path_node_indices(path)
        if not idx:
            return np.zeros(self.hidden + self.graph.edge_feat_dim()
                            + N_BUNDLE_FEATS)
        node_mean = emb[idx].mean(axis=0)
        return np.concatenate([node_mean, self.graph.path_edge_mean(path),
                               _bundle_feats(bundle)])

    def _score(self, z: np.ndarray) -> float:
        a = z @ self.W1 + self.b1
        r = np.maximum(0.0, a)
        return float((r @ self.W2 + self.b2).item())

    def predict_bundles(self, bundles: List[dict]) -> Dict[Tuple[str, str], float]:
        out = {}
        for b in bundles:
            path = b.get("path", [])
            z = self._path_embedding(path, bundle=b)
            out[(b["request_id"], b["bundle_id"])] = self._score(z)
        return out

    def score(self, path: List[str], bundle: Optional[dict] = None) -> float:
        return self._score(self._path_embedding(path, bundle=bundle))

    # ------------------------------------------------------------------
    # training (manual backprop, full-batch SGD)
    # ------------------------------------------------------------------
    def fit(self, bundles: List[dict], targets: Dict[Tuple[str, str], float],
            epochs: int = 200, log_every: int = 50) -> Dict:
        # Regression targets (bundle utilities) can be O(1-100) while the
        # raw node/edge features are O(1-10); fitting that gap by growing
        # W1/W2 during plain full-batch SGD reliably drives every hidden
        # unit's pre-activation negative ("dead ReLU"), after which the
        # score collapses to the bias term regardless of input. Only the
        # *ranking* (relative order) of scores is ever used downstream, so
        # training on a normalized target is equivalent for every caller
        # and removes the incentive to blow up the weights.
        raw_targets = {k: v for k, v in targets.items()}
        scale = max((abs(v) for v in raw_targets.values()), default=1.0) or 1.0
        targets = {k: v / scale for k, v in raw_targets.items()}

        samples = [(b, targets.get((b["request_id"], b["bundle_id"]), 0.0))
                   for b in bundles
                   if (b["request_id"], b["bundle_id"]) in targets]
        if not samples:
            raise ValueError("no training samples with targets")

        loss_hist = []
        for epoch in range(epochs):
            loss, gW, gW1, gb1, gW2, gb2 = self._train_step(samples)
            loss_hist.append(loss)
            self.W -= self.lr * gW
            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * gb1
            self.W2 -= self.lr * gW2
            self.b2 -= self.lr * gb2
        return {"final_loss": loss_hist[-1] if loss_hist else 0.0,
                "loss_history": loss_hist}

    def _train_step(self, samples):
        """One full-batch gradient step (returns loss and grads)."""
        X = self.graph.node_feats
        V = X.shape[0]

        # cache messages and activations for every node
        msgs = {}
        A_act = {}
        S_act = {}
        H = np.zeros((V, self.hidden))
        for i, node in enumerate(self.graph.nodes):
            mlist = [np.concatenate([X[self.graph.index[u]], e])
                     for u, e in self.graph.neighbors(node)]
            if not mlist:
                msgs[i] = None
                continue
            M = np.array(mlist)
            msgs[i] = M
            A = M.mean(axis=0)
            A_act[i] = A
            S = self.W.T @ A
            S_act[i] = S
            H[i] = np.maximum(0.0, S)

        N = len(samples)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW = np.zeros_like(self.W)

        total_loss = 0.0
        for b, y in samples:
            idx = self.graph.path_node_indices(b.get("path", []))
            if not idx:
                continue
            edge_mean = self.graph.path_edge_mean(b.get("path", []))
            node_mean = H[idx].mean(axis=0)
            z = np.concatenate([node_mean, edge_mean, _bundle_feats(b)])
            a = z @ self.W1 + self.b1
            r = np.maximum(0.0, a)
            out = float((r @ self.W2 + self.b2).item())

            err = (out - y) / N
            total_loss += 0.5 * (out - y) ** 2 / N

            dout = np.array([err])
            dW2 += np.outer(r, dout)
            db2 += dout.reshape(-1)
            dr = dout.reshape(-1) @ self.W2.T
            da = dr * (a > 0)
            dW1 += np.outer(z, da)
            db1 += da
            dz = da @ self.W1.T

            d_node_mean = dz[:self.hidden]
            per_node_grad = d_node_mean / max(len(idx), 1)
            for v in idx:
                if msgs[v] is None:
                    continue
                dS = per_node_grad * (S_act[v] > 0)
                dW += np.outer(A_act[v], dS)
                for row in msgs[v]:
                    dW += np.outer(row, dS) / len(msgs[v])

        return total_loss, dW, dW1, db1, dW2, db2


def gnn_guided_topk(topology: dict, bundles: List[dict], k: int = 8,
                    train_fraction: float = 0.7, epochs: int = 100,
                    seed: int = 42) -> Tuple[GraphSAGERanker, List[dict], float]:
    """Train the GNN on a subset of bundles (utility targets) and return the
    top-k per request scored candidates (the QUBO input set)."""
    rng = random.Random(seed)
    by_req: Dict[str, List[dict]] = defaultdict(list)
    for b in bundles:
        by_req[b["request_id"]].append(b)

    all_keys = [(b["request_id"], b["bundle_id"]) for b in bundles]
    rng.shuffle(all_keys)
    n_train = max(1, int(len(all_keys) * train_fraction))
    train_keys = set(all_keys[:n_train])

    train_bundles = [b for b in bundles if (b["request_id"], b["bundle_id"]) in train_keys]
    targets = {(b["request_id"], b["bundle_id"]): b.get("utility", 0.0)
               for b in train_bundles}

    ranker = GraphSAGERanker(topology, seed=seed)
    report = ranker.fit(train_bundles, targets, epochs=epochs)

    scores = ranker.predict_bundles(bundles)
    reduced = []
    for rid, bs in by_req.items():
        ranked = sorted(bs, key=lambda b: scores[(b["request_id"], b["bundle_id"])],
                        reverse=True)
        reduced.extend(ranked[:k])
    return ranker, reduced, report["final_loss"]


def gnn_guided_qubo(topology: dict, bundles: List[dict], edge_capacities: dict,
                    memory_capacities: dict, k: int = 8, num_reads: int = 30,
                    seed: int = 42) -> Dict:
    """GNN ranker -> top-k -> QUBO (with feasibility repair)."""
    import time
    from optimization.qubo_optimizer import QUBOOptimizer
    from optimization.openjij_solver import solve_sa

    ranker, reduced, train_loss = gnn_guided_topk(topology, bundles, k=k, seed=seed)
    t0 = time.perf_counter()
    optimizer = QUBOOptimizer(reduced, edge_capacities, memory_capacities)
    bqm = optimizer.to_bqm(
                           congestion_penalty=0.0, memory_congestion_penalty=0.0)
    response = solve_sa(bqm, num_reads=num_reads, seed=seed)
    selected = optimizer.decode_sample(response.first.sample, repair=True)
    elapsed = time.perf_counter() - t0

    util = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return {
        "selected": selected,
        "utility": sum(util.get(k_, 0.0) for k_ in selected),
        "served": len(set(k_[0] for k_ in selected)),
        "n_bundles_in": len(bundles),
        "n_bundles_reduced": len(reduced),
        "wall_time_s": elapsed,
        "gnn_training_loss": train_loss,
    }
