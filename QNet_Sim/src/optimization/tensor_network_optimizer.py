import itertools

import numpy as np
from collections import defaultdict


class TensorNetworkOptimizer:
    def __init__(self, bundles, edge_capacities, memory_capacities,
                 congestion_weight=0.0, congestion_threshold=0.7,
                 risk_weight=0.0, risk_tau=1.0, order_strategy="greedy"):
        if order_strategy not in ("greedy", "spectral"):
            raise ValueError(
                f"unknown order_strategy {order_strategy!r}; expected "
                "'greedy' or 'spectral'")
        self.bundles = bundles
        self.edge_capacities = {self._undirected_edge(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self.congestion_weight = congestion_weight
        self.congestion_threshold = congestion_threshold
        self.risk_weight = risk_weight
        self.risk_tau = risk_tau
        self.order_strategy = order_strategy
        self._greedy_sel = None
        self._validate()
        self._group_by_request()
        self._precompute()
        self._order_requests()

    def _validate(self):
        required = {"bundle_id", "request_id", "path", "edge_demands", "memory_demands", "utility"}
        for b in self.bundles:
            missing = required - b.keys()
            if missing:
                raise ValueError(f"Bundle {b.get('bundle_id', '?')} missing: {missing}")

    def _group_by_request(self):
        self.requests = []
        self.bundles_per_request = {}
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in self.bundles_per_request:
                self.bundles_per_request[rid] = []
                self.requests.append(rid)
            self.bundles_per_request[rid].append(b)

    @staticmethod
    def _undirected_edge(edge):
        return tuple(sorted(edge))

    def _precompute(self):
        self._util_of = {}
        self._edge_of = {}
        self._mem_of = {}
        self._latency_of = {}
        for b in self.bundles:
            key = (b["request_id"], b["bundle_id"])
            self._util_of[key] = b["utility"]
            self._edge_of[key] = {self._undirected_edge(e): d for e, d in b["edge_demands"].items()}
            self._mem_of[key] = dict(b["memory_demands"])
            self._latency_of[key] = b.get("latency", 0.0)

    def _soft_cong_pen(self, e, load):
        """Soft congestion: lambda_cong * max(0, L_e/C_e - theta)^2."""
        if self.congestion_weight <= 0:
            return 0.0
        cap = self.edge_capacities.get(e, 0)
        if cap <= 0:
            return 0.0
        ratio = load / cap
        if ratio <= self.congestion_threshold:
            return 0.0
        return self.congestion_weight * (ratio - self.congestion_threshold) ** 2

    def _risk_of(self, rid, bid):
        """Memory decoherence risk: lambda_risk * (1 - exp(-latency/tau))."""
        if self.risk_weight <= 0 or bid is None:
            return 0.0
        wait = self._latency_of.get((rid, bid), 0.0)
        import math
        return self.risk_weight * (1.0 - math.exp(-wait / self.risk_tau))

    def _get_options(self, rid):
        return [None] + [b["bundle_id"] for b in self.bundles_per_request[rid]]

    def _build_adjacency(self):
        rid_edges = {}
        rid_mems = {}
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in rid_edges:
                rid_edges[rid] = set()
                rid_mems[rid] = set()
            for e in b["edge_demands"]:
                rid_edges[rid].add(self._undirected_edge(e))
            for n in b["memory_demands"]:
                rid_mems[rid].add(n)

        coupling = np.zeros((len(self.requests), len(self.requests)), dtype=int)
        for i, r1 in enumerate(self.requests):
            for j, r2 in enumerate(self.requests):
                if i >= j:
                    continue
                shared = rid_edges[r1] & rid_edges[r2]
                shared |= rid_mems[r1] & rid_mems[r2]
                if shared:
                    coupling[i, j] = coupling[j, i] = len(shared)
        return coupling, rid_edges, rid_mems

    def _order_requests(self):
        coupling, self._rid_edges, self._rid_mems = self._build_adjacency()
        n = len(self.requests)
        if n <= 2:
            self._ordered_requests = list(self.requests)
            return
        if self.order_strategy == "spectral":
            self._ordered_requests = self._spectral_order(coupling)
        else:
            self._ordered_requests = self._greedy_order(coupling)

    def _greedy_order(self, coupling):
        n = len(self.requests)
        ordered = [self.requests[0]]
        remaining = set(range(1, n))
        current = 0
        while remaining:
            best_next = None
            best_weight = -1
            for j in remaining:
                if coupling[current, j] > best_weight:
                    best_weight = coupling[current, j]
                    best_next = j
            if best_next is None:
                best_next = remaining.pop()
            else:
                remaining.remove(best_next)
            ordered.append(self.requests[best_next])
            current = best_next
        return ordered

    def _spectral_order(self, coupling):
        """Structural/spectral heuristic for the MPS coupling order: sorts
        requests within each connected component of the coupling graph by
        their Fiedler-vector value (the eigenvector of the second-smallest
        eigenvalue of the graph Laplacian L = D - coupling). This is a
        training-free, principled stand-in for a "learned" coupling order --
        NOT a trained neural network -- based on the well-established
        spectral-graph-theory result that the Fiedler vector orders a graph's
        vertices to keep strongly-coupled nodes close together, which is
        exactly what an MPS chain wants to minimize the bond dimension
        needed for a given truncation accuracy.
        """
        n = len(self.requests)
        # Union-find to get connected components of the coupling>0 graph.
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if coupling[i, j] > 0:
                    union(i, j)

        components = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        # Order components deterministically: by descending total
        # intra-component coupling weight, then by smallest member index.
        def component_key(members):
            total_weight = sum(coupling[a, b] for a in members for b in members if a < b)
            return (-total_weight, min(members))

        ordered_components = sorted(components.values(), key=component_key)

        ordered_idx = []
        for members in ordered_components:
            if len(members) == 1:
                ordered_idx.extend(members)
                continue
            sub = coupling[np.ix_(members, members)].astype(float)
            degree = np.diag(sub.sum(axis=1))
            laplacian = degree - sub
            eigvals, eigvecs = np.linalg.eigh(laplacian)
            # eigh sorts eigenvalues ascending; index 0 is ~0 (constant
            # eigenvector for a connected component), index 1 is the
            # Fiedler vector.
            fiedler = eigvecs[:, 1] if len(members) > 1 else eigvecs[:, 0]
            member_order = sorted(range(len(members)),
                                  key=lambda k: (fiedler[k], members[k]))
            ordered_idx.extend(members[k] for k in member_order)

        return [self.requests[i] for i in ordered_idx]

    def _local_penalty(self, rid, bid):
        """Self-penalty: overload penalty plus quadratic congestion term (G1/G7).

        The congestion term C*load^2 is exactly decomposed as
        C*sum_b(d_b)^2 = sum_b(C*d_b^2) + sum_{b<b'} 2*C*d_b*d_b',
        so the per-request local C*d^2 term plus the pairwise 2*C*d*d' cross-term
        reproduce the paper's quadratic congestion penalty inside the tensor
        elements themselves.
        """
        if bid is None:
            return 0.0
        pen = 0.0
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            cap = self.edge_capacities.get(edge, 0)
            if d > cap:
                pen += self._B * (d - cap) ** 2
            pen += self._C * d * d
            pen += self._soft_cong_pen(edge, d)
        for node, d in self._mem_of.get((rid, bid), {}).items():
            cap = self.memory_capacities.get(node, 0)
            if d > cap:
                pen += self._D * (d - cap) ** 2
            pen += self._E * d * d
        pen += self._risk_of(rid, bid)
        return pen

    def _pairwise_penalty(self, rid_l, bid_l, rid_r, bid_r):
        """Pairwise penalty: shared-capacity overload plus 2*C*d_l*d_r cross-term."""
        if bid_l is None or bid_r is None:
            return 0.0
        pen = 0.0
        shared_edges = self._rid_edges[rid_l] & self._rid_edges[rid_r]
        shared_mems = self._rid_mems[rid_l] & self._rid_mems[rid_r]
        for edge in shared_edges:
            dl = self._edge_of.get((rid_l, bid_l), {}).get(edge, 0)
            dr = self._edge_of.get((rid_r, bid_r), {}).get(edge, 0)
            if dl + dr > self.edge_capacities.get(edge, 0):
                pen += self._B * (dl + dr - self.edge_capacities[edge]) ** 2
            pen += 2.0 * self._C * dl * dr
            pen += self._soft_cong_pen(edge, dl + dr)
        for node in shared_mems:
            dl = self._mem_of.get((rid_l, bid_l), {}).get(node, 0)
            dr = self._mem_of.get((rid_r, bid_r), {}).get(node, 0)
            if dl + dr > self.memory_capacities.get(node, 0):
                pen += self._D * (dl + dr - self.memory_capacities[node]) ** 2
            pen += 2.0 * self._E * dl * dr
        return pen

    def _selection_energy(self, selections):
        """Full Hamiltonian energy of a selection: -utility + hard/soft penalties
        + memory risk.  Mirrors the energy convention of the Metropolis annealer
        so the two solvers are directly comparable."""
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        util = 0.0
        risk = 0.0
        for rid, bid in selections.items():
            if bid is None:
                continue
            util += self._util_of.get((rid, bid), 0.0)
            for e, d in self._edge_of.get((rid, bid), {}).items():
                edge_load[e] += d
            for n_, d in self._mem_of.get((rid, bid), {}).items():
                mem_load[n_] += d
            risk += self._risk_of(rid, bid)
        pen = 0.0
        for e, load in edge_load.items():
            cap = self.edge_capacities.get(e, 0)
            if load > cap:
                pen += self._B * (load - cap) ** 2
            pen += self._C * load * load
            pen += self._soft_cong_pen(e, load)
        for n_, load in mem_load.items():
            cap = self.memory_capacities.get(n_, 0)
            if load > cap:
                pen += self._D * (load - cap) ** 2
            pen += self._E * load * load
        return -util + pen + risk

    def solve(self, edge_penalty=None, memory_penalty=None,
              congestion_penalty=0.05, memory_congestion_penalty=0.05,
              bond_dim=8, beta=5.0, max_sweeps=0, mf_iters=25,
              adaptive_bond_dim=False, trunc_eps=1e-4, max_bond_dim=64):
        # Penalty defaults are anchored to the utility scale (B, D just above
        # the largest positive bundle utility, the manuscript's rule), so
        # omitting them can never silently under-enforce the capacity
        # constraints on high-utility instances.
        if edge_penalty is None or memory_penalty is None:
            p0 = max((max(0.0, float(u)) for u in self._util_of.values()),
                     default=0.0)
            eps = max(1e-9, 1e-6 * p0)
            if edge_penalty is None:
                edge_penalty = p0 + eps
            if memory_penalty is None:
                memory_penalty = p0 + eps
        self._B = edge_penalty
        self._D = memory_penalty
        self._C = congestion_penalty
        self._E = memory_congestion_penalty
        n = len(self.requests)
        if n == 0:
            return {"selected": [], "selections": {}}

        ordered = self._ordered_requests
        opt_maps = [self._get_options(rid) for rid in ordered]
        dims = [len(opts) for opts in opt_maps]

        # 1. Greedy-baked mean field: the capacity penalties are baked into the
        #    site tensors relative to a feasible reference (the load-aware
        #    greedy selection), so the MPS never collapses onto all-None the
        #    way raw pairwise Boltzmann re-weighting did (RESEARCH_FINDINGS Q1).
        mu, baked_share = self._mean_field_bake(beta, mf_iters, ordered, opt_maps)

        mps = []
        for r_idx in range(n):
            t = np.zeros((dims[r_idx], 1, 1), dtype=np.float64)
            t[:, 0, 0] = mu[r_idx]
            mps.append(t)

        # 2. SVD renormalisation sweeps.  These are optional and default off:
        #    truncating a non-negative joint at chi << product(dim) loses so
        #    much feasible mass that the decode is worse than the greedy
        #    reference (the capacity coupling is a dense complete graph, which
        #    a low-rank chain MPS cannot represent), so the default keeps the
        #    exact product form and treats the sweeps purely as an accuracy
        #    knob for users who want to trade runtime for them.
        bond_dim_trace = [] if adaptive_bond_dim else None
        for _ in range(max_sweeps):
            for direction in ["left", "right"]:
                if direction == "left":
                    for i in range(n - 1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta,
                                            ordered, opt_maps, baked_share,
                                            adaptive_bond_dim=adaptive_bond_dim,
                                            trunc_eps=trunc_eps,
                                            max_bond_dim=max_bond_dim,
                                            trace=bond_dim_trace)
                else:
                    for i in range(n - 2, -1, -1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta,
                                            ordered, opt_maps, baked_share,
                                            adaptive_bond_dim=adaptive_bond_dim,
                                            trunc_eps=trunc_eps,
                                            max_bond_dim=max_bond_dim,
                                            trace=bond_dim_trace)

        # 3. Marginal decoder: contract the full MPS for per-option
        #    probabilities, committing requests by decimation (highest
        #    conditional marginal first), keeping feasibility by construction.
        selections = self._marginal_decoder(mps, opt_maps, ordered)

        # 4. Never return a worse selection than the feasible greedy baseline,
        #    then run a short deterministic add/drop/swap improvement pass so
        #    the result is competitive with the Metropolis annealer.
        if self._greedy_sel is None:
            self._greedy_sel = self._greedy_seed()
        greedy = self._greedy_sel
        mps_util = sum(
            self._util_of.get((r, b), 0.0) for r, b in selections.items() if b is not None
        )
        greedy_util = sum(
            self._util_of.get((r, b), 0.0) for r, b in greedy.items() if b is not None
        )
        if greedy_util > mps_util:
            selections = dict(greedy)
        selections = self._improve(selections)

        final_energy = self._selection_energy(selections)

        selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        result = {"selected": selected, "selections": selections, "energy": final_energy}
        if adaptive_bond_dim:
            result["bond_dim_trace"] = bond_dim_trace
            result["max_chi_used"] = max((t["chi_used"] for t in bond_dim_trace), default=0)
        return result

    def _mean_field_bake(self, beta, mf_iters, ordered, opt_maps):
        """Build the mean-field baseline used for the MPS site tensors and the
        contraction-time residual correction.

        The baseline is the load-aware greedy selection (a feasible
        configuration), not a self-consistent field iterate: the capacity
        coupling has no stable mixed fixed point for a plain simultaneous
        update (the all-serve and all-None poles repel each other), so iterated
        mean field collapses the reference onto all-None and the correction
        then re-introduces exactly the all-None collapse it was meant to
        remove.  Baking the penalties from the feasible greedy reference keeps
        every residual correction bounded and zero on the reference itself, so
        the MPS stays concentrated on the capacity-feasible region.

        Returns ``(mu, baked_share)`` where ``mu[i]`` are the normalized
        marginals for request ``i`` and ``baked_share[(i,j)][b_i]`` is the
        expected pairwise penalty of option ``b_i`` against request ``j``.
        """
        n = len(ordered)
        dims = [len(opts) for opts in opt_maps]
        if self._greedy_sel is None:
            self._greedy_sel = self._greedy_seed()
        g = self._greedy_sel

        mu = []
        for i, rid in enumerate(ordered):
            d = dims[i]
            v = np.zeros(d)
            v[0] = 1.0
            bid = g.get(rid)
            if bid is not None:
                for b_idx, b in enumerate(self.bundles_per_request[rid]):
                    if b["bundle_id"] == bid:
                        v[0] = 0.0
                        v[b_idx + 1] = 1.0
                        break
            # Keep a small uniform exploration mass so the marginal decoder can
            # entertain alternatives to the reference (e.g. swapping a
            # low-value greedy pick for a higher-value one); the reference
            # itself is never removed from the support.
            mu.append(0.9 * v + 0.1 * np.ones(d) / d)

        # Expected pairwise penalty of each option against each other request
        # (the mean-field baseline for the contraction correction).  With the
        # greedy one-hot mu this is exact for the reference configuration.
        baked_share = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rid_i, rid_j = ordered[i], ordered[j]
                share = np.zeros(dims[i])
                for b_idx, bid in enumerate(opt_maps[i]):
                    if bid is None:
                        continue
                    exp_pen = 0.0
                    for b2, b2id in enumerate(opt_maps[j]):
                        if b2id is None:
                            continue
                        exp_pen += mu[j][b2] * self._pairwise_penalty(
                            rid_i, bid, rid_j, b2id)
                    share[b_idx] = exp_pen
                baked_share[(i, j)] = share
        return mu, baked_share

    def _conditional_marginal(self, mps, opt_maps, idx, fixed):
        """Per-option marginal of request ``idx`` given fixed assignments, by
        contracting the full MPS with the fixed sites pinned (a full-marginal
        decoder rather than the old sequential sweep decode)."""
        n = len(mps)
        v = np.ones(1)
        for j in range(idx):
            t = mps[j]
            f = fixed.get(j)
            if f is not None:
                t = t[f:f + 1]
            v = np.einsum("i,bij->j", v, t)
        w = np.ones(1)
        for j in range(n - 1, idx, -1):
            t = mps[j]
            f = fixed.get(j)
            if f is not None:
                t = t[f:f + 1]
            w = np.einsum("j,bij->i", w, t)
        t_i = mps[idx]
        d = t_i.shape[0]
        P = np.array([np.einsum("i,ij,j->", v, t_i[b], w) for b in range(d)])
        s = P.sum()
        if s > 0:
            return P / s
        out = np.zeros(d)
        out[0] = 1.0
        return out

    def _marginal_decoder(self, mps, opt_maps, ordered):
        """Decimation decoder on the full-MPS conditional marginals.

        Repeatedly commits the unfixed request with the highest conditional
        marginal for its best still-feasible option, re-contracting the MPS
        after each commitment.  Feasibility is preserved by construction
        (only options that fit beside the committed selections are scored).
        """
        n = len(ordered)
        fixed = {}
        selections = {}
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        unfixed = list(range(n))

        while unfixed:
            best = None
            best_key = None
            for idx in unfixed:
                P = self._conditional_marginal(mps, opt_maps, idx, fixed)
                rid = ordered[idx]
                opts = opt_maps[idx]
                feasible_b = [b for b in range(len(opts))
                              if opts[b] is None or self._fits(rid, opts[b], edge_load, mem_load)]
                b_star = max(feasible_b, key=lambda b: (
                    P[b], self._util_of.get((rid, opts[b]), 0.0)))
                key = (P[b_star], self._util_of.get((rid, opts[b_star]), 0.0))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (idx, b_star)
            idx, b_star = best
            rid = ordered[idx]
            bid = opt_maps[idx][b_star]
            fixed[idx] = b_star
            selections[rid] = bid
            if bid is not None:
                self._commit(rid, bid, edge_load, mem_load)
            unfixed.remove(idx)
        return selections

    def _fits(self, rid, bid, edge_load, mem_load) -> bool:
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            if edge_load[edge] + d > self.edge_capacities.get(edge, 0):
                return False
        for node, d in self._mem_of.get((rid, bid), {}).items():
            if mem_load[node] + d > self.memory_capacities.get(node, 0):
                return False
        return True

    def _commit(self, rid, bid, edge_load, mem_load):
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            edge_load[edge] += d
        for node, d in self._mem_of.get((rid, bid), {}).items():
            mem_load[node] += d

    def _release(self, rid, bid, edge_load, mem_load):
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            edge_load[edge] -= d
        for node, d in self._mem_of.get((rid, bid), {}).items():
            mem_load[node] -= d

    def _loads_of(self, selections):
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        for rid, bid in selections.items():
            if bid is not None:
                self._commit(rid, bid, edge_load, mem_load)
        return edge_load, mem_load

    def _min_drop_for(self, rid, bid, selections, edge_load, mem_load):
        """Smallest-utility subset of served requests that must be dropped so
        that bundle ``bid`` of ``rid`` fits.  Returns the list of request ids."""
        return self._min_drop_joint([(rid, bid)], selections, edge_load, mem_load)

    def _min_drop_joint(self, bundles, selections, edge_load, mem_load):
        """Smallest-utility subset of served requests that must be dropped so
        that *all* ``(rid, bid)`` bundles in ``bundles`` fit jointly.  Returns
        the list of request ids, or ``[]`` if none of the bundles need room."""
        needed = defaultdict(int)
        for rid, bid in bundles:
            for edge, d in self._edge_of.get((rid, bid), {}).items():
                needed[edge] += d
            for node, d in self._mem_of.get((rid, bid), {}).items():
                needed[node] += d

        deficit = defaultdict(int)
        for e, d in needed.items():
            slack = self.edge_capacities.get(e, 0) - edge_load[e]
            if d > slack:
                deficit[e] = d - slack
        for nd, d in needed.items():
            slack = self.memory_capacities.get(nd, 0) - mem_load[nd]
            if d > slack:
                deficit[nd] = d - slack
        if not deficit:
            return []

        serving = []
        for r, b in selections.items():
            if b is None:
                continue
            e_dem = self._edge_of.get((r, b), {})
            m_dem = self._mem_of.get((r, b), {})
            if any(e in deficit and e_dem.get(e, 0) > 0 for e in e_dem) or \
               any(nd in deficit and m_dem.get(nd, 0) > 0 for nd in m_dem):
                serving.append(r)
        serving.sort(key=lambda r: self._util_of.get((r, selections[r]), 0.0))

        # Exact enumeration of small drop subsets (cheap: requests are few and
        # the drop only ever needs to free a handful of edges).  This finds
        # moves a greedy deficit-cover overshoots (e.g. dropping a request that
        # only touches an already-satisfied edge), which is what kept the swap
        # pass from closing the gap to the annealer.  All bundles are checked
        # against the *joint* post-drop loads (they may share constrained
        # resources), mirroring the exact state after they are committed.  The
        # lowest-utility feasible subset wins: a larger subset can be cheaper
        # than the first feasible singleton (e.g. two low-value requests vs one
        # mid-value one), and size-first enumeration would miss it.
        best_combo = None
        best_util = float("inf")
        for k in (1, 2, 3, 4):
            for combo in itertools.combinations(serving, min(k, len(serving))):
                el = defaultdict(int, edge_load)
                ml = defaultdict(int, mem_load)
                for r in combo:
                    self._release(r, selections[r], el, ml)
                for rid, bid in bundles:
                    self._commit(rid, bid, el, ml)
                if all(el[e] <= self.edge_capacities.get(e, 0) for e in el) and \
                   all(ml[nd] <= self.memory_capacities.get(nd, 0) for nd in ml):
                    u = sum(self._util_of.get((r, selections[r]), 0.0)
                            for r in combo)
                    if u < best_util:
                        best_util = u
                        best_combo = combo
        return list(best_combo) if best_combo is not None else []

    def _improve(self, selections, max_passes=5):
        """Deterministic add / drop / swap local search on top of the MPS
        decode.  Keeps feasibility at every move (only ever drops requests that
        free the exact capacity a higher-value request needs), so the result is
        at least as good as the input and typically matches or beats the
        Metropolis annealer."""
        cur = dict(selections)
        for _ in range(max_passes):
            edge_load, mem_load = self._loads_of(cur)
            changed = False

            # Phase A: add any unserved request whose best bundle fits.
            for rid in self.requests:
                if cur.get(rid) is not None:
                    continue
                best = max(
                    (b for b in self.bundles_per_request[rid]
                     if self._fits(rid, b["bundle_id"], edge_load, mem_load)),
                    key=lambda b: b["utility"], default=None)
                if best is not None:
                    cur[rid] = best["bundle_id"]
                    self._commit(rid, best["bundle_id"], edge_load, mem_load)
                    changed = True

            # Phase B: swap an unserved request in by dropping the smallest
            # utility subset that frees exactly the capacity it needs.
            for rid in self.requests:
                if cur.get(rid) is not None:
                    continue
                best = max(self.bundles_per_request[rid],
                           key=lambda b: b["utility"])
                drop = self._min_drop_for(rid, best["bundle_id"], cur,
                                          edge_load, mem_load)
                drop_util = sum(self._util_of.get((r, cur[r]), 0.0)
                                for r in drop)
                if drop and best["utility"] > drop_util:
                    for r in drop:
                        self._release(r, cur[r], edge_load, mem_load)
                        cur[r] = None
                    cur[rid] = best["bundle_id"]
                    self._commit(rid, best["bundle_id"], edge_load, mem_load)
                    changed = True

            # Phase C: swap a *pair* of unserved requests in together (the
            # two-for-k exchange that closes the last few cases where a single
            # add+drop is net-negative but the joint move is positive).  Only
            # the highest-value unserved requests are considered, bounded to
            # keep the pass cheap.
            unserved = [r for r in self.requests if cur.get(r) is None]
            unserved.sort(key=lambda r: max(b["utility"]
                                            for b in self.bundles_per_request[r]),
                          reverse=True)
            for a_idx in range(min(len(unserved), 8)):
                for b_idx in range(a_idx + 1, min(len(unserved), 8)):
                    ra, rb = unserved[a_idx], unserved[b_idx]
                    ba = max(self.bundles_per_request[ra], key=lambda x: x["utility"])
                    bb = max(self.bundles_per_request[rb], key=lambda x: x["utility"])
                    joint = self._min_drop_joint(
                        [(ra, ba["bundle_id"]), (rb, bb["bundle_id"])],
                        cur, edge_load, mem_load)
                    if not joint:
                        continue
                    # Verify both bundles fit the post-drop loads together
                    # (they are committed jointly and may share a constrained
                    # edge, so a per-bundle check is not enough).
                    el3 = defaultdict(int, edge_load)
                    ml3 = defaultdict(int, mem_load)
                    for r in joint:
                        self._release(r, cur[r], el3, ml3)
                    self._commit(ra, ba["bundle_id"], el3, ml3)
                    self._commit(rb, bb["bundle_id"], el3, ml3)
                    if not (all(el3[e] <= self.edge_capacities.get(e, 0)
                                for e in el3) and
                            all(ml3[nd] <= self.memory_capacities.get(nd, 0)
                                for nd in ml3)):
                        continue
                    drop_util = sum(self._util_of.get((r, cur[r]), 0.0)
                                    for r in joint)
                    gain = ba["utility"] + bb["utility"] - drop_util
                    if gain > 0:
                        for r in joint:
                            self._release(r, cur[r], edge_load, mem_load)
                            cur[r] = None
                        cur[ra] = ba["bundle_id"]
                        cur[rb] = bb["bundle_id"]
                        self._commit(ra, ba["bundle_id"], edge_load, mem_load)
                        self._commit(rb, bb["bundle_id"], edge_load, mem_load)
                        changed = True

            if not changed:
                break
        return cur

    def _greedy_seed(self):
        """Deterministic, load-aware greedy baseline (utility density, then utility, ids).

        Tracks cumulative edge/memory load so the returned selection is always
        capacity-feasible.
        """
        scored = []
        for b in self.bundles:
            total_demand = sum(b["edge_demands"].values()) + sum(b["memory_demands"].values())
            ud = b["utility"] / (total_demand + 1e-10)
            scored.append((ud, -b["utility"], b["request_id"], b["bundle_id"], b))
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        selections = {}
        assigned = set()
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        for _, _, _, _, b in scored:
            rid = b["request_id"]
            if rid in assigned:
                continue
            feasible = True
            for edge, d in b["edge_demands"].items():
                e = tuple(sorted(edge))
                if edge_load[e] + d > self.edge_capacities.get(e, 0):
                    feasible = False
                    break
            if feasible:
                for node, d in b["memory_demands"].items():
                    if mem_load[node] + d > self.memory_capacities.get(node, 0):
                        feasible = False
                        break
            if feasible:
                selections[rid] = b["bundle_id"]
                assigned.add(rid)
                for edge, d in b["edge_demands"].items():
                    edge_load[tuple(sorted(edge))] += d
                for node, d in b["memory_demands"].items():
                    mem_load[node] += d
        for rid in self.requests:
            if rid not in selections:
                selections[rid] = None
        return selections

    def _contract_bond(self, mps, left, right, bond_dim, beta, ordered, opt_maps,
                       baked_share=None, adaptive_bond_dim=False, trunc_eps=1e-4,
                       max_bond_dim=64, trace=None):
        tl = mps[left]
        tr = mps[right]
        dl = tl.shape[0]
        dr = tr.shape[0]
        chi_l = tl.shape[1]
        chi_r = tr.shape[2]
        rid_l = ordered[left]
        rid_r = ordered[right]

        combined = np.einsum("abf,cfd->acbd", tl, tr)
        combined = combined.reshape(dl * chi_l, dr * chi_r)

        if baked_share is not None:
            share_l = baked_share.get((left, right))
            share_r = baked_share.get((right, left))
            for bl in range(dl):
                bid_l = opt_maps[left][bl]
                for br in range(dr):
                    bid_r = opt_maps[right][br]
                    actual = self._pairwise_penalty(rid_l, bid_l, rid_r, bid_r)
                    expected = (share_l[bl] if share_l is not None else 0.0) + \
                               (share_r[br] if share_r is not None else 0.0)
                    # Only down-weight the mean-field residual; the factor is
                    # bounded below so the SVD never sees underflowed blocks.
                    residual = max(actual - expected, 0.0)
                    if residual > 0.0:
                        factor = max(np.exp(-beta * residual), 1e-300)
                        combined[bl * chi_l:(bl + 1) * chi_l,
                                 br * chi_r:(br + 1) * chi_r] *= factor

        combined = np.nan_to_num(combined, nan=0.0, posinf=1e150, neginf=-1e150)
        combined = np.maximum(combined, 0.0)

        try:
            u, s, vh = np.linalg.svd(combined, full_matrices=False)
        except np.linalg.LinAlgError:
            u, s, vh = np.linalg.svd(combined + 1e-12 * np.eye(combined.shape[0], combined.shape[1]),
                                     full_matrices=False)

        if adaptive_bond_dim:
            # Truncation-error-driven bond dimension: grow chi to the
            # smallest value that retains (1 - trunc_eps) of the singular
            # value weight (sum of squares), capped by max_bond_dim.
            total_weight = float(np.sum(s ** 2))
            if total_weight <= 0.0:
                k = 1
            else:
                cumulative = np.cumsum(s ** 2) / total_weight
                k = int(np.searchsorted(cumulative, 1.0 - trunc_eps) + 1)
                k = max(1, min(k, max_bond_dim, len(s)))
            if trace is not None:
                retained = float(np.sum(s[:k] ** 2))
                trunc_weight = 1.0 - (retained / total_weight if total_weight > 0 else 1.0)
                trace.append({"left": left, "right": right, "chi_used": k,
                              "trunc_weight": trunc_weight})
        else:
            k = min(bond_dim, len(s))

        u = u[:, :k]
        s = s[:k]
        vh = vh[:k, :]
        tl_new = (u * s).reshape(dl, chi_l, k)
        tr_new = vh.reshape(k, dr, chi_r).transpose(1, 0, 2)
        # A truncated SVD of a signed matrix can reintroduce negative
        # amplitudes; clamping keeps the MPS a non-negative tensor network so
        # the contraction marginals remain proper probabilities.
        tl_new = np.maximum(tl_new, 0.0)
        tr_new = np.maximum(tr_new, 0.0)
        mps[left] = tl_new
        mps[right] = tr_new

    def _feasibility_repair(self, selections):
        def is_feasible(sel):
            edge_load = defaultdict(int)
            mem_load = defaultdict(int)
            for rid, bid in sel.items():
                if bid is None:
                    continue
                for edge, d in self._edge_of.get((rid, bid), {}).items():
                    edge_load[edge] += d
                for node, d in self._mem_of.get((rid, bid), {}).items():
                    mem_load[node] += d
            for edge, load in edge_load.items():
                if load > self.edge_capacities.get(edge, 0):
                    return False
            for node, load in mem_load.items():
                if load > self.memory_capacities.get(node, 0):
                    return False
            return True

        if is_feasible(selections):
            return selections
        for rid in self._ordered_requests:
            bundles = self.bundles_per_request[rid]
            for b in bundles:
                trial = dict(selections)
                trial[rid] = b["bundle_id"]
                if is_feasible(trial):
                    return trial
        return {rid: None for rid in self._ordered_requests}

    def decode_sample(self, result):
        return result["selected"]
