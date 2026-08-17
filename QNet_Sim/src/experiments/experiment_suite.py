import sys, os, csv, math, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dimod

from experiments.instances import (
    generate_chain_topology, generate_grid_topology,
    generate_benchmark_instance, contention_sweep_instances,
)
from experiments.metrics import ExperimentTracker
from experiments.benchmark import build_metropolis, build_tensor_network

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiments"))
os.makedirs(OUT, exist_ok=True)


def run_contention_sweep():
    """Vary request count on chain topology at different edge capacities."""
    all_rows = []
    for cap in [4, 6, 10]:
        for n_nodes in [4, 6, 10]:
            topo_fn = lambda n=n_nodes, c=cap: generate_chain_topology(
                n_nodes=n, edge_capacity=c, latency=5.0, raw_fidelity=0.85
            )
            instances = contention_sweep_instances(topo_fn, [2, 4, 8, 16, 24], seed=42)
            for name, inst in instances.items():
                n_req = inst["n_requests"]
                n_bun = len(inst["bundles"])
                b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

                for seed in [42, 43, 44]:
                    meta_sf = build_metropolis(b, ec, mc, seed=seed)
                    t0 = time.perf_counter()
                    meta_r = meta_sf(
                                     congestion_penalty=0.0, memory_congestion_penalty=0.0,
                                     max_iterations=5000, cooling_rate=0.97,
                                     n_restarts=1, steps_per_temperature=10)
                    mt = time.perf_counter() - t0

                    tn_sf = build_tensor_network(b, ec, mc)
                    t0 = time.perf_counter()
                    tn_r = tn_sf(bond_dim=8, beta=5.0)
                    tt = time.perf_counter() - t0

                    for solver, r, t in [("Metropolis", meta_r, mt), ("TensorNetwork", tn_r, tt)]:
                        all_rows.append({
                            "topology": f"chain_{n_nodes}",
                            "n_nodes": n_nodes,
                            "edge_capacity": cap,
                            "n_requests": n_req,
                            "n_bundles": n_bun,
                            "solver": solver,
                            "seed": seed,
                            "served": len(r.get("selected", [])),
                            "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                            "utility": sum(_u(b, rid, bid) for rid, bid in r.get("selected", [])),
                            "time_s": t,
                        })

    path = os.path.join(OUT, "contention_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")
    return all_rows


def _u(bundles, rid, bid):
    for b in bundles:
        if b["request_id"] == rid and b["bundle_id"] == bid:
            return b["utility"]
    return 0.0


def run_bond_dim_sweep():
    """Vary TN bond dimension across request counts."""
    all_rows = []
    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6)
    instances = contention_sweep_instances(topo_fn, [4, 8, 16, 24], seed=42)

    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

        for bd in [1, 2, 4, 8, 16, 32]:
            tn_sf = build_tensor_network(b, ec, mc)
            t0 = time.perf_counter()
            tn_r = tn_sf(bond_dim=bd, beta=5.0)
            tt = time.perf_counter() - t0
            all_rows.append({
                "topology": "chain_8",
                "n_requests": n_req,
                "bond_dim": bd,
                "solver": f"TN(χ={bd})",
                "served": len(tn_r.get("selected", [])),
                "served_ratio": len(tn_r.get("selected", [])) / max(n_req, 1),
                "utility": sum(_u(b, rid, bid) for rid, bid in tn_r.get("selected", [])),
                "time_s": tt,
            })

        meta_sf = build_metropolis(b, ec, mc, seed=42)
        t0 = time.perf_counter()
        meta_r = meta_sf(
                         congestion_penalty=0.0, memory_congestion_penalty=0.0,
                         max_iterations=5000, cooling_rate=0.97,
                         n_restarts=1, steps_per_temperature=10)
        mt = time.perf_counter() - t0
        all_rows.append({
            "topology": "chain_8",
            "n_requests": n_req,
            "bond_dim": -1,
            "solver": "Metropolis-SA",
            "served": len(meta_r.get("selected", [])),
            "served_ratio": len(meta_r.get("selected", [])) / max(n_req, 1),
            "utility": sum(_u(b, rid, bid) for rid, bid in meta_r.get("selected", [])),
            "time_s": mt,
        })

    path = os.path.join(OUT, "bond_dim_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_adaptive_bond_dim_sweep():
    """Future work: adaptive bond dimension driven by truncation error.

    Sweeps the truncation-error target trunc_eps at a generous max_bond_dim
    cap and reports the bond dimension the solver actually uses (chi_used)
    versus a fixed-chi baseline at the same request counts."""
    all_rows = []
    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6)
    instances = contention_sweep_instances(topo_fn, [8, 16, 24], seed=42)

    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

        for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
            tn_sf = build_tensor_network(b, ec, mc)
            t0 = time.perf_counter()
            r = tn_sf(adaptive_bond_dim=True, trunc_eps=eps, max_bond_dim=64,
                     max_sweeps=2, beta=5.0)
            tt = time.perf_counter() - t0
            all_rows.append({
                "topology": "chain_8",
                "n_requests": n_req,
                "trunc_eps": eps,
                "max_chi_used": r["max_chi_used"],
                "served": len(r.get("selected", [])),
                "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                "utility": sum(_u(b, rid, bid) for rid, bid in r.get("selected", [])),
                "time_s": tt,
            })

    path = os.path.join(OUT, "adaptive_bond_dim_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_mps_ordering_experiments():
    """Future work: learned (spectral) MPS coupling order.

    Compares the existing greedy nearest-neighbor coupling order against a
    Fiedler-vector spectral ordering (a training-free, structural stand-in
    for a "learned" order) on solution quality and wall time."""
    from optimization.tensor_network_optimizer import TensorNetworkOptimizer

    all_rows = []
    for label, topo_fn in [
        ("chain_10", lambda: generate_chain_topology(n_nodes=10, edge_capacity=6)),
        ("grid_6x6", lambda: generate_grid_topology(rows=6, cols=6, edge_capacity=5)),
    ]:
        instances = contention_sweep_instances(topo_fn, [12, 16], seed=42)
        for name, inst in instances.items():
            n_req = inst["n_requests"]
            b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
            for strategy in ["greedy", "spectral"]:
                opt = TensorNetworkOptimizer(b, ec, mc, order_strategy=strategy)
                t0 = time.perf_counter()
                r = opt.solve(bond_dim=4, max_sweeps=2, beta=5.0)
                tt = time.perf_counter() - t0
                all_rows.append({
                    "topology": label,
                    "n_requests": n_req,
                    "order_strategy": strategy,
                    "served": len(r.get("selected", [])),
                    "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                    "utility": sum(_u(b, rid, bid) for rid, bid in r.get("selected", [])),
                    "energy": r["energy"],
                    "time_s": tt,
                })

    path = os.path.join(OUT, "mps_ordering_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_grid_comparison():
    """Compare solvers on grid topology."""
    all_rows = []
    for rows, cols in [(2, 3), (3, 3), (4, 4)]:
        topo_fn = lambda r=rows, c=cols: generate_grid_topology(
            rows=r, cols=c, edge_capacity=5, latency=5.0
        )
        instances = contention_sweep_instances(topo_fn, [8, 16], seed=42)
        for name, inst in instances.items():
            n_req = inst["n_requests"]
            b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

            meta_sf = build_metropolis(b, ec, mc, seed=42)
            t0 = time.perf_counter()
            meta_r = meta_sf(
                             congestion_penalty=0.0, memory_congestion_penalty=0.0,
                             max_iterations=5000, cooling_rate=0.97,
                             n_restarts=1, steps_per_temperature=10)
            mt = time.perf_counter() - t0

            tn_sf = build_tensor_network(b, ec, mc)
            t0 = time.perf_counter()
            tn_r = tn_sf(bond_dim=8, beta=5.0)
            tt = time.perf_counter() - t0

            for solver, r, t in [("Metropolis", meta_r, mt), ("TensorNetwork", tn_r, tt)]:
                all_rows.append({
                    "topology": f"grid_{rows}x{cols}",
                    "n_requests": n_req,
                    "solver": solver,
                    "served": len(r.get("selected", [])),
                    "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                    "utility": sum(_u(b, rid, bid) for rid, bid in r.get("selected", [])),
                    "time_s": t,
                })

    path = os.path.join(OUT, "grid_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_streaming_comparison():
    """Compare streaming vs batched on a request arrival sequence."""
    from optimization.streaming_annealer import StreamingAnnealer

    all_rows = []
    for n_req in [4, 8, 12, 16]:
        topo = generate_chain_topology(n_nodes=8, edge_capacity=6)
        pairs = []
        rng = __import__("random").Random(42)
        nodes = topo["nodes"]
        for _ in range(n_req):
            src, dst = rng.sample(nodes, 2)
            pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))

        from experiments.instances import generate_request_bundles
        all_bundles = []
        all_by_rid = {}
        for i, (src, dst, w, mf) in enumerate(pairs):
            rid = f"req_{i}"
            bs = generate_request_bundles(topo, src, dst, w, mf)
            for j, b in enumerate(bs):
                b["bundle_id"] = f"{rid}_b{j}"
                b["request_id"] = rid
            all_by_rid[rid] = bs
            all_bundles.extend(bs)

        # Streaming: add incrementally
        sa = StreamingAnnealer(topo["edge_capacities"], topo["memory_capacities"], seed=42)
        t0 = time.perf_counter()
        for rid, bs in all_by_rid.items():
            sa.add_request(rid, bs)
            sa.local_sweep(n_steps=50, temperature=2.0)
        st = time.perf_counter() - t0
        sa_sel = sa.get_selected()

        # Batched: solve from scratch
        opt = __import__("optimization.metropolis_annealer", fromlist=["MetropolisAnnealer"]).MetropolisAnnealer(
            all_bundles, topo["edge_capacities"], topo["memory_capacities"], seed=42
        )
        t0 = time.perf_counter()
        r = opt.solve(
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        bt = time.perf_counter() - t0

        for solver, sel, t in [("Streaming", sa_sel, st), ("Batched", r["selected"], bt)]:
            all_rows.append({
                "n_requests": n_req,
                "solver": solver,
                "served": len(sel),
                "served_ratio": len(sel) / max(n_req, 1),
                "utility": sum(_u(all_bundles, rid, bid) for rid, bid in sel),
                "time_s": t,
            })

    path = os.path.join(OUT, "streaming_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_hamiltonian_encoding_comparison():
    """Extension 12.1: slack-variable QUBO vs direct (slack-free) Hamiltonian.

    Both formulations are compiled through pyqubo and solved with OpenJij SA on
    identical instances, so the only difference is the constraint encoding:
    slack variables + squared-deviation terms (Eq. 5.3-5.4) vs the direct
    fixed-scale lambdas (Eq. 12.1).  Both use the at-most-one-bundle-per-request
    convention (requests may be left unserved).

    The one-per-request penalty is anchored to the utility scale
    (``A = scale * u_max + eps``, the manuscript's ``A > max_{r,b} u_{r,b}``
    rule), so the at-most-one constraint is enforced at every sweep point and
    the ``scale`` axis measures the penalty multiplier relative to utility
    rather than an absolute value that can fall below the utility scale.
    Raw multi-bundle conflict rates (before repair and before collapsing to
    the highest-utility bundle per request) are recorded explicitly so a
    weakly enforced constraint cannot be silently masked.  Every decoded
    selection is additionally run through the same deterministic
    capacity/conflict repair so the reported utility reflects a feasible plan.
    """
    from optimization.physics_hamiltonian import PhysicalHamiltonian
    from optimization.openjij_solver import solve_sa
    from optimization.conventional_calibrator import penalty_epsilon

    all_rows = []
    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=8,
                                              memory_capacity=12)
    instances = contention_sweep_instances(topo_fn, [4, 8, 12], seed=42)

    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
        p0 = max((max(0.0, float(bd["utility"])) for bd in b), default=0.0)

        for scale in [1.0, 10.0, 100.0, 1000.0]:
            # Penalty anchored to the utility scale: at scale=1 every hard
            # coefficient sits at the manuscript's A > max_{r,b} u_{r,b} rule.
            pen = scale * p0 + penalty_epsilon(p0)

            # (a) Slack-variable QUBO via QUBOOptimizer (uses Placeholder A/B/D).
            slack_opt = __import__("optimization.qubo_optimizer",
                                   fromlist=["QUBOOptimizer"]).QUBOOptimizer(b, ec, mc)
            try:
                bqm = slack_opt.to_bqm(penalty=pen, edge_penalty=pen,
                                       memory_penalty=pen,
                                       congestion_penalty=0.0,
                                       memory_congestion_penalty=0.0)
                resp = solve_sa(bqm, num_reads=50, seed=42)
                sel = slack_opt.decode_sample(resp.first.sample, repair=True)
                slack_vars = bqm.num_variables
                slack_conflict_rate, slack_mean_multi = _raw_conflict_stats(slack_opt, resp)
            except Exception:
                sel, slack_vars = [], 0
                slack_conflict_rate = slack_mean_multi = float("nan")

            # (b) Direct slack-free Hamiltonian (Eq. 12.1), same solver backend:
            # only |B| binary variables, no LogEncInteger slack variables.
            phys = PhysicalHamiltonian(b, ec, mc)
            try:
                q, offset = phys.to_qubo_slackfree(
                    utility_weight=1.0, one_per_request_weight=pen,
                    congestion_weight=pen, memory_ratio_weight=pen)
                dqm = dimod.BQM.from_qubo(q, offset)
                resp = solve_sa(dqm, num_reads=50, seed=42)
                var = resp.first.sample
                var["x_0"] = var.get("x_0", 0)
                sel2 = slack_opt.repair_selection([k for k in phys.decode(var)])
                direct_vars = dqm.num_variables
                direct_conflict_rate, direct_mean_multi = _raw_conflict_stats(phys, resp)
            except Exception:
                sel2, direct_vars = [], 0
                direct_conflict_rate = direct_mean_multi = float("nan")

            # (c) Direct Hamiltonian driven by the Metropolis annealer
            # (no slack variables by construction, incremental local moves;
            # enforces at-most-one by construction, so the raw conflict rate
            # is zero by design).
            meta_sf = build_metropolis(b, ec, mc, seed=42)
            meta_r = meta_sf(penalty=pen, edge_penalty=pen, memory_penalty=pen,
                             congestion_penalty=0.0, memory_congestion_penalty=0.0,
                             max_iterations=3000, n_restarts=1, steps_per_temperature=10)
            meta_sel = meta_r.get("selected", [])

            for formulation, sel, n_vars, cr, mm in [
                ("Slack-QUBO", sel, slack_vars, slack_conflict_rate, slack_mean_multi),
                ("Direct-QUBO", sel2, direct_vars, direct_conflict_rate, direct_mean_multi),
                ("Direct-Metropolis", meta_sel, len(b), 0.0, 0.0),
            ]:
                # A request is served if at least one of its bundles is
                # selected; keep the highest-utility bundle per request so
                # served ratio and utility are comparable across formulations
                # even when the one-per-request penalty is weakly enforced.
                # The raw conflict rate above reports the pre-repair samples.
                best = {}
                for k in sel:
                    u_k = _u_sum(b, [k])
                    if u_k > best.get(k[0], -1e18):
                        best[k[0]] = u_k
                served = len(best)
                plan_util = sum(best.values())
                all_rows.append({
                    "instance": name,
                    "n_requests": n_req,
                    "scale": scale,
                    "formulation": formulation,
                    "served": served,
                    "served_ratio": served / max(n_req, 1),
                    "utility": plan_util,
                    "n_vars": n_vars,
                    "raw_conflict_rate": cr,
                    "raw_mean_multi_bundle": mm,
                })

    path = os.path.join(OUT, "hamiltonian_encoding_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")
    return all_rows


def _raw_conflict_stats(decoder, resp):
    """Raw at-most-one violation statistics over all reads (no repair/masking).

    Returns (fraction of reads with at least one request selecting >1 bundle,
    mean number of requests selecting >1 bundle per read).
    """
    n_reads = 0
    n_conflict = 0
    n_multi_total = 0
    for sample in resp.samples():
        if hasattr(decoder, "decode_sample"):
            selected = decoder.decode_sample(sample)
        else:
            selected = decoder.decode(sample)
        counts = {}
        for rid, _bid in selected:
            counts[rid] = counts.get(rid, 0) + 1
        n_multi = sum(1 for c in counts.values() if c > 1)
        n_conflict += 1 if n_multi > 0 else 0
        n_multi_total += n_multi
        n_reads += 1
    if n_reads == 0:
        return float("nan"), float("nan")
    return n_conflict / n_reads, n_multi_total / n_reads


def _u_sum(bundles, selected):
    util_of = {(b2["request_id"], b2["bundle_id"]): b2["utility"] for b2 in bundles}
    return sum(util_of.get(k, 0.0) for k in selected)


def run_selfish_routing_experiments():
    """Extension 1 (D1): congestion games, price of anarchy and tolls.

    Two classes of experiments: (a) the atomic Pigou network, where a shared
    bottleneck produces pure-Nash equilibria whose social cost exceeds the
    optimum (PoA -> 4/3 as the number of players grows), and marginal-cost
    tolls recover the optimum; (b) Braess's paradox in the Wardrop
    (splittable-flow) model, where adding a zero-cost shortcut *raises* the
    equilibrium latency from 1.5 to 2.0 and tolls restore 1.5.
    """
    from extensions.selfish_routing import (
        braess_capacity_sweep, braess_marginal_toll_sweep,
        braess_toll_sweep, pigou_instance, pigou_poa_sweep, toll_sweep,
    )

    def _write(rows, name):
        path = os.path.join(OUT, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")

    _write(pigou_poa_sweep(), "selfish_routing_pigou_poa.csv")

    bundles, caps, ov = pigou_instance(n_players=3)
    _write(toll_sweep(bundles, caps, demand_model="unit",
                      beta_override=ov["beta"], base_override=ov["base"]),
           "selfish_routing_pigou_toll.csv")

    _write(braess_capacity_sweep(), "selfish_routing_braess_capacity.csv")
    _write(braess_toll_sweep(), "selfish_routing_braess_toll.csv")
    _write(braess_marginal_toll_sweep(), "selfish_routing_braess_wardrop_toll.csv")


def _fragile_bottleneck_bundles():
    bottleneck = tuple(sorted(("S", "B")))

    def mk(rid, bid, edges, util):
        return {"bundle_id": bid, "request_id": rid, "path": list(edges),
                "edge_demands": {tuple(sorted(e)): 1 for e in edges},
                "memory_demands": {}, "utility": util}

    bundles = [
        mk("R1", "high", [("S", "B"), ("B", "T1")], 100.0),
        mk("R1", "safe", [("S", "T1")], 60.0),
        mk("R2", "high", [("S", "B"), ("B", "T2")], 95.0),
        mk("R2", "safe", [("S", "T2")], 50.0),
    ]
    caps = {bottleneck: 1.0,
            tuple(sorted(("B", "T1"))): 10.0, tuple(sorted(("B", "T2"))): 10.0,
            tuple(sorted(("S", "T1"))): 10.0, tuple(sorted(("S", "T2"))): 10.0}
    mem = {"S": 100.0, "B": 100.0, "T1": 100.0, "T2": 100.0}
    return bundles, caps, mem


def _fragile_scenarios(bundles, failure_prob, severity, n_scenarios, seed):
    bottleneck = tuple(sorted(("S", "B")))
    rng = random.Random(seed)
    scenarios = []
    for _ in range(n_scenarios):
        fail = rng.random() < failure_prob
        s = {}
        for b in bundles:
            u = b["utility"]
            if fail and bottleneck in b["edge_demands"]:
                u *= (1.0 - severity)
            s[(b["request_id"], b["bundle_id"])] = u
        scenarios.append(s)
    return scenarios


def run_robust_routing_experiments():
    """Extension 2 (D2): robust routing under uncertain utilities.

    (a) On a fragile bottleneck, the nominal router maximises expected utility
    but lands in a poor worst case; the maximin router (gamma=1) raises the
    worst-case utility at a price in nominal utility (the price of robustness);
    a min-max-regret router trades between the two.  (b) We sweep the failure
    probability to show the robustness gain grows with uncertainty, and (c)
    compare nominal vs robust on ordinary chain instances under random noise.
    """
    import random as _random
    from extensions.robust_routing import (
        RobustRoutingModel, generate_scenarios,
    )

    def _write(rows, name):
        path = os.path.join(OUT, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")

    kw = dict(
              max_iterations=5000, n_restarts=3, steps_per_temperature=10)

    # (a) Decision-criterion comparison on the fragile bottleneck (exact).
    bundles, caps, mem = _fragile_bottleneck_bundles()
    scenarios = _fragile_scenarios(bundles, failure_prob=0.6, severity=0.9,
                                   n_scenarios=12, seed=3)
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=42)
    criterion_rows = []
    for label, sel in [("nominal", model.solve_exact(0.0)),
                       ("maximin", model.solve_exact(1.0)),
                       ("min-max-regret", model.solve_exact_regret())]:
        ev = model.evaluate(sel)
        reg = model.regret(sel, exact=True)
        criterion_rows.append({
            "criterion": label,
            "nominal_util": ev["nominal_util"],
            "worst_util": ev["worst_util"],
            "mean_util": ev["mean_util"],
            "max_regret": reg["max_regret"],
            "mean_regret": reg["mean_regret"],
        })
    _write(criterion_rows, "robust_routing_criteria.csv")

    # (b) Uncertainty sweep: robustness gain vs failure probability.
    noise_rows = []
    for failure_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = _fragile_scenarios(bundles, failure_prob, severity=0.9,
                               n_scenarios=12, seed=3)
        m = RobustRoutingModel(bundles, caps, mem, s, seed=42)
        g = m.robustness_gain(exact=True)
        noise_rows.append({
            "failure_prob": failure_prob,
            "worst_util_gain": g["worst_util_gain"],
            "nominal_util_loss": g["nominal_util_loss"],
            "nominal_worst_util": g["nominal_worst_util"],
            "robust_worst_util": g["robust_worst_util"],
        })
    _write(noise_rows, "robust_routing_uncertainty_sweep.csv")

    # (c) Pareto sweep on the fragile bottleneck (gamma sweep).
    _write(model.pareto_sweep(gammas=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], exact=True),
           "robust_routing_pareto.csv")

    # (d) Chain-instance comparison under random noise.
    instance_rows = []
    for n_req in [4, 6, 8]:
        topo = generate_chain_topology(n_nodes=8, edge_capacity=6, memory_capacity=10)
        rng = _random.Random(42)
        pairs = []
        for _ in range(n_req):
            src, dst = rng.sample(topo["nodes"], 2)
            pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))
        b2, ec2, mc2 = generate_benchmark_instance(topo, pairs, rng)
        scen = generate_scenarios(b2, n_scenarios=8, noise_scale=0.25,
                                  failure_prob=0.3, seed=7)
        m2 = RobustRoutingModel(b2, ec2, mc2, scen, seed=42)
        ev_n = m2.evaluate(m2.solve(0.0, **kw))
        ev_r = m2.evaluate(m2.solve(1.0, **kw))
        instance_rows.append({
            "n_requests": n_req,
            "n_bundles": len(b2),
            "nominal_util": ev_n["nominal_util"],
            "nominal_worst": ev_n["worst_util"],
            "robust_util": ev_r["nominal_util"],
            "robust_worst": ev_r["worst_util"],
            "worst_util_gain": ev_r["worst_util"] - ev_n["worst_util"],
            "nominal_util_loss": ev_n["nominal_util"] - ev_r["nominal_util"],
        })
    _write(instance_rows, "robust_routing_instances.csv")


def _write_rows(rows, name):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def run_joint_scheduling_experiments():
    """Extension 1 (E1): joint routing + temporal memory scheduling vs the
    memory-agnostic aggregate-slot router, plus the three control regimes
    (static / online / receding-horizon, E3) on identical arrival traces."""
    from optimization.joint_scheduler import run_joint_comparison
    from optimization.online_optimizers import run_regime_comparison

    topo_fn = lambda: generate_chain_topology(n_nodes=10, edge_capacity=8,
                                              memory_capacity=12,
                                              raw_fidelity=0.85)
    rows = []
    for mean_rate in [1.0, 1.8]:
        for tau_mem in [3.0, 8.0]:
            res = run_joint_comparison(topo_fn, n_slots=12, mean_rate=mean_rate,
                                       tau_mem=tau_mem, seed=42)
            for label in ("joint", "memory_agnostic"):
                rows.append({"mean_rate": mean_rate, "tau_mem": tau_mem,
                             "regime": label,
                             **res["trace"], **res[label]})
    _write_rows(rows, "joint_scheduling_comparison.csv")

    regime_rows = []
    for mean_rate in [0.8, 1.5]:
        res = run_regime_comparison(topo_fn, n_slots=10, mean_rate=mean_rate,
                                    tau_mem=5.0, window_size=3, seed=42)
        for r in res["rows"]:
            regime_rows.append({"mean_rate": mean_rate, "n_requests": res["n_requests"],
                                **r})
    _write_rows(regime_rows, "online_regimes_comparison.csv")


def run_adaptive_qubo_experiments():
    """Extension 7 (E7): adaptive candidate reduction for the QUBO."""
    from optimization.adaptive_qubo import run_topk_sweep

    rows = []
    for name, topo_fn in [("chain_10", lambda: generate_chain_topology(
                              n_nodes=10, edge_capacity=6, memory_capacity=10)),
                          ("grid_6x6", lambda: generate_grid_topology(
                              rows=6, cols=6, edge_capacity=6,
                              memory_capacity=10))]:
        res = run_topk_sweep(topo_fn, n_requests=12, num_reads=20, seed=42)
        for r in res["rows"]:
            rows.append({"topology": name, "n_bundles_in": res["n_bundles"],
                         **r})
    _write_rows(rows, "adaptive_qubo_topk.csv")


def run_hybrid_pipeline_experiments():
    """Extension 6 (E6): four-stage hybrid pipeline vs its ablations."""
    from optimization.hybrid_pipeline import run_hybrid_comparison

    rows = []
    for n_requests in [6, 12]:
        topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                                  memory_capacity=10)
        res = run_hybrid_comparison(topo_fn, n_requests=n_requests, seed=42)
        for stage, d in [("full_pipeline", res["full_pipeline"]),
                         ("qubo_only", res["qubo_only"]),
                         ("qubo_plus_repair", res["qubo_plus_repair"])]:
            rows.append({"n_requests": n_requests,
                         "n_bundles_in": res["n_bundles_in"],
                         "n_bundles_reduced": res["n_bundles_reduced"],
                         "stage": stage,
                         "utility": d["utility"],
                         "served": d["served"]})
    _write_rows(rows, "hybrid_pipeline_comparison.csv")


def run_gnn_experiments():
    """Extension 8 (E8): GNN-guided candidate reduction vs adaptive top-k and
    the full-candidate QUBO reference."""
    import random as _random
    from baselines.gnn_ranker import gnn_guided_qubo
    from optimization.adaptive_qubo import adaptive_qubo_solve, reference_solution
    from experiments.instances import generate_benchmark_instance

    rows = []
    for name, topo in [("chain_10", generate_chain_topology(
                           n_nodes=10, edge_capacity=6, memory_capacity=10)),
                       ("grid_5x5", generate_grid_topology(
                           rows=5, cols=5, edge_capacity=6,
                           memory_capacity=10))]:
        rng = _random.Random(7)
        pairs = []
        for _ in range(10):
            src, dst = rng.sample(topo["nodes"], 2)
            pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))
        b, ec, mc = generate_benchmark_instance(topo, pairs, rng)

        ref = reference_solution(b, ec, mc, num_reads=20, seed=42)
        adap = adaptive_qubo_solve(b, ec, mc, k=8, num_reads=20, seed=42)
        gnn = gnn_guided_qubo(topo, b, ec, mc, k=8, num_reads=20, seed=42)
        for label, r in [("full_qubo", ref), ("adaptive_topk", adap),
                         ("gnn_guided", gnn)]:
            rows.append({"topology": name, "method": label,
                         "n_bundles_in": len(b),
                         "n_bundles_in_qubo": r.get("n_bundles_reduced",
                                                    r.get("n_bundles_in_qubo", len(b))),
                         "utility": r["utility"],
                         "served": r["served"],
                         "n_qubo_variables": r.get("n_qubo_variables", ""),
                         "wall_time_s": r.get("wall_time_s", r.get("gnn_training_loss", ""))})
    _write_rows(rows, "gnn_candidate_reduction.csv")


def run_multi_objective_experiments():
    """Extension 9 (E9): Pareto frontiers and constraint queries."""
    import random as _random
    from extensions.multi_objective import (
        pareto_frontier, constraint_frontier, selection_objectives,
    )
    from experiments.instances import generate_benchmark_instance

    rng = _random.Random(3)
    topo = generate_grid_topology(rows=3, cols=3, edge_capacity=6,
                                  memory_capacity=10)
    pairs = []
    for _ in range(5):
        src, dst = rng.sample(topo["nodes"], 2)
        pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.4, 0.7)))
    b, ec, mc = generate_benchmark_instance(topo, pairs, rng)

    front = pareto_frontier(b, ec, mc, max_combos=20000)
    front_rows = []
    for i, p in enumerate(front):
        front_rows.append({"frontier_index": i,
                           "throughput": p["objectives"]["throughput"],
                           "fidelity": p["objectives"]["fidelity"],
                           "latency": p["objectives"]["latency"],
                           "memory": p["objectives"]["memory"]})
    _write_rows(front_rows, "multi_objective_frontier.csv")

    targets = [0.4, 0.5, 0.6, 0.7]
    rows = []
    for constrain, maximize in [("fidelity", "throughput"),
                                ("throughput", "fidelity")]:
        for r in constraint_frontier(b, ec, mc, targets, constrain=constrain,
                                     maximize=maximize, max_combos=20000):
            rows.append({"constrain": constrain, "maximize": maximize,
                         "target": r["target"], "feasible": r["feasible"],
                         "achieved_constraint": r.get(constrain),
                         "achieved_maximized": r.get(maximize)})
    _write_rows(rows, "multi_objective_constraint.csv")


def run_disjoint_paths_experiments():
    """Extension 11 (E11): k-disjoint path provisioning."""
    from extensions.disjoint_paths import run_disjoint_comparison

    topo = generate_grid_topology(rows=3, cols=4, edge_capacity=4,
                                  memory_capacity=12)
    rows = []
    for n_requests in [4, 8]:
        res = run_disjoint_comparison(topo, n_requests=n_requests,
                                      n_expected_disjoint=2, seed=42)
        for r in res["rows"]:
            rows.append({"n_requests": n_requests,
                         "n_bundles_single": res["n_bundles_single"],
                         "n_bundles_multipath": res["n_bundles_multipath"],
                         "n_multipath_selected": res["n_multipath_selected"],
                         **r})
    _write_rows(rows, "disjoint_paths_comparison.csv")


def run_qlearning_experiments():
    """Wave 3 (item iii): tabular Q-learning router vs streaming annealer.

    Was previously only produced by qlearning_router.py's own __main__
    block despite fig_qlearning() depending on its CSV; wired in here so
    it runs as part of the standard sweep."""
    from baselines.qlearning_router import run_rl_comparison

    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                              memory_capacity=10, latency=5.0)
    run_rl_comparison(topo_fn, n_slots=10, mean_rate=1.0, episodes=10,
                      seed=42, out_dir=OUT)


def run_rl_generalization_experiments():
    """Future work: function-approximation RL that generalizes across
    topologies. Trains one LinearQRouter on a chain topology, then
    zero-shot evaluates the frozen weights on ring/grid/scale-free
    topologies it never trained on, against an in-distribution tabular
    control and the streaming annealer baseline."""
    from baselines.qlearning_router import run_topology_generalization_study
    from extensions.topologies import (
        generate_ring_topology, generate_erdos_renyi_topology,
        generate_barabasi_albert_topology,
    )

    train_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6, memory_capacity=10)
    eval_fns = {
        "ring_8": lambda: generate_ring_topology(n_nodes=8, edge_capacity=6, memory_capacity=10),
        "erdos_renyi_8": lambda: generate_erdos_renyi_topology(n_nodes=8, p=0.4,
                                                               edge_capacity=6, memory_capacity=10, seed=1),
        "barabasi_albert_8": lambda: generate_barabasi_albert_topology(n_nodes=8, m_links=2,
                                                                       edge_capacity=6, memory_capacity=10, seed=1),
    }
    res = run_topology_generalization_study(
        train_topology_fns=[train_fn], eval_topology_fns=eval_fns,
        n_slots=15, mean_rate=1.2, episodes=40, seed=42)
    _write_rows(res["rows"], "rl_topology_generalization.csv")


def run_swapping_order_experiments():
    """Extension 17 (E17): swapping-strategy order effects."""
    from extensions.swapping_order import (
        run_swapping_order_sweep, run_path_fidelity_sweep,
        run_swapping_bundle_comparison,
    )

    _write_rows(run_swapping_order_sweep(path_lengths=[3, 4, 5, 6, 7, 8],
                                         n_trials=40, seed=42),
                "swapping_order_sweep.csv")
    _write_rows(run_path_fidelity_sweep(path_length=10, link_fidelity=0.85),
                "swapping_path_fidelity.csv")

    topo = generate_chain_topology(n_nodes=12, edge_capacity=6,
                                   memory_capacity=10, raw_fidelity=0.85)
    rows = []
    res = run_swapping_bundle_comparison(topo, n_requests=8, tau_mem=3.0,
                                         seed=42)
    for r in res["rows"]:
        rows.append({**r})
    _write_rows(rows, "swapping_bundle_comparison.csv")


def run_topology_evolution_experiments():
    """Extension 13 (E13): does the optimizer advantage survive topology
    shape changes?  Matched-size, matched-density families."""
    from extensions.topologies import (
        generate_ring_topology, generate_random_geometric_topology,
        generate_erdos_renyi_topology, generate_watts_strogatz_topology,
        generate_barabasi_albert_topology, topology_sweep,
    )

    topo_fns = {
        "chain": lambda: generate_chain_topology(n_nodes=12, edge_capacity=6,
                                                 memory_capacity=10),
        "ring": lambda: generate_ring_topology(n_nodes=12),
        "random_geometric": lambda: generate_random_geometric_topology(
            n_nodes=12, radius=0.35),
        "erdos_renyi": lambda: generate_erdos_renyi_topology(n_nodes=12, p=0.22),
        "watts_strogatz": lambda: generate_watts_strogatz_topology(n_nodes=12),
        "barabasi_albert": lambda: generate_barabasi_albert_topology(n_nodes=12),
    }
    rows = []
    for n_requests in [4, 8]:
        rows.extend(topology_sweep(topo_fns, n_requests=n_requests, seed=42))
    _write_rows(rows, "topology_evolution.csv")


def run_des_reliability_experiments():
    """Wave 3: discrete-event stochastic simulation of solver plans.

    Evaluate Metropolis plans through the event engine under a
    tau_mem x swap_success grid; report sampled vs parametric utility
    and SLA statistics."""
    from experiments.benchmark import build_metropolis
    from simulation.discrete_event_engine import StochasticEventSimulator

    rows = []
    for n_req in [6, 10]:
        topo = generate_chain_topology(n_nodes=10, edge_capacity=8,
                                       memory_capacity=12, raw_fidelity=0.85)
        inst = contention_sweep_instances(lambda: topo, [n_req],
                                          seed=42)[f"req{n_req}"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
        sf = build_metropolis(b, ec, mc, seed=42)
        r = sf(
               max_iterations=3000)
        for tau_mem in [5.0, 50.0]:
            for swap_success in [0.90, 0.95, 1.0]:
                sim = StochasticEventSimulator(topo, tau_mem=tau_mem,
                                               swap_success=swap_success, seed=42)
                agg = sim.simulate_plan(b, r["selected"], n_realizations=40)
                rows.append({
                    "n_requests": n_req,
                    "tau_mem": tau_mem,
                    "swap_success": swap_success,
                    "n_selected": len(r["selected"]),
                    "param_expected_utility": agg["param_expected_utility"],
                    "e_utility": agg["e_utility"],
                    "utility_gap": agg["utility_gap"],
                    "served_ratio": agg["served_ratio"],
                    "sla_violation_prob": agg["sla_violation_prob"],
                    "e_delivered_fidelity": agg["e_delivered_fidelity"],
                    "e_latency": agg["e_latency"],
                    "n_realizations": agg["n_realizations"],
                })
    _write_rows(rows, "des_reliability.csv")


def run_purification_experiments():
    """Wave 3: purification as a first-class optimization variable.

    Joint fidelity/memory purification scheduling vs entanglement-only
    provisioning, plus a path-length / fidelity cost trade-off sweep."""
    from optimization.purification_scheduler import (
        run_purification_comparison, run_purification_sweep,
    )

    rows = []
    for n_req in [6, 10]:
        topo = generate_chain_topology(n_nodes=10, edge_capacity=6,
                                       memory_capacity=10, raw_fidelity=0.85)
        res = run_purification_comparison(topo, n_requests=n_req, q_max=4, seed=42)
        rows.extend([{"n_requests": n_req, **r} for r in res["rows"]])
    _write_rows(rows, "purification_comparison.csv")
    _write_rows(run_purification_sweep(path_lengths=[3, 4, 5, 6], q_max=4,
                                       min_fidelity_values=[0.5, 0.7, 0.9]),
                "purification_sweep.csv")


def run_recourse_experiments():
    """Wave 3: adaptive recourse — local repair versus full reoptimization.

    Compare how often each failed request is recovered by local repair
    versus full replan, and the wall-time speedup."""
    from simulation.recourse import run_recourse_comparison

    rows, summary = [], []
    for n_req in [6, 10]:
        topo = generate_chain_topology(n_nodes=10, edge_capacity=6,
                                       memory_capacity=10, raw_fidelity=0.85)
        res = run_recourse_comparison(topo, n_requests=n_req, n_realizations=20,
                                      seed=42, tau_mem=50.0, swap_success=0.95)
        rows.extend([{"n_requests": n_req, **r} for r in res["rows"]])
        summary.append({
            "n_requests": n_req,
            "n_realizations": res["n_realizations"],
            "plan0_utility": res["plan0_utility"],
            "plan0_time_s": res["plan0_time_s"],
            "mean_failed_per_realization": res["mean_failed_per_realization"],
            "mean_t_local_s": res["mean_t_local_s"],
            "mean_t_full_s": res["mean_t_full_s"],
            "speedup": res["speedup"],
            "mean_u_local": res["mean_u_local"],
            "mean_u_full": res["mean_u_full"],
            "mean_recovery_rate": res["mean_recovery_rate"],
            "lost_utility_recovered_frac": res["lost_utility_recovered_frac"],
        })
    _write_rows(rows, "recourse_comparison.csv")
    _write_rows(summary, "recourse_summary.csv")


def run_optimality_experiments():
    """Wave 3: optimality-gap certification and stochastic reliability.

    Exact-ILP gap study across request counts, and parametric vs
    sampled reliability of solver plans under fidelity decay."""
    from experiments.optimality_benchmark import (
        run_gap_study, run_stochastic_reliability_benchmark,
    )

    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                              memory_capacity=10)
    gap_rows = run_gap_study(topo_fn, sizes=[4, 6, 8], n_instances=4,
                             seed=42, time_limit=30.0)
    _write_rows(gap_rows, "optimality_gap.csv")

    rows = []
    for n_req in [6, 10]:
        topo = generate_chain_topology(n_nodes=10, edge_capacity=8,
                                       memory_capacity=12, raw_fidelity=0.85)
        res = run_stochastic_reliability_benchmark(topo, n_requests=n_req,
                                                   n_realizations=40, seed=42,
                                                   tau_mem=50.0, swap_success=0.95)
        rows.extend(res["rows"])
    _write_rows(rows, "stochastic_reliability.csv")


def run_adaptive_budget_experiments():
    """Wave 3: adaptive QUBO candidate budget.

    Congestion/density-driven top-k budget vs fixed budgets and the
    full-candidate QUBO reference."""
    from optimization.adaptive_budget import run_adaptive_budget_study

    topo_fn = lambda: generate_chain_topology(n_nodes=10, edge_capacity=6,
                                              memory_capacity=10)
    res = run_adaptive_budget_study(topo_fn, n_requests_list=[8, 12],
                                    k_values=[2, 4, 6, 8], num_reads=20,
                                    solver="metropolis", seed=42)
    _write_rows(res["rows"], "adaptive_budget.csv")


def run_quantum_annealing_experiments():
    """Wave 3: quantum-annealing backend comparison.

    Minor-embedded QA (PIA sampler) vs simulated annealing and SQA on
    identical QUBO instances; reports qubits used and chain statistics."""
    from optimization.quantum_annealing_backend import run_quantum_annealing_sweep

    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                              memory_capacity=10)
    res = run_quantum_annealing_sweep(topo_fn, n_requests_list=[4, 6, 8],
                                      num_reads=20, num_sweeps=200, k=8, seed=42)
    _write_rows(res["rows"], "quantum_annealing_sweep.csv")


def run_hardware_literature_validation_experiments():
    """Future work: literature-calibrated hardware validation.

    NOT a physical hardware experiment -- re-parametrizes HardwareProfile
    from real, cited, published measured T1/T2/gate/readout numbers and
    checks the simulated fidelity-decay trend ranks consistently with the
    cited sources' T2 ordering."""
    from hardware.literature_validation import run_literature_calibration_report

    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                              memory_capacity=10, latency=5.0)
    res = run_literature_calibration_report(topo_fn, n_slots=15, mean_rate=1.0, seed=7)
    _write_rows(res["rows"], "hardware_literature_validation.csv")


def run_quantum_annealing_topology_experiments():
    """Pegasus/Zephyr hardware-lattice support.

    Extends the Chimera-only QA backend comparison with two more hardware
    lattices (Pegasus-like, Zephyr-like, both via dwave.graphs), reporting
    qubit count, chain-break fraction and utility per topology per problem
    size."""
    from optimization.quantum_annealing_backend import run_quantum_annealing_topology_sweep

    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                              memory_capacity=10)
    res = run_quantum_annealing_topology_sweep(
        topo_fn, n_requests_list=[4, 6, 8],
        topologies=("chimera", "pegasus", "zephyr"),
        num_reads=20, num_sweeps=200, k=8, seed=42)
    _write_rows(res["rows"], "quantum_annealing_topology_sweep.csv")


def run_chance_constrained_experiments():
    """Wave 3: SLA-calibration frontier of chance-constrained routing.

    Deterministic fidelity constraints implicitly tolerate up to ~1/2
    violation probability for razor-margin bundles; quantile (chance)
    constraints F >= F_min + z_eps*sigma deliver the requested reliability
    at a utility price, and eps > 1/2 trades reliability for capacity.
    Each policy is solved by exact ILP and executed in the discrete-event
    engine with matching fidelity noise."""
    from optimization.chance_constrained import run_chance_constrained_study

    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                              memory_capacity=10,
                                              raw_fidelity=0.85,
                                              generation_prob=0.8)
    res = run_chance_constrained_study(
        topo_fn, n_requests_list=[6, 10],
        eps_list=[0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
        sigma_list=[0.03, 0.05], n_instances=3, n_realizations=40,
        time_limit=30.0, seed=42)
    _write_rows(res["rows"], "chance_constrained.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("  Paper experiment sweep")
    print("=" * 60)

    print("\n1. Contention sweep (chain, vary cap/n_nodes)...")
    run_contention_sweep()

    print("\n2. Bond dimension sweep...")
    run_bond_dim_sweep()

    print("\n3. Grid comparison...")
    run_grid_comparison()

    print("\n4. Streaming vs batched comparison...")
    run_streaming_comparison()

    print("\n5. Hamiltonian encoding comparison (slack vs direct)...")
    run_hamiltonian_encoding_comparison()

    print("\n6. Selfish routing (D1): Pigou PoA + Braess paradox...")
    run_selfish_routing_experiments()

    print("\n7. Robust routing (D2): nominal vs maximin vs min-max-regret...")
    run_robust_routing_experiments()

    print("\n8. Joint scheduling (E1): routing + temporal memory vs memory-agnostic...")
    run_joint_scheduling_experiments()

    print("\n9. Adaptive QUBO (E7): candidate-space reduction top-k sweep...")
    run_adaptive_qubo_experiments()

    print("\n10. Hybrid pipeline (E6): full pipeline vs ablations...")
    run_hybrid_pipeline_experiments()

    print("\n11. GNN-guided reduction (E8): GNN vs adaptive top-k vs full QUBO...")
    run_gnn_experiments()

    print("\n12. Multi-objective (E9): Pareto frontier + constraint queries...")
    run_multi_objective_experiments()

    print("\n13. k-disjoint paths (E11): single vs multipath provisioning...")
    run_disjoint_paths_experiments()

    print("\n14. Swapping-order (E17): depth / coherence / concurrency...")
    run_swapping_order_experiments()

    print("\n15. Topology evolution (E13): chain vs ring vs G(n,p) vs small-world vs scale-free...")
    run_topology_evolution_experiments()

    print("\n16. DES (Wave 3): stochastic reliability of solver plans...")
    run_des_reliability_experiments()

    print("\n17. Purification (Wave 3): purification as a first-class variable...")
    run_purification_experiments()

    print("\n18. Adaptive recourse (Wave 3): local repair vs full reoptimization...")
    run_recourse_experiments()

    print("\n19. Optimality certification (Wave 3): exact-ILP gap + stochastic reliability...")
    run_optimality_experiments()

    print("\n20. Adaptive budget (Wave 3): candidate-space budget control...")
    run_adaptive_budget_experiments()

    print("\n21. Quantum annealing backend (Wave 3): embedded QA vs SA/SQA...")
    run_quantum_annealing_experiments()

    print("\n22. Chance-constrained routing (Wave 3): SLA-calibration frontier...")
    run_chance_constrained_experiments()

    print("\n23. Quantum annealing topologies : Chimera vs Pegasus vs Zephyr...")
    run_quantum_annealing_topology_experiments()

    print("\n24. Hardware literature validation: re-calibrated profiles vs cited sources...")
    run_hardware_literature_validation_experiments()

    print("\n25. Adaptive bond dimension: truncation-error-driven chi...")
    run_adaptive_bond_dim_sweep()

    print("\n26. MPS ordering: greedy vs spectral coupling order...")
    run_mps_ordering_experiments()

    print("\n27. Q-learning router (Wave 3, item iii): tabular vs streaming annealer...")
    run_qlearning_experiments()

    print("\n28. RL topology generalization (future work): zero-shot transfer...")
    run_rl_generalization_experiments()

    print(f"\nAll results in {OUT}/")



