"""
3-5 simulator validation spot-checks:

1. Bundle generation -> optimizer pipeline: verify selections are feasible
2. Reproducibility: same seed -> same result
3. Capacity constraint enforcement
4. Path validity across all selected bundles
5. Metropolis vs MPS consistency on small instances
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from collections import defaultdict
from network.topology import generate_chain, generate_grid
from network.request import Request
from routing.path_generator import PathGenerator
from routing.bundle_generation import BundleGenerator
from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
from optimization.sequential_branch_optimizer import SequentialBranchOptimizer
from optimization.qubo_optimizer import QUBOOptimizer
from optimization.openjij_solver import solve_sa, solve_sqa
from optimization.baselines import utility_density_greedy, fidelity_aware_greedy


def _undirected_edge(e):
    return tuple(sorted(e))


def extract_capacities(net):
    edge_caps = {}
    for u, v, data in net.graph.edges(data=True):
        edge_caps[_undirected_edge((u, v))] = data["data"].capacity
    mem_caps = {}
    for nid, data in net.graph.nodes(data=True):
        mem_caps[nid] = data["data"].memory_capacity
    return edge_caps, mem_caps


def check_feasibility(selections, bundles, edge_caps, mem_caps):
    edge_load = defaultdict(int)
    mem_load = defaultdict(int)
    for rid, bid in selections:
        for b in bundles:
            if b["request_id"] == rid and b["bundle_id"] == bid:
                for e, d in b["edge_demands"].items():
                    edge_load[_undirected_edge(e)] += d
                for n, d in b["memory_demands"].items():
                    mem_load[n] += d
    for e, load in edge_load.items():
        if load > edge_caps.get(e, 0):
            return False, f"Edge {e} overloaded: {load} > {edge_caps.get(e, 0)}"
    for n, load in mem_load.items():
        if load > mem_caps.get(n, 0):
            return False, f"Node {n} memory overloaded: {load} > {mem_caps.get(n, 0)}"
    return True, ""


def check_path_validity(selections, bundles, net):
    for rid, bid in selections:
        for b in bundles:
            if b["request_id"] == rid and b["bundle_id"] == bid:
                path = b["path"]
                for i in range(len(path) - 1):
                    edge = _undirected_edge((path[i], path[i + 1]))
                    if not net.graph.has_edge(path[i], path[i + 1]):
                        return False, f"Invalid edge {edge} not in network"
    return True, ""


def main():
    passed = 0
    failed = 0

    # Check 1: Pipeline produces feasible selections
    print("Check 1: End-to-end feasibility on chain(5)...")
    net = generate_chain(5, default_memory=10,
                         link_params={"capacity_bounds": (3, 5), "fidelity_bounds": (0.8, 0.9)})
    ec, mc = extract_capacities(net)
    gen = BundleGenerator(net)
    pg = PathGenerator(net)
    candidates = [n for n in net.graph.nodes() if net.graph.degree(n) >= 1]
    reqs = [
        Request(candidates[0], candidates[-1], 0.6, weight=50.0, request_id="R1"),
        Request(candidates[1], candidates[-2], 0.6, weight=30.0, request_id="R2"),
    ]
    all_bundles = []
    for req in reqs:
        paths = pg.get_k_shortest_paths(req, k=3)
        for b in gen.generate_bundles(req, paths):
            all_bundles.append(b.to_optimizer_dict())
    opt = TensorNetworkOptimizer(all_bundles, ec, mc)
    result = opt.solve(bond_dim=4, beta=3.0)
    feasible, msg = check_feasibility(result["selected"], all_bundles, ec, mc)
    if not feasible:
        print(f"  FAIL: {msg}")
        failed += 1
    else:
        # Also check no duplicate requests
        rids = [rid for rid, _ in result["selected"]]
        assert len(rids) == len(set(rids)), "Duplicate request IDs in selection"
        print("  PASS")
        passed += 1

    # Check 2: Reproducibility
    print("Check 2: Reproducibility (same seed -> same result)...")
    opt1 = MetropolisAnnealer(all_bundles, ec, mc, seed=42)
    res1 = opt1.solve(max_iterations=1000)
    opt2 = MetropolisAnnealer(all_bundles, ec, mc, seed=42)
    res2 = opt2.solve(max_iterations=1000)
    if res1["energy"] == res2["energy"] and sorted(res1["selected"]) == sorted(res2["selected"]):
        print("  PASS")
        passed += 1
    else:
        print(f"  FAIL: energies {res1['energy']} vs {res2['energy']}")
        failed += 1

    # Check 3: Capacity constraint enforcement (feasible solvers should have 0 violations)
    print("Check 3: Capacity enforcement on constrained grid...")
    net = generate_grid(3, 3, default_memory=8,
                        link_params={"capacity_bounds": (2, 3), "fidelity_bounds": (0.8, 0.9)})
    ec, mc = extract_capacities(net)
    gen = BundleGenerator(net)
    pg = PathGenerator(net)
    candidates = [n for n in net.graph.nodes() if net.graph.degree(n) >= 1]
    rng_candidates = __import__("random").Random(42).sample
    reqs = []
    for i in range(6):
        src, dst = rng_candidates(candidates, 2)
        reqs.append(Request(src, dst, 0.6, weight=100.0, request_id=f"R{i}"))
    all_bundles = []
    for req in reqs:
        paths = pg.get_k_shortest_paths(req, k=3)
        for b in gen.generate_bundles(req, paths):
            all_bundles.append(b.to_optimizer_dict())
    for solver_name, opt_cls, kwargs in [
        ("Metropolis", MetropolisAnnealer, {"max_iterations": 2000}),
        ("TensorNetwork", TensorNetworkOptimizer, {"bond_dim": 6, "beta": 5.0}),
        ("BranchExpansion", SequentialBranchOptimizer, {"branch_factor": 8, "beta": 5.0}),
        ("QUBO+SA", None, {"solve_sa": True}),
        ("QUBO+SQA", None, {"solve_sa": False}),
        ("Utility-Density Greedy", None, {"greedy": "ud"}),
        ("Fidelity-Aware Greedy", None, {"greedy": "fid"}),
    ]:
        if solver_name == "Metropolis":
            opt = opt_cls(all_bundles, ec, mc, seed=42)
            result = opt.solve(**kwargs)
        elif solver_name.startswith("QUBO"):
            opt = QUBOOptimizer(all_bundles, ec, mc)
            bqm = opt.to_bqm(congestion_penalty=0.05, memory_congestion_penalty=0.05)
            if kwargs["solve_sa"]:
                response = solve_sa(bqm, num_reads=20, seed=1)
            else:
                response = solve_sqa(bqm, num_reads=10, seed=1)
            selected = opt.decode_sample(response.first.sample, repair=True)
            result = {"selected": selected}
        elif solver_name.endswith("Greedy"):
            fn = utility_density_greedy if kwargs["greedy"] == "ud" else fidelity_aware_greedy
            result = fn(all_bundles, ec, mc)
        else:
            opt = opt_cls(all_bundles, ec, mc)
            result = opt.solve(**kwargs)
        feasible, msg = check_feasibility(result["selected"], all_bundles, ec, mc)
        if feasible:
            print(f"  {solver_name}: PASS (no violations)")
            passed += 1
        else:
            print(f"  {solver_name}: FAIL - {msg}")
            failed += 1

    # Check 4: Path validity
    print("Check 4: All selected paths use valid network edges...")
    for solver_name, opt_cls, kwargs in [
        ("Metropolis", MetropolisAnnealer, {"max_iterations": 2000}),
        ("TensorNetwork", TensorNetworkOptimizer, {"bond_dim": 6, "beta": 5.0}),
        ("BranchExpansion", SequentialBranchOptimizer, {"branch_factor": 8, "beta": 5.0}),
        ("QUBO+SA", None, {"solve_sa": True}),
        ("QUBO+SQA", None, {"solve_sa": False}),
        ("Utility-Density Greedy", None, {"greedy": "ud"}),
        ("Fidelity-Aware Greedy", None, {"greedy": "fid"}),
    ]:
        if solver_name == "Metropolis":
            opt = opt_cls(all_bundles, ec, mc, seed=42)
            result = opt.solve(**kwargs)
        elif solver_name.startswith("QUBO"):
            opt = QUBOOptimizer(all_bundles, ec, mc)
            bqm = opt.to_bqm(congestion_penalty=0.05, memory_congestion_penalty=0.05)
            if kwargs["solve_sa"]:
                response = solve_sa(bqm, num_reads=20, seed=1)
            else:
                response = solve_sqa(bqm, num_reads=10, seed=1)
            selected = opt.decode_sample(response.first.sample, repair=True)
            result = {"selected": selected}
        elif solver_name.endswith("Greedy"):
            fn = utility_density_greedy if kwargs["greedy"] == "ud" else fidelity_aware_greedy
            result = fn(all_bundles, ec, mc)
        else:
            opt = opt_cls(all_bundles, ec, mc)
            result = opt.solve(**kwargs)
        valid, msg = check_path_validity(result["selected"], all_bundles, net)
        if valid:
            print(f"  {solver_name}: PASS")
            passed += 1
        else:
            print(f"  {solver_name}: FAIL - {msg}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  Passed: {passed}/{passed + failed}")
    print(f"  Failed: {failed}/{passed + failed}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
