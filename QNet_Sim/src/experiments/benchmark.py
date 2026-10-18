import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Optional

from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
try:
    from optimization.openjij_solver import solve_sa, solve_sqa
    _HAS_OPENJIJ = True
except ImportError:
    _HAS_OPENJIJ = False

try:
    from optimization.qubo_optimizer import QUBOOptimizer
    _HAS_PYQUBO = True
except ImportError:
    _HAS_PYQUBO = False

from experiments.instances import (
    generate_chain_topology, contention_sweep_instances,
)
from experiments.metrics import ExperimentTracker


def build_metropolis(bundles, edge_caps, mem_caps, seed=42,
                     congestion_weight=0.0, congestion_threshold=0.7,
                     risk_weight=0.0, risk_tau=1.0):
    opt = MetropolisAnnealer(
        bundles, edge_caps, mem_caps, seed=seed,
        congestion_weight=congestion_weight,
        congestion_threshold=congestion_threshold,
        risk_weight=risk_weight, risk_tau=risk_tau,
    )
    return lambda **kw: opt.solve(**kw)


def build_tensor_network(bundles, edge_caps, mem_caps,
                         congestion_weight=0.0, congestion_threshold=0.7,
                         risk_weight=0.0, risk_tau=1.0):
    opt = TensorNetworkOptimizer(
        bundles, edge_caps, mem_caps,
        congestion_weight=congestion_weight,
        congestion_threshold=congestion_threshold,
        risk_weight=risk_weight, risk_tau=risk_tau,
    )
    return lambda **kw: opt.solve(**kw)


def build_openjij_sa(bundles, edge_caps, mem_caps, num_reads=50):
    if not _HAS_OPENJIJ:
        raise ImportError("openjij is not installed")
    if not _HAS_PYQUBO:
        raise ImportError("pyqubo is not installed")
    optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)
    bqm = optimizer.to_bqm(congestion_penalty=0.05, memory_congestion_penalty=0.05)
    return lambda **kw: _run_openjij(optimizer, bqm, "sa", num_reads, kw.get("seed", None))


def build_openjij_sqa(bundles, edge_caps, mem_caps, num_reads=50):
    if not _HAS_OPENJIJ:
        raise ImportError("openjij is not installed")
    if not _HAS_PYQUBO:
        raise ImportError("pyqubo is not installed")
    optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)
    bqm = optimizer.to_bqm(congestion_penalty=0.05, memory_congestion_penalty=0.05)
    return lambda **kw: _run_openjij(optimizer, bqm, "sqa", num_reads, kw.get("seed", None))


def _run_openjij(optimizer, bqm, solver_type, num_reads, seed):
    if solver_type == "sa":
        response = solve_sa(bqm, num_reads=num_reads, seed=seed)
    else:
        response = solve_sqa(bqm, num_reads=num_reads, seed=seed)
    sample = response.first.sample
    selected = optimizer.decode_sample(sample)
    energy = response.first.energy

    selections = {}
    for rid, bid in selected:
        selections[rid] = bid
    for b in optimizer.bundles:
        rid = b["request_id"]
        if rid not in selections:
            selections[rid] = None

    return {
        "selected": selected,
        "energy": energy,
        "selections": selections,
    }


def run_single_benchmark(tracker: ExperimentTracker, bundles, edge_caps, mem_caps,
                         instance_name: str, seed: int = 42,
                         congestion_weight=0.0, congestion_threshold=0.7,
                         risk_weight=0.0, risk_tau=1.0):
    solvers = {
        "Metropolis-SA": build_metropolis(
            bundles, edge_caps, mem_caps, seed=seed,
            congestion_weight=congestion_weight,
            congestion_threshold=congestion_threshold,
            risk_weight=risk_weight, risk_tau=risk_tau,
        ),
        "TensorNetwork": build_tensor_network(
            bundles, edge_caps, mem_caps,
            congestion_weight=congestion_weight,
            congestion_threshold=congestion_threshold,
            risk_weight=risk_weight, risk_tau=risk_tau,
        ),
    }
    if _HAS_OPENJIJ:
        solvers["OpenJij-SA"] = build_openjij_sa(bundles, edge_caps, mem_caps, num_reads=100)
        solvers["OpenJij-SQA"] = build_openjij_sqa(bundles, edge_caps, mem_caps, num_reads=100)

    for name, solver_fn in solvers.items():
        solver_kwargs = {
            "penalty": 100.0,
            "edge_penalty": 10.0,
            "memory_penalty": 10.0,
            "max_iterations": 5000,
            "initial_temperature": 10.0,
            "cooling_rate": 0.97,
        } if "Metropolis" in name else (
            {"bond_dim": 8, "beta": 5.0, "edge_penalty": 10.0, "memory_penalty": 10.0}
            if "Tensor" in name else
            {"seed": seed}
        )
        tracker.run_solver(
            solver_fn, bundles, edge_caps, mem_caps,
            solver_name=name, instance_name=instance_name,
            **solver_kwargs,
        )


def run_contention_benchmark(n_requests_list: Optional[List[int]] = None,
                             seed: int = 42) -> ExperimentTracker:
    if n_requests_list is None:
        n_requests_list = [2, 4, 8, 12, 16]

    topology_fn = lambda: generate_chain_topology(
        n_nodes=8, edge_capacity=6, memory_capacity=10,
        raw_fidelity=0.85, latency=5.0,
    )
    instances = contention_sweep_instances(topology_fn, n_requests_list, seed=seed)

    tracker = ExperimentTracker()
    for name, inst in instances.items():
        print(f"  Benchmarking {name} ({inst['n_requests']} requests, "
              f"{len(inst['bundles'])} bundles)...")
        run_single_benchmark(tracker, inst["bundles"], inst["edge_capacities"],
                             inst["memory_capacities"], name, seed=seed)
    return tracker


def run_hamiltonian_comparison(seed: int = 42) -> ExperimentTracker:
    topology_fn = lambda: generate_chain_topology(
        n_nodes=6, edge_capacity=8, memory_capacity=12,
    )
    instances = contention_sweep_instances(topology_fn, [4, 8, 12], seed=seed)

    tracker = ExperimentTracker()
    for name, inst in instances.items():
        print(f"  Instance {name}...")

        base_kwargs = {"penalty": 100.0, "edge_penalty": 10.0, "memory_penalty": 10.0,
                       "max_iterations": 5000, "initial_temperature": 10.0,
                       "cooling_rate": 0.97}

        for cong_w, cong_t in [(0.0, 0.7), (5.0, 0.7), (10.0, 0.5)]:
            for risk_w, risk_t in [(0.0, 1.0), (2.0, 5.0)]:
                label = f"Metropolis(cong={cong_w},thr={cong_t},risk={risk_w},τ={risk_t})"
                sf = build_metropolis(
                    inst["bundles"], inst["edge_capacities"], inst["memory_capacities"],
                    seed=seed, congestion_weight=cong_w,
                    congestion_threshold=cong_t, risk_weight=risk_w, risk_tau=risk_t,
                )
                tracker.run_solver(
                    sf, inst["bundles"], inst["edge_capacities"],
                    inst["memory_capacities"],
                    solver_name=label, instance_name=name,
                    **base_kwargs,
                )
    return tracker


def run_scaling_benchmark(topology_fn, n_requests_list: List[int],
                          bond_dims: List[int], seed: int = 42) -> ExperimentTracker:
    instances = contention_sweep_instances(topology_fn, n_requests_list, seed=seed)
    tracker = ExperimentTracker()

    for name, inst in instances.items():
        print(f"  Scaling {name} ({inst['n_requests']} requests)...")
        for bd in bond_dims:
            tn_fn = build_tensor_network(
                inst["bundles"], inst["edge_capacities"], inst["memory_capacities"],
            )
            tn_kwargs = {"bond_dim": bd, "beta": 5.0, "edge_penalty": 10.0,
                         "memory_penalty": 10.0}
            tracker.run_solver(
                tn_fn, inst["bundles"], inst["edge_capacities"],
                inst["memory_capacities"],
                solver_name=f"TN(χ={bd})", instance_name=name,
                **tn_kwargs,
            )

        meta_fn = build_metropolis(
            inst["bundles"], inst["edge_capacities"], inst["memory_capacities"],
            seed=seed,
        )
        meta_kwargs = {"penalty": 100.0, "edge_penalty": 10.0, "memory_penalty": 10.0,
                       "max_iterations": 5000, "initial_temperature": 10.0,
                       "cooling_rate": 0.97}
        tracker.run_solver(
            meta_fn, inst["bundles"], inst["edge_capacities"],
            inst["memory_capacities"],
            solver_name="Metropolis-SA", instance_name=name,
            **meta_kwargs,
        )
    return tracker


RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiments"))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("  QiOpt4QNet Benchmark Suite")
    print("=" * 60)

    print("\n--- Benchmark 1: Contention Scaling (Chain, 8 nodes) ---")
    t1 = run_contention_benchmark(n_requests_list=[2, 4, 8, 12, 16], seed=42)
    print("\nContention Benchmark Summary:")
    print(t1.table())
    t1.to_csv(os.path.join(RESULTS_DIR, "contention_benchmark.csv"))

    print("\n--- Benchmark 2: Hamiltonian Term Comparison ---")
    t2 = run_hamiltonian_comparison(seed=42)
    print("\nHamiltonian Comparison Summary:")
    print(t2.table())
    t2.to_csv(os.path.join(RESULTS_DIR, "hamiltonian_comparison.csv"))

    print("\n--- Benchmark 3: TN Scaling (χ sweep) ---")
    topo_fn = lambda: generate_chain_topology(n_nodes=10, edge_capacity=6)
    t3 = run_scaling_benchmark(topo_fn, [4, 8, 16], bond_dims=[2, 4, 8, 16], seed=42)
    print("\nScaling Benchmark Summary:")
    print(t3.table())
    t3.to_csv(os.path.join(RESULTS_DIR, "scaling_benchmark.csv"))

    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
