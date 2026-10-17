import sys
import os

# Ensure src is in the path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
sys.path.insert(0, src_dir)

from network.network import QuantumNetwork
from network.link import QuantumLink
from network.node import QuantumNode
from network.request import Request
from routing.bundle_generation import BundleGenerator
from optimization.qubo_optimizer import QUBOOptimizer
from optimization.openjij_solver import solve_sa

def main():
    print("==================================================")
    print("   QUANTUM BUNDLE GENERATION & QUBO DEMO          ")
    print("==================================================")

    # 1. Create Network & Links
    net = QuantumNetwork()
    net.add_node(QuantumNode("Alice", memory_capacity=10))
    net.add_node(QuantumNode("Repeater", memory_capacity=10))
    net.add_node(QuantumNode("Bob", memory_capacity=10))

    # Add fiber links (distance=10km, gen_prob=1.0, raw_fidelity=0.85, latency=5.0ms, capacity=6)
    net.add_link(QuantumLink("Alice", "Repeater", 10, 1.0, 0.85, 5.0, 6))
    net.add_link(QuantumLink("Repeater", "Bob", 10, 1.0, 0.85, 5.0, 6))

    # 2. Create Request & Generate Bundles (with higher weight so utility is positive)
    req = Request("Alice", "Bob", minimum_fidelity=0.6, weight=50.0, request_id="Req_1")
    gen = BundleGenerator(net)
    bundles = gen.generate_bundles(req, [["Alice", "Repeater", "Bob"]])

    print(f"\n---> STEP 1: GENERATED {len(bundles)} BUNDLES FOR {req.request_id} <---")
    for b in bundles:
        print(f"\n[{b.bundle_id}] (Purification Rounds: q={b.purification_rounds})")
        print(f"  * End-to-End Fidelity:  {b.fidelity:.4f}")
        print(f"  * Bell Pair Cost:       {b.bell_pair_cost} pairs")
        print(f"  * Success Probability:  {b.success_probability:.4f}")
        print(f"  * Utility Score (U):    {b.utility:.4f}")
        print(f"  * Memory Demands:       {b.memory_demands}")

    # 3. Export to Optimizer Format
    print("\n---> STEP 2: FEEDING TO QUBO OPTIMIZER <---")
    opt_bundles = [b.to_optimizer_dict() for b in bundles]

    # Define hardware limits (capacities)
    edge_caps = {("Alice", "Repeater"): 6, ("Repeater", "Bob"): 6}
    mem_caps = {"Alice": 10, "Repeater": 10, "Bob": 10}

    optimizer = QUBOOptimizer(opt_bundles, edge_caps, mem_caps)
    print(f"Bundles loaded into optimizer: {len(optimizer.bundles)}")

    # Build BQM and solve using OpenJij Simulated Annealing (SA)
    bqm = optimizer.to_bqm(congestion_penalty=0.05, memory_congestion_penalty=0.05)
    sa_response = solve_sa(bqm, num_reads=50)
    selected = optimizer.decode_sample(sa_response.first.sample)

    print("\n==================================================")
    print(f"   OPTIMIZER SELECTION RESULT: {selected}")
    print(f"   Best QUBO Energy: {sa_response.first.energy:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
