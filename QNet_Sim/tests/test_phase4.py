import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.insert(0, src_dir)

from optimization.qubo_optimizer import QUBOOptimizer
from optimization.openjij_solver import solve_sa, solve_sqa

def main():
    print("--- Testing QUBO Optimizer ---")

    #Hardcoded for now, will be replaced with generated bundles later
    valid_bundles = [
        {
            "bundle_id": "B1",
            "request_id": "R1",
            "path": ["Node_0", "Node_1", "Node_2"],
            "purification_profile": {
                ("Node_0", "Node_1"): 1,
                ("Node_1", "Node_2"): 0
            },
            "final_fidelity": 0.90,
            "success_probability": 0.75,
            "latency": 8.0,
            "bell_pair_cost": 3,
            "edge_demands": {
                ("Node_0", "Node_1"): 2,
                ("Node_1", "Node_2"): 1
            },
            "memory_demands": {
                "Node_0": 2,
                "Node_1": 3,
                "Node_2": 1
            },
            "utility": 1.5
        },
        {
            "bundle_id": "B2",
            "request_id": "R1",
            "path": ["Node_0", "Node_3", "Node_2"],
            "purification_profile": {
                ("Node_0", "Node_3"): 0,
                ("Node_3", "Node_2"): 0
            },
            "final_fidelity": 0.87,
            "success_probability": 0.82,
            "latency": 6.5,
            "bell_pair_cost": 2,
            "edge_demands": {
                ("Node_0", "Node_3"): 1,
                ("Node_3", "Node_2"): 1
            },
            "memory_demands": {
                "Node_0": 1,
                "Node_3": 2,
                "Node_2": 1
            },
            "utility": 1.2
        },
        {
            "bundle_id": "B3",
            "request_id": "R2",
            "path": ["Node_4", "Node_1", "Node_2"],
            "purification_profile": {
                ("Node_4", "Node_1"): 0,
                ("Node_1", "Node_2"): 0
            },
            "final_fidelity": 0.88,
            "success_probability": 0.78,
            "latency": 7.5,
            "bell_pair_cost": 2,
            "edge_demands": {
                ("Node_4", "Node_1"): 1,
                ("Node_1", "Node_2"): 1
            },
            "memory_demands": {
                "Node_4": 1,
                "Node_1": 2,
                "Node_2": 1
            },
            "utility": 1.3
        }
    ]
    
    edge_capacities = {
        ("Node_0", "Node_1"): 2,
        ("Node_1", "Node_2"): 1,
        ("Node_0", "Node_3"): 1,
        ("Node_2", "Node_3"): 1,
        ("Node_1", "Node_4"): 1
    }

    memory_capacities = {
        "Node_0": 2,
        "Node_1": 3,
        "Node_2": 2,
        "Node_3": 2,
        "Node_4": 1
    }

    optimizer = QUBOOptimizer(valid_bundles, edge_capacities, memory_capacities)
    print("Complete bundles passed")
    print(f"Bundles stored: {len(optimizer.bundles)}")

    qubo, offset = optimizer.to_qubo(1.21, 1.0, 1.0)
    bqm = optimizer.to_bqm(1.21, 1.0, 1.0)

    sa_response = solve_sa(bqm,100, 100)
    sqa_response = solve_sqa(bqm,100, 100)

    sa_selected = optimizer.decode_sample(sa_response.first.sample)
    sqa_selected = optimizer.decode_sample(sqa_response.first.sample)

    print(f"Request groups: {list(optimizer.bundles_by_request)}")
    print(f"Variables created: {len(optimizer.variables)}")
    print(f"Edges constrained: {len(optimizer.edge_demands)}")
    print(f"Nodes constrained: {len(optimizer.memory_demands)}")
    print(f"QUBO: {qubo}")
    print(f"Offset: {offset}")
    print(f"BQM variables: {len(bqm.variables)}")
    print(f"BQM variable type: {bqm.vartype}")

    print(f"SA best energy: {sa_response.first.energy}")
    print(f"SA best sample: {sa_response.first.sample}")
    print(f"SQA best energy: {sqa_response.first.energy}")
    print(f"SQA best sample: {sqa_response.first.sample}")
    
    print(f"SA selected bundles: {sa_selected}")
    print(f"SQA selected bundles: {sqa_selected}")

if __name__ == "__main__":
    main()