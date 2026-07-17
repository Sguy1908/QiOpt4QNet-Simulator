#path generations
import sys
import os

M = 3  # Number of rows in the heavy hex lattice
N = 4  # Number of columns in the heavy hex lattice

# Setup paths so it can find the src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.insert(0, src_dir)

from network.topology import generate_chain
from network.request import Request
from routing.path_generator import PathGenerator
from network.topology import generate_heavy_hex

def main():
    print("Generating a heavy hex topology...")
    # A chain looks like: Node_0 -- Node_1 -- Node_2 -- Node_3 -- Node_4
    network = generate_heavy_hex(M, N)
    
    # Node 0 wants to establish entanglement with Node 4
    req = Request(source="Node_H_0_0", destination="Node_H_4_7", minimum_fidelity=0.8, weight=1.0)
    print(f"\nCreated Request: {req}")
    
    print("\nGenerating candidate paths...")
    path_gen = PathGenerator(network)
    
    # Find up to 3 shortest paths (since it's a chain, it will only find 1)
    paths = path_gen.get_k_shortest_paths(req, k=3)
    
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {' -> '.join(path)}")
        
    network.visualize()

if __name__ == '__main__':
    main()
