#topology and network creation
import sys
import os

M = 3  # Number of rows in the heavy hex lattice
N = 4  # Number of columns in the heavy hex lattice

# 1. Get the directory containing the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with '../src' and make the result absolute
src_dir = os.path.abspath(os.path.join(current_dir, '../src'))

# 3. Add to sys.path
sys.path.insert(0, src_dir)

from network.topology import generate_chain
from network.topology import generate_heavy_hex

def main():
    print("Generating a chain topology of 5 nodes...")
    # network = generate_chain(5)
    network = generate_heavy_hex(M, N)

    print("\nNodes in the network:")
    network.display_nodes()

    print("\nLinks in the network:")
    network.display_links()

    print("\nVisualizing network...")
    network.visualize()

if __name__ == '__main__':
    main()