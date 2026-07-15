import sys
import os

# 1. Get the directory containing the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with '../src' and make the result absolute
src_dir = os.path.abspath(os.path.join(current_dir, '../src'))

# 3. Add to sys.path
sys.path.insert(0, src_dir)

from network.topology import generate_chain

def main():
    print("Generating a chain topology of 5 nodes...")
    network = generate_chain(5)

    print("\nNodes in the network:")
    network.display_nodes()

    print("\nLinks in the network:")
    network.display_links()

    print("\nVisualizing network...")
    network.visualize()

if __name__ == '__main__':
    main()