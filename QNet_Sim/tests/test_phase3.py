import sys
import os

# Setup paths so it can find the src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.insert(0, src_dir)

from fidelity.fidelity_model import FidelityModel

def main():
    print("--- Testing Fidelity Degradation (Entanglement Swapping) ---")
    # Imagine 4 physical links, each with 0.95 fidelity (requires 3 swaps to connect end-to-end)
    fidelities = [0.95, 0.95, 0.95, 0.95] 
    
    current_f = fidelities[0]
    for i, next_f in enumerate(fidelities[1:]):
        print(f"Nodes connected: {i+2}, Current Fidelity = {current_f:.4f}")
        current_f = FidelityModel.entanglement_swapping(current_f, next_f)
    print(f"Final End-to-End Fidelity: {current_f:.4f}\n")

    print("--- Testing Purification Boost ---")
    low_f = 0.70
    print(f"Starting Fidelity: {low_f:.4f}")
    
    boosted_f = FidelityModel.purification_bbpssw(low_f)
    print(f"After 1 Round of Purification: {boosted_f:.4f}")
    
    boosted_f_2 = FidelityModel.purification_bbpssw(boosted_f)
    print(f"After 2 Rounds of Purification: {boosted_f_2:.4f}")

if __name__ == '__main__':
    main()
