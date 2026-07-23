import simpy
import math
import sys
import os

# Ensure src is in the path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
sys.path.insert(0, src_dir)

from network.node import QuantumNode
from network.link import QuantumLink
from network.network import QuantumNetwork
from protocols.entanglement_generation import EntanglementGenerationProtocol
from protocols.entanglement_swapping import EntanglementSwappingProtocol

def main():
    print("==================================================")
    print("      Quantum Network Simulation Demo             ")
    print("==================================================")
    
    # 1. Initialize Simulation Environment
    env = simpy.Environment()
    net = QuantumNetwork()
    
    # 2. Create Nodes with T1/T2 Decoherence
    # We'll use realistic decoherence times (e.g., T1=100ms, T2=50ms)
    node_a = QuantumNode("Alice", memory_capacity=10, t1=100.0, t2=50.0)
    node_b = QuantumNode("Repeater", memory_capacity=10, t1=100.0, t2=50.0)
    node_c = QuantumNode("Bob", memory_capacity=10, t1=100.0, t2=50.0)
    
    net.add_node(node_a)
    net.add_node(node_b)
    net.add_node(node_c)
    
    # 3. Create Links
    # Alice <---> Repeater <---> Bob
    link_ab = QuantumLink("Alice", "Repeater", distance=10, generation_probability=1.0, raw_fidelity=1.0, latency=5.0, capacity=1)
    link_bc = QuantumLink("Repeater", "Bob", distance=15, generation_probability=1.0, raw_fidelity=1.0, latency=7.5, capacity=1)
    
    net.add_link(link_ab)
    net.add_link(link_bc)
    
    print(f"\n[Time: {env.now}] Starting protocols...")
    
    # 4. Schedule Entanglement Generation Protocols
    gen_protocol_ab = EntanglementGenerationProtocol(env, node_a, net, "Repeater")
    gen_protocol_bc = EntanglementGenerationProtocol(env, node_b, net, "Bob")
    
    gen_protocol_ab.start()
    gen_protocol_bc.start()
    
    # Run the simulation just enough for generation to finish (max latency 7.5)
    env.run(until=10.0)
    
    if not (gen_protocol_ab.success and gen_protocol_bc.success):
        print(f"\n[Time: {env.now}] Generation failed due to probabilistic link drops. Try running again!")
        return
        
    print(f"\n[Time: {env.now}] Entanglement Generation Successful!")
    mem_a, mem_b_a = gen_protocol_ab.generated_memory_ids
    mem_b_c, mem_c = gen_protocol_bc.generated_memory_ids
    
    state_ab = node_a.get_state(mem_a)
    state_bc = node_b.get_state(mem_b_c)
    
    # Apply time-based decoherence since generation
    # Alice-Repeater entanglement finished at t=5.0, now it's t=10.0 (dt=5.0)
    state_ab.apply_decoherence(node_a.t1, node_a.t2, dt=5.0)
    # Repeater-Bob finished at t=7.5, now it's t=10.0 (dt=2.5)
    state_bc.apply_decoherence(node_b.t1, node_b.t2, dt=2.5)
    
    print(f"Alice-Repeater Fidelity (Decohered for 5.0ms): {state_ab.fidelity_with_bell():.4f}")
    print(f"Repeater-Bob Fidelity (Decohered for 2.5ms): {state_bc.fidelity_with_bell():.4f}")
    
    # 5. Schedule Entanglement Swapping
    print(f"\n[Time: {env.now}] Repeater is starting Entanglement Swapping...")
    swap_protocol = EntanglementSwappingProtocol(
        env, node_b, net, "Alice", "Bob", mem_b_a, mem_b_c, mem_a, mem_c
    )
    swap_protocol.start()
    
    # Run simulation until swapping completes
    env.run()
    
    if swap_protocol.success:
        print(f"\n[Time: {env.now}] Swapping Complete!")
        final_state = swap_protocol.swapped_state
        print(f"Final End-to-End Fidelity (Alice <---> Bob): {final_state.fidelity_with_bell():.4f}")
        print(f"Repeater memory used: {node_b.memory_used} (Should be 0, as it was traced out during BSM)")
        print(f"Alice memory used: {node_a.memory_used}")
        print(f"Bob memory used: {node_c.memory_used}")

if __name__ == "__main__":
    main()
