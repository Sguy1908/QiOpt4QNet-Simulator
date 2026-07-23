import simpy
import math
import pytest
from network.node import QuantumNode
from network.link import QuantumLink
from network.network import QuantumNetwork
from protocols.entanglement_generation import EntanglementGenerationProtocol

def test_node_decoherence():
    env = simpy.Environment()
    node = QuantumNode("Node_A", 10, t1=100.0, t2=50.0)
    
    # Reserve memory at t=0
    mem_ids = node.reserve_memory(1, env.now)
    assert len(mem_ids) == 1
    mem_id = mem_ids[0]
    
    # Initial fidelity should be 1.0
    assert node.calculate_fidelity(mem_id, env.now) == 1.0
    
    # Advance time by 10 units
    current_time = env.now + 10.0
    fidelity = node.calculate_fidelity(mem_id, current_time)
    
    # Expected: 1.0 * exp(-10/100) * exp(-10/50)
    expected_fidelity = math.exp(-10/100) * math.exp(-10/50)
    assert math.isclose(fidelity, expected_fidelity, rel_tol=1e-5)
    
def test_entanglement_generation_protocol():
    env = simpy.Environment()
    
    network = QuantumNetwork()
    node_a = QuantumNode("Node_A", 10)
    node_b = QuantumNode("Node_B", 10)
    network.add_node(node_a)
    network.add_node(node_b)
    
    # 100% success rate, latency of 5.0
    link = QuantumLink("Node_A", "Node_B", distance=10, generation_probability=1.0, raw_fidelity=0.9, latency=5.0, capacity=1)
    network.add_link(link)
    
    protocol = EntanglementGenerationProtocol(env, node_a, network, "Node_B")
    protocol.start()
    
    # Run simulation until there are no more events
    env.run()
    
    # Since probability is 1.0, it should succeed
    assert protocol.success is True
    # Should have taken exactly 5 units of time
    assert env.now == 5.0
    # Both nodes should have 1 memory used
    assert node_a.memory_used == 1
    assert node_b.memory_used == 1
