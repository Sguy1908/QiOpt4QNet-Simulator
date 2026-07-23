import simpy
from typing import Optional
from network.node import QuantumNode
from network.network import QuantumNetwork
from protocols.base_protocol import Protocol
from models.quantum_state import QuantumState

class EntanglementSwappingProtocol(Protocol):
    """
    Simulates Entanglement Swapping at a middle node (B) between two outer nodes (A and C).
    """
    def __init__(self, env: simpy.Environment, node: QuantumNode, network: QuantumNetwork, 
                 node_a_id: str, node_c_id: str,
                 mem_id_b_to_a: int, mem_id_b_to_c: int,
                 mem_id_a: int, mem_id_c: int):
        super().__init__(env, node, network)
        self.node_a_id = node_a_id
        self.node_c_id = node_c_id
        
        # Memory IDs at the middle node (self.node)
        self.mem_id_b_to_a = mem_id_b_to_a
        self.mem_id_b_to_c = mem_id_b_to_c
        
        # Memory IDs at the endpoints
        self.mem_id_a = mem_id_a
        self.mem_id_c = mem_id_c
        
        self.success = False
        self.swapped_state: Optional[QuantumState] = None

    def run(self):
        print(f"[{self.env.now}] {self.node.id} initiating Entanglement Swapping between {self.node_a_id} and {self.node_c_id}...")
        
        # 1. Retrieve the states from memory
        state_ab = self.node.get_state(self.mem_id_b_to_a)
        state_bc = self.node.get_state(self.mem_id_b_to_c)
        
        if state_ab is None or state_bc is None:
            print(f"[{self.env.now}] Protocol Failed: Missing entanglement states at {self.node.id}.")
            return
            
        # 2. Perform BSM Operation
        # In a real hardware model, applying gates for BSM takes time. 
        # We'll assume a fixed 1.0 time unit for BSM processing.
        bsm_processing_time = 1.0
        yield self.env.timeout(bsm_processing_time)
        
        new_ac_state = QuantumState.entanglement_swap(state_ab, state_bc)
        
        # 3. Send classical results to A and C
        # The classical message takes time equal to the link latency
        # We find the links
        link_ab = self.network.graph[self.node_a_id][self.node.id]['data']
        link_bc = self.network.graph[self.node.id][self.node_c_id]['data']
        
        # The endpoints can only use the state once they receive the classical message.
        # We wait for the maximum of the two latencies to ensure both have received it.
        max_latency = max(link_ab.latency, link_bc.latency)
        yield self.env.timeout(max_latency)
        
        # 4. Assign new state to endpoints and free middle memory
        node_a = self.network.graph.nodes[self.node_a_id]['data']
        node_c = self.network.graph.nodes[self.node_c_id]['data']
        
        node_a.assign_state(self.mem_id_a, new_ac_state)
        node_c.assign_state(self.mem_id_c, new_ac_state)
        
        self.node.release_memory([self.mem_id_b_to_a, self.mem_id_b_to_c])
        
        self.success = True
        self.swapped_state = new_ac_state
        print(f"[{self.env.now}] Success: Swapping completed. {self.node_a_id} and {self.node_c_id} are now entangled.")
