import random
import simpy
from network.node import QuantumNode
from network.network import QuantumNetwork
from protocols.base_protocol import Protocol
from models.quantum_state import QuantumState

class EntanglementGenerationProtocol(Protocol):
    """
    Simulates the generation of entanglement between two adjacent nodes.
    It takes into account the link latency and generation probability.
    """
    def __init__(self, env: simpy.Environment, node: QuantumNode, network: QuantumNetwork, target_node_id: str):
        super().__init__(env, node, network)
        self.target_node_id = target_node_id
        self.success = False
        self.generated_memory_ids = []

    def run(self):
        # 1. Check if there's a link between self.node.id and self.target_node_id
        if not self.network.graph.has_edge(self.node.id, self.target_node_id):
            print(f"[{self.env.now}] Protocol Failed: No link between {self.node.id} and {self.target_node_id}")
            return
            
        link = self.network.graph[self.node.id][self.target_node_id]['data']
        
        print(f"[{self.env.now}] {self.node.id} attempting entanglement generation with {self.target_node_id}...")
        
        # 2. Wait for the link latency (simulating photon travel or classical signaling)
        yield self.env.timeout(link.latency)
        
        # 3. Determine success based on link's generation_probability
        if random.random() <= link.generation_probability:
            # 4. If successful, attempt to reserve memory on both nodes
            target_node = self.network.graph.nodes[self.target_node_id]['data']
            
            if self.node.available_memory() > 0 and target_node.available_memory() > 0:
                mem_id_source = self.node.reserve_memory(1, self.env.now)
                mem_id_target = target_node.reserve_memory(1, self.env.now)
                
                if mem_id_source and mem_id_target:
                    self.success = True
                    self.generated_memory_ids = (mem_id_source[0], mem_id_target[0])
                    
                    # Instantiate a Bell state and assign to nodes
                    shared_state = QuantumState.create_bell_state()
                    self.node.assign_state(self.generated_memory_ids[0], shared_state)
                    target_node.assign_state(self.generated_memory_ids[1], shared_state)
                    
                    print(f"[{self.env.now}] Success: Entanglement established between {self.node.id} and {self.target_node_id}.")
                else:
                    print(f"[{self.env.now}] Failure: Insufficient memory on one of the nodes.")
            else:
                print(f"[{self.env.now}] Failure: Memory full.")
        else:
            print(f"[{self.env.now}] Failure: Generation failed due to link probability.")
