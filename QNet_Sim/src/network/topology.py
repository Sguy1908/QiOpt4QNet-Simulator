import random
from typing import Union, Optional
from .node import QuantumNode
from .link import QuantumLink
from .network import QuantumNetwork

def _generate_default_link(u: str, v: str, link_params: Optional[dict] = None) -> QuantumLink:
    if link_params is None:
        link_params = {}

    dist_min, dist_max = link_params.get('distance_bounds', (10.0, 50.0))
    prob_min, prob_max = link_params.get('prob_bounds', (0.6, 0.95))
    fid_min, fid_max = link_params.get('fidelity_bounds', (0.8, 0.99))
    lat_min, lat_max = link_params.get('latency_bounds', (1.0, 5.0))
    cap_min, cap_max = link_params.get('capacity_bounds', (1, 10))

    return QuantumLink(
        source=u,
        destination=v,
        distance=random.uniform(dist_min, dist_max),
        generation_probability=round(random.uniform(prob_min, prob_max), 2),
        raw_fidelity=round(random.uniform(fid_min, fid_max), 2),
        latency=round(random.uniform(lat_min, lat_max), 2),
        capacity=random.randint(cap_min, cap_max)
    )

def _get_memory(memory_config: Union[int, tuple]) -> int:
    if isinstance(memory_config, tuple) and len(memory_config) == 2:
        return random.randint(memory_config[0], memory_config[1])
    elif isinstance(memory_config, int):
        return memory_config
    return 10

def generate_chain(n: int, default_memory: Union[int, tuple] = 10, link_params: Optional[dict] = None) -> QuantumNetwork:
    network = QuantumNetwork()
    for i in range(n):
        node_id = f"Node_{i}"
        mem = _get_memory(default_memory)
        network.add_node(QuantumNode(node_id, mem))

    for i in range(n-1):
        u = f"Node_{i}"
        v = f"Node_{i+1}"
        network.add_link(_generate_default_link(u, v, link_params))

    return network

def generate_grid(m: int, n: int, default_memory: Union[int, tuple] = 10, link_params: Optional[dict] = None) -> QuantumNetwork:
    network = QuantumNetwork()
    
    # Add nodes
    for i in range(m):
        for j in range(n):
            node_id = f"Node_{i}_{j}"
            mem = _get_memory(default_memory)
            network.add_node(QuantumNode(node_id, mem))
            
    # Add horizontal links
    for i in range(m):
        for j in range(n - 1):
            u = f"Node_{i}_{j}"
            v = f"Node_{i}_{j+1}"
            network.add_link(_generate_default_link(u, v, link_params))
            
    # Add vertical links
    for i in range(m - 1):
        for j in range(n):
            u = f"Node_{i}_{j}"
            v = f"Node_{i+1}_{j}"
            network.add_link(_generate_default_link(u, v, link_params))
            
    return network

def generate_random(n: int, p: float, default_memory: Union[int, tuple] = 10, link_params: Optional[dict] = None) -> QuantumNetwork:
    import networkx as nx
    network = QuantumNetwork()
    
    # We leverage networkx to generate the random Erdős-Rényi math for us
    er_graph = nx.erdos_renyi_graph(n, p)
    
    for i in range(n):
        node_id = f"Node_{i}"
        mem = _get_memory(default_memory)
        network.add_node(QuantumNode(node_id, mem))
        
    for u, v in er_graph.edges():
        network.add_link(_generate_default_link(f"Node_{u}", f"Node_{v}", link_params))
        
    return network

def generate_heavy_hex(m: int, n: int, default_memory: Union[int, tuple] = 10, link_params: Optional[dict] = None) -> QuantumNetwork:
    import networkx as nx
    network = QuantumNetwork()
    
    # 1. Generate base hexagonal lattice
    hex_graph = nx.hexagonal_lattice_graph(m, n)
    
    # The nodes are generated as (x,y) tuples. Let's map them to string IDs
    node_mapping = {node: f"Node_H_{node[0]}_{node[1]}" for node in hex_graph.nodes()}
    
    # Add the base hex nodes to our network
    for node, node_id in node_mapping.items():
        mem = _get_memory(default_memory)
        network.add_node(QuantumNode(node_id, mem))
        
    # 2. Subdivide each edge to make it "heavy"
    edge_idx = 0
    for u, v in hex_graph.edges():
        u_id = node_mapping[u]
        v_id = node_mapping[v]
        
        # Create a new node in the middle of the edge
        middle_id = f"Node_Edge_{edge_idx}"
        edge_idx += 1
        
        # Add the middle node to the network
        mem = _get_memory(default_memory)
        network.add_node(QuantumNode(middle_id, mem))
        
        # Connect the base nodes to the new middle node
        network.add_link(_generate_default_link(u_id, middle_id, link_params))
        network.add_link(_generate_default_link(middle_id, v_id, link_params))
        
    return network
