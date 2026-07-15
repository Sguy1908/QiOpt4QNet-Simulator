import networkx as nx
from typing import List
from network.network import QuantumNetwork
from network.request import Request

class PathGenerator:

    def __init__(self, network: QuantumNetwork):
        self.network = network

    def get_k_shortest_paths(self, request: Request, k: int = 3) -> List[List[str]]:
       
        source = request.source
        destination = request.destination
        
        # Check if nodes exist in the graph
        if source not in self.network.graph or destination not in self.network.graph:
            return []

        try:
            # shortest_simple_paths returns a generator
            path_generator = nx.shortest_simple_paths(self.network.graph, source, destination)
            
            k_paths = []
            for path in path_generator:
                k_paths.append(path)
                if len(k_paths) == k:
                    break
                    
            return k_paths
            
        except nx.NetworkXNoPath:
            # If there's no path between the nodes
            return []
