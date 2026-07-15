
import networkx as nx
import matplotlib.pyplot as plt
from .node import QuantumNode
from .link import QuantumLink

class QuantumNetwork:
    def __init__(self):
        self.graph = nx.Graph()
    def add_node(self, node: QuantumNode):
        self.graph.add_node(node.id, data=node)
    def add_link(self, link: QuantumLink):
        self.graph.add_edge(link.source, link.destination, data=link)
    def display_nodes(self):
        for node_id, data in self.graph.nodes(data=True):
            print(data['data'])
    def display_links(self):
        for u, v, data in self.graph.edges(data=True):
            print(data['data'])
    def visualize(self):
        pos = nx.spring_layout(self.graph)

        node_labels = {
            node_id: f"{node_id}\nMem:{data['data'].memory_capacity}"
            for node_id, data in self.graph.nodes(data=True)
        }
        edge_labels = {
            (u, v): f"F:{data['data'].raw_fidelity}\nP:{data['data'].generation_probability}\nC:{data['data'].capacity}"
            for u, v, data in self.graph.edges(data=True)
        }
        
        plt.figure(figsize=(10, 8))
        
        # Draw the graph elements
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(self.graph, pos, width=2, edge_color='gray')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Quantum Network Topology")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
            

            