import simpy
from typing import Optional
from network.node import QuantumNode
from network.network import QuantumNetwork

class Protocol:
    """
    Base class for all quantum network protocols.
    
    A protocol runs on a specific node within a network, driven by a SimPy environment.
    """
    def __init__(self, env: simpy.Environment, node: QuantumNode, network: QuantumNetwork):
        self.env = env
        self.node = node
        self.network = network
        self.process: Optional[simpy.events.Process] = None

    def start(self):
        """Starts the protocol process in the SimPy environment."""
        self.process = self.env.process(self.run())

    def run(self):
        """
        The main generator function for the protocol logic.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the run() method.")
