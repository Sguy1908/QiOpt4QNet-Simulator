class QuantumLink:
    def __init__(self, source: str, destination: str, distance: float, generation_probability: float, raw_fidelity: float, latency: float, capacity: float):
        self.source = source
        self.destination = destination
        self.distance = distance
        self.generation_probability = generation_probability
        self.raw_fidelity = raw_fidelity
        self.latency = latency
        self.capacity = capacity

    def __repr__(self):
        return (f"QuantumLink({self.source} -> {self.destination}, " 
                f"F={self.raw_fidelity}, P={self.generation_probability}, "
                f"C={self.capacity})")

