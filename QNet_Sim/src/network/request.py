class Request:
    def __init__(self, source: str, destination: str, minimum_fidelity: float, weight: float = 1.0):
        self.source = source
        self.destination = destination
        self.minimum_fidelity = minimum_fidelity
        self.weight = weight
    
    def __repr__(self):
        return f"Request({self.source} -> {self.destination}, min_F = {self.minimum_fidelity}, weight = {self.weight})"
