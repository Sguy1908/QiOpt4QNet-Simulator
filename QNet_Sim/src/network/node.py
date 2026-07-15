class QuantumNode:
    def __init__(self, node_id: str, memory_capacity: int):
        self.id = node_id
        self.memory_capacity = memory_capacity
        self.memory_used = 0
    
    def reserve_memory(self, amount: int = 1) -> bool:
        if self.available_memory() >= amount:
            self.memory_used += amount
            return True
        return False

    def release_memory(self, amount: int = 1):
        if self.memory_used >= amount:
            self.memory_used -= amount
        else:
            self.memory_used = 0

    def available_memory(self) -> int:
        return self.memory_capacity - self.memory_used

    def __repr__(self):
        return f"QuantumNode(id={self.id}, capacity = {self.memory_capacity}, available = {self.available_memory()})"


