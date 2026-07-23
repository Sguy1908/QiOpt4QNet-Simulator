import math
from typing import Optional
from models.quantum_state import QuantumState

class QuantumNode:
    def __init__(self, node_id: str, memory_capacity: int, t1: float = float('inf'), t2: float = float('inf')):
        self.id = node_id
        self.memory_capacity = memory_capacity
        self.t1 = t1
        self.t2 = t2
        
        # We still keep memory_used for backward compatibility with simple tests,
        # but we also introduce a dict to track details of individual memory qubits.
        self._memory_used_count = 0
        # memory_reservations[id] = {'creation_time': float, 'state': Optional[QuantumState]}
        self.memory_reservations = {}
        self._next_mem_id = 0
    
    @property
    def memory_used(self) -> int:
        return self._memory_used_count
        
    @memory_used.setter
    def memory_used(self, value: int):
        self._memory_used_count = value
    
    def reserve_memory(self, amount: int = 1, current_time: float = 0.0) -> list[int]:
        """Reserves memory and returns a list of memory IDs. Returns empty list if not enough memory."""
        if self.available_memory() >= amount:
            self._memory_used_count += amount
            reserved_ids = []
            for _ in range(amount):
                mem_id = self._next_mem_id
                self.memory_reservations[mem_id] = {'creation_time': current_time, 'state': None}
                reserved_ids.append(mem_id)
                self._next_mem_id += 1
            return reserved_ids
        return []
        
    def assign_state(self, mem_id: int, state: QuantumState):
        """Assigns a generated QuantumState to a reserved memory slot."""
        if mem_id in self.memory_reservations:
            self.memory_reservations[mem_id]['state'] = state
            
    def get_state(self, mem_id: int) -> Optional[QuantumState]:
        if mem_id in self.memory_reservations:
            return self.memory_reservations[mem_id]['state']
        return None

    def release_memory(self, memory_ids: Optional[list[int]] = None, amount: int = 1):
        """Release specific memory IDs, or just an amount for backward compatibility."""
        if memory_ids is not None:
            for mem_id in memory_ids:
                if mem_id in self.memory_reservations:
                    del self.memory_reservations[mem_id]
                    self._memory_used_count -= 1
        else:
            # Backward compatibility
            if self._memory_used_count >= amount:
                self._memory_used_count -= amount
                # We can't know which one to delete, so just remove the oldest ones
                for _ in range(amount):
                    if self.memory_reservations:
                        oldest_key = min(self.memory_reservations.keys(), key=lambda k: self.memory_reservations[k]['creation_time'])
                        del self.memory_reservations[oldest_key]
            else:
                self._memory_used_count = 0
                self.memory_reservations.clear()

    def available_memory(self) -> int:
        return self.memory_capacity - self._memory_used_count

    def calculate_fidelity(self, mem_id: int, current_time: float, initial_fidelity: float = 1.0) -> float:
        """Calculate the fidelity of a specific memory at the current time due to decoherence."""
        if mem_id not in self.memory_reservations:
            return 0.0
            
        creation_time = self.memory_reservations[mem_id]['creation_time']
        state = self.memory_reservations[mem_id]['state']
        
        dt = current_time - creation_time
        
        if state is not None and dt > 0:
            # Apply physics-based decoherence to the density matrix
            # Note: in a real sim, the state shouldn't be mutated every time we check fidelity
            # unless we are taking a copy. We'll mutate it for simplicity here as time strictly advances.
            # But wait, calculate_fidelity might be called multiple times at the same 'current_time'.
            # To be safe, we just return the theoretical decay if we want to preserve the old behavior.
            pass
            
        if dt <= 0:
            return initial_fidelity
            
        # Simplified exponential decay model for fidelity backward-compatibility
        decay = math.exp(-dt / self.t1) * math.exp(-dt / self.t2)
        return initial_fidelity * decay

    def __repr__(self):
        return f"QuantumNode(id={self.id}, capacity={self.memory_capacity}, available={self.available_memory()}, T1={self.t1}, T2={self.t2})"


