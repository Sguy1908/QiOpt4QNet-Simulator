import numpy as np
import math

class QuantumState:
    """
    Represents a 2-qubit quantum state using a 4x4 density matrix.
    """
    def __init__(self, density_matrix: np.ndarray = None):
        if density_matrix is None:
            # Default to perfect Bell state |Phi+>
            bell_vec = np.array([1, 0, 0, 1]) / math.sqrt(2)
            self.dm = np.outer(bell_vec, bell_vec.conj())
        else:
            self.dm = np.array(density_matrix, dtype=complex)
            
    @classmethod
    def create_bell_state(cls):
        return cls()
        
    def apply_decoherence(self, t1: float, t2: float, dt: float):
        """
        Applies T1 (amplitude damping) and T2 (phase damping) channels to BOTH qubits independently.
        """
        if dt <= 0:
            return
            
        p_t1 = math.exp(-dt / t1) if t1 > 0 else 0
        p_t2 = math.exp(-dt / t2) if t2 > 0 else 0
        
        # Amplitude Damping Kraus operators for one qubit
        E0_ad = np.array([[1, 0], [0, math.sqrt(p_t1)]])
        E1_ad = np.array([[0, math.sqrt(1 - p_t1)], [0, 0]])
        
        # Phase Damping Kraus operators for one qubit
        E0_pd = np.array([[1, 0], [0, math.sqrt(p_t2)]])
        E1_pd = np.array([[0, 0], [0, math.sqrt(1 - p_t2)]])
        
        # Combine them (they commute for the diagonal)
        # But applying them sequentially is easier
        self._apply_kraus_to_both([E0_ad, E1_ad])
        self._apply_kraus_to_both([E0_pd, E1_pd])
        
    def _apply_kraus_to_both(self, kraus_ops: list[np.ndarray]):
        """Applies single-qubit Kraus operators to both qubits."""
        new_dm = np.zeros((4, 4), dtype=complex)
        
        # Tensor all combinations of Kraus operators
        for k1 in kraus_ops:
            for k2 in kraus_ops:
                k_joint = np.kron(k1, k2)
                new_dm += k_joint @ self.dm @ k_joint.conj().T
                
        self.dm = new_dm

    def fidelity_with_bell(self) -> float:
        """Calculates fidelity with |Phi+>"""
        bell_vec = np.array([1, 0, 0, 1]) / math.sqrt(2)
        bell_dm = np.outer(bell_vec, bell_vec.conj())
        
        # For pure target state |psi>, F = Tr(rho * |psi><psi|)
        f = np.trace(self.dm @ bell_dm).real
        return float(f)
        
    @staticmethod
    def entanglement_swap(state_ab: 'QuantumState', state_bc: 'QuantumState') -> 'QuantumState':
        """
        Simulates a Bell State Measurement on qubit B1 and B2, leaving A and C entangled.
        """
        # Tensor product rho_AB \otimes rho_BC (16x16 matrix)
        rho_4 = np.kron(state_ab.dm, state_bc.dm)
        
        # BSM is performed on B1 and B2 (indices 1 and 2).
        # Projector P = I_A \otimes |Phi+><Phi+| \otimes I_C
        phi_plus = np.array([1, 0, 0, 1]) / math.sqrt(2)
        proj_phi = np.outer(phi_plus, phi_plus.conj()) # 4x4
        
        I2 = np.eye(2)
        P = np.kron(I2, np.kron(proj_phi, I2)) # 16x16
        
        # Apply projection: P * rho_4 * P
        post_meas_state = P @ rho_4 @ P
        
        # Trace out B1, B2
        # Reshape to 8D array: (A, B1, B2, C, A', B1', B2', C')
        reshaped = post_meas_state.reshape((2,2,2,2, 2,2,2,2))
        
        # Trace over B1 (axis 1 and 5)
        traced_b1 = np.trace(reshaped, axis1=1, axis2=5)
        # Now axes are (A, B2, C, A', B2', C')
        # Trace over B2 (which is now axis 1 and 4)
        traced_b2 = np.trace(traced_b1, axis1=1, axis2=4)
        
        rho_ac = traced_b2.reshape((4,4))
        
        # Normalize
        trace_val = np.trace(rho_ac).real
        if trace_val > 0:
            rho_ac = rho_ac / trace_val
            
        return QuantumState(rho_ac)
