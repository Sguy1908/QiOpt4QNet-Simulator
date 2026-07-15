class FidelityModel:

    
    @staticmethod
    def entanglement_swapping(f1: float, f2: float) -> float:
     
        # Theoretical formula for swapping two Werner states
        return f1 * f2 + ((1 - f1) * (1 - f2)) / 3.0

    @staticmethod
    def purification_bbpssw(f: float) -> float:
      
        if f <= 0.5:
            return f  # Purification mathematically fails (cannot improve) if F <= 0.5
            
        # BBPSSW analytic formula
        term1 = f**2 + ((1 - f) / 3.0)**2
        term2 = f**2 + (2.0 * f * (1 - f) / 3.0) + 5.0 * ((1 - f) / 3.0)**2
        return term1 / term2
        
    @staticmethod
    def end_to_end_fidelity(link_fidelities: list[float]) -> float:
       
        if not link_fidelities:
            return 0.0
            
        final_f = link_fidelities[0]
        for next_f in link_fidelities[1:]:
            final_f = FidelityModel.entanglement_swapping(final_f, next_f)
            
        return final_f
