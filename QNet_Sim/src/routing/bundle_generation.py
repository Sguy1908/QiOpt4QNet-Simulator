from dataclasses import dataclass
from typing import List, Optional
from network.network import QuantumNetwork
from network.request import Request
from fidelity.fidelity_model import FidelityModel

@dataclass
class Bundle:
    request: Request
    path: List[str]
    purification_rounds: int
    fidelity: float
    latency: float
    bell_pair_cost: int

class BundleGenerator:
    def __init__(self, network: QuantumNetwork):
        self.network = network

    def generate_bundles(self, request: Request, k_paths: List[List[str]]) -> List[Bundle]:
        bundles = []
        for path in k_paths:
            # We consider 0, 1, and 2 rounds of purification per link
            for q in [0, 1, 2]:
                bundle = self._evaluate_bundle(request, path, q)
                if bundle and bundle.fidelity >= request.minimum_fidelity:
                    bundles.append(bundle)
        
        return self.prune_dominated_bundles(bundles)

    def _evaluate_bundle(self, request: Request, path: List[str], q: int) -> Optional[Bundle]:
        if len(path) < 2:
            return None
            
        link_fidelities = []
        total_latency = 0.0
        total_bell_pairs = 0
        
        # Calculate properties for each link in the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.network.graph[u][v]['data']
            
            # Base raw fidelity
            f = edge_data.raw_fidelity
            
            # Apply purification q times
            for _ in range(q):
                f = FidelityModel.purification_bbpssw(f)
                
            link_fidelities.append(f)
            
            cost = 2 ** q 
            total_bell_pairs += cost
            
            total_latency += edge_data.latency * cost

        # Calculate end-to-end fidelity after swapping
        end_fidelity = FidelityModel.end_to_end_fidelity(link_fidelities)
        
        return Bundle(
            request=request,
            path=path,
            purification_rounds=q,
            fidelity=end_fidelity,
            latency=total_latency,
            bell_pair_cost=total_bell_pairs
        )

    def prune_dominated_bundles(self, bundles: List[Bundle]) -> List[Bundle]:
        """
        Removes dominated bundles. Bundle A dominates Bundle B if A has:
        - Higher or equal fidelity
        - Lower or equal cost
        - Lower or equal latency
        And is strictly better in at least one of these metrics.
        """
        pruned = []
        for i, b_candidate in enumerate(bundles):
            is_dominated = False
            for j, b_other in enumerate(bundles):
                if i == j:
                    continue
                    
                # Check if b_other dominates b_candidate
                better_fid = b_other.fidelity >= b_candidate.fidelity
                better_cost = b_other.bell_pair_cost <= b_candidate.bell_pair_cost
                better_lat = b_other.latency <= b_candidate.latency
                
                strictly_better = (b_other.fidelity > b_candidate.fidelity or 
                                   b_other.bell_pair_cost < b_candidate.bell_pair_cost or 
                                   b_other.latency < b_candidate.latency)
                                   
                if better_fid and better_cost and better_lat and strictly_better:
                    is_dominated = True
                    break
                    
            if not is_dominated:
                pruned.append(b_candidate)
                
        return pruned
