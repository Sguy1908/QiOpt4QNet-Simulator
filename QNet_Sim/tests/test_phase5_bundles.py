from network.network import QuantumNetwork
from network.link import QuantumLink
from network.node import QuantumNode
from network.request import Request
from routing.bundle_generation import BundleGenerator, Bundle

def test_bundle_generation():
    net = QuantumNetwork()
    net.add_node(QuantumNode("A", 10))
    net.add_node(QuantumNode("B", 10))
    net.add_node(QuantumNode("C", 10))
    
    # Links with low fidelity to ensure purification helps
    net.add_link(QuantumLink("A", "B", 10, 1.0, 0.8, 5.0, 1))
    net.add_link(QuantumLink("B", "C", 10, 1.0, 0.8, 5.0, 1))
    
    request = Request(source="A", destination="C", minimum_fidelity=0.5, weight=1.0)
    
    paths = [["A", "B", "C"]]
    
    generator = BundleGenerator(net)
    bundles = generator.generate_bundles(request, paths)
    
    # The higher cost/latency bundles should NOT be pruned if they offer higher fidelity
    # We should expect 3 bundles (q=0, q=1, q=2)
    assert len(bundles) == 3
    
    # Verify costs (q=0 -> 1 pair/link = 2. q=1 -> 2 pairs/link = 4. q=2 -> 4 pairs/link = 8)
    costs = [b.bell_pair_cost for b in bundles]
    assert sorted(costs) == [2, 4, 8]

def test_dominated_pruning():
    generator = BundleGenerator(QuantumNetwork())
    req = Request("A", "C", 0.5)
    
    # Bundle A: High fidelity, low cost (Dominates B)
    b_a = Bundle(req, ["A", "C"], 1, 0.9, 10.0, 2)
    # Bundle B: Low fidelity, high cost
    b_b = Bundle(req, ["A", "B", "C"], 0, 0.7, 20.0, 4)
    # Bundle C: Highest fidelity, high cost (Not dominated by A)
    b_c = Bundle(req, ["A", "D", "C"], 2, 0.95, 30.0, 8)
    
    pruned = generator.prune_dominated_bundles([b_a, b_b, b_c])
    
    # B should be pruned because A is strictly better in all metrics
    assert len(pruned) == 2
    assert b_a in pruned
    assert b_c in pruned
    assert b_b not in pruned
