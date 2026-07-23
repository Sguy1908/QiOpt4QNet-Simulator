from itertools import combinations
from pyqubo import Binary, Constraint, LogEncInteger, Placeholder

class QUBOOptimizer:
    required_params = {
        "bundle_id",
        "request_id",
        "path",
        "purification_profile",
        "final_fidelity",
        "success_probability",
        "latency",
        "bell_pair_cost",
        "edge_demands",
        "memory_demands",
        "utility"
    }
#Success prob./latency/b-pair cost/fidelity are used for utility. 
#Pruning is done with utility, fidelity, and b-pair cost
#Purification is used to calculate fidelity, prob., latency, and resource demand
#The only things really needed for optimization are the bundle_id, request_id, path, edge_demands, memory_demands, and utility. The rest are used to calculate these values

    def __init__(self, bundles, edge_capacities, memory_capacities):
        self.bundles = bundles
        self._bundle_review()
        self.edge_capacities = self._clean_edge_capacities(edge_capacities)
        self.memory_capacities = memory_capacities
        self.bundles_by_request = self._group_bundles()
        self.variables = self._binary_variables()
        self.variable_map = self._create_variable_map()
        self.edge_demands = self._group_edge_demands()
        self.memory_demands = self._group_memory_demands()
        self.model = self._build_hamiltonian().compile()

    def _bundle_review(self):
        for bundle in self.bundles:
            missing = self.required_params - bundle.keys()
            if missing:
                bundle_id = bundle.get("bundle_id", "unknown")
                raise ValueError(f"Bundle {bundle_id} is missing: {missing}")
            
    def _bundle_key(self, bundle):
        return (bundle["request_id"], bundle["bundle_id"])
    
    def _undirected_edge(self,edge):
        return tuple(sorted(edge))
#Edge order is undirected

    def _clean_edge_capacities(self, edge_capacities):
        capacities = {}
        for edge, capacity in edge_capacities.items():
            edge = self._undirected_edge(edge)
            if edge in capacities:
                raise ValueError(f"Edge {edge} has duplicate capacities")
            capacities[edge] = capacity
        return capacities
    
    def _group_bundles(self):
        groups = {}
        for bundle in self.bundles:
            groups.setdefault(bundle["request_id"], []).append(bundle)
        return groups
#For H_{one bundle per request}
    
    def _binary_variables(self):
        variables = {}
        for index,bundle in enumerate(self.bundles):
            key = self._bundle_key(bundle)
            if key in variables:
                raise ValueError(
                    f"Bundle {bundle['bundle_id']} Request {bundle['request_id']} duplicate"
                )
            variables[key] = Binary(f"x_{index}")
        return variables
    
    def _create_variable_map(self):
        variable_map = {}
        for index, bundle in enumerate(self.bundles):
            variable_map[f"x_{index}"] = self._bundle_key(bundle)
        return variable_map
#Used later when decoding solution

    def _group_edge_demands(self):
        edge_demands = {}
        for bundle in self.bundles:
            key = self._bundle_key(bundle)
            for edge, demand in bundle["edge_demands"].items():
                if demand == 0:
                    continue
                edge = self._undirected_edge(edge)
                if edge not in self.edge_capacities:
                    raise ValueError(f"Missing edge capacity for edge {edge}")
                edge_demands.setdefault(edge, []).append((key,demand))
        return edge_demands
    
    def _group_memory_demands(self):
        memory_demands = {}
        for bundle in self.bundles:
            key = self._bundle_key(bundle)
            for node, demand in bundle["memory_demands"].items():
                if demand == 0:
                    continue
                if node not in self.memory_capacities:
                    raise ValueError(f"Missing memory capacity for node {node}")
                memory_demands.setdefault(node, []).append((key,demand))
        return memory_demands

#Inequalities into equalities
    def _capacity_term(self, name, uses, capacity, penalty):
        load = sum(
            amount * self.variables[key]
            for key, amount in uses
        )
        slack = (
            LogEncInteger(f"{name}_slack", (0, capacity))
            if capacity > 0
            else 0
        )
        return penalty*Constraint(
            (load + slack - capacity)**2,
            label=name
        )

#H=H_{utility}+H_{one bundle per request}+H_{edge}+H_{memory}
    def _build_hamiltonian(self):
        hamiltonian = 0
        for bundle in self.bundles:
            key = self._bundle_key(bundle)
            hamiltonian -= bundle["utility"] * self.variables[key]
        penalty = Placeholder("A")
        for request_id, bundles in self.bundles_by_request.items():
            if len(bundles)<2:
                continue
            conflicts=0
            for first,second in combinations(bundles,2):
                first_key = self._bundle_key(first)
                second_key = self._bundle_key(second)
                conflicts += self.variables[first_key] * self.variables[second_key]
            hamiltonian += penalty * Constraint(
                conflicts,
                label=f"request_{request_id}"
            )
        edge_penalty = Placeholder("B")
        for index, (edge,uses) in enumerate(self.edge_demands.items()):
            hamiltonian += self._capacity_term(
                f"edge_{index}",
                uses,
                self.edge_capacities[edge],
                edge_penalty
            )
        memory_penalty = Placeholder("D")
        for index, (node,uses) in enumerate(self.memory_demands.items()):
            hamiltonian += self._capacity_term(
                f"memory_{index}",
                uses,
                self.memory_capacities[node],
                memory_penalty
            )
        return hamiltonian
    
    def decode_sample(self, sample):
        selected = []
        for variable, bundle_key in self.variable_map.items():
            if sample[variable] == 1:
                selected.append(bundle_key)
        return selected

    def to_qubo(self, penalty, edge_penalty, memory_penalty):
        return self.model.to_qubo(feed_dict={"A": penalty, "B": edge_penalty, "D": memory_penalty})

    def to_bqm(self, penalty, edge_penalty, memory_penalty):
        return self.model.to_bqm(feed_dict={"A": penalty, "B": edge_penalty, "D": memory_penalty})