from collections import defaultdict
from itertools import combinations
from pyqubo import Binary, Constraint, LogEncInteger, Num, Placeholder  # type: ignore

class QUBOOptimizer:
    required_params = {
        "bundle_id",
        "request_id",
        "path",
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
        self.memory_capacities = self._clean_memory_capacities(memory_capacities)
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
            if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
                raise ValueError("Edge capacity must be a nonnegative integer")
            edge = self._undirected_edge(edge)
            if edge in capacities:
                raise ValueError(f"Edge {edge} has duplicate capacities")
            capacities[edge] = capacity
        return capacities

    @staticmethod
    def _clean_memory_capacities(memory_capacities):
        for node, capacity in memory_capacities.items():
            if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
                raise ValueError(
                    f"Memory capacity for node {node} must be a nonnegative integer"
                )
        return memory_capacities
    
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

    def _congestion_term(self, name, uses, penalty):
        """Quadratic congestion term C*load^2 (and its memory analogue E*load^2).

        (sum_i d_i x_i)^2 = sum_i d_i^2 x_i + sum_{i<j} 2 d_i d_j x_i x_j,
        so the binary-square expansion reproduces the paper's congestion term
        inside the QUBO itself.
        """
        load = sum(
            amount * self.variables[key]
            for key, amount in uses
        )
        return penalty * (load * load)

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

        self.edge_index={}
        for index, (edge,uses) in enumerate(self.edge_demands.items()):
            self.edge_index[edge]=index
            hamiltonian += self._capacity_term(
                f"edge_{index}",
                uses,
                self.edge_capacities[edge],
                Placeholder(f"B_{index}")
            )
        self.memory_index={}
        for index, (node,uses) in enumerate(self.memory_demands.items()):
            self.memory_index[node]=index
            hamiltonian += self._capacity_term(
                f"memory_{index}",
                uses,
                self.memory_capacities[node],
                Placeholder(f"D_{index}")
            )
        congestion_penalty = Placeholder("C")
        for index, (edge,uses) in enumerate(self.edge_demands.items()):
            hamiltonian += self._congestion_term(
                f"cong_edge_{index}",
                uses,
                congestion_penalty
            )
        memory_congestion_penalty = Placeholder("E")
        for index, (node,uses) in enumerate(self.memory_demands.items()):
            hamiltonian += self._congestion_term(
                f"cong_mem_{index}",
                uses,
                memory_congestion_penalty
            )
        if isinstance(hamiltonian, (int, float)):
            return Num(hamiltonian)
        return hamiltonian
    
    def _demands_of(self, request_id, bundle_id):
        for bundle in self.bundles:
            if bundle["request_id"] == request_id and bundle["bundle_id"] == bundle_id:
                return bundle["edge_demands"], bundle["memory_demands"]
        return None

    def repair_selection(self, selected):
        """Deterministic greedy feasibility repair for a decoded sample.

        Requests are processed in descending order of the utility of their
        decoded bundle; a bundle is kept only if it still fits alongside the
        already-kept selections. The returned selection is always
        capacity-feasible, and the repair is deterministic.

        Bundles with non-positive utility are dropped: the encoding is
        at-most-one per request, so leaving the request unserved (zero
        contribution) strictly dominates provisioning a non-positive-utility
        bundle and only frees capacity for other requests.
        """
        selected_map = dict(selected)
        order = {}
        for bundle in self.bundles:
            key = self._bundle_key(bundle)
            if key[0] in selected_map and selected_map[key[0]] == key[1]:
                order[key[0]] = (bundle["utility"], key[0], key[1])
        ordered_requests = sorted(order, key=lambda rid: order[rid], reverse=True)

        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        repaired = []
        for rid in ordered_requests:
            utility, bid = order[rid][0], selected_map[rid]
            if utility <= 0.0:
                continue
            demands = self._demands_of(rid, bid)
            if demands is None:
                continue
            edge_demands, mem_demands = demands
            feasible = True
            for edge, d in edge_demands.items():
                e = self._undirected_edge(edge)
                if edge_load[e] + d > self.edge_capacities.get(e, 0):
                    feasible = False
                    break
            if feasible:
                for node, d in mem_demands.items():
                    if mem_load[node] + d > self.memory_capacities.get(node, 0):
                        feasible = False
                        break
            if feasible:
                repaired.append((rid, bid))
                for edge, d in edge_demands.items():
                    edge_load[self._undirected_edge(edge)] += d
                for node, d in mem_demands.items():
                    mem_load[node] += d
        return repaired

    def decode_sample(self, sample, repair=False):
        selected = []
        for variable, bundle_key in self.variable_map.items():
            if sample[variable] == 1:
                selected.append(bundle_key)
        if repair:
            selected = self.repair_selection(selected)
        return selected

    def solution_energy(self, selected, edge_penalty=10.0, memory_penalty=10.0,
                        congestion_penalty=0.05, memory_congestion_penalty=0.05):
        """Exact Hamiltonian energy of a decoded selection, matching the QUBO.

        For a capacity-feasible selection the request-conflict penalty (A) is
        identically zero, so only utility, overload, and congestion terms
        contribute---the same convention used by the other solvers.
        """
        selected_map = dict(selected)
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        utility = 0.0
        for bundle in self.bundles:
            rid, bid = self._bundle_key(bundle)
            if selected_map.get(rid) == bid:
                utility += bundle["utility"]
                for e, d in bundle["edge_demands"].items():
                    edge_load[self._undirected_edge(e)] += d
                for n, d in bundle["memory_demands"].items():
                    mem_load[n] += d
        pen = 0.0
        for e, load in edge_load.items():
            cap = self.edge_capacities.get(e, 0)
            if load > cap:
                pen += edge_penalty * (load - cap) ** 2
            pen += congestion_penalty * load * load
        for n, load in mem_load.items():
            cap = self.memory_capacities.get(n, 0)
            if load > cap:
                pen += memory_penalty * (load - cap) ** 2
            pen += memory_congestion_penalty * load * load
        return -utility + pen

    def _utility_scale_penalties(self):
        """(A, B, D) anchored to the utility scale: ``p0 + eps`` with
        ``p0 = max(0, u_{r,b})``, per the manuscript's ``A > max u`` rule.
        Used when callers omit penalty arguments so the at-most-one and
        capacity constraints can never be silently under-enforced on
        high-utility instances.
        """
        p0 = max((max(0.0, float(bundle["utility"])) for bundle in self.bundles),
                 default=0.0)
        eps = max(1e-9, 1e-6 * p0)
        return p0 + eps, p0 + eps, p0 + eps

    def _resolve_penalties(self, penalty, edge_penalty, memory_penalty):
        if penalty is None or edge_penalty is None or memory_penalty is None:
            a, b, d = self._utility_scale_penalties()
            return (a if penalty is None else penalty,
                    b if edge_penalty is None else edge_penalty,
                    d if memory_penalty is None else memory_penalty)
        return penalty, edge_penalty, memory_penalty

    def _feed(self,penalty, edge_penalty, memory_penalty, congestion_penalty, memory_congestion_penalty):
        feed={"A": penalty, "C": congestion_penalty, "E": memory_congestion_penalty}
        for edge, index in self.edge_index.items():
            feed[f"B_{index}"] = (edge_penalty[edge] if isinstance(edge_penalty, dict) else edge_penalty)
        for node, index in self.memory_index.items():
            feed[f"D_{index}"] = (memory_penalty[node] if isinstance(memory_penalty, dict) else memory_penalty)
        return feed

    def to_qubo(self, penalty=None, edge_penalty=None, memory_penalty=None,
                congestion_penalty=0.05, memory_congestion_penalty=0.05):
        penalty, edge_penalty, memory_penalty = self._resolve_penalties(
            penalty, edge_penalty, memory_penalty)
        return self.model.to_qubo(feed_dict=self._feed(penalty, edge_penalty, memory_penalty, congestion_penalty, memory_congestion_penalty))

    def to_bqm(self, penalty=None, edge_penalty=None, memory_penalty=None,
               congestion_penalty=0.05, memory_congestion_penalty=0.05):
        penalty, edge_penalty, memory_penalty = self._resolve_penalties(
            penalty, edge_penalty, memory_penalty)
        return self.model.to_bqm(feed_dict=self._feed(penalty, edge_penalty, memory_penalty, congestion_penalty, memory_congestion_penalty))
