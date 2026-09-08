"""Resource-aware penalty calibration for the QiOpt4QNet QUBO.

For each edge/memory resource and candidate bundle, ask how much the squared
capacity-overload penalty decreases when that bundle is removed from a
violating configuration. The coefficient is chosen above the largest ratio
of positive bundle utility to that penalty decrease over reachable competing
loads.

This is an instance-aware *single-bundle removal bound* for the capacity terms.
The request-conflict coefficient A intentionally remains identical to the
conventional utility-scale rule so that experiments isolate the effect of
resource-aware B/D calibration.

Reachable loads are computed with bounded dynamic programming rather than a
full Cartesian-product enumeration.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

from .conventional_calibrator import penalty_epsilon, positive_utility_scale

BundleKey = Tuple[str, str]


def _demands_by_request(
    key_demand_pairs: Sequence[Tuple[BundleKey, int]],
    excluded_request: str,
) -> Dict[str, set[int]]:
    """Possible contributions to one resource from each competing request."""
    grouped: Dict[str, set[int]] = {}
    for key, demand in key_demand_pairs:
        request_id = key[0]
        if request_id == excluded_request:
            continue
        demand = int(demand)
        if demand < 0:
            raise ValueError("resource demands must be nonnegative")
        # Zero is always an option: reject the request or use another bundle.
        grouped.setdefault(request_id, {0}).add(demand)
    return grouped


def possible_loads(
    key_demand_pairs: Sequence[Tuple[BundleKey, int]],
    excluded_request: str,
    *,
    capacity: int | None = None,
    candidate_demand: int | None = None,
) -> set[int]:
    """Return reachable resource loads from all requests except one.

    If ``capacity`` and ``candidate_demand`` are supplied, dynamic programming
    is safely capped. To find the smallest load capable of causing a violation,
    no load above ``capacity + max_competing_single_request_demand`` is needed.
    """
    grouped = _demands_by_request(key_demand_pairs, excluded_request)

    load_cap = None
    if capacity is not None and candidate_demand is not None:
        capacity = int(capacity)
        candidate_demand = int(candidate_demand)
        if capacity < 0 or candidate_demand < 0:
            raise ValueError("capacity and candidate_demand must be nonnegative")
        max_step = max((max(options) for options in grouped.values()), default=0)
        load_cap = capacity + max_step

    loads = {0}
    for options in grouped.values():
        updated = {load + demand for load in loads for demand in options}
        if load_cap is not None:
            updated = {load for load in updated if load <= load_cap}
        loads = updated

    return loads


def _penalty_drop(other_load: int, demand: int, capacity: int) -> float:
    """Squared-overload reduction obtained by removing one selected bundle."""
    before = max(0, other_load + demand - capacity) ** 2
    after = max(0, other_load - capacity) ** 2
    return float(before - after)

def resource_bounds(
    grouped_demands: Mapping[object, Sequence[Tuple[BundleKey, int]]],
    capacities: Mapping[object, int],
    utilities: Mapping[BundleKey, float],
) -> Dict[object, float]:
    """Return the single-removal coefficient bound for each resource.
    A resource whose capacity can never be exceeded maps to 0.0.
    """
    bounds: Dict[object, float] = {}
    for resource, key_demand_pairs in grouped_demands.items():
        if resource not in capacities:
            raise ValueError(f"missing capacity for resource {resource!r}")
        capacity = int(capacities[resource])
        if capacity < 0:
            raise ValueError("resource capacity must be nonnegative")
        bounds[resource] = 0.0
        loads_cache: Dict[Tuple[str, int], set[int]] = {}
        for key, demand_raw in key_demand_pairs:
            demand = int(demand_raw)
            if demand <=0:
                continue
            request_id = key[0]
            cache_key=(request_id, demand)
            if cache_key not in loads_cache:
                loads_cache[cache_key] = possible_loads(
                    key_demand_pairs,
                    request_id,
                    capacity=capacity,
                    candidate_demand=demand,
                )
            loads = loads_cache[cache_key]

            violating=[load for load in loads if load+demand>capacity]
            if not violating:
                continue
            other_load=min(violating)
            delta = _penalty_drop(other_load, demand, capacity)
            if delta <= 0:
                continue
            utility=max(0.0, float(utilities.get(key, 0.0)))
            bounds[resource] = max(bounds[resource], utility / delta)
    return bounds

def coefficient_bound(
        grouped_demands: Mapping[object, Sequence[Tuple[BundleKey, int]]],
        capacities: Mapping[object, int],
        utilities: Mapping[BundleKey, float],
) -> float:
    """Return the largest resource-aware single-removal coefficient bound."""
    return max(
        resource_bounds(grouped_demands, capacities, utilities).values(),
        default=0.0,
    )



def proposed_global_coefficients(
    optimizer,
    *,
    safety_factor: float = 1.0,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
) -> Dict[str, float]:
    """Return resource-aware QUBO coefficients for one problem instance."""
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")

    utilities: Dict[BundleKey, float] = {}
    for bundle in optimizer.bundles:
        key = optimizer._bundle_key(bundle)
        utilities[key] = max(0.0, float(bundle["utility"]))

    p0 = positive_utility_scale(optimizer)
    epsilon = penalty_epsilon(p0)

    edge_bound = coefficient_bound(
        optimizer.edge_demands,
        optimizer.edge_capacities,
        utilities,
    )
    memory_bound = coefficient_bound(
        optimizer.memory_demands,
        optimizer.memory_capacities,
        utilities,
    )

    return {
        "A": safety_factor * p0 + epsilon,
        "B": safety_factor * edge_bound + epsilon,
        "C": float(congestion_penalty),
        "D": safety_factor * memory_bound + epsilon,
        "E": float(memory_congestion_penalty),
    }

def proposed_resource_coefficients(
    optimizer,
    *,
    safety_factor: float = 1.0,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
) -> Dict[str, object]:
    """Return per-resource QUBO coefficients for one problem instance."""
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")

    utilities: Dict[BundleKey, float] = {}
    for bundle in optimizer.bundles:
        key = optimizer._bundle_key(bundle)
        utilities[key] = max(0.0, float(bundle["utility"]))

    p0 = positive_utility_scale(optimizer)
    epsilon = penalty_epsilon(p0)

    edge_bounds = resource_bounds(
        optimizer.edge_demands,
        optimizer.edge_capacities,
        utilities,
    )
    memory_bounds = resource_bounds(
        optimizer.memory_demands,
        optimizer.memory_capacities,
        utilities,
    )

    return {
        "A": safety_factor * p0 + epsilon,
        "B": {e: safety_factor * v + epsilon for e, v in edge_bounds.items()},
        "C": float(congestion_penalty),
        "D": {n: safety_factor * v + epsilon for n, v in memory_bounds.items()},
        "E": float(memory_congestion_penalty),
    }