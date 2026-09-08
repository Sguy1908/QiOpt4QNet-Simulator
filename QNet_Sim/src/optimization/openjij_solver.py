"""OpenJij helpers for QiOpt4QNet.

The low-level SA/SQA entry points remain available. The calibrated helpers make
three experiment modes explicit:

* conventional: utility-scale reference calibration;
* resource_aware: proposed edge/memory-aware calibration. one global coefficient per resource family;
* resource_aware_per: the same certtificate scoped per resource;
* fixed: user-supplied coefficients for legacy/sensitivity runs.

Soft congestion C/E is kept separate from hard-constraint A/B/D calibration.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import openjij as oj


def solve_sa(bqm, num_reads, seed=None):
    sampler = oj.SASampler()
    return sampler.sample(bqm, num_reads=num_reads, seed=seed)


def solve_sqa(bqm, num_reads, seed=None):
    sampler = oj.SQASampler()
    return sampler.sample(bqm, num_reads=num_reads, seed=seed)


def calibrated_coefficients(
    optimizer,
    strategy: str = "conventional",
    *,
    coefficient_scale: float = 1.0,
    fixed_coefficients: Optional[Mapping[str, float]] = None,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
) -> Dict[str, object]:
    """Return QUBO coefficients for a named calibration strategy."""
    if coefficient_scale <= 0:
        raise ValueError("coefficient_scale must be positive")

    strategy = strategy.lower().strip()

    if strategy == "conventional":
        from .conventional_calibrator import conventional_coefficients

        coeffs = conventional_coefficients(
            optimizer,
            congestion_penalty=congestion_penalty,
            memory_congestion_penalty=memory_congestion_penalty,
        )
    elif strategy in {"resource_aware", "resource-aware", "proposed"}:
        from .proposed_calibrator import proposed_global_coefficients

        coeffs = proposed_global_coefficients(
            optimizer,
            congestion_penalty=congestion_penalty,
            memory_congestion_penalty=memory_congestion_penalty,
        )
    elif strategy in {"resource_aware_per", "per_resource"}:
        from .proposed_calibrator import proposed_resource_coefficients

        coeffs = proposed_resource_coefficients(
            optimizer,
            congestion_penalty=congestion_penalty,
            memory_congestion_penalty=memory_congestion_penalty,
        )
    elif strategy == "fixed":
        if fixed_coefficients is None:
            raise ValueError("fixed_coefficients are required for strategy='fixed'")
        missing = {"A", "B", "D"} - set(fixed_coefficients)
        if missing:
            raise ValueError(
                f"fixed_coefficients missing required keys: {sorted(missing)}"
            )
        coeffs = {
            "A": float(fixed_coefficients["A"]),
            "B": float(fixed_coefficients["B"]),
            "C": float(fixed_coefficients.get("C", congestion_penalty)),
            "D": float(fixed_coefficients["D"]),
            "E": float(fixed_coefficients.get("E", memory_congestion_penalty)),
        }
    else:
        raise ValueError(
            "strategy must be one of: conventional, resource_aware, fixed"
        )

    # Sensitivity scaling applies only to hard constraints.
    for key in ("A", "B", "D"):
        value=coeffs[key]
        if isinstance(value,dict):
            coeffs[key] = {k: v * coefficient_scale for k, v in value.items()}
        else:
            coeffs[key] *= coefficient_scale

    return coeffs


def build_calibrated_bqm(
    optimizer,
    strategy: str = "conventional",
    *,
    coefficient_scale: float = 1.0,
    fixed_coefficients: Optional[Mapping[str, float]] = None,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
):
    """Build a BQM and return ``(bqm, coefficients)``."""
    coeffs = calibrated_coefficients(
        optimizer,
        strategy,
        coefficient_scale=coefficient_scale,
        fixed_coefficients=fixed_coefficients,
        congestion_penalty=congestion_penalty,
        memory_congestion_penalty=memory_congestion_penalty,
    )

    bqm = optimizer.to_bqm(
        penalty=coeffs["A"],
        edge_penalty=coeffs["B"],
        memory_penalty=coeffs["D"],
        congestion_penalty=coeffs["C"],
        memory_congestion_penalty=coeffs["E"],
    )
    return bqm, coeffs


def solve_calibrated(
    optimizer,
    *,
    sampler: str = "sa",
    strategy: str = "conventional",
    num_reads: int = 50,
    seed=None,
    coefficient_scale: float = 1.0,
    fixed_coefficients: Optional[Mapping[str, float]] = None,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
):
    """Build and solve a calibrated QUBO.

    Returns ``(response, coefficients)`` so experiments can always record the
    exact A/B/C/D/E values that were used.
    """
    bqm, coeffs = build_calibrated_bqm(
        optimizer,
        strategy,
        coefficient_scale=coefficient_scale,
        fixed_coefficients=fixed_coefficients,
        congestion_penalty=congestion_penalty,
        memory_congestion_penalty=memory_congestion_penalty,
    )

    sampler = sampler.lower().strip()
    if sampler == "sa":
        response = solve_sa(bqm, num_reads=num_reads, seed=seed)
    elif sampler == "sqa":
        response = solve_sqa(bqm, num_reads=num_reads, seed=seed)
    else:
        raise ValueError("sampler must be 'sa' or 'sqa'")

    return response, coeffs