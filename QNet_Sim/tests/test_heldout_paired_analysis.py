"""Tests for the subset option of the held-out paired analysis."""

import pytest

from experiments.analyze_penalty_calibration_heldout import (
    CALIBRATIONS,
    CONTRASTS,
    SAMPLERS,
    analyze_rows,
)
from experiments.run_penalty_calibration_heldout import NUM_READS, SOLVER_SEEDS


def _rows(instance_seeds, feasible):
    """Synthetic solver runs; ``feasible[calibration]`` is the raw feasible rate."""
    rows = []
    for instance_seed in instance_seeds:
        for sampler in SAMPLERS:
            for solver_seed in SOLVER_SEEDS:
                for calibration in CALIBRATIONS:
                    rate = feasible[calibration](instance_seed)
                    rows.append(
                        {
                            "topology": "chain",
                            "instance_seed": instance_seed,
                            "n_requests": 8,
                            "sampler": sampler,
                            "solver_seed": solver_seed,
                            "calibration": calibration,
                            "n_bundles": 10,
                            "bqm_variables": 20,
                            "oracle_utility": 5.0,
                            "oracle_status": "OPTIMAL",
                            "A": 1.0,
                            "C": 0.0,
                            "E": 0.0,
                            "coefficient_scale": 1.0,
                            "n_reads": NUM_READS,
                            "raw_feasible_rate": rate,
                            "raw_mean_overload_units": 0.0,
                            "raw_optimality_gap_pct": 0.0,
                            "repaired_optimality_gap_pct": 50.0 * (1.0 - rate),
                        }
                    )
    return rows


def _feasible(global_changed_seeds):
    return {
        "conventional": lambda seed: 0.0,
        # The global arm helps only where it changed a coefficient.
        "resource_aware": lambda seed: 1.0 if seed in global_changed_seeds else 0.0,
        "resource_aware_per": lambda seed: 1.0,
    }


def _global_vs_conventional(summary, metric, sampler="sa"):
    (row,) = [r for r in summary if r["metric"] == metric and r["sampler"] == sampler]
    return row


def test_restriction_removes_the_diluting_zeros():
    rows = _rows([1, 2, 3, 4], _feasible({1}))
    contrast = [CONTRASTS[0]]

    pooled = analyze_rows(rows, contrasts=contrast)
    subset = analyze_rows(rows, contrasts=contrast, restrict_to={("chain", 1, 8)})

    assert _global_vs_conventional(pooled, "raw_feasible_rate")["n_instances"] == 4
    assert _global_vs_conventional(pooled, "raw_feasible_rate")[
        "mean_paired_difference"
    ] == pytest.approx(0.25)
    restricted = _global_vs_conventional(subset, "raw_feasible_rate")
    assert restricted["n_instances"] == 1
    assert restricted["mean_paired_difference"] == pytest.approx(1.0)
    assert _global_vs_conventional(subset, "repaired_reference_gap_pct")[
        "mean_paired_difference"
    ] == pytest.approx(-50.0)


def test_default_arguments_keep_all_contrasts():
    summary = analyze_rows(_rows([1, 2], _feasible({1})))
    assert len(summary) == len(CONTRASTS) * len(SAMPLERS) * 3
