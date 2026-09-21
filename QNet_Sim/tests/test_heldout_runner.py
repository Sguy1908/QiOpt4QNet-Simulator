"""Tests for the held-out penalty-calibration runner.

The runner indexes the dict returned by ``contention_sweep_instances`` by
workload key. A mismatch between the two once made the whole experiment fail
with ``KeyError`` before doing any work, so these tests exercise that path on
a tiny configuration without touching ``results/``.
"""

import sys

import pytest

from experiments import (
    analyze_penalty_calibration_mechanism as mechanism,
    run_penalty_calibration_heldout as heldout,
)
from experiments.instances import (
    contention_sweep_instances,
    generate_chain_topology,
    instance_key,
)



def _tiny_topology():
    return generate_chain_topology(
        n_nodes=5, edge_capacity=3, memory_capacity=6,
        raw_fidelity=0.85, generation_prob=0.95,
    )


def test_instance_key_matches_generated_keys():
    counts = [2, 3]
    instances = contention_sweep_instances(_tiny_topology, counts, seed=10)
    assert set(instances) == {instance_key(n) for n in counts}
    for n in counts:
        assert instances[instance_key(n)]["n_requests"] == n


def test_heldout_scan_only_runs_on_tiny_config(monkeypatch):
    written = {}
    monkeypatch.setattr(heldout, "INSTANCE_SEEDS", [10, 11])
    monkeypatch.setattr(heldout, "REQUEST_COUNTS", [2, 3])
    monkeypatch.setattr(
        heldout, "_topology_cases", lambda full: [("chain5", _tiny_topology)]
    )
    monkeypatch.setattr(
        heldout, "_write_csv", lambda path, rows: written.setdefault(path, list(rows))
    )
    monkeypatch.setattr(sys, "argv", ["run_penalty_calibration_heldout", "--scan-only"])

    heldout.main()

    (rows,) = written.values()
    assert len(rows) == 4  # 2 seeds x 2 request counts
    assert {row["n_requests"] for row in rows} == {2, 3}
    # The CSV label format is part of the committed results; keep it stable.
    assert {row["instance"] for row in rows} == {"req2", "req3"}


def test_mechanism_analysis_uses_shared_key():
    # Guard against re-introducing a hand-formatted lookup key.
    import inspect

    assert "instances[instance_name]" not in inspect.getsource(mechanism)
    assert "instances[instance_name]" not in inspect.getsource(heldout)
