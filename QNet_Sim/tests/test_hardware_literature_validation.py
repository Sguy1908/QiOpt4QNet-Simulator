import pytest

from hardware.literature_validation import run_literature_calibration_report
from hardware.profiles import ALL_PROFILES


def _chain_topo():
    from experiments.instances import generate_chain_topology
    return lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)


def test_literature_report_runs_and_labeled():
    res = run_literature_calibration_report(_chain_topo(), n_slots=5,
                                            mean_rate=1.0, seed=7)
    rows = res["rows"]
    assert len(rows) == len(ALL_PROFILES)
    for row in rows:
        # Guardrail: this study must never claim to be a physical hardware
        # experiment, only a literature-calibrated one.
        assert row["label"] in ("literature-calibrated", "non-physical reference")
        if row["profile"] != "ideal":
            assert "LITERATURE-CALIBRATED" in row["label"].upper() or \
                   row["label"] == "literature-calibrated"
            assert row["source"] is not None
            assert row["source_url_or_doi"] is not None
        assert "rank_consistent" in row
        assert isinstance(row["rank_consistent"], bool)


def test_literature_report_fidelity_rank_consistent_with_t2_order():
    res = run_literature_calibration_report(_chain_topo(), n_slots=5,
                                            mean_rate=1.0, seed=7)
    rows = res["rows"]
    # every row must self-report as rank-consistent (delivered fidelity
    # never worse than a shorter-T2-cited profile)
    assert all(r["rank_consistent"] for r in rows)


def test_literature_report_only_ideal_lacks_a_source():
    res = run_literature_calibration_report(_chain_topo(), n_slots=5,
                                            mean_rate=1.0, seed=7)
    sourced = [r["profile"] for r in res["rows"] if r["source"] is not None]
    unsourced = [r["profile"] for r in res["rows"] if r["source"] is None]
    assert unsourced == ["ideal"]
    assert len(sourced) == len(ALL_PROFILES) - 1
