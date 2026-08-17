import pytest

from hardware.profiles import (
    IDEAL_PROFILE, SUPERCONDUCTING_PROFILE, HardwareProfile,
    run_hardware_profile_comparison, ALL_PROFILES,
)


def test_profile_fields_present():
    for p in (IDEAL_PROFILE, SUPERCONDUCTING_PROFILE):
        assert p.name
        assert p.t1_us > 0 and p.t2_us > 0
        assert p.bond_dim_max > 0


def test_ideal_has_longest_coherence():
    assert IDEAL_PROFILE.t1_us >= SUPERCONDUCTING_PROFILE.t1_us
    assert IDEAL_PROFILE.t2_us >= SUPERCONDUCTING_PROFILE.t2_us


def test_profile_comparison_runs():
    topo = _chain_topo()
    rows = run_hardware_profile_comparison(
        topo, n_slots=3, mean_rate=1.0,
        profiles=[IDEAL_PROFILE, SUPERCONDUCTING_PROFILE], seed=7)
    assert len(rows) == 2
    assert rows[0]["profile"] == "ideal"
    assert rows[1]["profile"] == "superconducting_2024"
    assert rows[0]["served_ratio"] >= 0.0


def test_short_coherence_reduces_delivered_fidelity():
    topo = _chain_topo()
    rows = run_hardware_profile_comparison(
        topo, n_slots=5, mean_rate=1.5,
        profiles=[IDEAL_PROFILE, SUPERCONDUCTING_PROFILE], seed=7)
    by_name = {r["profile"]: r for r in rows}
    assert by_name["ideal"]["mean_delivered_fidelity"] >= \
        by_name["superconducting_2024"]["mean_delivered_fidelity"]


def test_profiles_have_citations_except_ideal():
    for p in ALL_PROFILES:
        if p.name == "ideal":
            assert p.source is None
            assert p.source_url_or_doi is None
        else:
            assert p.source, f"{p.name} missing a literature citation"
            assert p.source_url_or_doi, f"{p.name} missing a checkable source URL/DOI"
            assert "LITERATURE-CALIBRATED" in p.description


def _chain_topo():
    from experiments.instances import generate_chain_topology
    return lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
