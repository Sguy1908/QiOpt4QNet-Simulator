import pytest

from experiments.instances import generate_grid_topology
from hardware.parasitic_emission import PAPER_OPERATING_POINTS
from hardware.parasitic_network import (
    apply_parasitic_emission, assign_link_lengths, bundle_edges, filtered_mu,
    link_is_secure, parasitic_link_fidelity, run_parasitic_network_study,
    secure_bundles,
)

WORST = PAPER_OPERATING_POINTS[-1]


def _grid(distance=25.0):
    return lambda: generate_grid_topology(3, 3, edge_capacity=4, memory_capacity=8,
                                          raw_fidelity=0.92, distance=distance)


def _links(topo):
    for u, v, d in topo["network"].graph.edges(data=True):
        yield tuple(sorted((u, v))), d["data"]


def test_no_emission_leaves_fidelity_untouched():
    assert parasitic_link_fidelity(0.9, 5.0, 0.0) == 0.9
    topo = _grid()()
    before = {e: l.raw_fidelity for e, l in _links(topo)}
    info = apply_parasitic_emission(topo, 0.0, 0.0)
    assert {e: l.raw_fidelity for e, l in _links(topo)} == before
    assert info["insecure_edges"] == set() and info["mean_delta_f"] == 0.0


def test_fidelity_degrades_with_mu_and_shrinks_with_distance():
    f_short = parasitic_link_fidelity(0.92, 2.0, WORST.mu)
    f_long = parasitic_link_fidelity(0.92, 25.0, WORST.mu)
    assert f_short < f_long <= 0.92
    assert parasitic_link_fidelity(0.92, 2.0, 0.01) > f_short
    assert parasitic_link_fidelity(0.30, 0.0, 10.0) >= 0.25   # floor


def test_filter_scales_mu_and_removes_insecurity():
    assert filtered_mu(2.0, 1.6e-9, 20.0) == pytest.approx(WORST.mu / 100.0)
    assert not link_is_secure(20.0, WORST.mu)
    assert link_is_secure(20.0, filtered_mu(2.0, 1.6e-9, 20.0))
    assert link_is_secure(500.0, 0.0)


def test_apply_marks_long_links_insecure_and_updates_link_params():
    topo = _grid(distance=20.0)()
    info = apply_parasitic_emission(topo, WORST.voltage, WORST.pulse_width_s)
    assert len(info["insecure_edges"]) == len(topo["edges"])
    for e in topo["edges"]:
        assert topo["link_params"][e]["raw_fidelity"] < 0.92


def test_assign_link_lengths_is_deterministic_and_in_range():
    a, b = _grid()(), _grid()()
    assign_link_lengths(a, (2.0, 30.0), seed=5)
    assign_link_lengths(b, (2.0, 30.0), seed=5)
    da = {e: l.distance for e, l in _links(a)}
    assert da == {e: l.distance for e, l in _links(b)}
    assert all(2.0 <= d <= 30.0 for d in da.values())


def test_secure_bundles_filters_on_edge_overlap():
    bundles = [{"edge_demands": {("A", "B"): 1}}, {"edge_demands": {("C", "B"): 1}}]
    kept = secure_bundles(bundles, {("B", "C")})
    assert kept == [bundles[0]]
    assert bundle_edges(bundles[1]) == {("B", "C")}


def test_study_scenarios_and_security_awareness():
    rows = run_parasitic_network_study(_grid(), WORST.voltage, WORST.pulse_width_s,
                                       n_requests=8, seed=3,
                                       allocators=["congestion_aware_greedy"],
                                       length_range=(2.0, 30.0))
    by = {r["scenario"]: r for r in rows}
    assert set(by) == {"clean", "blind", "aware"}
    assert all(r["feasible"] for r in rows)
    assert by["aware"]["insecure_admitted"] == 0
    assert by["blind"]["insecure_admitted"] > 0
    assert by["aware"]["total_utility"] <= by["blind"]["total_utility"] + 1e-9
    assert by["blind"]["total_utility"] <= by["clean"]["total_utility"] + 1e-9


def test_study_with_strong_filter_matches_clean_admission():
    rows = run_parasitic_network_study(_grid(), WORST.voltage, WORST.pulse_width_s,
                                       n_requests=8, seed=3, filter_db=30.0,
                                       allocators=["congestion_aware_greedy"],
                                       length_range=(2.0, 30.0))
    by = {r["scenario"]: r for r in rows}
    assert by["aware"]["n_insecure_edges"] == 0
    assert by["aware"]["accepted"] == by["clean"]["accepted"]
