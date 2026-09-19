import math

import pytest

from hardware.parasitic_emission import (
    DEFAULT_PARAMS, PAPER_OPERATING_POINTS, count_rate_cps, decoy_bounds,
    dual_source_gain_qber, h2, key_rate_dual_source, key_rate_ideal, key_rate_tha,
    max_secure_distance, mean_photon_number, single_photon_yield_error,
    tha_delta, tha_phase_error,
)


def test_mu_eve_matches_paper_table1_scenarios():
    # Paper: mu_Eve = 0.0048, 0.0388, 0.0977 for the three driving configurations.
    expected = [0.0048, 0.0388, 0.0977]
    for op, mu in zip(PAPER_OPERATING_POINTS, expected):
        assert op.mu == pytest.approx(mu, abs=5e-5)


def test_count_rate_hits_published_anchors_and_turn_on():
    assert count_rate_cps(1.4) == pytest.approx(2.38e7)
    assert count_rate_cps(2.0) == pytest.approx(5.82e7)
    assert count_rate_cps(0.8) == 0.0
    assert count_rate_cps(0.0) == 0.0
    assert count_rate_cps(1.7) > count_rate_cps(1.4)


def test_count_rate_ramps_linearly_from_turn_on_to_first_anchor():
    mid = 0.5 * (0.8 + 1.4)
    assert count_rate_cps(mid) == pytest.approx(0.5 * 2.38e7)
    assert 0.0 < count_rate_cps(1.0) < count_rate_cps(1.2) < count_rate_cps(1.4)


def test_mean_photon_number_edge_cases():
    assert mean_photon_number(0.5, 1e-9) == 0.0
    assert mean_photon_number(2.0, 1.0) == float("inf")   # C*dt >= 1


def test_h2_bounds():
    assert h2(0.0) == 0.0 and h2(1.0) == 0.0
    assert h2(0.5) == pytest.approx(1.0)


def test_tha_delta_zero_without_leakage_and_monotone():
    assert tha_delta(0.0) == pytest.approx(0.0, abs=1e-15)
    vals = [tha_delta(m) for m in (0.001, 0.01, 0.1, 1.0)]
    assert vals == sorted(vals) and all(v > 0 for v in vals)


def test_tha_phase_error_reduces_to_e_x_without_leakage_and_caps():
    assert tha_phase_error(0.02, 0.0, 0.3) == pytest.approx(0.02)
    assert tha_phase_error(0.02, 0.4, 0.3) == 0.5      # Delta' >= 0.5
    assert tha_phase_error(0.02, 0.01, 0.0) == 0.5     # no yield


def test_gain_qber_without_parasitic_light_is_standard_form():
    gamma, eta = 0.48, 0.5
    q, e = dual_source_gain_qber(gamma, 0.0, eta, 0.1)
    s = 1 - math.exp(-gamma * eta)
    q_ref = 1 - (1 - DEFAULT_PARAMS.y0) * (1 - s)
    assert q == pytest.approx(q_ref)
    # Signal errors e_d dominate; dark counts contribute ~Y0/2.
    assert e == pytest.approx(DEFAULT_PARAMS.e_d, abs=1e-6)


def test_noise_raises_gain_and_qber_toward_half():
    q0, e0 = dual_source_gain_qber(0.48, 0.0, 0.5, 0.1)
    q1, e1 = dual_source_gain_qber(0.48, 0.1, 0.5, 0.1)
    q2, e2 = dual_source_gain_qber(0.48, 5.0, 0.5, 0.9)
    assert q1 > q0 and e1 > e0
    assert 0.0 <= e2 <= 0.5


def test_single_photon_quantities_consistent_with_pure_signal():
    y1, e1 = single_photon_yield_error(0.0, 0.4, 0.1)
    assert y1 == pytest.approx(1 - (1 - 0.4) * (1 - DEFAULT_PARAMS.y0))
    assert e1 == pytest.approx(DEFAULT_PARAMS.e_d, abs=1e-6)


def test_decoy_bounds_recover_true_single_photon_yield_without_noise():
    p = DEFAULT_PARAMS
    eta = p.eta_signal(20.0)
    obs = {n: dual_source_gain_qber(g, 0.0, eta, 0.0, p.y0, p.e_d)
           for n, g in (("s", p.signal), ("v", p.decoys[0]), ("w", p.decoys[1]))}
    y1_l, e1_u = decoy_bounds(obs["s"][0], obs["v"][0], obs["w"][0],
                              obs["s"][1], obs["v"][1], obs["w"][1])
    y1_true, _ = single_photon_yield_error(0.0, eta, 0.0)
    assert 0.0 < y1_l <= y1_true * (1 + 1e-9)
    assert y1_l == pytest.approx(y1_true, rel=0.05)
    assert e1_u >= p.e_d * 0.9


def test_ideal_key_rate_matches_paper_scale():
    assert key_rate_ideal(0.0) == pytest.approx(0.1, rel=0.05)
    # Paper Fig. 6: ideal curve reaches ~320 km.
    assert 300.0 <= max_secure_distance(key_rate_ideal) <= 340.0
    assert key_rate_ideal(10.0) > key_rate_ideal(50.0) > key_rate_ideal(200.0)


def test_tha_reduces_rate_and_range_monotonically_in_mu():
    mus = [op.mu for op in PAPER_OPERATING_POINTS]
    reaches = [max_secure_distance(lambda L, m=m: key_rate_tha(L, m)) for m in mus]
    assert reaches == sorted(reaches, reverse=True)
    assert reaches[0] < max_secure_distance(key_rate_ideal)
    assert key_rate_tha(0.0, mus[-1]) < key_rate_tha(0.0, mus[0]) < key_rate_ideal(0.0)


def test_tha_with_zero_leakage_equals_ideal():
    for L in (0.0, 25.0, 150.0):
        assert key_rate_tha(L, 0.0) == pytest.approx(key_rate_ideal(L), rel=1e-9)


def test_dual_source_worst_case_halves_rate_at_short_range():
    mu = PAPER_OPERATING_POINTS[-1].mu
    ratio = key_rate_dual_source(0.0, mu, mode="aware") / key_rate_ideal(0.0)
    assert 0.4 <= ratio <= 0.6   # paper: ~50 % reduction at short distance


def test_dual_source_penalty_vanishes_at_distance():
    mu = PAPER_OPERATING_POINTS[-1].mu
    assert key_rate_dual_source(30.0, mu, mode="aware") == pytest.approx(
        key_rate_ideal(30.0), rel=0.02)


def test_aware_rate_never_exceeds_ideal_and_matches_it_without_noise():
    # Regression: the aware bound once used the exact single-photon yield while
    # key_rate_ideal uses the (lower) decoy bound, so parasitic light appeared
    # to *raise* the certified rate by ~0.8 % beyond ~30 km.
    for op in PAPER_OPERATING_POINTS:
        for mu in (op.mu, 0.3, 1.0):
            for length in range(0, 341, 5):
                assert key_rate_dual_source(float(length), mu, mode="aware")                     <= key_rate_ideal(float(length)) + 1e-15
    for length in (0.0, 50.0, 200.0):
        assert key_rate_dual_source(length, 0.0, mode="aware") == pytest.approx(
            key_rate_ideal(length), rel=1e-12)


def test_aware_rate_at_50km_is_at_most_ideal_for_every_operating_point():
    ideal = key_rate_ideal(50.0)
    assert ideal > 0.0
    for op in PAPER_OPERATING_POINTS:
        aware = key_rate_dual_source(50.0, op.mu, mode="aware")
        assert 0.0 < aware <= ideal
        # noise only adds errors, so the loss is tiny but never a gain
        assert aware == pytest.approx(ideal, rel=1e-2)


def test_aware_rate_is_monotone_non_increasing_in_noise():
    for length in (0.0, 5.0, 20.0, 50.0):
        rates = [key_rate_dual_source(length, m, mode="aware")
                 for m in (0.0, 0.005, 0.04, 0.1, 0.3)]
        assert rates == sorted(rates, reverse=True)


def test_dual_source_rejects_unknown_mode():
    with pytest.raises(ValueError):
        key_rate_dual_source(1.0, 0.01, mode="bogus")
