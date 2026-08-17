import pytest

from fidelity.fidelity_model import FidelityModel


def test_werner_parameter_bounds_and_identity():
    assert FidelityModel.werner_parameter(1.0) == pytest.approx(1.0)
    # F=1/4 is the maximally mixed Werner state -> werner parameter 0
    assert FidelityModel.werner_parameter(0.25) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        FidelityModel.werner_parameter(-0.1)
    with pytest.raises(ValueError):
        FidelityModel.werner_parameter(1.1)


def test_entanglement_swapping_symmetric_and_bounds():
    assert FidelityModel.entanglement_swapping(1.0, 1.0) == pytest.approx(1.0)
    assert FidelityModel.entanglement_swapping(0.9, 0.7) == pytest.approx(
        FidelityModel.entanglement_swapping(0.7, 0.9))
    # Werner-state swapping of two maximally-mixed links (F=1/4) stays at F=1/4
    assert FidelityModel.entanglement_swapping(0.25, 0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        FidelityModel.entanglement_swapping(1.5, 0.9)


def test_purification_bbpssw_improves_above_threshold():
    improved = FidelityModel.purification_bbpssw(0.9)
    assert improved > 0.9
    assert improved <= 1.0
    # identity at exactly the 0.5 threshold
    assert FidelityModel.purification_bbpssw(0.5) == pytest.approx(0.5)


def test_purification_bbpssw_rejects_below_half():
    with pytest.raises(ValueError):
        FidelityModel.purification_bbpssw(0.3)


def test_end_to_end_fidelity_empty_and_single_link():
    assert FidelityModel.end_to_end_fidelity([]) == 0.0
    assert FidelityModel.end_to_end_fidelity([0.9]) == pytest.approx(0.9)


def test_end_to_end_fidelity_matches_repeated_swapping():
    links = [0.95, 0.9, 0.85]
    expected = FidelityModel.entanglement_swapping(
        FidelityModel.entanglement_swapping(links[0], links[1]), links[2])
    assert FidelityModel.end_to_end_fidelity(links) == pytest.approx(expected)
