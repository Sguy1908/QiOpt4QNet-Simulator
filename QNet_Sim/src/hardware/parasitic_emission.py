"""Parasitic-emission (VOA electroluminescence) side-channel model.

Implements the analytic model of Li et al., "Security risks of VOA-induced
luminescence in chip-based quantum key distribution", npj Quantum Information
(2026), doi:10.1038/s41534-026-01365-1.  A forward-biased p-i-n variable
optical attenuator (VOA) in a photonic QKD transmitter emits spontaneous
electroluminescence (EL, centred near 1107 nm).  The paper analyses two
consequences that this module reproduces:

* **Passive Trojan-horse attack (THA)** -- EL emitted upstream of the encoder
  is modulated with the signal and readable by an eavesdropper (Eqs. 12-19).
* **Dual-source flaw** -- EL emitted downstream of the encoder reaches Bob as an
  unmodulated, independent Poissonian noise source that biases decoy-state
  parameter estimation (Eqs. 20-24, 60-68).

What is and is not taken from the paper
---------------------------------------
Taken directly: the closed-form gain/QBER expressions (Eqs. 23-24), the THA
leakage bound (Eqs. 16-19), the pulse-count-rate to mean-photon-number map
(Eqs. 12-14, 20), the Table 1 system parameters and the two measured count
rates ``C(1.4 V)`` and ``C(2 V)``.

Reconstructed here (the authors' code is not public): the two-decoy-state
estimator (Ma et al., PRA 72, 012326 -- the paper's ref. 85), the interpolation
of ``C(U)`` between the two published anchors, and the "naive" versus
"noise-aware" key-rate comparison for the dual-source flaw.  The paper states
that the dual-source rate has no explicit analytic expression, so the numbers
produced here for that scenario are a reconstruction, not a reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Table 1 of the paper.
E_D = 0.0061          # intrinsic misalignment error
Y0 = 2e-8             # detector background (dark-count) yield
ETA_BOB = 0.78        # Bob's detection efficiency at the signal wavelength
F_EC = 1.2            # error-correction inefficiency
ALPHA_DB_KM = 0.2     # signal-band fibre loss (1550 nm)
# Parasitic (1077 nm) light, Fig. 7/8 scenario.
ALPHA_PARASITIC_DB_KM = 0.8
ETA_BOB_PARASITIC = 0.25
# Decoy intensities used in the paper's simulations.
SIGNAL_INTENSITY = 0.48
DECOY_INTENSITIES = (0.02, 0.001)

# Measured VOA emission count rates (cps), Table 1.
_MEASURED_COUNT_RATE = {1.4: 2.38e7, 2.0: 5.82e7}
# Turn-on voltage of the p-i-n junction (Fig. 3c-d): no detectable EL below it.
TURN_ON_VOLTAGE = 0.8


def h2(x: float) -> float:
    """Binary entropy in bits, clamped so that h2(0) = h2(1) = 0."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def count_rate_cps(voltage: float) -> float:
    """EL photon count rate C(U) of the VOA at drive voltage ``U``.

    Only C(1.4 V) and C(2.0 V) are published numerically (Table 1).  Between
    them, and above 2 V, C is interpolated/extrapolated log-linearly; between
    the 0.8 V turn-on and 1.4 V it is interpolated linearly from zero to the
    1.4 V anchor, and it is zero below the turn-on.  This is a model of the
    trend visible in Fig. 3c, not a measurement: use the two anchors when
    exactness matters.
    """
    if voltage <= TURN_ON_VOLTAGE:
        return 0.0
    (u1, c1), (u2, c2) = sorted(_MEASURED_COUNT_RATE.items())
    if voltage < u1:
        return c1 * (voltage - TURN_ON_VOLTAGE) / (u1 - TURN_ON_VOLTAGE)
    slope = math.log(c2 / c1) / (u2 - u1)
    return c1 * math.exp(slope * (voltage - u1))


def mean_photon_number(voltage: float, pulse_width_s: float) -> float:
    """Mean EL photons per signal-pulse window, ``mu = -ln(1 - C(U) dt)``
    (Eqs. 12-14 and 20).  Saturates at a large value if ``C dt >= 1``."""
    q = count_rate_cps(voltage) * pulse_width_s
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return float("inf")
    return -math.log(1.0 - q)


@dataclass(frozen=True)
class QKDParams:
    """Decoy-state BB84 link parameters (defaults: Table 1 of the paper)."""

    e_d: float = E_D
    y0: float = Y0
    eta_bob: float = ETA_BOB
    f_ec: float = F_EC
    alpha_db_km: float = ALPHA_DB_KM
    alpha_parasitic_db_km: float = ALPHA_PARASITIC_DB_KM
    eta_bob_parasitic: float = ETA_BOB_PARASITIC
    signal: float = SIGNAL_INTENSITY
    decoys: Tuple[float, float] = DECOY_INTENSITIES
    # Efficient-BB84 sifting: q = 1/2 with p_Z -> 1 gives R(0 km) ~ 0.1 (Fig. 8).
    sift_factor: float = 0.5

    def eta_signal(self, length_km: float) -> float:
        return self.eta_bob * 10.0 ** (-self.alpha_db_km * length_km / 10.0)

    def eta_parasitic(self, length_km: float) -> float:
        return self.eta_bob_parasitic * 10.0 ** (-self.alpha_parasitic_db_km * length_km / 10.0)


DEFAULT_PARAMS = QKDParams()


# --------------------------------------------------------------------------
# Dual-source gain and QBER (Eqs. 23-24)
# --------------------------------------------------------------------------

def dual_source_gain_qber(gamma: float, mu_el: float, eta: float, eta_p: float,
                          y0: float = Y0, e_d: float = E_D,
                          e0: float = 0.5) -> Tuple[float, float]:
    """Overall gain ``Q`` and QBER ``E`` at Bob for a signal of mean photon
    number ``gamma`` plus independent parasitic noise ``mu_el`` (Eqs. 23-24).

    ``eta``/``eta_p`` are the end-to-end transmittances of signal / parasitic
    light.  Noise photons and dark counts err with probability ``e0`` = 1/2.
    """
    s = 1.0 - math.exp(-gamma * eta)      # P(signal click)
    n = 1.0 - math.exp(-mu_el * eta_p)    # P(noise click)
    q = 1.0 - (1.0 - y0) * (1.0 - s) * (1.0 - n)
    eq = (y0 * e0 + e_d * s + e0 * n
          - y0 * e0 * e_d * s
          - y0 * e0 * e0 * n
          - e_d * e0 * s * n
          + y0 * e0 * e0 * e_d * s * n)
    return q, (eq / q if q > 0 else 0.0)


def single_photon_yield_error(mu_el: float, eta: float, eta_p: float,
                              y0: float = Y0, e_d: float = E_D,
                              e0: float = 0.5) -> Tuple[float, float]:
    """Exact single-photon yield ``Y1`` and error rate ``e1`` in the presence
    of noise (Eq. 66 with ``i = 1`` averaged over the noise Poisson law)."""
    n = 1.0 - math.exp(-mu_el * eta_p)
    y1 = 1.0 - (1.0 - eta) * (1.0 - y0) * (1.0 - n)
    e1y1 = (y0 * e0 + eta * e_d + n * e0
            - eta * n * e_d * e0
            - eta * y0 * e_d * e0
            - n * y0 * e0 * e0
            + eta * n * y0 * e_d * e0 * e0)
    return y1, (e1y1 / y1 if y1 > 0 else 0.0)


# --------------------------------------------------------------------------
# Two-decoy-state estimator (Ma et al. 2005)
# --------------------------------------------------------------------------

def decoy_bounds(qs: float, qv: float, qw: float, es: float, ev: float, ew: float,
                 params: QKDParams = DEFAULT_PARAMS) -> Tuple[float, float]:
    """Lower bound ``Y1^L`` on the single-photon yield and upper bound
    ``e1^U`` on its error rate from the gains / QBERs at the signal ``s`` and
    the two decoy intensities ``v > w``.  Clamped to physical ranges."""
    s, (v, w) = params.signal, params.decoys
    denom = s * v - s * w - v * v + w * w
    y1_l = (s / denom) * (qv * math.exp(v) - qw * math.exp(w)
                          - ((v * v - w * w) / (s * s)) * (qs * math.exp(s) - params.y0))
    y1_l = max(y1_l, 0.0)
    if y1_l <= 0.0:
        return 0.0, 0.5
    e1_u = (ev * qv * math.exp(v) - ew * qw * math.exp(w)) / ((v - w) * y1_l)
    return y1_l, min(max(e1_u, 0.0), 0.5)


# --------------------------------------------------------------------------
# Passive Trojan-horse attack (Eqs. 16-19)
# --------------------------------------------------------------------------

def tha_delta(mu_eve: float) -> float:
    """Basis-dependence ``Delta`` of Alice's source when Eve holds a coherent
    copy of the polarisation state with mean photon number ``mu_eve`` (Eq. 19).
    """
    r = math.exp(-mu_eve) * (math.cosh(mu_eve / math.sqrt(2.0))
                             + 0.5 * math.sinh(mu_eve / math.sqrt(2.0)))
    return 0.5 * (1.0 - r)


def tha_phase_error(e_x: float, delta: float, y1: float) -> float:
    """Phase-error bound ``e_X'`` under the GLLP-Koashi approach (Eqs. 17-18).
    Returns 0.5 (no key) when ``Delta/Y1`` leaves the valid range."""
    if y1 <= 0.0:
        return 0.5
    dp = delta / y1
    if dp >= 0.5:
        return 0.5
    e = (e_x + 4.0 * dp * (1.0 - dp) * (1.0 - 2.0 * e_x)
         + 4.0 * (1.0 - 2.0 * dp) * math.sqrt(dp * (1.0 - dp) * e_x * (1.0 - e_x)))
    return min(e, 0.5)


def mu_eve_from_voa(voltage: float, pulse_width_s: float) -> float:
    """Eve's mean photon number per pulse for a given VOA operating point
    (Eq. 14).  Worst case: ideal detector with unit efficiency, no dark counts.
    """
    return mean_photon_number(voltage, pulse_width_s)


# --------------------------------------------------------------------------
# Secure key rates
# --------------------------------------------------------------------------

def _observed(length_km: float, mu_el: float, params: QKDParams):
    eta = params.eta_signal(length_km)
    eta_p = params.eta_parasitic(length_km)
    out = {}
    for name, g in (("s", params.signal), ("v", params.decoys[0]), ("w", params.decoys[1])):
        out[name] = dual_source_gain_qber(g, mu_el, eta, eta_p, params.y0, params.e_d)
    return out


def key_rate_ideal(length_km: float, params: QKDParams = DEFAULT_PARAMS) -> float:
    """Decoy-state BB84 rate with no parasitic light (red curve, Figs. 6/8)."""
    return key_rate_dual_source(length_km, 0.0, params, mode="naive")


def key_rate_tha(length_km: float, mu_eve: float,
                 params: QKDParams = DEFAULT_PARAMS) -> float:
    """Decoy-state BB84 rate under the passive THA (Eq. 16 / Eq. 50).

    Bob's statistics are unaffected (Eve only reads her copy), so the decoy
    estimates ``Y1^L`` and ``e1^U`` come from clean gains; the leakage enters
    through the phase-error bound ``e_X'``.
    """
    obs = _observed(length_km, 0.0, params)
    y1, e1 = decoy_bounds(obs["s"][0], obs["v"][0], obs["w"][0],
                          obs["s"][1], obs["v"][1], obs["w"][1], params)
    if y1 <= 0.0:
        return 0.0
    e_prime = tha_phase_error(e1, tha_delta(mu_eve), y1)
    q1 = params.signal * math.exp(-params.signal) * y1
    qs, es = obs["s"]
    rate = params.sift_factor * (q1 * (1.0 - h2(e_prime)) - qs * params.f_ec * h2(es))
    return max(rate, 0.0)


def key_rate_dual_source(length_km: float, mu_el: float,
                         params: QKDParams = DEFAULT_PARAMS,
                         mode: str = "naive") -> float:
    """Decoy-state BB84 rate with parasitic noise at Bob (Eq. 25).

    ``mode="naive"``: Alice and Bob feed the *observed* (noise-inflated) gains
    into the standard two-decoy estimator, i.e. they trust the source model --
    the situation the paper calls the dual-source flaw.

    ``mode="aware"``: the parasitic source is characterised separately (its
    ``mu_EL`` is known), so its clicks are removed from the decoy statistics and
    the signal-photon contribution ``Q1 = s e^-s Y1^L`` is the *same* certified
    bound as the noise-free analysis.  Noise cannot create key: it only adds
    errors, and in the worst case all of them are charged to the single-photon
    key bits, ``e1 = e1^U + (E1 Y1)_noise / Y1^L`` (capped at 1/2), where
    ``(E1 Y1)_noise`` is the increase of Eq. 66's single-photon error mass over
    its ``mu_EL = 0`` value.  Error correction is charged for the *observed*
    (noisy) signal gain and QBER.  By construction the certified rate never
    exceeds ``key_rate_ideal`` at the same distance and equals it at
    ``mu_EL = 0``.
    """
    obs = _observed(length_km, mu_el, params)
    qs, es = obs["s"]
    if mode == "naive":
        y1, e1 = decoy_bounds(qs, obs["v"][0], obs["w"][0],
                              es, obs["v"][1], obs["w"][1], params)
    elif mode == "aware":
        clean = _observed(length_km, 0.0, params)
        y1, e1_clean = decoy_bounds(clean["s"][0], clean["v"][0], clean["w"][0],
                                    clean["s"][1], clean["v"][1], clean["w"][1], params)
        if y1 <= 0.0:
            return 0.0
        eta = params.eta_signal(length_km)
        eta_p = params.eta_parasitic(length_km)
        y_n, e_n = single_photon_yield_error(mu_el, eta, eta_p, params.y0, params.e_d)
        y_0, e_0 = single_photon_yield_error(0.0, eta, eta_p, params.y0, params.e_d)
        extra_errors = max(e_n * y_n - e_0 * y_0, 0.0)
        e1 = min(e1_clean + extra_errors / y1, 0.5)
    else:
        raise ValueError(f"mode must be 'naive' or 'aware' (got {mode!r})")
    q1 = params.signal * math.exp(-params.signal) * y1
    rate = params.sift_factor * (q1 * (1.0 - h2(e1)) - qs * params.f_ec * h2(es))
    return max(rate, 0.0)


def max_secure_distance(rate_fn, l_max_km: float = 400.0, step_km: float = 0.5) -> float:
    """Largest distance (km) on a ``step_km`` grid at which ``rate_fn(L) > 0``,
    scanning outwards from 0 and stopping at the first zero."""
    last = 0.0
    length = 0.0
    while length <= l_max_km:
        if rate_fn(length) <= 0.0:
            return last
        last = length
        length += step_km
    return last


# --------------------------------------------------------------------------
# VOA operating points used in the paper (Table 1 / Figs. 6-8)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatingPoint:
    label: str
    voltage: float
    pulse_width_s: float

    @property
    def mu(self) -> float:
        return mean_photon_number(self.voltage, self.pulse_width_s)


PAPER_OPERATING_POINTS = (
    OperatingPoint("1.4 V, 200 ps", 1.4, 200e-12),
    OperatingPoint("1.4 V, 1.6 ns", 1.4, 1.6e-9),
    OperatingPoint("2.0 V, 1.6 ns", 2.0, 1.6e-9),
)
