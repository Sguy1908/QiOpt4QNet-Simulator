"""Hardware testbed parameter profiles (paper Future Work item iv).

Bridges the idealized simulation parameters to physically realistic values for
current quantum hardware families.  Each profile captures the coherence times
(T1/T2) that govern end-to-end fidelity decay, the per-anneal duration, the
gate/readout error rates and the maximum tensor-network bond dimension the
platform can realistically sustain.

``run_hardware_profile_comparison`` replays the decoherence-aware streaming
experiment under each profile and reports how the delivered fidelity and
served utility degrade as coherence times drop.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@dataclass
class HardwareProfile:
    name: str
    t1_us: float
    t2_us: float
    anneal_time_us: float
    gate_error_rate: float
    readout_error_rate: float
    bond_dim_max: int
    description: str
    source: Optional[str] = None
    source_url_or_doi: Optional[str] = None


IDEAL_PROFILE = HardwareProfile(
    name="ideal",
    t1_us=1e9, t2_us=1e9,
    anneal_time_us=1.0, gate_error_rate=0.0, readout_error_rate=0.0,
    bond_dim_max=512,
    description="Noiseless reference: infinite coherence, no gate/readout errors. "
                "Not a physical platform -- no literature source applies.")

SUPERCONDUCTING_PROFILE = HardwareProfile(
    name="superconducting_2024",
    t1_us=262.69, t2_us=176.67,
    anneal_time_us=200.0, gate_error_rate=7.57e-3, readout_error_rate=1.35e-2,
    bond_dim_max=128,
    description="LITERATURE-CALIBRATED (not a physical hardware experiment). "
                "IBM 127-qubit Eagle r3 processor (ibm_sherbrooke), reported "
                "2024-08-01: median T1=262.69us, median T2=176.67us, median "
                "two-qubit (ECR) error=7.57e-3, median readout error=1.35e-2.",
    source="IBM Quantum, \"Eagle's quantum performance progress\" (ibm_sherbrooke, Eagle r3)",
    source_url_or_doi="https://www.ibm.com/quantum/blog/eagle-quantum-processor-performance")

ION_TRAP_PROFILE = HardwareProfile(
    name="ion_trap",
    t1_us=5.0e7, t2_us=5.0e7,
    anneal_time_us=50.0, gate_error_rate=1e-6, readout_error_rate=7e-4,
    bond_dim_max=256,
    description="LITERATURE-CALIBRATED (not a physical hardware experiment). "
                "Harty et al. (Oxford, 43Ca+ hyperfine qubit): memory coherence "
                "T2*=50s, single-qubit gate error=1e-6 (99.9999% fidelity), "
                "state-preparation-and-measurement (readout) error=7e-4 "
                "(99.93% fidelity). The source characterizes a single T2* "
                "dephasing time and does not separately report T1 (population "
                "relaxation); T1 is conservatively set equal to T2 here as a "
                "physically-motivated lower bound (T1 >= T2 always holds), not "
                "an independently measured value.",
    source="Harty et al., \"High-Fidelity Preparation, Gates, Memory, and "
           "Readout of a Trapped-Ion Quantum Bit\", Phys. Rev. Lett. 113, "
           "220501 (2014)",
    source_url_or_doi="https://arxiv.org/abs/1403.1524")

NEUTRAL_ATOM_PROFILE = HardwareProfile(
    name="neutral_atom",
    t1_us=1.49e6, t2_us=1.49e6,
    anneal_time_us=100.0, gate_error_rate=5e-3, readout_error_rate=1e-2,
    bond_dim_max=256,
    description="LITERATURE-CALIBRATED (not a physical hardware experiment). "
                "Coherence T2=1.49(8)s from Bluvstein et al. (Nature 604, 451, "
                "2022, coherent atom transport); two-qubit Rydberg gate "
                "error=5e-3 (99.5% fidelity) from Evered et al. (Nature 622, "
                "268, 2023); readout error~1e-2, a typical value from a 2026 "
                "review of the platform rather than one specific measurement. "
                "T1 is not separately reported by these sources and is "
                "conservatively set equal to T2 (T1 >= T2 lower bound), not "
                "an independently measured value.",
    source="Bluvstein et al., Nature 604, 451 (2022); Evered et al., Nature "
           "622, 268 (2023)",
    source_url_or_doi="https://doi.org/10.1038/s41586-022-04592-6")

PHOTONIC_PROFILE = HardwareProfile(
    name="photonic",
    t1_us=2.0e4, t2_us=2.0e4,
    anneal_time_us=10.0, gate_error_rate=1e-5, readout_error_rate=0.15,
    bond_dim_max=512,
    description="LITERATURE-CALIBRATED (not a physical hardware experiment). "
                "Rare-earth-doped-crystal (151Eu3+:Y2SiO5) atomic-frequency-comb "
                "quantum memory: storage of a photonic time-bin qubit for 20ms "
                "with average output fidelity 85(2)% (readout_error_rate=0.15 "
                "represents this retrieval infidelity, not a logic-gate error "
                "-- this platform is a passive memory with no active "
                "entangling gate reported here, so gate_error_rate is left at "
                "an unsourced nominal placeholder).",
    source="Ortu, Holzapfel, Etesse & Afzelius, \"Storage of photonic "
           "time-bin qubits for up to 20 ms in a rare-earth doped crystal\", "
           "npj Quantum Information 8, 29 (2022)",
    source_url_or_doi="https://doi.org/10.1038/s41534-022-00541-3")


ALL_PROFILES = [IDEAL_PROFILE, SUPERCONDUCTING_PROFILE, ION_TRAP_PROFILE,
                NEUTRAL_ATOM_PROFILE, PHOTONIC_PROFILE]


def run_hardware_profile_comparison(topology_fn: Callable, n_slots: int = 15,
                                    mean_rate: float = 1.5,
                                    profiles: Optional[List[HardwareProfile]] = None,
                                    seed: int = 7,
                                    out_dir: Optional[str] = None) -> List[dict]:
    """Replay the decoherence-aware experiment under each hardware profile.

    The risk-aware router is run with the profile's T2 (via ``tau_mem``) and
    T1 (via ``t1_us``); the delivered fidelity is then recomputed under the
    same coherence times, so short-coherence platforms show a clear penalty.
    """
    import csv
    from optimization.time_dependent_optimizer import run_time_dependent_comparison

    profiles = profiles or ALL_PROFILES
    rows = []
    for prof in profiles:
        res = run_time_dependent_comparison(
            topology_fn, n_slots=n_slots, mean_rate=mean_rate,
            tau_mem=prof.t2_us, t1_us=prof.t1_us, seed=seed)
        dec = res["decoherence_aware"]
        rows.append({
            "profile": prof.name,
            "t1_us": prof.t1_us,
            "t2_us": prof.t2_us,
            "anneal_time_us": prof.anneal_time_us,
            "served_ratio": dec["served_ratio"],
            "utility": dec["utility"],
            "mean_fidelity": dec["mean_fidelity"],
            "mean_delivered_fidelity": dec["mean_delivered_fidelity"],
            "fidelity_retention": dec["mean_delivered_fidelity"] / max(dec["mean_fidelity"], 1e-12),
        })
        print(f"{prof.name:>20}: served {dec['served_ratio']:.2f}, "
              f"util {dec['utility']:.1f}, delivered fid "
              f"{dec['mean_delivered_fidelity']:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "hardware_profile_comparison.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")
    return rows


if __name__ == "__main__":
    from experiments.instances import generate_chain_topology
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "results", "experiments"))
    topo = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
    run_hardware_profile_comparison(topo, n_slots=15, mean_rate=1.0,
                                    profiles=[IDEAL_PROFILE, SUPERCONDUCTING_PROFILE,
                                              ION_TRAP_PROFILE, PHOTONIC_PROFILE],
                                    out_dir=out_dir)
