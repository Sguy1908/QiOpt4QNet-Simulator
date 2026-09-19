"""Parasitic-emission (VOA electroluminescence) study.

1. Point-to-point: secure key rate vs distance under the passive THA and the
   dual-source flaw, for the three VOA operating points of Li et al.,
   npj Quantum Inf. (2026), doi:10.1038/s41534-026-01365-1 (Figs. 6, 8).
2. Network-level: how the same device effect changes admission, utility and
   security of allocations made by the repo's solvers, with and without a
   security-aware bundle filter and with an optional band-pass filter.

Writes CSVs and figures under ``results/experiments/``.

    python experiments/run_parasitic_emission.py
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.classical_baselines import ALL_BASELINES
from experiments.instances import generate_chain_topology, generate_grid_topology
from hardware.parasitic_emission import (
    PAPER_OPERATING_POINTS, key_rate_dual_source, key_rate_ideal, key_rate_tha,
    max_secure_distance,
)
from hardware.parasitic_network import run_parasitic_network_study

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments'))
FIG = os.path.join(OUT, 'figures')

ALLOCATORS = ["utility_per_resource_greedy", "congestion_aware_greedy", "greedy_local_search"]
if "cp_sat_exact" in ALL_BASELINES:
    ALLOCATORS.append("cp_sat_exact")

# Single allocator used for the headline summary and figures.  The exact CP-SAT
# solver when OR-Tools is installed; otherwise a fixed heuristic, never a mix.
REFERENCE = "cp_sat_exact" if "cp_sat_exact" in ALLOCATORS else "congestion_aware_greedy"
REFERENCE_LABEL = ("exact CP-SAT allocator" if REFERENCE == "cp_sat_exact"
                   else f"{REFERENCE} (OR-Tools unavailable; not exact)")

TOPOLOGIES = {
    "chain8": lambda: generate_chain_topology(8, edge_capacity=4, memory_capacity=8, raw_fidelity=0.92),
    "grid3x3": lambda: generate_grid_topology(3, 3, edge_capacity=4, memory_capacity=8, raw_fidelity=0.92),
}
LENGTH_RANGE = (2.0, 30.0)   # km, per-link draw
FILTERS_DB = (0.0, 10.0, 20.0)
SEEDS = range(10)
N_REQUESTS = 10


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def key_rate_sweep():
    rows = []
    for length in [x * 1.0 for x in range(0, 341)]:
        row = {"length_km": length, "ideal": key_rate_ideal(length)}
        for op in PAPER_OPERATING_POINTS:
            row[f"tha[{op.label}]"] = key_rate_tha(length, op.mu)
            row[f"dual_naive[{op.label}]"] = key_rate_dual_source(length, op.mu, mode="naive")
            row[f"dual_aware[{op.label}]"] = key_rate_dual_source(length, op.mu, mode="aware")
        rows.append(row)
    _write(os.path.join(OUT, "parasitic_key_rate.csv"), rows)

    summary = [{"operating_point": "no parasitic light", "mu": 0.0,
                "max_distance_tha_km": max_secure_distance(key_rate_ideal),
                "rate_0km_tha": key_rate_ideal(0.0),
                "rate_0km_dual": key_rate_ideal(0.0)}]
    for op in PAPER_OPERATING_POINTS:
        summary.append({
            "operating_point": op.label, "mu": op.mu,
            "max_distance_tha_km": max_secure_distance(lambda L, m=op.mu: key_rate_tha(L, m)),
            "rate_0km_tha": key_rate_tha(0.0, op.mu),
            "rate_0km_dual": key_rate_dual_source(0.0, op.mu, mode="aware"),
        })
    _write(os.path.join(OUT, "parasitic_summary.csv"), summary)
    return rows, summary


def network_sweep():
    rows = []
    for topo_name, topo_fn in TOPOLOGIES.items():
        for op in PAPER_OPERATING_POINTS:
            for filt in FILTERS_DB:
                for seed in SEEDS:
                    for r in run_parasitic_network_study(
                            topo_fn, op.voltage, op.pulse_width_s, n_requests=N_REQUESTS,
                            seed=seed, filter_db=filt, allocators=ALLOCATORS,
                            length_range=LENGTH_RANGE):
                        r.update({"topology": topo_name, "operating_point": op.label, "seed": seed})
                        rows.append(r)
        print(f"  {topo_name} done")
    _write(os.path.join(OUT, "parasitic_network.csv"), rows)
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _summarize(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        row = dict(zip(keys, key))
        row.update({
            "mean_accepted": _mean([r["accepted"] for r in rs]),
            "mean_utility": _mean([r["total_utility"] for r in rs]),
            "mean_fidelity": _mean([r["avg_fidelity"] for r in rs]),
            "mean_insecure_admitted": _mean([r["insecure_admitted"] for r in rs]),
            "mean_insecure_edges": _mean([r["n_insecure_edges"] for r in rs]),
            "mean_delta_f": _mean([r["mean_delta_f"] for r in rs]),
            "n": len(rs),
        })
        out.append(row)
    return out


def aggregate(rows):
    """Summaries per (operating point, filter, scenario).

    The headline summary uses the single ``REFERENCE`` allocator (exact CP-SAT
    when OR-Tools is installed, else ``congestion_aware_greedy``); allocators are
    never averaged together.  The greedy heuristics reorder their picks under ~1e-6 utility perturbations, so
    their means carry a tie-break noise floor (reported separately per
    allocator).
    """
    ref_rows = [r for r in rows if r["allocator"] == REFERENCE]
    if not ref_rows:
        raise RuntimeError(f"reference allocator {REFERENCE!r} produced no rows")
    keys = ["operating_point", "filter_db", "scenario"]
    out = _summarize(ref_rows, keys)
    for row in out:
        row["reference_allocator"] = REFERENCE
    _write(os.path.join(OUT, "parasitic_network_summary.csv"), out)
    _write(os.path.join(OUT, "parasitic_network_by_allocator.csv"),
           _summarize(rows, ["allocator"] + keys))
    return out


def figures(rate_rows, agg):
    os.makedirs(FIG, exist_ok=True)
    colors = ["#1f77b4", "#222222", "#2ca02c"]
    L = [r["length_km"] for r in rate_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogy(L, [max(r["ideal"], 1e-12) for r in rate_rows], color="#d62728", lw=2,
                label="no parasitic light")
    for op, c in zip(PAPER_OPERATING_POINTS, colors):
        ax.semilogy(L, [max(r[f"tha[{op.label}]"], 1e-12) for r in rate_rows], color=c, lw=1.6,
                    label=f"THA, {op.label} ($\\mu_{{Eve}}$={op.mu:.4f})")
    ax.set_ylim(1e-8, 0.2)
    ax.set_xlabel("Distance L (km)")
    ax.set_ylabel("Secure key rate (bit/pulse)")
    ax.set_title("Passive Trojan-horse attack (cf. paper Fig. 6)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    short = [r for r in rate_rows if r["length_km"] <= 30]
    Ls = [r["length_km"] for r in short]
    ax.plot(Ls, [r["ideal"] for r in short], color="#d62728", lw=2, label="no parasitic light")
    for op, c in zip(PAPER_OPERATING_POINTS, colors):
        ax.plot(Ls, [r[f"dual_aware[{op.label}]"] for r in short], color=c, lw=1.6,
                label=f"dual-source, {op.label}")
    ax.set_xlabel("Distance L (km)")
    ax.set_ylabel("Secure key rate (bit/pulse)")
    ax.set_title("Dual-source flaw at short range (cf. paper Fig. 8)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG, "parasitic_key_rate.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # Network figure: accepted requests / insecure admissions per scenario.
    ops = [op.label for op in PAPER_OPERATING_POINTS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    lookup = {(a["operating_point"], a["filter_db"], a["scenario"]): a for a in agg}
    width = 0.27
    for ax, key, title in ((axes[0], "mean_accepted", "Admitted requests"),
                           (axes[1], "mean_utility", "Total utility"),
                           (axes[2], "mean_insecure_admitted",
                            "Admissions over insecure links")):
        for i, (scen, col) in enumerate((("clean", "#7f7f7f"), ("blind", "#d62728"),
                                         ("aware", "#1f77b4"))):
            vals = [lookup[(o, 0.0, scen)][key] for o in ops]
            ax.bar([j + (i - 1) * width for j in range(len(ops))], vals, width,
                   color=col, label=scen)
        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels(ops, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 10)
    axes[0].legend(fontsize=8, title="scenario", ncol=3, loc="upper center")
    fig.suptitle("Network-level effect of VOA parasitic emission "
                 f"(no filter; {REFERENCE_LABEL}, mean over topologies and seeds)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "parasitic_network.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # Mitigation figure: security-blind vs filter strength at the worst-case point.
    worst = PAPER_OPERATING_POINTS[-1].label
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(FILTERS_DB, [lookup[(worst, f, "blind")]["mean_insecure_admitted"]
                         for f in FILTERS_DB], "o-", color="#d62728",
            label="insecure admissions (blind)")
    ax.plot(FILTERS_DB, [lookup[(worst, f, "aware")]["mean_accepted"]
                         for f in FILTERS_DB], "s-", color="#1f77b4",
            label="admitted, security-aware")
    ax.plot(FILTERS_DB, [lookup[(worst, f, "clean")]["mean_accepted"]
                         for f in FILTERS_DB], "--", color="#7f7f7f", label="admitted, no EL")
    ax.set_xlabel("Band-pass filter attenuation of EL (dB)")
    ax.set_ylabel("Requests (mean)")
    ax.set_title(f"Filtering the EL restores capacity ({worst})\n{REFERENCE_LABEL}",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG, "parasitic_filter.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    print("Point-to-point key rates ...")
    rate_rows, summary = key_rate_sweep()
    for s in summary:
        print(f"  {s['operating_point']:>20}: mu={s['mu']:.4f}  "
              f"Lmax(THA)={s['max_distance_tha_km']:.1f} km  "
              f"R0(THA)={s['rate_0km_tha']:.4f}  R0(dual)={s['rate_0km_dual']:.4f}")
    print("Network study ...")
    rows = network_sweep()
    agg = aggregate(rows)
    figures(rate_rows, agg)
    print("\nno filter:")
    for a in agg:
        if a["filter_db"] == 0.0:
            print(f"  {a['operating_point']:>14} {a['scenario']:>5}: "
                  f"accepted={a['mean_accepted']:.2f} util={a['mean_utility']:.2f} "
                  f"insecure_admitted={a['mean_insecure_admitted']:.2f} dF={a['mean_delta_f']:.4f}")


if __name__ == "__main__":
    main()
