import csv, os, math
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiments"))
OUT = os.path.join(DATA, "figures")
os.makedirs(OUT, exist_ok=True)


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _agg(rows, group_keys, val_key, agg_fn="mean"):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys) if isinstance(group_keys, (list, tuple)) else r[group_keys]
        groups[key].append(float(r[val_key]))
    result = {}
    for k, vals in groups.items():
        if agg_fn == "mean":
            result[k] = sum(vals) / len(vals)
        elif agg_fn == "sum":
            result[k] = sum(vals)
        elif agg_fn == "std":
            result[k] = np.std(vals) if len(vals) > 1 else 0.0
    return result


def fig_contention_scaling():
    if not HAS_MPL:
        print("matplotlib not available")
        return
    rows = _load(os.path.join(DATA, "contention_sweep.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = {"Metropolis": "#1f77b4", "TensorNetwork": "#ff7f0e"}
    markers = {"Metropolis": "o", "TensorNetwork": "s"}
    styles = {"Metropolis": "-", "TensorNetwork": "--"}

    for col, cap in enumerate([4, 6, 10]):
        for row, metric in enumerate(["served_ratio", "time_s"]):
            ax = axes[row][col]
            for solver in ["Metropolis", "TensorNetwork"]:
                subset = [r for r in rows
                          if int(r["edge_capacity"]) == cap
                          and r["solver"] == solver]
                by_n = _agg(subset, ["n_requests", "n_nodes"], metric)
                for n_nodes in [4, 6, 10]:
                    pts = [(int(k[0]), v) for k, v in by_n.items() if int(k[1]) == n_nodes]
                    if not pts:
                        continue
                    pts.sort(key=lambda x: x[0])
                    xs, ys = zip(*pts)
                    label = f"{solver} N={n_nodes}" if col == 0 else None
                    ax.plot(xs, ys, label=label,
                            color=colors[solver], marker=markers[solver],
                            linestyle=styles[solver], markersize=4)

            ax.set_xlabel("Number of requests")
            if metric == "served_ratio":
                ax.set_ylabel("Served ratio")
                ax.set_ylim(0, 1.05)
                ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
            else:
                ax.set_ylabel("Wall-clock time (s)")
                ax.set_yscale("log")
            ax.set_title(f"Edge capacity = {cap}")
            ax.legend(fontsize=7) if col == 0 else None
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "contention_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_bond_dimension_scaling():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "bond_dim_sweep.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    bond_dims = sorted(set(int(r["bond_dim"]) for r in rows if int(r["bond_dim"]) > 0))

    n_requests = sorted(set(int(r["n_requests"]) for r in rows))
    for i, bd in enumerate(bond_dims):
        subset = [r for r in rows if int(r["bond_dim"]) == bd]
        by_n = _agg(subset, "n_requests", "served_ratio")
        pts = sorted((int(k), v) for k, v in by_n.items())
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker="o", color=colors[i], label=f"$\\chi={bd}$")

    meta = [r for r in rows if r["solver"] == "Metropolis-SA"]
    by_n = _agg(meta, "n_requests", "served_ratio")
    pts = sorted((int(k), v) for k, v in by_n.items())
    xs, ys = zip(*pts)
    ax1.plot(xs, ys, marker="s", color="red", linewidth=2, label="Metropolis", linestyle="--")

    ax1.set_xlabel("Number of requests")
    ax1.set_ylabel("Served ratio")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Solution quality vs bond dimension")

    for i, bd in enumerate(bond_dims):
        subset = [r for r in rows if int(r["bond_dim"]) == bd]
        by_n = _agg(subset, "n_requests", "time_s")
        pts = sorted((int(k), v) for k, v in by_n.items())
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, marker="o", color=colors[i], label=f"$\\chi={bd}$")

    meta_by_n = _agg(meta, "n_requests", "time_s")
    pts = sorted((int(k), v) for k, v in meta_by_n.items())
    xs, ys = zip(*pts)
    ax2.plot(xs, ys, marker="s", color="red", linewidth=2, label="Metropolis", linestyle="--")

    ax2.set_xlabel("Number of requests")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Runtime vs bond dimension")

    plt.tight_layout()
    path = os.path.join(OUT, "bond_dimension_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_adaptive_bond_dimension():
    """Future work: adaptive bond dimension driven by truncation error."""
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "adaptive_bond_dim_sweep.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    eps_vals = sorted(set(float(r["trunc_eps"]) for r in rows), reverse=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(ns)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, n in enumerate(ns):
        subset = sorted([r for r in rows if int(r["n_requests"]) == n],
                        key=lambda r: -float(r["trunc_eps"]))
        xs = [float(r["trunc_eps"]) for r in subset]
        ys = [float(r["max_chi_used"]) for r in subset]
        ax1.plot(xs, ys, marker="o", color=colors[i], label=f"N={n}")
    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("Truncation-error target (trunc_eps)")
    ax1.set_ylabel("Bond dimension used (max chi)")
    ax1.set_title("Adaptive bond dimension vs truncation-error target")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for i, n in enumerate(ns):
        subset = sorted([r for r in rows if int(r["n_requests"]) == n],
                        key=lambda r: -float(r["trunc_eps"]))
        xs = [float(r["trunc_eps"]) for r in subset]
        ys = [float(r["served_ratio"]) for r in subset]
        ax2.plot(xs, ys, marker="s", color=colors[i], label=f"N={n}")
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("Truncation-error target (trunc_eps)")
    ax2.set_ylabel("Served ratio")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Solution quality vs truncation-error target")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "adaptive_bond_dimension.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_mps_ordering():
    """Future work: learned (spectral) MPS coupling order vs greedy."""
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "mps_ordering_comparison.csv"))
    topologies = sorted(set(r["topology"] for r in rows))
    strategies = ["greedy", "spectral"]
    colors = {"greedy": "#4c72b0", "spectral": "#c44e52"}
    x = range(len(topologies))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, strat in enumerate(strategies):
        util = []
        for topo in topologies:
            subset = [r for r in rows if r["topology"] == topo and r["order_strategy"] == strat]
            util.append(sum(float(r["utility"]) for r in subset) / max(len(subset), 1))
        ax1.bar([xi + (i - 0.5) * width for xi in x], util, width,
               label=strat, color=colors[strat], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(topologies)
    ax1.set_ylabel("Mean utility")
    ax1.set_title("MPS coupling order: greedy vs spectral (not a trained model)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    for i, strat in enumerate(strategies):
        t = []
        for topo in topologies:
            subset = [r for r in rows if r["topology"] == topo and r["order_strategy"] == strat]
            t.append(sum(float(r["time_s"]) for r in subset) / max(len(subset), 1))
        ax2.bar([xi + (i - 0.5) * width for xi in x], t, width,
               label=strat, color=colors[strat], alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(topologies)
    ax2.set_ylabel("Mean wall-clock time (s)")
    ax2.set_title("Runtime: greedy vs spectral ordering")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "mps_ordering.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_topology_comparison():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "grid_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    topologies = sorted(set(r["topology"] for r in rows))
    x = range(len(topologies))
    width = 0.35

    for idx, metric in enumerate(["served_ratio", "time_s"]):
        ax = [ax1, ax2][idx]
        for si, solver in enumerate(["Metropolis", "TensorNetwork"]):
            vals = []
            for topo in topologies:
                subset = [r for r in rows if r["topology"] == topo and r["solver"] == solver]
                if metric == "served_ratio":
                    v = sum(float(r["served"]) for r in subset) / max(sum(int(r["n_requests"]) for r in subset), 1)
                else:
                    v = sum(float(r["time_s"]) for r in subset) / max(len(subset), 1)
                vals.append(v)
            offset = (si - 0.5) * width
            bars = ax.bar([xi + offset for xi in x], vals, width,
                          label=solver, color=["#1f77b4", "#ff7f0e"][si], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(topologies)
        if metric == "served_ratio":
            ax.set_ylabel("Served ratio")
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylabel("Mean wall-clock time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "topology_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_streaming_throughput():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "streaming_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    solvers = ["Streaming", "Batched"]
    colors = {"Streaming": "#2ca02c", "Batched": "#1f77b4"}
    markers = {"Streaming": "^", "Batched": "o"}

    for metric, ax in [("served_ratio", ax1), ("time_s", ax2)]:
        for solver in solvers:
            subset = [r for r in rows if r["solver"] == solver]
            pts = sorted((int(r["n_requests"]), float(r[metric])) for r in subset)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=markers[solver], color=colors[solver],
                    label=solver, linewidth=2, markersize=6)

        ax.set_xlabel("Number of requests")
        if metric == "served_ratio":
            ax.set_ylabel("Served ratio")
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylabel("Wall-clock time (s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "streaming_throughput.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def generate_tables():
    rows = _load(os.path.join(DATA, "contention_sweep.csv"))
    rows_bd = _load(os.path.join(DATA, "bond_dim_sweep.csv"))
    rows_grid = _load(os.path.join(DATA, "grid_comparison.csv"))
    rows_stream = _load(os.path.join(DATA, "streaming_comparison.csv"))

    lines = []

    lines.append("=== Table 1: Contention scaling (chain, cap=6, N=10) ===")
    hdr = f"{'Req':>5} {'Metro served':>14} {'TN served':>12} {'Metro time':>12} {'TN time':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    subset = [r for r in rows if int(r["edge_capacity"]) == 6 and int(r["n_nodes"]) == 10]
    by_n = defaultdict(lambda: {"Metropolis": [], "TensorNetwork": []})
    for r in subset:
        by_n[int(r["n_requests"])][r["solver"]].append(r)
    for n in sorted(by_n.keys()):
        m = by_n[n]["Metropolis"]
        t = by_n[n]["TensorNetwork"]
        if m and t:
            ms = sum(float(r["served_ratio"]) for r in m) / len(m)
            ts = sum(float(r["served_ratio"]) for r in t) / len(t)
            mt = sum(float(r["time_s"]) for r in m) / len(m)
            tt = sum(float(r["time_s"]) for r in t) / len(t)
            lines.append(f"{n:>5} {ms:>14.3f} {ts:>12.3f} {mt:>12.4f} {tt:>12.4f}")

    lines.append("")
    lines.append("=== Table 2: Bond dimension analysis (chain_8) ===")
    hdr = f"{'Req':>5} {'χ=2 served':>14} {'χ=8 served':>14} {'χ=32 served':>15} {'Metro served':>15}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    by_n_bd = defaultdict(lambda: defaultdict(list))
    for r in rows_bd:
        by_n_bd[int(r["n_requests"])][r["solver"]].append(r)
    for n in sorted(by_n_bd.keys()):
        d = by_n_bd[n]
        def _sv(s):
            v = d.get(s, [])
            return sum(float(x["served_ratio"]) for x in v) / max(len(v), 1)
        lines.append(f"{n:>5} {_sv('TN(χ=2)'):>14.3f} {_sv('TN(χ=8)'):>14.3f} "
                     f"{_sv('TN(χ=32)'):>15.3f} {_sv('Metropolis-SA'):>15.3f}")

    lines.append("")
    lines.append("=== Table 3: Streaming vs batched ===")
    hdr = f"{'Req':>5} {'Stream served':>15} {'Batch served':>14} {'Stream time':>13} {'Batch time':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    by_n_s = defaultdict(lambda: defaultdict(list))
    for r in rows_stream:
        by_n_s[int(r["n_requests"])][r["solver"]].append(r)
    for n in sorted(by_n_s.keys()):
        d = by_n_s[n]
        ss = sum(float(x["served_ratio"]) for x in d.get("Streaming", [])) / max(len(d.get("Streaming", [])), 1)
        bs = sum(float(x["served_ratio"]) for x in d.get("Batched", [])) / max(len(d.get("Batched", [])), 1)
        st = sum(float(x["time_s"]) for x in d.get("Streaming", [])) / max(len(d.get("Streaming", [])), 1)
        bt = sum(float(x["time_s"]) for x in d.get("Batched", [])) / max(len(d.get("Batched", [])), 1)
        lines.append(f"{n:>5} {ss:>15.3f} {bs:>14.3f} {st:>13.4f} {bt:>12.4f}")

    path = os.path.join(OUT, "summary_tables.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")
    print("\n".join(lines))


def fig_hamiltonian_encoding():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "hamiltonian_encoding_comparison.csv"))
    forms = ["Slack-QUBO", "Direct-QUBO", "Direct-Metropolis"]
    colors = {"Slack-QUBO": "#d62728", "Direct-QUBO": "#2ca02c", "Direct-Metropolis": "#1f77b4"}
    def _req_count(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0
    instances = sorted(set(r["instance"] for r in rows), key=_req_count)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    x = range(len(instances))
    width = 0.26
    for fi, form in enumerate(forms):
        nvars = []
        for inst in instances:
            sub = [r for r in rows if r["formulation"] == form and r["instance"] == inst
                   and r["scale"] == "1.0"]
            nvars.append(float(sub[0]["n_vars"]) if sub else 0.0)
        ax1.bar([xi + (fi - 1) * width for xi in x], nvars, width,
                label=form, color=colors[form], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(instances)
    ax1.set_xlabel("Instance (requests)")
    ax1.set_ylabel("QUBO variables")
    ax1.set_title("Encoding size (scale = 1)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    scales = sorted(set(float(r["scale"]) for r in rows))
    for fi, form in enumerate(forms):
        for inst in instances:
            sub = sorted([r for r in rows if r["formulation"] == form and r["instance"] == inst],
                         key=lambda r: float(r["scale"]))
            xs = [float(r["scale"]) for r in sub]
            ys = [float(r["served_ratio"]) for r in sub]
            ax2.plot(xs, ys, marker="o", color=colors[form], markersize=4,
                     linestyle="-", alpha=0.85 if fi == 1 else 0.45,
                     label=form if inst == instances[0] else None)
    ax2.set_xscale("log")
    ax2.set_xlabel("Demand scale")
    ax2.set_ylabel("Served ratio")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Scale robustness of each encoding")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(OUT, "hamiltonian_encoding.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_scaling_analysis():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "scaling_curves.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for n in sorted(set(int(r["n_requests"]) for r in rows if r["solver"] == "Metropolis")):
        sub = [r for r in rows if r["solver"] == "Metropolis" and int(r["n_requests"]) == n]
        pts = sorted((int(r["budget"]), float(r["utility"])) for r in sub)
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker="o", label=f"N={n}")

    for n in sorted(set(int(r["n_requests"]) for r in rows if r["solver"] == "TensorNetwork")):
        sub = [r for r in rows if r["solver"] == "TensorNetwork" and int(r["n_requests"]) == n]
        pts = sorted((int(r["bond_dim"]), float(r["utility"])) for r in sub)
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, marker="s", linestyle="--", label=f"N={n}")

    ax1.set_xscale("log")
    ax1.set_xlabel("Metropolis budget (steps)")
    ax1.set_ylabel("Aggregate utility")
    ax1.set_title("Metropolis quality vs budget")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("MPS bond dimension $\\chi$")
    ax2.set_ylabel("Aggregate utility")
    ax2.set_title("Tensor-network quality vs $\\chi$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "scaling_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_time_dependent():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "time_dependent_slots.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    slots = [int(r["slot"]) for r in rows]
    dec = [int(r["served_dec"]) for r in rows]
    st = [int(r["served_static"]) for r in rows]

    width = 0.4
    ax1.bar([s - width / 2 for s in slots], dec, width, label="Decoherence-aware",
            color="#1f77b4", alpha=0.85)
    ax1.bar([s + width / 2 for s in slots], st, width, label="Static",
            color="#ff7f0e", alpha=0.85)
    ax1.set_xlabel("Time slot")
    ax1.set_ylabel("Requests served")
    ax1.set_title("Per-slot routing decisions")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    cdec = []
    cst = []
    acc = 0
    for v in dec:
        acc += v
        cdec.append(acc)
    acc = 0
    for v in st:
        acc += v
        cst.append(acc)
    ax2.plot(slots, cdec, marker="o", color="#1f77b4", label="Decoherence-aware")
    ax2.plot(slots, cst, marker="s", color="#ff7f0e", label="Static")
    ax2.set_xlabel("Time slot")
    ax2.set_ylabel("Cumulative requests served")
    ax2.set_title("Cumulative throughput")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "time_dependent.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_risk_tradeoff():
    if not HAS_MPL:
        return
    path_csv = os.path.join(DATA, "risk_gain_tradeoff.csv")
    if not os.path.exists(path_csv):
        print("risk_gain_tradeoff.csv missing; skipping")
        return
    rows = _load(path_csv)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key, label, color in [("dec", "Decoherence-aware", "#1f77b4"),
                              ("static", "Static", "#ff7f0e")]:
        xs = [float(r[f"fid_{key}"]) for r in rows]
        ys = [float(r[f"utility_{key}"]) for r in rows]
        ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel("Mean delivered fidelity")
    ax.set_ylabel("Aggregate utility")
    ax.set_title("Fidelity-utility tradeoff under risk gain sweep")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "risk_tradeoff.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_constrained_mps():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "constrained_mps_comparison.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    solvers = ["Penalty-MPS", "Constrained-MPS"]
    colors = {"Penalty-MPS": "#1f77b4", "Constrained-MPS": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = range(len(ns))
    width = 0.36
    for idx, (metric, ylab, title) in enumerate([
        ("served_ratio", "Served ratio", "Solution quality"),
        ("time_s", "Wall-clock time (s)", "Runtime"),
        ("utility", "Aggregate utility", "Aggregate utility"),
    ]):
        ax = axes[idx]
        for si, solver in enumerate(solvers):
            vals = []
            for n in ns:
                sub = [r for r in rows if r["solver"] == solver and int(r["n_requests"]) == n]
                vals.append(sum(float(r[metric]) for r in sub) / max(len(sub), 1))
            ax.bar([xi + (si - 0.5) * width for xi in x], vals, width,
                   label=solver, color=colors[solver], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"N={n}" for n in ns])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "constrained_mps.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_qlearning():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "qlearning_comparison.csv"))
    eps = [int(r["episode"]) for r in rows]
    served = [int(r["served"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(eps, served, marker="o", color="#1f77b4", label="Served requests")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Served requests")
    ax1.set_title("Q-learning router training progress")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eps, reward, marker="s", color="#d62728", alpha=0.7, label="Episode reward")
    ax2.set_ylabel("Episode reward", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    plt.tight_layout()
    path = os.path.join(OUT, "qlearning.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_rl_generalization():
    """Future work: function-approximation RL that generalizes across
    topologies. Zero-shot linear-router served ratio vs an in-distribution
    tabular control and the streaming-annealer baseline, per eval topology
    the linear router never trained on."""
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "rl_topology_generalization.csv"))
    topos = [r["eval_topology"] for r in rows]
    x = range(len(topos))
    width = 0.25
    series = [
        ("Linear (zero-shot)", "linear_zero_shot_served_ratio", "#c44e52"),
        ("Tabular (in-distribution)", "tabular_in_distribution_served_ratio", "#4c72b0"),
        ("Streaming annealer", "streaming_annealer_served_ratio", "#55a868"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (label, key, color) in enumerate(series):
        vals = [float(r[key]) for r in rows]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=label,
              color=color, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(topos)
    ax.set_ylabel("Served ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("RL generalization: zero-shot transfer to unseen topologies")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "rl_topology_generalization.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_hardware_profiles():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "hardware_profile_comparison.csv"))
    profiles = [r["profile"] for r in rows]
    fid = [float(r["mean_delivered_fidelity"]) for r in rows]
    ret = [float(r["fidelity_retention"]) for r in rows]
    raw = [float(r["mean_fidelity"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = range(len(profiles))
    width = 0.38
    ax1.bar([xi - width / 2 for xi in x], raw, width, label="Generated fidelity",
            color="#1f77b4", alpha=0.7)
    ax1.bar([xi + width / 2 for xi in x], fid, width, label="Delivered fidelity",
            color="#2ca02c", alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(profiles, rotation=12, fontsize=8)
    ax1.set_ylabel("Mean fidelity")
    ax1.set_title("Fidelity loss across hardware profiles")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, [r * 100 for r in ret], color="#9467bd", alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(profiles, rotation=12, fontsize=8)
    ax2.set_ylabel("Fidelity retention (%)")
    ax2.set_title("Retention = delivered / generated")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "hardware_profiles.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_selfish_routing():
    if not HAS_MPL:
        return
    base = os.path.join(DATA, "selfish_routing")
    rows_poa = _load(os.path.join(DATA, "selfish_routing_pigou_poa.csv"))
    rows_toll = _load(os.path.join(DATA, "selfish_routing_pigou_toll.csv"))
    rows_cap = _load(os.path.join(DATA, "selfish_routing_braess_capacity.csv"))
    rows_flat = _load(os.path.join(DATA, "selfish_routing_braess_toll.csv"))
    rows_wtoll = _load(os.path.join(DATA, "selfish_routing_braess_wardrop_toll.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))

    ax = axes[0][0]
    xs = [int(r["n_players"]) for r in rows_poa]
    ys = [float(r["poa_worst"]) for r in rows_poa]
    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2)
    ax.axhline(4 / 3, color="gray", linestyle=":", label="4/3 (Pigou bound)")
    ax.set_xlabel("Number of players")
    ax.set_ylabel("Price of anarchy (worst Nash)")
    ax.set_title("Atomic Pigou: PoA vs players")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    xs = [float(r["lambda"]) for r in rows_toll]
    ys = [float(r["poa_worst"]) for r in rows_toll]
    ax.plot(xs, ys, marker="o", color="#2ca02c", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_xlabel("Marginal-cost toll coefficient $\\lambda$")
    ax.set_ylabel("Price of anarchy")
    ax.set_title("Pigou (n=3): tolls recover optimum")
    ax.grid(True, alpha=0.3)

    ax = axes[0][2]
    xs = [float(r["cross_base_latency"]) for r in rows_cap]
    ys = [float(r["eq_social_cost"]) for r in rows_cap]
    yt = [float(r["tolled_social_cost"]) for r in rows_cap]
    ax.plot(xs, ys, marker="o", color="#d62728", linewidth=2, label="Selfish")
    ax.plot(xs, yt, marker="s", color="#2ca02c", linewidth=2, label="Marginal toll")
    ax.set_xlabel("Shortcut latency")
    ax.set_ylabel("Equilibrium latency")
    ax.set_title("Braess: adding capacity hurts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    xs = [float(r["cross_base_latency"]) for r in rows_cap]
    ys = [float(r["poa_worst"]) for r in rows_cap]
    ax.plot(xs, ys, marker="o", color="#d62728", linewidth=2)
    ax.axhline(4 / 3, color="gray", linestyle=":", label="4/3")
    ax.set_xlabel("Shortcut latency")
    ax.set_ylabel("Price of anarchy")
    ax.set_title("Braess: PoA vs shortcut capacity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    xs = [float(r["flat_toll"]) for r in rows_flat]
    ys = [float(r["poa_worst"]) for r in rows_flat]
    ax.plot(xs, ys, marker="o", color="#9467bd", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_xlabel("Flat toll on shortcut")
    ax.set_ylabel("Price of anarchy")
    ax.set_title("Braess: flat toll removes paradox")
    ax.grid(True, alpha=0.3)

    ax = axes[1][2]
    xs = [float(r["lambda"]) for r in rows_wtoll]
    ys = [float(r["eq_social_cost"]) for r in rows_wtoll]
    ax.plot(xs, ys, marker="o", color="#2ca02c", linewidth=2)
    ax.axhline(1.5, color="gray", linestyle=":", label="Social optimum (1.5)")
    ax.set_xlabel("Marginal-cost toll coefficient $\\lambda$")
    ax.set_ylabel("Equilibrium latency")
    ax.set_title("Braess: marginal toll restores optimum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "selfish_routing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_robust_routing():
    if not HAS_MPL:
        return
    rows_crit = _load(os.path.join(DATA, "robust_routing_criteria.csv"))
    rows_noise = _load(os.path.join(DATA, "robust_routing_uncertainty_sweep.csv"))
    rows_pareto = _load(os.path.join(DATA, "robust_routing_pareto.csv"))
    rows_inst = _load(os.path.join(DATA, "robust_routing_instances.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

    ax = axes[0][0]
    labels = [r["criterion"] for r in rows_crit]
    x = range(len(labels))
    width = 0.38
    nom = [float(r["nominal_util"]) for r in rows_crit]
    worst = [float(r["worst_util"]) for r in rows_crit]
    ax.bar([xi - width / 2 for xi in x], nom, width, label="Nominal utility",
           color="#1f77b4", alpha=0.85)
    ax.bar([xi + width / 2 for xi in x], worst, width, label="Worst-case utility",
           color="#d62728", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Aggregate utility")
    ax.set_title("Decision criteria on a fragile bottleneck")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0][1]
    xs = [float(r["failure_prob"]) for r in rows_noise]
    gain = [float(r["worst_util_gain"]) for r in rows_noise]
    loss = [float(r["nominal_util_loss"]) for r in rows_noise]
    ax.plot(xs, gain, marker="o", color="#2ca02c", label="Worst-case gain")
    ax.plot(xs, loss, marker="s", color="#ff7f0e", label="Nominal loss (price of robustness)")
    ax.set_xlabel("Bottleneck failure probability")
    ax.set_ylabel("Utility (robust vs nominal)")
    ax.set_title("Robustness gain grows with uncertainty")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    xs = [float(r["gamma"]) for r in rows_pareto]
    nom = [float(r["nominal_util"]) for r in rows_pareto]
    worst = [float(r["worst_util"]) for r in rows_pareto]
    ax.plot(xs, nom, marker="o", color="#1f77b4", label="Nominal utility")
    ax.plot(xs, worst, marker="s", color="#d62728", label="Worst-case utility")
    ax.set_xlabel("Robustness budget $\\gamma$")
    ax.set_ylabel("Utility")
    ax.set_title("Gamma-robust trade-off frontier")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    xs = [int(r["n_requests"]) for r in rows_inst]
    gain = [float(r["worst_util_gain"]) for r in rows_inst]
    loss = [float(r["nominal_util_loss"]) for r in rows_inst]
    ax.plot(xs, gain, marker="o", color="#2ca02c", label="Worst-case gain")
    ax.plot(xs, loss, marker="s", color="#ff7f0e", label="Nominal loss")
    ax.set_xlabel("Number of requests")
    ax.set_ylabel("Utility (robust vs nominal)")
    ax.set_title("Chain instances under random noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "robust_routing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_joint_scheduling():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "joint_scheduling_comparison.csv"))
    labels = sorted(set((r["mean_rate"], r["tau_mem"]) for r in rows))
    by_regime = defaultdict(dict)
    for r in rows:
        by_regime[r["regime"]][(r["mean_rate"], r["tau_mem"])] = float(r["served_ratio"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = range(len(labels))
    width = 0.36
    colors = {"joint": "#1f77b4", "memory_agnostic": "#ff7f0e"}
    for i, regime in enumerate(["joint", "memory_agnostic"]):
        vals = [by_regime[regime].get(k, 0.0) for k in labels]
        ax1.bar([xi + (i - 0.5) * width for xi in x], vals, width,
                label=regime, color=colors[regime], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"rate={mr}, $\\tau$={tau}" for mr, tau in labels], fontsize=8)
    ax1.set_ylabel("Served ratio")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Memory-aware vs agnostic scheduling")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    by_tau = defaultdict(list)
    for r in rows:
        if r["regime"] == "joint" and r["mean_delivered_fidelity"] \
                and float(r["mean_delivered_fidelity"]) > 0:
            by_tau[r["tau_mem"]].append(float(r["mean_delivered_fidelity"]))
    taus = sorted(by_tau.keys())
    ax2.bar(range(len(taus)), [sum(by_tau[t]) / len(by_tau[t]) for t in taus],
            color="#2ca02c", alpha=0.85)
    ax2.set_xticks(list(range(len(taus))))
    ax2.set_xticklabels([f"$\\tau$={t}" for t in taus])
    ax2.set_ylabel("Mean delivered fidelity")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Fidelity of served requests (joint)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "joint_scheduling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_online_regimes():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "online_regimes_comparison.csv"))
    rates = sorted(set(float(r["mean_rate"]) for r in rows))
    regimes = ["static", "online", "receding_horizon"]
    colors = {"static": "#1f77b4", "online": "#2ca02c", "receding_horizon": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(rates))
    width = 0.26
    for i, regime in enumerate(regimes):
        vals = []
        for rate in rates:
            sub = [r for r in rows if r["regime"] == regime and float(r["mean_rate"]) == rate]
            vals.append(sum(float(r["served_ratio"]) for r in sub) / max(len(sub), 1))
        ax.bar([xi + (i - 1) * width for xi in x], vals, width,
               label=regime, color=colors[regime], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"rate={r}" for r in rates])
    ax.set_ylabel("Served ratio")
    ax.set_ylim(0, 1.1)
    ax.set_title("Static vs online vs receding-horizon scheduling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "online_regimes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_adaptive_qubo():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "adaptive_qubo_topk.csv"))
    topos = sorted(set(r["topology"] for r in rows))
    markers = {"chain_10": "o", "grid_6x6": "s"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for topo in topos:
        sub = [r for r in rows if r["topology"] == topo]
        pts = sorted((int(r["k"]), float(r["relative_gap"])) for r in sub)
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker=markers[topo], label=topo, linewidth=2, markersize=5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Candidate budget $k$")
    ax1.set_ylabel("Relative optimality gap")
    ax1.set_title("Quality of adaptive candidate reduction")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for topo in topos:
        sub = sorted([r for r in rows if r["topology"] == topo],
                     key=lambda r: int(r["k"]))
        xs = [int(r["k"]) for r in sub]
        ys = [int(r["n_bundles_in_qubo"]) for r in sub]
        full = max(int(r["n_bundles_in"]) for r in rows if r["topology"] == topo)
        ax2.plot(xs, ys, marker=markers[topo], label=topo, linewidth=2, markersize=5)
        ax2.axhline(full, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Candidate budget $k$")
    ax2.set_ylabel("QUBO bundles ($n_{b}$)")
    ax2.set_title("QUBO size under top-k reduction")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "adaptive_qubo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_hybrid_pipeline():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "hybrid_pipeline_comparison.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    stages = ["full_pipeline", "qubo_only", "qubo_plus_repair"]
    colors = {"full_pipeline": "#1f77b4", "qubo_only": "#ff7f0e", "qubo_plus_repair": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(ns))
    width = 0.26
    for i, stage in enumerate(stages):
        vals = []
        for n in ns:
            sub = [r for r in rows if r["stage"] == stage and int(r["n_requests"]) == n]
            vals.append(sum(float(r["utility"]) for r in sub) / max(len(sub), 1))
        ax.bar([xi + (i - 1) * width for xi in x], vals, width,
               label=stage, color=colors[stage], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Aggregate utility")
    ax.set_title("Hybrid candidate reduction + QUBO pipeline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "hybrid_pipeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_gnn_reduction():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "gnn_candidate_reduction.csv"))
    topos = sorted(set(r["topology"] for r in rows))
    methods = ["full_qubo", "adaptive_topk", "gnn_guided"]
    colors = {"full_qubo": "#1f77b4", "adaptive_topk": "#ff7f0e", "gnn_guided": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(topos))
    width = 0.26
    for i, method in enumerate(methods):
        vals = []
        for topo in topos:
            sub = [r for r in rows if r["method"] == method and r["topology"] == topo]
            vals.append(sum(float(r["utility"]) for r in sub) / max(len(sub), 1))
        ax.bar([xi + (i - 1) * width for xi in x], vals, width,
               label=method, color=colors[method], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(topos)
    ax.set_ylabel("Aggregate utility")
    ax.set_title("GNN-guided candidate reduction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "gnn_reduction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_multi_objective():
    if not HAS_MPL:
        return
    front = _load(os.path.join(DATA, "multi_objective_frontier.csv"))
    cons = _load(os.path.join(DATA, "multi_objective_constraint.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    sc = ax1.scatter([float(r["fidelity"]) for r in front],
                     [float(r["throughput"]) for r in front],
                     c=[float(r["latency"]) for r in front], s=8, alpha=0.6,
                     cmap="viridis")
    ax1.set_xlabel("Mean delivered fidelity")
    ax1.set_ylabel("Throughput (served)")
    ax1.set_title("Pareto frontier (enumerated selections)")
    fig.colorbar(sc, ax=ax1, label="Latency")

    for key, color in [("fidelity", "#1f77b4"), ("throughput", "#d62728")]:
        sub = [r for r in cons if r["constrain"] == key and r["feasible"] == "True"]
        pts = sorted((float(r["target"]), float(r["achieved_maximized"])) for r in sub)
        if not pts:
            continue
        xs, ys = zip(*pts)
        max_key = sub[0]["maximize"]
        ax2.plot(xs, ys, marker="o", color=color, linewidth=2,
                 label=f"maximize {max_key} s.t. {key} ≥ t")
    ax2.set_xlabel("Constraint target $t$")
    ax2.set_ylabel("Achieved maximized objective")
    ax2.set_title("$\\epsilon$-constraint frontiers")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "multi_objective.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_disjoint_paths():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "disjoint_paths_comparison.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    sets_ = ["single_path", "multipath"]
    colors = {"single_path": "#1f77b4", "multipath": "#2ca02c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = range(len(ns))
    width = 0.36
    for i, cs in enumerate(sets_):
        sr = []
        for n in ns:
            sub = [r for r in rows if r["candidate_set"] == cs and int(r["n_requests"]) == n]
            sr.append(sum(float(r["served_ratio"]) for r in sub) / max(len(sub), 1))
        ax1.bar([xi + (i - 0.5) * width for xi in x], sr, width,
                label=cs, color=colors[cs], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Served ratio")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Coverage: single vs multipath candidates")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    for i, cs in enumerate(sets_):
        ut = []
        for n in ns:
            sub = [r for r in rows if r["candidate_set"] == cs and int(r["n_requests"]) == n]
            ut.append(sum(float(r["utility"]) for r in sub) / max(len(sub), 1))
        ax2.bar([xi + (i - 0.5) * width for xi in x], ut, width,
                label=cs, color=colors[cs], alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"N={n}" for n in ns])
    ax2.set_ylabel("Aggregate utility")
    ax2.set_title("Utility with k-disjoint redundancy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "disjoint_paths.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_swapping_order():
    if not HAS_MPL:
        return
    sweep = _load(os.path.join(DATA, "swapping_order_sweep.csv"))
    pfid = _load(os.path.join(DATA, "swapping_path_fidelity.csv"))
    strategies = ["linear", "balanced", "optimal"]
    colors = {"linear": "#d62728", "balanced": "#ff7f0e", "optimal": "#2ca02c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    by = defaultdict(list)
    for r in sweep:
        by[(int(r["path_length"]), r["strategy"])].append(float(r["delivered_fidelity"]))
    for strat in strategies:
        pts = sorted((k[0], sum(v) / len(v)) for k, v in by.items() if k[1] == strat)
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker="o", color=colors[strat], label=strat, linewidth=2, markersize=5)
    ax1.set_xlabel("Path length (links)")
    ax1.set_ylabel("Mean delivered fidelity")
    ax1.set_title("Swapping order vs decay loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    by_tau = defaultdict(list)
    for r in pfid:
        by_tau[r["strategy"]].append((float(r["tau_mem"]), float(r["delivered_fidelity"])))
    for strat in strategies:
        pts = sorted(by_tau[strat])
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, marker="s", color=colors[strat], label=strat, linewidth=2, markersize=5)
    ax2.set_xlabel("Memory coherence time $\\tau_{\\mathrm{mem}}$")
    ax2.set_ylabel("Delivered fidelity")
    ax2.set_title("T1/T2 decay vs swapping order")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "swapping_order.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_topology_evolution():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "topology_evolution.csv"))
    topo_fams = sorted(set(r["topology"] for r in rows))
    colors = plt.cm.tab10(np.linspace(0, 1, len(topo_fams)))
    fam_color = {t: colors[i] for i, t in enumerate(topo_fams)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    x = range(len(ns))
    width = 0.8 / max(len(topo_fams), 1)
    for i, fam in enumerate(topo_fams):
        vals = []
        for n in ns:
            sub = [r for r in rows if r["topology"] == fam and int(r["n_requests"]) == n]
            vals.append(sum(float(r["served_ratio"]) for r in sub) / max(len(sub), 1))
        ax1.bar([xi + (i - len(topo_fams) / 2) * width for xi in x], vals, width,
                label=fam, color=fam_color[fam], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Served ratio")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Solver quality across topology families")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3, axis="y")

    for fam in topo_fams:
        sub = [r for r in rows if r["topology"] == fam]
        ax2.scatter([float(r["density"]) for r in sub],
                    [float(r["served_ratio"]) for r in sub],
                    color=fam_color[fam], label=fam, s=40)
    ax2.set_xlabel("Edge density")
    ax2.set_ylabel("Served ratio")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Served ratio vs topology density")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "topology_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_des_reliability():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "des_reliability.csv"))
    tau_mems = sorted(set(r["tau_mem"] for r in rows), key=float)
    colors = plt.cm.tab10(np.linspace(0, 1, len(tau_mems)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, tm in enumerate(tau_mems):
        sub = [r for r in rows if r["tau_mem"] == tm]
        served = {float(k): v for k, v in _agg(sub, "swap_success", "served_ratio").items()}
        gap = {float(k): v for k, v in _agg(sub, "swap_success", "utility_gap").items()}
        xs = sorted(served)
        ax1.plot(xs, [served[x] for x in xs], "o-", color=colors[i], label=f"tau_mem={tm}")
        ax2.plot(xs, [gap[x] for x in xs], "s--", color=colors[i], label=f"tau_mem={tm}")
    ax1.set_xlabel("Swap success probability")
    ax1.set_ylabel("Sampled served ratio")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("DES: delivered request ratio")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.set_xlabel("Swap success probability")
    ax2.set_ylabel("E[U] parametric $-$ sampled")
    ax2.set_title("DES: parametric vs sampled utility gap")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "des_reliability.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_purification():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "purification_comparison.csv"))
    sweep = _load(os.path.join(DATA, "purification_sweep.csv"))
    regimes = ["purification_agnostic", "default_q012", "purification_aware"]
    regimes = [rg for rg in regimes if rg in {r["regime"] for r in rows}]
    colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    x = range(len(ns))
    width = 0.8 / max(len(regimes), 1)
    for i, rg in enumerate(regimes):
        vals = []
        for n in ns:
            sub = [r for r in rows if r["regime"] == rg and int(r["n_requests"]) == n]
            vals.append(sum(float(r["served_ratio"]) for r in sub) / max(len(sub), 1))
        ax1.bar([xi + (i - len(regimes) / 2) * width for xi in x], vals, width,
                label=rg, color=colors[i], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Served ratio")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Purification regime: served ratio")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    plens = sorted(set(int(r["path_length"]) for r in sweep))
    for pl in plens:
        sub = [r for r in sweep if int(r["path_length"]) == pl]
        sub.sort(key=lambda r: float(r["fidelity"]))
        ax2.plot([float(r["fidelity"]) for r in sub],
                 [float(r["bell_pair_cost"]) for r in sub],
                 "o-", label=f"path={pl}")
    ax2.set_xlabel("Delivered fidelity")
    ax2.set_ylabel("Bell-pair cost (purification rounds)")
    ax2.set_title("Fidelity vs purification cost")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "purification.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_recourse():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "recourse_comparison.csv"))
    summary = _load(os.path.join(DATA, "recourse_summary.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in sorted(set(int(r["n_requests"]) for r in rows)):
        sub = [r for r in rows if int(r["n_requests"]) == n]
        ax1.scatter([float(r["n_failed"]) for r in sub],
                    [float(r["n_local_recovered"]) for r in sub],
                    s=36, alpha=0.8, label=f"N={n}")
    lim = max(float(r["n_failed"]) for r in rows) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", alpha=0.4)
    ax1.set_xlabel("Requests failed per realization")
    ax1.set_ylabel("Requests recovered by local repair")
    ax1.set_title("Adaptive recourse: local repair recovery")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ns = [int(r["n_requests"]) for r in summary]
    speedup = [float(r["speedup"]) for r in summary]
    recovery = [float(r["mean_recovery_rate"]) for r in summary]
    x = range(len(ns))
    ax2.bar([xi - 0.18 for xi in x], speedup, 0.36, label="speedup (t_full/t_local)", alpha=0.85)
    ax2.bar([xi + 0.18 for xi in x], recovery, 0.36, label="mean recovery rate", alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"N={n}" for n in ns])
    ax2.set_ylabel("Ratio")
    ax2.set_title("Recourse: local-repair speedup and recovery")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "recourse.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_optimality_gap():
    if not HAS_MPL:
        return
    gap_rows = _load(os.path.join(DATA, "optimality_gap.csv"))
    rel_rows = _load(os.path.join(DATA, "stochastic_reliability.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    solvers = sorted(set(r["solver"] for r in gap_rows))
    ns = sorted(set(int(r["n_requests"]) for r in gap_rows))
    x = range(len(ns))
    width = 0.8 / max(len(solvers), 1)
    for i, s in enumerate(solvers):
        vals = []
        for n in ns:
            sub = [r for r in gap_rows if r["solver"] == s and int(r["n_requests"]) == n]
            vals.append(sum(float(r["gap_rel"]) for r in sub) / max(len(sub), 1))
        ax1.bar([xi + (i - len(solvers) / 2) * width for xi in x], vals, width,
                label=s, alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Mean relative gap vs exact ILP")
    ax1.set_title("Optimality-gap certification")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    rel_solvers = sorted(set(r["solver"] for r in rel_rows))
    rel_ns = sorted(set(int(r["n_requests"]) for r in rel_rows))
    x2 = range(len(rel_ns))
    for i, s in enumerate(rel_solvers):
        vals = []
        for n in rel_ns:
            sub = [r for r in rel_rows if r["solver"] == s and int(r["n_requests"]) == n]
            vals.append(sum(float(r["reliability_gap"]) for r in sub) / max(len(sub), 1))
        ax2.bar([xi + (i - len(rel_solvers) / 2) * width for xi in x2], vals, width,
                label=s, alpha=0.85)
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels([f"N={n}" for n in rel_ns])
    ax2.set_ylabel("Reliability gap (E[U] param $-$ sampled)")
    ax2.set_title("Stochastic reliability of solver plans")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "optimality_gap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_adaptive_budget():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "adaptive_budget.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    kvals = sorted(set(int(r["k"]) for r in rows if r["method"].startswith("k")))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ns)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, n in enumerate(ns):
        sub = [r for r in rows if int(r["n_requests"]) == n and r["method"].startswith("k")]
        sub.sort(key=lambda r: int(r["k"]))
        ax1.plot([int(r["k"]) for r in sub], [float(r["utility"]) for r in sub],
                 "o-", color=colors[i], label=f"N={n}")
        ad = [r for r in rows if int(r["n_requests"]) == n and r["method"] == "adaptive"]
        if ad:
            ax1.axhline(float(ad[0]["utility"]), color=colors[i], ls=":", alpha=0.6)
        ax2.plot([int(r["k"]) for r in sub],
                 [100 * float(r["relative_gap_vs_full"]) for r in sub],
                 "s--", color=colors[i], label=f"N={n}")
    ax1.set_xlabel("Top-k candidate budget")
    ax1.set_ylabel("Utility (dashed: adaptive)")
    ax1.set_title("Adaptive budget: utility vs fixed k")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Top-k candidate budget")
    ax2.set_ylabel("Relative utility gap vs full QUBO (%)")
    ax2.set_title("Adaptive budget: gap to full candidate set")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "adaptive_budget.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_quantum_annealing():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "quantum_annealing_sweep.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    x = range(len(ns))
    width = 0.25
    backends = [("QA (embedded PIA)", "qa_utility", "#c44e52"),
                ("SA (openjij)", "sa_utility", "#4c72b0"),
                ("SQA (openjij)", "sqa_utility", "#55a868")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, (label, key, color) in enumerate(backends):
        vals = [float([r for r in rows if int(r["n_requests"]) == n][0][key]) for n in ns]
        ax1.bar([xi + (i - 1) * width for xi in x], vals, width,
                label=label, color=color, alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Utility")
    ax1.set_title("Embedded QA vs SA / SQA")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    qubits = [float([r for r in rows if int(r["n_requests"]) == n][0]["qa_n_qubits"]) for n in ns]
    ax2.bar(x, qubits, 0.5, alpha=0.85)
    for xi, q in zip(x, qubits):
        ax2.text(xi, q, f"{int(q)}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"N={n}" for n in ns])
    ax2.set_ylabel("Hardware qubits (minor-embedded)")
    ax2.set_title("QA embedding resource use")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "quantum_annealing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_quantum_annealing_topologies():
    """Chimera vs Pegasus vs Zephyr hardware-lattice comparison."""
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "quantum_annealing_topology_sweep.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    topologies = ["chimera", "pegasus", "zephyr"]
    colors = {"chimera": "#4c72b0", "pegasus": "#c44e52", "zephyr": "#55a868"}
    x = range(len(ns))
    width = 0.25

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, topo in enumerate(topologies):
        util = [float([r for r in rows if int(r["n_requests"]) == n
                       and r["topology"] == topo][0]["utility"]) for n in ns]
        ax1.bar([xi + (i - 1) * width for xi in x], util, width,
                label=topo, color=colors[topo], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Utility")
    ax1.set_title("Embedded QA utility by hardware lattice")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    for i, topo in enumerate(topologies):
        qubits = [float([r for r in rows if int(r["n_requests"]) == n
                         and r["topology"] == topo][0]["n_qubits"]) for n in ns]
        ax2.bar([xi + (i - 1) * width for xi in x], qubits, width,
                label=topo, color=colors[topo], alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"N={n}" for n in ns])
    ax2.set_ylabel("Hardware qubits (minor-embedded)")
    ax2.set_title("Embedding resource use by lattice")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    for i, topo in enumerate(topologies):
        cb = [float([r for r in rows if int(r["n_requests"]) == n
                     and r["topology"] == topo][0]["chain_break_fraction"]) for n in ns]
        ax3.bar([xi + (i - 1) * width for xi in x], cb, width,
                label=topo, color=colors[topo], alpha=0.85)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels([f"N={n}" for n in ns])
    ax3.set_ylabel("Chain-break fraction")
    ax3.set_title("Chain breaks by lattice")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "quantum_annealing_topologies.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_hardware_literature_validation():
    """Future work: literature-calibrated hardware validation.

    NOT a physical hardware experiment -- delivered fidelity under
    HardwareProfile constants re-calibrated from cited, published measured
    parameters (see hardware/profiles.py)."""
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "hardware_literature_validation.csv"))
    rows = sorted(rows, key=lambda r: float(r["t2_us"]))
    names = [r["profile"] for r in rows]
    fid = [float(r["mean_delivered_fidelity"]) for r in rows]
    labels = [f"{r['profile']}\n({r['source'][:28]}...)" if r["source"] else r["profile"]
             for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#8c8c8c" if r["profile"] == "ideal" else "#4c72b0" for r in rows]
    ax.bar(range(len(rows)), fid, color=colors, alpha=0.85)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Mean delivered fidelity")
    ax.set_title("Hardware profiles, literature-calibrated (NOT a physical "
                 "hardware experiment) -- ordered by cited T2")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT, "hardware_literature_validation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_chance_constrained():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "chance_constrained.csv"))
    groups = sorted({(int(r["n_requests"]), float(r["sigma"])) for r in rows})
    eps_all = sorted({float(r["eps"]) for r in rows if r["policy"] == "chance"})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, (n, sig) in enumerate(groups):
        ch = [r for r in rows if r["policy"] == "chance"
              and int(r["n_requests"]) == n and float(r["sigma"]) == sig]
        hard = [r for r in rows if r["policy"] == "hard"
                and int(r["n_requests"]) == n and float(r["sigma"]) == sig]
        nom = [r for r in rows if r["policy"] == "nominal"
               and int(r["n_requests"]) == n and float(r["sigma"]) == sig]
        eps = [float(r["eps"]) for r in ch]
        sla = [sum(float(r["sla_violation_prob"]) for r in ch)
               / max(len(ch), 1)] * len(eps)
        eU = [sum(float(r["e_utility"]) for r in ch) / max(len(ch), 1)] * len(eps)
        label = f"N={n}, $\\sigma$={sig}"
        ax1.plot(eps, sla, "o-", color=colors[i], label=label, markersize=5)
        ax2.plot(eps, eU, "o-", color=colors[i], label=label, markersize=5)
        if hard:
            h_sla = sum(float(r["sla_violation_prob"]) for r in hard) / len(hard)
            h_u = sum(float(r["e_utility"]) for r in hard) / len(hard)
            ax1.axhline(h_sla, color=colors[i], ls=":", lw=1, alpha=0.7)
            ax2.axhline(h_u, color=colors[i], ls=":", lw=1, alpha=0.7)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, label="y = eps (calibrated)")
    ax1.set_xlabel("Requested violation bound $\\epsilon$")
    ax1.set_ylabel("Empirical SLA violation probability")
    ax1.set_title("Chance constraint calibration (dotted: hard)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Requested violation bound $\\epsilon$")
    ax2.set_ylabel("Expected utility E[U] (dotted: hard)")
    ax2.set_title("Utility price of the reliability guarantee")
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "chance_constrained.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_contention_scaling()
    fig_bond_dimension_scaling()
    fig_topology_comparison()
    fig_streaming_throughput()
    fig_hamiltonian_encoding()
    fig_scaling_analysis()
    fig_time_dependent()
    fig_risk_tradeoff()
    fig_constrained_mps()
    fig_qlearning()
    fig_hardware_profiles()
    fig_selfish_routing()
    fig_robust_routing()
    fig_joint_scheduling()
    fig_online_regimes()
    fig_adaptive_qubo()
    fig_hybrid_pipeline()
    fig_gnn_reduction()
    fig_multi_objective()
    fig_disjoint_paths()
    fig_swapping_order()
    fig_topology_evolution()
    fig_des_reliability()
    fig_purification()
    fig_recourse()
    fig_optimality_gap()
    fig_adaptive_budget()
    fig_quantum_annealing()
    fig_chance_constrained()
    fig_quantum_annealing_topologies()
    fig_hardware_literature_validation()
    fig_adaptive_bond_dimension()
    fig_mps_ordering()
    fig_rl_generalization()
    print("\nGenerating tables...")
    generate_tables()
    print(f"\nAll outputs in {OUT}/")
