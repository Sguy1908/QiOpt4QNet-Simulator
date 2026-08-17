# QiOpt4QNet-Simulator

A quantum-network simulation and optimization framework. It models quantum repeater networks (nodes with limited quantum memory, lossy/noisy links, entanglement generation and swapping), builds candidate route + purification "bundles" for service requests, and compares multiple ways of deciding which bundles to admit under capacity constraints, from exact MILP solving and quantum/QUBO annealing to classical heuristics and a learned ranking model.

## What's in here

| Layer | Purpose |
|---|---|
| `network/` | Graph-based network model: nodes (quantum memory), links (fidelity, success probability, capacity, latency), request objects, and topology generators |
| `models/` | Physical quantum state model (density matrices), decoherence, entanglement swapping |
| `protocols/` | SimPy-driven discrete-event protocols: entanglement generation and swapping over a simulated timeline |
| `fidelity/` | Werner-state fidelity math: entanglement swapping, BBPSSW purification, end-to-end path fidelity |
| `routing/` | Candidate path generation, per-path/per-purification-level "bundle" construction, resource-demand and success-probability models, and the utility function used to score bundles |
| `optimization/` | Solvers that decide which bundle to admit per request under edge/memory capacity constraints: exact QUBO formulation (pyqubo), simulated/quantum annealing (OpenJij), a Metropolis annealer, and a tensor-network optimizer |
| `baselines/` | Classical (non-quantum) allocation baselines, an exact CP-SAT reference solver, shared feasibility/metrics tooling, and a starter ML candidate-ranking model — all built to slot into the same interface as `optimization/` |
| `tests/` | Unit tests for every layer above |

## Core data model

**`QuantumNetwork`** wraps a `networkx.Graph`. Nodes are `QuantumNode`
(memory capacity, T1/T2 decoherence times, per-qubit memory reservations).
Edges are `QuantumLink` (fidelity, generation probability, latency,
capacity).

**`Request`** is a service request: `source`, `destination`,
`minimum_fidelity`, `weight` (priority).

**`Bundle`** (in `routing/bundle_generation.py`) is a *candidate solution*
for a request: a specific path with a specific number of purification
rounds per hop, carrying its resulting end-to-end fidelity, latency,
success probability, resource demand (edges + memory), and a computed
`utility` score. Every optimizer and baseline in this repo consumes lists
of bundles and decides which single bundle (if any) to admit per request.

Two dict views are used depending on how much detail a solver needs:

- `Bundle.to_optimizer_dict()` — the minimal fields every solver requires:
  `bundle_id, request_id, path, edge_demands, memory_demands, utility`
- `Bundle.to_dict()` — the full set, additionally including `fidelity`,
  `success_probability`, `latency`, `bell_pair_cost`, `purification_rounds`

## Pipeline

topology.generate_*() -> QuantumNetwork
|
requests: list[Request] (baselines/request_generator.py, or built by hand)
|
PathGenerator / CandidatePathGenerator
-> candidate paths per request
|
BundleGenerator.generate_bundles(request, candidate_paths)
-> list[Bundle] (one per path x purification-level combination)
|
[b.to_optimizer_dict() for b in bundles], edge_capacities, memory_capacities
|
v
┌─────────────────────────────┬──────────────────────────────┐
│ optimization/ │ baselines/ │
│ - QUBOOptimizer (pyqubo) │ - 8 classical allocators │
│ - MetropolisAnnealer │ - CPSATAllocator (exact) │
│ - TensorNetworkOptimizer │ - BundleRankingModel (ML) │
│ - OpenJij SA / SQA sampler │ │
└─────────────────────────────┴──────────────────────────────┘
|
result: {"selected": [(request_id, bundle_id), ...], "total_utility": ...}
|
baselines/feasibility.py: compute_metrics() / compare_all() / print_comparison()


Every solver — quantum or classical — returns the same
`{"selected": [...], "total_utility": ...}` shape, so any of them can be
dropped into `baselines/feasibility.py`'s comparison tooling interchangeably.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows

pip install -r requirements.txt        # core solvers (pinned)
pip install "qiopt4qnet[baselines]"    # optional: CP-SAT oracle + ML ranking baseline
pip install pytest                     # only needed for the test suite
```

Versions are pinned in `pyproject.toml`/`requirements.txt` to the
environment `RESEARCH_FINDINGS.md` was run on. `minorminer` (quantum-annealing
minor embedding) is a required core dependency; `ortools` (exact CP-SAT
baseline) and `scikit-learn` (legacy ML ranking baseline) are optional and
install via the `[baselines]` extra.

(If your environment enforces PEP 668, add `--break-system-packages` to
the pip commands, or just use a venv as above.)

Tests and demo scripts assume `src/` is on the Python path. `pyproject.toml`
already configures this for pytest, and the demo scripts add it themselves
at the top of the file, so no manual `PYTHONPATH` is needed for those. For
your own scripts, either run from `QNet_Sim/` with `PYTHONPATH=src`, or
add `sys.path.insert(0, "src")` before importing.

## Running things

**Discrete-event protocol simulation** (entanglement generation/swapping
over simulated time):
```bash
python3 run_simulation_demo.py
```

**Quantum/QUBO optimizers** (QUBOOptimizer, OpenJij SA/SQA,
MetropolisAnnealer, TensorNetworkOptimizer) on a generated instance:
```bash
python3 run_quantum_optimizers_demo.py
```

**Bundle generation walkthrough** (candidate paths → bundles → one
optimizer):
```bash
python3 run_bundle_optimizer_demo.py
```

**Classical baselines vs. quantum optimizers**, side by side on the same
instance:
```bash
python3 run_classical_baselines_demo.py
```

**Interactive demo**:
```bash
python3 run_interactive_demo.py
```

**Tests:**
```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Classical baselines (`baselines/`)

These exist to answer "is the quantum/QUBO solver actually worth it?" —
they share the exact interface `QUBOOptimizer` and `MetropolisAnnealer`
use, so they can be benchmarked head-to-head with no glue code.

```python
from baselines.classical_baselines import run_all_baselines
from baselines.feasibility import compare_all, print_comparison

opt_bundles = [b.to_optimizer_dict() for b in bundles]
results = run_all_baselines(opt_bundles, edge_capacities, memory_capacities, seed=42)
metrics = compare_all(results, opt_bundles, edge_capacities, memory_capacities)
print_comparison(metrics)
```

Included allocators:

- `random_feasible` — random order, random pick; sanity-check lower bound
- `shortest_feasible_path` — fewest hops first
- `highest_fidelity_first` / `highest_success_first` / `lowest_resource_cost_first` — per-request greedy heuristics
- `utility_per_resource_greedy` — global greedy by utility-per-resource
- `congestion_aware_greedy` — global greedy, rescored against *residual* capacity after every pick
- `greedy_local_search` — greedy seed + hill-climbing swaps
- `cp_sat_exact` — exact MILP via OR-Tools CP-SAT (requires `pip install ortools`); this is the reference/oracle solution

`baselines/feasibility.py` provides `FeasibilityChecker`, `compute_metrics`
(acceptance rate, total utility, average fidelity, capacity violations,
edge/memory utilization, optimality gap vs. a reference), `compare_all`,
and `print_comparison`.

`baselines/request_generator.py` and `baselines/path_strategies.py`
provide reusable low/medium/high-load workload generation and additional
candidate-path strategies (highest-fidelity, highest-success,
lowest-latency, capacity-aware) on top of `PathGenerator`'s k-shortest-paths.

## ML candidate-ranking baseline

`baselines/dataset_generator.py` runs the CP-SAT exact solver as an oracle
over many random instances and writes per-candidate-bundle features +
whether the oracle selected them:

```python
from baselines.dataset_generator import generate_dataset
path = generate_dataset(n_instances=50, out_path="dataset.jsonl", n_nodes=8, load="medium")
```

`baselines/ranking_model.py` trains a small MLP to imitate the oracle's
selections, and a feasibility-aware greedy decoder turns its scores into a
valid (capacity-respecting) allocation:

```python
from baselines.ranking_model import load_dataset, BundleRankingModel, feasibility_aware_decode

records = load_dataset(path)
model = BundleRankingModel().fit(records)
scores = model.score_bundles(candidate_dicts)   # {(request_id, bundle_id): score}
result = feasibility_aware_decode(opt_bundles, scores, edge_capacities, memory_capacities)
```

This is a flat-feature starter model. The work plan calls for a graph
encoder (GraphSAGE/GAT) over network structure eventually — swap it in for
`BundleRankingModel` without touching the decoder, since
`feasibility_aware_decode` only needs a `{(request_id, bundle_id): score}`
mapping regardless of how the scores were produced.

## Extensions and the experiment suite

Beyond the core solver layers, `src/extensions/` and `src/routing/` +
`src/optimization/` hold paper-style extensions that plug into the same
bundle interface:

| Module | Extension |
|---|---|
| `extensions/adaptive_qubo.py` | Adaptive candidate reduction: keep the top-`k` bundles per request (by fidelity/congestion filters) before building and annealing the QUBO, then compare against the full-size reference QUBO |
| `extensions/hybrid_pipeline.py` | Hybrid pipeline: candidate reduction → small QUBO solve → feasibility repair/refine, benchmarked against QUBO-only ablations |
| `extensions/multi_objective.py` | Multi-objective selection: throughput/fidelity/success/latency/memory trade-offs via Pareto-frontier enumeration and `ε`-constraint queries |
| `extensions/disjoint_paths.py` | `k`-disjoint-path provisioning: composite bundles over edge-disjoint routes with success-probability redundancy |
| `extensions/swapping_order.py` | Entanglement-swap-tree ordering (linear / balanced / optimal) under T1/T2 memory decay and fidelity-vs-`τ_mem` sweeps |
| `extensions/topologies.py` | Generative topology families (ring, random-geometric, Erdős–Rényi, Watts–Strogatz, Barabási–Albert) to test solver robustness to network shape |
| `routing/temporal_request.py`, `routing/memory_scheduler.py` | Time-aware requests (arrival/deadline/priority) and a temporal memory scheduler with fidelity decay and T1/T2 windows |
| `optimization/joint_scheduler.py`, `optimization/online_optimizers.py` | Joint routing + temporal-memory scheduling vs memory-agnostic baselines; static / online / receding-horizon regimes |
| `baselines/gnn_ranker.py` | GraphSAGE candidate ranker that scores bundles from graph features and feeds a feasibility-aware greedy / QUBO decode |
| `simulation/discrete_event_engine.py` | Stochastic discrete-event engine that samples the entanglement pipeline (generation, swapping, purification, delivery) and reports sampled vs parametric utility and SLA statistics |
| `simulation/recourse.py` | Adaptive recourse: local repair of failed requests vs full reoptimization, with recovery-rate and wall-time speedup |
| `optimization/purification_scheduler.py` | Purification as a first-class variable: joint fidelity/memory purification scheduling vs entanglement-only provisioning, plus cost/fidelity sweeps |
| `experiments/optimality_benchmark.py` | Exact-ILP optimality-gap certification and parametric-vs-sampled stochastic reliability benchmarks |
| `optimization/adaptive_budget.py` | Congestion/density-driven adaptive QUBO candidate budget vs fixed top-k and full-candidate baselines |
| `optimization/quantum_annealing_backend.py` | Quantum-annealing backend: minor-embedding onto a hardware lattice, PIA sampler with chain-break-aware decode, compared against SA/SQA |
| `optimization/chance_constrained.py` | Chance-constrained routing: replaces the hard rule `F ≥ F_min` with the quantile constraint `P(F_r ≥ F_min) ≥ 1-ε` under truncated-normal fidelity noise; hard, chance(ε), and nominal policies solved by exact ILP and executed in the DES engine to certify that the empirical SLA stays ≤ ε |
| `optimization/conflict_purification_scheduler.py` | Network-level, conflict-aware purification scheduling (PSC-style): fixed Dijkstra routing, then a per-link conflict metric (aggregated purification need across concurrent requests, scaled by remaining Bell-pair scarcity) drives probabilistic purification decisions shared across every request using a link -- vs `threshold`/`greedy` baselines that decide per-request with no cross-request coordination |

**Run the full experiment battery** (writes ~42 CSVs to `results/experiments/`):

```bash
PYTHONPATH=src python3 src/experiments/experiment_suite.py
```

**Regenerate all figures and summary tables** (writes PNGs + `summary_tables.txt`
to `results/experiments/figures/`):

```bash
PYTHONPATH=src python3 src/experiments/plot_results.py
```

Each `extensions/` module and the temporal/joint scheduling stack is covered
by unit tests under `tests/` (`test_extensions_*.py`, `test_joint_scheduling.py`,
`test_adaptive_qubo.py`, `test_hybrid_pipeline.py`, `test_gnn_ranker.py`,
`test_discrete_event_simulation.py`, `test_recourse.py`,
`test_purification_scheduler.py`, `test_optimality_benchmark.py`,
`test_adaptive_budget.py`, `test_quantum_annealing_backend.py`,
`test_chance_constrained.py`, `test_conflict_purification_scheduler.py`).

## Physics notes

- Fidelities are modeled as Werner-state parameters. Entanglement swapping
  combines two link fidelities via `f1*f2 + (1-f1)(1-f2)/3`; BBPSSW
  purification improves fidelity probabilistically (only above F > 0.5)
  at a known success-probability cost.
- `models/quantum_state.py` additionally implements a full density-matrix
  simulation (2-qubit Bell states, T1/T2 decoherence channels, and
  entanglement swapping via Kraus operators and partial trace) used by the
  SimPy protocol simulation in `protocols/`.
- `routing/utility.py` scores a bundle as
  `weight * success_probability * (1 + beta * max(0, fidelity - min_fidelity)) - lambda_latency * latency - lambda_cost * bell_pair_cost` —
  rewarding fidelity margin above the request's requirement while
  penalizing latency and resource cost.

## Project status

Actively developed. The entanglement-routing formulation (fidelity models,
purification rules, memory constraints, objective function) is expected to
evolve — the bundle/optimizer interface is intentionally decoupled from
those physical models so they can change without breaking the optimization
or baseline layers.