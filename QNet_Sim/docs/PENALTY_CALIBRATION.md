# OpenJij Penalty Calibration Benchmark

## Purpose

OpenJij SA/SQA should serve as a standardized QUBO baseline. The repository
should therefore preserve a simple conventional coefficient rule and compare it
against the proposed resource-aware rules under identical solver settings.

The calibration study should answer a narrow question:

> Does resource-aware edge/memory calibration reduce sufficient penalty
> coefficients while preserving exactness, and do those reductions affect raw
> feasibility, solution quality, or robustness under fixed solver settings?

## Calibration variants

All three variants leave the request-conflict penalty `A` on the conventional
utility scale, so experiments isolate the effect of edge/memory calibration.

### Conventional reference

`conventional_coefficients()` sets the hard request, edge, and memory penalties
just above the largest positive bundle utility. This is intentionally simple
and transparent.

### Global resource-aware proposal

`proposed_global_coefficients()` calculates single `B` and `D` values from the
edge/memory capacities, bundle demands, reachable competing loads, and bundle
utilities. The implementation uses bounded dynamic programming to avoid the
exponential full-load enumeration used in the initial version.

### Per-resource proposal

`proposed_resource_coefficients()` applies the same single-removal certificate
independently to each edge and each memory node, returning `B` and `D` as
mappings rather than scalars. `QUBOOptimizer` carries one placeholder per
resource (`B_0`, `B_1`, ..., `D_0`, ...), and `_feed` accepts either a scalar
or a per-resource mapping.

By construction the global rule is the maximum of the per-resource ones:

```text
B_global = max_e B_e     and     D_global = max_v D_v
```

so `B_e <= B_global <= B_ref` for every resource. The per-resource rule
therefore **cannot** lower the largest coefficient; what it lowers are the
non-binding resources.

### Soft congestion is separate

`C` and `E` are soft load-balancing/congestion regularizers. They are **not**
part of the hard-constraint calibration claim. For the clean calibration
ablation, run with `C = E = 0`. Congestion can then be restored in a separate
experiment.

## Main experiment

From `QNet_Sim`:

```bash
PYTHONPATH=src python src/experiments/run_penalty_calibration.py
```

This runs the quick comparison over chain/grid instances using:

- OpenJij SA and SQA;
- conventional and resource-aware calibration;
- coefficient sensitivity multipliers `0.25x, 0.5x, 1x, 2x, 4x`;
- multiple independently generated network/request instances;
- CP-SAT as the exact reference when OR-Tools is installed.

A larger paper-style sweep is:

```bash
PYTHONPATH=src python src/experiments/run_penalty_calibration.py --full --reads 100
```

Outputs are written to:

```text
results/penalty_calibration/calibration_runs.csv
results/penalty_calibration/calibration_summary.csv
```

Note that this driver runs the `conventional` and `resource_aware` strategies
only. Adding `resource_aware_per` to its strategy list also requires wrapping
its `coeffs["B"]` / `coeffs["D"]` CSV fields in `coefficient_stats()`, because
those coefficients are mappings rather than scalars.

## Metrics that matter

Do not judge calibration only after feasibility repair. Repair can hide poor
penalty choices.

The benchmark therefore reports:

- raw feasible-read rate;
- raw request conflicts;
- raw edge/memory violations and overload;
- best raw-feasible utility;
- repaired utility as a secondary operational metric;
- CP-SAT optimality gap where available;
- coefficient magnitudes;
- QUBO compile/calibration time;
- SA/SQA sampling time.

The primary evidence for calibration quality is the size of the sufficient
penalty coefficients together with raw feasibility and solution quality at `1x`.
A tighter coefficient rule is useful even if heuristic solver quality is
unchanged, provided that the same hard-constraint guarantee is preserved.

## Held-out validation

A separate held-out experiment evaluates the calibration rules on instance
seeds `10` through `99`, which were not used in the initial exploratory
calibration experiments. The request-count dependence of the resource-aware
bound was predicted before the held-out evaluation: sparse reachable-load sets
were expected to permit stronger tightening, while increasing contention was
expected to make reachable loads denser and drive the minimum realizable
penalty drop toward `1`.

From `QNet_Sim`:

```bash
PYTHONPATH=src python3 src/experiments/run_penalty_calibration_heldout.py
```

A coefficient-only scan can be run with:

```bash
PYTHONPATH=src python3 src/experiments/run_penalty_calibration_heldout.py --scan-only
```

The held-out coefficient scan covers 1,080 instances across four topology
configurations, 90 fixed held-out instance seeds, and request counts `8`, `16`,
and `24`.

### Coefficient tightening at the exact `1x` scale

Under the **global** rule:

- `B` was strictly lower in `121 / 1080` instances (`11.20%`);
- `D` was strictly lower in `67 / 1080` instances (`6.20%`);
- both were lower simultaneously in `61 / 1080` instances (`5.65%`);
- at least one was lower in `127 / 1080` instances (`11.76%`);
- conditional on strict tightening, `B` was reduced by `38.42%` on average;
- conditional on strict tightening, `D` was reduced by `48.45%` on average.

Under the **per-resource** rule, at least one edge coefficient and at least one
memory coefficient were strictly lower in `1078 / 1080` instances (`99.81%`).

### Read the coefficient reduction carefully

| family | mean over resources | max over resources | global rule |
| --- | --- | --- | --- |
| edge `B` | `49.51%` reduction | `4.30%` | `4.30%` |
| memory `D` | `46.34%` reduction | `3.01%` | `3.01%` |

The max-over-resources column equals the global rule exactly, because
`B_global = max_e B_e` by definition. The large mean reduction is therefore an
average over an instance's `12.4` edge and `10.4` memory resources, of which
`11.0` and `8.3` respectively are tightened. It reflects lowering **non-binding**
resource penalties, which compresses the BQM's coefficient dynamic range; it is
not a reduction of the binding penalty, which is mathematically impossible under
this construction.

### Matched solver study

The 1,078 instances with a resource-coefficient treatment difference were
evaluated using matched conventional, global, and per-resource QUBOs. Selection
depended only on whether coefficients changed, not on solver performance.

Each selected instance used SA and SQA, five fixed solver seeds
(`101, 202, 303, 404, 505`), and 100 reads per run. CP-SAT was used as an exact
reference when optimality was certified.

### Paired solver analysis

The completed solver runs can be analyzed without rerunning OpenJij:

```bash
PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_heldout.py
```

Conventional, global and per-resource runs are paired using the same topology,
instance seed, request count, sampler, and solver seed. The five solver-seed
differences are averaged within each instance, leaving 1,078 instance-level
paired comparisons per contrast and sampler.

Three contrasts are reported, because per-resource calibration must be judged
against the **global** rule rather than against the conventional reference:

- `resource_aware - conventional`;
- `resource_aware_per - conventional`;
- `resource_aware_per - resource_aware`.

That gives `3 contrasts x 2 samplers x 3 metrics = 18` comparisons. **No
multiplicity correction is applied**, so roughly one interval in twenty would be
expected to exclude zero by chance alone.

Intervals are percentile bootstrap CIs over 10,000 resamples. Resampling is at
the level of **instance-seed blocks** rather than individual instances, so
instances generated from the same held-out seed stay together. The statistic is
the ratio-of-sums across resampled blocks, which equals the overall mean paired
difference. Each comparison's bootstrap seed is derived from the comparison
identity itself (`contrast_seed()`), so the reported intervals are reproducible
across processes.

Headline results for the contrast that isolates the per-resource contribution
(`resource_aware_per - resource_aware`):

| sampler | metric | mean difference | 95% CI |
| --- | --- | --- | --- |
| SA | raw feasible rate | `+0.1160` | `[+0.1005, +0.1315]` |
| SA | repaired reference gap | `-10.33 pp` | `[-11.20, -9.46]` |
| SA | raw overload units | `-1.534` | `[-1.680, -1.386]` |
| SQA | raw feasible rate | `+0.0686` | `[+0.0515, +0.0863]` |
| SQA | repaired reference gap | `-5.95 pp` | `[-6.58, -5.35]` |
| SQA | raw overload units | `+0.0173` | `[+0.0004, +0.0362]` |

The global rule alone produces much smaller effects relative to conventional
(SA raw feasibility `+0.0041`, SQA `+0.0071`). One comparison is not
distinguishable from zero: `resource_aware_per - conventional` on SQA raw
overload units has a CI lower bound of `-5.1e-19`, which straddles zero at
floating-point resolution and should be described as indistinguishable rather
than significant.

### Reachable-load mechanism analysis

The request-count dependence of the coefficient tightening can be analyzed
without running SA, SQA, or CP-SAT:

```bash
PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_mechanism.py
```

This analysis regenerates all 1,080 held-out instances and directly measures the
minimum realizable resource-penalty drop `delta` used by the resource-aware
bound.

For edge constraints, the mean fraction of eligible bundle/resource pairs with
`delta = 1` increased with request count:

- `n = 8`: `75.32%`;
- `n = 16`: `90.93%`;
- `n = 24`: `96.95%`.

For memory constraints:

- `n = 8`: `57.75%`;
- `n = 16`: `83.71%`;
- `n = 24`: `94.13%`.

At the same time, strict global coefficient tightening decreased sharply.

For `B`:

- `n = 8`: `26.11%` of instances;
- `n = 16`: `8.61%`;
- `n = 24`: `2.22%`.

For `D`:

- `n = 8`: `12.50%`;
- `n = 16`: `0.83%`;
- `n = 24`: `0.28%`.

The most direct saturation diagnostic is whether a maximum-utility bundle
(`P0`) also has a realizable `delta = 1` case. Such a pair contributes
`P0 / 1 = P0` and therefore pins the resource-aware bound to the conventional
scale.

The observed `P0`-pin frequencies were:

- edge: `73.89%`, `91.39%`, and `97.78%` for
  `n = 8, 16, 24`, respectively;
- memory: `87.50%`, `99.17%`, and `99.72%`.

Within each request-count/family group, the `P0`-pin frequency was the exact
complement of the strict-tightening frequency. The held-out data therefore
directly support the predicted saturation mechanism: increasing request count
makes unit penalty drops increasingly reachable, which progressively removes
the opportunity for **global** coefficient tightening. Per-resource calibration
is far less exposed to this saturation, because a single `P0`-pinned resource
no longer raises the coefficient of every other resource.

The held-out experiment and follow-up analyses write:

```text
results/penalty_calibration/heldout_coefficient_scan.csv
results/penalty_calibration/heldout_tightening_solver_runs.csv
results/penalty_calibration/heldout_tightening_solver_summary_global.csv
results/penalty_calibration/heldout_tightening_solver_summary_by_regime.csv
results/penalty_calibration/heldout_paired_analysis.csv
results/penalty_calibration/heldout_mechanism_analysis.csv
```

In `heldout_tightening_solver_runs.csv`, `B_ratio` and `D_ratio` are
instance-level ratios describing the **global** rule, so they are constant
across the three arms of an instance. The arm-specific ratios are
`B_ratio_arm` and `D_ratio_arm`.

## Tests

```bash
PYTHONPATH=src pytest tests/test_penalty_calibration.py -v
```

The tests cover utility-scale calibration, reachable-load calculation,
separate edge/memory bounds, no-overload cases, sensitivity scaling for both
scalar and per-resource coefficients, scalar/mapping equivalence, reversed edge
keys, missing-coefficient errors, and API validation.

Two tests check ground-state correspondence for per-resource calibration:

- `test_per_resource_preserves_ground_state` enumerates every selection and
  compares the minimum-energy selection against the maximum-utility feasible
  selection. Its instance is constructed so that two bundles of the same
  request are jointly capacity-feasible and jointly worth more than the best
  legal selection, so the test fails if the at-most-one penalty `A` is dropped.
- `test_per_resource_ground_state_of_compiled_qubo` brute-forces the **compiled**
  QUBO returned by `to_qubo()` rather than a hand-written copy of the
  Hamiltonian, so a mis-wired `B_i` / `D_i` placeholder cannot pass unnoticed.

## Recommended paper framing

Use:

- **Conventional calibration** = required reference baseline.
- **Global resource-aware calibration** = first proposed rule.
- **Per-resource calibration** = the scoped variant, evaluated against the
  global rule and not only against the conventional reference.
- **OpenJij SA/SQA** = standardized QUBO solver family, not itself the novel
  contribution.

The defensible claim is that resource-aware calibration gives a provably
no-larger sufficient resource-penalty calibration while preserving the same
exact ground-state guarantee, and that scoping it per resource additionally
improves SA and SQA raw feasibility and repaired optimality gap relative to the
global rule under fixed solver settings.

Report both the frequency and magnitude of tightening and the measured
saturation mechanism, and state explicitly that the per-resource rule does not
reduce the largest coefficient.
