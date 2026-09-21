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

By default this driver runs the `conventional` and `resource_aware` strategies.
Add the per-resource rule with `--strategies conventional resource_aware
resource_aware_per`. For mapping-valued coefficients the `B` / `D` columns hold
the mean over resources, alongside `B_min`, `B_max`, `B_n` (and likewise for
`D`); for scalar coefficients `min = max = mean` and `n = 1`.

## Solver-budget sensitivity

To check that the per-resource advantage is not just compensating for an
under-budgeted annealer, `run_penalty_calibration_budget.py` reruns matched
conventional and per-resource QUBOs at several budgets (reads, and SA sweeps)
and reports paired per-resource-minus-conventional effects with bootstrap
intervals:

```bash
# about 40 randomly sampled held-out instances (slow: minutes to hours)
PYTHONPATH=src python3 src/experiments/run_penalty_calibration_budget.py

# every held-out instance selected by the per-resource criterion
PYTHONPATH=src python3 src/experiments/run_penalty_calibration_budget.py --full
```

Outputs: `results/penalty_calibration/budget_sensitivity_runs.csv` and
`budget_sensitivity_summary.csv`. Each run row records `n_samples`, the number of
samples OpenJij actually returned, and the summary column `samples_match_reads`
is `True` only if it equals the requested reads in every row. This makes a
budget that fails to reach the solver visible. `solve_sa` accepts an optional
`num_sweeps`.

**`num_reads` does reach the solver, but with a fixed seed the reads are
identical.** For both SA and SQA, requesting 1, 100 or 1000 reads returns
exactly that many samples (`n_samples == reads`), and in every case checked
(chain8 and grid3x3 instances, `seed=101`) the returned samples are a single
distinct state (`n_distinct_samples == 1`). The earlier observation that 100 and
1000 reads give identical effects is therefore explained by the seeded call
returning the same read repeatedly, not by the read count being ignored and not
(as suggested in the discussion of PR #20) by greedy repair saturating. It
also explains why every archived call in the held-out solver study is uniformly
feasible or uniformly infeasible (`raw_feasible_rate` is 0 or 1; the analysis
asserts this): a "100-read" call is one sample per solver seed. Within-call
reads are not independent replicates, and the read budget in the held-out study
is not a budget in the usual sense. `--independent-reads` draws each read as
its own call with its own seed (`seed, seed+1, ...`), which makes extra reads
real but is slow: a single-read call costs about `17` ms on a 65-variable
instance and about `230` ms on a 209-variable one, so 1000 independent reads
take roughly `17` s and `230` s respectively.

Do not quote a small-sample magnitude as the effect size: the PR #20 discussion
reports about `-16` pp for the repaired-gap effect on a 40-instance subsample,
against `-10.8` pp on the full cohort.

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

That gives `3 contrasts x 2 samplers x 3 metrics = 18` comparisons. The
restricted analysis below adds `1 contrast x 2 samplers x 3 metrics = 6` more,
for `24` in total. **No multiplicity correction is applied**, so roughly one
interval in twenty would be expected to exclude zero by chance alone.

The five solver seeds are fixed and averaged within each instance, so the
intervals reflect variation over generated instances (resampled as
generation-seed blocks) only, not over solver seeds. The percentile bootstrap
uses 10,000 resamples of the 90 generation-seed blocks.

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

### Global rule restricted to instances where it changed a coefficient

The pooled global-versus-conventional effect above is averaged over all 1,078
selected instances, of which `951` (`88.2%`) have coefficients identical to the
conventional reference (they were selected on the per-resource criterion) and so
contribute exact zeros. The same contrast restricted to the `127` instances
where the global rule tightened `B` or `D`
(`heldout_paired_analysis_global_changed.csv`, written by the same script):

| sampler | metric | mean difference | 95% CI |
| --- | --- | --- | --- |
| SA | raw feasible rate | `+0.0346` | `[+0.0063, +0.0656]` |
| SA | repaired reference gap | `-4.27 pp` | `[-6.98, -1.94]` |
| SA | raw overload units | `+0.0110` | `[-0.1687, +0.2276]` |
| SQA | raw feasible rate | `+0.0598` | `[+0.0133, +0.1079]` |
| SQA | repaired reference gap | `-6.77 pp` | `[-9.96, -3.88]` |
| SQA | raw overload units | `+0.0047` | `[-0.0228, +0.0344]` |

So the global rule does help where it does something (`+3.5` and `+6.0` pp raw
feasibility for SA and SQA), and the pooled `+0.41` / `+0.71` pp is that effect
diluted by instances it never touched. Even here the per-resource gain against
the global rule is larger (`+11.6` / `+6.9` pp pooled).

**Why the earlier 139-instance analysis found no benefit.** The superseded
artifact (commit `3e6536a`) reported a raw-feasibility change of `-1.3` pp for SA
under the global rule. It is not comparable to the current results, for three
reasons that can be checked from the archived CSVs:

1. *Different instance set.* It selected `139` instances, of which only `46` are
   among today's `127`. The B/D ratios in the two scan artifacts differ on `220`
   of the same `1,080` instances, i.e. the calibration itself changed.
2. *Dilution.* The current pooled figure includes `951` instances with a zero
   difference by construction.
3. *Run-to-run variability.* Re-evaluating the same `139` instances with the
   current code gives `+1.3` pp for SA raw feasibility (`+0.29` pp for SQA,
   against `-0.43` pp archived): the sign flips. The conventional arm's raw
   feasibility also differs from the archived run in `311` of `1,390`
   (instance, sampler, seed) cells, although its coefficients are unchanged. A
   `1` pp effect on `139` instances, with one distinct sample per call, is within
   that variability. The cause of the run-to-run differences was not isolated.

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

- `n = 8`: `67.76%`;
- `n = 16`: `88.32%`;
- `n = 24`: `95.39%`.

For memory constraints:

- `n = 8`: `48.00%`;
- `n = 16`: `78.03%`;
- `n = 24`: `90.37%`.

At the same time, strict global coefficient tightening decreased sharply.

For `B`:

- `n = 8`: `24.17%` of instances;
- `n = 16`: `6.94%`;
- `n = 24`: `2.50%`.

For `D`:

- `n = 8`: `16.67%`;
- `n = 16`: `1.94%`;
- `n = 24`: `0.00%`.

The most direct saturation diagnostic is whether a maximum-utility bundle
(`P0`) also has a realizable `delta = 1` case. Such a pair contributes
`P0 / 1 = P0` and therefore pins the resource-aware bound to the conventional
scale.

The observed `P0`-pin frequencies were:

- edge: `75.28%`, `93.06%`, and `97.50%` for
  `n = 8, 16, 24`, respectively;
- memory: `82.78%`, `98.06%`, and `100.00%`.

Within each request-count/family group, the `P0`-pin frequency and the
strict-tightening frequency sum to `100%` for `n = 16` and `n = 24`, and to
`99.4%` for `n = 8` (two instances per family are neither pinned nor
tightened). The held-out data therefore
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
