# Run configurations

A run is described by one TOML file. `python run.py FILE.toml` prints the plan and a time estimate, asks to proceed, computes every combination, and draws the figures. Every list is a set of values. The run computes every combination of the lists in `[markets]` and `[design]`, the ζ values of each distribution, and the `[[optimizers]]` or `[[thresholds]]` tables. `python run.py --options` prints the defaults below.

| File | Purpose |
|---|---|
| [standard_test.toml](standard_test.toml) | standard test of the code: every feature on small grids, about 1 minute |
| [evaluate_example.toml](evaluate_example.toml) | evaluate run: given thresholds, and the optima of an earlier run |
| [optimize_example.toml](optimize_example.toml) | research-size optimize run on the tweet data (several ζ, per-node τ) |

## Top level

| Key | Values | Meaning |
|---|---|---|
| `kind` | `"optimize"`, `"evaluate"` | optimize: search for τ in each combination, then evaluate it; evaluate: evaluate given τ |
| `name` | text | part of the run folder name `<date>_<time>_<kind>_<name>` |
| `description` | text | shown in the run's README |

## `[[distributions]]` (one table per distribution; at least one)

| Key | Default | Meaning |
|---|---|---|
| `source` | required | a synthetic family (`a`–`f`, `h`, `example_n`, `independent`, `harm_rises`, `harmful_middle`, `heavy_tail`), the data set `tweets`, or a path to a CSV file with columns `e_score`, `v_score` |
| `label` | family name / data set name / file name | folder name of the distribution |
| `externality_scales` | `[1.0]` | the externality scales ζ (e = ζ e_score); one sub-folder `z<ζ>` each |
| `num_externality_nodes` | 151 (synthetic), 241 (empirical) | externality grid of the piecewise-uniform distribution |
| `num_valuation_cells` | 150 (synthetic), 400 (empirical) | valuation cells per externality node |
| `options` | `{}` | synthetic only: family parameters, e.g. `options = {correlation = -0.4}` for `harm_rises` (see `audited_auctions/README.md`) |
| `valuation_scale` | 1.0 | empirical only: value of one action (v = valuation_scale × v_score) |
| `truncation_quantile` | 0.995 | empirical: training rows with v above this quantile of all rows are dropped; the quantile is the valuation bound v̄ |
| `evaluation_truncation_quantile` | same as `truncation_quantile` | empirical: the same for the test rows (1.0 keeps every row) |
| `train_fraction` | 0.5 | empirical: share of rows in the training half |
| `smoothing` | `"kernel"` | empirical: training half kernel-smoothed (`"kernel"`) or used as atoms (`"none"`) |
| `bandwidth_scale` | 1.0 | empirical: multiplier of the kernel bandwidth |
| `evaluation` | `"atomic"` | empirical: test half as atoms (`"atomic"`) or smoothed like the training half (`"smoothed"`) |

## `[markets]`

| Key | Default | Meaning |
|---|---|---|
| `num_bidders` | `[8]` | numbers of bidders n |
| `num_goods` | `[1]` | numbers of identical goods k |

## `[design]` (optimize runs)

| Key | Default | Meaning |
|---|---|---|
| `threshold_forms` | `["knots6"]` | forms of τ: `poly<d>` (polynomial of degree d), `knots<K>` (continuous piecewise linear with K knots + exclusion cutoff), `bins<K>` (constant on K bins), `nodes` (free at every externality node) |
| `objectives` | `["vcg_pa"]` | auction whose exact expected welfare τ maximizes: `vcg_pa`, `ivcg_pa` |

## `[[optimizers]]` (optimize runs; default: one best-response optimizer)

Combinations of a method with a form it cannot search are skipped and listed in the plan: the genetic search takes `poly`, `knots`, `bins`; best-response ascent takes `knots`, `bins`, `nodes`.

| Key | Default | Meaning |
|---|---|---|
| `label` | `ga`, `br`, `ga-br` | folder-name label of the optimizer |
| `method` | required | `genetic` (genetic stage + Nelder–Mead), `best_response` (best-response ascent + gradient polish), `genetic+best_response` (ascent also started from a genetic result) |
| `num_generations` | 500 | genetic: maximum generations |
| `population_size` | 50 | genetic: population size |
| `num_parents_mating` | 15 | genetic: parents per generation |
| `mutation_probability` | 0.25 | genetic: mutation probability per gene |
| `range_multiplier` | 3.0 | genetic: width of the search ranges of polynomial coefficients |
| `local_refinement` | `"nelder_mead"` | genetic: `"nelder_mead"` or `"none"` |
| `stall_generations` | 60 | genetic: stop after this many generations without improvement |
| `num_local_starts` | 5 | genetic: Nelder–Mead starts |
| `local_max_evaluations` | 2000 | genetic: evaluations per Nelder–Mead start |
| `num_restarts` | 1 | genetic: independent restarts (seeds `search`, `search + 1`, …) |
| `starts` | `["zero", "no_competition"]` | best response: starts (`zero`: τ ≡ 0; `no_competition`: τ(e) = max(0, −e); `short_genetic`) |
| `max_iterations` | 100 | best response: iterations per start |
| `relative_tolerance` | 1e-10 | best response: stop when relative welfare gains fall below this |
| `polish_rounds` | 3 | best response: gradient polishes |
| `short_genetic_generations` | 40 | best response: generations of the `short_genetic` start |
| `genetic_form` | same form, or `knots6` for `nodes` | `genetic+best_response`: form of the genetic stage |
| `warm_start_across_zeta` | false | best response: also start from the optimum at the previous ζ, rescaled (τ(e) = τ_prev(e ζ_prev / ζ)); the ζ values of a distribution then run in sequence |

## `[[thresholds]]` (evaluate runs; at least one)

Each table has a `label` (folder name `tau-<label>`) and exactly one source:

| Key | Example | Meaning |
|---|---|---|
| `polynomial` | `[0.0, -1.0]` | τ(e) = Σ_m β_m e^m (coefficients from degree 0) |
| `constant` | `0.3` | τ(e) = 0.3 |
| `tabulated` | `{externalities = [-1, 0], values = [1, 0]}` | piecewise linear through the points, constant outside |
| `tau_file` | `"runs/<run>/<path>/tau.json"` | a τ saved by an earlier run |
| `from_run` | `"runs/<run>"` | every τ of an earlier run with the same distribution label, ζ, n and k (no `label`: the original design label is used) |

A τ value at or above v̄ excludes the externality (τ = v̄⁺ in the paper).

## `[evaluation]`

| Key | Default | Meaning |
|---|---|---|
| `num_simulated_auctions` | 2000 | simulated auctions for the per-auction figures (the same auctions for every design of a market) |
| `num_bootstrap` | 200 | empirical: bootstrap resamples of the test rows |
| `confidence_level` | 0.95 | level of the bootstrap percentile intervals |
| `bootstrap_all_outcomes` | true | bootstrap every metric (revenue, penalties, welfare parts), not only welfare |

## `[checks]`

| Key | Default | Meaning |
|---|---|---|
| `grid_refinement` | false | synthetic: recompute on a finer grid with τ fixed |
| `refinement_factor` | 2 | refinement of both grids |
| `grid_relative_tolerance` | 0.01 | pass if every gain over VCG changes by less than this share |
| `bandwidth_sensitivity` | `[]` | empirical, optimize runs: redesign τ with these bandwidth multipliers, e.g. `[0.5, 2.0]` |
| `first_order_tolerance` | 1e-4 | `nodes` form: pass if max \|M(e, τ(e))\| / v̄ at interior thresholds is below this |
| `monte_carlo_z` | 4.0 | pass if every simulated mean is within this many standard errors of the exact value |

## `[seeds]`

| Key | Default | Used by |
|---|---|---|
| `split` | 0 | train/test split of empirical rows |
| `search` | 0 | genetic search and `short_genetic` starts |
| `simulation` | 1234 | simulated auctions |
| `bootstrap` | 0 | bootstrap resamples |

## `[output]`

| Key | Default | Meaning |
|---|---|---|
| `base` | `"runs"` | folder of the run folders (relative to the code folder) |
| `workers` | 0 | parallel processes; 0: number of cores − 1, at most 8 (`--workers` overrides) |
| `search_history_curves` | 400 | tried thresholds kept for `search/tried-taus.pdf` |
| `report` | true | draw figures after computing (`--no-report` overrides) |
