# `audited_auctions` — exact computation, simulation and τ optimization for a common threshold

Given a joint distribution H of bidder types (e, v) and a common participation threshold function τ(e), this package computes the outcomes of four mechanisms with n exchangeable bidders and k identical goods (unit demand):

| Key | Mechanism | Participants | Ranking score | Penalty |
|-----|-----------|--------------|---------------|---------|
| `vcg` | VCG without audits | all bidders | bid v | none |
| `vcg_pa` | VCG-PA | types with v ≥ τ(e) | bid v | r(e) inducing τ, charged to participants |
| `ivcg_pa` | IVCG-PA | types with v ≥ τ(e) | ironed welfare contribution ω̄^τ(v) | r(e) inducing τ, charged to participants |
| `first_best` | optimal recipient audit | all bidders | w = v + e | −e, charged to winners |

All four are score-based auctions: the k participants with the highest positive scores win, ties are broken uniformly at random, and payments are critical-value (Myerson) payments, so each auction is DSIC. The notation follows the WEB27 core draft (`WEB27/drafting/web27_aa_1p_core/`).

The penalties are outputs, not inputs: since τ is given, the package computes the penalty function that induces τ in each participant-audit auction. It also searches for the τ that maximizes exact expected welfare, in a polynomial, knot, bin or per-node form, by genetic search or best-response ascent. Whole experiments (sets of distributions, markets and designs, with saved data, checks and figures) run from configuration files through `run.py` and the `experiments` subpackage; see the top-level README.

## Quick start

```python
from audited_auctions import evaluate_threshold, synthetic_distribution, polynomial_threshold

distribution = synthetic_distribution("harm_rises")      # piecewise-uniform, v ∈ [0, v̄]
threshold = polynomial_threshold([0.5, -1.0])            # τ(e) = 0.5 − e
evaluation = evaluate_threshold(distribution, threshold, num_bidders=8, num_goods=2,
                                num_simulated_draws=20000, random_seed=0)
print(evaluation.summary_table())          # exact outcomes, gains over VCG, simulated means
print(evaluation.welfare_gains_over_vcg()) # W_m − W_VCG and share of the first-best gain
```

Experiments with figures: `python run.py configs/standard_test.toml` (top-level README).

## Modules

| Module | Contents |
|--------|----------|
| `type_distributions.py` | `TypeDistribution` (cells: an externality, a valuation interval, a probability); populations (atomic), gridded densities (piecewise uniform), discrete externality levels; the synthetic families (`SYNTHETIC_FAMILIES`, `synthetic_distribution`). |
| `smoothing.py` | `kernel_density_distribution`: kernel-smoothed piecewise-uniform distribution of a population. |
| `empirical.py` | Empirical pipeline: load a CSV, train/test split, truncation, smoothed or atomic distribution, bootstrap of exact welfare. |
| `threshold_functions.py` | τ(e) from polynomial coefficients, tabulated values, or a constant. τ = +inf excludes an externality (the threshold v̄⁺ of the paper). |
| `scoring_rules.py` | `IdentityScore` (rank by bid) and `PiecewiseLinearScore` (rank by the ironed welfare contribution, a nondecreasing function of the bid), with the generalized inverses needed for payments. |
| `exact_computation.py` | Quantile representation of a mechanism, exact ironing, interim allocation with ties, exact integration of welfare, revenue and the allocation integral R(b) (the penalty). |
| `auction_simulation.py` | Vectorized Monte Carlo of a score-based auction on sampled type profiles. |
| `threshold_evaluation.py` | `evaluate_threshold` (main entry point), `ThresholdEvaluation`, `expected_welfare` (optimization objective), `ivcg_scoring_rule_for` (IVCG-PA design to apply elsewhere), `penalty_from_table`, `grid_refinement_check`. |
| `threshold_forms.py` | Forms of τ searched by the optimizers: `poly<d>`, `knots<K>` (+ exclusion cutoff), `bins<K>`, `nodes`; `build_threshold_form`, `threshold_from_spec`. |
| `genetic_search.py` | Genetic search over a form's parameters with the exact welfare objective: genetic stage, then Nelder–Mead refinement (default). |
| `marginal_contribution.py` | Exact marginal contribution M^τ(e, v) = (v + e) X^τ(v) − L^τ(v) of a participating type for a fixed auction (rival welfare loss L^τ), as Chebyshev series between breakpoints. |
| `best_response_ascent.py` | Best-response ascent over knots, bins or nodes, with an exact-gradient polish; `first_order_check` (zero marginal contribution and Pigouvian penalty at each threshold). |
| `pipeline.py` | Building blocks of a combination: `prepare_distributions` (training and evaluation distributions), `optimize_threshold` (search in a form with either optimizer), `designed_ivcg_rule`, `bandwidth_sensitivity`. |
| `figures.py` | Every figure of a run, drawn from saved data (style: fixed color and line style per mechanism, light type distribution behind, shared legends). |
| `experiments/config.py` | TOML configurations: defaults, validation, expansion into combinations and folder names. |
| `experiments/compute.py` | One combination: τ (searched or given), exact outcomes, simulation, bootstrap, diagnostics, checks; writes `tau.json`, `tau.csv`, `results.json`, `first-order.csv`, `data.npz`. |
| `experiments/runner.py` | A run: plan, time estimate, parallel computation, `run.toml`, `summary.csv`, `checks.csv`, `runs/index.csv`. |
| `experiments/report.py` | Figures and README of a run from its saved files (`python run.py --report`). |
| `experiments/provenance.py` | Git version, software versions, host (local or BU SCC), data hashes; TOML writer. |

## Type distributions

A `TypeDistribution` is a list of cells j with externality e_j, valuation interval [lower_j, upper_j] and probability p_j; given the cell, the valuation is uniform on the interval. Valuations must be non-negative and bounded (v̄ = largest upper end), as in the model; constructing a distribution with a negative valuation raises an error. Two kinds:

- **Atomic** (lower = upper): a population used as is. Bidders are i.i.d. draws of its rows. Ties between equal valuations have positive probability.
- **Piecewise uniform** (lower < upper): the externality takes the values of a grid of nodes and, given the externality, the valuation has a piecewise-constant density. Valuations have no atoms, so ties between valuation bids have probability zero, and every quantity below is computed exactly *for this distribution*. Relative to a continuous density it approximates, the error shrinks with the grid; `grid_refinement_check` measures it by recomputing on a twice finer grid.

Mixed distributions are not supported. `distribution.with_externality_scale(ζ)` multiplies externalities by ζ.

### Synthetic families

All families have v ∈ [0, v̄]. Grid options: `num_externality_nodes` (default 151) and `num_valuation_cells` (default 150); scripts: `--num-externality-nodes`, `--num-valuation-cells`. The two grids have different errors: coarser externality nodes pool nearby externalities (error about proportional to the spacing) and fix how finely a per-node τ can vary; coarser valuation cells make the conditional density of v a coarser step function (error about proportional to the squared spacing; zero when v given e is already piecewise uniform, as in `independent` and `harmful_middle`), and τ stays continuous in v. Accuracy at a fixed τ (n = 8, k = 2), maximum change of the welfare gains over VCG against 601 × 1200 on `harm_rises`, `harmful_middle`, `d`, `independent`: 1.1·10⁻⁴ at 151 × 150 (10–16 ms per evaluation of the four mechanisms), 2.2·10⁻⁴ at 101 × 200 and 101 × 100, 1.7·10⁻⁴ at 201 × 100, 7.4·10⁻⁵ at 201 × 150; the gains are 0.06–0.5. Density jumps (the band of `harmful_middle`) are cell edges. `synthetic_distribution(name, externality_scale=ζ, **options)` builds one; family parameters are keyword options.

| Name | Description | Why it is useful |
|------|-------------|------------------|
| `a` | single bivariate normal, mean (e, v) = (0, 1), variances 0.1 | baseline, no correlation |
| `b` | two normals side by side, means (−1, 1), (0, 1), variances 0.05 | harmful and benign groups with equal valuations |
| `c` | two normals on a diagonal, means (−1, 1), (0, 2) | harmful types have low valuations (VCG already does well) |
| `d` | two normals on the other diagonal, means (−1, 2), (0, 1) | harmful types have high valuations (VCG does badly) |
| `e` | two normals stacked, means (0, 2), (0, 1) | no harm; audits should not matter |
| `f` | four normals at (−1, 1), (−1, 2), (0, 2), (0, 1) | mixture of the above |
| `h` | single normal, mean (0, 2), variances 0.1 | shifted baseline |
| `example_n` | Example N of the core draft: v ~ U[0, 1], e ∈ {0, −c} with probability ½ each (option `harm` = c, default 0.6) | analytic benchmark: with n = 2, k = 1, VCG-PA welfare 2/3 − c/2 + (c/2)t² − t³/4 at τ(−c) = t, optimum t* = 4c/3, penalty (3/4)t² |
| `independent` | v ~ U[0, 1] independent of e ~ N(−0.3, 0.5²) | independence case of the open question on optimal τ for VCG-PA |
| `harm_rises` | bivariate normal, means (−0.3, 1), sds (0.5, 0.4), correlation −0.6 (option `correlation`) | harm grows with valuation (misinformation-style) |
| `harmful_middle` | v ~ U[0, 1]; given v, e ~ N(−1.5, 0.2²) with probability 0.8 for v ∈ [0.5, 0.75) and 0.1 otherwise, else e ~ N(0, 0.1²) | ŵ^τ is non-monotone, so IVCG-PA irons and beats VCG-PA (a continuous analogue of Example T) |
| `heavy_tail` | log v ~ N(0, 1) truncated at its 99.5% quantile; given v, e ~ N(−0.5 √v, 0.4²) | skew like engagement data; harm grows with valuation |

The mixtures a–f and h are those of `legacy/alternate_distributions.ipynb` with valuation means shifted up by 1 (h unchanged), truncated to v ≥ 0 and to ±5 standard deviations, and renormalized. The shift keeps the shapes while truncation at v = 0 removes almost no mass; the old unshifted CSV samples with negative valuations are in `legacy/data/samples/` and are not used.

### Empirical populations

`empirical.py` and `pipeline.prepare_distributions` (used when the source is a CSV with columns `e_score`, `v_score`):

1. **Scale** (`load_empirical_sample`): e = ζ e_score, v = action_cost × v_score; rows with v ≤ 0 are dropped.
2. **Split** (`split_sample`): a random train/test split of all rows (default ½/½) with a seed common to all combinations of a run (`[seeds] split`). The split is made before truncation, so it does not depend on the truncation level.
3. **Truncate** (`valuation_bound`, `truncate_sample`): each half drops rows with v above a quantile of the full sample's valuations; that quantile is the bound v̄. Training: `truncation_quantile` (default 0.995; on the Community Notes data the top 0.5% is 60 rows from 393k to 1.6M that would dominate the design and double the bootstrap noise). Test: `evaluation_truncation_quantile` (default the same; 1.0 keeps every row, a robustness check that reports welfare for the whole population at the cost of more noise).
4. **Design distribution** (training half): kernel-smoothed by default (`--smoothing none` keeps atoms). τ is optimized on it, and the IVCG-PA scoring rule ω̄^τ is ironed on it (`ivcg_scoring_rule_for`).
5. **Evaluation distribution** (test half): the atomic rows by default (`--evaluation smoothed` smooths them like the training half). τ and the IVCG-PA scoring rule designed on the training half are applied unchanged (`ivcg_scoring_rule=` arguments), so evaluation is out of sample and has no smoothing bias. Penalties r(e) are computed on the evaluation distribution, the population that plays the auction.
6. **Bootstrap** (`bootstrap_welfare`): resample the test rows with replacement and recompute exact welfare of all four mechanisms, with τ and the IVCG-PA scoring rule fixed at the training design. Reports per-resample welfare, paired gains over VCG and shares of the first-best gain, and percentile intervals. Fast on atomic test rows (no smoothing per resample).
7. **Bandwidth sensitivity** (`pipeline.bandwidth_sensitivity`, configuration `[checks] bandwidth_sensitivity = [0.5, 2.0]`): redo the design (the search and the IVCG-PA scores) on the training half smoothed with each bandwidth multiplier and evaluate each design exactly on the same test rows. Shows whether the results depend on the smoothing choice; it costs one extra search per scale.

**Why IVCG-PA is designed on the training half.** With distinct valuations, each valuation level of an atomic population is a single row, so ŵ^τ(v) = v + e of that row: IVCG-PA ironed on the rows it is evaluated on would rank bidders by their true welfare contribution and approach the first best in sample. Designing the scoring rule on the smoothed training half and applying it to the test rows avoids this.

**Kernel smoothing** (`smoothing.py`). Coordinates u = asinh(e/s_e) (s_e = median |e|) and y = log v; Gaussian kernel with Scott's bandwidth matrix (× `bandwidth_scale`), computed by linear binning and FFT convolution; reflection at y = log v̄; cells with externality nodes on a uniform u-grid (default 241, `--num-externality-nodes`) and valuation cells on a uniform y-grid (default 400, `--num-valuation-cells`); on the Community Notes data (ζ = 5, n = 8, k = 1, τ(e) = −0.4e) the VCG-PA gain changes by 0.1% at 241 nodes and 1.1% at 101 nodes, against 481, the valuation uniform within a cell. The estimator is variance-preserving: the data are shrunk toward their mean by (1 + h²)^(−1/2), so the smoothed (u, y) keep the sample mean and covariance. Without this, smoothing in log v inflates the upper tail of v (expected VCG welfare +10% on the Community Notes data at bandwidth scale 1); with it, smoothed and atomic VCG welfare differ by under 1%. Check sensitivity with `bandwidth_scale` 0.5 and 2.

## What is computed, and how

**Quantile representation.** Let P be the participation probability. Participating types are ordered by bid; the quantile q ∈ [0, P] is the participating mass below a bid. The bid distribution is a list of segments in q: one per distinct bid for atoms, and one per elementary bid interval (between consecutive cell end points and thresholds) for piecewise-uniform distributions, on which the bid is linear in q. On each segment the expected welfare contribution ŵ^τ(b) = E[v + e | bid = b, participate] is linear in q (b + E[e | b] for valuation bids, b for the first best) and E[e | b] is constant.

**Ironing (IVCG-PA).** ω̄^τ is the nondecreasing L2 projection of ŵ^τ in quantile space, the derivative of the lower convex hull of Ψ(q) = ∫_0^q ŵ^τ. `iron_quantile_function` computes it exactly for functions that are linear on segments (pool-adjacent-violators with pools that may start or end inside a segment; each pool level is the root of a monotone equation, solved to machine precision). Pooled types have identical scores and tie. As a function of the bid, the score is linear with slope one where not ironed and constant on ironing intervals; a bid in a gap of the support takes the score approached from below, and a bid below the support is never allocated.

**Interim allocation.** For a participant with score s, let a = Pr[a rival participates with score > s] and z = Pr[a rival participates with score = s] (z > 0 only on atoms and ironing intervals). With A rivals above and Z tied, the participant wins with probability min(1, (k − A)/(Z + 1)) if A < k and s > 0, else 0. X(s) sums this over the binomial distribution of A and the conditional binomial distribution of Z; without ties X(s) = Pr[Bin(n − 1, a) ≤ k − 1], a polynomial of degree n − 1 in q.

**Exact integration.** Every expectation is an integral over segments of a polynomial of degree at most n, so Gauss–Legendre quadrature with n//2 + 2 nodes per segment is exact up to floating point:

    welfare             n ∫ ŵ^τ(q) X(q) dq                (valuation and externality parts likewise)
    R(b)                ∫_0^b X(t) dt                     (tabulated at the knots of X)
    revenue             n [∫ b(q) X(q) dq − ∫_0^∞ X(t) Pr(bid > t) dt]
    penalty paid (PA)   n Σ_j Pr[cell j, participate] r(e_j)
    penalty paid (RA)   n E[x (−e)] = −(externality welfare of the first best)
    participants        n P

**Penalties.** With the normalized DSIC payment (zero bid pays zero), the penalty inducing τ is

    r(e) = R(τ(e)) = ∫_0^{τ(e)} X^τ(b) db,

with τ clipped to [0, v̄]; an excluded externality (τ ≥ v̄, including +inf) gets R(v̄). `ThresholdEvaluation.penalty_tables[mechanism]["allocation_integral"]` is R as a callable; `penalty_from_table` evaluates r(e).

This replaces the old estimator R(e) = A(e)(τ(e) − P(e)) built from realized (k+1)-th admitted prices, which was biased upward (`WEB27/drafting/new_direction_proposals/audit-extensions.tex`, section "Three corrections").

**Welfare comparisons.** All comparisons are gains over unaudited VCG: `welfare_gains_over_vcg()` gives W_m − W_VCG and the share of the first-best gain, (W_m − W_VCG)/(W_FB − W_VCG); the summary table and the paired per-auction differences use the same baseline.

**Caching of τ-independent results.** Everyone participates in VCG and the first best, so their outcomes do not depend on τ. `tau_independent_outcomes` computes them once per (distribution, n, k) and stores them in `distribution.cache`; every τ evaluated on the same distribution object (all designs of a run with the same distribution and ζ, in the same worker process) reuses them. The bootstrap does the same per resample: resamples are determined by the seed, so VCG and first-best welfare of each resample are stored in `sample.cache` and reused for every τ bootstrapped with that seed (about 5× faster for the second and later τ). Transformed distributions (`with_externality_scale`, `dataclasses.replace`) start with an empty cache.

**Simulation.** Used only for per-auction welfare distributions and as a check of the exact values. Each draw is one auction; allocations are probabilities (the tie-breaking lottery is averaged out); payments are critical-value payments given the realized rival bids. All mechanisms run on the same draws (common random numbers). For piecewise-uniform cells, valuations are drawn uniformly within the cell.

**Revenue units.** For the first best, revenue is the critical payment in effective-valuation units (w = v + e); the transfer −e of winners is reported separately as penalty paid.

## Optimizing τ

### Threshold forms (`threshold_forms.py`)

| Label | τ(e) | Parameters | Optimizers |
|-------|------|------------|------------|
| `poly<d>` | Σ_m β_m e^m | d + 1 | genetic |
| `knots<K>` | continuous, piecewise linear between K knots at evenly spaced quantiles of the marginal externality distribution (levels (k − 1)/(K − 1)), constant beyond the end knots; exclusion cutoff ê: τ = v̄⁺ for e < ê | K + 1 | genetic, best response |
| `bins<K>` | constant on K bins with edges at the quantiles of levels j/K | K | genetic, best response |
| `nodes` | one value per externality node of the distribution (linear interpolation in between, for evaluation on other distributions) | number of nodes | best response |

The externality of every distribution takes finitely many values (grid nodes, or the distinct values of a population), so only τ at the nodes matters: `nodes` is the unrestricted problem, and the other forms restrict it. Values at or above v̄(1 + 10⁻⁹) exclude the externality; the threshold function returns +inf there, so excluded externalities stay excluded on a distribution with a larger v̄ (test rows). Knot values may lie outside [0, v̄]: they change the interpolated thresholds between knots. A record stores `threshold_spec` (form, knots / bin edges / nodes, parameters); `threshold_from_spec` rebuilds τ.

### Genetic search (`genetic_search.py`)

`run_genetic_search(distribution, degree, n, k, objective_mechanism="vcg_pa", threshold_form=None)` searches the parameters of a form (default the polynomial of degree `degree`) in two stages and returns a `SearchResult` (coefficients = parameters, welfare, trajectory, summary).

1. **Genetic stage (global).** The genetic algorithm of the original code: pygad, tournament selection, single-point crossover, Gaussian mutation scaled to each parameter's range, warm start, restarts in the pipeline. Search ranges: polynomials β_0 ∈ [v_min − sd(v)/2, v̄ + sd(v)/2], β_m ∈ ±3 sd(v)/E|e|^m; knot and bin values in the β_0 range; the cutoff ê between e_min − (e_max − e_min)/4 (no exclusion) and e_max. It stops at `num_generations` (500) or once the best welfare has not improved for `stall_generations` (60) generations.
2. **Nelder–Mead stage (local, default).** Nelder–Mead (scipy, adaptive) from the `num_local_starts` (5) best distinct candidates of the final population, always including the genetic optimum, in coordinates where each parameter's range is [0, 1] (initial simplex steps 5% of the range); tolerances 10⁻⁷ in those coordinates and 10⁻¹⁰ relative in welfare; at most `local_max_evaluations` (2000) evaluations per start. The refined τ is kept only if its welfare is higher.

The genetic stage locates a good region but converges slowly inside it; Nelder–Mead converges quickly from a good start but is local. Neither stage guarantees the global optimum of a non-convex objective. `local_refinement="none"` runs the genetic stage alone for all generations. The genetic search is practical only for few parameters (polynomials, knots, bins), not for `nodes`.

### Best-response ascent (`best_response_ascent.py`)

Theory (core draft v7, §5.1 and App. E). Hold the auction and the rivals' thresholds fixed: welfare is linear in one bidder's participation set, W = W(bidder out) + Σ_e g_e(τ(e)) with g_e(t) = E[1{v ≥ t, e_i = e} M^τ(e, v)] (Prop. 5.4), and the marginal contribution M^τ(e, v) = A^τ(v) + e X^τ(v) is affine in e (Def. 5.1). `marginal_contribution.py` computes X^τ and the rival welfare loss L^τ exactly for a score-based auction with k identical goods: the bidder displaces the rival with the k-th highest score S_(k) when it wins and S_(k) > 0, so L(s) = ∫_{0<S_(k)<s} ĉ dPr[S_(k)] + ĉ(s)(X(s) − Pr[S_(k) < s]), with ĉ = E[v + e | score] (pooled on ironing intervals). For VCG-PA, L^τ is the Clarke payment plus the displaced externality. Checks: Example N's M^τ(−c, t) = (3/4)t² − ct exactly; L^τ equals the Clarke payment without externalities; −n f M^τ equals finite differences of W to 7 digits for VCG-PA and IVCG-PA.

Each iteration:
1. builds the objective auction for the current τ and computes A^τ, X^τ on [0, v̄] as Chebyshev series between breakpoints (cell edges, segment ends, scoring-rule knots); for IVCG-PA the ironed scores of the current τ are held fixed (justified at a joint optimum, core draft §5.4; elsewhere an unproved envelope argument);
2. computes the best response of one bidder within the form: `nodes` and `bins` exactly (candidates: breakpoints and the zeros of M^τ where it changes sign from − to +, since g′ = −f M^τ; exclusion included), `knots` by dynamic programming over the knot values (the payoff is a sum of terms in adjacent knot pairs; the cutoff is a state of the recursion) on a grid of valuation quantiles plus values outside [0, v̄], refined twice around the optimum;
3. moves all bidders: τ ← τ + α(τ′ − τ) in parameter space, with α the first of 1, ½, …, 1/64 that raises the exact W.

It stops when no step raises W, when two consecutive relative gains are below `relative_tolerance` (10⁻¹⁰), or after `max_iterations` (100). A best response moves one bidder while the common τ moves all n, so the ascent can stop short of the best common τ (Thm 5.2: the best common τ satisfies the first-order condition but need not be a profile of best responses). **Gradient polish:** L-BFGS-B on W with the exact gradient ∂W/∂τ(e_j) = −n f_j(τ_j) M^τ(e_j, τ_j), chained to the form's parameters (knots: interpolation weights, cutoff fixed; bins: summed), then more ascent if the polish gained; at most `polish_rounds` (3) rounds; piecewise-uniform distributions only.

**Starts:** `zero` (τ ≡ 0), `no_competition` (τ(e) = max(0, −e), excluded if −e > v̄; first best without competition, Prop. 5.6), optional `short_genetic` (a 40-generation genetic search in the same form, `knots6` for `nodes`), plus warm starts: the genetic search's result (optimizer `genetic+best_response`) and, with `warm_start_across_zeta = true`, the optimum at the previous ζ rescaled to the new externality scale (τ(e) = τ_prev(e ζ_prev/ζ)). Each start runs to convergence; the best is kept. The genetic search and the ascent are independent: `best_response` needs no genetic run.

**Speed and accuracy** (151 × 150 grid, n = 8, k = 2): `nodes` converges in 0.1–1.1 s (VCG-PA and IVCG-PA, two starts), with interior marginal contribution below 3·10⁻⁵ and best-response gap below 10⁻¹¹; `knots6` in 1–3 s, reaching the Nelder–Mead optimum of the same form to 10⁻¹¹; Example N's optimum t* = 4c/3 exactly in every form. On the Community Notes data (241 nodes, ζ = 5, n = 8, k = 1) `nodes` takes 0.2 s.

**First-order check.** `first_order_check(distribution, τ, n, k, mechanism)` reports per externality node: τ and its status (interior, zero, excluded), X^τ(τ), the penalty r(e) = R(τ(e)), the own-allocation charge −e X^τ(τ), the displaced term L^τ(τ) − p(τ) (the displaced externality Δ^τ in VCG-PA), the marginal contribution M^τ(e, τ(e)) = r − (−e X^τ + displaced) and the best-response gap. At the optimum of the `nodes` form, interior thresholds have M^τ ≈ 0: the penalty equals the Pigouvian charge −e X^τ + L^τ − P^τ, which is −e X^τ + Δ^τ in VCG-PA (Thm 5.2, Rem. 5.3, Cor. E.2). For a restricted form the residuals measure how far the form is from the per-node optimum. Pipeline records store it for the objective mechanism on the training distribution (`first_order_check`).

### Pipeline

`pipeline.optimize_threshold(training, form, n, k, objective_mechanism, optimizer)` runs the search; `optimizer` is `auto` (genetic for polynomials, best response otherwise), `genetic`, `best_response`, or `genetic+best_response` (genetic search in `genetic_form`, default the same form or `knots6` for `nodes`, then the ascent with the genetic result as an extra start). `experiments/compute.py` adds the evaluation of each combination:

The objective is the **exact** expected welfare of the objective mechanism on the distribution (`expected_welfare`; one exact evaluation of all four mechanisms takes 10–16 ms on the default 151 × 150 grid), not an average over simulated auctions, so it has no Monte Carlo noise and does not fit a particular sample of auctions.

| Source | τ optimized on | τ evaluated on | Uncertainty reported |
|--------|----------------|----------------|----------------------|
| synthetic family | the family's distribution | the same distribution | none (H known); optional grid-refinement check |
| empirical CSV | training half (smoothed by default); IVCG-PA scoring rule designed there too | test half (atomic by default) | bootstrap over test rows (exact welfare per resample); optional bandwidth sensitivity |

and in both cases one simulation of `num_simulated_draws` auctions on the evaluation distribution for the per-auction welfare figures.

## Figures

`figures.py` draws the figures of a run from the files that `experiments/compute.py` saves; the list of figures, the folder layout and the checks are in the top-level README. Exact expected values are marked in every distribution figure; bootstrap percentile intervals over test rows (empirical data) give the uncertainty of the expected values, and simulated auctions (the same auctions for every mechanism) give the per-auction distributions. Paired differences from VCG are exactly zero in many auctions (same winners); the figures show the density of the nonzero differences scaled by their share and state the share of zeros.

Accounting identity used in the expected-outcomes figure, for every mechanism and auction: valuation part Σvx = bidder net utility Σu + revenue Σp + penalties Σr, and welfare W = Σvx + Σex. For the first best, revenue is the critical payment in effective-valuation units and the penalty is −e per unit of allocation, so the identity holds there too.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

`tests/test_audited_auctions.py`: Example T (ironing pools valuations 2 and 3; IVCG-PA welfare 40/16; first best 41/16; VCG-PA ≤ 39/16); the second-price penalty τ² (exact for uniform valuations); Example N welfare and penalty (exact); exact ironing of piecewise-linear ŵ^τ against ironing of a fine atomic discretization; ironing binds and helps in `harmful_middle`; exact expectations equal simulated means for all mechanisms and metrics (atomic and piecewise-uniform); a threshold type is indifferent (its simulated auction-utility equals r(e)); IVCG-PA ≥ VCG-PA; inverses of the piecewise-linear score; negative valuations are rejected.

`tests/test_pipeline.py`: every family has bounded non-negative continuous valuations; variance-preserving smoothing keeps the moments; the split is deterministic; bootstrap gains are paired; smoothed and atomic VCG welfare agree; the genetic search recovers the Example N optimum t* = 4c/3 (to 10⁻⁴ in τ and 10⁻¹⁰ in welfare with Nelder–Mead refinement); an IVCG-PA scoring rule designed on a distribution and applied to the same distribution reproduces its ironing exactly (welfare, revenue, penalties; atomic and piecewise uniform); cached VCG and first-best results equal fresh ones and are reused across τ and in the bootstrap.

`tests/test_experiments.py`: configuration errors name the problem; expansion runs every valid combination and skips invalid form–method pairs; the TOML run record re-reads to the same configuration; a synthetic combination writes all files, satisfies the accounting identity and passes its checks, and its `tau.json` rebuilds τ exactly; an empirical combination with bootstrap of every metric and bandwidth sensitivity; a whole optimize run with figures and README, then an evaluate run that reuses its optima (`from_run`) and reproduces their welfare.

`tests/test_best_response.py`: form labels, spec round trip and exclusion cutoff; projection keeps the excluded prefix; marginal contribution equals Example N's formula; the rival welfare loss equals the Clarke payment without externalities (n, k ∈ {(3,1), (8,2), (5,3)}); the gradient equals finite differences (VCG-PA, IVCG-PA); best-response ascent recovers Example N's optimum in the nodes, knots and bins forms; interior thresholds at the nodes optimum have zero marginal contribution and zero best-response gap; genetic+best_response ≥ genetic, nodes ≥ knots; optimizer–form combinations are validated.

## Conventions and limitations

- Only positive scores are allocated (VCG maximizes reported welfare, so a nonpositive bid never wins).
- Externalities of piecewise-uniform distributions are discrete (grid nodes); only valuations are continuous given the externality. τ is continuous in v (a threshold inside a cell admits the part of the cell above it exactly).
- The truncation bound v̄ of empirical data matters for the exclusion penalty R(v̄).
- A polynomial τ cannot represent exclusion (τ = v̄⁺) or jumps; knots represent exclusion of low externalities through the cutoff, bins and nodes represent both.
- The optimizers maximize VCG-PA welfare by default; IVCG-PA is then evaluated at that τ (`objective_mechanism="ivcg_pa"` optimizes for IVCG-PA instead; the ascent then holds the ironed scores of the current τ fixed in each step, an envelope argument that is not proved in general; the line search on exact W* protects each step).
- Best-response ascent and the polish find a local optimum; several starts reduce the risk.
