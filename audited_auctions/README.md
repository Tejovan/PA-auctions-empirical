# `audited_auctions` — exact computation, simulation and τ optimization for a common threshold

Given a joint distribution H of bidder types (e, v) and a common participation threshold function τ(e), this package computes the outcomes of four mechanisms with n exchangeable bidders and k identical goods (unit demand):

| Key | Mechanism | Participants | Ranking score | Penalty |
|-----|-----------|--------------|---------------|---------|
| `vcg` | VCG without audits | all bidders | bid v | none |
| `vcg_pa` | VCG-PA | types with v ≥ τ(e) | bid v | r(e) inducing τ, charged to participants |
| `ivcg_pa` | IVCG-PA | types with v ≥ τ(e) | ironed welfare contribution ω̄^τ(v) | r(e) inducing τ, charged to participants |
| `first_best` | optimal recipient audit | all bidders | w = v + e | −e, charged to winners |

All four are score-based auctions: the k participants with the highest positive scores win, ties are broken uniformly at random, and payments are critical-value (Myerson) payments, so each auction is DSIC. The notation follows the paper (`WEB27/arxiv_WEB/arxiv_WEB_v1.tex` of the paper repository); citations give the v1 number and the stable LaTeX label in brackets.

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
| `type_distributions.py` | `TypeDistribution` (cells: an externality, a valuation interval, a probability); populations (atomic), gridded densities (piecewise uniform), discrete externality levels; continuous versions of gridded distributions (`continuous_version`) and their discrete approximation (`refined_version`); the synthetic families (`SYNTHETIC_FAMILIES`, `synthetic_distribution`). |
| `smoothing.py` | `kernel_density_distribution`: kernel-smoothed piecewise-uniform distribution of a population. |
| `empirical.py` | Empirical pipeline: load a CSV, train/test split, truncation, smoothed or atomic distribution, bootstrap of exact welfare. |
| `threshold_functions.py` | τ(e) from polynomial coefficients, tabulated values, or a constant. τ = +inf excludes an externality (the threshold v̄⁺ of the paper). `threshold_kinks`: the externalities where τ has kinks or jumps (used by the continuous evaluation). |
| `scoring_rules.py` | `IdentityScore` (rank by bid) and `PiecewiseLinearScore` (rank by the ironed welfare contribution, a nondecreasing function of the bid), with the generalized inverses needed for payments. |
| `exact_computation.py` | Quantile representation of a mechanism, exact ironing, interim allocation with ties, exact integration of welfare, revenue and the allocation integral R(b) (the penalty); `continuous_bid_segments` for continuous versions. |
| `auction_simulation.py` | Vectorized Monte Carlo of a score-based auction on sampled type profiles. |
| `threshold_evaluation.py` | `evaluate_threshold` (main entry point), `ThresholdEvaluation`, `expected_welfare` (optimization objective), `ivcg_scoring_rule_for` (IVCG-PA design to apply elsewhere), `penalty_from_table`, `grid_refinement_check`. |
| `threshold_forms.py` | Forms of τ searched by the optimizers: `poly<d>`, `knots<K>` (+ exclusion cutoff), `bins<K>`, `cutoff`, `nodes`; `build_threshold_form`, `threshold_from_spec`; `ex_ante_cutoff`, `ex_ante_guarantee` (paper Thm 8.9 [thm:k-goods]). |
| `genetic_search.py` | Genetic search over a form's parameters with the exact welfare objective: genetic stage, then Nelder–Mead refinement (default). |
| `marginal_contribution.py` | Exact marginal contribution M^τ(e, v) = (v + e) X^τ(v) − L^τ(v) of a participating type for a fixed auction (rival welfare loss L^τ), as Chebyshev series between breakpoints. |
| `best_response_ascent.py` | Best-response ascent over knots, bins or nodes, with an exact-gradient polish; `first_order_check` (zero marginal contribution and Pigouvian penalty at each threshold); `participation_gap` (largest welfare gain from one bidder changing its participation set: zero exactly when participants have M ≥ 0 and abstainers M ≤ 0, the sufficient condition for VCG-PA). |
| `symmetric_relaxation.py` | Best symmetric participation rule for VCG-PA with k goods, a concave program solved as a linear program (tangent cuts): an estimate of an upper bound on VCG-PA welfare over all common τ, and the thresholds read from its solution (the `relaxation` start of best-response ascent and the `relaxation` optimizer method). |
| `pipeline.py` | Building blocks of a combination: `prepare_distributions` (training, design and evaluation distributions), `optimize_threshold` (search in a form with any method), `designed_ivcg_rule`, `bandwidth_sensitivity`. |
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

### Continuous versions

A piecewise-uniform grid discretizes a continuous density with two kinds of jumps: each externality node is a point mass in e, and the valuation density jumps at cell edges. Under a continuous τ, a whole node enters the participants at its threshold, and E[e | v] jumps wherever the mix of externalities changes, so ŵ^τ has small drops at every node threshold and every cell edge, and IVCG-PA irons in many small bands. `distribution.continuous_version()` removes both jumps:

- **externalities:** each node's mass is uniform on its externality interval, from the midpoint to the previous node to the midpoint to the next (the end nodes get half a spacing outside); cell attributes `externality_lower`, `externality_upper`;
- **valuations:** within each node's column of cells, the density is the piecewise-linear interpolation of the cell densities between the cell midpoints, constant from the outer midpoints to the ends of a run of adjacent cells; cell attributes `valuation_shape_lower`, `valuation_shape_upper` (density at the cell ends relative to the midpoint); cell masses are renormalized.

The continuous version differs from the grid by an error of second order in the spacing, the order of the grid's own error. Distributions with finitely many externality values (`discrete_externality_distribution`, e.g. `example_n`) keep their externality values; atomic distributions have no continuous version. `sample_types` draws from the continuous density. `distribution.refined_version(m, r)` is a discrete approximation: m point externalities per interval × r uniform valuation pieces per cell (used for the first best and for the participation gap).

### Synthetic families

All families have v ∈ [0, v̄]. Grid options: `num_externality_nodes` (default 151) and `num_valuation_cells` (default 150); scripts: `--num-externality-nodes`, `--num-valuation-cells`. The two grids have different errors: coarser externality nodes pool nearby externalities (error about proportional to the spacing) and fix how finely a per-node τ can vary; coarser valuation cells make the conditional density of v a coarser step function (error about proportional to the squared spacing; zero when v given e is already piecewise uniform, as in `independent`), and τ stays continuous in v. Accuracy at a fixed τ (n = 8, k = 2), maximum change of the welfare gains over VCG against 601 × 1200: on `harm_rises`, `d`, `independent` at most 1.1·10⁻⁴ at 151 × 150 (10–16 ms per evaluation of the four mechanisms), 2.2·10⁻⁴ at 101 × 200 and 101 × 100, 1.7·10⁻⁴ at 201 × 100, 7.4·10⁻⁵ at 201 × 150 (gains 0.06–0.5); on `harmful_middle` with welfare cutoffs w ∈ {0, 0.4}, 8.4·10⁻⁵ at 151 × 150 (about 50 ms), 7.9·10⁻⁵ at 101 × 200, 1.9·10⁻⁴ at 101 × 100, 2.1·10⁻⁴ at 201 × 100, 1.1·10⁻⁴ at 201 × 150 (gains 0.12–1.18). `synthetic_distribution(name, externality_scale=ζ, **options)` builds one; family parameters are keyword options.

| Name | Description | Why it is useful |
|------|-------------|------------------|
| `a` | single bivariate normal, mean (e, v) = (0, 1), variances 0.1 | baseline, no correlation |
| `b` | two normals side by side, means (−1, 1), (0, 1), variances 0.05 | harmful and benign groups with equal valuations |
| `c` | two normals on a diagonal, means (−1, 1), (0, 2) | harmful types have low valuations (VCG already does well) |
| `d` | two normals on the other diagonal, means (−1, 2), (0, 1) | harmful types have high valuations (VCG does badly) |
| `e` | two normals stacked, means (0, 2), (0, 1) | no harm; audits should not matter |
| `f` | four normals at (−1, 1), (−1, 2), (0, 2), (0, 1) | mixture of the above |
| `h` | single normal, mean (0, 2), variances 0.1 | shifted baseline |
| `example_n` | Example N of the paper (Ex. 7.11 [ex:N]): v ~ U[0, 1], e ∈ {0, −c} with probability ½ each (option `harm` = c, default 0.6) | analytic benchmark: with n = 2, k = 1, VCG-PA welfare 2/3 − c/2 + (c/2)t² − t³/4 at τ(−c) = t, optimum t* = 4c/3, penalty (3/4)t² |
| `independent` | v ~ U[0, 1] independent of e ~ N(−0.3, 0.5²) | independence case of the open question on optimal τ for VCG-PA |
| `harm_rises` | bivariate normal, means (−0.3, 1), sds (0.5, 0.4), correlation −0.6 (option `correlation`) | harm grows with valuation (misinformation-style) |
| `harmful_middle` | v ~ N(1, 0.45²) truncated to v ≥ 0; given v, e ~ N(m(v), 0.4²) with m(v) = −0.3 − 0.9 (v − 1) − 0.5 x e^{(1 − x²)/2}, x = (v − 1)/0.15: a steep ridge like `harm_rises` with a wiggle at v = 1 (options `valuation_mean`, `valuation_sd`, `externality_mean`, `slope`, `externality_sd`, `wiggle_amplitude`, `wiggle_width`, `wiggle_center`) | m(v) falls with slope −0.9 − 0.5·e^{1/2}/0.15 ≈ −6.4 at the wiggle, so ŵ^τ decreases there, IVCG-PA irons and beats VCG-PA, also at optimized τ (a continuous analogue of Example T, Ex. 6.4 [ex:T-pool]) |
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

**Continuous versions** (`continuous_bid_segments`; VCG, VCG-PA, IVCG-PA). τ is evaluated at the ends and midpoint of each cell's externality interval and at its kinks and jumps there (`threshold_kinks`: knots, nodes, bin edges, the exclusion cutoff; for `cutoff`, e = w and e = w − v̄), and is linear in between: exact for the piecewise-linear forms. Each cell is cut into pieces (externality sub-interval × valuation half) with a density uniform in e and linear in v. At bid b, the participating externalities of a piece are an interval whose ends are piecewise linear in b, so the participating density is piecewise quadratic in b and E[e | bid = b] is a ratio of polynomials. The bid line is cut at cell edges and midpoints, at the values of τ at the interval points, and once more inside each range where τ crosses an interval. Each segment gets its exact participating mass (Simpson's rule) with a constant bid density, and E[e | bid] is evaluated exactly at the segment ends (at an end where no type participates, as the limit of the entering externalities) and interpolated linearly in between. ŵ^τ is then continuous in the bid, and IVCG-PA irons only where ŵ^τ decreases. Accuracy: second order in the segment widths. An evaluation takes about 5× as long as on the grid (0.07 s vs 0.015 s for the four mechanisms at 161 × 100, n = 4). The first best (bids v + e) uses `refined_version(4, 2)`. Penalties paid: n Σ over participating pieces of mass × r(τ(mean externality of the piece)).

**Ironing (IVCG-PA).** ω̄^τ is the nondecreasing L2 projection of ŵ^τ in quantile space, the derivative of the lower convex hull of Ψ(q) = ∫_0^q ŵ^τ. `iron_quantile_function` computes it exactly for functions that are linear on segments (pool-adjacent-violators with pools that may start or end inside a segment; each pool level is the root of a monotone equation, solved to machine precision). A segment on which ŵ^τ decreases (continuous versions only) is first replaced by its mean: the projection is constant on it, so this leaves the projection unchanged. Pooled types have identical scores and tie. As a function of the bid, the score is linear with slope one where not ironed and constant on ironing intervals; a bid in a gap of the support takes the score approached from below, and a bid below the support is never allocated.

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

**Simulation.** Used only for per-auction welfare distributions and as a check of the exact values. Each draw is one auction; allocations are probabilities (the tie-breaking lottery is averaged out); payments are critical-value payments given the realized rival bids. All mechanisms run on the same draws (common random numbers). For piecewise-uniform cells, valuations are drawn uniformly within the cell; for continuous versions, from the cell's piecewise-linear density, with the externality uniform on the cell's interval.

**Revenue units.** For the first best, revenue is the critical payment in effective-valuation units (w = v + e); the transfer −e of winners is reported separately as penalty paid.

## Optimizing τ

### Threshold forms (`threshold_forms.py`)

| Label | τ(e) | Parameters | Optimizers |
|-------|------|------------|------------|
| `poly<d>` | Σ_m β_m e^m | d + 1 | genetic |
| `knots<K>` | continuous, piecewise linear between K knots at evenly spaced quantiles of the marginal externality distribution (levels (k − 1)/(K − 1)), constant beyond the end knots; exclusion cutoff ê: τ = v̄⁺ for e < ê | K + 1 | genetic, best response |
| `bins<K>` | constant on K bins with edges at the quantiles of levels j/K | K | genetic, best response |
| `cutoff` | common welfare cutoff w: max(0, w − e), excluded if w − e > v̄; w = 0 is the zero-welfare threshold τ⁰ (§7.2 [sec:restrictions]) | 1 | genetic |
| `nodes` | one value per externality node of the distribution (linear interpolation in between, for evaluation on other distributions) | number of nodes | best response |

The externality of every distribution takes finitely many values (grid nodes, or the distinct values of a population), so only τ at the nodes matters: `nodes` is the unrestricted problem, and the other forms restrict it. Values at or above v̄(1 + 10⁻⁹) exclude the externality; the threshold function returns +inf there, so excluded externalities stay excluded on a distribution with a larger v̄ (test rows). Knot values may lie outside [0, v̄]: they change the interpolated thresholds between knots. A record stores `threshold_spec` (form, knots / bin edges / nodes, parameters); `threshold_from_spec` rebuilds τ.

### Genetic search (`genetic_search.py`)

`run_genetic_search(distribution, degree, n, k, objective_mechanism="vcg_pa", threshold_form=None)` searches the parameters of a form (default the polynomial of degree `degree`) in two stages and returns a `SearchResult` (coefficients = parameters, welfare, trajectory, summary).

1. **Genetic stage (global).** The genetic algorithm of the original code: pygad, tournament selection, single-point crossover, Gaussian mutation scaled to each parameter's range, warm start; `num_restarts` (4) independent restarts with seeds `search`, `search + 1`, …, run simultaneously. Search ranges: polynomials β_0 ∈ [v_min − sd(v)/2, v̄ + sd(v)/2], β_m ∈ ±3 sd(v)/E|e|^m; knot and bin values in the β_0 range; the cutoff ê between e_min − (e_max − e_min)/4 (no exclusion) and e_max. It stops at `num_generations` (500) or once the best welfare has risen by at most `stall_tolerance` (10⁻⁴ of the first-best gain FB − W_VCG) over `stall_generations` (60) generations.
2. **Nelder–Mead stage (local, default).** Nelder–Mead (scipy, adaptive) from the `num_local_starts` (5) best distinct candidates of the final population, always including the genetic optimum, in coordinates where each parameter's range is [0, 1] (initial simplex steps 5% of the range); tolerances 10⁻⁷ in those coordinates and 10⁻¹⁰ relative in welfare; at most `local_max_evaluations` (2000) evaluations per start. The refined τ is kept only if its welfare is higher.

The genetic stage locates a good region but converges slowly inside it; Nelder–Mead converges quickly from a good start but is local. Neither stage guarantees the global optimum of a non-convex objective. `local_refinement="none"` runs the genetic stage alone for all generations. The genetic search is practical only for few parameters (polynomials, knots, bins), not for `nodes`.

### Best-response ascent (`best_response_ascent.py`)

Theory (paper §7.1 [sec:choosing], §7.3 and App. E). Hold the auction and the rivals' thresholds fixed: welfare is linear in one bidder's participation set, W = W(bidder out) + Σ_e g_e(τ(e)) with g_e(t) = E[1{v ≥ t, e_i = e} M^τ(e, v)] (Prop. 7.4 [prop:decomposition]), and the marginal contribution M^τ(e, v) = A^τ(v) + e X^τ(v) is affine in e (Def. 7.1 [def:marginal-contribution]). `marginal_contribution.py` computes X^τ and the rival welfare loss L^τ exactly for a score-based auction with k identical goods: the bidder displaces the rival with the k-th highest score S_(k) when it wins and S_(k) > 0, so L(s) = ∫_{0<S_(k)<s} ĉ dPr[S_(k)] + ĉ(s)(X(s) − Pr[S_(k) < s]), with ĉ = E[v + e | score] (pooled on ironing intervals). For VCG-PA, L^τ is the Clarke payment plus the displaced externality. Checks: Example N's M^τ(−c, t) = (3/4)t² − ct exactly; L^τ equals the Clarke payment without externalities; −n f M^τ equals finite differences of W to 7 digits for VCG-PA and IVCG-PA.

Each iteration:
1. builds the objective auction for the current τ and computes A^τ, X^τ on [0, v̄] as Chebyshev series between breakpoints (cell edges, segment ends, scoring-rule knots); for IVCG-PA the ironed scores of the current τ are held fixed (justified at a joint optimum, §7.6 [sec:joint]; elsewhere an unproved envelope argument);
2. computes the best response of one bidder within the form: `nodes` and `bins` exactly (candidates: breakpoints and the zeros of M^τ where it changes sign from − to +, since g′ = −f M^τ; exclusion included), `knots` by dynamic programming over the knot values (the payoff is a sum of terms in adjacent knot pairs; the cutoff is a state of the recursion) on a grid of valuation quantiles plus values outside [0, v̄], refined twice around the optimum;
3. moves all bidders: τ ← τ + α(τ′ − τ) in parameter space, with α the first of 1, ½, …, 1/64 that raises the exact W.

It stops when no step raises W, when two consecutive relative gains are below `relative_tolerance` (10⁻¹⁰), or after `max_iterations` (100). A best response moves one bidder while the common τ moves all n, so the ascent can stop short of the best common τ (Prop. 7.9 [prop:zero-marginal-equal]: the best common τ satisfies the first-order condition but need not be a profile of best responses). **Gradient polish:** L-BFGS-B on W with the exact gradient ∂W/∂τ(e_j) = −n f_j(τ_j) M^τ(e_j, τ_j), chained to the form's parameters (knots: interpolation weights, cutoff fixed; bins: summed), then more ascent if the polish gained; at most `polish_rounds` (3) rounds; piecewise-uniform distributions only.

**Starts:** `zero` (τ ≡ 0), `zero_welfare` (the zero-welfare threshold τ⁰(e) = max(0, −e), excluded if −e > v̄, §7.2 [sec:restrictions]; first best without competition, Prop. 7.10 [prop:no-competition]; alias `no_competition`), `short_genetic` (a 40-generation genetic search in the same form, `knots6` for `nodes`), `relaxation` (the thresholds of the best symmetric participation rule for VCG-PA, below; skipped for atomic distributions and k ≥ n), `num_random_starts` (3) random starts (τ piecewise linear through 6 externalities spread over the nodes, values uniform on [0, 95% quantile of v]), plus warm starts: the genetic search's result (optimizer `genetic+best_response`) and, with `warm_start_across_zeta = true`, the optimum at the previous ζ rescaled to the new externality scale (τ(e) = τ_prev(e ζ_prev/ζ)). Each start runs to convergence, simultaneously in separate processes; the best is kept. The genetic search and the ascent are independent: `best_response` needs no genetic run.

**Speed and accuracy** (151 × 150 grid, n = 8, k = 2): `nodes` converges in 0.1–1.1 s (VCG-PA and IVCG-PA, two starts), with interior marginal contribution below 3·10⁻⁵ and best-response gap below 10⁻¹¹; `knots6` in 1–3 s, reaching the Nelder–Mead optimum of the same form to 10⁻¹¹; Example N's optimum t* = 4c/3 exactly in every form. On the Community Notes data (241 nodes, ζ = 5, n = 8, k = 1) `nodes` takes 0.2 s.

**Parallel stages.** `optimize_threshold(..., workers=W)`: the genetic restarts run simultaneously (each with the workers left over for its Nelder–Mead starts), then the best-response starts; `bootstrap_welfare(..., workers=W)` splits the resamples into contiguous chunks. Results equal the serial ones (test `test_inner_parallelism_matches_serial`). In a run, `runner` uses one process per combination and gives each combination workers // (number of combinations) for these stages, so a run with few, large combinations (on the SCC: one combination, 16–28 slots) still uses every core. Process start-up costs about 1 s per parallel stage.

**Time limit and stop reasons.** `optimize_threshold(..., time_limit=seconds)` (configuration `time_limit_minutes`, default 60 per combination) sets one deadline for the whole search: the genetic stage stops after its current generation, Nelder–Mead and L-BFGS-B after their current iteration, the ascent before its next iteration; later restarts and starts are skipped; the best τ so far is returned. Every stage records `stop_reason`: genetic `stalled`, `max_generations`, `time_limit`; Nelder–Mead `converged`, `max_evaluations`, `time_limit`; ascent `no_improving_step`, `small_gain`, `max_iterations`, `time_limit` (`skipped (time limit)` for starts not begun).

**Per-iteration diagnostics** (`search.history` in results.json; figure `search/iterations.pdf`, one line per start). All stages: welfare. Best-response ascent, at the τ of each iteration: *stationarity* v̄ Σ_j |∂W/∂θ_j| / (FB − W_VCG), the projected gradient in the form's parameters θ (at τ(e) = 0 and at exclusion only the one-sided derivative into [0, v̄] that raises W counts); *first-order violation* max over nodes of |M(e, τ(e))| at interior thresholds, max(0, −M(e, 0)) at τ = 0, max(0, M(e, v̄)) at exclusion, / v̄; *best-response gap* / (FB − W_VCG). τ ≡ 0 meets both first-order conditions (a bidder with v = 0 never wins, so M(e, 0) = 0) without being optimal, and an optimal common τ need not be a best response, so none of the three is a stopping rule; the L-BFGS-B polish stops on its own projected gradient (`gtol`).

**First-order check.** `first_order_check(distribution, τ, n, k, mechanism)` reports per externality node: τ and its status (interior, zero, excluded), X^τ(τ), the penalty r(e) = R(τ(e)), the own-allocation charge −e X^τ(τ), the displaced term L^τ(τ) − p(τ) (the displaced externality Δ^τ in VCG-PA), the marginal contribution M^τ(e, τ(e)) = r − (−e X^τ + displaced) and the best-response gap. At the optimum of the `nodes` form, interior thresholds have M^τ ≈ 0: the penalty equals the Pigouvian charge −e X^τ + L^τ − P^τ, which is −e X^τ + Δ^τ in VCG-PA (Prop. 7.9 [prop:zero-marginal-equal], Thm 7.2 [thm:zero-marginal], Rem. 7.3 [rem:pigouvian], Cor. E.2 [cor:pigouvian]). For a restricted form the residuals measure how far the form is from the per-node optimum. Pipeline records store it for the objective mechanism on the training distribution (`first_order_check`).

### Symmetric relaxation (`symmetric_relaxation.py`)

A symmetric participation rule lets each type participate with a probability α(e, v), the same for all bidders; a common τ is the rule 1{v ≥ τ(e)}. For VCG-PA with k goods, expected welfare over all such rules is a concave program after the change of variables y(v) = Φ(Γ(v)) (paper v3, proof of the sufficiency theorem), and the best rule bounds the welfare of every common τ from above. On valuation intervals (cell edges, each split; 600 in total by default) it is a linear program (HiGHS): tangent cuts for the concave allocation term and the perspective of the externality term. It is a midpoint rule, so the value is an estimate with error of order (interval width)², not a certified bound; about 5 s per distribution, n, k. τ read off the solution: each node's threshold is the first valuation at which at least half of the node participates. If every node's participating fraction is nondecreasing in v (`is_threshold_rule`), the best symmetric rule is a threshold rule and that τ is a best common τ for VCG-PA up to the discretization.

Uses: the `relaxation` start of best-response ascent (one start among several; the best is kept), the `relaxation` optimizer method (that start alone, followed by best-response ascent; `optimize_threshold(..., optimizer="relaxation")`, which records the program's value, `threshold_rule`, `largest_drop` and seconds under `summary["relaxation"]`), and the `symmetric_relaxation` check (share of the relaxation's gain over VCG that τ attains on the training grid). The relaxation is solved on the grid (piecewise-uniform training distribution). It is not faster than best-response ascent (5 s against 0.1–1.1 s per start on the `nodes` form); its use is that it is a convex program, so its solution does not depend on a start, while the ascent is local. Its τ is a best common τ for VCG-PA only where its solution is a threshold rule; elsewhere the ascent improves it. For the IVCG-PA objective it is a start only.

### Pipeline

`pipeline.optimize_threshold(training, form, n, k, objective_mechanism, optimizer)` runs the search; `optimizer` is `auto` (genetic for polynomials, best response otherwise), `genetic`, `best_response`, `genetic+best_response` (genetic search in `genetic_form`, default the same form or `knots6` for `nodes`, then the ascent with the genetic result as an extra start), or `relaxation` (above). `experiments/compute.py` adds the evaluation of each combination:

The objective is the **exact** expected welfare of the objective mechanism on the distribution (`expected_welfare`; one exact evaluation of all four mechanisms takes 10–16 ms on the default 151 × 150 grid), not an average over simulated auctions, so it has no Monte Carlo noise and does not fit a particular sample of auctions.

| Source | τ optimized on | τ evaluated on | Uncertainty reported |
|--------|----------------|----------------|----------------------|
| synthetic family | the family's grid | its continuous version (`continuous = true`, default), else the grid | none (H known); optional grid-refinement check |
| empirical CSV | training half (smoothed by default); IVCG-PA scoring rule designed on its continuous version (`continuous = true`) | test half (atomic by default; `evaluation = "smoothed"`: the continuous version of the smoothed test half) | bootstrap over test rows (exact welfare per resample); optional bandwidth sensitivity |

`PreparedDistributions.design` is the distribution of the IVCG-PA scoring rule, the ironing figure, the marginal-contribution maps and the participation gap (the continuous version of the training distribution when `continuous`); the first-order check and the relaxation use the training grid, where the search runs.

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

`tests/test_experiments.py`: configuration errors name the problem (unknown keys, a form the optimizer cannot search, forms in `[design]`); expansion runs every optimizer with each of its forms and skips the relaxation where k ≥ n; the TOML run record re-reads to the same configuration; a synthetic combination writes all files, satisfies the accounting identity and passes its checks, and its `tau.json` rebuilds τ exactly; an empirical combination with bootstrap of every metric and bandwidth sensitivity; a whole optimize run with figures and README, then an evaluate run that reuses its optima (`from_run`) and reproduces their welfare.

`tests/test_best_response.py`: form labels, spec round trip and exclusion cutoff; projection keeps the excluded prefix; marginal contribution equals Example N's formula; the rival welfare loss equals the Clarke payment without externalities (n, k ∈ {(3,1), (8,2), (5,3)}); the gradient equals finite differences (VCG-PA, IVCG-PA); best-response ascent recovers Example N's optimum in the nodes, knots and bins forms; interior thresholds at the nodes optimum have zero marginal contribution and zero best-response gap; genetic+best_response ≥ genetic, nodes ≥ knots; optimizer–form combinations are validated.

`tests/test_continuous.py`: the continuous version keeps total mass and samples continuous externalities; with independent externalities and a welfare cutoff, IVCG-PA = VCG-PA on the continuous version (to 10⁻⁸) while the grid irons, and the two agree in level to the grid's error; exact outcomes of the continuous version equal simulated means and satisfy the accounting identity; ironing pools a decreasing segment to its mean; the relaxation method reports the bound and reaches it on Example N.

## Conventions and limitations

- Only positive scores are allocated (VCG maximizes reported welfare, so a nonpositive bid never wins).
- Externalities of piecewise-uniform distributions are discrete (grid nodes); only valuations are continuous given the externality. τ is continuous in v (a threshold inside a cell admits the part of the cell above it exactly). Continuous versions make both continuous; the search still runs on the grid, so τ is optimal for the grid, and its welfare on the continuous version differs by the discretization error.
- The truncation bound v̄ of empirical data matters for the exclusion penalty R(v̄).
- A polynomial τ cannot represent exclusion (τ = v̄⁺) or jumps; knots represent exclusion of low externalities through the cutoff, bins and nodes represent both.
- The optimizers maximize VCG-PA welfare by default; IVCG-PA is then evaluated at that τ (`objective_mechanism="ivcg_pa"` optimizes for IVCG-PA instead; the ascent then holds the ironed scores of the current τ fixed in each step, an envelope argument that is not proved in general; the line search on exact W* protects each step).
- Best-response ascent and the polish find a local optimum; several starts reduce the risk.
