# `audited_auctions` — exact computation, simulation and τ optimization for a common threshold

Given a joint distribution H of bidder types (e, v) and a common participation threshold function τ(e), this package computes the outcomes of four mechanisms with n exchangeable bidders and k identical goods (unit demand):

| Key | Mechanism | Participants | Ranking score | Penalty |
|-----|-----------|--------------|---------------|---------|
| `vcg` | VCG without audits | all bidders | bid v | none |
| `vcg_pa` | VCG-PA | types with v ≥ τ(e) | bid v | r(e) inducing τ, charged to participants |
| `ivcg_pa` | IVCG-PA | types with v ≥ τ(e) | ironed welfare contribution ω̄^τ(v) | r(e) inducing τ, charged to participants |
| `first_best` | optimal recipient audit | all bidders | w = v + e | −e, charged to winners |

All four are score-based auctions: the k participants with the highest positive scores win, ties are broken uniformly at random, and payments are critical-value (Myerson) payments, so each auction is DSIC. The notation follows the WEB27 core draft (`WEB27/drafting/web27_aa_1p_core/`).

The penalties are outputs, not inputs: since τ is given, the package computes the penalty function that induces τ in each participant-audit auction. It also searches for the polynomial τ that maximizes exact expected welfare (genetic algorithm), and runs the full synthetic and empirical pipelines (`pipeline.py`).

## Quick start

```python
from audited_auctions import evaluate_threshold, synthetic_distribution, polynomial_threshold
from audited_auctions.figures import plot_threshold_and_penalties, plot_welfare_distributions

distribution = synthetic_distribution("harm_rises")      # piecewise-uniform, v ∈ [0, v̄]
threshold = polynomial_threshold([0.5, -1.0])            # τ(e) = 0.5 − e
evaluation = evaluate_threshold(distribution, threshold, num_bidders=8, num_goods=2,
                                num_simulated_draws=20000, random_seed=0)
print(evaluation.summary_table())          # exact outcomes, gains over VCG, simulated means
print(evaluation.welfare_gains_over_vcg()) # W_m − W_VCG and share of the first-best gain
plot_threshold_and_penalties(evaluation, distribution, "threshold_penalties.png")
plot_welfare_distributions(evaluation, "welfare_distributions.png")
```

Command line: `python evaluate_threshold.py --distribution harm_rises --tau-coefficients 0.5,-1 --num-bidders 8 --num-goods 2` (one τ), `python Collateralized_Auction_genetic2.py ...` (optimize one cell), `python run_sweep_genetic2.py ...` (sweeps); see the top-level README.

## Modules

| Module | Contents |
|--------|----------|
| `type_distributions.py` | `TypeDistribution` (cells: an externality, a valuation interval, a probability); populations (atomic), gridded densities (piecewise uniform), discrete externality levels; the synthetic families (`SYNTHETIC_FAMILIES`, `synthetic_distribution`). |
| `smoothing.py` | `kernel_density_distribution`: kernel-smoothed piecewise-uniform distribution of a population. |
| `empirical.py` | Empirical pipeline: load and truncate a CSV, train/test split, smoothed or atomic distribution, bootstrap of exact welfare. |
| `threshold_functions.py` | τ(e) from polynomial coefficients, tabulated values, or a constant. τ = +inf excludes an externality (the threshold v̄⁺ of the paper). |
| `scoring_rules.py` | `IdentityScore` (rank by bid) and `PiecewiseLinearScore` (rank by the ironed welfare contribution, a nondecreasing function of the bid), with the generalized inverses needed for payments. |
| `exact_computation.py` | Quantile representation of a mechanism, exact ironing, interim allocation with ties, exact integration of welfare, revenue and the allocation integral R(b) (the penalty). |
| `auction_simulation.py` | Vectorized Monte Carlo of a score-based auction on sampled type profiles. |
| `threshold_evaluation.py` | `evaluate_threshold` (main entry point), `ThresholdEvaluation`, `expected_welfare` (optimization objective), `penalty_from_table`, `grid_refinement_check`. |
| `genetic_search.py` | Genetic algorithm over polynomial τ with the exact welfare objective. |
| `pipeline.py` | One optimization-and-evaluation cell: `prepare_distributions`, `optimize_and_evaluate`. |
| `figures.py` | `plot_threshold_and_penalties`, `plot_welfare_distributions`. |
| `sweep_support.py` | Glue for `run_sweep_genetic2.py` and `post_optimization_analysis.py`. |

## Type distributions

A `TypeDistribution` is a list of cells j with externality e_j, valuation interval [lower_j, upper_j] and probability p_j; given the cell, the valuation is uniform on the interval. Valuations must be non-negative and bounded (v̄ = largest upper end), as in the model; constructing a distribution with a negative valuation raises an error. Two kinds:

- **Atomic** (lower = upper): a population used as is. Bidders are i.i.d. draws of its rows. Ties between equal valuations have positive probability.
- **Piecewise uniform** (lower < upper): the externality takes the values of a grid of nodes and, given the externality, the valuation has a piecewise-constant density. Valuations have no atoms, so ties between valuation bids have probability zero, and every quantity below is computed exactly *for this distribution*. Relative to a continuous density it approximates, the error shrinks with the grid; `grid_refinement_check` measures it by recomputing on a twice finer grid.

Mixed distributions are not supported. `distribution.with_externality_scale(ζ)` multiplies externalities by ζ.

### Synthetic families

All families have v ∈ [0, v̄]. Grid options: `num_externality_nodes` (default 301) and `num_valuation_cells` (default 600). `synthetic_distribution(name, externality_scale=ζ, **options)` builds one; family parameters are keyword options.

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

The mixtures a–f and h are those of `alternate_distributions.ipynb` with valuation means shifted up by 1 (h unchanged), truncated to v ≥ 0 and to ±5 standard deviations, and renormalized. The shift keeps the shapes while truncation at v = 0 removes almost no mass; the CSV samples in `data/samples/` are the old unshifted versions with negative valuations and are not used by the pipeline.

### Empirical populations

`empirical.py` (used by the pipeline when the source is a CSV with columns `e_score`, `v_score`):

1. **Scale and truncate** (`load_empirical_sample`): e = ζ e_score, v = action_cost × v_score; rows with v ≤ 0 or v above the `truncation_quantile` (default 0.995) of v are dropped, and that quantile is the bound v̄. On the Community Notes data this drops 60 of 11,811 rows.
2. **Split** (`split_sample`): a random train/test split (default ½/½) with a seed common to all sweep cells.
3. **Distribution** (`sample_distribution`): kernel-smoothed (default) or atomic.
4. **Bootstrap** (`bootstrap_welfare`): resample the test rows with replacement, rebuild the distribution the same way (re-smoothing with the bandwidth re-chosen on the resample), and recompute exact welfare of all four mechanisms. Reports per-resample welfare, paired gains over VCG and shares of the first-best gain, and percentile intervals. v̄ stays fixed. About 40 ms per resample on the Community Notes test half.

**Kernel smoothing** (`smoothing.py`). Coordinates u = asinh(e/s_e) (s_e = median |e|) and y = log v; Gaussian kernel with Scott's bandwidth matrix (× `bandwidth_scale`), computed by linear binning and FFT convolution; reflection at y = log v̄; cells with externality nodes on a uniform u-grid (241) and valuation cells on a uniform y-grid (400), the valuation uniform within a cell. The estimator is variance-preserving: the data are shrunk toward their mean by (1 + h²)^(−1/2), so the smoothed (u, y) keep the sample mean and covariance. Without this, smoothing in log v inflates the upper tail of v (expected VCG welfare +10% on the Community Notes data at bandwidth scale 1); with it, smoothed and atomic VCG welfare differ by under 1%. Check sensitivity with `bandwidth_scale` 0.5 and 2.

**Why smoothing is the default.** With distinct valuations, each valuation level of an atomic population is a single row, so ŵ^τ(v) = v + e of that row and an atomic IVCG-PA ranks bidders by their true welfare contribution: in-sample it approaches the first best and it overfits. Smoothing makes E[e | v] a conditional mean. The atomic distribution remains available (`--smoothing none`) as a robustness check, mainly for VCG and VCG-PA.

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

**Simulation.** Used only for per-auction welfare distributions and as a check of the exact values. Each draw is one auction; allocations are probabilities (the tie-breaking lottery is averaged out); payments are critical-value payments given the realized rival bids. All mechanisms run on the same draws (common random numbers). For piecewise-uniform cells, valuations are drawn uniformly within the cell.

**Revenue units.** For the first best, revenue is the critical payment in effective-valuation units (w = v + e); the transfer −e of winners is reported separately as penalty paid.

## Optimizing τ

`genetic_search.run_genetic_search(distribution, degree, n, k, objective_mechanism="vcg_pa")` searches polynomial coefficients τ(e) = Σ β_m e^m with the genetic algorithm of the original code (pygad, tournament selection, single-point crossover, Gaussian mutation scaled to each coefficient's range, warm start, restarts). The fitness is the **exact** expected welfare of the objective mechanism on the distribution (`expected_welfare`; about 3–10 ms for VCG-PA and 10–25 ms for IVCG-PA on 180k cells), not an average over simulated auctions, so it has no Monte Carlo noise and does not fit a particular sample of auctions. Search ranges: β_0 ∈ [v_min − sd(v)/2, v̄ + sd(v)/2], β_m ∈ ±3 sd(v)/E|e|^m.

`pipeline.optimize_and_evaluate` runs one cell:

| Source | τ optimized on | τ evaluated on | Uncertainty reported |
|--------|----------------|----------------|----------------------|
| synthetic family | the family's distribution | the same distribution | none (H known); optional grid-refinement check |
| empirical CSV | training half (smoothed or atomic) | test half, same construction | bootstrap over test rows (exact welfare per resample) |

and in both cases one simulation of `num_simulated_draws` auctions on the evaluation distribution for the per-auction welfare figures.

## Figures

`plot_threshold_and_penalties(evaluation, distribution, path)`:
- **(a)** type distribution in the (e, v) plane (contours for synthetic densities; log shading for smoothed and atomic populations), τ(e) with the abstention region shaded, and the first-best boundary v = −e.
- **(b)** penalty functions r(e) of VCG-PA and IVCG-PA, and the recipient penalty −e.
- **(c)** exact expected welfare of the four mechanisms.

`plot_welfare_distributions(evaluation, path)`:
- **(a)** Gaussian-KDE densities of per-auction welfare, with means.
- **(b)** paired per-auction welfare gains over unaudited VCG (VCG-PA − VCG, IVCG-PA − VCG, first best − VCG). Differences are exactly zero in many auctions (same winners); the panel shows the density of the nonzero differences scaled by their share and reports the share of zeros.

Colors are a fixed, color-vision-checked assignment per mechanism (`MECHANISM_COLORS`); each series also has its own line style.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

`tests/test_audited_auctions.py`: Example T (ironing pools valuations 2 and 3; IVCG-PA welfare 40/16; first best 41/16; VCG-PA ≤ 39/16); the second-price penalty τ² (exact for uniform valuations); Example N welfare and penalty (exact); exact ironing of piecewise-linear ŵ^τ against ironing of a fine atomic discretization; ironing binds and helps in `harmful_middle`; exact expectations equal simulated means for all mechanisms and metrics (atomic and piecewise-uniform); a threshold type is indifferent (its simulated auction-utility equals r(e)); IVCG-PA ≥ VCG-PA; inverses of the piecewise-linear score; negative valuations are rejected.

`tests/test_pipeline.py`: every family has bounded non-negative continuous valuations; variance-preserving smoothing keeps the moments; the split is deterministic; bootstrap gains are paired; smoothed and atomic VCG welfare agree; the genetic search recovers the Example N optimum t* = 4c/3; a synthetic pipeline cell with grid refinement.

## Conventions and limitations

- Only positive scores are allocated (VCG maximizes reported welfare, so a nonpositive bid never wins).
- Externalities of piecewise-uniform distributions are discrete (grid nodes); only valuations are continuous given the externality.
- The truncation bound v̄ of empirical data matters for the exclusion penalty R(v̄).
- τ is a polynomial; it cannot represent exclusion (τ = v̄⁺) or jumps. A grid representation of τ and the best-response ascent based on the first-order condition of Appendix E are planned next.
- The genetic search maximizes VCG-PA welfare by default; IVCG-PA is then evaluated at that τ (`objective_mechanism="ivcg_pa"` optimizes for IVCG-PA instead).
