# `audited_auctions` — exact computation and simulation for a common threshold τ

Given a joint distribution H of bidder types (e, v) and a common participation threshold function τ(e), this package computes the outcomes of four mechanisms with n exchangeable bidders and k identical goods (unit demand):

| Key | Mechanism | Participants | Ranking score | Penalty |
|-----|-----------|--------------|---------------|---------|
| `vcg` | VCG without audits | all bidders | bid v | none |
| `vcg_pa` | VCG-PA | types with v ≥ τ(e) | bid v | r(e) inducing τ, charged to participants |
| `ivcg_pa` | IVCG-PA | types with v ≥ τ(e) | ironed welfare contribution ω̄^τ(v) | r(e) inducing τ, charged to participants |
| `first_best` | optimal recipient audit | all bidders | w = v + e | −e, charged to winners |

All four are score-based auctions: the k participants with the highest positive scores win, ties are broken uniformly at random, and payments are critical-value (Myerson) payments, so each auction is DSIC. The notation follows the WEB27 core draft (`WEB27/drafting/web27_aa_1p_core/`).

The penalties are outputs, not inputs: since τ is given, the package computes the penalty function that induces τ in each participant-audit auction.

## Quick start

```python
from audited_auctions import evaluate_threshold, synthetic_distribution, polynomial_threshold
from audited_auctions.figures import plot_threshold_and_penalties, plot_welfare_distributions

distribution = synthetic_distribution("d")             # gridded Gaussian mixture
threshold = polynomial_threshold([0.0])                # τ(e) = 0
evaluation = evaluate_threshold(distribution, threshold, num_bidders=8, num_goods=2,
                                num_simulated_draws=20000, random_seed=0)
print(evaluation.summary_table())                       # exact and simulated results
plot_threshold_and_penalties(evaluation, distribution, "threshold_penalties.png")
plot_welfare_distributions(evaluation, "welfare_distributions.png")
```

The same from the command line: `python evaluate_threshold.py --distribution d --tau-coefficients 0 --num-bidders 8 --num-goods 2` (see `python evaluate_threshold.py --help`; it also accepts a population CSV and a sweep result pickle as the source of τ).

## Modules

| Module | Contents |
|--------|----------|
| `type_distributions.py` | `TypeDistribution`: a finite list of atoms (e_j, v_j) with probabilities. Constructors for populations (`population_distribution`, `population_distribution_from_csv`) and for gridded densities (`gridded_density_distribution`, `gaussian_mixture_distribution`, `synthetic_distribution` for the samples a–f, h of `alternate_distributions.ipynb`). |
| `threshold_functions.py` | τ(e) from polynomial coefficients, from tabulated values (any other source), or a constant. τ = +inf excludes an externality (the threshold v̄⁺ of the paper). |
| `scoring_rules.py` | `IdentityScore` (rank by bid) and `StepScore` (rank by a nondecreasing step function of the bid, used for ω̄^τ), with the generalized inverses needed for payments. |
| `exact_computation.py` | Ironing (lower convex hull), interim allocation with ties, `ScoreAuctionModel` with exact expected welfare, revenue, allocation integral R(b) and penalties. |
| `auction_simulation.py` | Vectorized Monte Carlo of a score-based auction on sampled type profiles: per-draw allocation probabilities, critical-value payments, welfare, revenue, penalties, participation. |
| `threshold_evaluation.py` | `evaluate_threshold` (main entry point) and the `ThresholdEvaluation` result (exact results, penalty curves and tables, ŵ^τ and ω̄^τ, simulated draws; `summary_table`, `paired_welfare_differences`, `save`/`load`). |
| `figures.py` | `plot_threshold_and_penalties`, `plot_welfare_distributions`. |
| `sweep_support.py` | Glue for `run_sweep_genetic2.py` and `post_optimization_analysis.py` (flat per-draw keys, exact penalty curves of a sweep record). |

## What is computed, and how

**Distributions are discrete.** A population (Community Notes data, a CSV sample) is used as is: bidders are i.i.d. draws from its rows. A continuous density is discretized onto a grid of (e, v) nodes (default 301 × 601), each node an atom. Exact computation and simulation use the same atoms, so they agree up to Monte Carlo error; the discretization error relative to the continuous density is of the order of the grid spacing.

**Participation and levels.** Under the threshold strategy, a type participates iff v ≥ τ(e) and bids v (the first best: everyone, effective bid w = v + e). Participating atoms are grouped by bid into levels b_g with mass μ_g = Pr[participate, bid = b_g] and welfare mass E[(v + e) 1{participate, bid = b_g}].

**Ironing (IVCG-PA).** The expected welfare contribution is ŵ^τ(b_g) = b_g + E[e | v = b_g, v ≥ τ(e)] = welfare mass / μ_g. It conditions on participation, so it changes with τ, and it need not be monotone. In quantile space (cumulative μ on the x-axis, cumulative welfare mass on the y-axis, i.e. Ψ of the core draft up to the factor 1/q), the ironed welfare contribution ω̄^τ is the slope of the lower convex hull over each level's quantile interval. Pooled levels get identical scores and tie. A bid between levels takes the score of the largest participating level below it; a bid below all levels scores −inf.

**Interim allocation.** For a participant with score s, let a = Pr[a rival participates with score > s] and z = Pr[a rival participates with score = s]. With A rivals above and Z tied, the participant wins with probability min(1, (k − A)/(Z + 1)) if A < k and s > 0, else 0. The interim allocation X(s) sums this over the binomial distribution of A and the conditional binomial distribution of Z.

**Penalties.** With the normalized DSIC payment (zero bid pays zero), the penalty inducing τ is

    r(e) = R(τ(e)) = ∫_0^{τ(e)} X^τ(b) db,

with τ clipped to [0, v̄]; an excluded externality (τ ≥ v̄, including +inf) gets R(v̄). R is piecewise linear and is tabulated exactly at the breakpoints of X^τ (`allocation_integral_table`, `penalty_from_table`). For VCG-PA with one good, X^τ(b) is the probability that no participating rival bids above b; the rivals' participation depends on their own externalities, so the expectation is over H, not over the valuation marginal.

This replaces the old estimator R(e) = A(e)(τ(e) − P(e)) built from the realized (k+1)-th admitted price of each draw, which compared τ(e) with the (k+1)-th highest admitted valuation among the n drawn bidders instead of the k-th highest participating valuation among the n − 1 rivals of a bidder at the threshold, and was biased upward (`WEB27/drafting/new_direction_proposals/audit-extensions.tex`, section "Three corrections").

**Expected outcomes** (all n bidders together):

    welfare            = n Σ_g X(b_g) · welfare mass_g
    revenue            = n Σ_g μ_g [b_g X(b_g) − R(b_g)]
    penalty paid (PA)  = n E[r(e) 1{v ≥ τ(e)}]
    penalty paid (RA)  = n E[x · (−e)]   (negative values are net subsidies)
    participants       = n Pr[v ≥ τ(e)]

**Simulation.** Each draw is one auction. Allocations are probabilities (the tie-breaking lottery is averaged out). Payments are the critical-value payments given the realized rival bids: with c the k-th highest rival score, bidder i pays b_i x_i − ∫_0^{b_i} x_i(t) dt, where x_i(t) = 1 when score(t) > max(c, 0), the tie share when score(t) = c > 0, and 0 otherwise. All mechanisms run on the same draws (common random numbers), so per-draw differences are paired.

**Revenue units.** For the first best, revenue is the critical payment in effective-valuation units (w = v + e); the transfer −e of winners is reported separately as penalty paid.

## Figures

`plot_threshold_and_penalties(evaluation, distribution, path)`:
- **(a)** type distribution in the (e, v) plane (contours for gridded densities, a log-shaded histogram for populations), τ(e) with the abstention region shaded, and the first-best boundary v = −e.
- **(b)** penalty functions r(e) of VCG-PA and IVCG-PA, and the recipient penalty −e.
- **(c)** exact expected welfare of the four mechanisms.

`plot_welfare_distributions(evaluation, path)`:
- **(a)** Gaussian-KDE densities of per-draw welfare, with means.
- **(b)** paired per-draw welfare gains over unaudited VCG (VCG-PA − VCG, IVCG-PA − VCG, first best − VCG); all welfare comparisons use VCG as the baseline. Differences are exactly zero in many draws (same winners); the panel shows the density of the nonzero differences scaled by their share and reports the share of zeros.

Colors are a fixed, color-vision-checked assignment per mechanism (`MECHANISM_COLORS`); each series also has its own line style.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

Checks: Example T of the core draft (ironing pools valuations 2 and 3; IVCG-PA welfare 40/16; first best 41/16; VCG-PA ≤ 39/16); the second-price penalty τ² for two bidders with uniform valuations; exact expectations equal simulated means for all mechanisms and metrics (including a population with ties); a type at the threshold is indifferent (its simulated auction-utility equals r(e)); IVCG-PA welfare ≥ VCG-PA welfare.

## Conventions and limitations

- Only positive scores are allocated (VCG maximizes reported welfare, so a nonpositive bid never wins). Synthetic samples a–f contain negative valuations; those bidders are never allocated by VCG or VCG-PA.
- Valuations are bounded by the largest atom valuation v̄. For heavy-tailed data the exclusion penalty R(v̄) is large.
- Truncating distributions to the model's assumptions (v ∈ [0, v̄], no atoms) and smoothing the population are not done yet.
- ŵ^τ for a population is the average externality among participating atoms with exactly that valuation; with few atoms per valuation it is noisy, and ironing then pools noise.
