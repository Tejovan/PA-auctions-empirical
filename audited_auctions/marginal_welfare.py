"""
marginal_welfare.py

Marginal welfare of a participating type (core draft, Appendix E), computed
exactly for a score-based auction model (exact_computation.ScoreAuctionModel)
held fixed:

    M^τ(e, v) = (v + e) X^τ(v) − L^τ(v) = A^τ(v) + e X^τ(v),   A^τ = v X^τ − L^τ,

where X^τ(v) is the interim allocation of a bid v against n − 1 rivals playing
the TPT strategy for τ, and L^τ(v) is the expected welfare that the rivals lose
when the bidder participates with bid v instead of abstaining. M^τ is affine in
the externality e, so it is given by the two functions A^τ, X^τ of the
valuation (core draft, Rem. E.6).

Rival welfare loss (k identical goods, unit demand, uniform tie-breaking, only
positive scores allocated). Let S_(k) be the k-th highest rival score
(−∞ if fewer than k rivals participate). A participant with score s > 0 is
allocated surely if S_(k) < s, with some probability if S_(k) = s (ties), and
never if S_(k) > s. When it is allocated and S_(k) > 0, it displaces one rival
with score S_(k), whose expected welfare contribution is ĉ(S_(k)) =
E[v + e | score = S_(k), participate]; otherwise it takes a good no rival
would have received. Hence

    L(s) = ∫_{0 < S_(k) < s} ĉ dPr[S_(k)] + ĉ(s) (X(s) − Pr[S_(k) < s]).

In quantile space (q = participating rival mass below a score, P the
participation probability) Pr[k-th highest rival quantile <= q] =
B(q) = Pr[Bin(n − 1, P − q) <= k − 1], so dB = (n − 1) Bin(k − 1; n − 2, P − q) dq.
ĉ is the welfare contribution ŵ^τ on segments with increasing scores, and the
mass-weighted mean of ŵ^τ over a run of segments with the same constant score
(an atom, or an ironing pool of IVCG-PA, where the mean equals the ironed
score). For VCG-PA, L = P + Δ (Clarke payment plus displaced externality) and
M = Ū + e X − Δ (Cor. E.4).

The marginal welfare holds the auction fixed: for IVCG-PA, the ironed scores
of the current τ (envelope argument; core draft §5, joint choice).

Exactness. Between consecutive breakpoints (cell edges of the distribution,
segment ends of the model, knots of the scoring rule), X and L are polynomials
in v of degree at most n. ValuationFunctions stores them, and the
antiderivatives of A and X, as Chebyshev series on each interval, interpolated
at n + 3 Chebyshev points (exact up to rounding).
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev
from scipy.stats import binom

from .exact_computation import _gauss_legendre, interim_allocation_of_scores, quadrature_order
from .scoring_rules import PiecewiseLinearScore


class RivalWelfareLoss:
    """X(v) and L(v) of a bid v against the rivals of a fixed ScoreAuctionModel."""

    def __init__(self, model):
        self.model = model
        s = model.segments
        self.num_bidders, self.num_goods = model.num_bidders, model.num_goods
        self.participation = model.participation_probability
        self.profile = model.rival_profile
        self.quantile_start, self.quantile_end = s["quantile_start"], s["quantile_end"]
        self.contribution_start, self.contribution_end = self._conditional_contributions(s)
        self.positive = s["score_end"] > 0
        nodes, weights = _gauss_legendre(self.quantile_start, self.quantile_end,
                                         quadrature_order(self.num_bidders))
        integrals = np.sum(weights * self._integrand(np.arange(len(self.quantile_start)), nodes),
                           axis=1) * self.positive
        self.cumulative_start = np.concatenate([[0.0], np.cumsum(integrals)])

    @staticmethod
    def _conditional_contributions(s):
        """ĉ = E[v + e | score] at the segment ends (pooled mean on constant-score runs)."""
        start, end = s["welfare_start"].copy(), s["welfare_end"].copy()
        scores_start, scores_end = s["score_start"], s["score_end"]
        scale = max(1.0, float(np.max(np.abs(scores_end[np.isfinite(scores_end)]), initial=1.0)))
        constant = np.abs(scores_end - scores_start) <= 1e-12 * scale
        index = 0
        num_segments = len(start)
        while index < num_segments:
            if not constant[index]:
                index += 1
                continue
            run_end = index + 1
            while (run_end < num_segments and constant[run_end]
                   and abs(scores_start[run_end] - scores_start[index]) <= 1e-12 * scale):
                run_end += 1
            lengths = s["quantile_end"][index:run_end] - s["quantile_start"][index:run_end]
            total = lengths.sum()
            mean = (np.sum((start[index:run_end] + end[index:run_end]) / 2 * lengths) / total
                    if total > 0 else start[index])
            start[index:run_end] = mean
            end[index:run_end] = mean
            index = run_end
        return start, end

    def _contribution_at(self, segment, quantiles):
        q_start, q_end = self.quantile_start[segment], self.quantile_end[segment]
        length = q_end - q_start
        length = length[..., None] if quantiles.ndim > np.ndim(segment) else length
        q_start_b = q_start[..., None] if quantiles.ndim > np.ndim(segment) else q_start
        c_start = self.contribution_start[segment]
        c_end = self.contribution_end[segment]
        if quantiles.ndim > np.ndim(segment):
            c_start, c_end = c_start[..., None], c_end[..., None]
        fraction = np.where(length > 0, (quantiles - q_start_b) / np.where(length > 0, length, 1.0),
                            0.0)
        return c_start + (c_end - c_start) * fraction

    def _displacement_density(self, quantiles):
        """dB/dq = (n − 1) Bin(k − 1; n − 2, P − q)."""
        n, k = self.num_bidders, self.num_goods
        if n < 2 or k > n - 1:
            return np.zeros_like(quantiles)
        mass_above = np.clip(self.participation - quantiles, 0.0, 1.0)
        return (n - 1) * binom.pmf(k - 1, n - 2, mass_above)

    def _integrand(self, segment, quantiles):
        return self._contribution_at(segment, quantiles) * self._displacement_density(quantiles)

    def __call__(self, bids):
        """(X(v), L(v)) for an array of bids v."""
        bids = np.asarray(bids, dtype=float)
        scores = self.model.scoring_rule.score_of_bid(bids)
        allocation = interim_allocation_of_scores(scores, self.profile, self.num_bidders,
                                                  self.num_goods)
        mass_below = np.clip(self.profile.mass_below(scores), 0.0, self.participation)
        num_segments = len(self.quantile_start)
        segment = np.minimum(np.searchsorted(self.quantile_end, mass_below, side="left"),
                             num_segments - 1)
        nodes, weights = _gauss_legendre(self.quantile_start[segment],
                                         np.maximum(mass_below, self.quantile_start[segment]),
                                         quadrature_order(self.num_bidders))
        partial = np.sum(weights * self._integrand(segment, nodes), axis=1) * self.positive[segment]
        displaced = self.cumulative_start[segment] + partial
        surely_allocated = np.array(binom.cdf(self.num_goods - 1, self.num_bidders - 1,
                                              np.clip(self.participation - mass_below, 0.0, 1.0)))
        tie_allocation = np.maximum(allocation - surely_allocated, 0.0)
        tied_segment = np.minimum(np.searchsorted(self.quantile_end, mass_below, side="right"),
                                  num_segments - 1)
        loss = displaced + self.contribution_start[tied_segment] * tie_allocation
        positive = scores > 0
        return np.where(positive, allocation, 0.0), np.where(positive, loss, 0.0)


@dataclass
class ValuationFunctions:
    """X, L and A = vX − L on [0, v̄] as Chebyshev series between breakpoints.

    breakpoints : (N + 1,) increasing, from 0 to v̄
    coefficients_x, coefficients_a : (degree + 1, N) series in x ∈ [−1, 1],
        v = midpoint + half_width · x
    antiderivative_x, antiderivative_a : series of ∫_{left end}^{v} (in v units)
    """

    breakpoints: np.ndarray
    coefficients_x: np.ndarray
    coefficients_a: np.ndarray
    antiderivative_x: np.ndarray
    antiderivative_a: np.ndarray
    integral_x: np.ndarray
    integral_a: np.ndarray
    rival_welfare_loss: RivalWelfareLoss

    @property
    def num_intervals(self):
        return len(self.breakpoints) - 1

    def local_coordinate(self, interval, valuations):
        left, right = self.breakpoints[interval], self.breakpoints[interval + 1]
        half = (right - left) / 2
        return np.clip((valuations - left) / np.where(half > 0, half, 1.0) - 1.0, -1.0, 1.0)

    def interval_of(self, valuations):
        return np.clip(np.searchsorted(self.breakpoints, valuations, side="right") - 1, 0,
                       self.num_intervals - 1)

    def partial_integrals(self, interval, valuations):
        """(∫_v^{right end} A, ∫_v^{right end} X) within each valuation's interval."""
        x = self.local_coordinate(interval, valuations)
        below_a = chebyshev.chebval(x, self.antiderivative_a[:, interval], tensor=False)
        below_x = chebyshev.chebval(x, self.antiderivative_x[:, interval], tensor=False)
        return self.integral_a[interval] - below_a, self.integral_x[interval] - below_x


def valuation_breakpoints(model, type_distribution):
    """0, v̄, cell edges, segment ends, scoring-rule knots and zero crossings, in [0, v̄]."""
    maximum = float(type_distribution.maximum_valuation)
    s = model.segments
    points = [np.array([0.0, maximum]), type_distribution.valuation_lower,
              type_distribution.valuation_upper, s["bid_start"], s["bid_end"]]
    rule = model.scoring_rule
    if isinstance(rule, PiecewiseLinearScore):
        points.append(rule.knots)
        lengths = np.append(np.diff(rule.knots), 0.0)
        crossing = (rule.start_scores < 0) & (rule.end_scores > 0) & (lengths > 0)
        points.append(rule.knots[crossing] + lengths[crossing] * (-rule.start_scores[crossing])
                      / (rule.end_scores[crossing] - rule.start_scores[crossing]))
    points = np.unique(np.clip(np.concatenate(points), 0.0, maximum))
    keep = np.concatenate([[True], np.diff(points) > 1e-12 * max(maximum, 1.0)])
    points = points[keep]
    points[-1] = maximum
    return points


def valuation_functions(model, type_distribution):
    """ValuationFunctions of a fixed ScoreAuctionModel (VCG-PA or IVCG-PA, valuation bids)."""
    loss = RivalWelfareLoss(model)
    breakpoints = valuation_breakpoints(model, type_distribution)
    degree = model.num_bidders + 2
    unit_points = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    vandermonde = chebyshev.chebvander(unit_points, degree)
    interpolation = np.linalg.inv(vandermonde)
    left, right = breakpoints[:-1], breakpoints[1:]
    half = (right - left) / 2
    valuations = ((left + right) / 2)[:, None] + half[:, None] * unit_points
    allocation, rival_loss = loss(valuations.ravel())
    allocation = allocation.reshape(valuations.shape)
    marginal_base = valuations * allocation - rival_loss.reshape(valuations.shape)
    coefficients_x = interpolation @ allocation.T
    coefficients_a = interpolation @ marginal_base.T
    antiderivative_x = chebyshev.chebint(coefficients_x, lbnd=-1, axis=0) * half
    antiderivative_a = chebyshev.chebint(coefficients_a, lbnd=-1, axis=0) * half
    ones = np.ones(len(left))
    return ValuationFunctions(
        breakpoints=breakpoints, coefficients_x=coefficients_x, coefficients_a=coefficients_a,
        antiderivative_x=antiderivative_x, antiderivative_a=antiderivative_a,
        integral_x=chebyshev.chebval(ones, antiderivative_x, tensor=False),
        integral_a=chebyshev.chebval(ones, antiderivative_a, tensor=False),
        rival_welfare_loss=loss)
