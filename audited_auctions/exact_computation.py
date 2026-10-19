"""
exact_computation.py

Exact expected outcomes of score-based auctions with exchangeable bidders and a
common threshold function τ, for a type distribution made of cells (see
type_distributions.py): atoms, or cells with a uniform valuation.

Setting (as in the paper, arxiv_WEB v1): n bidders with i.i.d. types (e, v) ~ H,
k identical goods, unit demand. Under the participant-truthful threshold (TPT)
strategy a bidder participates iff v >= τ(e) and then bids v. The first best
(recipient audit with penalty −e) is the same kind of auction in which every
bidder participates and bids w = v + e.

Quantile representation
-----------------------
Let P be the probability that a bidder participates. Order the participating
types by bid and let q ∈ [0, P] be the participating mass below a bid (the
"quantile"). The bid distribution is a list of segments in q, sorted by bid:

  * atoms (atomic distributions): one segment per distinct bid, of q-length
    equal to its mass, with a constant bid;
  * elementary bid intervals (piecewise-uniform distributions): the bid line is
    cut at every cell end point (and at the thresholds); on each piece the
    participating bid density is constant, so the bid is linear in q.

On each segment the expected welfare contribution
      ŵ^τ(b) = E[v + e | bid = b, participate]
is linear in q (b + const for valuation bids, b for the first best), and so is
the expected valuation; the expected externality is constant.

Scores: VCG, VCG-PA and the first best score a bid by itself. IVCG-PA scores a
participant by the ironed welfare contribution ω̄^τ, the nondecreasing L2
projection of ŵ^τ in quantile space; equivalently, ω̄^τ is the derivative of
the lower convex hull of Ψ(q) = ∫_0^q ŵ^τ (the paper's Ψ of §6 [thm:ironing], with the unnormalized quantile q).
`iron_quantile_function` computes it exactly (pool-adjacent-violators for
functions that are linear on each segment; the pool levels are roots of a
monotone equation solved to machine precision).

Interim allocation
------------------
A rival participates with score above s with probability a(s) and with score
equal to s with probability z(s) (z > 0 only on atoms and ironing intervals).
With A rivals above and Z rivals tied, a participant with positive score s is
allocated with probability min(1, (k − A)/(Z + 1)) if A < k, else 0:
      X(s) = Σ_{A<k} Bin(A; n−1, a) Σ_Z Bin(Z; n−1−A, z/(1−a)) min(1, (k−A)/(Z+1)).
If z = 0 this is Pr[Bin(n−1, a) <= k−1]. On a segment without ties a = P − q,
so X is a polynomial of degree n − 1 in q.

Exact integration
-----------------
Every expectation below is an integral, over segments, of a polynomial of
degree at most n in q (or in the bid), so Gauss–Legendre quadrature with
n//2 + 2 nodes per segment is exact (up to floating point):

  welfare             n ∫ ŵ^τ(q) X(q) dq
  allocation integral R(b) = ∫_0^b X(t) dt   (tabulated at the knots of X)
  revenue             n [∫ b(q) X(q) dq − ∫_0^∞ X(t) Pr(bid > t) dt]
                      (the critical-value payment is p(b) = b X(b) − R(b),
                       and E[R(bid)] = ∫ X(t) Pr(bid > t) dt)

Facts used (paper Thm 5.9 [thm:dsic]): the penalty inducing τ in a normalized DSIC
auction is r(e) = R(τ(e)), with τ clipped to [0, v̄] and r(e) = R(v̄) for an
excluded externality; the critical-value payment is b X(b) − R(b).
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.stats import binom

from .scoring_rules import IdentityScore, PiecewiseLinearScore

VALUATION_BIDS = "valuation"
WELFARE_CONTRIBUTION_BIDS = "welfare_contribution"


def quadrature_order(num_bidders):
    """Gauss–Legendre nodes per segment; exact for polynomials of degree <= n + 3."""
    return num_bidders // 2 + 2


# ─── Interim allocation from rival masses ─────────────────────────────────────


def allocation_given_rival_masses(mass_above, mass_tied, num_bidders, num_goods):
    """X = Pr[allocated] for a positive score with rival masses a (above) and z (tied)."""
    mass_above = np.clip(np.asarray(mass_above, dtype=float), 0.0, 1.0)
    mass_tied = np.clip(np.asarray(mass_tied, dtype=float), 0.0, 1.0)
    num_rivals = num_bidders - 1
    allocation = np.array(binom.cdf(num_goods - 1, num_rivals, mass_above), dtype=float)
    has_ties = mass_tied > 0
    if has_ties.any():
        above, tied = mass_above[has_ties], mass_tied[has_ties]
        tied_given_not_above = np.clip(tied / np.maximum(1.0 - above, 1e-300), 0.0, 1.0)
        with_ties = np.zeros_like(above)
        for num_above in range(min(num_goods, num_rivals + 1)):
            remaining_rivals = num_rivals - num_above
            expected_share = np.zeros_like(above)
            for num_tied in range(remaining_rivals + 1):
                share = min(1.0, (num_goods - num_above) / (num_tied + 1))
                expected_share += share * binom.pmf(num_tied, remaining_rivals,
                                                    tied_given_not_above)
            with_ties += binom.pmf(num_above, num_rivals, above) * expected_share
        allocation[has_ties] = with_ties
    return allocation


@dataclass
class RivalScoreProfile:
    """Distribution of a rival's score: a nondecreasing piecewise-linear score
    ω(q) on quantile segments [quantile_start, quantile_end] ⊂ [0, P]."""

    quantile_start: np.ndarray
    quantile_end: np.ndarray
    score_start: np.ndarray
    score_end: np.ndarray
    participation_mass: float

    def mass_at_or_below(self, scores):
        """Q⁺(s) = sup{q : ω(q) <= s}: participating mass with score <= s."""
        scores = np.asarray(scores, dtype=float)
        piece = np.searchsorted(self.score_start, scores, side="right") - 1
        safe = np.clip(piece, 0, len(self.score_start) - 1)
        start, end = self.score_start[safe], self.score_end[safe]
        q_start, q_end = self.quantile_start[safe], self.quantile_end[safe]
        rise = end - start
        inside = (rise > 0) & (scores < end)
        fraction = np.where(inside, (scores - start) / np.where(rise > 0, rise, 1.0), 1.0)
        return np.where(piece >= 0, q_start + fraction * (q_end - q_start), 0.0)

    def mass_below(self, scores):
        """Q⁻(s) = inf{q : ω(q) >= s}: participating mass with score < s."""
        scores = np.asarray(scores, dtype=float)
        piece = np.searchsorted(self.score_end, scores, side="left")
        has_piece = piece < len(self.score_end)
        safe = np.minimum(piece, len(self.score_end) - 1)
        start, end = self.score_start[safe], self.score_end[safe]
        q_start, q_end = self.quantile_start[safe], self.quantile_end[safe]
        rise = end - start
        fraction = np.where((start < scores) & (rise > 0),
                            (scores - start) / np.where(rise > 0, rise, 1.0), 0.0)
        return np.where(has_piece, q_start + fraction * (q_end - q_start), self.participation_mass)

    def rival_masses(self, scores):
        """(a, z): probabilities that a rival participates with score above / equal to s."""
        at_or_below = self.mass_at_or_below(scores)
        tied = np.maximum(at_or_below - self.mass_below(scores), 0.0)
        tied = np.where(tied > 1e-12, tied, 0.0)  # interpolation noise on strictly increasing pieces
        return self.participation_mass - at_or_below, tied


def interim_allocation_of_scores(scores, rival_profile, num_bidders, num_goods):
    """X(s) of a participant with score s (0 for s <= 0)."""
    scores = np.asarray(scores, dtype=float)
    above, tied = rival_profile.rival_masses(scores)
    allocation = allocation_given_rival_masses(above, tied, num_bidders, num_goods)
    return np.where(scores > 0, allocation, 0.0)


# ─── Ironing ──────────────────────────────────────────────────────────────────


def _crossing(q_start, q_end, value_start, value_end, level):
    """Point in [q_start, q_end] where a linear function from value_start to value_end equals level."""
    if value_end == value_start:
        return q_start
    return q_start + (level - value_start) / (value_end - value_start) * (q_end - q_start)


def iron_quantile_function(quantile_start, quantile_end, value_start, value_end):
    """Nondecreasing L2 projection of a function linear and nondecreasing on each segment.

    Input segments are contiguous in q; on segment i the function rises
    linearly from value_start[i] to value_end[i] (constant for atoms). Returns
    pieces (quantile_start, quantile_end, value_start, value_end) of the
    projection ω̄: each piece is a restriction of an input segment (not
    ironed) or a constant pool level (ironed). ω̄ is the derivative of the lower
    convex hull of the integral Ψ(q) = ∫_0^q value.

    Pool-adjacent-violators, left to right: when a new segment starts below the
    current projection, the pool level m solves D(m) = 0, where D(m) is the
    integral of (value − m) over the region that the pool covers at level m
    (from the point where the projection so far crosses m, to the point where
    the new segment crosses m). D is decreasing in m, so m is unique.
    """
    stack = []  # pieces [q_start, q_end, value_start, value_end], nondecreasing

    def excess(level, segment):
        q_a, q_b, v_a, v_b = segment
        total = 0.0
        for p_a, p_b, w_a, w_b in reversed(stack):
            if w_b <= level:
                break
            if w_a >= level:
                total += ((w_a + w_b) / 2 - level) * (p_b - p_a)
            else:
                crossing = _crossing(p_a, p_b, w_a, w_b, level)
                total += (w_b - level) / 2 * (p_b - crossing)
                break
        if v_b <= level:
            total += ((v_a + v_b) / 2 - level) * (q_b - q_a)
        else:
            crossing = _crossing(q_a, q_b, v_a, v_b, level)
            total += (v_a - level) / 2 * (crossing - q_a)
        return total

    for segment in zip(quantile_start, quantile_end, value_start, value_end):
        q_a, q_b, v_a, v_b = (float(x) for x in segment)
        if q_b <= q_a:
            continue
        if not stack or v_a >= stack[-1][3]:
            stack.append([q_a, q_b, v_a, v_b])
            continue

        segment = (q_a, q_b, v_a, v_b)
        low, high = v_a, stack[-1][3]
        if excess(high, segment) >= 0:
            level = high
        else:
            scale = max(1.0, abs(low), abs(high))
            level = brentq(excess, low, high, args=(segment,), xtol=1e-15 * scale,
                           rtol=8.9e-16, maxiter=200)

        # Remove the part of the projection above the pool level.
        pool_start = stack[-1][1]
        while stack and stack[-1][3] > level:
            p_a, p_b, w_a, w_b = stack.pop()
            if w_a < level:
                pool_start = _crossing(p_a, p_b, w_a, w_b, level)
                if pool_start > p_a:
                    stack.append([p_a, pool_start, w_a, level])
                break
            pool_start = p_a
        pool_end = q_b if v_b <= level else _crossing(q_a, q_b, v_a, v_b, level)
        if pool_end > pool_start:
            stack.append([pool_start, pool_end, level, level])
        if pool_end < q_b:
            stack.append([pool_end, q_b, level, v_b])

    pieces = np.array(stack, dtype=float).reshape(-1, 4)
    return pieces[:, 0], pieces[:, 1], pieces[:, 2], pieces[:, 3]


def ironed_welfare_contributions(level_mass, level_welfare_mass):
    """Ironed welfare contribution of each atom level (atomic distributions).

    level_mass[g] = Pr[participate, bid = b_g]; level_welfare_mass[g] =
    E[(v + e) 1{participate, bid = b_g}]; levels sorted by bid. Levels pooled by
    ironing receive exactly the same floating-point score, so they tie.
    """
    level_mass = np.asarray(level_mass, dtype=float)
    expected = np.asarray(level_welfare_mass, dtype=float) / level_mass
    quantile_end = np.cumsum(level_mass)
    quantile_start = quantile_end - level_mass
    _, piece_end, piece_value, _ = iron_quantile_function(quantile_start, quantile_end,
                                                          expected, expected)
    midpoints = (quantile_start + quantile_end) / 2
    return piece_value[np.minimum(np.searchsorted(piece_end, midpoints), len(piece_end) - 1)]


# ─── Bid segments ─────────────────────────────────────────────────────────────


def _bid_segments(bid_lower, bid_upper, masses, externalities, atomic, extra_breakpoints=()):
    """Aggregate participating cells into segments sorted by bid.

    Returns arrays bid_start, bid_end, mass, mean_externality. For atomic
    distributions a segment is a distinct bid; otherwise an elementary interval
    of the bid line with positive density (gaps with zero density are dropped).
    """
    if atomic:
        bids, inverse = np.unique(bid_lower, return_inverse=True)
        mass = np.bincount(inverse, masses, len(bids))
        externality_mass = np.bincount(inverse, masses * externalities, len(bids))
        return bids, bids.copy(), mass, externality_mass / mass

    breakpoints = np.unique(np.concatenate([bid_lower, bid_upper,
                                            np.asarray(extra_breakpoints, dtype=float)]))
    start_index = np.searchsorted(breakpoints, bid_lower)
    end_index = np.searchsorted(breakpoints, bid_upper)
    density = masses / (bid_upper - bid_lower)
    num_intervals = len(breakpoints) - 1
    density_steps = (np.bincount(start_index, density, num_intervals + 1)
                     - np.bincount(end_index, density, num_intervals + 1))
    externality_steps = (np.bincount(start_index, density * externalities, num_intervals + 1)
                         - np.bincount(end_index, density * externalities, num_intervals + 1))
    interval_density = np.cumsum(density_steps)[:num_intervals]
    interval_externality_density = np.cumsum(externality_steps)[:num_intervals]
    # Cumulative sums leave rounding residue in gaps; treat it as zero density.
    positive = interval_density > 1e-12 * max(density.max(), 1e-300)
    widths = np.diff(breakpoints)
    mass = interval_density * widths
    return (breakpoints[:-1][positive], breakpoints[1:][positive], mass[positive],
            interval_externality_density[positive] / interval_density[positive])


def _gauss_legendre(starts, ends, order):
    """Nodes and weights, each of shape (len(starts), order), on each interval."""
    unit_nodes, unit_weights = leggauss(order)
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    half = (ends - starts)[:, None] / 2
    nodes = (starts + ends)[:, None] / 2 + half * unit_nodes
    return nodes, half * unit_weights


def _linear_at(start_values, end_values, starts, ends, points):
    """Evaluate per-segment linear functions at points (one row per segment)."""
    length = (ends - starts)[:, None]
    fraction = np.where(length > 0, (points - starts[:, None]) / np.where(length > 0, length, 1.0),
                        0.0)
    return start_values[:, None] + (end_values - start_values)[:, None] * fraction


# ─── Allocation integral R(b) ─────────────────────────────────────────────────


@dataclass
class AllocationIntegral:
    """R(b) = ∫_0^b X(t) dt, exact: tabulated at knots between which X is a
    polynomial, and integrated by Gauss–Legendre within an interval."""

    knots: np.ndarray
    integral_at_knots: np.ndarray
    scoring_rule: object
    rival_profile: RivalScoreProfile
    num_bidders: int
    num_goods: int

    def interim_allocation(self, bids):
        return interim_allocation_of_scores(self.scoring_rule.score_of_bid(bids),
                                            self.rival_profile, self.num_bidders, self.num_goods)

    def __call__(self, bids):
        bids = np.asarray(bids, dtype=float)
        flat = np.maximum(bids.ravel(), 0.0)
        interval = np.searchsorted(self.knots, flat, side="right") - 1
        start = self.knots[interval]
        nodes, weights = _gauss_legendre(start, flat, quadrature_order(self.num_bidders))
        partial = (self.interim_allocation(nodes) * weights).sum(axis=1)
        integral = self.integral_at_knots[interval] + partial
        return np.where(bids.ravel() > 0, integral, 0.0).reshape(bids.shape)


def build_allocation_integral(scoring_rule, rival_profile, knots, num_bidders, num_goods):
    knots = np.asarray(knots, dtype=float)
    knots = np.unique(np.concatenate([[0.0], knots[knots > 0]]))
    nodes, weights = _gauss_legendre(knots[:-1], knots[1:], quadrature_order(num_bidders))
    allocation = interim_allocation_of_scores(scoring_rule.score_of_bid(nodes), rival_profile,
                                              num_bidders, num_goods)
    interval_integrals = (allocation * weights).sum(axis=1)
    return AllocationIntegral(
        knots=knots,
        integral_at_knots=np.concatenate([[0.0], np.cumsum(interval_integrals)]),
        scoring_rule=scoring_rule, rival_profile=rival_profile,
        num_bidders=num_bidders, num_goods=num_goods,
    )


# ─── Score-based auction model ────────────────────────────────────────────────


@dataclass
class ScoreAuctionModel:
    """Exact model of one score-based auction under TPT strategies.

    segments: dict of per-segment arrays, contiguous in q and sorted by bid:
      quantile_start/end, bid_start/end, score_start/end,
      welfare_start/end (ŵ^τ), valuation_start/end (E[v | bid]), externality (E[e | bid]).
    cell_participation_mass: per cell of the distribution, Pr[cell, participate].
    """

    mechanism: str
    num_bidders: int
    num_goods: int
    segments: dict
    participation_probability: float
    scoring_rule: object
    cell_participation_mass: np.ndarray
    maximum_bid: float
    _allocation_integral: object = field(default=None, repr=False)

    # Interim allocation and its integral -------------------------------------

    @property
    def rival_profile(self) -> RivalScoreProfile:
        s = self.segments
        return RivalScoreProfile(s["quantile_start"], s["quantile_end"], s["score_start"],
                                 s["score_end"], self.participation_probability)

    def interim_allocation_of_scores(self, scores):
        return interim_allocation_of_scores(scores, self.rival_profile, self.num_bidders,
                                            self.num_goods)

    def interim_allocation_of_bids(self, bids):
        """X(b) for arbitrary bids b (not only bids in the support)."""
        return self.interim_allocation_of_scores(self.scoring_rule.score_of_bid(bids))

    @property
    def allocation_integral_function(self) -> AllocationIntegral:
        """R(b) as a callable object (picklable; stored as the penalty table)."""
        if self._allocation_integral is None:
            s = self.segments
            knots = np.concatenate([s["bid_start"], s["bid_end"], [self.maximum_bid]])
            if isinstance(self.scoring_rule, PiecewiseLinearScore):
                knots = np.concatenate([knots, self.scoring_rule.knots])
            self._allocation_integral = build_allocation_integral(
                self.scoring_rule, self.rival_profile, knots, self.num_bidders, self.num_goods)
        return self._allocation_integral

    def allocation_integral(self, bids):
        """R(b) = ∫_0^b X(t) dt; R(b) = 0 for b <= 0."""
        return self.allocation_integral_function(bids)

    def critical_value_payment(self, bids):
        """Expected payment p(b) = b X(b) − R(b) of the normalized DSIC auction."""
        bids = np.asarray(bids, dtype=float)
        payment = bids * self.interim_allocation_of_bids(bids) - self.allocation_integral(bids)
        return np.where(bids > 0, payment, 0.0)

    # Expected outcomes (all n bidders together) -------------------------------

    def _segment_quadrature(self):
        s = self.segments
        nodes, weights = _gauss_legendre(s["quantile_start"], s["quantile_end"],
                                         quadrature_order(self.num_bidders))

        def at(name):
            return _linear_at(s[f"{name}_start"], s[f"{name}_end"], s["quantile_start"],
                              s["quantile_end"], nodes)

        allocation = self.interim_allocation_of_scores(at("score"))
        return weights, at, allocation

    def expected_welfare(self) -> float:
        """n ∫ ŵ^τ X dq (fast path used as the optimization objective)."""
        weights, at, allocation = self._segment_quadrature()
        return float(self.num_bidders * np.sum(weights * allocation * at("welfare")))

    def expected_outcomes(self) -> dict:
        n = self.num_bidders
        weights, at, allocation = self._segment_quadrature()
        allocated_weight = weights * allocation
        valuation_welfare = n * np.sum(allocated_weight * at("valuation"))
        externality_welfare = n * np.sum(allocated_weight * self.segments["externality"][:, None])
        bid_times_allocation = np.sum(allocated_weight * np.maximum(at("bid"), 0.0))
        return {
            "welfare": float(valuation_welfare + externality_welfare),
            "valuation_welfare": float(valuation_welfare),
            "externality_welfare": float(externality_welfare),
            "revenue": float(n * (bid_times_allocation - self._expected_allocation_integral())),
            "num_allocated": float(n * np.sum(allocated_weight)),
            "participation_probability": float(self.participation_probability),
            "num_participants": float(n * self.participation_probability),
        }

    def _bid_mass_at_or_below(self, bids):
        """F(t): participating mass with bid <= t."""
        s = self.segments
        segment = np.searchsorted(s["bid_start"], bids, side="right") - 1
        safe = np.clip(segment, 0, len(s["bid_start"]) - 1)
        b_start, b_end = s["bid_start"][safe], s["bid_end"][safe]
        length = b_end - b_start
        fraction = np.where((length > 0) & (bids < b_end),
                            (bids - b_start) / np.where(length > 0, length, 1.0), 1.0)
        mass = s["quantile_start"][safe] + fraction * (s["quantile_end"][safe]
                                                       - s["quantile_start"][safe])
        return np.where(segment >= 0, mass, 0.0)

    def _expected_allocation_integral(self) -> float:
        """E[R(bid) 1{participate}] = ∫_0^∞ X(t) Pr(bid > t) dt."""
        table = self.allocation_integral_function
        nodes, weights = _gauss_legendre(table.knots[:-1], table.knots[1:],
                                         quadrature_order(self.num_bidders))
        survival = self.participation_probability - self._bid_mass_at_or_below(nodes)
        return float(np.sum(weights * table.interim_allocation(nodes) * survival))

    def welfare_contribution_curves(self) -> dict:
        """ŵ^τ and the score along the bid axis, at segment end points (for plots)."""
        s = self.segments
        return {key: s[key].copy() for key in (
            "bid_start", "bid_end", "welfare_start", "welfare_end", "score_start", "score_end",
            "quantile_start", "quantile_end")}


# ─── Building a model ─────────────────────────────────────────────────────────


def participating_cells(distribution, cell_thresholds):
    """Participating part of each cell under v >= τ(e).

    Returns (lowest participating valuation, participating mass) per cell; the
    mass is zero for cells that do not participate.
    """
    lower, upper = distribution.valuation_lower, distribution.valuation_upper
    thresholds = np.asarray(cell_thresholds, dtype=float)
    participating_lower = np.minimum(np.maximum(lower, thresholds), upper)
    if distribution.is_atomic:
        mass = np.where(lower >= thresholds, distribution.probabilities, 0.0)
    else:
        fraction = np.clip((upper - participating_lower) / (upper - lower), 0.0, 1.0)
        mass = distribution.probabilities * fraction
    return participating_lower, mass


def _interpolate_rows(start_values, end_values, q_start, q_end, index, points):
    length = q_end[index] - q_start[index]
    fraction = np.where(length > 0, (points - q_start[index]) / np.where(length > 0, length, 1.0),
                        0.0)
    return start_values[index] + (end_values[index] - start_values[index]) * fraction


def _refine(segments, piece_start, piece_end, piece_score_start, piece_score_end):
    """Cut segments at the boundaries of score pieces and attach the scores."""
    q_breaks = np.unique(np.concatenate([segments["quantile_start"], segments["quantile_end"],
                                         piece_start, piece_end]))
    total = segments["quantile_end"][-1]
    # Merge boundaries that differ only by rounding.
    q_breaks = q_breaks[np.concatenate([[True], np.diff(q_breaks) > 1e-14 * max(total, 1e-300)])]
    starts, ends = q_breaks[:-1], q_breaks[1:]
    midpoints = (starts + ends) / 2
    segment = np.minimum(np.searchsorted(segments["quantile_end"], midpoints),
                         len(segments["quantile_end"]) - 1)
    piece = np.minimum(np.searchsorted(piece_end, midpoints), len(piece_end) - 1)

    refined = {"quantile_start": starts, "quantile_end": ends,
               "externality": segments["externality"][segment]}
    for name in ("bid", "welfare", "valuation"):
        for label, points in (("start", starts), ("end", ends)):
            refined[f"{name}_{label}"] = _interpolate_rows(
                segments[f"{name}_start"], segments[f"{name}_end"], segments["quantile_start"],
                segments["quantile_end"], segment, points)
    for label, points in (("start", starts), ("end", ends)):
        refined[f"score_{label}"] = _interpolate_rows(piece_score_start, piece_score_end,
                                                      piece_start, piece_end, piece, points)
    return refined


def _split_at_zero_score(segments):
    """Cut segments whose score crosses 0, so that X is a polynomial on each segment."""
    crossing = np.flatnonzero((segments["score_start"] < 0) & (segments["score_end"] > 0))
    if len(crossing) == 0:
        return segments
    fraction = -segments["score_start"][crossing] / (
        segments["score_end"][crossing] - segments["score_start"][crossing])
    order = np.argsort(np.concatenate([np.arange(len(segments["quantile_start"])), crossing + 0.5]))
    split = {}
    for key, values in segments.items():
        if key.endswith("_start"):
            name = key[: -len("_start")]
            cut = values[crossing] + fraction * (segments[f"{name}_end"][crossing] - values[crossing])
            starts = values.copy()
            split[key] = np.concatenate([starts, cut])[order]
            ends = segments[f"{name}_end"].copy()
            ends[crossing] = cut
            split[f"{name}_end"] = np.concatenate([ends, segments[f"{name}_end"][crossing]])[order]
        elif not key.endswith("_end"):
            split[key] = np.concatenate([values, values[crossing]])[order]
    return split


def _enforce_monotone(start_values, end_values):
    """Remove rounding violations of monotonicity along start_0, end_0, start_1, end_1, ..."""
    interleaved = np.maximum.accumulate(np.column_stack([start_values, end_values]).ravel())
    return interleaved[0::2].copy(), interleaved[1::2].copy()


def _bid_space_score(segments):
    """The ironed score as a function of the bid (PiecewiseLinearScore).

    Walk the segments in bid order: a segment with positive bid length is a
    linear piece; a gap after it (no participating bids) keeps the score
    approached from below; an atom starts a constant piece that lasts until the
    next participating bid.
    """
    bid_start, bid_end = segments["bid_start"], segments["bid_end"]
    score_start, score_end = segments["score_start"], segments["score_end"]
    knots, starts, ends = [], [], []
    num_segments = len(bid_start)
    for index in range(num_segments):
        next_bid = bid_start[index + 1] if index + 1 < num_segments else np.inf
        if bid_end[index] > bid_start[index]:
            knots.append(bid_start[index])
            starts.append(score_start[index])
            ends.append(score_end[index])
            if bid_end[index] < next_bid:  # gap: hold the left limit
                knots.append(bid_end[index])
                starts.append(score_end[index])
                ends.append(score_end[index])
        else:  # atom
            knots.append(bid_start[index])
            starts.append(score_start[index])
            ends.append(score_start[index])
    knots, starts, ends = np.array(knots), np.array(starts), np.array(ends)
    # Keep the last piece at each knot (an atom split by ironing has one score).
    keep = np.append(knots[1:] > knots[:-1], True)
    knots, starts, ends = knots[keep], starts[keep], ends[keep]
    # Remove rounding violations of monotonicity.
    for index in range(1, len(knots)):
        starts[index] = max(starts[index], ends[index - 1])
        ends[index] = max(ends[index], starts[index])
    ends[-1] = starts[-1]
    return PiecewiseLinearScore(knots, starts, ends)


# Finite stand-in for a score of −inf (bids below the support of a fixed scoring
# rule), so that per-segment linear interpolation stays finite.
NEVER_ALLOCATED_SCORE = -1e300


def build_score_auction_model(mechanism, distribution, cell_thresholds, bid_kind, ironed,
                              num_bidders, num_goods, fixed_scoring_rule=None):
    """Build the exact model of one score-based auction.

    mechanism       : label stored in the model
    distribution    : TypeDistribution
    cell_thresholds : (num_cells,) τ(e_j) of each cell (−inf: everyone participates)
    bid_kind        : VALUATION_BIDS (bid v) or WELFARE_CONTRIBUTION_BIDS (bid w = v + e)
    ironed          : False -> score = bid; True -> score = ironed welfare contribution
    fixed_scoring_rule : optional PiecewiseLinearScore used instead of ironing this
                      distribution (an IVCG-PA designed on another distribution, e.g.
                      the training data, evaluated on this one)
    """
    participating_lower, mass = participating_cells(distribution, cell_thresholds)
    participates = mass > 0
    if not participates.any():
        raise ValueError(f"{mechanism}: no type participates under this threshold")
    externalities = distribution.externalities[participates]
    shift = externalities if bid_kind == WELFARE_CONTRIBUTION_BIDS else 0.0
    bid_lower = participating_lower[participates] + shift
    bid_upper = distribution.valuation_upper[participates] + shift
    extra_breakpoints = [0.0] if bid_kind == WELFARE_CONTRIBUTION_BIDS else []
    if fixed_scoring_rule is not None:
        # Cut at the knots of the scoring rule so the score is linear on each segment.
        extra_breakpoints = list(extra_breakpoints) + list(fixed_scoring_rule.knots)
    bid_start, bid_end, segment_mass, mean_externality = _bid_segments(
        bid_lower, bid_upper, mass[participates], externalities, distribution.is_atomic,
        extra_breakpoints)

    quantile_end = np.cumsum(segment_mass)
    quantile_start = np.concatenate([[0.0], quantile_end[:-1]])
    welfare_offset = mean_externality if bid_kind == VALUATION_BIDS else 0.0
    valuation_offset = 0.0 if bid_kind == VALUATION_BIDS else -mean_externality
    segments = {
        "quantile_start": quantile_start, "quantile_end": quantile_end,
        "bid_start": bid_start, "bid_end": bid_end,
        "welfare_start": bid_start + welfare_offset, "welfare_end": bid_end + welfare_offset,
        "valuation_start": bid_start + valuation_offset,
        "valuation_end": bid_end + valuation_offset,
        "externality": mean_externality,
    }

    if fixed_scoring_rule is not None:
        segments["score_start"] = np.maximum(fixed_scoring_rule.score_of_bid(bid_start),
                                             NEVER_ALLOCATED_SCORE)
        segments["score_end"] = np.where(
            bid_end > bid_start, np.maximum(fixed_scoring_rule.left_limit(bid_end),
                                            NEVER_ALLOCATED_SCORE),
            segments["score_start"])
        segments["score_start"], segments["score_end"] = _enforce_monotone(
            segments["score_start"], segments["score_end"])
        scoring_rule = fixed_scoring_rule
    elif ironed:
        pieces = iron_quantile_function(quantile_start, quantile_end, segments["welfare_start"],
                                        segments["welfare_end"])
        segments = _split_at_zero_score(_refine(segments, *pieces))
        segments["score_start"], segments["score_end"] = _enforce_monotone(
            segments["score_start"], segments["score_end"])
        scoring_rule = _bid_space_score(segments)
    else:
        segments["score_start"] = bid_start.copy()
        segments["score_end"] = bid_end.copy()
        scoring_rule = IdentityScore()

    return ScoreAuctionModel(
        mechanism=mechanism, num_bidders=num_bidders, num_goods=num_goods, segments=segments,
        participation_probability=float(segment_mass.sum()), scoring_rule=scoring_rule,
        cell_participation_mass=mass, maximum_bid=float(np.max(bid_end)),
    )
