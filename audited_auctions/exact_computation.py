"""
exact_computation.py

Exact expected outcomes of score-based auctions with exchangeable bidders and a
common threshold function τ, for a discrete type distribution (see
type_distributions.py).

Setting (as in the WEB27 core draft): n bidders with i.i.d. types (e, v) ~ H,
k identical goods, unit demand. Under the participant-truthful threshold (TPT)
strategy a bidder participates iff v >= τ(e) and then bids v.

Notation used in this file
--------------------------
bid levels b_g            distinct bids of participating atoms, increasing
level mass μ_g            Pr[participate and bid = b_g]
welfare mass              E[(v + e) 1{participate, bid = b_g}]
expected welfare
  contribution ŵ^τ(b_g)   welfare mass / μ_g = b_g + E[e | v = b_g, v >= τ(e)]
ironed welfare
  contribution ω̄^τ        slope of the lower convex hull of the cumulative
                          welfare mass as a function of cumulative mass
interim allocation X(b)   probability that a participant bidding b is
                          allocated, given rivals' TPT strategies
allocation integral R(b)  ∫_0^b X(t) dt

Facts used (from the core draft)
--------------------------------
* Penalty inducing τ in a normalized DSIC auction (Thm on DSIC penalties):
      r(e) = R(τ(e)),  with τ clipped to [0, v̄]; r(e) = R(v̄) excludes e.
* Critical-value payment of a normalized DSIC auction:
      p(b) = b X(b) - R(b).
* Expected welfare: n Σ_g X(b_g) · welfare mass_g, because bidders are
  independent, so E[(v + e) x | bid] = ŵ^τ(bid) X(bid).

Interim allocation with ties
----------------------------
A rival participates and has a score above s with probability a(s), and a score
equal to s with probability z(s). With A rivals above and Z rivals tied, a
participant with positive score s is allocated with probability
min(1, (k - A) / (Z + 1)) if A < k, and 0 otherwise. So
      X(s) = Σ_{A<k} Bin(A; n-1, a) Σ_Z Bin(Z; n-1-A, z/(1-a)) min(1, (k-A)/(Z+1)).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from .scoring_rules import IdentityScore, StepScore


# ─── Ironing ──────────────────────────────────────────────────────────────────


def lower_convex_hull_vertices(x_values, y_values):
    """Indices of the vertices of the lower convex hull of points sorted by x.

    x_values must be strictly increasing. Collinear middle points are dropped,
    so consecutive hull segments have strictly increasing slopes.
    """
    hull = []
    for index in range(len(x_values)):
        while len(hull) >= 2:
            origin, middle = hull[-2], hull[-1]
            cross_product = (x_values[middle] - x_values[origin]) * (
                y_values[index] - y_values[origin]
            ) - (y_values[middle] - y_values[origin]) * (
                x_values[index] - x_values[origin]
            )
            if cross_product <= 0:  # middle point is on or above the chord
                hull.pop()
            else:
                break
        hull.append(index)
    return hull


def ironed_welfare_contributions(level_mass, level_welfare_mass):
    """Ironed welfare contribution ω̄^τ at each bid level.

    Works in quantile space: cumulative mass on the x-axis, cumulative welfare
    mass (Ψ in the core draft, up to the factor 1/q) on the y-axis. The ironed
    welfare contribution of a level is the slope of the lower convex hull over
    that level's quantile interval. Levels pooled by ironing receive exactly the
    same floating-point score, so they tie in the auction.
    """
    level_mass = np.asarray(level_mass, dtype=float)
    cumulative_mass = np.concatenate([[0.0], np.cumsum(level_mass)])
    cumulative_welfare = np.concatenate([[0.0], np.cumsum(level_welfare_mass)])
    hull = lower_convex_hull_vertices(cumulative_mass, cumulative_welfare)

    ironed = np.empty(len(level_mass))
    for left_vertex, right_vertex in zip(hull[:-1], hull[1:]):
        slope = (cumulative_welfare[right_vertex] - cumulative_welfare[left_vertex]) / (
            cumulative_mass[right_vertex] - cumulative_mass[left_vertex]
        )
        ironed[left_vertex:right_vertex] = slope
    return ironed


# ─── Interim allocation ───────────────────────────────────────────────────────


def interim_allocation_for_scores(
    own_scores, rival_scores, rival_mass, num_bidders, num_goods
):
    """Interim allocation X(s) of a participant with score s.

    own_scores  : (M,) scores at which to evaluate
    rival_scores: (G,) score of each rival bid level (any order, may repeat)
    rival_mass  : (G,) probability that a rival participates at that level
                  (the remaining probability is abstention)
    """
    own_scores = np.atleast_1d(np.asarray(own_scores, dtype=float))
    # Group rival levels with equal scores (pooled levels tie).
    unique_scores, inverse = np.unique(rival_scores, return_inverse=True)
    unique_mass = np.bincount(inverse, weights=rival_mass, minlength=len(unique_scores))
    mass_at_or_above = np.concatenate([np.cumsum(unique_mass[::-1])[::-1], [0.0]])

    first_above = np.searchsorted(unique_scores, own_scores, side="right")
    first_at_or_above = np.searchsorted(unique_scores, own_scores, side="left")
    # Clip away floating-point overshoot of cumulative sums (binom needs p in [0, 1]).
    probability_above = np.clip(mass_at_or_above[first_above], 0.0, 1.0)
    probability_tied = np.clip(
        mass_at_or_above[first_at_or_above] - probability_above, 0.0, 1.0
    )

    num_rivals = num_bidders - 1
    probability_tied_given_not_above = np.where(
        probability_above < 1.0,
        probability_tied / np.maximum(1.0 - probability_above, 1e-300),
        0.0,
    )
    probability_tied_given_not_above = np.clip(probability_tied_given_not_above, 0, 1)

    allocation = np.zeros_like(own_scores)
    for num_above in range(min(num_goods, num_rivals + 1)):
        probability_num_above = binom.pmf(num_above, num_rivals, probability_above)
        remaining_rivals = num_rivals - num_above
        expected_share = np.zeros_like(own_scores)
        for num_tied in range(remaining_rivals + 1):
            share = min(1.0, (num_goods - num_above) / (num_tied + 1))
            expected_share += share * binom.pmf(
                num_tied, remaining_rivals, probability_tied_given_not_above
            )
        allocation += probability_num_above * expected_share

    # Only positive scores are allocated.
    return np.where(own_scores > 0, allocation, 0.0)


def allocation_integral_from_table(table, bids):
    """Evaluate R(b) = ∫_0^b X(t) dt from ScoreAuctionModel.allocation_integral_table().

    X is constant on [breakpoints[j], breakpoints[j+1]) (the last interval is
    unbounded), so R is linear there.
    """
    bids = np.asarray(bids, dtype=float)
    breakpoints = table["breakpoints"]
    clipped = np.maximum(bids, 0.0)
    interval = np.searchsorted(breakpoints, clipped, side="right") - 1
    integral = table["integral_at_breakpoints"][interval] + table[
        "interval_allocation"
    ][interval] * (clipped - breakpoints[interval])
    return np.where(bids > 0, integral, 0.0)


# ─── Score-based auction model ────────────────────────────────────────────────


@dataclass
class ScoreAuctionModel:
    """Exact model of one score-based auction under TPT strategies.

    All level arrays are indexed by the participating bid levels g.
    """

    mechanism: str
    num_bidders: int
    num_goods: int
    bid_levels: np.ndarray
    level_mass: np.ndarray
    level_valuation_mass: np.ndarray
    level_externality_mass: np.ndarray
    level_scores: np.ndarray
    scoring_rule: object
    atom_level_index: np.ndarray  # level of each atom; -1 for abstaining atoms

    def __post_init__(self):
        self.level_allocation = self.interim_allocation_of_scores(self.level_scores)
        self._build_allocation_integral()

    # Interim allocation and its integral -------------------------------------

    def interim_allocation_of_scores(self, scores):
        return interim_allocation_for_scores(
            scores, self.level_scores, self.level_mass, self.num_bidders, self.num_goods
        )

    def interim_allocation_of_bids(self, bids):
        """X(b) for arbitrary nonnegative bids b (not only bid levels)."""
        return self.interim_allocation_of_scores(self.scoring_rule.score_of_bid(bids))

    def _build_allocation_integral(self):
        """Tabulate R(b) = ∫_0^b X(t) dt at the breakpoints of X.

        X is constant on each open interval between consecutive positive bid
        levels (for StepScore it is constant on [b_g, b_{g+1})), so R is
        piecewise linear with these breakpoints.
        """
        positive_levels = self.bid_levels[self.bid_levels > 0]
        self._breakpoints = np.unique(np.concatenate([[0.0], positive_levels]))
        interval_midpoints = np.append(
            (self._breakpoints[:-1] + self._breakpoints[1:]) / 2,
            self._breakpoints[-1] + 1.0,  # representative bid of the last interval
        )
        self._interval_allocation = self.interim_allocation_of_bids(interval_midpoints)
        interval_lengths = np.diff(self._breakpoints)
        self._integral_at_breakpoints = np.concatenate(
            [[0.0], np.cumsum(self._interval_allocation[:-1] * interval_lengths)]
        )

    def allocation_integral(self, bids):
        """R(b) = ∫_0^b X(t) dt; R(b) = 0 for b <= 0."""
        return allocation_integral_from_table(self.allocation_integral_table(), bids)

    def allocation_integral_table(self) -> dict:
        """Arrays that determine R(b) exactly (small; stored in sweep results)."""
        return {
            "breakpoints": self._breakpoints,
            "integral_at_breakpoints": self._integral_at_breakpoints,
            "interval_allocation": self._interval_allocation,
        }

    def critical_value_payment(self, bids):
        """Expected payment p(b) = b X(b) - R(b) of the normalized DSIC auction."""
        bids = np.asarray(bids, dtype=float)
        payment = bids * self.interim_allocation_of_bids(bids) - self.allocation_integral(bids)
        return np.where(bids > 0, payment, 0.0)

    # Expected outcomes (all n bidders together) -------------------------------

    @property
    def participation_probability(self) -> float:
        return float(self.level_mass.sum())

    def expected_outcomes(self) -> dict:
        n = self.num_bidders
        level_payment = np.where(
            self.bid_levels > 0,
            self.bid_levels * self.level_allocation
            - self.allocation_integral(self.bid_levels),
            0.0,
        )
        valuation_welfare = n * np.sum(self.level_allocation * self.level_valuation_mass)
        externality_welfare = n * np.sum(
            self.level_allocation * self.level_externality_mass
        )
        return {
            "welfare": valuation_welfare + externality_welfare,
            "valuation_welfare": valuation_welfare,
            "externality_welfare": externality_welfare,
            "revenue": n * np.sum(self.level_mass * level_payment),
            "num_allocated": n * np.sum(self.level_mass * self.level_allocation),
            "participation_probability": self.participation_probability,
            "num_participants": n * self.participation_probability,
        }

    def atom_allocation(self) -> np.ndarray:
        """Interim allocation of each atom (0 for abstaining atoms)."""
        safe_index = np.maximum(self.atom_level_index, 0)
        return np.where(
            self.atom_level_index >= 0, self.level_allocation[safe_index], 0.0
        )

    def expected_welfare_contribution(self) -> np.ndarray:
        """ŵ^τ at each bid level (welfare mass / level mass)."""
        return (self.level_valuation_mass + self.level_externality_mass) / self.level_mass


def build_score_auction_model(
    mechanism,
    distribution,
    atom_participates,
    atom_bids,
    ironed,
    num_bidders,
    num_goods,
):
    """Aggregate participating atoms by bid level and build a ScoreAuctionModel.

    mechanism         : label stored in the model
    distribution      : TypeDistribution
    atom_participates : (num_atoms,) bool; participation of each atom
    atom_bids         : (num_atoms,) bid of each atom when participating
    ironed            : False -> score = bid (VCG ranking);
                        True  -> score = ironed welfare contribution (IVCG)
    """
    participating = np.flatnonzero(atom_participates)
    if len(participating) == 0:
        raise ValueError(f"{mechanism}: no type participates under this threshold")

    bid_levels, level_of_participant = np.unique(
        atom_bids[participating], return_inverse=True
    )
    probabilities = distribution.probabilities[participating]
    num_levels = len(bid_levels)
    level_mass = np.bincount(level_of_participant, probabilities, num_levels)
    level_valuation_mass = np.bincount(
        level_of_participant,
        probabilities * distribution.valuations[participating],
        num_levels,
    )
    level_externality_mass = np.bincount(
        level_of_participant,
        probabilities * distribution.externalities[participating],
        num_levels,
    )

    atom_level_index = np.full(distribution.num_atoms, -1, dtype=int)
    atom_level_index[participating] = level_of_participant

    if ironed:
        level_scores = ironed_welfare_contributions(
            level_mass, level_valuation_mass + level_externality_mass
        )
        scoring_rule = StepScore(bid_levels, level_scores)
    else:
        level_scores = bid_levels.copy()
        scoring_rule = IdentityScore()

    return ScoreAuctionModel(
        mechanism=mechanism,
        num_bidders=num_bidders,
        num_goods=num_goods,
        bid_levels=bid_levels,
        level_mass=level_mass,
        level_valuation_mass=level_valuation_mass,
        level_externality_mass=level_externality_mass,
        level_scores=level_scores,
        scoring_rule=scoring_rule,
        atom_level_index=atom_level_index,
    )
