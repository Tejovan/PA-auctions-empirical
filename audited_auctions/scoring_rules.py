"""
scoring_rules.py

Scoring rules of score-based auctions. A score-based auction with k identical
goods and unit demand allocates to the participants with the k highest scores,
allocates only to positive scores, and breaks ties uniformly at random.

  * VCG and VCG-PA rank participants by bid:        score(b) = b.
  * IVCG-PA ranks by ironed welfare contribution:   score(b) = ω̄^τ(b), a
    nondecreasing function of the bid (see exact_computation.py): linear with
    slope one where not ironed, constant on ironing (pooling) intervals.
  * The first-best auction (recipient audit with penalty −e) ranks all bidders
    by effective bid w = v + e:                     score(w) = w.

Payments are critical-value (Myerson) payments, so each auction is DSIC. They
need the generalized inverse of the score function, which each scoring rule
provides through two methods:

  smallest_bid_with_score_above(c)    = inf{ b : score(b) >  c }
  smallest_bid_with_score_at_least(c) = inf{ b : score(b) >= c }

(+inf when no bid reaches the score).
"""

import numpy as np


class IdentityScore:
    """score(b) = b."""

    name = "identity"

    def score_of_bid(self, bids):
        return np.asarray(bids, dtype=float)

    def smallest_bid_with_score_above(self, score_values):
        return np.asarray(score_values, dtype=float)

    def smallest_bid_with_score_at_least(self, score_values):
        return np.asarray(score_values, dtype=float)


class PiecewiseLinearScore:
    """Nondecreasing, right-continuous, piecewise-linear score function of the bid.

    Piece i covers bids [knots[i], knots[i+1]) (the last piece [knots[-1], ∞)),
    on which the score rises linearly from start_scores[i] to end_scores[i]
    (the left limit at knots[i+1]); the last piece is constant. Bids below
    knots[0] score −inf (never allocated).

    This is the paper's convention for bids outside the support of the
    participating valuations: a bid in a gap of the support takes the score
    approached from below (the left limit of the score at the gap), and a bid
    below the support is never allocated. A step function (atomic
    distributions) is the special case start_scores = end_scores.
    """

    name = "piecewise_linear"

    def __init__(self, knots, start_scores, end_scores):
        self.knots = np.asarray(knots, dtype=float)
        self.start_scores = np.asarray(start_scores, dtype=float)
        self.end_scores = np.asarray(end_scores, dtype=float)
        if np.any(np.diff(self.knots) <= 0):
            raise ValueError("knots must be strictly increasing")
        if self.start_scores[-1] != self.end_scores[-1]:
            raise ValueError("the last piece must be constant")
        if np.any(self.end_scores < self.start_scores) or np.any(
            self.start_scores[1:] < self.end_scores[:-1]
        ):
            raise ValueError("score function must be nondecreasing")
        self._piece_lengths = np.append(np.diff(self.knots), np.inf)

    def score_of_bid(self, bids):
        bids = np.asarray(bids, dtype=float)
        piece = np.searchsorted(self.knots, bids, side="right") - 1
        safe = np.clip(piece, 0, len(self.knots) - 1)
        rise = self.end_scores[safe] - self.start_scores[safe]
        fraction = np.where(rise > 0,
                            (bids - self.knots[safe]) / np.where(np.isfinite(self._piece_lengths[safe]),
                                                                 self._piece_lengths[safe], 1.0),
                            0.0)
        score = self.start_scores[safe] + rise * fraction
        return np.where(piece >= 0, score, -np.inf)

    def left_limit(self, bids):
        """lim_{t ↑ b} score(t): the score approached from below (−inf at or below knots[0])."""
        bids = np.asarray(bids, dtype=float)
        piece = np.searchsorted(self.knots, bids, side="left") - 1
        safe = np.clip(piece, 0, len(self.knots) - 1)
        rise = self.end_scores[safe] - self.start_scores[safe]
        length = np.where(np.isfinite(self._piece_lengths[safe]), self._piece_lengths[safe], 1.0)
        score = self.start_scores[safe] + np.where(rise > 0, rise * (bids - self.knots[safe]) / length,
                                                   0.0)
        return np.where(piece >= 0, score, -np.inf)

    def _inverse(self, score_values, strict):
        """inf{b : score(b) > c} (strict) or inf{b : score(b) >= c}."""
        score_values = np.asarray(score_values, dtype=float)
        side = "right" if strict else "left"
        # First piece whose left-limit end score exceeds (reaches) c.
        piece = np.searchsorted(self.end_scores, score_values, side=side)
        has_piece = piece < len(self.knots)
        safe = np.minimum(piece, len(self.knots) - 1)
        start = self.start_scores[safe]
        starts_past = (start > score_values) if strict else (start >= score_values)
        rise = self.end_scores[safe] - start
        length = np.where(np.isfinite(self._piece_lengths[safe]), self._piece_lengths[safe], 0.0)
        interpolated = self.knots[safe] + np.where(
            rise > 0, (score_values - start) / np.where(rise > 0, rise, 1.0), 0.0) * length
        bid = np.where(starts_past, self.knots[safe], interpolated)
        return np.where(has_piece, bid, np.inf)

    def smallest_bid_with_score_above(self, score_values):
        return self._inverse(score_values, strict=True)

    def smallest_bid_with_score_at_least(self, score_values):
        return self._inverse(score_values, strict=False)
