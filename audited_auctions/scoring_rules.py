"""
scoring_rules.py

Scoring rules of score-based auctions. A score-based auction with k identical
goods and unit demand allocates to the participants with the k highest scores,
allocates only to positive scores, and breaks ties uniformly at random.

  * VCG-PA ranks participants by bid:            score(b) = b.
  * IVCG-PA ranks by ironed welfare contribution: score(b) = ω̄^τ(b), a
    nondecreasing step function of the bid (see exact_computation.py).
  * The first-best auction (recipient audit with penalty -e) ranks all bidders
    by effective bid w = v + e:                  score(w) = w.

Payments are critical-value (Myerson) payments, so each auction is DSIC. They
need the generalized inverse of the score function, which each scoring rule
provides through two methods:

  smallest_bid_with_score_above(c)    = inf{ b >= 0 ... : score(b) >  c }
  smallest_bid_with_score_at_least(c) = inf{ b : score(b) >= c }
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


class StepScore:
    """Nondecreasing step score: score(b) = level_scores[g] for b in [bid_levels[g], bid_levels[g+1]).

    Bids below bid_levels[0] score -inf (never allocated). This is the paper's
    convention for bids outside the support of the participating valuation
    distribution: a bid takes the score of the largest supported valuation
    that does not exceed it.
    """

    name = "step"

    def __init__(self, bid_levels, level_scores):
        self.bid_levels = np.asarray(bid_levels, dtype=float)
        self.level_scores = np.asarray(level_scores, dtype=float)
        if np.any(np.diff(self.level_scores) < 0):
            raise ValueError("level_scores must be nondecreasing")
        # Sentinel bid +inf for "no bid achieves this score".
        self._bid_levels_with_sentinel = np.append(self.bid_levels, np.inf)

    def score_of_bid(self, bids):
        bids = np.asarray(bids, dtype=float)
        level_index = np.searchsorted(self.bid_levels, bids, side="right") - 1
        safe_index = np.clip(level_index, 0, len(self.level_scores) - 1)
        return np.where(level_index >= 0, self.level_scores[safe_index], -np.inf)

    def smallest_bid_with_score_above(self, score_values):
        first_level = np.searchsorted(self.level_scores, score_values, side="right")
        return self._bid_levels_with_sentinel[first_level]

    def smallest_bid_with_score_at_least(self, score_values):
        first_level = np.searchsorted(self.level_scores, score_values, side="left")
        return self._bid_levels_with_sentinel[first_level]
