"""
auction_simulation.py

Monte Carlo simulation of score-based auctions on sampled type profiles.

Each simulated draw is one auction: n bidders with types (e_i, v_i). For each
mechanism the simulation computes, per draw,

  allocation x_i   probability that bidder i is allocated, averaging over the
                   uniform tie-breaking lottery (so welfare per draw is the
                   expectation over that lottery only)
  payment p_i      critical-value payment b_i x_i(b) - ∫_0^{b_i} x_i(t, b_-i) dt,
                   the DSIC payment given the realized rival bids

and from these the per-draw metrics listed in DRAW_METRICS.

Allocation rule (any score-based auction): a participant with score s > 0 is
allocated with probability min(1, (k - A)/(Z + 1)) when A < k rivals have
higher scores and Z rivals have equal scores, and 0 otherwise.

Payment rule: let c be the k-th highest rival score (-inf if fewer than k
rivals participate). As a function of its own bid t, bidder i is allocated
with probability 1 when score(t) > max(c, 0), with the tie share when
score(t) = c > 0, and 0 otherwise. The integral of this step function uses the
generalized inverse of the scoring rule (scoring_rules.py).
"""

import numpy as np

DRAW_METRICS = (
    "welfare",  # Σ_i x_i (v_i + e_i)
    "valuation_welfare",  # Σ_i x_i v_i
    "externality_welfare",  # Σ_i x_i e_i
    "revenue",  # Σ_i p_i (auction payments only)
    "penalty_paid",  # participant audit: Σ_{participants} r(e_i); recipient audit: Σ_i x_i (-e_i)
    "bidder_utility",  # Σ_i (v_i x_i − p_i − penalty_i): bidders' net utility
    "num_participants",
    "num_allocated",  # Σ_i x_i
)


def allocation_probabilities(scores, num_goods):
    """Allocation probability of each bidder in each draw; scores -inf = abstain.

    scores : (num_draws, num_bidders)
    """
    # higher[d, i] = number of bidders j with scores[d, j] > scores[d, i]
    higher = (scores[:, None, :] > scores[:, :, None]).sum(axis=2)
    tied = (scores[:, None, :] == scores[:, :, None]).sum(axis=2) - 1
    share = np.minimum(1.0, (num_goods - higher) / (tied + 1.0))
    return np.where((scores > 0) & (higher < num_goods), share, 0.0)


def critical_value_payments(bids, scores, allocation, scoring_rule, num_goods):
    """Critical-value payment of each bidder given realized rival bids.

    bids, scores, allocation : (num_draws, num_bidders); scores -inf = abstain
    """
    num_draws, num_bidders = scores.shape
    # Sorted scores, padded with -inf so that position k always exists.
    padded = np.concatenate(
        [scores, np.full((num_draws, num_goods + 1), -np.inf)], axis=1
    )
    sorted_descending = -np.sort(-padded, axis=1)
    kth_highest = sorted_descending[:, num_goods - 1][:, None]
    next_highest = sorted_descending[:, num_goods][:, None]
    # Removing bidder i shifts position k only if bidder i is among the top k.
    rival_kth_score = np.where(scores >= kth_highest, next_highest, kth_highest)

    # Tie share at the rival k-th score, counted among rivals only.
    rivals_above = (scores[:, None, :] > rival_kth_score[:, :, None]).sum(axis=2) - (
        scores > rival_kth_score
    )
    rivals_tied = (scores[:, None, :] == rival_kth_score[:, :, None]).sum(axis=2) - (
        scores == rival_kth_score
    )
    tie_share = np.clip((num_goods - rivals_above) / (rivals_tied + 1.0), 0.0, 1.0)

    cutoff_positive = rival_kth_score > 0
    reserve_or_cutoff = np.where(cutoff_positive, rival_kth_score, 0.0)
    full_allocation_start = np.maximum(
        scoring_rule.smallest_bid_with_score_above(reserve_or_cutoff), 0.0
    )
    integral = np.maximum(bids - full_allocation_start, 0.0)

    safe_cutoff = np.where(cutoff_positive, rival_kth_score, 1.0)
    tie_start = np.maximum(scoring_rule.smallest_bid_with_score_at_least(safe_cutoff), 0.0)
    tie_end = scoring_rule.smallest_bid_with_score_above(safe_cutoff)
    tie_length = np.maximum(np.minimum(bids, tie_end) - tie_start, 0.0)
    integral = integral + np.where(cutoff_positive, tie_share * tie_length, 0.0)

    payments = bids * allocation - integral
    return np.where(np.isfinite(scores) & (bids > 0), payments, 0.0)


def simulate_score_auction(
    externalities,
    valuations,
    participates,
    bids,
    scoring_rule,
    num_goods,
    penalty_per_bidder,
    chunk_size=4000,
):
    """Simulate one score-based auction on all draws; return dict metric -> (num_draws,).

    externalities, valuations, bids : (num_draws, num_bidders)
    participates       : (num_draws, num_bidders) bool
    penalty_per_bidder : participant audit: (num_draws, num_bidders) array of
                         penalties r(e_i), charged to participants only;
                         recipient audit: the string "recipient", which charges
                         -e_i to each winner in proportion to its allocation.
    """
    num_draws = externalities.shape[0]
    metrics = {name: np.empty(num_draws) for name in DRAW_METRICS}

    for start in range(0, num_draws, chunk_size):
        rows = slice(start, min(start + chunk_size, num_draws))
        draw_bids = bids[rows]
        scores = np.where(
            participates[rows], scoring_rule.score_of_bid(draw_bids), -np.inf
        )
        allocation = allocation_probabilities(scores, num_goods)
        payments = critical_value_payments(
            draw_bids, scores, allocation, scoring_rule, num_goods
        )
        metrics["valuation_welfare"][rows] = (allocation * valuations[rows]).sum(axis=1)
        metrics["externality_welfare"][rows] = (allocation * externalities[rows]).sum(
            axis=1
        )
        metrics["revenue"][rows] = payments.sum(axis=1)
        metrics["num_participants"][rows] = participates[rows].sum(axis=1)
        metrics["num_allocated"][rows] = allocation.sum(axis=1)
        if isinstance(penalty_per_bidder, str) and penalty_per_bidder == "recipient":
            metrics["penalty_paid"][rows] = (allocation * -externalities[rows]).sum(axis=1)
        else:
            metrics["penalty_paid"][rows] = np.where(
                participates[rows], penalty_per_bidder[rows], 0.0
            ).sum(axis=1)

    metrics["welfare"] = metrics["valuation_welfare"] + metrics["externality_welfare"]
    metrics["bidder_utility"] = (metrics["valuation_welfare"] - metrics["revenue"]
                                 - metrics["penalty_paid"])
    return metrics
