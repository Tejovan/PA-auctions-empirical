"""
Tests for the audited_auctions package. Run from WEB27/code with

    .venv/bin/python -m pytest tests -q

Ground truths:
  * Example T of the paper, Ex. 6.4 [ex:T-pool] (ironing, IVCG-PA welfare 40/16, first best 41/16).
  * Second-price auction, two bidders, uniform valuations: a constant threshold
    τ is induced by the penalty τ² (paper Ex. 5.4 [ex:first-price]).
  * Example N of the paper, Ex. 7.11 [ex:N]: VCG-PA welfare 2/3 − c/2 + (c/2)t² − t³/4 and
    penalty (3/4)t² (piecewise-uniform distribution, exact).
  * Ironing of piecewise-linear welfare contributions agrees with ironing of a
    fine atomic discretization.
  * Exact expectations agree with simulated means (all mechanisms, all metrics).
  * A type at the threshold is indifferent: its simulated expected
    auction-utility equals the computed penalty r(e).
  * IVCG-PA welfare >= VCG-PA welfare for the same τ.
"""

import numpy as np
import pytest

from audited_auctions import (
    MECHANISMS,
    build_mechanism_models,
    evaluate_threshold,
    gridded_density_distribution,
    penalty_from_table,
    polynomial_threshold,
    population_distribution,
    synthetic_distribution,
)
from audited_auctions.auction_simulation import (
    allocation_probabilities,
    critical_value_payments,
)
from audited_auctions.exact_computation import (
    iron_quantile_function,
    ironed_welfare_contributions,
)
from audited_auctions.scoring_rules import PiecewiseLinearScore


def example_t_distribution():
    # Types (e, v): A=(0,2), B=(-2,3), C=(0,4), D=(-10,1), each with probability 1/4.
    return population_distribution([0, -2, 0, -10], [2, 3, 4, 1], name="Example T")


def example_t_threshold(externalities):
    # Excludes D (e = -10); admits A, B, C at every valuation.
    return np.where(np.asarray(externalities) < -5, np.inf, 0.0)


def test_ironing_pools_example_t():
    ironed = ironed_welfare_contributions([1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 3, 4 / 3])
    np.testing.assert_allclose(ironed, [1.5, 1.5, 4.0])
    assert ironed[0] == ironed[1]  # pooled levels tie exactly


def test_example_t_welfare():
    evaluation = evaluate_threshold(example_t_distribution(), example_t_threshold, 2, 1)
    assert evaluation.exact["ivcg_pa"]["welfare"] == pytest.approx(40 / 16)
    assert evaluation.exact["first_best"]["welfare"] == pytest.approx(41 / 16)
    assert evaluation.exact["vcg_pa"]["welfare"] <= 39 / 16 + 1e-12


def uniform_valuation_distribution(num_valuation_cells=10):
    # Valuations uniform on [0, 1], externality identically 0 (exact: piecewise uniform).
    return gridded_density_distribution(
        lambda e, v: np.ones_like(v), [0.0], np.linspace(0, 1, num_valuation_cells + 1)
    )


@pytest.mark.parametrize("threshold_value", [0.3, 0.6, 0.9])
def test_second_price_constant_threshold_penalty(threshold_value):
    evaluation = evaluate_threshold(
        uniform_valuation_distribution(), threshold_value, num_bidders=2, num_goods=1
    )
    expected_penalty = threshold_value**2
    np.testing.assert_allclose(
        evaluation.penalty_curves["vcg_pa"], expected_penalty, rtol=1e-12
    )


@pytest.mark.parametrize(
    "distribution_factory, threshold, num_bidders, num_goods",
    [
        (lambda: synthetic_distribution("c", num_externality_nodes=121, num_valuation_cells=240),
         polynomial_threshold([0.3, -0.8, 0.5]), 4, 2),
        (lambda: synthetic_distribution("h", num_externality_nodes=121, num_valuation_cells=240),
         polynomial_threshold([2.0, -1.0]), 6, 1),
        # A small population: valuations repeat, so ties have positive probability.
        (lambda: population_distribution(
            np.random.default_rng(1).normal(0, 1, 25).round(1),
            np.random.default_rng(2).integers(1, 6, 25).astype(float)),
         polynomial_threshold([2.0, -1.5]), 5, 2),
        (example_t_distribution, example_t_threshold, 2, 1),
    ],
)
def test_exact_matches_simulation(distribution_factory, threshold, num_bidders, num_goods):
    evaluation = evaluate_threshold(
        distribution_factory(), threshold, num_bidders, num_goods,
        num_simulated_draws=60000, random_seed=3,
    )
    for mechanism in MECHANISMS:
        for metric in ("welfare", "revenue", "penalty_paid", "num_participants", "num_allocated"):
            draws = evaluation.simulated[mechanism][metric]
            standard_error = draws.std(ddof=1) / np.sqrt(len(draws))
            exact_value = evaluation.exact[mechanism][metric]
            assert abs(draws.mean() - exact_value) <= 5 * standard_error + 1e-9, (
                mechanism, metric, draws.mean(), exact_value
            )


@pytest.mark.parametrize("mechanism", ["vcg_pa", "ivcg_pa"])
def test_threshold_type_is_indifferent(mechanism):
    """Simulated auction-utility of a threshold type equals the computed penalty."""
    distribution = synthetic_distribution("c", num_externality_nodes=121, num_valuation_cells=240)
    threshold = polynomial_threshold([0.4, -0.5])
    num_bidders, num_goods = 4, 1
    models, _ = build_mechanism_models(distribution, threshold, num_bidders, num_goods)
    model = models[mechanism]
    rng = np.random.default_rng(5)

    for externality in (-1.0, -0.3, 0.2):
        threshold_valuation = float(threshold(np.array(externality)))
        externalities, valuations = distribution.sample_types(100000, num_bidders, rng)
        externalities[:, 0], valuations[:, 0] = externality, threshold_valuation
        participates = valuations >= threshold(externalities)
        participates[:, 0] = True
        # Bidder 0 has the threshold type; its auction-utility is v x_0 - p_0.
        scores = np.where(participates, model.scoring_rule.score_of_bid(valuations), -np.inf)
        allocation = allocation_probabilities(scores, num_goods)
        payments = critical_value_payments(valuations, scores, allocation, model.scoring_rule, num_goods)
        utility = threshold_valuation * allocation[:, 0] - payments[:, 0]
        penalty = model.allocation_integral(np.array([max(threshold_valuation, 0.0)]))[0]
        standard_error = utility.std(ddof=1) / np.sqrt(len(utility))
        assert abs(utility.mean() - penalty) <= 5 * standard_error + 1e-9


@pytest.mark.parametrize("letter", ["b", "c", "d", "f"])
def test_ironing_never_lowers_welfare(letter):
    distribution = synthetic_distribution(letter, num_externality_nodes=121, num_valuation_cells=240)
    rng = np.random.default_rng(7)
    for _ in range(5):
        threshold = polynomial_threshold(rng.normal(0, 1, 3))
        try:
            evaluation = evaluate_threshold(distribution, threshold, 5, 2)
        except ValueError:  # nobody participates
            continue
        assert evaluation.exact["ivcg_pa"]["welfare"] >= evaluation.exact["vcg_pa"]["welfare"] - 1e-12
        assert evaluation.exact["first_best"]["welfare"] >= evaluation.exact["ivcg_pa"]["welfare"] - 1e-12


@pytest.mark.parametrize("harm", [0.3, 0.6])
@pytest.mark.parametrize("threshold_value", [0.0, 0.25, 0.8])
def test_example_n_welfare_and_penalty(harm, threshold_value):
    distribution = synthetic_distribution("example_n", harm=harm, num_valuation_cells=7)

    def threshold(externalities):
        return np.where(np.asarray(externalities) < 0, threshold_value, -np.inf)

    evaluation = evaluate_threshold(distribution, threshold, num_bidders=2, num_goods=1)
    expected_welfare = 2 / 3 - harm / 2 + harm / 2 * threshold_value**2 - threshold_value**3 / 4
    assert evaluation.exact["vcg_pa"]["welfare"] == pytest.approx(expected_welfare, rel=1e-12)
    penalty = penalty_from_table(evaluation.penalty_tables["vcg_pa"], np.array([threshold_value]))
    assert penalty[0] == pytest.approx(0.75 * threshold_value**2, abs=1e-12)


def test_continuous_ironing_matches_fine_atomic_ironing():
    """Exact ironing of piecewise-linear ŵ^τ agrees with ironing many small atoms."""
    rng = np.random.default_rng(11)
    edges = np.sort(np.concatenate([[0.0, 1.0], rng.random(6)]))
    mass = rng.random(7)
    jumps = rng.normal(0, 0.6, 7)  # E[e | bid] per interval: ŵ jumps down and up
    quantile_end = np.cumsum(mass)
    quantile_start = quantile_end - mass
    pieces = iron_quantile_function(quantile_start, quantile_end, edges[:-1] + jumps,
                                    edges[1:] + jumps)
    # Fine atomic version: split each interval into many atoms of equal mass.
    atoms = 4000
    fine_mass = np.repeat(mass / atoms, atoms)
    fine_values = np.concatenate([
        np.linspace(a, b, atoms, endpoint=False) + (b - a) / (2 * atoms) + j
        for a, b, j in zip(edges[:-1], edges[1:], jumps)])
    fine_ironed = ironed_welfare_contributions(fine_mass, fine_mass * fine_values)
    fine_quantiles = np.cumsum(fine_mass) - fine_mass / 2
    piece_index = np.minimum(np.searchsorted(pieces[1], fine_quantiles), len(pieces[1]) - 1)
    span = pieces[1][piece_index] - pieces[0][piece_index]
    fraction = (fine_quantiles - pieces[0][piece_index]) / span
    exact = pieces[2][piece_index] + (pieces[3][piece_index] - pieces[2][piece_index]) * fraction
    assert np.max(np.abs(exact - fine_ironed)) < 2e-3
    assert np.all(np.diff(np.concatenate([[p0, p1] for p0, p1 in zip(pieces[2], pieces[3])])) >= -1e-12)


def test_harmful_middle_irons_and_ironing_helps():
    distribution = synthetic_distribution("harmful_middle", num_externality_nodes=121,
                                          num_valuation_cells=240)
    evaluation = evaluate_threshold(distribution, -np.inf, num_bidders=4, num_goods=1)
    curves = evaluation.welfare_contribution_curves
    pooled = (curves["score_end"] == curves["score_start"]) & (
        curves["welfare_end"] > curves["welfare_start"])
    assert pooled.any()
    assert evaluation.exact["ivcg_pa"]["welfare"] > evaluation.exact["vcg_pa"]["welfare"] + 1e-3


def test_piecewise_linear_score_inverses():
    score = PiecewiseLinearScore([0.0, 1.0, 2.0, 3.0], [0.5, 1.0, 1.0, 4.0], [1.0, 1.0, 3.0, 4.0])
    bids = np.array([-0.5, 0.0, 0.5, 1.5, 2.5, 3.0, 9.0])
    np.testing.assert_allclose(score.score_of_bid(bids), [-np.inf, 0.5, 0.75, 1.0, 2.0, 4.0, 4.0])
    levels = np.array([0.0, 0.75, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0])
    np.testing.assert_allclose(score.smallest_bid_with_score_above(levels),
                               [0.0, 0.5, 2.0, 2.5, 3.0, 3.0, np.inf, np.inf])
    np.testing.assert_allclose(score.smallest_bid_with_score_at_least(levels),
                               [0.0, 0.5, 1.0, 2.5, 3.0, 3.0, 3.0, np.inf])


def test_negative_valuations_are_rejected():
    with pytest.raises(ValueError):
        population_distribution([0.0, 1.0], [-1.0, 2.0])
