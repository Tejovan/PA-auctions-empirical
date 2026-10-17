"""
Tests for the audited_auctions package. Run from WEB27/code with

    .venv/bin/python -m pytest tests -q

Ground truths:
  * Example T of the core draft (ironing, IVCG-PA welfare 40/16, first best 41/16).
  * Second-price auction, two bidders, uniform valuations: a constant threshold
    τ is induced by the penalty τ² (core draft, example after Thm 3.3).
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
    polynomial_threshold,
    population_distribution,
    synthetic_distribution,
)
from audited_auctions.auction_simulation import (
    allocation_probabilities,
    critical_value_payments,
)
from audited_auctions.exact_computation import ironed_welfare_contributions


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


def uniform_valuation_distribution(num_valuation_nodes=2001):
    # Valuations uniform on [0, 1], externality identically 0.
    return gridded_density_distribution(
        lambda e, v: np.ones_like(v), [0.0], np.linspace(0, 1, num_valuation_nodes)
    )


@pytest.mark.parametrize("threshold_value", [0.3, 0.6, 0.9])
def test_second_price_constant_threshold_penalty(threshold_value):
    evaluation = evaluate_threshold(
        uniform_valuation_distribution(), threshold_value, num_bidders=2, num_goods=1
    )
    expected_penalty = threshold_value**2
    np.testing.assert_allclose(
        evaluation.penalty_curves["vcg_pa"], expected_penalty, atol=2e-3
    )


@pytest.mark.parametrize(
    "distribution_factory, threshold, num_bidders, num_goods",
    [
        (lambda: synthetic_distribution("c", num_externality_nodes=121, num_valuation_nodes=241),
         polynomial_threshold([0.3, -0.8, 0.5]), 4, 2),
        (lambda: synthetic_distribution("h", num_externality_nodes=121, num_valuation_nodes=241),
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
    distribution = synthetic_distribution("c", num_externality_nodes=121, num_valuation_nodes=241)
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
    distribution = synthetic_distribution(letter, num_externality_nodes=121, num_valuation_nodes=241)
    rng = np.random.default_rng(7)
    for _ in range(5):
        threshold = polynomial_threshold(rng.normal(0, 1, 3))
        try:
            evaluation = evaluate_threshold(distribution, threshold, 5, 2)
        except ValueError:  # nobody participates
            continue
        assert evaluation.exact["ivcg_pa"]["welfare"] >= evaluation.exact["vcg_pa"]["welfare"] - 1e-12
        assert evaluation.exact["first_best"]["welfare"] >= evaluation.exact["ivcg_pa"]["welfare"] - 1e-12
