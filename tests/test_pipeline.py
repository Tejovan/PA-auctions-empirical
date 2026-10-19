"""
Tests of the distribution families, kernel smoothing, the empirical pipeline and
the exact-objective genetic search. Run from WEB27/code with

    .venv/bin/python -m pytest tests -q
"""

import numpy as np
import pytest

from audited_auctions import SYNTHETIC_FAMILIES, synthetic_distribution
from audited_auctions.empirical import (
    EmpiricalSample,
    bootstrap_welfare,
    sample_distribution,
    split_sample,
)
from audited_auctions.genetic_search import run_genetic_search
from audited_auctions.pipeline import prepare_distributions
from audited_auctions.smoothing import kernel_density_distribution
from audited_auctions.threshold_functions import polynomial_threshold


@pytest.mark.parametrize("family", sorted(SYNTHETIC_FAMILIES))
def test_families_have_bounded_nonnegative_continuous_valuations(family):
    options = {} if family == "example_n" else {"num_externality_nodes": 41}
    distribution = synthetic_distribution(family, num_valuation_cells=60, **options)
    assert distribution.kind == "piecewise_uniform"
    assert distribution.valuation_lower.min() >= 0
    assert np.isfinite(distribution.maximum_valuation)
    assert distribution.probabilities.sum() == pytest.approx(1.0)


def lognormal_sample(num_rows=4000, seed=0):
    rng = np.random.default_rng(seed)
    log_valuations = rng.normal(0, 1, num_rows)
    externalities = -0.5 * np.exp(0.5 * log_valuations) + rng.normal(0, 0.3, num_rows)
    valuations = np.exp(log_valuations)
    bound = float(np.quantile(valuations, 0.995))
    keep = valuations <= bound
    return EmpiricalSample(externalities[keep], valuations[keep], bound, "synthetic.csv", {})


def test_variance_preserving_smoothing_keeps_moments():
    sample = lognormal_sample()
    distribution = kernel_density_distribution(sample.externalities, sample.valuations,
                                               sample.maximum_valuation)
    assert distribution.valuation_lower.min() > 0
    assert distribution.maximum_valuation == pytest.approx(sample.maximum_valuation)
    mean_valuation = np.sum(distribution.probabilities * distribution.valuation_midpoints)
    assert mean_valuation == pytest.approx(sample.valuations.mean(), rel=0.03)
    mean_externality = np.sum(distribution.probabilities * distribution.externalities)
    assert mean_externality == pytest.approx(sample.externalities.mean(), rel=0.05)


def test_split_is_deterministic_and_disjoint():
    sample = lognormal_sample(1000)
    train_a, test_a = split_sample(sample, 0.5, seed=3)
    train_b, _ = split_sample(sample, 0.5, seed=3)
    np.testing.assert_array_equal(train_a.valuations, train_b.valuations)
    assert train_a.num_rows + test_a.num_rows == sample.num_rows


def test_bootstrap_gains_are_paired():
    sample = lognormal_sample(1500)
    result = bootstrap_welfare(sample, polynomial_threshold([0.5, -1.0]), 4, 1, num_resamples=5,
                               seed=1, num_externality_nodes=61, num_valuation_cells=80)
    table = result.resamples
    assert len(table) == 5
    for mechanism in ("vcg_pa", "ivcg_pa", "first_best"):
        np.testing.assert_allclose(table[f"gain_{mechanism}"],
                                   table[f"welfare_{mechanism}"] - table["welfare_vcg"])
    assert (table["welfare_first_best"] >= table["welfare_ivcg_pa"] - 1e-12).all()
    assert {"mean", "sd"} <= set(result.summary().columns)


def test_atomic_and_smoothed_vcg_welfare_are_close():
    sample = lognormal_sample(6000)
    threshold = polynomial_threshold([-1.0])  # everyone participates
    from audited_auctions.empirical import exact_welfare_by_mechanism
    atomic = exact_welfare_by_mechanism(sample_distribution(sample, "none"), threshold, 8, 1)
    smoothed = exact_welfare_by_mechanism(sample_distribution(sample, "kernel"), threshold, 8, 1)
    assert smoothed["vcg"] == pytest.approx(atomic["vcg"], rel=0.03)


def test_genetic_search_recovers_example_n_optimum():
    """Example N: the optimal threshold for e = −c is t* = 4c/3 (c = 0.6: t* = 0.8)."""
    distribution = synthetic_distribution("example_n", harm=0.6, num_valuation_cells=10)
    optimum = 2 / 3 - 0.3 + 0.3 * 0.8**2 - 0.8**3 / 4
    genetic_only = run_genetic_search(distribution, 1, 2, 1, num_generations=60, seed=1,
                                      local_refinement="none", verbose=False)
    assert genetic_only.welfare == pytest.approx(optimum, abs=2e-5)
    assert genetic_only.summary["generations_run"] == 60
    refined = run_genetic_search(distribution, 1, 2, 1, num_generations=60, seed=1,
                                 verbose=False)
    assert refined.welfare >= genetic_only.welfare - 1e-12
    assert refined.welfare == pytest.approx(optimum, abs=1e-10)
    tau_at_harm = refined.coefficients[0] - 0.6 * refined.coefficients[1]
    assert tau_at_harm == pytest.approx(0.8, abs=1e-4)


@pytest.mark.parametrize("family_or_population", ["harmful_middle", "population"])
def test_fixed_ivcg_rule_from_same_distribution_reproduces_ironing(family_or_population):
    from audited_auctions import evaluate_threshold, ivcg_scoring_rule_for, population_distribution
    if family_or_population == "population":
        rng = np.random.default_rng(4)
        distribution = population_distribution(rng.normal(-0.3, 0.6, 300).round(2),
                                               rng.integers(1, 40, 300) / 10)
    else:
        distribution = synthetic_distribution("harmful_middle", num_externality_nodes=61,
                                              num_valuation_cells=120)
    threshold = polynomial_threshold([0.1, -0.2])
    rule = ivcg_scoring_rule_for(distribution, threshold, 5, 2)
    own = evaluate_threshold(distribution, threshold, 5, 2)
    fixed = evaluate_threshold(distribution, threshold, 5, 2, ivcg_scoring_rule=rule)
    for metric in ("welfare", "revenue", "penalty_paid", "num_allocated"):
        assert fixed.exact["ivcg_pa"][metric] == pytest.approx(own.exact["ivcg_pa"][metric],
                                                               rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(fixed.penalty_curves["ivcg_pa"], own.penalty_curves["ivcg_pa"],
                               rtol=1e-9, atol=1e-12)


def test_tau_independent_results_are_cached_and_unchanged():
    from audited_auctions import build_mechanism_models, evaluate_threshold
    distribution = synthetic_distribution("harm_rises", num_externality_nodes=61,
                                          num_valuation_cells=120)
    first = evaluate_threshold(distribution, polynomial_threshold([0.4, -1.0]), 5, 2)
    assert ("vcg", 5, 2, "outcomes") in distribution.cache
    second = evaluate_threshold(distribution, polynomial_threshold([0.1, -0.5]), 5, 2)
    models, _ = build_mechanism_models(distribution, polynomial_threshold([0.1]), 5, 2)
    for mechanism in ("vcg", "first_best"):
        assert second.exact[mechanism] == first.exact[mechanism]
        assert first.exact[mechanism]["revenue"] == pytest.approx(
            models[mechanism].expected_outcomes()["revenue"], rel=1e-12)
    # A rescaled distribution starts with an empty cache.
    assert distribution.with_externality_scale(2.0).cache == {}


def test_bootstrap_reuses_baselines_with_identical_results():
    sample = lognormal_sample(1200)
    arguments = dict(num_resamples=4, seed=2)
    first = bootstrap_welfare(sample, polynomial_threshold([0.5, -1.0]), 4, 1, **arguments)
    assert len(sample.cache) == 1
    again = bootstrap_welfare(sample, polynomial_threshold([0.5, -1.0]), 4, 1, **arguments)
    other = bootstrap_welfare(sample, polynomial_threshold([0.2, -0.5]), 4, 1, **arguments)
    assert again.resamples.equals(first.resamples)
    np.testing.assert_array_equal(other.resamples["welfare_vcg"], first.resamples["welfare_vcg"])
