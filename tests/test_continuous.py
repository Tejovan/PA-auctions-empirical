"""
Tests of the continuous version of a gridded distribution
(TypeDistribution.continuous_version, exact_computation.continuous_bid_segments)
and of the relaxation optimizer method.
"""

import numpy as np
import pytest

from PA_auctions import synthetic_distribution
from PA_auctions.exact_computation import iron_quantile_function
from PA_auctions.pipeline import optimize_threshold
from PA_auctions.threshold_evaluation import evaluate_threshold, exact_outcomes
from PA_auctions.threshold_forms import ThresholdForm


def cutoff(distribution, welfare_cutoff):
    form = ThresholdForm("cutoff", 1, np.empty(0), distribution.maximum_valuation, (-3.0, 3.0))
    return form.threshold([welfare_cutoff])


def test_continuous_version_keeps_masses_and_supports():
    gridded = synthetic_distribution("harm_rises", num_externality_nodes=21,
                                     num_valuation_cells=30)
    continuous = gridded.continuous_version()
    assert continuous.is_continuous and not gridded.is_continuous
    assert continuous.probabilities.sum() == pytest.approx(1.0)
    assert np.all(continuous.externality_lower < continuous.externality_upper)
    rng = np.random.default_rng(0)
    e, v = continuous.sample_types(20000, 1, rng)
    assert v.min() >= 0 and v.max() <= continuous.maximum_valuation
    assert len(np.unique(e)) > 1000  # externalities are not on the nodes
    # The mean externality of the samples matches the cells' interval midpoints.
    assert e.mean() == pytest.approx(continuous.mean_externality, abs=0.02)


def test_continuous_version_removes_grid_ironing():
    """Independent externalities with a log-concave density and a welfare cutoff (a steep τ):
    ŵ is nondecreasing (paper Thm [thm:dmrl]), so IVCG-PA = VCG-PA. The grid irons at
    every node threshold; the continuous version does not."""
    gridded = synthetic_distribution("independent", num_externality_nodes=41,
                                     num_valuation_cells=60)
    continuous = gridded.continuous_version()
    tau = cutoff(gridded, 0.4)
    grid_gap = (exact_outcomes(gridded, tau, 4, 1)["ivcg_pa"]["welfare"]
                - exact_outcomes(gridded, tau, 4, 1)["vcg_pa"]["welfare"])
    outcomes = exact_outcomes(continuous, tau, 4, 1)
    assert grid_gap > 1e-5
    assert abs(outcomes["ivcg_pa"]["welfare"] - outcomes["vcg_pa"]["welfare"]) < 1e-8
    # Same distribution up to the discretization: welfare levels agree to the grid's error.
    assert outcomes["vcg_pa"]["welfare"] == pytest.approx(
        exact_outcomes(gridded, tau, 4, 1)["vcg_pa"]["welfare"], rel=5e-3)


def test_continuous_exact_matches_simulation_and_accounting():
    continuous = synthetic_distribution("harmful_middle", num_externality_nodes=21,
                                        num_valuation_cells=30).continuous_version()
    evaluation = evaluate_threshold(continuous, cutoff(continuous, 0.2), 4, 2,
                                    num_simulated_draws=40000, random_seed=1)
    for mechanism, outcomes in evaluation.exact.items():
        draws = evaluation.simulated[mechanism]["welfare"]
        error = draws.std(ddof=1) / np.sqrt(len(draws))
        assert abs(draws.mean() - outcomes["welfare"]) < 4 * error, mechanism
        if mechanism == "first_best":  # an allocation: no payments
            assert "revenue" not in outcomes and "penalty_paid" not in outcomes
            continue
        assert outcomes["valuation_welfare"] == pytest.approx(
            outcomes["bidder_utility"] + outcomes["revenue"] + outcomes["penalty_paid"])


def test_ironing_pools_decreasing_segments():
    """A segment that decreases inside is pooled to its mean before the projection."""
    pieces = iron_quantile_function(np.array([0.0, 1.0]), np.array([1.0, 2.0]),
                                    np.array([1.0, 2.0]), np.array([0.0, 3.0]))
    starts, ends, low, high = pieces
    # Segment 1 pools to 0.5; segment 2 rises from 2 to 3: projection unchanged.
    assert low[0] == pytest.approx(0.5) and high[0] == pytest.approx(0.5)
    assert ends[-1] == pytest.approx(2.0) and high[-1] == pytest.approx(3.0)


def test_relaxation_method_reports_the_bound():
    distribution = synthetic_distribution("example_n", num_valuation_cells=60)
    result = optimize_threshold(distribution, "nodes", 2, 1, optimizer="relaxation",
                                best_response_options={"max_iterations": 10}, verbose=False)
    summary = result["search_summaries"][0]
    assert [start["start"] for start in summary["starts"]] == ["relaxation"]
    relaxation = summary["relaxation"]
    assert relaxation["threshold_rule"]
    # Example N (paper Ex. [ex:N]): the relaxation value equals the best common τ.
    assert result["welfare"] == pytest.approx(relaxation["value"], rel=1e-4)
