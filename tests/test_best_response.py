"""
Tests of the threshold forms, the marginal contribution, best-response ascent and the
gradient polish. Run from WEB27/code with

    .venv/bin/python -m pytest tests -q
"""

import numpy as np
import pytest

from audited_auctions import synthetic_distribution
from audited_auctions.best_response_ascent import (
    first_order_check,
    run_best_response_ascent,
    welfare_and_gradient,
)
from audited_auctions.marginal_contribution import RivalWelfareLoss
from audited_auctions.pipeline import optimize_threshold, prepare_distributions
from audited_auctions.threshold_evaluation import build_mechanism_model, expected_welfare
from audited_auctions.threshold_forms import (
    build_threshold_form,
    externality_nodes,
    parse_form_label,
    threshold_from_spec,
)

SMALL_GRID = {"num_externality_nodes": 31, "num_valuation_cells": 60}


def example_n_welfare(c, t):
    """Paper, Example N (Ex. 7.11 [ex:N]): W(t) = 2/3 − c/2 + (c/2) t² − t³/4."""
    return 2 / 3 - c / 2 + c / 2 * t ** 2 - t ** 3 / 4


def test_form_labels_and_spec_round_trip():
    assert parse_form_label("poly2") == ("polynomial", 2)
    assert parse_form_label("knots6") == ("knots", 6)
    assert parse_form_label("nodes") == ("nodes", None)
    with pytest.raises(ValueError):
        parse_form_label("splines3")
    distribution = synthetic_distribution("harm_rises", **SMALL_GRID)
    form = build_threshold_form("knots5", distribution)
    parameters = [1.4, 1.0, 0.6, 0.2, 0.0, -1.0]
    rebuilt = threshold_from_spec(form.spec(parameters))
    externalities = np.linspace(-2, 2, 50)
    np.testing.assert_allclose(rebuilt(externalities), form.threshold(parameters)(externalities))
    assert np.all(np.isinf(rebuilt(np.array([-1.5]))))  # below the exclusion cutoff


def test_projection_keeps_excluded_prefix():
    distribution = synthetic_distribution("harm_rises", **SMALL_GRID)
    form = build_threshold_form("knots4", distribution)
    nodes, _, _ = externality_nodes(distribution)
    target = lambda e: np.where(e < -1.0, np.inf, np.maximum(-np.asarray(e), 0.0))
    parameters = form.project(target, distribution)
    values = form.threshold(parameters)(nodes)
    assert np.all(np.isinf(values[nodes < -1.0]))
    assert np.all(np.isfinite(values[nodes > -0.9]))


def test_marginal_contribution_matches_example_n():
    c = 0.6
    distribution = synthetic_distribution("example_n", harm=c, num_valuation_cells=60)
    for t in (0.3, 0.8):
        threshold = lambda e, t=t: np.where(np.asarray(e) < -c / 2, t, 0.0)
        model = build_mechanism_model("vcg_pa", distribution, threshold, 2, 1)
        allocation, loss = RivalWelfareLoss(model)(np.array([t]))
        marginal = (t - c) * allocation - loss
        assert marginal[0] == pytest.approx(0.75 * t ** 2 - c * t, abs=1e-12)


@pytest.mark.parametrize("num_bidders,num_goods", [(3, 1), (8, 2), (5, 3)])
def test_rival_welfare_loss_is_clarke_payment_without_externalities(num_bidders, num_goods):
    distribution = synthetic_distribution("independent", externality_scale=1e-12, **SMALL_GRID)
    threshold = lambda e: np.full(np.shape(e), 0.3)
    model = build_mechanism_model("vcg_pa", distribution, threshold, num_bidders, num_goods)
    bids = np.array([0.35, 0.6, 0.9, 1.5])
    allocation, loss = RivalWelfareLoss(model)(bids)
    np.testing.assert_allclose(loss, bids * allocation - model.allocation_integral(bids),
                               atol=1e-12)


@pytest.mark.parametrize("mechanism", ["vcg_pa", "ivcg_pa"])
def test_gradient_matches_finite_differences(mechanism):
    distribution = synthetic_distribution("harm_rises", **SMALL_GRID)
    form = build_threshold_form("knots4", distribution)
    parameters = np.array([1.4, 0.93, 0.51, 0.13, -1.37])
    _, gradient = welfare_and_gradient(distribution, form, parameters, 6, 2, mechanism)
    for index in range(4):
        step = np.zeros_like(parameters)
        step[index] = 1e-6
        difference = (expected_welfare(distribution, form.threshold(parameters + step), 6, 2,
                                       mechanism)
                      - expected_welfare(distribution, form.threshold(parameters - step), 6, 2,
                                         mechanism)) / 2e-6
        assert gradient[index] == pytest.approx(difference, rel=1e-4, abs=1e-8)


@pytest.mark.parametrize("label", ["nodes", "knots3", "bins2"])
def test_best_response_ascent_recovers_example_n(label):
    c = 0.6
    distribution = synthetic_distribution("example_n", harm=c, num_valuation_cells=60)
    result = run_best_response_ascent(distribution, label, 2, 1, verbose=False)
    assert result.welfare == pytest.approx(example_n_welfare(c, 4 * c / 3), abs=1e-10)


def test_optimum_has_zero_marginal_contribution_at_interior_thresholds():
    distribution = synthetic_distribution("harmful_middle", **SMALL_GRID)
    result = run_best_response_ascent(distribution, "nodes", 6, 2, verbose=False)
    form = build_threshold_form("nodes", distribution)
    check = first_order_check(distribution, form.threshold(result.coefficients), 6, 2)
    interior = check[check.status == "interior"]
    assert len(interior) > 0
    assert check.first_order_violation.max() < 1e-4
    assert check.best_response_gap.sum() < 1e-8


def test_best_response_beats_or_matches_polynomial_genetic_search():
    prepared = prepare_distributions("harm_rises", **SMALL_GRID)
    ga_options = {"num_generations": 15}
    genetic = optimize_threshold(prepared.training, "poly1", 5, 2, optimizer="genetic",
                                 ga_options=ga_options, ga_seed=0, verbose=False)
    combined = optimize_threshold(prepared.training, "knots4", 5, 2,
                                  optimizer="genetic+best_response", genetic_form="poly1",
                                  ga_options=ga_options, ga_seed=0, verbose=False)
    nodes = optimize_threshold(prepared.training, "nodes", 5, 2, verbose=False)
    assert combined["optimizer"] == "genetic+best_response"
    assert combined["welfare"] >= genetic["welfare"] - 1e-12
    assert nodes["welfare"] >= combined["welfare"] - 1e-9
    assert nodes["form"].kind == "nodes"
    assert {spec["form"] for spec in combined["trajectory_forms"]} == {"polynomial", "knots"}
    traces = [summary["trace"] for summary in combined["search_summaries"]]
    assert all(len(trace) > 0 for trace in traces)


def test_genetic_search_rejects_nodes_and_best_response_rejects_polynomials():
    prepared = prepare_distributions("harm_rises", **SMALL_GRID)
    with pytest.raises(ValueError):
        optimize_threshold(prepared.training, "nodes", 4, 1, optimizer="genetic", verbose=False)
    with pytest.raises(ValueError):
        optimize_threshold(prepared.training, "poly1", 4, 1, optimizer="best_response",
                           verbose=False)


def test_ascent_history_reaches_first_order_conditions():
    distribution = synthetic_distribution("harmful_middle", **SMALL_GRID)
    result = run_best_response_ascent(distribution, "nodes", 6, 2, verbose=False)
    for start in result.summary["starts"]:
        assert start["stop_reason"] in ("no_improving_step", "small_gain")
        last = start["history"][-1]
        assert last["max_marginal"] < 1e-4 and last["stationarity"] < 1e-4
    # τ ≡ 0 satisfies the first-order conditions (a bidder with v = 0 never wins, so
    # M(e, 0) = 0) but is not optimal; the best-response gap detects it.
    zero = next(s for s in result.summary["starts"] if s["start"] == "zero")
    assert zero["history"][0]["stationarity"] < 1e-12
    assert zero["history"][0]["gap"] > 1e-3
    assert zero["welfare"] > zero["start_welfare"]


def test_time_limit_stops_gracefully():
    import time

    distribution = synthetic_distribution("harmful_middle", **SMALL_GRID)
    result = run_best_response_ascent(distribution, "nodes", 6, 2, verbose=False,
                                      deadline=time.time())
    reasons = [s["stop_reason"] for s in result.summary["starts"]]
    assert reasons[0] == "time_limit" and len(reasons) == 7  # 4 named + 3 random starts
    assert set(reasons[1:]) == {"skipped (time limit)"}
    assert np.isfinite(result.welfare)
    search = optimize_threshold(distribution, "poly1", 6, 2, optimizer="genetic",
                                ga_options={"num_generations": 50}, ga_seed=0, verbose=False,
                                time_limit=0.0)
    summary = search["search_summaries"][0]
    assert summary["stop_reason"] == "time_limit" and summary["generations_run"] == 1
    assert summary["local_starts"] == []


def test_participation_gap_and_relaxation():
    from audited_auctions.best_response_ascent import participation_gap
    from audited_auctions.symmetric_relaxation import symmetric_relaxation
    from audited_auctions.threshold_functions import polynomial_threshold, tabulated_threshold

    distribution = synthetic_distribution("example_n", num_valuation_cells=60)
    # Example N with n = 2, k = 1: the best τ(−c) is 4c/3 and τ(0) = 0 (paper Ex. 7.11).
    harm = 0.6
    relaxation = symmetric_relaxation(distribution, 2, 1, num_intervals=600)
    assert relaxation.is_threshold_rule
    best = relaxation.thresholds[np.argmin(relaxation.nodes)]
    assert abs(best - 4 * harm / 3) < 0.01
    optimum = tabulated_threshold([-harm, 0.0], [4 * harm / 3, 0.0])
    welfare = 2 / 3 - harm / 2 + (harm / 2) * (0.8) ** 2 - 0.8 ** 3 / 4
    assert abs(relaxation.value - welfare) < 1e-4
    gap, _, _ = participation_gap(distribution, optimum, 2, 1)
    assert gap < 1e-9
    zero_welfare, _, _ = participation_gap(distribution, polynomial_threshold([0.0, -1.0]), 2, 1)
    assert zero_welfare > 1e-3


def test_nodes_thresholds_take_the_smallest_equivalent_value():
    """A node whose whole support participates gets τ = 0 (same welfare, smallest
    penalty), whether the search ends there or the relaxation start puts τ there."""
    distribution = synthetic_distribution("harmful_middle", **SMALL_GRID)
    nodes, _, node_of_cell = externality_nodes(distribution)
    positive = distribution.probabilities > 0
    lowest = np.full(len(nodes), np.inf)
    np.minimum.at(lowest, node_of_cell[positive], distribution.valuation_lower[positive])
    for optimizer in ("best_response", "relaxation"):
        result = optimize_threshold(distribution, "nodes", 4, 1, optimizer=optimizer,
                                    best_response_options={"max_iterations": 20},
                                    verbose=False)
        parameters = np.asarray(result["parameters"])
        assert not np.any((parameters > 0) & (parameters <= lowest)), optimizer
        form = build_threshold_form("nodes", distribution)
        assert expected_welfare(distribution, form.threshold(parameters), 4, 1, "vcg_pa") == (
            pytest.approx(result["welfare"], rel=1e-12))
