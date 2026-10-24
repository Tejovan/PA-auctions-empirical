"""Grid search over the cutoff and poly1 forms (grid_search.py) and its configuration."""

import numpy as np
import pytest

from PA_auctions import synthetic_distribution
from PA_auctions.experiments.config import ConfigError, expand, resolve_config
from PA_auctions.genetic_search import run_genetic_search
from PA_auctions.grid_search import GridObjective, local_maxima, polish_starts, run_grid_search
from PA_auctions.threshold_evaluation import expected_welfare
from PA_auctions.threshold_forms import build_threshold_form

SMALL = {"num_externality_nodes": 31, "num_valuation_cells": 40}


@pytest.fixture(scope="module")
def distribution():
    return synthetic_distribution("harmful_middle", **SMALL)


def test_line_coordinates_map_to_the_described_lines(distribution):
    form = build_threshold_form("poly1", distribution)
    objective = GridObjective(distribution, form, 4, 1, "vcg_pa")
    e_min, e_max = form.externality_range
    v_bar = form.maximum_valuation
    # θ = π/2: constant τ = u v̄
    intercept, slope = objective.parameters([np.pi / 2, 0.3])
    assert slope == pytest.approx(0.0, abs=1e-12) and intercept == pytest.approx(0.3 * v_bar)
    # θ = π/4: the diagonal through the box corners (e_min, v̄) and (e_max, 0) at u = 1/2
    intercept, slope = objective.parameters([np.pi / 4, 0.5])
    assert intercept + slope * e_min == pytest.approx(v_bar)
    assert intercept + slope * e_max == pytest.approx(0.0, abs=1e-9)
    # u = 0: every type participates; u = 1: every externality is excluded
    nodes = np.unique(distribution.externalities)
    for angle in (0.1, 1.0, 2.5):
        tau = form.threshold(objective.parameters([angle, 0.0]))(nodes)
        assert np.all(tau <= 1e-9)
        tau = form.threshold(objective.parameters([angle, 1.0]))(nodes)
        assert np.all(tau >= v_bar - 1e-9)


def test_polish_starts_are_separated_local_maxima():
    values = np.zeros((9, 9))
    values[2, 2], values[2, 3], values[7, 7] = 3.0, 2.5, 2.0
    assert local_maxima(values)[2, 2] and not local_maxima(values)[2, 3]
    starts, _ = polish_starts(values, 3)
    assert [tuple(s) for s in starts[:2]] == [(2, 2), (7, 7)]


def test_grid_search_matches_or_beats_the_genetic_search(distribution):
    for label in ("cutoff", "poly1"):
        form = build_threshold_form(label, distribution)
        grid = run_grid_search(distribution, form, 4, 1, grid_points=25, num_local_starts=3)
        genetic = run_genetic_search(distribution, None, 4, 1, threshold_form=label,
                                     num_generations=40, sol_per_pop=20, verbose=False)
        assert grid.welfare >= genetic.welfare - 1e-6 * abs(genetic.welfare)
        assert grid.welfare >= grid.summary["grid_welfare"]
        welfare = expected_welfare(distribution, form.threshold(grid.coefficients), 4, 1, "vcg_pa")
        assert welfare == pytest.approx(grid.welfare, rel=1e-12)
        assert grid.summary["grid_shape"] == ([25] if label == "cutoff" else [25, 25])


def test_grid_optimizer_configuration():
    raw = {"kind": "optimize", "distributions": [{"source": "harmful_middle", **SMALL}],
           "markets": {"pairs": [[4, 1], [4, 3]]},
           "design": {"objectives": ["vcg_pa", "ivcg_pa"]},
           "optimizers": [{"method": "best_response", "threshold_forms": ["nodes"]},
                          {"method": "grid", "threshold_forms": ["poly1", "cutoff"],
                           "objectives": ["vcg_pa"], "grid_points": 31}]}
    config = resolve_config(raw)
    combinations, _ = expand(config)
    # 2 markets × (br nodes × 2 objectives + grid × 2 forms × VCG-PA)
    assert len(combinations) == 2 * (2 + 2)
    paths = {c.path for c in combinations}
    assert "harmful-middle/z1/n4-k3/vcgpa_poly1_grid" in paths
    assert "harmful-middle/z1/n4-k3/ivcgpa_poly1_grid" not in paths
    assert "harmful-middle/z1/n4-k3/ivcgpa_nodes_br" in paths
    # one label for two grid tables with different objectives; equal folders are an error
    shared = {**raw, "optimizers": [
        {"method": "grid", "threshold_forms": ["poly1"], "objectives": ["vcg_pa"]},
        {"method": "grid", "threshold_forms": ["cutoff"]}]}
    assert len(expand(resolve_config(shared))[0]) == 2 * (1 + 2)
    with pytest.raises(ConfigError, match="same combination"):
        expand(resolve_config({**raw, "optimizers": [{"method": "grid"}, {"method": "grid"}]}))
    with pytest.raises(ConfigError, match="different methods"):
        resolve_config({**raw, "optimizers": [{"method": "grid", "label": "x"},
                                              {"method": "best_response", "label": "x"}]})
    with pytest.raises(ConfigError, match="cannot search form 'poly2'"):
        resolve_config({**raw, "optimizers": [{"method": "grid", "threshold_forms": ["poly2"]}]})
    with pytest.raises(ConfigError, match="objectives must be in"):
        resolve_config({**raw, "optimizers": [{"method": "grid", "objectives": ["vcg"]}]})
