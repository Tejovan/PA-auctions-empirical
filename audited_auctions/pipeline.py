"""
pipeline.py

One optimization-and-evaluation cell, shared by run_sweep_genetic2.py and
Collateralized_Auction_genetic2.py:

  prepare_distributions   build the distribution τ is optimized on (training)
                          and the one it is evaluated on (evaluation)
  optimize_and_evaluate   genetic search for τ with the exact objective, then
                          exact evaluation of the four mechanisms, one
                          simulation for per-auction welfare distributions,
                          and (empirical data) a bootstrap of exact welfare

Synthetic families: training = evaluation = the family's piecewise-uniform
distribution (H is known; no sampling uncertainty). Optional grid-refinement
check of the discretization error.

Empirical CSV: rows are scaled and truncated at a valuation quantile (v̄),
split into training and test halves (the same split for every cell when the
split seed is fixed), and each half is smoothed (kernel, default) or used as
atoms. τ is optimized on the training distribution and evaluated on the test
distribution; the bootstrap resamples the test rows.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .empirical import (
    EmpiricalSample,
    bootstrap_welfare,
    load_empirical_sample,
    sample_distribution,
    split_sample,
)
from .genetic_search import run_genetic_search
from .threshold_evaluation import evaluate_threshold, grid_refinement_check
from .threshold_functions import polynomial_threshold
from .type_distributions import SYNTHETIC_FAMILIES, TypeDistribution, synthetic_distribution

SYNTHETIC_GRID_DEFAULTS = {"num_externality_nodes": 301, "num_valuation_cells": 600}
EMPIRICAL_GRID_DEFAULTS = {"num_externality_nodes": 241, "num_valuation_cells": 400}


def is_synthetic_source(source) -> bool:
    return source in SYNTHETIC_FAMILIES


@dataclass
class PreparedDistributions:
    """Training and evaluation distributions of one (source, ζ) combination."""

    source: str
    training: TypeDistribution
    evaluation: TypeDistribution
    test_sample: Optional[EmpiricalSample]
    settings: dict = field(default_factory=dict)

    @property
    def is_empirical(self) -> bool:
        return self.test_sample is not None


def prepare_distributions(source, externality_scale=1.0, valuation_scale=1.0,
                          truncation_quantile=0.995, train_fraction=0.5, split_seed=0,
                          smoothing="kernel", bandwidth_scale=1.0, num_externality_nodes=None,
                          num_valuation_cells=None, family_options=None):
    """Distributions for one cell; `source` is a synthetic family name or a CSV path."""
    if is_synthetic_source(source):
        grid = {key: value for key, value in (("num_externality_nodes", num_externality_nodes),
                                              ("num_valuation_cells", num_valuation_cells))
                if value is not None}
        grid = {**SYNTHETIC_GRID_DEFAULTS, **grid}
        if source == "example_n":
            grid.pop("num_externality_nodes")
        distribution = synthetic_distribution(source, externality_scale=externality_scale,
                                              **grid, **(family_options or {}))
        return PreparedDistributions(
            source=source, training=distribution, evaluation=distribution, test_sample=None,
            settings={"kind": "synthetic", "family": source, "externality_scale": externality_scale,
                      "grid": grid, "family_options": dict(family_options or {})})

    grid = {"num_externality_nodes": num_externality_nodes
            or EMPIRICAL_GRID_DEFAULTS["num_externality_nodes"],
            "num_valuation_cells": num_valuation_cells
            or EMPIRICAL_GRID_DEFAULTS["num_valuation_cells"]}
    sample = load_empirical_sample(source, externality_scale, valuation_scale, truncation_quantile)
    train, test = split_sample(sample, train_fraction, split_seed)
    options = dict(smoothing=smoothing, bandwidth_scale=bandwidth_scale, **grid)
    return PreparedDistributions(
        source=str(source),
        training=sample_distribution(train, **options),
        evaluation=sample_distribution(test, **options),
        test_sample=test,
        settings={"kind": "empirical", "csv": str(source), "externality_scale": externality_scale,
                  "valuation_scale": valuation_scale, "truncation_quantile": truncation_quantile,
                  "maximum_valuation": sample.maximum_valuation, "train_fraction": train_fraction,
                  "split_seed": split_seed, "num_train_rows": train.num_rows,
                  "num_test_rows": test.num_rows, **sample.settings, **options},
    )


def optimize_and_evaluate(prepared, polynomial_degree, num_bidders, num_goods,
                          objective_mechanism="vcg_pa", ga_options=None, ga_seed=None,
                          num_restarts=1, initial_solution=None, num_simulated_draws=2000,
                          simulation_seed=None, num_bootstrap=200, bootstrap_seed=0,
                          grid_refinement=False, num_type_sample=4000, verbose=True):
    """Optimize a polynomial τ on prepared.training and evaluate it on prepared.evaluation.

    Returns a record dict (see run_sweep_genetic2.py README for the fields).
    """
    ga_options = dict(ga_options or {})
    best = (None, -np.inf)
    trajectory = []
    for restart in range(num_restarts):
        coefficients, welfare, restart_trajectory = run_genetic_search(
            prepared.training, polynomial_degree, num_bidders, num_goods,
            objective_mechanism=objective_mechanism,
            seed=None if ga_seed is None else ga_seed + restart,
            initial_solution=initial_solution if restart == 0 else None,
            verbose=verbose, **ga_options)
        trajectory.extend(restart_trajectory)
        if welfare > best[1]:
            best = (coefficients, welfare)
    tau_coefficients, training_welfare = best
    threshold = polynomial_threshold(tau_coefficients)

    evaluation = evaluate_threshold(prepared.evaluation, threshold, num_bidders, num_goods,
                                    num_simulated_draws=num_simulated_draws,
                                    random_seed=simulation_seed)

    bootstrap = None
    if prepared.is_empirical and num_bootstrap > 0:
        settings = prepared.settings
        bootstrap = bootstrap_welfare(
            prepared.test_sample, threshold, num_bidders, num_goods, num_resamples=num_bootstrap,
            seed=bootstrap_seed, smoothing=settings["smoothing"],
            bandwidth_scale=settings["bandwidth_scale"],
            num_externality_nodes=settings["num_externality_nodes"],
            num_valuation_cells=settings["num_valuation_cells"])

    refinement = None
    if grid_refinement and not prepared.is_empirical:
        settings = prepared.settings
        refinement = grid_refinement_check(
            settings["family"], threshold, num_bidders, num_goods,
            externality_scale=settings["externality_scale"], **settings["grid"],
            **settings["family_options"])

    sample_externalities, sample_valuations = prepared.evaluation.sample_type_points(
        num_type_sample, np.random.default_rng(0))
    return {
        "source": prepared.source,
        "distribution_settings": prepared.settings,
        "objective_mechanism": objective_mechanism,
        "polynomial_degree": polynomial_degree,
        "num_bidders": num_bidders,
        "num_goods": num_goods,
        "tau_coeffs": tau_coefficients,
        "training_welfare": training_welfare,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "bootstrap": bootstrap,
        "grid_refinement": refinement,
        "type_sample": {"externalities": sample_externalities, "valuations": sample_valuations},
        "ga_seed": ga_seed,
        "num_restarts": num_restarts,
        "simulation_seed": simulation_seed,
        "bootstrap_seed": bootstrap_seed,
    }
