"""
pipeline.py

One optimization-and-evaluation cell, shared by run_sweep_genetic2.py and
Collateralized_Auction_genetic2.py:

  prepare_distributions   build the distribution τ is designed on (training)
                          and the one it is evaluated on (evaluation)
  optimize_and_evaluate   genetic search for τ with the exact objective, then
                          exact evaluation of the four mechanisms, one
                          simulation for per-auction welfare distributions,
                          and (empirical data) a bootstrap of exact welfare and
                          an optional bandwidth sensitivity check

Synthetic families: training = evaluation = the family's piecewise-uniform
distribution (H is known; no sampling uncertainty). Optional grid-refinement
check of the discretization error.

Empirical CSV: all rows are split into training and test halves (the same
split for every cell when the split seed is fixed); each half is truncated at
a valuation quantile of the full sample (training: truncation_quantile,
default 0.995; test: evaluation_truncation_quantile, default the same). The
training half is kernel-smoothed (or atomic with smoothing="none"); τ and the
IVCG-PA scoring rule are designed on it. The test half is atomic by default
(evaluation="atomic"; "smoothed" smooths it like the training half), and the
designed τ and IVCG-PA scoring rule are evaluated on it; the bootstrap
resamples the test rows.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import pandas as pd

from .empirical import (
    EmpiricalSample,
    bootstrap_welfare,
    exact_welfare_by_mechanism,
    load_empirical_sample,
    sample_distribution,
    split_sample,
    truncate_sample,
    valuation_bound,
)
from .genetic_search import run_genetic_search
from .threshold_evaluation import (
    AUDITED_MECHANISMS,
    MECHANISMS,
    evaluate_threshold,
    grid_refinement_check,
    ivcg_scoring_rule_for,
)
from .threshold_functions import polynomial_threshold
from .type_distributions import SYNTHETIC_FAMILIES, TypeDistribution, synthetic_distribution

SYNTHETIC_GRID_DEFAULTS = {"num_externality_nodes": 301, "num_valuation_cells": 600}
EMPIRICAL_GRID_DEFAULTS = {"num_externality_nodes": 241, "num_valuation_cells": 400}
EVALUATION_CHOICES = ("atomic", "smoothed")


def is_synthetic_source(source) -> bool:
    return source in SYNTHETIC_FAMILIES


@dataclass
class PreparedDistributions:
    """Training and evaluation distributions of one (source, ζ) combination."""

    source: str
    training: TypeDistribution
    evaluation: TypeDistribution
    test_sample: Optional[EmpiricalSample]
    training_sample: Optional[EmpiricalSample] = None
    settings: dict = field(default_factory=dict)

    @property
    def is_empirical(self) -> bool:
        return self.test_sample is not None


def prepare_distributions(source, externality_scale=1.0, valuation_scale=1.0,
                          truncation_quantile=0.995, evaluation_truncation_quantile=None,
                          train_fraction=0.5, split_seed=0, smoothing="kernel",
                          bandwidth_scale=1.0, evaluation="atomic", num_externality_nodes=None,
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

    if evaluation not in EVALUATION_CHOICES:
        raise ValueError(f"evaluation must be one of {EVALUATION_CHOICES}")
    if evaluation_truncation_quantile is None:
        evaluation_truncation_quantile = truncation_quantile
    grid = {"num_externality_nodes": num_externality_nodes
            or EMPIRICAL_GRID_DEFAULTS["num_externality_nodes"],
            "num_valuation_cells": num_valuation_cells
            or EMPIRICAL_GRID_DEFAULTS["num_valuation_cells"]}
    full = load_empirical_sample(source, externality_scale, valuation_scale)
    train_full, test_full = split_sample(full, train_fraction, split_seed)
    train = truncate_sample(train_full, valuation_bound(full, truncation_quantile))
    test = truncate_sample(test_full, valuation_bound(full, evaluation_truncation_quantile))
    smoothing_options = dict(smoothing=smoothing, bandwidth_scale=bandwidth_scale, **grid)
    evaluation_options = (dict(smoothing="none") if evaluation == "atomic"
                          else smoothing_options)
    return PreparedDistributions(
        source=str(source),
        training=sample_distribution(train, **smoothing_options),
        evaluation=sample_distribution(test, **evaluation_options),
        test_sample=test, training_sample=train,
        settings={"kind": "empirical", "csv": str(source), "externality_scale": externality_scale,
                  "valuation_scale": valuation_scale, "truncation_quantile": truncation_quantile,
                  "evaluation_truncation_quantile": evaluation_truncation_quantile,
                  "maximum_valuation": train.maximum_valuation,
                  "evaluation_maximum_valuation": test.maximum_valuation,
                  "train_fraction": train_fraction, "split_seed": split_seed,
                  "num_rows_loaded": full.settings["num_rows_loaded"],
                  "num_train_rows": train.num_rows, "num_test_rows": test.num_rows,
                  "evaluation": evaluation, "evaluation_smoothing": evaluation_options["smoothing"],
                  **smoothing_options},
    )


def _designed_ivcg_rule(prepared, design_distribution, threshold, num_bidders, num_goods):
    """IVCG-PA scoring rule designed on the training side (empirical data only)."""
    if not prepared.is_empirical:
        return None
    return ivcg_scoring_rule_for(design_distribution, threshold, num_bidders, num_goods)


def optimize_and_evaluate(prepared, polynomial_degree, num_bidders, num_goods,
                          objective_mechanism="vcg_pa", ga_options=None, ga_seed=None,
                          num_restarts=1, initial_solution=None, num_simulated_draws=2000,
                          simulation_seed=None, num_bootstrap=200, bootstrap_seed=0,
                          grid_refinement=False, sensitivity_bandwidth_scales=(),
                          num_type_sample=4000, verbose=True):
    """Optimize a polynomial τ on prepared.training and evaluate it on prepared.evaluation.

    sensitivity_bandwidth_scales: empirical data with kernel smoothing only;
    for each scale, rerun the genetic search on the training half smoothed with
    that bandwidth multiplier and evaluate the resulting design exactly on the
    same evaluation distribution (record["bandwidth_sensitivity"]).

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
    ivcg_rule = _designed_ivcg_rule(prepared, prepared.training, threshold, num_bidders,
                                    num_goods)

    evaluation = evaluate_threshold(prepared.evaluation, threshold, num_bidders, num_goods,
                                    num_simulated_draws=num_simulated_draws,
                                    random_seed=simulation_seed, ivcg_scoring_rule=ivcg_rule)

    bootstrap = None
    if prepared.is_empirical and num_bootstrap > 0:
        settings = prepared.settings
        bootstrap = bootstrap_welfare(
            prepared.test_sample, threshold, num_bidders, num_goods, num_resamples=num_bootstrap,
            seed=bootstrap_seed, smoothing=settings["evaluation_smoothing"],
            bandwidth_scale=settings["bandwidth_scale"], ivcg_scoring_rule=ivcg_rule,
            num_externality_nodes=settings["num_externality_nodes"],
            num_valuation_cells=settings["num_valuation_cells"])

    sensitivity = None
    if (prepared.is_empirical and sensitivity_bandwidth_scales
            and prepared.settings["smoothing"] == "kernel"):
        sensitivity = bandwidth_sensitivity(
            prepared, sensitivity_bandwidth_scales, polynomial_degree, num_bidders, num_goods,
            objective_mechanism, ga_options, ga_seed, num_restarts,
            base_result=(tau_coefficients, training_welfare, evaluation), verbose=verbose)

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
        "bandwidth_sensitivity": sensitivity,
        "type_sample": {"externalities": sample_externalities, "valuations": sample_valuations},
        "ga_seed": ga_seed,
        "num_restarts": num_restarts,
        "simulation_seed": simulation_seed,
        "bootstrap_seed": bootstrap_seed,
    }


def _sensitivity_row(scale, coefficients, training_welfare, welfare):
    row = {"bandwidth_scale": scale, "tau_coeffs": list(coefficients),
           "training_welfare": training_welfare}
    row.update({f"welfare_{m}": welfare[m] for m in MECHANISMS})
    first_best_gain = welfare["first_best"] - welfare["vcg"]
    for mechanism in AUDITED_MECHANISMS:
        gain = welfare[mechanism] - welfare["vcg"]
        row[f"gain_{mechanism}"] = gain
        row[f"share_{mechanism}"] = gain / first_best_gain if first_best_gain else float("nan")
    return row


def bandwidth_sensitivity(prepared, bandwidth_scales, polynomial_degree, num_bidders, num_goods,
                          objective_mechanism, ga_options, ga_seed, num_restarts, base_result,
                          verbose=True):
    """Redesign τ (and the IVCG-PA scores) on the training half smoothed with each
    bandwidth multiplier, and evaluate each design exactly on the same evaluation
    distribution. One row per bandwidth scale, including the base design."""
    settings = prepared.settings
    base_coefficients, base_training_welfare, base_evaluation = base_result
    rows = [_sensitivity_row(settings["bandwidth_scale"], base_coefficients, base_training_welfare,
                             {m: base_evaluation.exact[m]["welfare"] for m in MECHANISMS})]
    for scale in bandwidth_scales:
        if scale == settings["bandwidth_scale"]:
            continue
        training = sample_distribution(prepared.training_sample, "kernel", scale,
                                       num_externality_nodes=settings["num_externality_nodes"],
                                       num_valuation_cells=settings["num_valuation_cells"])
        best = (None, -np.inf)
        for restart in range(num_restarts):
            coefficients, welfare, _ = run_genetic_search(
                training, polynomial_degree, num_bidders, num_goods,
                objective_mechanism=objective_mechanism,
                seed=None if ga_seed is None else ga_seed + restart, verbose=verbose,
                **(ga_options or {}))
            if welfare > best[1]:
                best = (coefficients, welfare)
        threshold = polynomial_threshold(best[0])
        ivcg_rule = ivcg_scoring_rule_for(training, threshold, num_bidders, num_goods)
        welfare = exact_welfare_by_mechanism(prepared.evaluation, threshold, num_bidders,
                                             num_goods, ivcg_rule)
        rows.append(_sensitivity_row(scale, best[0], best[1], welfare))
    return pd.DataFrame(rows).sort_values("bandwidth_scale").set_index("bandwidth_scale")
