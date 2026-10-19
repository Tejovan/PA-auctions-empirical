"""
pipeline.py

Building blocks of one combination of a run (used by experiments/compute.py):

  prepare_distributions   build the distribution τ is designed on (training)
                          and the one it is evaluated on (evaluation)
  optimize_threshold      search for τ in a threshold form (threshold_forms.py)
                          with the exact objective: genetic search,
                          best-response ascent, or both
  bandwidth_sensitivity   redesign τ on the training half smoothed with other
                          bandwidths and evaluate each design on the test half

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
    exact_welfare_by_mechanism,
    load_empirical_sample,
    sample_distribution,
    split_sample,
    truncate_sample,
    valuation_bound,
)
from .best_response_ascent import run_best_response_ascent
from .genetic_search import run_genetic_search
from .threshold_evaluation import AUDITED_MECHANISMS, MECHANISMS, ivcg_scoring_rule_for
from .threshold_forms import BEST_RESPONSE_FORMS, GENETIC_FORMS, build_threshold_form
from .type_distributions import SYNTHETIC_FAMILIES, TypeDistribution, synthetic_distribution

SYNTHETIC_GRID_DEFAULTS = {"num_externality_nodes": 151, "num_valuation_cells": 150}
EMPIRICAL_GRID_DEFAULTS = {"num_externality_nodes": 241, "num_valuation_cells": 400}
EVALUATION_CHOICES = ("atomic", "smoothed")
OPTIMIZERS = ("auto", "genetic", "best_response", "genetic+best_response")


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


def designed_ivcg_rule(prepared, design_distribution, threshold, num_bidders, num_goods):
    """IVCG-PA scoring rule designed on the training side (empirical data only)."""
    if not prepared.is_empirical:
        return None
    return ivcg_scoring_rule_for(design_distribution, threshold, num_bidders, num_goods)


def resolve_optimizer(optimizer, form_kind):
    """'auto': genetic search for polynomials, best-response ascent for the other forms."""
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if optimizer == "auto":
        return "genetic" if form_kind == "polynomial" else "best_response"
    if optimizer == "genetic" and form_kind not in GENETIC_FORMS:
        raise ValueError(f"the genetic search needs a form in {GENETIC_FORMS}")
    if optimizer != "genetic" and form_kind not in BEST_RESPONSE_FORMS:
        raise ValueError(f"best-response ascent needs a form in {BEST_RESPONSE_FORMS}")
    return optimizer


def optimize_threshold(training, threshold_form, num_bidders, num_goods,
                       objective_mechanism="vcg_pa", optimizer="auto", ga_options=None,
                       ga_seed=None, num_restarts=1, initial_solution=None, genetic_form=None,
                       best_response_options=None, warm_starts=(), verbose=True):
    """Search for τ in threshold_form on the training distribution.

    threshold_form : label ("poly2", "knots6", "bins10", "nodes") or polynomial degree
    optimizer      : "auto", "genetic", "best_response" or "genetic+best_response"
                     (genetic search in genetic_form, default the same form or knots6
                     for nodes, then best-response ascent started from its result
                     as well as from the default starts)
    initial_solution : genetic search warm start (parameters of the genetic form)
    warm_starts    : threshold functions used as extra best-response starts
    best_response_options : keyword options of run_best_response_ascent (starts,
                     max_iterations, relative_tolerance, short_genetic_generations)
    Returns a dict: form, parameters, welfare, optimizer, search_summaries, and the
    trajectory: candidate parameter vectors as (form index, parameters) pairs, with
    trajectory_forms the specs of the forms (genetic and best-response forms differ
    for genetic+best_response).
    """
    form = build_threshold_form(threshold_form, training)
    optimizer = resolve_optimizer(optimizer, form.kind)
    ga_options = dict(ga_options or {})
    best_response_options = dict(best_response_options or {})
    trajectory, summaries, trajectory_forms = [], [], []
    warm_starts = list(warm_starts)

    def record_trajectory(search_form, candidates):
        trajectory_forms.append(search_form.spec([]))
        index = len(trajectory_forms) - 1
        trajectory.extend((index, list(candidate)) for candidate in candidates)

    if optimizer in ("genetic", "genetic+best_response"):
        if optimizer == "genetic":
            search_form = form
        else:
            search_form = build_threshold_form(
                genetic_form or (form.label if form.kind in GENETIC_FORMS else "knots6"), training)
        best = (None, -np.inf)
        for restart in range(num_restarts):
            search = run_genetic_search(
                training, None, num_bidders, num_goods, objective_mechanism=objective_mechanism,
                threshold_form=search_form,
                seed=None if ga_seed is None else ga_seed + restart,
                initial_solution=initial_solution if restart == 0 else None,
                verbose=verbose, **ga_options)
            record_trajectory(search_form, search.trajectory)
            summaries.append(search.summary)
            if search.welfare > best[1]:
                best = (search.coefficients, search.welfare)
        if optimizer == "genetic":
            return {"form": form, "parameters": list(best[0]), "welfare": best[1],
                    "optimizer": optimizer, "trajectory": trajectory,
                    "trajectory_forms": trajectory_forms, "search_summaries": summaries}
        genetic_threshold = search_form.threshold(best[0])
        genetic_threshold.start_name = f"genetic ({search_form.label})"
        warm_starts.insert(0, genetic_threshold)

    search = run_best_response_ascent(
        training, form, num_bidders, num_goods, objective_mechanism=objective_mechanism,
        warm_starts=warm_starts, seed=ga_seed, verbose=verbose, **best_response_options)
    record_trajectory(form, search.trajectory)
    summaries.append(search.summary)
    return {"form": form, "parameters": list(search.coefficients), "welfare": search.welfare,
            "optimizer": optimizer, "trajectory": trajectory,
            "trajectory_forms": trajectory_forms, "search_summaries": summaries}


def _sensitivity_row(scale, spec, training_welfare, welfare):
    row = {"bandwidth_scale": scale, "tau_parameters": list(spec["parameters"]),
           "threshold_spec": spec, "training_welfare": training_welfare}
    row.update({f"welfare_{m}": welfare[m] for m in MECHANISMS})
    first_best_gain = welfare["first_best"] - welfare["vcg"]
    for mechanism in AUDITED_MECHANISMS:
        gain = welfare[mechanism] - welfare["vcg"]
        row[f"gain_{mechanism}"] = gain
        row[f"share_{mechanism}"] = gain / first_best_gain if first_best_gain else float("nan")
    return row


def bandwidth_sensitivity(prepared, bandwidth_scales, threshold_form, num_bidders, num_goods,
                          search_options, base_result):
    """Redesign τ (and the IVCG-PA scores) on the training half smoothed with each
    bandwidth multiplier, and evaluate each design exactly on the same evaluation
    distribution. One row per bandwidth scale, including the base design.

    search_options: keyword options of optimize_threshold
    base_result: (threshold spec, training welfare, mechanism -> exact welfare) of
    the base design"""
    settings = prepared.settings
    base_spec, base_training_welfare, base_welfare = base_result
    rows = [_sensitivity_row(settings["bandwidth_scale"], base_spec, base_training_welfare,
                             base_welfare)]
    for scale in bandwidth_scales:
        if scale == settings["bandwidth_scale"]:
            continue
        training = sample_distribution(prepared.training_sample, "kernel", scale,
                                       num_externality_nodes=settings["num_externality_nodes"],
                                       num_valuation_cells=settings["num_valuation_cells"])
        result = optimize_threshold(training, threshold_form, num_bidders, num_goods,
                                    **search_options)
        threshold = result["form"].threshold(result["parameters"])
        ivcg_rule = ivcg_scoring_rule_for(training, threshold, num_bidders, num_goods)
        welfare = exact_welfare_by_mechanism(prepared.evaluation, threshold, num_bidders,
                                             num_goods, ivcg_rule)
        rows.append(_sensitivity_row(scale, result["form"].spec(result["parameters"]),
                                     result["welfare"], welfare))
    return pd.DataFrame(rows).sort_values("bandwidth_scale").set_index("bandwidth_scale")
