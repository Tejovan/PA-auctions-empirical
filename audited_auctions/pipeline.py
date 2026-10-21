"""
pipeline.py

Building blocks of one combination of a run (used by experiments/compute.py):

  prepare_distributions   build the distribution τ is designed on (training)
                          and the one it is evaluated on (evaluation)
  optimize_threshold      search for τ in a threshold form (threshold_forms.py)
                          with the exact objective: genetic search,
                          best-response ascent, both, or the symmetric
                          relaxation followed by best-response ascent
  bandwidth_sensitivity   redesign τ on the training half smoothed with other
                          bandwidths and evaluate each design on the test half

Synthetic families: training = evaluation = the family's piecewise-uniform
distribution (H is known; no sampling uncertainty). Optional grid-refinement
check of the discretization error.

Continuous evaluation (continuous=True, the default of runs): the search uses
the piecewise-uniform training distribution, where the best-response ascent's
per-node best responses and gradients are exact. Everything after the search
uses continuous versions (TypeDistribution.continuous_version): the evaluation
distribution (unless atomic), and the design distribution on which the IVCG-PA
scoring rule, the ironing figure, the marginal-contribution maps and the
participation gap are computed. This removes the many small ironing bands of
the grid (see type_distributions.py).

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

import time
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
from .parallel import split_workers, parallel_map
from .symmetric_relaxation import symmetric_relaxation
from .threshold_evaluation import (AUDITED_MECHANISMS, MECHANISMS, expected_welfare,
                                   ivcg_scoring_rule_for)
from .threshold_forms import BEST_RESPONSE_FORMS, GENETIC_FORMS, build_threshold_form
from .type_distributions import SYNTHETIC_FAMILIES, TypeDistribution, synthetic_distribution

SYNTHETIC_GRID_DEFAULTS = {"num_externality_nodes": 151, "num_valuation_cells": 150}
EMPIRICAL_GRID_DEFAULTS = {"num_externality_nodes": 241, "num_valuation_cells": 400}
EVALUATION_CHOICES = ("atomic", "smoothed")
OPTIMIZERS = ("auto", "genetic", "best_response", "genetic+best_response", "relaxation")


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
    design: Optional[TypeDistribution] = None  # default: training

    def __post_init__(self):
        if self.design is None:
            self.design = self.training

    @property
    def is_empirical(self) -> bool:
        return self.test_sample is not None


def prepare_distributions(source, externality_scale=1.0, valuation_scale=1.0,
                          truncation_quantile=0.995, evaluation_truncation_quantile=None,
                          train_fraction=0.5, split_seed=0, smoothing="kernel",
                          bandwidth_scale=1.0, evaluation="atomic", num_externality_nodes=None,
                          num_valuation_cells=None, family_options=None, continuous=False):
    """Distributions for one cell; `source` is a synthetic family name or a CSV path.

    continuous: evaluate on continuous versions (module notes); the training
    distribution of the search stays piecewise uniform."""
    if is_synthetic_source(source):
        grid = {key: value for key, value in (("num_externality_nodes", num_externality_nodes),
                                              ("num_valuation_cells", num_valuation_cells))
                if value is not None}
        grid = {**SYNTHETIC_GRID_DEFAULTS, **grid}
        if source == "example_n":
            grid.pop("num_externality_nodes")
        distribution = synthetic_distribution(source, externality_scale=externality_scale,
                                              **grid, **(family_options or {}))
        design = distribution.continuous_version() if continuous else distribution
        return PreparedDistributions(
            source=source, training=distribution, evaluation=design, design=design,
            test_sample=None,
            settings={"kind": "synthetic", "family": source, "externality_scale": externality_scale,
                      "grid": grid, "family_options": dict(family_options or {}),
                      "continuous": continuous})

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
    training = sample_distribution(train, **smoothing_options)
    evaluated = sample_distribution(test, **evaluation_options)
    return PreparedDistributions(
        source=str(source),
        training=training,
        evaluation=evaluated.continuous_version() if continuous else evaluated,
        design=training.continuous_version() if continuous else training,
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
                  "continuous": continuous, **smoothing_options},
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
    if optimizer != "genetic" and form_kind not in BEST_RESPONSE_FORMS:  # incl. relaxation
        raise ValueError(f"best-response ascent needs a form in {BEST_RESPONSE_FORMS}")
    return optimizer


def _genetic_restart(options):
    """One genetic restart (module level, so that restarts can run in worker processes)."""
    return run_genetic_search(**options)


def optimize_threshold(training, threshold_form, num_bidders, num_goods,
                       objective_mechanism="vcg_pa", optimizer="auto", ga_options=None,
                       ga_seed=None, num_restarts=1, initial_solution=None, genetic_form=None,
                       best_response_options=None, warm_starts=(), verbose=True,
                       time_limit=None, workers=1):
    """Search for τ in threshold_form on the training distribution.

    threshold_form : label ("poly2", "knots6", "bins10", "nodes") or polynomial degree
    optimizer      : "auto", "genetic", "best_response", "genetic+best_response"
                     (genetic search in genetic_form, default the same form or knots6
                     for nodes, then best-response ascent started from its result
                     as well as from the default starts) or "relaxation" (the
                     thresholds of the best symmetric participation rule for VCG-PA,
                     symmetric_relaxation.py, projected on the form, then
                     best-response ascent from that single start; piecewise-uniform
                     training distributions with k < n)
    initial_solution : genetic search warm start (parameters of the genetic form)
    warm_starts    : threshold functions used as extra best-response starts
    best_response_options : keyword options of run_best_response_ascent (starts,
                     max_iterations, relative_tolerance, short_genetic_generations)
    time_limit     : seconds for the whole search; past it, every stage stops gracefully
                     and the best τ found so far is returned (None: no limit)
    workers        : processes: the genetic restarts run simultaneously (the workers
                     left per restart run its Nelder–Mead starts), then the
                     best-response starts
    Returns a dict: form, parameters, welfare, optimizer, search_summaries,
    benchmarks (training welfare of VCG and the first best), and the
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
    deadline = None if time_limit is None else time.time() + time_limit
    benchmarks = {m: expected_welfare(training, lambda e: np.zeros(np.shape(e)), num_bidders,
                                      num_goods, m) for m in ("vcg", "first_best")}
    gain = benchmarks["first_best"] - benchmarks["vcg"]
    scale = gain if gain > 1e-12 * max(abs(benchmarks["first_best"]), 1.0) else None

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
        processes, per_restart = split_workers(workers, num_restarts)
        jobs = [dict(type_distribution=training, polynomial_degree=None,
                     num_bidders=num_bidders, num_goods=num_goods,
                     objective_mechanism=objective_mechanism, threshold_form=search_form,
                     seed=None if ga_seed is None else ga_seed + restart,
                     initial_solution=initial_solution if restart == 0 else None,
                     verbose=verbose and processes == 1, deadline=deadline,
                     welfare_scale=scale, workers=per_restart, **ga_options)
                for restart in range(num_restarts)]
        for search in parallel_map(_genetic_restart, jobs, processes):
            record_trajectory(search_form, search.trajectory)
            summaries.append(search.summary)
            if search.welfare > best[1]:
                best = (search.coefficients, search.welfare)
        if optimizer == "genetic":
            return {"form": form, "parameters": list(best[0]), "welfare": best[1],
                    "optimizer": optimizer, "trajectory": trajectory, "benchmarks": benchmarks,
                    "trajectory_forms": trajectory_forms, "search_summaries": summaries}
        genetic_threshold = search_form.threshold(best[0])
        genetic_threshold.start_name = f"genetic ({search_form.label})"
        warm_starts.insert(0, genetic_threshold)

    relaxation_summary = None
    if optimizer == "relaxation":
        if training.is_atomic or num_goods >= num_bidders:
            raise ValueError("the relaxation needs a piecewise-uniform distribution and k < n")
        begin = time.time()
        key = ("symmetric_relaxation", num_bidders, num_goods)
        if key not in training.cache:
            training.cache[key] = symmetric_relaxation(training, num_bidders, num_goods)
        relaxation = training.cache[key]
        relaxation_summary = {"value": relaxation.value,
                              "threshold_rule": bool(relaxation.is_threshold_rule),
                              "largest_drop": relaxation.largest_drop,
                              "seconds": time.time() - begin}
        best_response_options.update(starts=["relaxation"], num_random_starts=0)
        warm_starts = []
    search = run_best_response_ascent(
        training, form, num_bidders, num_goods, objective_mechanism=objective_mechanism,
        warm_starts=warm_starts, seed=ga_seed, verbose=verbose, deadline=deadline, scale=scale,
        workers=workers, **best_response_options)
    record_trajectory(form, search.trajectory)
    if relaxation_summary is not None:
        search.summary["relaxation"] = relaxation_summary
    summaries.append(search.summary)
    return {"form": form, "parameters": list(search.coefficients), "welfare": search.welfare,
            "optimizer": optimizer, "trajectory": trajectory, "benchmarks": benchmarks,
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
        design = training.continuous_version() if settings.get("continuous") else training
        ivcg_rule = ivcg_scoring_rule_for(design, threshold, num_bidders, num_goods)
        welfare = exact_welfare_by_mechanism(prepared.evaluation, threshold, num_bidders,
                                             num_goods, ivcg_rule)
        rows.append(_sensitivity_row(scale, result["form"].spec(result["parameters"]),
                                     result["welfare"], welfare))
    return pd.DataFrame(rows).sort_values("bandwidth_scale").set_index("bandwidth_scale")
