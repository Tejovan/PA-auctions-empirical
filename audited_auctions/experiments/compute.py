"""
compute.py

Computation of one combination of a run, and the files it saves. Nothing here
plots: figures are drawn afterwards from the saved files (report.py), so they
can be redrawn without recomputing.

Steps for one combination (distribution at ζ, market n, k, design):
  1. distributions: training (τ and the IVCG-PA scores are designed on it) and
     evaluation (outcomes are reported on it); identical for synthetic families
  2. τ: searched (optimize runs) or given (evaluate runs)
  3. exact expected outcomes of VCG, VCG-PA, IVCG-PA and the first best on the
     evaluation distribution; per-auction outcomes of simulated auctions
     (common random numbers: the same draws for every design of a market)
  4. empirical data: bootstrap over the test rows (exact outcomes per resample)
  5. diagnostics: first-order check, marginal-contribution map, ironing,
     interim allocations, checks with verdicts, search history

Files in the combination's folder:
  tau.json             τ, rebuildable with load_threshold_file
  tau.csv              τ(e) and the penalties r(e) at the externality nodes
  results.json         inputs, exact outcomes, comparisons with VCG, bootstrap
                       intervals, checks, search summary, timing
  first-order.csv      per externality node and mechanism: penalty, Pigouvian
                       charge, marginal contribution of the threshold type
  data.npz             arrays for the figures: τ and penalties on the plotting
                       grid, simulated per-auction outcomes, bootstrap resamples,
                       ironing segments, interim allocations, marginal-contribution
                       maps, tried thresholds of the search
  grid-refinement.csv, bandwidth-sensitivity.csv   when those checks run
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..best_response_ascent import first_order_check, participation_gap
from ..empirical import bootstrap_welfare
from ..marginal_contribution import RivalWelfareLoss
from ..pipeline import (
    bandwidth_sensitivity,
    designed_ivcg_rule,
    optimize_threshold,
    prepare_distributions,
)
from ..threshold_evaluation import (
    AUDITED_MECHANISMS,
    MECHANISMS,
    OUTCOME_METRICS,
    PARTICIPANT_AUDIT_MECHANISMS,
    build_mechanism_model,
    evaluate_threshold,
    exact_outcomes,
    grid_refinement_check,
    penalty_from_table,
    tau_independent_outcomes,
)
from ..symmetric_relaxation import symmetric_relaxation
from ..threshold_forms import (
    ThresholdForm,
    ex_ante_cutoff,
    ex_ante_guarantee,
    externality_nodes,
    threshold_from_spec,
)
from ..threshold_functions import tabulated_threshold
from .config import CODE_ROOT

# Metrics shown in figures and tables (OUTCOME_METRICS without the counts).
VALUE_METRICS = ("welfare", "valuation_welfare", "externality_welfare", "revenue",
                 "penalty_paid", "bidder_utility")

_PREPARED = {}  # process-local cache: (distribution label, ζ) -> PreparedDistributions


# ─── Distributions and plotting geometry ────────────────────────────────────────


def prepared_distributions(distribution, zeta, split_seed):
    """Training and evaluation distributions of a resolved distribution entry at ζ."""
    key = (distribution["label"], zeta, split_seed)
    if key in _PREPARED:
        return _PREPARED[key]
    grid = {"num_externality_nodes": distribution["num_externality_nodes"],
            "num_valuation_cells": distribution["num_valuation_cells"]}
    if distribution["kind"] == "synthetic":
        prepared = prepare_distributions(distribution["source"], externality_scale=zeta,
                                         family_options=distribution["options"], **grid)
    else:
        path = Path(distribution["path"])
        path = path if path.is_absolute() or path.exists() else CODE_ROOT / path
        prepared = prepare_distributions(
            str(path), externality_scale=zeta,
            valuation_scale=distribution["valuation_scale"],
            truncation_quantile=distribution["truncation_quantile"],
            evaluation_truncation_quantile=distribution["evaluation_truncation_quantile"],
            train_fraction=distribution["train_fraction"], split_seed=split_seed,
            smoothing=distribution["smoothing"], bandwidth_scale=distribution["bandwidth_scale"],
            evaluation=distribution["evaluation"], **grid)
    _PREPARED[key] = prepared
    return prepared


def plot_geometry(prepared):
    """Axis limits and grids of the type plane, shared by all figures of a distribution.

    Limits cover the training distribution between its 0.1% and 99.9% quantiles
    (synthetic) or 2.5% and 97.5% quantiles (empirical, heavy-tailed), plus a margin.
    """
    distribution = prepared.training
    tail = 0.001 if not prepared.is_empirical else 0.025
    quantile = distribution.weighted_quantile
    e_low = quantile(distribution.externalities, tail)
    e_high = quantile(distribution.externalities, 1 - tail)
    v_tail = 0.001 if not prepared.is_empirical else 0.005  # log axis: the tail fits
    v_low = quantile(distribution.valuation_lower, v_tail)
    v_high = min(quantile(distribution.valuation_upper, 1 - v_tail),
                 distribution.maximum_valuation)
    if e_high <= e_low:
        e_low, e_high = e_low - 0.5, e_high + 0.5
    # Heavy-tailed valuations (empirical data, heavy_tail family): single-panel type plots
    # use a logarithmic valuation axis from the v_tail quantile of the valuations.
    heavy_tailed = prepared.is_empirical or prepared.settings.get("family") == "heavy_tail"
    v_log_lower = max(quantile(distribution.valuation_midpoints, v_tail), 1e-12)
    e_pad, v_pad = 0.04 * (e_high - e_low), 0.04 * (v_high - v_low)
    e_limits = (e_low - e_pad, e_high + e_pad)
    v_limits = (v_low - v_pad, v_high + v_pad)
    return {"e_limits": np.array(e_limits), "v_limits": np.array(v_limits),
            "e_grid": np.linspace(*e_limits, 400), "v_grid": np.linspace(*v_limits, 240),
            "v_log": heavy_tailed, "v_log_lower": v_log_lower,
            "v_linear_upper": quantile(distribution.valuation_midpoints, 0.975)}


def type_background(prepared, geometry):
    """Arrays for the type-distribution background of the figures (saved as types.npz)."""
    distribution = prepared.training
    e_limits, v_limits = geometry["e_limits"], geometry["v_limits"]
    grid = distribution.density_grid
    if grid is not None:
        e_axis, v_axis = grid["externality_grid"], grid["valuation_grid"]
        density = grid["density"]
        shading = "log" if distribution.metadata.get("smoothing") == "kernel" else "linear"
    else:
        density, e_edges, v_edges = np.histogram2d(
            distribution.externalities, distribution.valuation_midpoints, bins=[90, 90],
            range=[e_limits, v_limits], weights=distribution.probabilities)
        density = density.T
        e_axis, v_axis = 0.5 * (e_edges[1:] + e_edges[:-1]), 0.5 * (v_edges[1:] + v_edges[:-1])
        shading = "log"
    nodes, masses, _ = externality_nodes(distribution)
    if prepared.is_empirical:
        settings = prepared.settings
        description = (f"types: {settings['smoothing']}-smoothed training half "
                       f"({settings['num_train_rows']:,} rows)"
                       if settings["smoothing"] == "kernel"
                       else f"types: training half ({settings['num_train_rows']:,} rows)")
    else:
        description = (f"types: {prepared.source} on a {len(nodes)} × "
                       f"{prepared.settings['grid'].get('num_valuation_cells')} grid")
    return {**geometry, "background_e": np.asarray(e_axis), "background_v": np.asarray(v_axis),
            "background_density": np.asarray(density), "shading": shading,
            "description": description, "externality_nodes": nodes, "node_masses": masses,
            "maximum_valuation": distribution.maximum_valuation}


# ─── Thresholds ─────────────────────────────────────────────────────────────────


def threshold_record(label, threshold_kind, payload, context):
    """Contents of tau.json: τ in a form load_threshold_file rebuilds."""
    return {"label": label, "kind": threshold_kind, **payload, "context": context}


def load_threshold_file(path):
    """Threshold function stored in a tau.json file."""
    with open(path) as file:
        record = json.load(file)
    return threshold_from_record(record)


def threshold_from_record(record):
    if record["kind"] == "form":
        return threshold_from_spec(record["spec"])
    if record["kind"] == "tabulated":
        values = [np.inf if v is None else v for v in record["values"]]
        return tabulated_threshold(record["externalities"], values)
    raise ValueError(f"unknown threshold kind {record['kind']!r}")


def given_threshold(entry, training, num_bidders, num_goods):
    """(threshold function, tau.json payload, kind) of a [[thresholds]] entry (evaluate runs).

    cutoff: τ(e) = max(0, w − e) (excluded if w − e > v̄); "ex_ante" computes the welfare
    cutoff w* of the ex ante relaxation on the training distribution for this n, k.
    """
    source, value = entry["source"], entry["value"]
    nodes, _, _ = externality_nodes(training)
    externality_range = (float(nodes[0]), float(nodes[-1]))
    if source == "cutoff":
        cutoff = (ex_ante_cutoff(training, num_bidders, num_goods) if value == "ex_ante"
                  else float(value))
        form = ThresholdForm("cutoff", 1, np.empty(0), float(training.maximum_valuation),
                             externality_range)
        return form.threshold([cutoff]), {"spec": form.spec([cutoff])}, "form"
    if source in ("polynomial", "constant"):
        coefficients = [float(value)] if source == "constant" else [float(c) for c in value]
        form = ThresholdForm("polynomial", len(coefficients) - 1, np.empty(0),
                             float(training.maximum_valuation), externality_range)
        spec = form.spec(coefficients)
        return form.threshold(coefficients), {"spec": spec}, "form"
    if source == "tabulated":
        externalities = [float(e) for e in value["externalities"]]
        values = [float(v) for v in value["values"]]
        payload = {"externalities": externalities,
                   "values": [None if np.isinf(v) else v for v in values]}
        return tabulated_threshold(externalities, values), payload, "tabulated"
    path = Path(value)
    path = path if path.is_absolute() or path.exists() else CODE_ROOT / path
    with open(path) as file:
        record = json.load(file)
    payload = {key: record[key] for key in ("spec", "externalities", "values") if key in record}
    payload["copied_from"] = str(path)
    return threshold_from_record(record), payload, record["kind"]


def rescaled_threshold(threshold, externality_ratio):
    """τ(e) = τ_prev(e · ζ_prev / ζ): the previous ζ's optimum as a warm start."""

    from ..threshold_functions import Rescaled

    return Rescaled(threshold, externality_ratio, start_name="previous ζ")


# ─── One combination ────────────────────────────────────────────────────────────


def _search(combination, prepared, config, warm_start):
    optimizer = combination.optimizer
    genetic = dict(optimizer["genetic"])
    num_restarts = genetic.pop("num_restarts")
    genetic["sol_per_pop"] = genetic.pop("population_size")
    warm_starts = [warm_start] if (warm_start is not None
                                   and optimizer["warm_start_across_zeta"]
                                   and optimizer["method"] != "genetic") else []
    search_options = dict(objective_mechanism=combination.objective,
                          optimizer=optimizer["method"], ga_options=genetic,
                          ga_seed=config["seeds"]["search"], num_restarts=num_restarts,
                          genetic_form=optimizer["genetic_form"],
                          best_response_options=dict(optimizer["best_response"]), verbose=False,
                          time_limit=(60.0 * optimizer["time_limit_minutes"]
                                      if optimizer["time_limit_minutes"] else None),
                          workers=config.get("inner_workers", 1))
    begin = time.time()
    result = optimize_threshold(prepared.training, combination.threshold_form,
                                combination.num_bidders, combination.num_goods,
                                warm_starts=warm_starts, **search_options)
    result["seconds"] = time.time() - begin
    result["begin"] = begin
    result["search_options"] = search_options
    return result


def _trace(result):
    """Convergence trace of a search: best welfare so far against seconds."""
    points = []
    for index, summary in enumerate(result["search_summaries"]):
        source = summary.get("optimizer", "search")
        for entry in summary.get("trace", []):
            points.append({"optimizer": source, "run": index, "stage": entry["stage"],
                           "seconds": entry["time"] - result["begin"],
                           "welfare": entry["welfare"]})
    return points


def _history(result):
    """Per start of the search: welfare and stopping diagnostics at each iteration
    (generation for the genetic stage), for search/iterations.pdf."""
    lines = []
    for index, summary in enumerate(result["search_summaries"]):
        if summary.get("optimizer") == "best_response":
            lines.extend({"optimizer": "best_response", "start": s["start"],
                          "stop_reason": s["stop_reason"], "points": s["history"]}
                         for s in summary["starts"] if s["history"])
            continue
        lines.append({"optimizer": "genetic", "start": f"genetic run {index}",
                      "stop_reason": summary["stop_reason"], "points": summary["history"]})
        lines.extend({"optimizer": "nelder_mead", "start": f"run {index}, local start {i}",
                      "stop_reason": s["stop_reason"], "points": s["history"]}
                     for i, s in enumerate(summary.get("local_starts", [])))
    for line in lines:
        begin = line["points"][0]["time"] if line["points"] else 0.0
        line["points"] = [{**{k: v for k, v in point.items() if k != "time"},
                           "seconds": point["time"] - begin} for point in line["points"]]
    return lines


def _search_summary(result):
    """Search statistics for results.json (without parameter vectors and traces)."""
    starts = []
    for summary in result["search_summaries"]:
        if summary.get("optimizer") == "best_response":
            starts.extend({"optimizer": "best_response", "start": s["start"],
                           "start_welfare": s["start_welfare"], "welfare": s["welfare"],
                           "iterations": s["iterations"], "seconds": s["seconds"],
                           "polishes": len(s["polish_gains"]), "stop_reason": s["stop_reason"],
                           **({key: s["history"][-1][key]
                               for key in ("gap", "stationarity", "max_marginal")}
                              if s["history"] else {})}
                          for s in summary["starts"])
        else:
            starts.append({"optimizer": "genetic", "start": "genetic stage",
                           "welfare": summary["genetic_welfare"],
                           "generations": summary["generations_run"],
                           "stop_reason": summary["stop_reason"],
                           "evaluations": summary.get("num_objective_evaluations")})
            starts.extend({"optimizer": "nelder_mead", "start": f"local start {i}",
                           "start_welfare": s["start_welfare"], "welfare": s["welfare"],
                           "evaluations": s["evaluations"], "converged": s["converged"],
                           "stop_reason": s["stop_reason"]}
                          for i, s in enumerate(summary.get("local_starts", [])))
    return {"method": result["optimizer"], "training_welfare": result["welfare"],
            "training_benchmarks": result["benchmarks"],
            "time_limit_seconds": result["search_options"]["time_limit"],
            "seconds": result["seconds"], "starts": starts,
            "num_tried": len(result["trajectory"])}


def _tried_thresholds(result, e_grid, max_curves):
    """τ of up to max_curves candidates of the search on the plotting grid (NaN = excluded)."""
    trajectory = result["trajectory"]
    if not trajectory:
        return np.empty((0, len(e_grid)), dtype=np.float32)
    rng = np.random.default_rng(0)
    if len(trajectory) > max_curves:
        trajectory = [trajectory[i] for i in sorted(rng.choice(len(trajectory), max_curves,
                                                               replace=False))]
    forms = result["trajectory_forms"]
    curves = np.array([threshold_from_spec(forms[i], p)(e_grid) for i, p in trajectory])
    return np.where(np.isfinite(curves), curves, np.nan).astype(np.float32)


def _comparisons(exact):
    """Changes over VCG of every metric, share of the first-best gain, relative revenue change."""
    change = {metric: {m: exact[m][metric] - exact["vcg"][metric] for m in AUDITED_MECHANISMS}
              for metric in OUTCOME_METRICS}
    first_best_gain = change["welfare"]["first_best"]
    share = {m: (change["welfare"][m] / first_best_gain if abs(first_best_gain) > 1e-15
                 else None) for m in AUDITED_MECHANISMS}
    vcg_revenue = exact["vcg"]["revenue"]
    revenue_relative = {m: (change["revenue"][m] / vcg_revenue if abs(vcg_revenue) > 1e-15
                            else None) for m in AUDITED_MECHANISMS}
    return {"change_over_vcg": change, "share_of_first_best_gain": share,
            "revenue_relative_change": revenue_relative}


def _bootstrap(prepared, threshold, combination, config, ivcg_rule):
    """Bootstrap resamples (DataFrame) and percentile intervals of levels, changes and shares."""
    settings = prepared.settings
    evaluation = config["evaluation"]
    result = bootstrap_welfare(
        prepared.test_sample, threshold, combination.num_bidders, combination.num_goods,
        num_resamples=evaluation["num_bootstrap"], seed=config["seeds"]["bootstrap"],
        smoothing=settings["evaluation_smoothing"], bandwidth_scale=settings["bandwidth_scale"],
        confidence_level=evaluation["confidence_level"], ivcg_scoring_rule=ivcg_rule,
        all_outcomes=evaluation["bootstrap_all_outcomes"],
        num_externality_nodes=settings["num_externality_nodes"],
        num_valuation_cells=settings["num_valuation_cells"],
        workers=config.get("inner_workers", 1))
    resamples = result.resamples
    tail = (1 - evaluation["confidence_level"]) / 2

    def interval(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return None
        return [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]

    metrics = [m for m in VALUE_METRICS if f"{m}_vcg" in resamples]
    levels = {metric: {m: interval(resamples[f"{metric}_{m}"]) for m in MECHANISMS}
              for metric in metrics}
    changes = {metric: {m: interval(resamples[f"{metric}_{m}"] - resamples[f"{metric}_vcg"])
                        for m in AUDITED_MECHANISMS} for metric in metrics}
    shares = {m: interval(resamples[f"share_{m}"]) for m in AUDITED_MECHANISMS}
    summary = {"num_resamples": int(len(resamples)),
               "confidence_level": evaluation["confidence_level"],
               "level_intervals": levels, "change_intervals": changes,
               "share_intervals": shares}
    return resamples, summary


def _marginal_contribution_map(training, threshold, combination, geometry):
    """M^τ(e, v) = (v + e) X^τ(v) − L^τ(v) on a grid, for VCG-PA and IVCG-PA (fixed scores)."""
    e_values = np.linspace(*geometry["e_limits"], 120)
    v_low = max(geometry["v_limits"][0], 0.0)
    v_high = min(geometry["v_limits"][1], training.maximum_valuation)
    if geometry["v_log"]:  # log-spaced, matching the logarithmic valuation axis
        v_values = np.geomspace(geometry["v_log_lower"], v_high, 180)
    else:
        v_values = np.linspace(v_low, v_high, 160)
    maps = {"mc_e": e_values, "mc_v": v_values}
    for mechanism in PARTICIPANT_AUDIT_MECHANISMS:
        model = build_mechanism_model(mechanism, training, threshold, combination.num_bidders,
                                      combination.num_goods)
        allocation, loss = RivalWelfareLoss(model)(v_values)
        maps[f"mc_{mechanism}"] = ((v_values[None, :] + e_values[:, None]) * allocation[None, :]
                                   - loss[None, :]).astype(np.float32)
    return maps


def _ironing(training, threshold, combination):
    """IVCG-PA design segments in quantile space, and whether ironing binds."""
    model = build_mechanism_model("ivcg_pa", training, threshold, combination.num_bidders,
                                  combination.num_goods)
    curves = model.welfare_contribution_curves()
    welfare = np.concatenate([curves["welfare_start"], curves["welfare_end"]])
    score = np.concatenate([curves["score_start"], curves["score_end"]])
    scale = max(1.0, float(np.nanmax(np.abs(welfare))))
    finite = np.isfinite(score)
    binds = bool(np.any(np.abs(welfare[finite] - score[finite]) > 1e-9 * scale))
    return {f"iron_{key}": np.asarray(value) for key, value in curves.items()}, binds


def _sufficiency(training, threshold, combination, relaxation_wanted):
    """Distance of τ from the sufficient condition for VCG-PA (paper v3 Thm
    [thm:vcg-sufficient]), on the training distribution.

    participation_gap: G(τ) = E[M^- 1{participate} + M^+ 1{abstain}], the largest welfare
      gain from one bidder changing its participation set (best_response_ascent.
      participation_gap); share = G / (FB − W_VCG). G = 0 for VCG-PA if and only if τ
      maximizes VCG-PA welfare over all symmetric participation rules.
    relaxation: best symmetric participation rule for VCG-PA (symmetric_relaxation), an
      estimate of an upper bound on VCG-PA welfare over all common τ; share_attained =
      (W^VCG(τ) − W_VCG) / (relaxation − W_VCG). Piecewise-uniform distributions only.
    """
    n, k = combination.num_bidders, combination.num_goods
    vcg = tau_independent_outcomes(training, "vcg", n, k, welfare_only=True)
    gain = tau_independent_outcomes(training, "first_best", n, k, welfare_only=True) - vcg
    out = {}
    for mechanism in PARTICIPANT_AUDIT_MECHANISMS:
        try:
            total, _, wrong_mass = participation_gap(training, threshold, n, k, mechanism)
        except (ValueError, IndexError):
            continue
        out[f"participation_gap_{mechanism}"] = total
        out[f"participation_gap_share_{mechanism}"] = total / gain if abs(gain) > 1e-15 else None
        out[f"wrong_sign_mass_{mechanism}"] = wrong_mass
    if relaxation_wanted and not training.is_atomic and k < n:
        key = ("symmetric_relaxation", n, k)
        if key not in training.cache:
            training.cache[key] = symmetric_relaxation(training, n, k)
        relaxation = training.cache[key]
        welfare = exact_outcomes(training, threshold, n, k)["vcg_pa"]["welfare"]
        out.update({
            "relaxation_value": relaxation.value,
            "relaxation_threshold_rule": bool(relaxation.is_threshold_rule),
            "relaxation_largest_drop": relaxation.largest_drop,
            "training_welfare_vcg_pa": welfare,
            "relaxation_share_attained": ((welfare - vcg) / (relaxation.value - vcg)
                                          if relaxation.value - vcg > 1e-15 else None)})
    return out


def _check(path, name, value, tolerance, verdict, note):
    return {"path": path, "check": name, "value": value, "tolerance": tolerance,
            "verdict": verdict, "note": note}


def _checks(combination, config, prepared, threshold, evaluation, first_order, search_summary,
            sensitivity, refinement, bootstrap_summary, sufficiency=None):
    path, checks = combination.path, []
    tolerances = config["checks"]

    # Monte Carlo: simulated means agree with the exact expectations.
    simulated = evaluation.simulated
    if simulated is not None:
        worst = 0.0
        for mechanism in MECHANISMS:
            for metric in ("welfare", "revenue"):
                draws = simulated[mechanism][metric]
                error = draws.std(ddof=1) / np.sqrt(len(draws))
                difference = draws.mean() - evaluation.exact[mechanism][metric]
                z = 0.0 if error == 0 and abs(difference) < 1e-9 else abs(difference) / max(
                    error, 1e-300)
                worst = max(worst, z)
        checks.append(_check(path, "monte_carlo_vs_exact", worst, tolerances["monte_carlo_z"],
                             "pass" if worst < tolerances["monte_carlo_z"] else "fail",
                             "largest |z| of simulated mean − exact (welfare, revenue; "
                             "4 mechanisms)"))

    # First-order condition at interior thresholds.
    if first_order is not None:
        mechanisms = ([combination.objective] if combination.objective
                      else list(PARTICIPANT_AUDIT_MECHANISMS))
        rows = first_order[first_order.mechanism.isin(mechanisms)
                           & (first_order.status == "interior") & (first_order.mass > 0)]
        scale = max(float(prepared.training.maximum_valuation), 1e-12)
        worst = float(rows.marginal_contribution.abs().max() / scale) if len(rows) else 0.0
        restricted = combination.threshold_form not in ("nodes",)
        verdict = ("info" if restricted or combination.objective is None
                   else "pass" if worst < tolerances["first_order_tolerance"] else "fail")
        note = ("max |M^τ(e, τ(e))| / v̄ at interior thresholds; zero at a per-node optimum "
                "(paper: zero marginal contribution at interior optimal thresholds)")
        if verdict == "info":
            note += ("; restricted form or given τ: measures the distance from the "
                     "per-node optimum")
        checks.append(_check(path, "first_order", worst, tolerances["first_order_tolerance"],
                             verdict, note))

    # Different starts of the search.
    if search_summary is not None:
        finals = [s["welfare"] for s in search_summary["starts"] if s["welfare"] is not None]
        benchmarks = search_summary["training_benchmarks"]
        gain = benchmarks["first_best"] - benchmarks["vcg"]
        if len(finals) > 1:
            best = max(finals)
            spread = (best - min(finals)) / (gain if gain > 0 else max(abs(best), 1e-300))
            checks.append(_check(path, "search_starts_spread", spread, 1e-4, "info",
                                 "welfare spread of the final points of the starts, in units of "
                                 "the first-best gain FB − W_VCG (training); large = local optima "
                                 "(the best start is kept)"))
        reasons = [s["stop_reason"] for s in search_summary["starts"]]
        timed = sum("time" in reason for reason in reasons)
        capped = sum(reason in ("max_iterations", "max_generations", "max_evaluations")
                     for reason in reasons)
        checks.append(_check(path, "search_stop", timed + capped, 0,
                             "fail" if timed else "info" if capped else "pass",
                             f"{timed} stages stopped or skipped by the time limit, {capped} at "
                             f"an iteration cap, of {len(reasons)}; pass = every stage stopped "
                             f"by a convergence rule"))

    # Ex ante relaxation: IVCG-PA for τ* attains max{β_{n,k}, 1 − 1/e} of the first best
    # (paper Thm 8.9 [thm:k-goods]); checked on the training distribution, where τ* is set.
    entry = combination.threshold
    if entry is not None and entry["source"] == "cutoff" and entry["value"] == "ex_ante":
        training_outcomes = exact_outcomes(prepared.training, threshold, combination.num_bidders,
                                           combination.num_goods)
        ratio = (training_outcomes["ivcg_pa"]["welfare"]
                 / training_outcomes["first_best"]["welfare"])
        bound = ex_ante_guarantee(combination.num_bidders, combination.num_goods)
        atomic = prepared.training.is_atomic
        checks.append(_check(path, "ex_ante_guarantee", ratio, bound,
                             "info" if atomic else "pass" if ratio >= bound - 1e-9 else "fail",
                             "W*(τ*)/FB on the training distribution against max{β_{n,k}, "
                             "1 − 1/e} (paper Thm 8.9)" + ("; atomic: no boundary mixing"
                                                          if atomic else "")))

    # Sufficient condition for VCG-PA (paper v3 Thm [thm:vcg-sufficient]): informational.
    sufficiency = sufficiency or {}
    share = sufficiency.get("participation_gap_share_vcg_pa")
    if share is not None:
        checks.append(_check(path, "sufficient_condition", share, None, "info",
                             "G(τ)/(FB − W_VCG), G = largest welfare gain from one bidder "
                             "changing its participation set (M^τ < 0 participants, M^τ > 0 "
                             "abstainers); 0 iff τ is best among all symmetric participation "
                             "rules for VCG-PA"))
    attained = sufficiency.get("relaxation_share_attained")
    if attained is not None:
        rule = ("a threshold rule" if sufficiency["relaxation_threshold_rule"]
                else "not a threshold rule")
        checks.append(_check(path, "symmetric_relaxation", attained, None, "info",
                             "(W^VCG(τ) − W_VCG)/(best symmetric rule − W_VCG) on the training "
                             f"distribution (midpoint estimate of the bound); best symmetric "
                             f"rule is {rule}"))

    if refinement is not None:
        gain_base = refinement.gain_over_vcg_base.drop(index=refinement.index[0])
        change = refinement.gain_change.drop(index=refinement.index[0])
        relative = float((change.abs() / gain_base.abs().clip(lower=1e-12)).max())
        tolerance = tolerances["grid_relative_tolerance"]
        checks.append(_check(path, "grid_refinement", relative, tolerance,
                             "pass" if relative < tolerance else "fail",
                             f"largest |change of gain over VCG| / gain on a "
                             f"{tolerances['refinement_factor']}× finer grid (τ fixed)"))

    if sensitivity is not None:
        objective = combination.objective or "vcg_pa"
        gains = sensitivity[f"gain_{objective}"]
        spread = float(gains.max() - gains.min())
        interval = (bootstrap_summary or {}).get("change_intervals", {}).get(
            "welfare", {}).get(objective)
        width = None if interval is None else interval[1] - interval[0]
        checks.append(_check(path, "bandwidth_sensitivity", spread, width,
                             "info" if width is None else "pass" if spread <= width else "fail",
                             "spread of the gain over VCG across smoothing bandwidths, against "
                             "the width of its bootstrap interval"))
    return checks


def compute_combination(combination, config, run_folder, warm_start=None):
    """Compute one combination and write its files; returns (summary row, checks, threshold)."""
    begin = time.time()
    folder = Path(run_folder) / combination.path
    folder.mkdir(parents=True, exist_ok=True)
    n, k = combination.num_bidders, combination.num_goods
    prepared = prepared_distributions(combination.distribution, combination.externality_scale,
                                      config["seeds"]["split"])
    geometry = plot_geometry(prepared)
    arrays, timing = {}, {}

    # τ ------------------------------------------------------------------------
    search_result = search_summary = None
    if combination.threshold is None:
        search_result = _search(combination, prepared, config, warm_start)
        form = search_result["form"]
        parameters = search_result["parameters"]
        threshold = form.threshold(parameters)
        tau = threshold_record(combination.design_label, "form", {"spec": form.spec(parameters)},
                               combination.keys())
        search_summary = _search_summary(search_result)
        arrays["search_e"] = geometry["e_grid"][::2]
        arrays["search_tried"] = _tried_thresholds(
            search_result, arrays["search_e"], config["output"]["search_history_curves"])
        timing["search"] = search_result["seconds"]
    else:
        threshold, payload, kind = given_threshold(combination.threshold, prepared.training,
                                                   n, k)
        tau = threshold_record(combination.threshold["label"], kind, payload, combination.keys())

    # Exact outcomes and simulation ----------------------------------------------
    stage = time.time()
    ivcg_rule = designed_ivcg_rule(prepared, prepared.training, threshold, n, k)
    evaluation = evaluate_threshold(
        prepared.evaluation, threshold, n, k,
        num_simulated_draws=config["evaluation"]["num_simulated_auctions"],
        random_seed=config["seeds"]["simulation"], externality_grid=geometry["e_grid"],
        ivcg_scoring_rule=ivcg_rule)
    exact = evaluation.exact
    timing["evaluation"] = time.time() - stage

    bootstrap_summary = None
    if prepared.is_empirical and config["evaluation"]["num_bootstrap"] > 0:
        stage = time.time()
        resamples, bootstrap_summary = _bootstrap(prepared, threshold, combination, config,
                                                  ivcg_rule)
        arrays.update({f"boot_{column}": resamples[column].to_numpy()
                       for column in resamples.columns})
        timing["bootstrap"] = time.time() - stage

    # Diagnostics -----------------------------------------------------------------
    stage = time.time()
    first_order = None
    try:
        first_order = pd.concat(
            [first_order_check(prepared.training, threshold, n, k, mechanism).assign(
                mechanism=mechanism) for mechanism in PARTICIPANT_AUDIT_MECHANISMS],
            ignore_index=True)
        first_order["pigouvian_charge"] = first_order.penalty - first_order.marginal_contribution
        first_order.to_csv(folder / "first-order.csv", index=False)
    except (ValueError, IndexError) as error:  # e.g. atomic training data
        first_order_error = str(error)
    else:
        first_order_error = None
    sufficiency = _sufficiency(prepared.training, threshold, combination,
                               config["checks"]["symmetric_relaxation"])
    # The relaxation is solved on the training distribution; it bounds the evaluation
    # welfare only where the two coincide (synthetic data).
    sufficiency["relaxation_on_evaluation"] = not prepared.is_empirical
    try:
        arrays.update(_marginal_contribution_map(prepared.training, threshold, combination,
                                                 geometry))
    except (ValueError, IndexError):
        pass
    ironing_arrays, ironing_binds = _ironing(prepared.training, threshold, combination)
    arrays.update(ironing_arrays)
    allocation_bids = np.linspace(0.0, prepared.evaluation.maximum_valuation, 300)
    arrays["alloc_bids"] = allocation_bids
    for mechanism in PARTICIPANT_AUDIT_MECHANISMS:
        table = evaluation.penalty_tables[mechanism]["allocation_integral"]
        arrays[f"alloc_{mechanism}"] = table.interim_allocation(allocation_bids)

    refinement = None
    if config["checks"]["grid_refinement"] and not prepared.is_empirical:
        settings = prepared.settings
        refinement = grid_refinement_check(
            settings["family"], threshold, n, k, externality_scale=settings["externality_scale"],
            refinement_factor=config["checks"]["refinement_factor"], **settings["grid"],
            **settings["family_options"])
        refinement.to_csv(folder / "grid-refinement.csv")
    sensitivity = None
    if (search_result is not None and prepared.is_empirical
            and config["checks"]["bandwidth_sensitivity"]
            and prepared.settings["smoothing"] == "kernel"):
        sensitivity = bandwidth_sensitivity(
            prepared, config["checks"]["bandwidth_sensitivity"], combination.threshold_form, n,
            k, search_result["search_options"],
            base_result=(tau["spec"], search_result["welfare"],
                         {m: exact[m]["welfare"] for m in MECHANISMS}))
        sensitivity.drop(columns=["tau_parameters", "threshold_spec"]).to_csv(
            folder / "bandwidth-sensitivity.csv")
    checks = _checks(combination, config, prepared, threshold, evaluation, first_order,
                     search_summary, sensitivity, refinement, bootstrap_summary, sufficiency)
    timing["diagnostics"] = time.time() - stage

    # Files -------------------------------------------------------------------------
    curves = evaluation.penalty_curves
    arrays.update({"tau_grid": np.asarray(curves["threshold"], dtype=float),
                   "e_grid": geometry["e_grid"],
                   "penalty_vcg_pa": curves["vcg_pa"], "penalty_ivcg_pa": curves["ivcg_pa"]})
    for mechanism in MECHANISMS:
        for metric, values in evaluation.simulated[mechanism].items():
            arrays[f"sim_{mechanism}_{metric}"] = values
    np.savez_compressed(folder / "data.npz", **arrays)

    nodes, masses, _ = externality_nodes(prepared.training)
    node_thresholds = threshold(nodes)
    pd.DataFrame({
        "externality": nodes, "mass": masses,
        "threshold": np.where(np.isfinite(node_thresholds), node_thresholds, np.nan),
        "excluded": ~np.isfinite(node_thresholds),
        **{f"penalty_{m}": penalty_from_table(evaluation.penalty_tables[m], node_thresholds)
           for m in PARTICIPANT_AUDIT_MECHANISMS},
    }).to_csv(folder / "tau.csv", index=False)
    write_json(folder / "tau.json", {**tau, "training_welfare": (
        None if search_result is None else search_result["welfare"])})

    timing["total"] = time.time() - begin
    results = {
        "combination": combination.keys(),
        "distribution_settings": prepared.settings,
        "exact": exact,
        **_comparisons(exact),
        "bootstrap": bootstrap_summary,
        "num_simulated_auctions": config["evaluation"]["num_simulated_auctions"],
        "ivcg_design": "training half" if ivcg_rule is not None else "same distribution",
        "ironing_binds": ironing_binds,
        "search": None if search_summary is None else {
            **search_summary, "trace": _trace(search_result),
            "history": _history(search_result)},
        "first_order_error": first_order_error,
        "sufficiency": sufficiency,
        "checks": checks,
        "seconds": timing,
    }
    write_json(folder / "results.json", results)
    return summary_row(results), checks, threshold


def summary_row(results):
    """One flat row of summary.csv."""
    row = {key: value for key, value in results["combination"].items() if value is not None}
    exact = results["exact"]
    for metric in VALUE_METRICS + ("num_participants",):
        for mechanism in MECHANISMS:
            row[f"{metric}_{mechanism}"] = exact[mechanism][metric]
    for mechanism in AUDITED_MECHANISMS:
        row[f"welfare_gain_{mechanism}"] = results["change_over_vcg"]["welfare"][mechanism]
        row[f"share_{mechanism}"] = results["share_of_first_best_gain"][mechanism]
        row[f"revenue_change_{mechanism}"] = results["change_over_vcg"]["revenue"][mechanism]
    bootstrap = results["bootstrap"]
    if bootstrap is not None:
        for mechanism in MECHANISMS:
            interval = bootstrap["level_intervals"]["welfare"][mechanism]
            if interval:
                row[f"welfare_{mechanism}_lower"], row[f"welfare_{mechanism}_upper"] = interval
        for mechanism in AUDITED_MECHANISMS:
            interval = bootstrap["change_intervals"]["welfare"][mechanism]
            if interval:
                row[f"welfare_gain_{mechanism}_lower"], row[f"welfare_gain_{mechanism}_upper"] = (
                    interval)
    if results["search"] is not None:
        row["training_welfare"] = results["search"]["training_welfare"]
        row["search_seconds"] = results["search"]["seconds"]
    row["ironing_binds"] = results["ironing_binds"]
    sufficiency = results.get("sufficiency") or {}
    row.update({key: value for key, value in sufficiency.items()
                if key in ("participation_gap_share_vcg_pa", "participation_gap_share_ivcg_pa",
                           "relaxation_share_attained", "relaxation_threshold_rule")})
    if sufficiency.get("relaxation_on_evaluation") and "relaxation_value" in sufficiency:
        row["relaxation_value"] = sufficiency["relaxation_value"]
    row["seconds"] = results["seconds"]["total"]
    return row


# ─── JSON ───────────────────────────────────────────────────────────────────────


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    raise TypeError(f"not serializable: {type(value)}")


def write_json(path, data):
    with open(path, "w") as file:
        json.dump(data, file, indent=1, default=_json_default)


def read_json(path):
    with open(path) as file:
        return json.load(file)
