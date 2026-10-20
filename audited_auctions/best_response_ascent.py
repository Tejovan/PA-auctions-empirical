"""
best_response_ascent.py

Best-response ascent: search for a common threshold function τ (in a
threshold_forms.ThresholdForm of kind knots, bins or nodes) that maximizes the
exact expected welfare of VCG-PA or IVCG-PA.

Theory (paper arxiv_WEB v1, §7.1 [sec:choosing], §7.3 and App. E; LaTeX labels in brackets).
Hold the auction and the rivals' thresholds fixed. Then expected welfare is linear in one
bidder's participation set (Prop. 7.4 [prop:decomposition]):

    W(τ_i, τ_{-i}) = W(bidder i out) + Σ_nodes g_e(τ_i(e)),
    g_e(t) = E[1{v >= t, e_i = e} M^τ(e, v)],

with the marginal contribution M^τ(e, v) = A^τ(v) + e X^τ(v) (marginal_contribution.py).
The best response of one bidder maximizes Σ_e g_e(τ(e)) over the form
(Cor. 7.5 [cor:best-response]). For a common τ (all bidders move together), the derivative
of W in τ(e) is −n Pr[e] f(τ(e) | e) M^τ(e, τ(e)) (Prop. 7.9 [prop:zero-marginal-equal]),
so the best response
is an ascent direction but not an improvement guarantee: each iteration

  1. computes A^τ, X^τ exactly for the current τ (one model of the objective
     auction; for IVCG-PA the ironed scores of the current τ are held fixed,
     justified at a joint optimum, §7.6 [sec:joint]),
  2. computes the best response τ' within the form:
       nodes, bins : for each node (bin), the exact maximizer of g (of the sum
                     of g over the bin's nodes) over t ∈ [0, v̄] ∪ {v̄⁺}:
                     candidates are the breakpoints of the valuation grid and
                     the zeros of M where it changes sign from − to + (g' = −f M);
       knots       : dynamic programming over the knots (the payoff is a sum of
                     terms in pairs of adjacent knot values, plus the exclusion
                     cutoff), on a grid of knot values refined twice around the
                     optimum,
  3. moves τ ← τ + α (τ' − τ) in parameter space, with α the first of
     1, 1/2, ..., 1/64 that raises the exact objective W.

It stops when no step raises W, when the relative gain of two consecutive
iterations is below relative_tolerance, or after max_iterations.

Gradient polish. A profile of best responses need not maximize W over common
thresholds (a best response moves one bidder; the common τ moves all n), so the
ascent can stop short of the optimum. After it stops, L-BFGS-B maximizes W
over the form's parameters with the exact gradient

    ∂W/∂τ(e_j) = −n f_j(τ(e_j)) M^τ(e_j, τ(e_j))        (Prop. 7.9 [prop:zero-marginal-equal]),

f_j the density of v at node j (zero below 0 and above v̄), chained to the
form's parameters (knots: interpolation weights, cutoff held fixed; bins: sum
over the bin's nodes). If the polish raises W, the ascent restarts from its
result (at most polish_rounds times). Piecewise-uniform distributions only (W is
a step function of τ for atomic distributions).

Each start is run to convergence and the best result is kept. Starts:
  "zero"            τ ≡ 0 (VCG-PA without exclusions; welfare of VCG)
  "zero_welfare"    the zero-welfare threshold τ⁰(e) = max(0, −e), excluded if −e > v̄
                    (§7.2 [sec:restrictions]; the first best without competition,
                    Prop. 7.10 [prop:no-competition]); alias "no_competition"
  "short_genetic"   a short genetic search in the same form (knots6 for nodes)
plus any warm starts passed as threshold functions (for example the genetic
search's result, or the optimum at the previous externality scale of a run).

Stopping. The ascent stops for one of the stop_reason values: "no_improving_step",
"small_gain" (relative gain below relative_tolerance in two consecutive
iterations), "max_iterations" or "time_limit" (deadline, a time.time() value:
the current iteration or polish ends, later starts are skipped, and the best
point so far is kept). The polish (L-BFGS-B) stops on its projected gradient
(gtol), which is the first-order condition in the form's parameters.

Per-iteration diagnostics (history of each start), at the current τ:
  stationarity      v̄ Σ_j |∂W/∂θ_j| / scale, the projected gradient of W in the form's
                    parameters θ (at τ(e) = 0 and exclusion only the one-sided
                    derivative into [0, v̄] that raises W counts): the
                    rate of change of W if every parameter moves by v̄ in its
                    ascent direction, in units of scale (the first-best gain
                    FB − W_VCG when given, else |W|). Zero at a stationary point.
  max_marginal      largest violation, / v̄, over nodes with positive mass, of the
                    first-order conditions of an optimal common τ: |M^τ(e, τ(e))| at
                    interior thresholds (zero marginal contribution, Prop. 7.9
                    [prop:zero-marginal-equal]), max(0, −M(e, 0)) at τ(e) = 0 and
                    max(0, M(e, v̄)) at exclusion.
  gap               best-response gap Σ_e max_t g_e(t) − Σ_e g_e(τ(e)), in units of
                    scale: the gain one bidder could get by deviating alone.
Not zero at an optimum in general: a common optimum need not be a best response.

Diagnostics: first_order_check reports, at each externality node, the marginal
contribution of the threshold type, the penalty, the own-allocation charge −e X^τ,
the displaced welfare term, and the best-response gap.
"""

import time

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev

from .genetic_search import SearchResult, run_genetic_search
from .marginal_contribution import valuation_functions
from .threshold_evaluation import build_mechanism_model, expected_welfare
from .threshold_forms import BEST_RESPONSE_FORMS, build_threshold_form, externality_nodes

DEFAULT_STARTS = ("zero", "zero_welfare")
DEFAULT_POLISH_ROUNDS = 3
STEP_SIZES = tuple(0.5 ** m for m in range(7))
KNOT_GRID_SIZE = 48
KNOT_ZOOM_GRID_SIZE = 17
KNOT_ZOOM_PASSES = 2


# ─── Participation payoffs g ──────────────────────────────────────────────────


class ParticipationPayoffs:
    """g_G(t) = E[1{v >= t, type in group G} M^τ(e, v)] for groups of cells.

    groups: (num_cells,) group index of each cell (node or bin of its externality).
    """

    def __init__(self, type_distribution, functions, groups, num_groups):
        self.distribution = type_distribution
        self.functions = functions
        self.groups = np.asarray(groups)
        self.num_groups = num_groups
        self.maximum_valuation = float(type_distribution.maximum_valuation)
        self.atomic = type_distribution.is_atomic
        if self.atomic:
            self._build_atomic()
        else:
            self._build_piecewise_uniform()

    # Piecewise-uniform cells ----------------------------------------------------

    def _build_piecewise_uniform(self):
        d, f = self.distribution, self.functions
        breaks = f.breakpoints
        tolerance = 1e-12 * max(self.maximum_valuation, 1.0)
        lower_index = np.searchsorted(breaks, d.valuation_lower - tolerance, side="left")
        upper_index = np.searchsorted(breaks, d.valuation_upper - tolerance, side="left")
        density = d.probabilities / (d.valuation_upper - d.valuation_lower)
        num_intervals = f.num_intervals
        steps = np.zeros((2, self.num_groups, num_intervals + 1))
        for row, weight in enumerate((density, density * d.externalities)):
            np.add.at(steps[row], (self.groups, lower_index), weight)
            np.add.at(steps[row], (self.groups, upper_index), -weight)
        cumulative = np.cumsum(steps, axis=2)[:, :, :num_intervals]
        self.density, self.externality_density = cumulative[0], cumulative[1]
        contributions = (self.density * f.integral_a + self.externality_density * f.integral_x)
        tail = np.cumsum(contributions[:, ::-1], axis=1)[:, ::-1]
        self.tail = np.concatenate([tail, np.zeros((self.num_groups, 1))], axis=1)

    def _payoff_piecewise_uniform(self, group, thresholds):
        f = self.functions
        t = np.clip(thresholds, 0.0, self.maximum_valuation)
        interval = f.interval_of(t)
        partial_a, partial_x = f.partial_integrals(interval, t)
        payoff = (self.tail[group, interval + 1] + self.density[group, interval] * partial_a
                  + self.externality_density[group, interval] * partial_x)
        return np.where(thresholds >= self.maximum_valuation, 0.0, payoff)

    def _best_piecewise_uniform(self):
        """(best threshold, best payoff) of every group over [0, v̄] ∪ {v̄⁺}."""
        f = self.functions
        values = self.tail  # payoff at t = breakpoint i
        best_index = np.argmax(values, axis=1)
        best_t = f.breakpoints[best_index]
        best_value = values[np.arange(self.num_groups), best_index]

        ends = np.array([-1.0, 1.0])
        a_ends = chebyshev.chebval(ends, f.coefficients_a)  # (N, 2)
        x_ends = chebyshev.chebval(ends, f.coefficients_x)
        h_left = self.density * a_ends[:, 0] + self.externality_density * x_ends[:, 0]
        h_right = self.density * a_ends[:, 1] + self.externality_density * x_ends[:, 1]
        group, interval = np.nonzero((h_left < 0) & (h_right > 0))
        if len(group):
            coefficients = (self.density[group, interval] * f.coefficients_a[:, interval]
                            + self.externality_density[group, interval]
                            * f.coefficients_x[:, interval])
            low, high = -np.ones(len(group)), np.ones(len(group))
            for _ in range(60):
                middle = (low + high) / 2
                negative = chebyshev.chebval(middle, coefficients, tensor=False) < 0
                low, high = np.where(negative, middle, low), np.where(negative, high, middle)
            left, right = f.breakpoints[interval], f.breakpoints[interval + 1]
            roots = left + (right - left) * ((low + high) / 2 + 1) / 2
            root_values = self._payoff_piecewise_uniform(group, roots)
            order = np.lexsort((root_values, group))  # best root last within each group
            last = np.append(group[order][1:] != group[order][:-1], True)
            g, t, v = group[order][last], roots[order][last], root_values[order][last]
            better = v > best_value[g]
            best_t[g[better]], best_value[g[better]] = t[better], v[better]
        return best_t, best_value

    # Atomic cells -----------------------------------------------------------------

    def _build_atomic(self):
        d, f = self.distribution, self.functions
        allocation, loss = f.rival_welfare_loss(d.valuation_lower)
        marginal = (d.valuation_lower + d.externalities) * allocation - loss
        self.unique_valuations = np.unique(d.valuation_lower)
        rank = np.searchsorted(self.unique_valuations, d.valuation_lower)
        self.rank_span = len(self.unique_valuations) + 1
        key = self.groups * self.rank_span + rank
        order = np.argsort(key, kind="stable")
        self.sorted_key = key[order]
        weighted = (d.probabilities * marginal)[order]
        self.prefix = np.concatenate([[0.0], np.cumsum(weighted)])
        group_end = np.searchsorted(self.sorted_key, (np.arange(self.num_groups) + 1)
                                    * self.rank_span, side="left")
        self.group_end_prefix = self.prefix[group_end]
        self.atom_group, self.atom_valuation = self.groups, d.valuation_lower

    def _payoff_atomic(self, group, thresholds):
        rank = np.searchsorted(self.unique_valuations, np.asarray(thresholds), side="left")
        position = np.searchsorted(self.sorted_key, group * self.rank_span + rank, side="left")
        return self.group_end_prefix[group] - self.prefix[position]

    def _best_atomic(self):
        candidates = self._payoff_atomic(self.atom_group, self.atom_valuation)
        best_t = np.full(self.num_groups, np.inf)
        best_value = np.zeros(self.num_groups)  # exclusion
        zero = self._payoff_atomic(np.arange(self.num_groups), np.zeros(self.num_groups))
        better = zero > best_value
        best_t[better], best_value[better] = 0.0, zero[better]
        order = np.lexsort((candidates, self.atom_group))
        last = np.append(self.atom_group[order][1:] != self.atom_group[order][:-1], True)
        g, t, v = self.atom_group[order][last], self.atom_valuation[order][last], candidates[order][last]
        better = v > best_value[g]
        best_t[g[better]], best_value[g[better]] = t[better], v[better]
        return best_t, best_value

    # Interface ----------------------------------------------------------------------

    def payoff(self, group, thresholds):
        group = np.asarray(group)
        thresholds = np.asarray(thresholds, dtype=float)
        if self.atomic:
            return self._payoff_atomic(group, thresholds)
        return self._payoff_piecewise_uniform(group, thresholds)

    def best(self):
        """(best threshold, best payoff) of each group; v̄ and above mean exclusion."""
        return self._best_atomic() if self.atomic else self._best_piecewise_uniform()


# ─── Best response within a form ──────────────────────────────────────────────


def _group_of_cells(form, type_distribution):
    nodes, _, node_of_cell = externality_nodes(type_distribution)
    if form.kind == "bins":
        return np.searchsorted(form.locations, type_distribution.externalities, side="right"), form.size
    return node_of_cell, len(nodes)


def _knot_value_grid(type_distribution, form, size):
    probabilities = type_distribution.probabilities
    valuations = type_distribution.valuation_midpoints
    order = np.argsort(valuations)
    cumulative = np.cumsum(probabilities[order])
    levels = np.linspace(0, 1, size)[1:-1]
    quantiles = valuations[order][np.minimum(np.searchsorted(cumulative, levels), len(order) - 1)]
    maximum = form.maximum_valuation
    # Values outside [0, v̄] change the interpolated thresholds between knots.
    outside = maximum * np.array([-0.5, -0.25, -0.1, 1.25, 1.5, 2.0, 3.0, 4.0])
    return np.unique(np.concatenate([[0.0], quantiles, [maximum, form.exclusion_value],
                                     outside]))


def _knot_dynamic_program(payoffs, form, node_values, grids):
    """Exact maximization of Σ_nodes g over knot values in the given grids and the cutoff."""
    knots = form.locations
    num_knots = len(knots)
    segment = np.clip(np.searchsorted(knots, node_values, side="right") - 1, 0, num_knots - 2)
    weight = np.clip((knots[segment + 1] - node_values) / (knots[segment + 1] - knots[segment]),
                     0.0, 1.0)
    num_nodes = len(node_values)
    future = np.zeros(len(grids[-1]))
    pointers, cut_values, cut_choices = [None] * (num_knots - 1), [], []
    for s in range(num_knots - 2, -1, -1):
        members = np.flatnonzero(segment == s)
        left, right = grids[s][:, None, None], grids[s + 1][None, :, None]
        w = weight[members][None, None, :]
        thresholds = w * left + (1 - w) * right
        values = payoffs.payoff(np.broadcast_to(members, thresholds.shape), thresholds)
        suffix = np.concatenate([np.cumsum(values[:, :, ::-1], axis=2)[:, :, ::-1],
                                 np.zeros(values.shape[:2] + (1,))], axis=2)
        full = suffix[:, :, 0] + future[None, :]
        cut = suffix + future[None, :, None]
        best_cut = np.unravel_index(np.argmax(cut), cut.shape)
        cut_values.append(cut[best_cut])
        cut_choices.append((s, best_cut, members))
        pointers[s] = np.argmax(full, axis=1)
        future = full[np.arange(len(grids[s])), pointers[s]]
    best = int(np.argmax(cut_values))
    s, (a, b, m), members = cut_choices[best]
    values = np.empty(num_knots)
    values[: s + 1] = grids[s][a]
    values[s + 1] = grids[s + 1][b]
    index = b
    for k in range(s + 1, num_knots - 1):
        index = pointers[k][index]
        values[k + 1] = grids[k + 1][index]
    num_excluded = int(np.sum(segment < s)) + int(m)
    if num_excluded == 0:
        cutoff = form.no_cutoff_value
    elif num_excluded >= num_nodes:
        cutoff = form.externality_range[1] + 1.0
    else:
        cutoff = 0.5 * (node_values[num_excluded - 1] + node_values[num_excluded])
    return np.append(values, cutoff), float(cut_values[best])


def _knot_best_response(payoffs, form, type_distribution, current_parameters):
    node_values, _, _ = externality_nodes(type_distribution)
    base = _knot_value_grid(type_distribution, form, KNOT_GRID_SIZE)
    grids = [np.unique(np.append(base, value)) for value in current_parameters[:-1]]
    parameters, value = _knot_dynamic_program(payoffs, form, node_values, grids)
    for _ in range(KNOT_ZOOM_PASSES):
        zoomed = []
        for grid, chosen in zip(grids, parameters[:-1]):
            index = int(np.argmin(np.abs(grid - chosen)))
            low, high = grid[max(index - 1, 0)], grid[min(index + 1, len(grid) - 1)]
            zoomed.append(np.unique(np.append(np.linspace(low, high, KNOT_ZOOM_GRID_SIZE), chosen)))
        grids = zoomed
        parameters, value = _knot_dynamic_program(payoffs, form, node_values, grids)
    return parameters, value


def best_response(type_distribution, form, current_parameters, num_bidders, num_goods,
                  mechanism):
    """Best-response parameters of the form against the current τ, the current and best
    one-bidder payoffs Σ g, and the payoffs object (for diagnostics)."""
    threshold = form.threshold(current_parameters)
    model = build_mechanism_model(mechanism, type_distribution, threshold, num_bidders, num_goods)
    functions = valuation_functions(model, type_distribution)
    node_values, _, _ = externality_nodes(type_distribution)
    if form.kind == "knots":
        node_payoffs = ParticipationPayoffs(type_distribution, functions,
                                            externality_nodes(type_distribution)[2],
                                            len(node_values))
        current_value = float(np.sum(node_payoffs.payoff(
            np.arange(len(node_values)), form.finite_values(current_parameters, node_values))))
        parameters, best_value = _knot_best_response(node_payoffs, form, type_distribution,
                                                     current_parameters)
        return parameters, current_value, max(best_value, current_value), node_payoffs
    groups, num_groups = _group_of_cells(form, type_distribution)
    payoffs = ParticipationPayoffs(type_distribution, functions, groups, num_groups)
    current = np.asarray(current_parameters, dtype=float)
    current_payoff = payoffs.payoff(np.arange(num_groups), current)
    best_t, best_value = payoffs.best()
    scale = max(np.max(np.abs(best_value)), 1e-300)
    keep = best_value <= current_payoff + 1e-13 * scale
    parameters = np.where(best_t >= form.maximum_valuation, form.exclusion_value, best_t)
    parameters = np.where(keep, current, parameters)
    best_value = np.where(keep, current_payoff, best_value)
    return parameters, float(current_payoff.sum()), float(best_value.sum()), payoffs


# ─── Ascent ───────────────────────────────────────────────────────────────────


def _start_thresholds(starts, type_distribution, form, num_bidders, num_goods, mechanism,
                      short_genetic_generations, seed, deadline=None):
    maximum = type_distribution.maximum_valuation
    thresholds = []
    for start in starts:
        if callable(start):
            thresholds.append((getattr(start, "start_name", "warm start"), start))
        elif start == "zero":
            thresholds.append(("zero", lambda e: np.zeros(np.shape(e))))
        elif start in ("zero_welfare", "no_competition"):  # τ⁰; old name accepted
            thresholds.append(("zero_welfare", lambda e: np.where(
                -np.asarray(e) > maximum, np.inf, np.maximum(-np.asarray(e), 0.0))))
        elif start == "short_genetic":
            genetic_form = form if form.kind != "nodes" else build_threshold_form(
                "knots6", type_distribution)
            search = run_genetic_search(type_distribution, None, num_bidders, num_goods,
                                        objective_mechanism=mechanism,
                                        threshold_form=genetic_form,
                                        num_generations=short_genetic_generations,
                                        sol_per_pop=30, num_parents_mating=10, seed=seed,
                                        local_refinement="none", verbose=False,
                                        deadline=deadline)
            thresholds.append(("short_genetic", genetic_form.threshold(search.coefficients)))
        else:
            raise ValueError(f"unknown start {start!r}")
    return thresholds


def _past(deadline):
    return deadline is not None and time.time() >= deadline


def first_order_measures(type_distribution, form, parameters, num_bidders, functions, welfare,
                         scale=None):
    """(stationarity, max_marginal) of τ = form.threshold(parameters); see the module
    docstring. functions: valuation_functions of the objective model at this τ.
    Stationarity is None for atomic distributions (W is a step function of τ)."""
    nodes, masses, node_of_cell = externality_nodes(type_distribution)
    maximum = float(type_distribution.maximum_valuation)
    values = form.finite_values(parameters, nodes)
    clipped = np.clip(values, 0.0, maximum)
    allocation, loss = functions.rival_welfare_loss(clipped)
    marginal = (clipped + nodes) * allocation - loss
    # Violation of the first-order conditions: |M| at interior thresholds; at τ = 0,
    # M(e, 0) < 0 (raising τ helps); at exclusion, M(e, v̄) > 0 (admitting types helps).
    violation = np.where(values <= 0, np.maximum(-marginal, 0.0),
                         np.where(values >= maximum, np.maximum(marginal, 0.0), np.abs(marginal)))
    violation = violation[(masses > 0) & np.isfinite(violation)]
    max_marginal = float(np.max(violation) / maximum) if len(violation) else 0.0
    if type_distribution.is_atomic:
        return None, max_marginal
    payoffs = ParticipationPayoffs(type_distribution, functions, node_of_cell, len(nodes))
    gradient = _parameter_gradient(type_distribution, form, parameters, num_bidders, functions,
                                   payoffs, bounds=True)
    scale = scale if scale else max(abs(welfare), 1e-300)
    return float(maximum * np.sum(np.abs(gradient)) / scale), max_marginal


def ascend(type_distribution, form, parameters, num_bidders, num_goods, mechanism,
           max_iterations=100, relative_tolerance=1e-10, verbose=False, deadline=None,
           scale=None, first_iteration=0):
    """Best-response ascent from one start; returns (parameters, welfare, history,
    stop_reason). History entries: iteration, welfare, step, gap, stationarity,
    max_marginal (measures at that entry's τ), parameters, time."""
    parameters = np.asarray(parameters, dtype=float)

    def welfare_of(candidate):
        return expected_welfare(type_distribution, form.threshold(candidate), num_bidders,
                                num_goods, mechanism)

    welfare = welfare_of(parameters)
    history = [{"iteration": first_iteration, "welfare": welfare, "step": None, "gap": None,
                "stationarity": None, "max_marginal": None,
                "parameters": parameters.tolist(), "time": time.time()}]
    unit = scale if scale else max(abs(welfare), 1e-300)
    small_gains, stop_reason = 0, "max_iterations"
    for iteration in range(first_iteration + 1, first_iteration + max_iterations + 1):
        if _past(deadline):
            stop_reason = "time_limit"
            break
        target, current_payoff, best_payoff, payoffs = best_response(
            type_distribution, form, parameters, num_bidders, num_goods, mechanism)
        gap = best_payoff - current_payoff
        stationarity, max_marginal = first_order_measures(
            type_distribution, form, parameters, num_bidders, payoffs.functions, welfare, scale)
        history[-1].update(gap=gap / unit, stationarity=stationarity, max_marginal=max_marginal)
        accepted = None
        for step in STEP_SIZES:
            candidate = parameters + step * (target - parameters)
            candidate_welfare = welfare_of(candidate)
            if candidate_welfare > welfare + 1e-14 * max(abs(welfare), 1.0):
                accepted = (step, candidate, candidate_welfare)
                break
        if accepted is None:
            stop_reason = "no_improving_step"
            break
        step, parameters, new_welfare = accepted
        gain = (new_welfare - welfare) / max(abs(welfare), 1e-300)
        welfare = new_welfare
        history.append({"iteration": iteration, "welfare": welfare, "step": step, "gap": None,
                        "stationarity": None, "max_marginal": None,
                        "parameters": parameters.tolist(), "time": time.time()})
        if verbose:
            print(f"    iteration {iteration:3d}  welfare={welfare:.10g}  step={step:g}  "
                  f"best-response gap={gap:.3g}")
        small_gains = small_gains + 1 if gain < relative_tolerance else 0
        if small_gains >= 2:
            stop_reason = "small_gain"
            break
    return parameters, welfare, history, stop_reason


def welfare_and_gradient(type_distribution, form, parameters, num_bidders, num_goods,
                         mechanism):
    """Exact W(τ) and its gradient in the form's parameters (piecewise-uniform cells).

    Node gradient −n f_j M^τ(e_j, τ_j) (Prop. 7.9; one-sided at cell edges: the
    density of the cell above τ_j); knots: chained through the interpolation
    weights of included nodes, zero for the cutoff; bins: summed over the bin.
    """
    parameters = np.asarray(parameters, dtype=float)
    threshold = form.threshold(parameters)
    try:
        model = build_mechanism_model(mechanism, type_distribution, threshold, num_bidders,
                                      num_goods)
    except ValueError:  # nobody participates
        return 0.0, np.zeros_like(parameters)
    welfare = model.expected_welfare()
    functions = valuation_functions(model, type_distribution)
    nodes, _, node_of_cell = externality_nodes(type_distribution)
    payoffs = ParticipationPayoffs(type_distribution, functions, node_of_cell, len(nodes))
    return welfare, _parameter_gradient(type_distribution, form, parameters, num_bidders,
                                        functions, payoffs)


def _parameter_gradient(type_distribution, form, parameters, num_bidders, functions, payoffs,
                        bounds=False):
    """∂W/∂θ for the form's parameters θ; payoffs: ParticipationPayoffs grouped by node.

    bounds=False: zero where τ(e) is clipped to 0 or excluded (W is flat in the
    parameter there). bounds=True: at τ(e) = 0 and at exclusion, the one-sided
    derivative into [0, v̄] if it raises W, else zero (the projected gradient of the
    first-order (KKT) conditions with bounds 0 and v̄)."""
    parameters = np.asarray(parameters, dtype=float)
    nodes, _, _ = externality_nodes(type_distribution)
    values = form.finite_values(parameters, nodes)
    maximum = form.maximum_valuation
    at_zero = np.abs(values) <= 1e-12 * maximum
    excluded = values >= maximum
    active = ((values > 0) & (values < maximum)) | (bounds & (at_zero | excluded))
    clipped = np.clip(values, 0.0, form.maximum_valuation)
    interval = functions.interval_of(clipped)
    x = functions.local_coordinate(interval, clipped)
    allocation = chebyshev.chebval(x, functions.coefficients_x[:, interval], tensor=False)
    base = chebyshev.chebval(x, functions.coefficients_a[:, interval], tensor=False)
    density = payoffs.density[np.arange(len(nodes)), interval]
    node_gradient = np.where(active, -num_bidders * density * (base + nodes * allocation), 0.0)
    if bounds:  # only feasible, improving directions: raise τ from 0, lower τ from v̄
        node_gradient = np.where(at_zero, np.maximum(node_gradient, 0.0),
                                 np.where(excluded, np.minimum(node_gradient, 0.0), node_gradient))
    if form.kind == "nodes":
        return np.interp(form.locations, nodes, node_gradient) if len(
            form.locations) != len(nodes) else node_gradient
    if form.kind == "bins":
        bin_of_node = np.searchsorted(form.locations, nodes, side="right")
        return np.bincount(bin_of_node, node_gradient, form.size)
    knots = form.locations
    segment = np.clip(np.searchsorted(knots, nodes, side="right") - 1, 0, len(knots) - 2)
    weight = np.clip((knots[segment + 1] - nodes) / (knots[segment + 1] - knots[segment]),
                     0.0, 1.0)
    included = nodes >= parameters[-1]
    gradient = np.zeros_like(parameters)
    np.add.at(gradient, segment[included], (node_gradient * weight)[included])
    np.add.at(gradient, segment[included] + 1, (node_gradient * (1 - weight))[included])
    return gradient


def polish(type_distribution, form, parameters, num_bidders, num_goods, mechanism,
           max_iterations=200, deadline=None):
    """L-BFGS-B on W with the exact gradient (knots: cutoff fixed); returns (parameters, W)."""
    from scipy.optimize import minimize

    parameters = np.asarray(parameters, dtype=float)
    start_welfare = expected_welfare(type_distribution, form.threshold(parameters), num_bidders,
                                     num_goods, mechanism)
    scale = max(abs(start_welfare), 1e-12)

    def objective(candidate):
        welfare, gradient = welfare_and_gradient(type_distribution, form, candidate,
                                                 num_bidders, num_goods, mechanism)
        return -welfare / scale, -gradient / scale

    bounds = [(None, None)] * len(parameters)
    if form.kind == "knots":
        bounds[-1] = (parameters[-1], parameters[-1])
    if form.kind in ("nodes", "bins"):
        bounds = [(0.0, form.exclusion_value)] * len(parameters)
    def callback(intermediate_result):  # noqa: ARG001
        if _past(deadline):
            raise StopIteration

    outcome = minimize(objective, parameters, jac=True, method="L-BFGS-B", bounds=bounds,
                       callback=callback,
                       options={"maxiter": max_iterations, "ftol": 1e-15, "gtol": 1e-12})
    welfare = expected_welfare(type_distribution, form.threshold(outcome.x), num_bidders,
                               num_goods, mechanism)
    if welfare > start_welfare:
        return np.asarray(outcome.x, dtype=float), welfare
    return parameters, start_welfare


def _complete_measures(type_distribution, form, entry, num_bidders, num_goods, mechanism,
                       scale):
    """Fill the gap and first-order measures of a history entry that has none."""
    if entry["max_marginal"] is not None:
        return
    target, current_payoff, best_payoff, payoffs = best_response(
        type_distribution, form, entry["parameters"], num_bidders, num_goods, mechanism)
    stationarity, max_marginal = first_order_measures(
        type_distribution, form, entry["parameters"], num_bidders, payoffs.functions,
        entry["welfare"], scale)
    unit = scale if scale else max(abs(entry["welfare"]), 1e-300)
    entry.update(gap=(best_payoff - current_payoff) / unit, stationarity=stationarity,
                 max_marginal=max_marginal)


def run_best_response_ascent(type_distribution, threshold_form, num_bidders, num_goods,
                             objective_mechanism="vcg_pa", starts=DEFAULT_STARTS,
                             warm_starts=(), max_iterations=100, relative_tolerance=1e-10,
                             short_genetic_generations=40, polish_rounds=DEFAULT_POLISH_ROUNDS,
                             seed=None, verbose=True, deadline=None, scale=None):
    """Maximize the exact welfare of objective_mechanism over τ in threshold_form.

    threshold_form : a ThresholdForm or a label ("knots6", "bins10", "nodes")
    starts         : names among "zero", "zero_welfare", "short_genetic"
    warm_starts    : threshold functions (callables) used as additional starts;
                     an attribute start_name labels them in the summary
    polish_rounds  : gradient polishes (each followed by more ascent); 0 = none;
                     skipped for atomic distributions
    deadline       : time.time() value after which the search stops gracefully
                     (starts not begun are listed with stop_reason "skipped (time limit)")
    scale          : welfare unit of the diagnostics (the first-best gain FB − W_VCG)
    Returns a genetic_search.SearchResult (coefficients = parameters of the form).
    """
    form = (threshold_form if not isinstance(threshold_form, str)
            else build_threshold_form(threshold_form, type_distribution))
    if form.kind not in BEST_RESPONSE_FORMS:
        raise ValueError(f"best-response ascent needs a form in {BEST_RESPONSE_FORMS}")
    start_list = _start_thresholds(list(starts) + list(warm_starts), type_distribution, form,
                                   num_bidders, num_goods, objective_mechanism,
                                   short_genetic_generations, seed, deadline)
    if verbose:
        print(f"Best-response ascent ({objective_mechanism}, {form.label}, "
              f"{form.num_parameters} parameters, {len(start_list)} starts)")
    best, summaries, trajectory, trace = None, [], [], []

    def record_trace(name, entries):
        trace.extend({"stage": name, "iteration": entry["iteration"], "time": entry["time"],
                      "welfare": float(entry["welfare"])} for entry in entries)

    def diagnostics(entries, stage):
        return [{"iteration": e["iteration"], "stage": stage, "time": e["time"],
                 "welfare": float(e["welfare"]), "gap": e["gap"],
                 "stationarity": e["stationarity"], "max_marginal": e["max_marginal"]}
                for e in entries]

    for name, threshold in start_list:
        if _past(deadline) and best is not None:
            summaries.append({"start": name, "start_welfare": None, "welfare": None,
                              "iterations": 0, "polish_gains": [], "final_gap": None,
                              "seconds": 0.0, "parameters": None,
                              "stop_reason": "skipped (time limit)", "history": []})
            continue
        begin = time.time()
        initial = form.project(threshold, type_distribution)
        parameters, welfare, history, stop_reason = ascend(
            type_distribution, form, initial, num_bidders, num_goods, objective_mechanism,
            max_iterations, relative_tolerance, verbose=False, deadline=deadline, scale=scale)
        start_welfare, iterations = history[0]["welfare"], len(history) - 1
        trajectory.extend(entry["parameters"] for entry in history)
        record_trace(name, history)
        records = diagnostics(history, "ascent")
        polish_gains = []
        rounds = 0 if type_distribution.is_atomic else polish_rounds
        for _ in range(rounds):
            if _past(deadline):
                stop_reason = "time_limit"
                break
            polished, polished_welfare = polish(type_distribution, form, parameters,
                                                num_bidders, num_goods, objective_mechanism,
                                                deadline=deadline)
            gain = polished_welfare - welfare
            if gain <= relative_tolerance * max(abs(welfare), 1e-300):
                break
            polish_gains.append(gain)
            parameters, welfare, history, stop_reason = ascend(
                type_distribution, form, polished, num_bidders, num_goods, objective_mechanism,
                max_iterations, relative_tolerance, deadline=deadline, scale=scale,
                first_iteration=records[-1]["iteration"] + 1)
            iterations += len(history) - 1
            trajectory.extend(entry["parameters"] for entry in history)
            record_trace(name, history)
            records += diagnostics(history[:1], "polish") + diagnostics(history[1:], "ascent")
        _complete_measures(type_distribution, form, history[-1], num_bidders, num_goods,
                           objective_mechanism, scale)
        records[-1].update({key: history[-1][key]
                            for key in ("gap", "stationarity", "max_marginal")})
        summaries.append({"start": name, "start_welfare": start_welfare,
                          "welfare": welfare, "iterations": iterations,
                          "polish_gains": polish_gains,
                          "final_gap": history[-1]["gap"], "seconds": time.time() - begin,
                          "parameters": parameters.tolist(), "stop_reason": stop_reason,
                          "history": records})
        if verbose:
            print(f"  start {name:15s}: welfare {start_welfare:.10g} -> {welfare:.10g} "
                  f"in {iterations} iterations, {len(polish_gains)} polishes "
                  f"({time.time() - begin:.1f}s, {stop_reason})")
        if best is None or welfare > best[1]:
            best = (parameters, welfare)
    summary = {"optimizer": "best_response", "form": form.spec(best[0]), "starts": summaries,
               "trace": trace}
    return SearchResult([float(p) for p in best[0]], float(best[1]), trajectory, summary)


# ─── First-order check ──────────────────────────────────────────────────────────


def first_order_check(type_distribution, threshold_function, num_bidders, num_goods,
                      mechanism="vcg_pa"):
    """Per externality node: the threshold type's marginal contribution and penalty terms.

    Columns: externality, mass, threshold (clipped to [0, v̄]), status (interior:
    0 < τ < v̄; zero; excluded), interim allocation X^τ(τ), penalty r(e) =
    R(τ(e)), own-allocation charge −e X^τ(τ), displaced term L^τ(τ) − p(τ)
    (rival welfare loss minus the critical-value payment; the displaced
    externality Δ^τ in VCG-PA), marginal contribution M^τ(e, τ(e)) = r − (−e X^τ +
    displaced), and the best-response gap max_t g_e(t) − g_e(τ(e)).

    At an optimal common τ, interior thresholds where the density of v given e
    is positive have zero marginal contribution (paper Prop. 7.9 [prop:zero-marginal-equal];
    Thm 7.2 [thm:zero-marginal] for optimal profiles): the penalty equals the Pigouvian
    charge −e X^τ + L^τ − P^τ (Rem. 7.3 [rem:pigouvian], Cor. E.2 [cor:pigouvian]), which is
    −e X^τ + Δ^τ in VCG-PA.
    """
    model = build_mechanism_model(mechanism, type_distribution, threshold_function, num_bidders,
                                  num_goods)
    functions = valuation_functions(model, type_distribution)
    nodes, masses, node_of_cell = externality_nodes(type_distribution)
    maximum = float(type_distribution.maximum_valuation)
    raw = np.asarray(threshold_function(nodes), dtype=float)
    clipped = np.clip(raw, 0.0, maximum)
    allocation, loss = functions.rival_welfare_loss(clipped)
    penalty = model.allocation_integral(clipped)
    payment = clipped * allocation - penalty
    marginal = (clipped + nodes) * allocation - loss
    payoffs = ParticipationPayoffs(type_distribution, functions, node_of_cell, len(nodes))
    best_t, best_value = payoffs.best()
    current = payoffs.payoff(np.arange(len(nodes)), np.where(raw >= maximum, maximum, clipped))
    status = np.where(raw >= maximum, "excluded", np.where(raw <= 0, "zero", "interior"))
    return pd.DataFrame({
        "externality": nodes, "mass": masses, "threshold": clipped, "status": status,
        "allocation": allocation, "penalty": penalty, "own_allocation_charge": -nodes * allocation,
        "displaced_term": loss - payment,
        "marginal_contribution": np.where(raw >= maximum, np.nan, marginal),
        "best_response_gap": np.maximum(best_value - current, 0.0),
    })
