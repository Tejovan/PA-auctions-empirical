"""
grid_search.py

Exhaustive grid search, then Nelder–Mead polish, for the two small threshold forms
(threshold_forms.py), with the exact objective (threshold_evaluation.expected_welfare):

  cutoff  τ(e) = max(0, w − e), excluded where w − e > v̄: a grid of grid_points
          welfare cutoffs w ∈ [0, w_max] (the genetic search range)
  poly1   τ(e) = β_0 + β_1 e, clipped to [0, v̄] (excluded above v̄): a grid of
          grid_points × grid_points lines

Lines. With x = (e − e_min)/(e_max − e_min) and y = v/v̄, the box of the type plane
[e_min, e_max] × [0, v̄] (e_min, e_max the extreme externality nodes) is [0, 1]². A line
is {(x, y): x cos θ + y sin θ = ρ} with angle θ ∈ (0, π), and participation is
x cos θ + y sin θ ≥ ρ (the types above the line): θ → 0 is a vertical boundary with
participation to its right (τ decreasing, steep), θ = π/2 a constant τ, θ → π a
vertical boundary with participation to its left (τ increasing). For each θ, the
offset ρ = lo(θ) + u (hi(θ) − lo(θ)), u ∈ [0, 1], runs between the lowest and highest
corner of the box, lo = min(0, cos θ), hi = max(0, cos θ) + sin θ: u = 0 is τ ≤ 0 on the
box (all participate), u = 1 is exclusion of every externality. The grid is uniform in
(θ, u), θ at the midpoints of grid_points equal subintervals of (0, π), u at grid_points
points of [0, 1] with both ends. So every line through the box is near a grid point at
the same relative spacing; a grid in (β_0, β_1) would space steep lines (β_1 → −∞, the
optimum when v and e are independent) poorly. In the form's parameters,

    β_1 = −v̄ cos θ / (sin θ (e_max − e_min)),   β_0 = v̄ (ρ + cos θ e_min/(e_max − e_min)) / sin θ.

Polish. Nelder–Mead (genetic_search.refine_with_nelder_mead; coordinates scaled to the
grid ranges, initial simplex two grid spacings) from the best grid points that are
strict local maxima of the grid (better than their 3^d − 1 neighbors; plus the best grid
point) and more than three grid spacings from a better start, up to num_local_starts of
them. The result is the best of the grid and the polished points.

Evidence of the global optimum: the summary records the best grid welfare, the number
of grid local maxima, the welfare of each polished start and its gain over its grid
point. A grid is no proof (the objective has kinks and no known Lipschitz bound), but
with a fine grid every basin of attraction wider than a few grid spacings contains a
start.
"""

import time

import numpy as np

from .genetic_search import SearchResult, refine_with_nelder_mead
from .parallel import parallel_map
from .threshold_evaluation import expected_welfare

GRID_FORMS = ("poly1", "cutoff")
DEFAULT_GRID_POINTS = 121
DEFAULT_NUM_LOCAL_STARTS = 5
DEFAULT_LOCAL_MAX_EVALUATIONS = 500
START_SEPARATION = 3   # grid spacings between polish starts (Chebyshev distance in indices)
ANGLE_MARGIN = 1e-6    # θ is kept in [margin, π − margin] (finite slope)


class GridObjective:
    """Exact welfare at a grid coordinate: (θ, u) for poly1, (w,) for cutoff (picklable)."""

    def __init__(self, type_distribution, form, num_bidders, num_goods, mechanism):
        self.arguments = (type_distribution, form, num_bidders, num_goods, mechanism)
        low, high = form.externality_range
        self.e_min, self.width = float(low), max(float(high - low), 1e-12)
        self.maximum_valuation = float(form.maximum_valuation)

    def parameters(self, point):
        """Parameters of the form at a grid coordinate."""
        form = self.arguments[1]
        if form.kind == "cutoff":
            return np.array([max(float(point[0]), 0.0)])
        angle = float(np.clip(point[0], ANGLE_MARGIN, np.pi - ANGLE_MARGIN))
        cosine, sine = np.cos(angle), np.sin(angle)
        low, high = min(0.0, cosine), max(0.0, cosine) + sine
        offset = low + float(point[1]) * (high - low)
        slope = -self.maximum_valuation * cosine / (sine * self.width)
        intercept = self.maximum_valuation * (offset + cosine * self.e_min / self.width) / sine
        return np.array([intercept, slope])

    def __call__(self, point):
        d, form, n, k, mechanism = self.arguments
        return expected_welfare(d, form.threshold(self.parameters(point)), n, k, mechanism)


def _evaluate_points(job):
    objective = job["objective"]
    return [objective(point) for point in job["points"]]


def _polish_job(job):
    results = refine_with_nelder_mead(job["objective"], [job["start"]], job["lows"],
                                      job["highs"], job["max_evaluations"], job["deadline"],
                                      simplex_step=job["simplex_step"])
    return results[0] if results else None


def grid_axes(form, type_distribution, grid_points):
    """Grid coordinates along each axis and their ranges (lows, highs)."""
    if form.kind == "cutoff":
        lows, highs = form.parameter_bounds(type_distribution)
        return [np.linspace(lows[0], highs[0], grid_points)], lows, highs
    angles = np.pi * (np.arange(grid_points) + 0.5) / grid_points
    offsets = np.linspace(0.0, 1.0, grid_points)
    return [angles, offsets], np.array([0.0, 0.0]), np.array([np.pi, 1.0])


def local_maxima(values):
    """Boolean array: strict local maxima of the grid, better than all their neighbors (no
    wrap-around), and the best grid point (which may tie with a neighbor, e.g. on a plateau)."""
    padded = np.pad(values, 1, constant_values=-np.inf)
    keep = np.ones(values.shape, dtype=bool)
    for shift in np.ndindex(*(3,) * values.ndim):
        if all(s == 1 for s in shift):
            continue
        window = tuple(slice(s, s + n) for s, n in zip(shift, values.shape))
        keep &= values > padded[window]
    keep.flat[int(np.argmax(values))] = True
    return keep


def polish_starts(values, count, separation=START_SEPARATION):
    """Indices of up to `count` grid local maxima, best first, pairwise farther apart than
    `separation` grid spacings."""
    candidates = np.argwhere(local_maxima(values))
    order = np.argsort(-values[tuple(candidates.T)], kind="stable")
    chosen = []
    for index in candidates[order]:
        if all(np.max(np.abs(index - other)) > separation for other in chosen):
            chosen.append(index)
        if len(chosen) == count:
            break
    return chosen, len(candidates)


def run_grid_search(type_distribution, threshold_form, num_bidders, num_goods,
                    objective_mechanism="vcg_pa", grid_points=DEFAULT_GRID_POINTS,
                    num_local_starts=DEFAULT_NUM_LOCAL_STARTS,
                    local_max_evaluations=DEFAULT_LOCAL_MAX_EVALUATIONS, welfare_scale=None,
                    deadline=None, workers=1, verbose=False):
    """Maximize the exact welfare of objective_mechanism over τ in the cutoff or poly1 form:
    every grid point, then Nelder–Mead from the best separated grid local maxima.

    welfare_scale : unit of the reported gains (pipeline: the first-best gain FB − W_VCG)
    Returns a SearchResult (coefficients in the form's parameters).
    """
    form = threshold_form
    if form.label not in GRID_FORMS:
        raise ValueError(f"the grid search needs a form in {GRID_FORMS}, got {form.label}")
    objective = GridObjective(type_distribution, form, num_bidders, num_goods,
                              objective_mechanism)
    axes, lows, highs = grid_axes(form, type_distribution, grid_points)
    shape = tuple(len(axis) for axis in axes)
    points = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    begin = time.time()
    chunks = np.array_split(points, max(1, min(len(points), 8 * max(workers, 1))))
    values, stop_reason = [], "grid complete"
    if workers > 1:
        for result in parallel_map(_evaluate_points, [{"objective": objective, "points": c}
                                                      for c in chunks], workers):
            values.extend(result)
    else:
        for chunk in chunks:
            if deadline is not None and time.time() >= deadline:
                stop_reason = "time_limit"
                break
            values.extend(_evaluate_points({"objective": objective, "points": chunk}))
    values = np.asarray(values + [-np.inf] * (len(points) - len(values))).reshape(shape)
    grid_seconds = time.time() - begin
    best_index = np.unravel_index(int(np.argmax(values)), shape)
    best_point = points[np.ravel_multi_index(best_index, shape)]
    best = {"point": best_point, "welfare": float(values[best_index])}
    grid_best = dict(best)
    trace = [{"stage": "grid", "evaluations": len(points), "time": time.time(),
              "welfare": best["welfare"]}]

    starts, num_maxima = polish_starts(values, num_local_starts)
    spacing = 2.0 / max(grid_points - 1, 1)
    local_results = []
    if stop_reason != "time_limit" and starts:
        jobs = [{"objective": objective, "start": points[np.ravel_multi_index(tuple(i), shape)],
                 "lows": lows, "highs": highs, "max_evaluations": local_max_evaluations,
                 "deadline": deadline, "simplex_step": spacing} for i in starts]
        local_results = [r for r in parallel_map(_polish_job, jobs, workers) if r]
        for result in local_results:
            result["point"], result["start_point"] = result["coefficients"], result["start"]
            result["coefficients"] = objective.parameters(result["point"]).tolist()
            result["start"] = objective.parameters(result["start_point"]).tolist()
            if result["welfare"] > best["welfare"]:
                best = {"point": np.asarray(result["point"]), "welfare": result["welfare"]}
            trace.append({"stage": "nelder_mead", "evaluations": None,
                          "time": result["history"][-1]["time"] if result["history"]
                          else time.time(), "welfare": max(best["welfare"], result["welfare"])})
    scale = welfare_scale or max(abs(best["welfare"]), 1e-300)
    coefficients = objective.parameters(best["point"])
    second = sorted((r["welfare"] for r in local_results), reverse=True)
    summary = {
        "optimizer": "grid", "grid_shape": list(shape), "grid_axes": "w" if form.kind == "cutoff"
        else "angle theta in (0, pi), offset u in [0, 1]", "grid_seconds": grid_seconds,
        "grid_best_point": np.asarray(grid_best["point"]).tolist(),
        "grid_best_parameters": objective.parameters(grid_best["point"]).tolist(),
        "grid_welfare": grid_best["welfare"], "num_grid_local_maxima": int(num_maxima),
        "polish_gain": (best["welfare"] - grid_best["welfare"]) / scale,
        "second_local_optimum_gap": ((second[0] - second[1]) / scale if len(second) > 1
                                     else None),
        "best_point": np.asarray(best["point"]).tolist(), "stop_reason": stop_reason,
        "local_starts": local_results, "num_objective_evaluations": int(len(points) + sum(
            r["evaluations"] for r in local_results)), "trace": trace,
        "form": form.spec(coefficients)}
    if verbose:
        print(f"Grid search ({objective_mechanism}, {form.label}): {len(points)} points in "
              f"{grid_seconds:.1f}s, best {grid_best['welfare']:.8g}; {num_maxima} local maxima; "
              f"polished {best['welfare']:.8g}")
    trajectory = [objective.parameters(point).tolist() for point in points]
    return SearchResult([float(c) for c in coefficients], float(best["welfare"]), trajectory,
                        summary)
