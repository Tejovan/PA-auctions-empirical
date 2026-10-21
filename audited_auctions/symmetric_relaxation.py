"""
symmetric_relaxation.py

Best symmetric participation rule for VCG-PA with k identical goods (paper
arxiv_WEB v3, §9.2 [sec:vcg-thresholds], Thm [thm:vcg-sufficient] and its proof
in App. [app:vcg-thresholds]).

A symmetric participation rule lets each type (e, v) participate with a
probability α(e, v), independently across bidders, with truthful bids. Every
common threshold function τ is such a rule, α = 1{v >= τ(e)}. Expected VCG-PA
welfare over all symmetric rules is a concave problem after the change of
variables y(v) = Φ(Γ(v)), with Γ(v) = Pr[participate, valuation > v],
φ(γ) = Pr[Bin(n − 1, γ) <= k − 1] and Φ(γ) = ∫_γ^1 φ = E[(k − Bin(n, γ))^+]/n:

    W = n ∫ v dy + n ∫ t(y) B_v(y' / t(y)) dv,     t(y) = φ(Φ^{-1}(y)),

where B_v(a) is the largest externality mass of participants with valuation v
and participation density a (the largest externalities participate first).
t is concave because φ is log-concave, and s B(a / s) is concave and
nondecreasing in s. So the best symmetric rule solves a convex program, and its
value bounds the VCG-PA welfare of every common τ from above.

Discretization. On valuation intervals I_t (cell edges of the distribution,
each split into equal parts), the program becomes a linear program: y at the
interval ends, s_t <= t(ȳ_t) at the midpoint ȳ_t (tangent cuts of the concave
t), the perspective of the piecewise-linear B_t as the minimum of its lines,
and n v_mid Δy_t for the valuation part. This is a midpoint rule: the value is
an estimate of the best symmetric welfare with error of order (interval
width)², not a certified bound. Piecewise-uniform distributions only.

Used by the "relaxation" start of best-response ascent, the "relaxation" optimizer
method (pipeline.optimize_threshold) and the symmetric_relaxation check.

Result: the value, the participating fraction of each externality node in each
interval, and the threshold function read from it (the first valuation at which
a node participates with fraction at least 1/2). If every fraction is
nondecreasing in v, the best symmetric rule is a threshold rule, and by the
theorem the threshold function read from it is a best common τ for VCG-PA.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.sparse import coo_matrix
from scipy.stats import binom

from .threshold_forms import externality_nodes
from .threshold_functions import tabulated_threshold


def allocation_probability(gamma, num_bidders, num_goods):
    """φ(γ) = Pr[Bin(n − 1, γ) <= k − 1]."""
    return binom.cdf(num_goods - 1, num_bidders - 1, gamma)


def allocation_slope(gamma, num_bidders, num_goods):
    """|φ'(γ)| = (n − 1) Pr[Bin(n − 2, γ) = k − 1]."""
    return (num_bidders - 1) * binom.pmf(num_goods - 1, num_bidders - 2, gamma)


def goods_left(gamma, num_bidders, num_goods):
    """Φ(γ) = E[(k − Bin(n, γ))^+] / n."""
    counts = np.arange(num_goods)
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    pmf = binom.pmf(counts[None, :], num_bidders, gamma[:, None])
    return (pmf * (num_goods - counts)).sum(axis=1) / num_bidders


@dataclass
class RelaxationResult:
    value: float                   # estimate of the best symmetric VCG-PA welfare
    nodes: np.ndarray              # externality nodes
    valuation_edges: np.ndarray    # (T + 1,) interval ends
    participation: np.ndarray      # (T, num_nodes) participating fraction
    thresholds: np.ndarray         # per node; +inf = excluded
    largest_drop: float            # largest decrease in v of a participating fraction
    status: str = ""
    threshold_function: object = field(default=None, repr=False)

    @property
    def is_threshold_rule(self):
        return self.largest_drop <= 1e-6


def _interval_masses(distribution, nodes_of_cell, num_nodes, target_intervals):
    edges = np.unique(np.concatenate([distribution.valuation_lower,
                                      distribution.valuation_upper]))
    parts = max(1, int(np.ceil(target_intervals / (len(edges) - 1))))
    fine = np.concatenate([np.linspace(edges[i], edges[i + 1], parts + 1)[:-1]
                           for i in range(len(edges) - 1)] + [[edges[-1]]])
    tolerance = 1e-12 * max(edges[-1], 1.0)
    lower = np.searchsorted(fine, distribution.valuation_lower - tolerance)
    upper = np.searchsorted(fine, distribution.valuation_upper - tolerance)
    density = distribution.probabilities / (distribution.valuation_upper
                                            - distribution.valuation_lower)
    steps = np.zeros((num_nodes, len(fine)))
    np.add.at(steps, (nodes_of_cell, lower), density)
    np.add.at(steps, (nodes_of_cell, upper), -density)
    node_density = np.cumsum(steps, axis=1)[:, :-1]           # (nodes, T)
    return fine, (node_density * np.diff(fine)[None, :]).T    # (T, nodes)


def _tangent_cuts(num_bidders, num_goods, num_cuts):
    """Points y0 and tangents (t(y0), t'(y0)) of the concave t(y) = φ(Φ^{-1}(y))."""
    total = num_goods / num_bidders
    y0 = total * np.linspace(1e-3, 1.0, num_cuts) ** 2
    gamma = np.array([0.0 if y >= total else
                      brentq(lambda g: goods_left(g, num_bidders, num_goods)[0] - y, 0.0, 1.0)
                      for y in y0])
    value = allocation_probability(gamma, num_bidders, num_goods)
    slope = allocation_slope(gamma, num_bidders, num_goods) / np.maximum(value, 1e-300)
    return y0, value, slope


def symmetric_relaxation(distribution, num_bidders, num_goods, num_intervals=600,
                         num_cuts=80):
    """Best symmetric participation rule for VCG-PA (see the module docstring)."""
    if distribution.is_atomic:
        raise ValueError("symmetric_relaxation needs a piecewise-uniform distribution")
    n, k = num_bidders, num_goods
    nodes, _, node_of_cell = externality_nodes(distribution)
    edges, mass = _interval_masses(distribution, node_of_cell, len(nodes), num_intervals)
    T = len(edges) - 1
    midpoints = (edges[1:] + edges[:-1]) / 2
    y0, t0, dt0 = _tangent_cuts(n, k, num_cuts)

    ny = T + 1
    def y(i): return i
    def s(t): return ny + t
    def z(t): return ny + T + t
    num_variables = ny + 2 * T
    cost = np.zeros(num_variables)
    cost[1:ny] -= n * midpoints
    cost[:ny - 1] += n * midpoints
    cost[ny + T:] = -n

    rows, cols, vals, rhs = [], [], [], []
    row = 0
    def add(columns, values, bound):
        nonlocal row
        rows.extend([row] * len(columns)); cols.extend(columns); vals.extend(values)
        rhs.append(bound); row += 1

    descending = np.argsort(-nodes)
    for t in range(T):
        add([y(t), y(t + 1)], [1.0, -1.0], 0.0)                       # Δy >= 0
        total = mass[t].sum()
        add([y(t + 1), y(t), s(t)], [1.0, -1.0, -total], 0.0)        # mass <= available
        present = descending[mass[t, descending] > 0]
        cumulative_e = np.concatenate([[0.0], np.cumsum(nodes[present] * mass[t, present])])
        cumulative_m = np.concatenate([[0.0], np.cumsum(mass[t, present])])
        for index, node in enumerate(present):                       # z <= α s + β Δy
            alpha = cumulative_e[index] - nodes[node] * cumulative_m[index]
            beta = nodes[node]
            add([z(t), s(t), y(t + 1), y(t)], [1.0, -alpha, -beta, beta], 0.0)
        if len(present) == 0:
            add([z(t)], [1.0], 0.0)
        for q in range(len(y0)):                                     # s <= t(ȳ) by tangents
            add([s(t), y(t), y(t + 1)], [1.0, -dt0[q] / 2, -dt0[q] / 2],
                t0[q] - dt0[q] * y0[q])
    matrix = coo_matrix((vals, (rows, cols)), shape=(row, num_variables)).tocsr()
    bounds = [(0.0, k / n)] * ny + [(0.0, 1.0)] * T + [(None, None)] * T
    bounds[y(T)] = (k / n, k / n)
    solution = linprog(cost, A_ub=matrix, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if solution.x is None:
        raise RuntimeError(f"symmetric relaxation failed: {solution.message}")

    x = solution.x
    increments = np.maximum(np.diff(x[:ny]), 0.0)
    scale = x[ny:ny + T]
    participating = np.where(scale > 1e-12, increments / np.maximum(scale, 1e-12), 0.0)
    fraction = np.zeros((T, len(nodes)))
    for t in range(T):
        remaining = min(participating[t], mass[t].sum())
        for node in descending:
            if mass[t, node] <= 0:
                continue
            take = min(remaining, mass[t, node])
            fraction[t, node] = take / mass[t, node]
            remaining -= take
            if remaining <= 0:
                break
    present = mass > 0
    drops = [np.max(np.maximum.accumulate(f[p]) - f[p]) if p.any() else 0.0
             for f, p in zip(fraction.T, present.T)]
    thresholds = np.full(len(nodes), np.inf)
    for node in range(len(nodes)):
        admitted = np.flatnonzero(present[:, node] & (fraction[:, node] >= 0.5))
        if len(admitted):
            # Admitted from the node's lowest valuation on: τ = 0 gives the same participants
            # with the smallest penalty (the edge would put τ at the bottom of the support).
            lowest = np.flatnonzero(present[:, node])[0]
            thresholds[node] = 0.0 if admitted[0] == lowest else edges[admitted[0]]
    return RelaxationResult(
        value=float(-solution.fun), nodes=nodes, valuation_edges=edges, participation=fraction,
        thresholds=thresholds, largest_drop=float(max(drops)), status=solution.message,
        threshold_function=tabulated_threshold(nodes, thresholds))
