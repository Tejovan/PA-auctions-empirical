"""
threshold_forms.py

Parametric families ("forms") of the common threshold function τ(e) searched
by the optimizers (genetic_search.py, best_response_ascent.py).

  polynomial (label "poly<d>")  τ(e) = Σ_m β_m e^m, d + 1 parameters.
                                 Genetic search only.
  knots      (label "knots<K>") continuous and piecewise linear in e, with
                                 values θ_1..θ_K at K knots, constant beyond the
                                 end knots, plus an exclusion cutoff ê: types
                                 with e < ê are excluded (τ = v̄⁺). K + 1
                                 parameters (θ_1..θ_K, ê).
  bins       (label "bins<K>")  constant on each of K bins, K parameters.
  cutoff     (label "cutoff")   a welfare cutoff w: τ(e) = max(0, w − e), excluded
                                 (τ = v̄⁺) when w − e > v̄; 1 parameter. Participation is
                                 v + e >= w. w = 0 is the zero-welfare threshold τ⁰
                                 (paper §7.2 [sec:restrictions]); the cutoff of the ex
                                 ante relaxation is ex_ante_cutoff (Thm 8.9
                                 [thm:k-goods]). Genetic search only.
  nodes      (label "nodes")    one value per externality node of the
                                 distribution (its distinct externality values),
                                 linear interpolation in between. Best-response
                                 ascent only.

Knots and bin edges are evenly spaced quantiles of the marginal distribution of
the externality: knot k is the quantile of level (k − 1)/(K − 1), bin edges
are the quantiles of levels j/K. The quantile function interpolates the
cumulative distribution of the externality nodes linearly (each node's mass
centered on the node), so knots are distinct even when the externality has few
values.

Exclusion. The distributions are discretized in the externality: only τ at the
externality nodes matters for a given distribution. A value at or above
exclusion_value = v̄ (1 + 1e-9) + 1e-12 excludes the externality; the callable
returned by ThresholdForm.threshold maps such values to +inf (the threshold
v̄⁺ of the paper), so the externality stays excluded on another distribution
with a larger v̄ (for example test rows). Parameters stay finite, so that line
searches can interpolate between them.

A form is fixed by its kind, its size and its locations (knots, bin edges or
nodes); ThresholdForm.spec stores these with the parameters in a plain dict,
and threshold_from_spec rebuilds the threshold function from it.
"""

import re
from dataclasses import dataclass

import numpy as np

FORM_KINDS = ("polynomial", "knots", "bins", "cutoff", "nodes")
GENETIC_FORMS = ("polynomial", "knots", "bins", "cutoff")
BEST_RESPONSE_FORMS = ("knots", "bins", "nodes")


def parse_form_label(label):
    """'poly2' -> ('polynomial', 2); 'knots6' -> ('knots', 6); 'bins10'; 'nodes' -> ('nodes', None)."""
    if isinstance(label, (int, np.integer)):
        return "polynomial", int(label)
    text = str(label).strip().lower()
    if text == "nodes":
        return "nodes", None
    if text == "cutoff":
        return "cutoff", 1
    match = re.fullmatch(r"(poly|polynomial|knots|bins)(\d+)", text)
    if not match:
        raise ValueError(f"unknown τ form {label!r}: use poly<d>, knots<K>, bins<K>, cutoff "
                         f"or nodes")
    kind = "polynomial" if match.group(1).startswith("poly") else match.group(1)
    size = int(match.group(2))
    if kind == "knots" and size < 2:
        raise ValueError("knots<K> needs K >= 2")
    if kind == "bins" and size < 1:
        raise ValueError("bins<K> needs K >= 1")
    return kind, size


def externality_nodes(type_distribution):
    """Distinct externality values of the distribution and their total masses."""
    nodes, inverse = np.unique(type_distribution.externalities, return_inverse=True)
    masses = np.bincount(inverse, type_distribution.probabilities, len(nodes))
    return nodes, masses, inverse


def externality_quantiles(nodes, masses, levels):
    """Quantiles of the node distribution, by linear interpolation of the cumulative masses."""
    if len(nodes) == 1:
        return np.full(len(levels), nodes[0])
    centered = (np.cumsum(masses) - masses / 2) / masses.sum()
    return np.interp(levels, centered, nodes)


def exclusion_value_for(maximum_valuation):
    return float(maximum_valuation) * (1 + 1e-9) + 1e-12


@dataclass
class ThresholdForm:
    """A family of threshold functions, fixed by kind, size and locations.

    locations: knots (knots), the K − 1 interior bin edges (bins), the nodes
    (nodes), or empty (polynomial).
    """

    kind: str
    size: int
    locations: np.ndarray
    maximum_valuation: float
    externality_range: tuple

    # Construction and description -------------------------------------------

    @property
    def label(self):
        return {"polynomial": f"poly{self.size}", "knots": f"knots{self.size}",
                "bins": f"bins{self.size}", "cutoff": "cutoff", "nodes": "nodes"}[self.kind]

    @property
    def num_parameters(self):
        return {"polynomial": self.size + 1, "knots": self.size + 1, "bins": self.size,
                "cutoff": 1, "nodes": len(self.locations)}[self.kind]

    @property
    def exclusion_value(self):
        return exclusion_value_for(self.maximum_valuation)

    @property
    def no_cutoff_value(self):
        """Exclusion cutoff below every externality node (no exclusion), knots only."""
        low, high = self.externality_range
        return low - 0.25 * max(high - low, 1e-12)

    def spec(self, parameters):
        return {"form": self.kind, "size": self.size, "locations": self.locations.tolist(),
                "maximum_valuation": self.maximum_valuation,
                "externality_range": list(self.externality_range),
                "parameters": [float(p) for p in parameters], "label": self.label}

    @classmethod
    def from_spec(cls, spec):
        return cls(spec["form"], spec["size"], np.asarray(spec["locations"], dtype=float),
                   float(spec["maximum_valuation"]), tuple(spec["externality_range"]))

    # Values -------------------------------------------------------------------

    def finite_values(self, parameters, externalities):
        """τ(e) in the finite representation (values >= exclusion_value exclude)."""
        parameters = np.asarray(parameters, dtype=float)
        externalities = np.asarray(externalities, dtype=float)
        if self.kind == "polynomial":
            values = np.zeros_like(externalities)
            for coefficient in parameters[::-1]:
                values = values * externalities + coefficient
            return values
        if self.kind == "knots":
            values = np.interp(externalities, self.locations, parameters[:-1])
            return np.where(externalities < parameters[-1], self.exclusion_value, values)
        if self.kind == "bins":
            return parameters[np.searchsorted(self.locations, externalities, side="right")]
        if self.kind == "cutoff":
            values = parameters[0] - externalities
            return np.where(values > self.maximum_valuation, self.exclusion_value,
                            np.maximum(values, 0.0))
        return np.interp(externalities, self.locations, parameters)

    def threshold(self, parameters):
        """Vectorized threshold function τ(e); excluded externalities map to +inf."""
        parameters = np.asarray(parameters, dtype=float).copy()
        exclusion = self.exclusion_value

        def threshold(externalities):
            values = self.finite_values(parameters, externalities)
            return np.where(values >= exclusion, np.inf, values)

        threshold.description = f"{self.label} " + str(list(np.round(parameters, 6)))
        threshold.form_spec = self.spec(parameters)
        return threshold

    # Search ranges and projections ----------------------------------------------

    def parameter_bounds(self, type_distribution, range_multiplier=3.0):
        """(lows, highs) of the genetic search; polynomial ranges as in genetic_search."""
        if self.kind == "polynomial":
            from .genetic_search import compute_gene_space
            space = compute_gene_space(type_distribution, self.size, range_multiplier)
            return (np.array([g["low"] for g in space]), np.array([g["high"] for g in space]))
        if self.kind == "cutoff":  # w >= 0, up to the largest welfare contribution
            return np.array([0.0]), np.array([self.maximum_valuation
                                              + max(self.externality_range[1], 0.0)])
        probabilities = type_distribution.probabilities
        valuations = type_distribution.valuation_midpoints
        mean = np.sum(probabilities * valuations)
        margin = 0.5 * (float(np.sqrt(np.sum(probabilities * (valuations - mean) ** 2))) or 1.0)
        low_value = float(type_distribution.valuation_lower.min()) - margin
        high_value = type_distribution.maximum_valuation + margin
        num_values = self.num_parameters - (1 if self.kind == "knots" else 0)
        lows, highs = np.full(num_values, low_value), np.full(num_values, high_value)
        if self.kind == "knots":
            lows = np.append(lows, self.no_cutoff_value)
            highs = np.append(highs, self.externality_range[1])
        return lows, highs

    def project(self, threshold_function, type_distribution):
        """Parameters of this form closest to a threshold function, at the nodes.

        Node values are clipped to [0, exclusion_value]. knots: the cutoff
        excludes the longest run of excluded nodes starting at the lowest
        externality; knot values are τ at the knots (the first included
        node's value for knots below the cutoff). bins: mass-weighted mean
        of the node values in each bin. polynomial: weighted least squares on
        the included nodes.
        """
        nodes, masses, _ = externality_nodes(type_distribution)
        values = np.asarray(threshold_function(nodes), dtype=float)
        values = np.clip(np.where(np.isnan(values), 0.0, values), 0.0, self.exclusion_value)
        excluded = values >= self.exclusion_value
        if self.kind == "nodes":
            return np.interp(self.locations, nodes, values)
        if self.kind == "cutoff":  # mass-weighted mean of τ(e) + e at interior nodes
            interior = (values > 0) & ~excluded
            if not interior.any():
                return np.array([0.0])
            return np.array([max(0.0, float(np.average(values[interior] + nodes[interior],
                                                       weights=masses[interior])))])
        if self.kind == "knots":
            num_excluded = int(np.argmin(excluded)) if not excluded.all() else len(nodes)
            if num_excluded == 0:
                cutoff = self.no_cutoff_value
            elif num_excluded == len(nodes):
                return np.append(np.full(self.size, self.exclusion_value),
                                 self.externality_range[1] + 1.0)
            else:
                cutoff = 0.5 * (nodes[num_excluded - 1] + nodes[num_excluded])
            included_nodes, included_values = nodes[num_excluded:], values[num_excluded:]
            knot_values = np.interp(self.locations, included_nodes, included_values)
            return np.append(knot_values, cutoff)
        if self.kind == "bins":
            bin_of_node = np.searchsorted(self.locations, nodes, side="right")
            weight = np.bincount(bin_of_node, masses, self.size)
            total = np.bincount(bin_of_node, masses * values, self.size)
            fallback = np.interp(np.arange(self.size), bin_of_node, values)
            return np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), fallback)
        included = ~excluded
        if included.sum() <= self.size:
            return np.append(values[included].mean() if included.any() else 0.0,
                             np.zeros(self.size))
        return np.polynomial.polynomial.polyfit(nodes[included], values[included], self.size,
                                                w=np.sqrt(masses[included]))


def build_threshold_form(label, type_distribution):
    """The form named by label ('poly2', 'knots6', 'bins10', 'nodes') for this distribution."""
    kind, size = parse_form_label(label)
    nodes, masses, _ = externality_nodes(type_distribution)
    externality_range = (float(nodes[0]), float(nodes[-1]))
    if kind in ("polynomial", "cutoff"):
        locations = np.empty(0)
    elif kind == "knots":
        locations = np.unique(externality_quantiles(nodes, masses, np.linspace(0, 1, size)))
        if len(locations) < size:  # a single externality value
            locations = np.linspace(nodes[0] - 0.5, nodes[-1] + 0.5, size)
    elif kind == "bins":
        locations = externality_quantiles(nodes, masses, np.linspace(0, 1, size + 1)[1:-1])
    else:
        locations, size = nodes.copy(), len(nodes)
    return ThresholdForm(kind, size, np.asarray(locations, dtype=float),
                         float(type_distribution.maximum_valuation), externality_range)


def threshold_from_spec(spec, parameters=None):
    """Threshold function of a stored spec (ThresholdForm.spec), optionally with other parameters."""
    form = ThresholdForm.from_spec(spec)
    return form.threshold(spec["parameters"] if parameters is None else parameters)


def ex_ante_cutoff(type_distribution, num_bidders, num_goods):
    """Welfare cutoff w* of the ex ante relaxation for k identical goods (paper Thm 8.9
    [thm:k-goods]): Pr[v + e > w*] = q* = min{k/n, Pr[v + e > 0]}, w* >= 0.

    The welfare contribution w = v + e is uniform on [lower + e, upper + e] within a
    cell. For atomic distributions, Pr[w > c] is a step function; the cutoff is then the
    smallest c with Pr[w > c] <= q*, without the boundary mixing of the paper.
    """
    low = type_distribution.valuation_lower + type_distribution.externalities
    high = type_distribution.valuation_upper + type_distribution.externalities
    probabilities = type_distribution.probabilities

    def mass_above(c):
        width = high - low
        fraction = np.where(width > 0, np.clip((high - c) / np.where(width > 0, width, 1.0),
                                               0.0, 1.0), (low > c).astype(float))
        return float(np.sum(probabilities * fraction))

    target = min(num_goods / num_bidders, mass_above(0.0))
    if mass_above(0.0) <= target:
        return 0.0
    left, right = 0.0, float(high.max())
    for _ in range(200):
        middle = 0.5 * (left + right)
        if mass_above(middle) > target:
            left = middle
        else:
            right = middle
    return right


def ex_ante_guarantee(num_bidders, num_goods):
    """max{β_{n,k}, 1 − 1/e} of paper Thm 8.9 [thm:k-goods]:
    β_{n,k} = 1 − C(n, k) (1 − k/n)^{n+1−k} (k/n)^k."""
    from math import comb, exp

    n, k = num_bidders, num_goods
    beta = 1 - comb(n, k) * (1 - k / n) ** (n + 1 - k) * (k / n) ** k
    return max(beta, 1 - exp(-1))
