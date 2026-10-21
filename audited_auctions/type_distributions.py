"""
type_distributions.py

Representations of the joint distribution H of bidder types (e, v), where e is
the externality and v >= 0 is the valuation.

Every distribution used by the package is a finite list of cells. Cell j has an
externality e_j, a valuation interval [lower_j, upper_j] and a probability p_j;
given cell j, the valuation is uniform on [lower_j, upper_j]. Two kinds:

  * atomic: every interval is a single point (lower_j = upper_j). Used for a
    finite population (for example the Community Notes data) used as is:
    bidders are i.i.d. draws of the population rows.

  * piecewise_uniform: every interval has positive length. Used for continuous
    densities (the synthetic families below, and the kernel-smoothed empirical
    distribution of smoothing.py). The externality takes the values of a grid of
    nodes; given the externality, the valuation has a piecewise-constant
    density, so the valuation distribution has no atoms and ties between
    valuation bids have probability zero.

Mixed distributions (some atoms, some intervals) are not supported.

Continuous version. A piecewise-uniform distribution on a grid of externality
nodes is a discretization of a continuous density: the externality takes the
node values, and the valuation density jumps at the cell edges. Under a
continuous τ, a whole node then enters the participants at its threshold, and
E[e | v] jumps at every cell edge where the mix of externalities changes, so
the expected welfare contribution ŵ^τ has small drops at every node threshold
and every cell edge, and IVCG-PA irons in many small bands (a discretization
effect of order (spacing)²). continuous_version removes both jumps:

  * externalities: the mass of each node is spread uniformly over its
    externality interval (from the midpoint to the previous node to the
    midpoint to the next; the end nodes get half a spacing on the outside);
  * valuations: within each node's column of cells, the density is the
    piecewise-linear interpolation of the cell densities between the cell
    midpoints (constant from the outer midpoints to the ends of a run of
    adjacent cells); cell masses are renormalized to sum to one.

Each cell is then a rectangle [externality_lower_j, externality_upper_j] ×
[lower_j, upper_j] with a density uniform in e and piecewise linear in v (its
values at lower_j, the midpoint and upper_j are proportional to
valuation_shape_lower_j, 1 and valuation_shape_upper_j). The distribution
differs from the gridded one by a discretization error of the same order as
the grid's own. Evaluation: exact_computation.continuous_bid_segments.
Distributions with genuinely finitely many externality values
(discrete_externality_distribution, metadata externality_support = "discrete")
keep their externality values (only valuations become piecewise linear).
refined_version gives a discrete approximation of the continuous version
(point externalities and uniform valuation pieces on a finer grid; used for
the first best).

Model assumptions enforced here: valuations are non-negative and bounded, with
upper bound v̄ = max_j upper_j (TypeDistribution.maximum_valuation).

Both the exact computations (exact_computation.py) and the Monte Carlo
simulation (auction_simulation.py) use the same cells, so exact expectations
and simulated means agree up to Monte Carlo error. For a continuous density,
the discretization error relative to that density shrinks with the grid
spacing; check it with a finer grid (threshold_evaluation.grid_refinement_check).

Bidders are exchangeable: all n bidders draw their types independently from
the same distribution.

Synthetic families (SYNTHETIC_FAMILIES, synthetic_distribution) are documented
in audited_auctions/README.md, section "Type distributions".
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

ATOMIC = "atomic"
PIECEWISE_UNIFORM = "piecewise_uniform"


@dataclass
class TypeDistribution:
    """A joint distribution of bidder types (e, v) made of cells.

    Attributes
    ----------
    externalities   : (num_cells,) externality e_j of each cell
    valuation_lower : (num_cells,) lower end of the valuation interval
    valuation_upper : (num_cells,) upper end (equal to lower for atoms)
    probabilities   : (num_cells,) probability of each cell; sums to one
    name            : short label used in figure titles
    density_grid    : optional, for plotting contours only: dict with keys
                      "externality_grid", "valuation_grid", "density"
                      (density has shape (len(valuation_grid), len(externality_grid)))
    metadata        : free-form provenance (family parameters, smoothing settings)
    externality_lower, externality_upper : optional (num_cells,) externality interval
                      of each cell (continuous version, see the module notes);
                      None: the externality of a cell is the point externalities_j
    valuation_shape_lower, valuation_shape_upper : optional (num_cells,) valuation
                      density at the cell's ends relative to its midpoint value
                      (continuous version); None: uniform valuations
    cache           : results that depend only on this distribution (not on τ),
                      e.g. VCG and first-best outcomes; filled by
                      threshold_evaluation.tau_independent_outcomes. Not copied
                      by dataclasses.replace, so transformed distributions start empty.
    """

    externalities: np.ndarray
    valuation_lower: np.ndarray
    valuation_upper: np.ndarray
    probabilities: np.ndarray
    name: str = ""
    density_grid: Optional[dict] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict, repr=False)
    externality_lower: Optional[np.ndarray] = field(default=None, repr=False)
    externality_upper: Optional[np.ndarray] = field(default=None, repr=False)
    valuation_shape_lower: Optional[np.ndarray] = field(default=None, repr=False)
    valuation_shape_upper: Optional[np.ndarray] = field(default=None, repr=False)
    cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.externalities = np.asarray(self.externalities, dtype=float).ravel()
        self.valuation_lower = np.asarray(self.valuation_lower, dtype=float).ravel()
        self.valuation_upper = np.asarray(self.valuation_upper, dtype=float).ravel()
        self.probabilities = np.asarray(self.probabilities, dtype=float).ravel()
        if not (
            self.externalities.shape
            == self.valuation_lower.shape
            == self.valuation_upper.shape
            == self.probabilities.shape
        ):
            raise ValueError("externalities, valuation bounds and probabilities must align")
        if np.any(self.probabilities < 0) or not np.all(np.isfinite(self.probabilities)):
            raise ValueError("probabilities must be finite and nonnegative")
        if np.any(self.valuation_upper < self.valuation_lower):
            raise ValueError("valuation_upper must be >= valuation_lower")
        if np.any(self.valuation_lower < 0):
            raise ValueError(
                "valuations must be non-negative (model assumption); "
                f"smallest valuation is {self.valuation_lower.min():g}"
            )
        if not np.all(np.isfinite(self.valuation_upper)):
            raise ValueError("valuations must be bounded")
        # Drop zero-probability cells and renormalize.
        keep = self.probabilities > 0
        if not keep.any():
            raise ValueError("all probabilities are zero")
        self.externalities = self.externalities[keep]
        self.valuation_lower = self.valuation_lower[keep]
        self.valuation_upper = self.valuation_upper[keep]
        for name in ("externality_lower", "externality_upper", "valuation_shape_lower",
                     "valuation_shape_upper"):
            if getattr(self, name) is not None:
                setattr(self, name, np.asarray(getattr(self, name), dtype=float).ravel()[keep])
        self.probabilities = self.probabilities[keep] / self.probabilities[keep].sum()
        point_cells = self.valuation_upper == self.valuation_lower
        if point_cells.all():
            self.kind = ATOMIC
        elif not point_cells.any():
            self.kind = PIECEWISE_UNIFORM
        else:
            raise ValueError("mixed distributions (atoms and intervals) are not supported")
        if self.is_continuous and self.kind == ATOMIC:
            raise ValueError("a continuous version needs piecewise-uniform cells")

    # Basic properties ----------------------------------------------------------

    @property
    def num_cells(self) -> int:
        return len(self.probabilities)

    @property
    def is_atomic(self) -> bool:
        return self.kind == ATOMIC

    @property
    def is_continuous(self) -> bool:
        """A continuous version (continuous_version): density continuous in v, and in e
        unless the externality takes finitely many values."""
        return self.valuation_shape_lower is not None

    @property
    def has_externality_intervals(self) -> bool:
        return self.externality_lower is not None

    def cell_externality_intervals(self):
        """(lower, upper) externality of each cell; equal for point externalities."""
        if self.has_externality_intervals:
            return self.externality_lower, self.externality_upper
        return self.externalities, self.externalities

    def cell_valuation_shapes(self):
        """(lower, upper) valuation density at the cell ends relative to the midpoint."""
        if self.valuation_shape_lower is not None:
            return self.valuation_shape_lower, self.valuation_shape_upper
        ones = np.ones(self.num_cells)
        return ones, ones

    @property
    def valuation_midpoints(self) -> np.ndarray:
        return (self.valuation_lower + self.valuation_upper) / 2

    @property
    def valuations(self) -> np.ndarray:
        """Valuation of each atom (atomic distributions only)."""
        if not self.is_atomic:
            raise AttributeError("valuations is defined for atomic distributions; "
                                 "use valuation_lower/valuation_upper or valuation_midpoints")
        return self.valuation_lower

    @property
    def maximum_valuation(self) -> float:
        """Upper bound v̄ of the valuations."""
        return float(self.valuation_upper.max())

    @property
    def mean_valuation(self) -> float:
        return float(np.sum(self.probabilities * self.valuation_midpoints))

    @property
    def mean_externality(self) -> float:
        return float(np.sum(self.probabilities * self.externalities))

    # Sampling ------------------------------------------------------------------

    def sample_types(self, num_draws, num_bidders, rng):
        """Draw (externalities, valuations), each of shape (num_draws, num_bidders)."""
        cells = rng.choice(self.num_cells, size=(num_draws, num_bidders), p=self.probabilities)
        lower, upper = self.valuation_lower[cells], self.valuation_upper[cells]
        uniform = rng.random(cells.shape)
        if self.is_continuous:
            valuations = _sample_hat(lower, upper, self.valuation_shape_lower[cells],
                                     self.valuation_shape_upper[cells], uniform,
                                     rng.random(cells.shape))
        else:
            valuations = lower + uniform * (upper - lower)
        if self.has_externality_intervals:
            low, high = self.externality_lower[cells], self.externality_upper[cells]
            return low + rng.random(cells.shape) * (high - low), valuations
        return self.externalities[cells], valuations

    def sample_type_points(self, num_points, rng):
        """num_points i.i.d. types as two flat arrays (for scatter plots)."""
        externalities, valuations = self.sample_types(num_points, 1, rng)
        return externalities[:, 0], valuations[:, 0]

    def weighted_quantile(self, values, quantile) -> float:
        """Quantile of a cell-level quantity (e.g. externalities) under this distribution."""
        values = np.asarray(values, dtype=float)
        order = np.argsort(values)
        cumulative = np.cumsum(self.probabilities[order])
        position = np.searchsorted(cumulative, quantile, side="left")
        return float(values[order][min(position, len(order) - 1)])

    # Transformations -----------------------------------------------------------

    def with_externality_scale(self, externality_scale, name=None):
        """The distribution of (ζ e, v): externalities multiplied by ζ > 0."""
        if externality_scale <= 0:
            raise ValueError("externality_scale must be positive")
        grid = None
        if self.density_grid is not None:
            grid = {
                "externality_grid": self.density_grid["externality_grid"] * externality_scale,
                "valuation_grid": self.density_grid["valuation_grid"],
                "density": self.density_grid["density"] / externality_scale,
            }
        intervals = {}
        if self.has_externality_intervals:
            intervals = {"externality_lower": self.externality_lower * externality_scale,
                         "externality_upper": self.externality_upper * externality_scale}
        return replace(
            self,
            externalities=self.externalities * externality_scale,
            name=name or f"{self.name} (ζ = {externality_scale:g})",
            density_grid=grid,
            metadata={**self.metadata, "externality_scale": externality_scale},
            **intervals,
        )

    def continuous_version(self):
        """The continuous version of a piecewise-uniform distribution (module notes);
        self for atomic distributions and continuous versions."""
        if self.is_atomic or self.is_continuous:
            return self
        intervals = {}
        nodes, node_of_cell = np.unique(self.externalities, return_inverse=True)
        node_of_cell = node_of_cell.ravel()
        spread = (self.metadata.get("externality_support") != "discrete" and len(nodes) > 1)
        if spread:
            edges = np.concatenate([[nodes[0] - (nodes[1] - nodes[0]) / 2],
                                    (nodes[:-1] + nodes[1:]) / 2,
                                    [nodes[-1] + (nodes[-1] - nodes[-2]) / 2]])
            intervals = {"externality_lower": edges[node_of_cell],
                         "externality_upper": edges[node_of_cell + 1]}
        shape_lower, shape_upper = _interpolated_shapes(
            node_of_cell, self.valuation_lower, self.valuation_upper,
            self.probabilities / (self.valuation_upper - self.valuation_lower))
        probabilities = self.probabilities * (shape_lower + 2 + shape_upper) / 4
        externalities = (self.externalities if not spread
                         else (intervals["externality_lower"] + intervals["externality_upper"]) / 2)
        return replace(self, externalities=externalities, probabilities=probabilities,
                       valuation_shape_lower=shape_lower, valuation_shape_upper=shape_upper,
                       metadata={**self.metadata, "version": "continuous"}, **intervals)

    def refined_version(self, externality_points=4, valuation_pieces=2):
        """Discrete approximation of a continuous version: each cell split into
        externality_points equally spaced point externalities × valuation_pieces
        uniform valuation pieces, with the cell's mass on each piece. self if not
        continuous."""
        if not self.is_continuous:
            return self
        m, r = (int(externality_points) if self.has_externality_intervals else 1,
                int(valuation_pieces))
        low_e, high_e = self.cell_externality_intervals()
        e = low_e[:, None] + (high_e - low_e)[:, None] * ((np.arange(m) + 0.5) / m)  # (C, m)
        # Valuation pieces and their masses under the piecewise-linear density.
        cuts = self.valuation_lower[:, None] + (self.valuation_upper - self.valuation_lower)[
            :, None] * (np.arange(r + 1) / r)                                            # (C, r+1)
        density = _hat_density(self.valuation_lower[:, None], self.valuation_upper[:, None],
                               self.valuation_shape_lower[:, None],
                               self.valuation_shape_upper[:, None], cuts)
        middle = _hat_density(self.valuation_lower[:, None], self.valuation_upper[:, None],
                              self.valuation_shape_lower[:, None],
                              self.valuation_shape_upper[:, None], (cuts[:, 1:] + cuts[:, :-1]) / 2)
        piece_mass = (density[:, :-1] + 4 * middle + density[:, 1:]) / 6   # Simpson; exact
        piece_mass = piece_mass / piece_mass.sum(axis=1, keepdims=True)
        cells = self.num_cells
        mass = (self.probabilities[:, None, None] * piece_mass[:, None, :] / m) * np.ones((1, m, 1))
        return TypeDistribution(
            np.broadcast_to(e[:, :, None], (cells, m, r)).ravel(),
            np.broadcast_to(cuts[:, None, :-1], (cells, m, r)).ravel(),
            np.broadcast_to(cuts[:, None, 1:], (cells, m, r)).ravel(), mass.ravel(),
            name=self.name, density_grid=self.density_grid,
            metadata={**self.metadata, "version": f"refined ({m} × {r} per cell)"})


def _interpolated_shapes(column, lower, upper, density):
    """Valuation density at the ends of each cell, relative to the cell's density, of the
    piecewise-linear interpolation between cell midpoints within each column (runs of
    adjacent cells); constant from the outer midpoints of a run to its ends."""
    order = np.lexsort((lower, column))
    col, low, high, dens = column[order], lower[order], upper[order], density[order]
    middle = (low + high) / 2
    scale = max(float(np.max(high)), 1.0)
    adjacent = (col[1:] == col[:-1]) & (np.abs(low[1:] - high[:-1]) <= 1e-12 * scale)
    edge = dens[:-1] + (dens[1:] - dens[:-1]) * (high[:-1] - middle[:-1]) / (middle[1:]
                                                                                 - middle[:-1])
    upper_value = dens.copy()
    lower_value = dens.copy()
    upper_value[:-1] = np.where(adjacent, edge, dens[:-1])
    lower_value[1:] = np.where(adjacent, edge, dens[1:])
    shape_lower, shape_upper = np.empty(len(order)), np.empty(len(order))
    shape_lower[order] = lower_value / dens
    shape_upper[order] = upper_value / dens
    return shape_lower, shape_upper


def _hat_density(lower, upper, shape_lower, shape_upper, valuations):
    """Unnormalized valuation density of a cell: linear from shape_lower at lower to 1 at
    the midpoint to shape_upper at upper."""
    middle = (lower + upper) / 2
    half = (upper - lower) / 2
    left = shape_lower + (1 - shape_lower) * (valuations - lower) / half
    right = 1 + (shape_upper - 1) * (valuations - middle) / half
    return np.where(valuations <= middle, left, right)


def _sample_linear(start, end, uniform):
    """t in [0, 1] with density proportional to start + (end − start) t."""
    slope = end - start
    total = (start + end) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        root = (-start + np.sqrt(np.maximum(start ** 2 + 2 * slope * uniform * total, 0.0))) / slope
    return np.where(np.abs(slope) > 1e-12 * np.maximum(total, 1e-300), root, uniform)


def _sample_hat(lower, upper, shape_lower, shape_upper, uniform, choice):
    """Valuations from the piecewise-linear density of each cell (see _hat_density)."""
    left_mass, right_mass = shape_lower + 1, 1 + shape_upper
    left = choice * (left_mass + right_mass) < left_mass
    half = (upper - lower) / 2
    t = np.where(left, _sample_linear(shape_lower, np.ones_like(shape_lower), uniform),
                 _sample_linear(np.ones_like(shape_upper), shape_upper, uniform))
    return np.where(left, lower, lower + half) + half * t


# ─── Populations (atomic) ─────────────────────────────────────────────────────


def population_distribution(externalities, valuations, name="population", metadata=None):
    """Uniform distribution over the rows of a finite population (atomic)."""
    externalities = np.asarray(externalities, dtype=float)
    valuations = np.asarray(valuations, dtype=float)
    probabilities = np.full(len(externalities), 1.0 / len(externalities))
    return TypeDistribution(externalities, valuations, valuations, probabilities,
                            name=name, metadata=metadata or {})


def load_population_csv(csv_path, externality_scale=1.0, valuation_scale=1.0):
    """(externalities, valuations) arrays from a CSV with columns e_score and v_score.

    The externality is e = externality_scale * e_score (the externality cost ζ
    of a run configuration); the valuation is v = valuation_scale * v_score (the
    value of one action).
    """
    data = pd.read_csv(csv_path, usecols=["e_score", "v_score"])
    return (
        externality_scale * data["e_score"].to_numpy(dtype=float),
        valuation_scale * data["v_score"].to_numpy(dtype=float),
    )


def population_distribution_from_csv(csv_path, externality_scale=1.0, valuation_scale=1.0,
                                     name=None):
    """Atomic population distribution of a CSV (no truncation; see empirical.py)."""
    externalities, valuations = load_population_csv(csv_path, externality_scale, valuation_scale)
    return population_distribution(externalities, valuations, name=name or str(csv_path))


# ─── Gridded densities (piecewise uniform) ────────────────────────────────────


def _node_weights(nodes):
    """Integration weights of grid nodes: the length of each node's Voronoi cell."""
    nodes = np.asarray(nodes, dtype=float)
    if len(nodes) == 1:
        return np.ones(1)
    edges = np.concatenate([[nodes[0]], (nodes[:-1] + nodes[1:]) / 2, [nodes[-1]]])
    weights = np.diff(edges)
    # End nodes represent half cells on a closed grid; give them full spacing so
    # that a grid spanning ±5 standard deviations is not biased at its ends.
    weights[0] = nodes[1] - nodes[0]
    weights[-1] = nodes[-1] - nodes[-2]
    return weights


def gridded_density_distribution(density_function, externality_nodes, valuation_edges,
                                 name="density", externality_weights=None, metadata=None):
    """Piecewise-uniform approximation of a joint density h(e, v).

    density_function  : callable (E, V) -> density, broadcasting over arrays
    externality_nodes : (num_e,) increasing externality values (the cells' e)
    valuation_edges   : (num_v + 1,) increasing valuation cell edges, edges[0] >= 0
    externality_weights : optional weights of the externality nodes (default:
                        node spacing); pass the probabilities of a discrete
                        externality distribution here.

    Cell (e_j, [edges_i, edges_{i+1}]) gets probability proportional to
    h(e_j, midpoint_i) × (edges_{i+1} − edges_i) × weight_j (midpoint rule), and
    the valuation is uniform within the cell.
    """
    externality_nodes = np.asarray(externality_nodes, dtype=float)
    valuation_edges = np.asarray(valuation_edges, dtype=float)
    if np.any(np.diff(valuation_edges) <= 0):
        raise ValueError("valuation_edges must be strictly increasing")
    weights = (_node_weights(externality_nodes) if externality_weights is None
               else np.asarray(externality_weights, dtype=float))
    midpoints = (valuation_edges[:-1] + valuation_edges[1:]) / 2
    widths = np.diff(valuation_edges)
    externality_mesh, midpoint_mesh = np.meshgrid(externality_nodes, midpoints)
    density = np.asarray(density_function(externality_mesh, midpoint_mesh), dtype=float)
    density = np.where(np.isfinite(density) & (density > 0), density, 0.0)
    cell_mass = density * widths[:, None] * weights[None, :]
    lower_mesh = np.broadcast_to(valuation_edges[:-1][:, None], cell_mass.shape)
    upper_mesh = np.broadcast_to(valuation_edges[1:][:, None], cell_mass.shape)
    total = cell_mass.sum()
    return TypeDistribution(
        externality_mesh.ravel(), lower_mesh.ravel(), upper_mesh.ravel(), cell_mass.ravel(),
        name=name,
        density_grid={
            "externality_grid": externality_nodes,
            "valuation_grid": midpoints,
            "density": density * (cell_mass.sum() / total if total > 0 else 1.0),
        },
        metadata=metadata or {},
    )


def discrete_externality_distribution(externality_levels, level_probabilities,
                                      conditional_valuation_density, valuation_edges,
                                      name="discrete externalities", metadata=None):
    """Finitely many externality levels, each with a continuous valuation density.

    conditional_valuation_density : callable (E, V) -> density of v given e
    (each conditional density is renormalized on the valuation edges).
    """
    levels = np.asarray(externality_levels, dtype=float)
    level_probabilities = np.asarray(level_probabilities, dtype=float)
    valuation_edges = np.asarray(valuation_edges, dtype=float)
    midpoints = (valuation_edges[:-1] + valuation_edges[1:]) / 2
    widths = np.diff(valuation_edges)
    level_mesh, midpoint_mesh = np.meshgrid(levels, midpoints)
    conditional = np.asarray(conditional_valuation_density(level_mesh, midpoint_mesh), dtype=float)
    conditional_mass = conditional * widths[:, None]
    conditional_mass /= conditional_mass.sum(axis=0, keepdims=True)
    cell_mass = conditional_mass * level_probabilities[None, :]
    lower_mesh = np.broadcast_to(valuation_edges[:-1][:, None], cell_mass.shape)
    upper_mesh = np.broadcast_to(valuation_edges[1:][:, None], cell_mass.shape)
    return TypeDistribution(
        level_mesh.ravel(), lower_mesh.ravel(), upper_mesh.ravel(), cell_mass.ravel(),
        name=name, metadata={**(metadata or {}), "externality_support": "discrete"},
    )


# ─── Density building blocks ──────────────────────────────────────────────────


def _normal_pdf(x, mean, standard_deviation):
    z = (x - mean) / standard_deviation
    return np.exp(-0.5 * z * z) / (standard_deviation * np.sqrt(2 * np.pi))


def gaussian_mixture_density(component_means, component_covariances, component_weights):
    """Callable density h(e, v) of a mixture of bivariate normals; coordinates (e, v)."""
    means = [np.asarray(m, dtype=float) for m in component_means]
    covariances = [np.asarray(c, dtype=float) for c in component_covariances]
    weights = np.asarray(component_weights, dtype=float)
    weights = weights / weights.sum()

    def density(externality, valuation):
        total = np.zeros(np.broadcast(externality, valuation).shape)
        for mean, covariance, weight in zip(means, covariances, weights):
            inverse = np.linalg.inv(covariance)
            determinant = np.linalg.det(covariance)
            de = externality - mean[0]
            dv = valuation - mean[1]
            quadratic_form = (inverse[0, 0] * de * de + 2 * inverse[0, 1] * de * dv
                              + inverse[1, 1] * dv * dv)
            total += weight * np.exp(-0.5 * quadratic_form) / (2 * np.pi * np.sqrt(determinant))
        return total

    return density


def gaussian_mixture_distribution(component_means, component_covariances, component_weights=None,
                                  num_externality_nodes=151, num_valuation_cells=150,
                                  num_standard_deviations=5.0, name="gaussian mixture",
                                  metadata=None):
    """Gaussian mixture over (e, v), truncated to v >= 0 and discretized.

    The grid spans every component mean ± num_standard_deviations standard
    deviations, with the valuation range cut at 0; the density is truncated to
    that rectangle and renormalized, so v̄ = largest mean + 5 sd.
    """
    if component_weights is None:
        component_weights = np.ones(len(component_means))
    means = np.array(component_means, dtype=float)
    standard_deviations = np.array(
        [np.sqrt(np.diag(np.asarray(c, dtype=float))) for c in component_covariances])
    lower = (means - num_standard_deviations * standard_deviations).min(axis=0)
    upper = (means + num_standard_deviations * standard_deviations).max(axis=0)
    lower[1] = max(lower[1], 0.0)
    return gridded_density_distribution(
        gaussian_mixture_density(component_means, component_covariances, component_weights),
        np.linspace(lower[0], upper[0], num_externality_nodes),
        np.linspace(lower[1], upper[1], num_valuation_cells + 1),
        name=name, metadata=metadata,
    )


# ─── Synthetic families ───────────────────────────────────────────────────────
# Each family is a function of grid options (and its own parameters) returning a
# TypeDistribution with valuations in [0, v̄]. See README "Type distributions".

_WIDE = [[0.1, 0.0], [0.0, 0.1]]
_NARROW = [[0.05, 0.0], [0.0, 0.05]]

# The mixtures a–f of legacy/alternate_distributions.ipynb, with valuation means shifted
# up by 1 so that truncation at v = 0 removes almost no mass and the shapes are
# kept (legacy/data/samples/ has the unshifted, untruncated CSV versions).
_MIXTURES = {
    "a": ("A single modal normal", [[0, 1]], _WIDE),
    "b": ("B two modal (side by side)", [[-1, 1], [0, 1]], _NARROW),
    "c": ("C two modal (diagonal .')", [[-1, 1], [0, 2]], _NARROW),
    "d": ("D two modal (diagonal '.)", [[-1, 2], [0, 1]], _NARROW),
    "e": ("E two modal (stacked)", [[0, 2], [0, 1]], _NARROW),
    "f": ("F four modal", [[-1, 1], [-1, 2], [0, 2], [0, 1]], _NARROW),
    "h": ("H shifted normal", [[0, 2]], _WIDE),
}


def _mixture_family(letter):
    title, means, covariance = _MIXTURES[letter]

    def build(num_externality_nodes=151, num_valuation_cells=150):
        return gaussian_mixture_distribution(
            means, [covariance] * len(means),
            num_externality_nodes=num_externality_nodes,
            num_valuation_cells=num_valuation_cells,
            name=title,
            metadata={"family": letter, "means": means, "covariance": covariance},
        )

    return build


def example_n_distribution(harm=0.6, num_externality_nodes=None, num_valuation_cells=150):
    """Example N of the paper (Ex. 7.11 [ex:N]): v ~ U[0,1], e ∈ {0, −harm} w.p. ½ each, independent.

    With n = 2, k = 1 and τ(0) <= 0, τ(−harm) = t, VCG-PA welfare is
    2/3 − c/2 + (c/2) t² − t³/4 (c = harm), maximized at t* = 4c/3 for c <= 3/4,
    and the penalty inducing t is (3/4) t². Used as an exact benchmark.
    """
    del num_externality_nodes  # two levels; kept for a uniform signature
    return discrete_externality_distribution(
        [-harm, 0.0], [0.5, 0.5], lambda e, v: np.ones_like(v),
        np.linspace(0.0, 1.0, num_valuation_cells + 1),
        name=f"N two externality levels (c = {harm:g})",
        metadata={"family": "example_n", "harm": harm},
    )


def independent_distribution(externality_mean=-0.3, externality_sd=0.5,
                             num_externality_nodes=151, num_valuation_cells=150):
    """v ~ U[0, 1] independent of e ~ N(externality_mean, externality_sd²)."""
    return gridded_density_distribution(
        lambda e, v: _normal_pdf(e, externality_mean, externality_sd) * np.ones_like(v),
        np.linspace(externality_mean - 5 * externality_sd, externality_mean + 5 * externality_sd,
                    num_externality_nodes),
        np.linspace(0.0, 1.0, num_valuation_cells + 1),
        name="I independent: v ~ U[0,1], e normal",
        metadata={"family": "independent", "externality_mean": externality_mean,
                  "externality_sd": externality_sd},
    )


def harm_rises_distribution(correlation=-0.6, means=(-0.3, 1.0), standard_deviations=(0.5, 0.4),
                            num_externality_nodes=151, num_valuation_cells=150):
    """Bivariate normal (e, v) with negative correlation, truncated to v ∈ [0, v̄].

    Harm grows with valuation: the bidders who would win under VCG tend to have
    the most negative externalities.
    """
    covariance = [[standard_deviations[0] ** 2,
                   correlation * standard_deviations[0] * standard_deviations[1]],
                  [correlation * standard_deviations[0] * standard_deviations[1],
                   standard_deviations[1] ** 2]]
    distribution = gaussian_mixture_distribution(
        [list(means)], [covariance],
        num_externality_nodes=num_externality_nodes, num_valuation_cells=num_valuation_cells,
        name=f"R harm rises with valuation (ρ = {correlation:g})",
        metadata={"family": "harm_rises", "correlation": correlation, "means": list(means),
                  "standard_deviations": list(standard_deviations)},
    )
    return distribution


def harmful_middle_distribution(valuation_mean=1.0, valuation_sd=0.45, externality_mean=-0.3,
                                slope=-0.9, externality_sd=0.4, wiggle_amplitude=0.5,
                                wiggle_width=0.26, wiggle_center=1.0, wiggle_spread=0.38,
                                num_externality_nodes=151, num_valuation_cells=150):
    """v ~ N(valuation_mean, valuation_sd²) truncated to v ∈ [0, v̄];
    given v, e ~ N(m(v), s(v)²).

    The conditional mean m(v) = externality_mean + slope (v − valuation_mean)
    − wiggle_amplitude x exp((1 − x²)/2), x = (v − wiggle_center)/wiggle_width, falls
    steeply like harm_rises, with a wiggle in the middle: m(v) is
    wiggle_amplitude above the line at v = wiggle_center − wiggle_width and as far
    below it at v = wiggle_center + wiggle_width. Between the two, m(v) falls with
    slope slope − e^{1/2} wiggle_amplitude / wiggle_width, so the expected welfare
    contribution v + E[e | v] decreases there and IVCG-PA irons (a continuous
    analogue of Example T, Ex. 6.4 [ex:T-pool]). The conditional standard deviation
    s(v) = externality_sd + wiggle_spread exp(−x²/2) widens the ridge at the wiggle.
    """
    def conditional_mean(v):
        x = (v - wiggle_center) / wiggle_width
        return (externality_mean + slope * (v - valuation_mean)
                - wiggle_amplitude * x * np.exp(0.5 - 0.5 * x * x))

    def conditional_sd(v):
        x = (v - wiggle_center) / wiggle_width
        return externality_sd + wiggle_spread * np.exp(-0.5 * x * x)

    def density(e, v):
        return (_normal_pdf(v, valuation_mean, valuation_sd)
                * _normal_pdf(e, conditional_mean(v), conditional_sd(v)))

    valuation_edges = np.linspace(max(0.0, valuation_mean - 5 * valuation_sd),
                                  valuation_mean + 5 * valuation_sd, num_valuation_cells + 1)
    valuations = np.linspace(valuation_edges[0], valuation_edges[-1], 1001)
    means, sds = conditional_mean(valuations), conditional_sd(valuations)
    return gridded_density_distribution(
        density,
        np.linspace((means - 5 * sds).min(), (means + 5 * sds).max(), num_externality_nodes),
        valuation_edges,
        name="M harm rises with a wiggle in the middle",
        metadata={"family": "harmful_middle", "valuation_mean": valuation_mean,
                  "valuation_sd": valuation_sd, "externality_mean": externality_mean,
                  "slope": slope, "externality_sd": externality_sd,
                  "wiggle_amplitude": wiggle_amplitude, "wiggle_width": wiggle_width,
                  "wiggle_center": wiggle_center, "wiggle_spread": wiggle_spread},
    )


def heavy_tail_distribution(log_valuation_sd=1.0, truncation_quantile=0.995, harm_slope=0.5,
                            externality_sd=0.4, num_externality_nodes=151,
                            num_valuation_cells=150):
    """Lognormal valuations (log v ~ N(0, s²)) truncated at a high quantile v̄,
    with e | v ~ N(−harm_slope √v, externality_sd²).

    Mimics the skew of engagement data; harm grows with valuation. Valuation
    cells are geometrically spaced, plus one cell [0, e^{−4s}].
    """
    from scipy.stats import norm

    maximum_valuation = float(np.exp(log_valuation_sd * norm.ppf(truncation_quantile)))
    smallest_edge = float(np.exp(-4 * log_valuation_sd))
    valuation_edges = np.concatenate(
        [[0.0], np.geomspace(smallest_edge, maximum_valuation, num_valuation_cells)])

    def density(e, v):
        safe_v = np.maximum(v, 1e-300)
        valuation_density = np.exp(-0.5 * (np.log(safe_v) / log_valuation_sd) ** 2) / (
            safe_v * log_valuation_sd * np.sqrt(2 * np.pi))
        return valuation_density * _normal_pdf(e, -harm_slope * np.sqrt(v), externality_sd)

    lowest_e = -harm_slope * np.sqrt(maximum_valuation) - 5 * externality_sd
    return gridded_density_distribution(
        density, np.linspace(lowest_e, 5 * externality_sd, num_externality_nodes),
        valuation_edges,
        name="L heavy-tailed valuations",
        metadata={"family": "heavy_tail", "log_valuation_sd": log_valuation_sd,
                  "truncation_quantile": truncation_quantile, "harm_slope": harm_slope,
                  "externality_sd": externality_sd},
    )


SYNTHETIC_FAMILIES = {
    **{letter: _mixture_family(letter) for letter in _MIXTURES},
    "example_n": example_n_distribution,
    "independent": independent_distribution,
    "harm_rises": harm_rises_distribution,
    "harmful_middle": harmful_middle_distribution,
    "heavy_tail": heavy_tail_distribution,
}


def synthetic_distribution(family, externality_scale=1.0, **options):
    """A synthetic family by name (see SYNTHETIC_FAMILIES), optionally with e scaled by ζ.

    options: grid options (num_externality_nodes, num_valuation_cells) and the
    family's own parameters.
    """
    if family not in SYNTHETIC_FAMILIES:
        raise KeyError(f"unknown family {family!r}; choose from {sorted(SYNTHETIC_FAMILIES)}")
    distribution = SYNTHETIC_FAMILIES[family](**options)
    distribution.metadata = {**distribution.metadata, "options": dict(options)}
    if externality_scale != 1.0:
        distribution = distribution.with_externality_scale(externality_scale)
    return distribution
