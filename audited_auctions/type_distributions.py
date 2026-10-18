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
    """

    externalities: np.ndarray
    valuation_lower: np.ndarray
    valuation_upper: np.ndarray
    probabilities: np.ndarray
    name: str = ""
    density_grid: Optional[dict] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict, repr=False)

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
        self.probabilities = self.probabilities[keep] / self.probabilities[keep].sum()
        point_cells = self.valuation_upper == self.valuation_lower
        if point_cells.all():
            self.kind = ATOMIC
        elif not point_cells.any():
            self.kind = PIECEWISE_UNIFORM
        else:
            raise ValueError("mixed distributions (atoms and intervals) are not supported")

    # Basic properties ----------------------------------------------------------

    @property
    def num_cells(self) -> int:
        return len(self.probabilities)

    @property
    def is_atomic(self) -> bool:
        return self.kind == ATOMIC

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
        valuations = lower + rng.random(cells.shape) * (upper - lower)
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
        return replace(
            self,
            externalities=self.externalities * externality_scale,
            name=name or f"{self.name} (ζ = {externality_scale:g})",
            density_grid=grid,
            metadata={**self.metadata, "externality_scale": externality_scale},
        )


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
    of the sweep scripts); the valuation is v = valuation_scale * v_score (the
    action cost of the sweep scripts).
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
        name=name, metadata=metadata or {},
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
                                  num_externality_nodes=301, num_valuation_cells=600,
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

    def build(num_externality_nodes=301, num_valuation_cells=600):
        return gaussian_mixture_distribution(
            means, [covariance] * len(means),
            num_externality_nodes=num_externality_nodes,
            num_valuation_cells=num_valuation_cells,
            name=title,
            metadata={"family": letter, "means": means, "covariance": covariance},
        )

    return build


def example_n_distribution(harm=0.6, num_externality_nodes=None, num_valuation_cells=600):
    """Example N of the core draft: v ~ U[0,1], e ∈ {0, −harm} w.p. ½ each, independent.

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
                             num_externality_nodes=301, num_valuation_cells=600):
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
                            num_externality_nodes=301, num_valuation_cells=600):
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


def harmful_middle_distribution(band=(0.5, 0.75), harmful_share_in_band=0.8,
                                harmful_share_outside=0.1, harmful_mean=-1.5, harmful_sd=0.2,
                                benign_sd=0.1, num_externality_nodes=301, num_valuation_cells=600):
    """v ~ U[0, 1]; harmful types concentrate in a middle valuation band.

    Given v, the externality is N(harmful_mean, harmful_sd²) with probability
    π(v) (harmful_share_in_band for v in the band, harmful_share_outside
    otherwise) and N(0, benign_sd²) otherwise. The expected welfare
    contribution v + E[e | v] drops at the band, so IVCG-PA irons (a continuous
    analogue of Example T).
    """
    def density(e, v):
        harmful_share = np.where((v >= band[0]) & (v < band[1]),
                                 harmful_share_in_band, harmful_share_outside)
        return (harmful_share * _normal_pdf(e, harmful_mean, harmful_sd)
                + (1 - harmful_share) * _normal_pdf(e, 0.0, benign_sd))

    return gridded_density_distribution(
        density,
        np.linspace(harmful_mean - 5 * harmful_sd, 5 * benign_sd, num_externality_nodes),
        np.linspace(0.0, 1.0, num_valuation_cells + 1),
        name="M harmful middle band",
        metadata={"family": "harmful_middle", "band": list(band),
                  "harmful_share_in_band": harmful_share_in_band,
                  "harmful_share_outside": harmful_share_outside,
                  "harmful_mean": harmful_mean, "harmful_sd": harmful_sd, "benign_sd": benign_sd},
    )


def heavy_tail_distribution(log_valuation_sd=1.0, truncation_quantile=0.995, harm_slope=0.5,
                            externality_sd=0.4, num_externality_nodes=301,
                            num_valuation_cells=600):
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
