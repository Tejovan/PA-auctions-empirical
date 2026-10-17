"""
type_distributions.py

Discrete representations of the joint distribution H of bidder types (e, v),
where e is the externality and v is the valuation.

Every distribution used by the package is a finite list of atoms
(externality_j, valuation_j) with probabilities probability_j. Two sources:

  * A finite population (for example the Community Notes data, or one of the
    synthetic CSV samples in data/samples/). Bidders are drawn i.i.d. from the
    population, so the atoms are the population rows with equal probability.

  * A continuous density (for example a Gaussian mixture). The density is
    discretized onto a fine (e, v) grid: each grid node becomes one atom whose
    probability is proportional to the density at that node.

Both the exact computations (exact_computation.py) and the Monte Carlo
simulation (auction_simulation.py) use the same atoms, so exact expectations
and simulated means agree up to Monte Carlo error. For a continuous density,
the discretization error is of the order of the grid spacing.

Bidders are exchangeable: all n bidders draw their types independently from
the same distribution.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TypeDistribution:
    """A discrete joint distribution of bidder types (e, v).

    Attributes
    ----------
    externalities  : (num_atoms,) externality e of each atom
    valuations     : (num_atoms,) valuation v of each atom
    probabilities  : (num_atoms,) probability of each atom; sums to one
    name           : short human-readable label used in figure titles
    density_grid   : for gridded densities only: dict with keys
                     "externality_grid", "valuation_grid", "density"
                     (density has shape (len(valuation_grid), len(externality_grid))).
                     Used only for plotting contours.
    """

    externalities: np.ndarray
    valuations: np.ndarray
    probabilities: np.ndarray
    name: str = ""
    density_grid: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        self.externalities = np.asarray(self.externalities, dtype=float)
        self.valuations = np.asarray(self.valuations, dtype=float)
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if not (
            self.externalities.shape
            == self.valuations.shape
            == self.probabilities.shape
        ):
            raise ValueError("externalities, valuations, probabilities must align")
        if np.any(self.probabilities < 0):
            raise ValueError("probabilities must be nonnegative")
        # Drop zero-probability atoms and renormalize.
        keep = self.probabilities > 0
        self.externalities = self.externalities[keep]
        self.valuations = self.valuations[keep]
        self.probabilities = self.probabilities[keep] / self.probabilities[keep].sum()

    @property
    def num_atoms(self) -> int:
        return len(self.probabilities)

    @property
    def welfare_contributions(self) -> np.ndarray:
        """Welfare contribution w = v + e of each atom."""
        return self.valuations + self.externalities

    def sample_type_indices(self, num_draws, num_bidders, rng) -> np.ndarray:
        """Draw atom indices for num_draws auctions of num_bidders i.i.d. bidders.

        Returns an integer array of shape (num_draws, num_bidders).
        """
        flat_indices = rng.choice(
            self.num_atoms, size=num_draws * num_bidders, p=self.probabilities
        )
        return flat_indices.reshape(num_draws, num_bidders)

    def sample_types(self, num_draws, num_bidders, rng):
        """Draw (externalities, valuations), each of shape (num_draws, num_bidders)."""
        indices = self.sample_type_indices(num_draws, num_bidders, rng)
        return self.externalities[indices], self.valuations[indices]

    def weighted_quantile(self, values, quantile) -> float:
        """Quantile of an atom-level quantity (e.g. valuations) under this distribution."""
        order = np.argsort(values)
        cumulative = np.cumsum(self.probabilities[order])
        position = np.searchsorted(cumulative, quantile, side="left")
        return float(values[order][min(position, len(order) - 1)])


# ─── Constructors ─────────────────────────────────────────────────────────────


def population_distribution(externalities, valuations, name="population"):
    """Uniform distribution over the rows of a finite population."""
    externalities = np.asarray(externalities, dtype=float)
    probabilities = np.full(len(externalities), 1.0 / len(externalities))
    return TypeDistribution(externalities, valuations, probabilities, name=name)


def population_distribution_from_csv(
    csv_path, externality_scale=1.0, valuation_scale=1.0, name=None
):
    """Population distribution from a CSV with columns e_score and v_score.

    The externality is e = externality_scale * e_score (the externality cost ζ in
    the sweep scripts) and the valuation is v = valuation_scale * v_score (the
    action cost in the sweep scripts).
    """
    data = pd.read_csv(csv_path)
    return population_distribution(
        externality_scale * data["e_score"].to_numpy(),
        valuation_scale * data["v_score"].to_numpy(),
        name=name or str(csv_path),
    )


def gridded_density_distribution(
    density_function, externality_grid, valuation_grid, name="density"
):
    """Discretize a joint density h(e, v) onto the nodes of a rectangular grid.

    density_function : callable (E, V) -> density, broadcasting over arrays
    externality_grid : (num_e,) increasing grid of externalities
    valuation_grid   : (num_v,) increasing grid of valuations

    Each node (e_i, v_j) becomes an atom with probability proportional to
    h(e_i, v_j). The grid must be uniform in each coordinate for these weights
    to be a midpoint-rule discretization.
    """
    externality_grid = np.asarray(externality_grid, dtype=float)
    valuation_grid = np.asarray(valuation_grid, dtype=float)
    externality_mesh, valuation_mesh = np.meshgrid(externality_grid, valuation_grid)
    density = density_function(externality_mesh, valuation_mesh)
    return TypeDistribution(
        externality_mesh.ravel(),
        valuation_mesh.ravel(),
        density.ravel(),
        name=name,
        density_grid={
            "externality_grid": externality_grid,
            "valuation_grid": valuation_grid,
            "density": density / density.sum(),
        },
    )


def gaussian_mixture_density(component_means, component_covariances, component_weights):
    """Return a callable density h(e, v) of a mixture of bivariate normals.

    Coordinates are ordered (e, v), as in the CSV samples.
    """
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
            quadratic_form = (
                inverse[0, 0] * de * de
                + 2 * inverse[0, 1] * de * dv
                + inverse[1, 1] * dv * dv
            )
            total += weight * np.exp(-0.5 * quadratic_form) / (
                2 * np.pi * np.sqrt(determinant)
            )
        return total

    return density


def gaussian_mixture_distribution(
    component_means,
    component_covariances,
    component_weights=None,
    minimum_valuation=None,
    num_externality_nodes=301,
    num_valuation_nodes=601,
    num_standard_deviations=5.0,
    name="gaussian mixture",
):
    """Gaussian mixture over (e, v), discretized onto a grid.

    The grid spans every component mean plus/minus num_standard_deviations
    standard deviations. If minimum_valuation is given, the density is
    truncated to v > minimum_valuation (and renormalized).
    """
    if component_weights is None:
        component_weights = np.ones(len(component_means))
    means = np.array(component_means, dtype=float)
    standard_deviations = np.array(
        [np.sqrt(np.diag(np.asarray(c, dtype=float))) for c in component_covariances]
    )
    lower = (means - num_standard_deviations * standard_deviations).min(axis=0)
    upper = (means + num_standard_deviations * standard_deviations).max(axis=0)
    if minimum_valuation is not None:
        lower[1] = max(lower[1], minimum_valuation)

    externality_grid = np.linspace(lower[0], upper[0], num_externality_nodes)
    valuation_grid = np.linspace(lower[1], upper[1], num_valuation_nodes)
    mixture_density = gaussian_mixture_density(
        component_means, component_covariances, component_weights
    )

    def density(externality, valuation):
        values = mixture_density(externality, valuation)
        if minimum_valuation is not None:
            values = np.where(valuation > minimum_valuation, values, 0.0)
        return values

    return gridded_density_distribution(
        density, externality_grid, valuation_grid, name=name
    )


# ─── Named synthetic distributions ────────────────────────────────────────────
# Parametric versions of the samples in data/samples/, with the parameters used
# in alternate_distributions.ipynb. Coordinates are (e, v); mixtures have equal
# component weights.

_WIDE = [[0.1, 0.0], [0.0, 0.1]]
_NARROW = [[0.05, 0.0], [0.0, 0.05]]

SYNTHETIC_DISTRIBUTION_PARAMETERS = {
    "a": dict(name="A single modal normal", means=[[0, 0]], covariance=_WIDE),
    "b": dict(name="B two modal (side by side)", means=[[-1, 0], [0, 0]], covariance=_NARROW),
    "c": dict(name="C two modal (diagonal .')", means=[[-1, 0], [0, 1]], covariance=_NARROW),
    "d": dict(name="D two modal (diagonal '.)", means=[[-1, 1], [0, 0]], covariance=_NARROW),
    "e": dict(name="E two modal (stacked)", means=[[0, 1], [0, 0]], covariance=_NARROW),
    "f": dict(
        name="F four modal",
        means=[[-1, 0], [-1, 1], [0, 1], [0, 0]],
        covariance=_NARROW,
    ),
    "h": dict(
        name="H shifted normal, v > 0",
        means=[[0, 2]],
        covariance=_WIDE,
        minimum_valuation=0.0,
    ),
}


def synthetic_distribution(letter, **grid_options):
    """Gridded parametric version of synthetic sample `letter` in {a,...,f,h}."""
    parameters = SYNTHETIC_DISTRIBUTION_PARAMETERS[letter]
    return gaussian_mixture_distribution(
        component_means=parameters["means"],
        component_covariances=[parameters["covariance"]] * len(parameters["means"]),
        minimum_valuation=parameters.get("minimum_valuation"),
        name=parameters["name"],
        **grid_options,
    )
