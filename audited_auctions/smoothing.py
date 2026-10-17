"""
smoothing.py

Kernel smoothing of an empirical population into a piecewise-uniform type
distribution (type_distributions.TypeDistribution), so that valuations have no
atoms, as the model assumes, and the expected welfare contribution
ŵ^τ(v) = v + E[e | v, participate] is a smooth conditional mean rather than the
externality of the single row with valuation v.

Method
------
1. Transform to coordinates where the data are roughly symmetric:
       u = asinh(e / s_e)   (s_e = median |e|; a signed logarithm for |e| >> s_e)
       y = log v            (valuations must be positive)
   Both transforms are scale-equivariant: multiplying e by ζ changes s_e by ζ
   and leaves u unchanged.
2. Gaussian kernel density estimate in (u, y) with bandwidth matrix
       H = (bandwidth_scale · N^(−1/6))² · Cov(u, y)        (Scott's rule, 2-D),
   computed on a grid by linear binning and FFT convolution.
   With preserve_variance (default), the data are first shrunk toward their
   mean by the factor (1 + h²)^(−1/2), h = bandwidth_scale · N^(−1/6), and
   the kernel covariance is shrunk by the same factor squared, so the smoothed
   (u, y) keep the sample mean and covariance (the variance-corrected kernel
   estimator; Jones 1991). Without it, smoothing adds the kernel variance to
   log v, which inflates the upper tail of v: on the Community Notes data the
   expected VCG welfare rises by about 10% at bandwidth_scale = 1.
3. Boundary at the valuation bound v̄ (y_max = log v̄): reflection. The data
   are mirrored at y_max before smoothing, so no mass is lost above v̄ and the
   density has zero slope at v̄. There is no boundary at v = 0 in log
   coordinates.
4. Cells: externality nodes e_j = s_e sinh(u_j) on a uniform u-grid (dense near
   e = 0, sparse in the tails); valuation cells [exp(y_i), exp(y_{i+1})] on a
   uniform y-grid, with the valuation uniform within a cell (the piecewise-
   uniform approximation of the smoothed density; cells are narrow in log
   scale). Cell probability ∝ density(u_j, y_mid_i) Δu Δy.

The bandwidth is a modeling choice: report results for bandwidth_scale = 1 and
check sensitivity with 0.5 and 2.
"""

import numpy as np
from scipy.signal import fftconvolve

from .type_distributions import TypeDistribution


def _linear_binning(points_u, points_y, u_nodes, y_nodes):
    """Counts on grid nodes, each point split linearly between its 4 neighbors."""
    du, dy = u_nodes[1] - u_nodes[0], y_nodes[1] - y_nodes[0]
    position_u = (points_u - u_nodes[0]) / du
    position_y = (points_y - y_nodes[0]) / dy
    counts = np.zeros((len(y_nodes), len(u_nodes)))
    base_u, base_y = np.floor(position_u).astype(int), np.floor(position_y).astype(int)
    weight_u, weight_y = position_u - base_u, position_y - base_y
    for shift_u, factor_u in ((0, 1 - weight_u), (1, weight_u)):
        for shift_y, factor_y in ((0, 1 - weight_y), (1, weight_y)):
            index_u, index_y = base_u + shift_u, base_y + shift_y
            inside = (index_u >= 0) & (index_u < len(u_nodes)) & (index_y >= 0) & (
                index_y < len(y_nodes))
            np.add.at(counts, (index_y[inside], index_u[inside]), (factor_u * factor_y)[inside])
    return counts


def kernel_density_distribution(externalities, valuations, maximum_valuation, bandwidth_scale=1.0,
                                num_externality_nodes=241, num_valuation_cells=400,
                                externality_unit=None, preserve_variance=True,
                                name="smoothed population", relative_mass_floor=1e-12):
    """Piecewise-uniform distribution from a kernel density estimate of (e, v).

    externalities, valuations : population rows (valuations in (0, v̄])
    maximum_valuation         : the valuation bound v̄ (reflecting boundary)
    bandwidth_scale           : multiplier on Scott's bandwidth
    externality_unit          : s_e of the asinh transform (default median |e|)
    preserve_variance         : shrink the data toward their mean before
                                smoothing so that the smoothed (u, y) have the
                                sample mean and covariance (see module notes)
    relative_mass_floor       : cells with probability below this fraction of
                                the largest cell are dropped (negligible mass)
    """
    externalities = np.asarray(externalities, dtype=float)
    valuations = np.asarray(valuations, dtype=float)
    if np.any(valuations <= 0):
        raise ValueError("kernel smoothing works in log v and needs positive valuations")
    if np.any(valuations > maximum_valuation * (1 + 1e-12)):
        raise ValueError("valuations above maximum_valuation; truncate first")
    externality_unit = externality_unit or float(np.median(np.abs(externalities))) or 1.0
    u = np.arcsinh(externalities / externality_unit)
    y = np.log(valuations)
    y_max = float(np.log(maximum_valuation))

    num_points = len(u)
    factor = bandwidth_scale * num_points ** (-1.0 / 6.0)
    bandwidth = np.cov(np.vstack([u, y])) * factor**2
    if preserve_variance:
        shrink = 1.0 / np.sqrt(1.0 + factor**2)
        u = u.mean() + shrink * (u - u.mean())
        y = y.mean() + shrink * (y - y.mean())
        bandwidth = bandwidth * shrink**2
    sd_u, sd_y = np.sqrt(np.diag(bandwidth))

    u_nodes = np.linspace(u.min() - 4 * sd_u, u.max() + 4 * sd_u, num_externality_nodes)
    y_edges = np.linspace(min(y.min(), y_max) - 4 * sd_y, y_max, num_valuation_cells + 1)
    y_mids = (y_edges[:-1] + y_edges[1:]) / 2
    du, dy = u_nodes[1] - u_nodes[0], y_edges[1] - y_edges[0]

    # Extend the grid by the kernel radius so that reflected points are binned.
    pad_u, pad_y = int(np.ceil(4 * sd_u / du)) + 1, int(np.ceil(4 * sd_y / dy)) + 1
    u_extended = u_nodes[0] + du * np.arange(-pad_u, num_externality_nodes + pad_u)
    y_extended = y_mids[0] + dy * np.arange(-pad_y, num_valuation_cells + pad_y)
    counts = _linear_binning(np.concatenate([u, u]), np.concatenate([y, 2 * y_max - y]),
                             u_extended, y_extended)

    offsets_u = du * np.arange(-pad_u, pad_u + 1)
    offsets_y = dy * np.arange(-pad_y, pad_y + 1)
    mesh_u, mesh_y = np.meshgrid(offsets_u, offsets_y)
    inverse = np.linalg.inv(bandwidth)
    quadratic = (inverse[0, 0] * mesh_u**2 + 2 * inverse[0, 1] * mesh_u * mesh_y
                 + inverse[1, 1] * mesh_y**2)
    kernel = np.exp(-0.5 * quadratic) / (2 * np.pi * np.sqrt(np.linalg.det(bandwidth)))

    density_extended = fftconvolve(counts, kernel, mode="same") / num_points
    density = np.clip(density_extended[pad_y:pad_y + num_valuation_cells,
                                       pad_u:pad_u + num_externality_nodes], 0.0, None)

    cell_mass = density * du * dy
    keep = cell_mass > relative_mass_floor * cell_mass.max()
    externality_nodes = externality_unit * np.sinh(u_nodes)
    valuation_edges = np.exp(y_edges)
    rows, columns = np.nonzero(keep)

    # Density in (e, v) coordinates for contour plots: f(u, y) |du/de| |dy/dv|.
    jacobian = 1.0 / (externality_unit * np.cosh(u_nodes)[None, :] * np.exp(y_mids)[:, None])
    return TypeDistribution(
        externality_nodes[columns], valuation_edges[rows], valuation_edges[rows + 1],
        cell_mass[rows, columns],
        name=name,
        density_grid={"externality_grid": externality_nodes, "valuation_grid": np.exp(y_mids),
                      "density": density * jacobian},
        metadata={"smoothing": "kernel", "bandwidth_scale": bandwidth_scale,
                  "preserve_variance": preserve_variance,
                  "bandwidth_matrix_asinh_log": bandwidth.tolist(),
                  "externality_unit": externality_unit, "maximum_valuation": maximum_valuation,
                  "num_points": num_points, "num_externality_nodes": num_externality_nodes,
                  "num_valuation_cells": num_valuation_cells},
    )
