"""
figures.py

Publication figures for one ThresholdEvaluation.

plot_threshold_and_penalties
    (a) the type distribution in the (e, v) plane, the threshold τ(e) with the
        abstention region shaded, and the first-best participation boundary
        v = -e of the optimal recipient audit;
    (b) the penalty functions r(e) inducing τ in VCG-PA and IVCG-PA, and the
        recipient-audit penalty -e;
    (c) expected welfare of VCG (no audit), VCG-PA, IVCG-PA and the first best
        (exact values, with simulated 95% intervals when available).

plot_welfare_distributions
    (a) smoothed densities (Gaussian KDE) of per-draw welfare;
    (b) smoothed densities of paired per-draw welfare gains over unaudited VCG
        on the same draws (VCG-PA − VCG, IVCG-PA − VCG, first best − VCG),
        which remove the common variation across mechanisms.

Colors: one fixed color per mechanism (MECHANISM_COLORS), validated for
color-vision deficiencies; every series also has a distinct line style and a
legend entry, so identity never relies on color alone.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from .threshold_evaluation import MECHANISM_LABELS, MECHANISMS

MECHANISM_COLORS = {
    "vcg": "#7a7974",  # neutral gray: baseline without audits
    "vcg_pa": "#2a78d6",  # blue
    "ivcg_pa": "#c0392b",  # red
    "first_best": "#1baf7a",  # aqua
}
MECHANISM_LINESTYLES = {"vcg": ":", "vcg_pa": "-", "ivcg_pa": "--", "first_best": "-."}
THRESHOLD_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

# Welfare gains are measured against unaudited VCG; each gain wears the color
# and line style of its audited mechanism.
DIFFERENCE_STYLES = {
    f"{mechanism} - vcg": dict(color=MECHANISM_COLORS[mechanism],
                               linestyle=MECHANISM_LINESTYLES[mechanism],
                               label=f"{label} − VCG")
    for mechanism, label in (("vcg_pa", "VCG-PA"), ("ivcg_pa", "IVCG-PA"),
                             ("first_best", "First best"))
}

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _compact_number(value, _position=None):
    """Tick label: 12k, 1.5M for large numbers; plain otherwise."""
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value / 1e6:g}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def _use_compact_ticks(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_number))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_number))


def _title_suffix(evaluation):
    return f"n = {evaluation.num_bidders}, k = {evaluation.num_goods}"


# ─── Panel (a): type distribution and threshold ──────────────────────────────


def _type_plot_limits(type_distribution):
    # Populations (e.g. engagement data) can be heavy-tailed: show the central 96%.
    tail_probability = 0.005 if type_distribution.density_grid is not None else 0.02
    quantile = type_distribution.weighted_quantile
    e_low = quantile(type_distribution.externalities, tail_probability)
    e_high = quantile(type_distribution.externalities, 1 - tail_probability)
    v_low = quantile(type_distribution.valuation_midpoints, tail_probability)
    v_high = quantile(type_distribution.valuation_midpoints, 1 - tail_probability)
    e_pad, v_pad = 0.05 * (e_high - e_low), 0.05 * (v_high - v_low)
    return (e_low - e_pad, e_high + e_pad), (v_low - v_pad, v_high + v_pad)


def draw_type_distribution(ax, type_distribution, e_limits, v_limits):
    """Shade the type density: filled contours for gridded synthetic densities,
    log-shaded density for kernel-smoothed populations, and a log-shaded 2-D
    histogram for atomic populations."""
    grid = type_distribution.density_grid
    if grid is not None and type_distribution.metadata.get("smoothing") == "kernel":
        # Smoothed empirical densities are skewed: logarithmic shading, as for populations.
        density = np.ma.masked_less_equal(grid["density"], grid["density"].max() * 1e-5)
        ax.pcolormesh(grid["externality_grid"], grid["valuation_grid"], density, cmap="Greys",
                      shading="auto", norm=LogNorm(vmin=density.min(), vmax=density.max()),
                      alpha=0.75, rasterized=True, zorder=1)
        return
    if grid is not None:
        density = grid["density"]
        levels = np.linspace(0, density.max(), 9)[1:]
        ax.contourf(grid["externality_grid"], grid["valuation_grid"], density,
                    levels=np.append(levels, density.max() * 1.01), cmap="Greys",
                    alpha=0.7, zorder=1)
        return
    histogram, e_edges, v_edges = np.histogram2d(
        type_distribution.externalities,
        type_distribution.valuation_midpoints,
        bins=[90, 90],
        range=[e_limits, v_limits],
        weights=type_distribution.probabilities,
    )
    # Logarithmic shading keeps sparse regions of skewed populations visible.
    histogram = np.ma.masked_less_equal(histogram.T, 0)
    ax.pcolormesh(e_edges, v_edges, histogram, cmap="Greys", shading="auto",
                  norm=LogNorm(vmin=histogram.min(), vmax=histogram.max()),
                  alpha=0.75, rasterized=True, zorder=1)


def draw_threshold_panel(ax, evaluation, type_distribution):
    e_limits, v_limits = _type_plot_limits(type_distribution)
    draw_type_distribution(ax, type_distribution, e_limits, v_limits)

    curves = evaluation.penalty_curves
    externality_grid = curves["externality_grid"]
    threshold = np.asarray(curves["threshold"], dtype=float)
    shown_threshold = np.clip(threshold, v_limits[0], v_limits[1])
    ax.fill_between(externality_grid, v_limits[0], shown_threshold, color=MECHANISM_COLORS["vcg_pa"],
                    alpha=0.10, linewidth=0, zorder=2, label="abstain (v < τ(e))")
    finite = np.isfinite(threshold)
    ax.plot(externality_grid[finite], threshold[finite], color=THRESHOLD_COLOR,
            linewidth=2.0, zorder=4, label="threshold τ(e)")
    if (~finite).any():
        ax.plot(externality_grid[~finite], np.full((~finite).sum(), v_limits[1]),
                color=THRESHOLD_COLOR, linewidth=4, solid_capstyle="butt", zorder=4,
                label="excluded (τ = v̄⁺)")
    ax.plot(externality_grid, -externality_grid, color=MECHANISM_COLORS["first_best"],
            linestyle="-.", linewidth=1.6, zorder=3, label="first best: v = −e")

    ax.set_xlim(e_limits)
    ax.set_ylim(v_limits)
    ax.set_xlabel("externality e")
    ax.set_ylabel("valuation v")
    ax.set_title("(a) Types and threshold τ(e)", loc="left")
    _use_compact_ticks(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)


# ─── Panel (b): penalty functions ─────────────────────────────────────────────


def draw_penalty_panel(ax, evaluation, e_limits=None):
    curves = evaluation.penalty_curves
    externality_grid = curves["externality_grid"]
    for mechanism, key in (("vcg_pa", "vcg_pa"), ("ivcg_pa", "ivcg_pa")):
        ax.plot(externality_grid, curves[key], color=MECHANISM_COLORS[mechanism],
                linestyle=MECHANISM_LINESTYLES[mechanism],
                label=f"r(e) for {MECHANISM_LABELS[mechanism]}")
    ax.plot(externality_grid, curves["recipient"], color=MECHANISM_COLORS["first_best"],
            linestyle="-.", linewidth=1.6, label="recipient penalty −e")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.7, zorder=0)
    ax.set_xlim(e_limits or (externality_grid[0], externality_grid[-1]))
    ax.set_xlabel("externality e")
    ax.set_ylabel("penalty")
    _use_compact_ticks(ax)
    ax.set_title("(b) Penalties inducing τ", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)


# ─── Panel (c): expected welfare ──────────────────────────────────────────────


def _format_number(value):
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def draw_welfare_panel(ax, evaluation):
    """Exact expected welfare per mechanism (simulated draws appear in the
    welfare-distribution figure; their confidence intervals can be wider than
    the differences between mechanisms, which the paired differences resolve)."""
    positions = np.arange(len(MECHANISMS))[::-1]
    exact = np.array([evaluation.exact[m]["welfare"] for m in MECHANISMS])
    span = exact.max() - exact.min()
    # Zoom on the differences when welfare levels are far from zero.
    left = exact.min() - 0.6 * span if exact.min() > 2 * span else min(exact.min(), 0.0)
    for position, mechanism, value in zip(positions, MECHANISMS, exact):
        ax.plot([left, value], [position, position], color=MECHANISM_COLORS[mechanism],
                linewidth=2.0, solid_capstyle="butt")
        ax.plot(value, position, "o", markersize=8, color=MECHANISM_COLORS[mechanism],
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.annotate(_format_number(value), (value, position), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
    ax.set_xlim(left, exact.max() + 0.3 * (exact.max() - left))
    ax.set_yticks(positions)
    ax.set_yticklabels([MECHANISM_LABELS[m] for m in MECHANISMS])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("expected welfare (exact)")
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_number))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_title("(c) Expected welfare", loc="left")


def plot_threshold_and_penalties(evaluation, type_distribution, output_path, title=None):
    """Three-panel figure: types and τ, penalties, expected welfare."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                                 gridspec_kw={"width_ratios": [1.1, 1.0, 0.95]})
        draw_threshold_panel(axes[0], evaluation, type_distribution)
        draw_penalty_panel(axes[1], evaluation, _type_plot_limits(type_distribution)[0])
        draw_welfare_panel(axes[2], evaluation)
        fig.suptitle(title or f"{evaluation.distribution_name}  ({_title_suffix(evaluation)})",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


# ─── Welfare distributions ────────────────────────────────────────────────────


def _draw_smoothed_density(ax, values, plot_range, color, linestyle, label, fill=True):
    """Gaussian KDE line (with light fill); a vertical line if the values are constant."""
    values = np.asarray(values, dtype=float)
    if np.ptp(values) <= 1e-12 * max(1.0, np.abs(values).max()):
        ax.axvline(values[0], color=color, linestyle=linestyle, label=f"{label} (constant)")
        return
    grid = np.linspace(*plot_range, 400)
    density = gaussian_kde(values)(grid)
    ax.plot(grid, density, color=color, linestyle=linestyle, label=label)
    if fill:
        ax.fill_between(grid, density, color=color, alpha=0.12, linewidth=0)
    ax.axvline(values.mean(), color=color, linestyle=linestyle, linewidth=0.9, alpha=0.8)


def _central_range(arrays, tail_probability=0.005):
    stacked = np.concatenate(arrays)
    low, high = np.quantile(stacked, [tail_probability, 1 - tail_probability])
    pad = 0.05 * (high - low) if high > low else 1.0
    return low - pad, high + pad


def plot_welfare_distributions(evaluation, output_path, include_vcg=True, title=None):
    """Smoothed welfare densities per mechanism and of paired differences."""
    if evaluation.simulated is None:
        raise ValueError("no simulated draws; rerun evaluate_threshold with num_simulated_draws > 0")
    mechanisms = [m for m in MECHANISMS if include_vcg or m != "vcg"]
    welfare = {m: evaluation.simulated[m]["welfare"] for m in mechanisms}
    differences = evaluation.paired_welfare_differences()

    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        welfare_range = _central_range(list(welfare.values()))
        for mechanism in mechanisms:
            _draw_smoothed_density(axes[0], welfare[mechanism], welfare_range,
                                   MECHANISM_COLORS[mechanism], MECHANISM_LINESTYLES[mechanism],
                                   f"{MECHANISM_LABELS[mechanism]} (mean {welfare[mechanism].mean():.4g})",
                                   fill=mechanism != "vcg")
        axes[0].set_xlim(welfare_range)
        axes[0].xaxis.set_major_formatter(FuncFormatter(_compact_number))
        axes[0].set_xlabel("welfare per auction")
        axes[0].set_ylabel("density")
        axes[0].set_title("(a) Welfare per auction", loc="left")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

        # Mechanisms often choose the same winners, so a difference is exactly
        # zero in many auctions. The panel shows the density of the nonzero
        # differences, scaled by their share, and reports the share of zeros.
        shown = list(DIFFERENCE_STYLES)
        zero_tolerance = 1e-9 * max(1.0, float(np.abs(welfare["first_best"]).max()))
        nonzero = {key: differences[key][np.abs(differences[key]) > zero_tolerance] for key in shown}
        nonempty = [nonzero[key] for key in shown if len(nonzero[key]) > 1]
        difference_range = _central_range(nonempty) if nonempty else (-1.0, 1.0)
        grid = np.linspace(*difference_range, 400)
        for key in shown:
            style = DIFFERENCE_STYLES[key]
            share_zero = 1 - len(nonzero[key]) / len(differences[key])
            label = (f"{style['label']}: mean {differences[key].mean():.3g}, "
                     f"zero in {100 * share_zero:.0f}%")
            if len(nonzero[key]) > 1 and np.ptp(nonzero[key]) > 0:
                density = (1 - share_zero) * gaussian_kde(nonzero[key])(grid)
                axes[1].plot(grid, density, color=style["color"], linestyle=style["linestyle"],
                             label=label)
                axes[1].fill_between(grid, density, color=style["color"], alpha=0.10, linewidth=0)
            else:
                axes[1].plot([], [], color=style["color"], linestyle=style["linestyle"], label=label)
        axes[1].axvline(0, color=TEXT_SECONDARY, linewidth=0.7, zorder=0)
        axes[1].set_xlim(difference_range)
        axes[1].xaxis.set_major_formatter(FuncFormatter(_compact_number))
        axes[1].set_xlabel("welfare gain over VCG on the same auction (nonzero part)")
        axes[1].set_ylabel("share × density")
        axes[1].set_title("(b) Paired welfare gains over VCG", loc="left")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)

        num_draws = len(next(iter(welfare.values())))
        fig.suptitle(title or f"{evaluation.distribution_name}  ({_title_suffix(evaluation)}, "
                     f"{num_draws:,} simulated auctions; vertical lines: means)",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    return output_path
