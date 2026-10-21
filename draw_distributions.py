"""
draw_distributions.py

Previews of the synthetic families for the README (section 6): one WebP per
family in docs/distributions/, drawn at ζ = 1 on the default grid with the
type-plane geometry of the run figures, and a grid of harmful_middle over its
wiggle options (harmful_middle_options.webp).

Usage:
    python draw_distributions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from audited_auctions import SYNTHETIC_FAMILIES, figures
from audited_auctions.experiments.compute import plot_geometry, type_background
from audited_auctions.experiments.config import family_options
from audited_auctions.pipeline import prepare_distributions

FOLDER = Path(__file__).resolve().parent / "docs" / "distributions"

# harmful_middle option grid: rows vary wiggle_width, columns wiggle_spread; the last row
# varies wiggle_amplitude at the default width and spread.
WIGGLE_WIDTHS = [0.15, 0.26, 0.4]
WIGGLE_SPREADS = [0.0, 0.2, 0.38, 0.6]
WIGGLE_AMPLITUDES = [0.0, 0.25, 0.5, 0.8]


def harmful_middle_options_grid(path):
    defaults = family_options("harmful_middle")
    panels = [{"wiggle_width": w, "wiggle_spread": s} for w in WIGGLE_WIDTHS for s in WIGGLE_SPREADS]
    panels += [{"wiggle_amplitude": a} for a in WIGGLE_AMPLITUDES]
    with plt.rc_context(figures.STYLE):
        fig, axes = plt.subplots(len(WIGGLE_WIDTHS) + 1, len(WIGGLE_SPREADS),
                                 figsize=(2.6 * len(WIGGLE_SPREADS), 2.2 * (len(WIGGLE_WIDTHS) + 1)),
                                 sharex=True, sharey=True, constrained_layout=True)
        for ax, options in zip(axes.flat, panels):
            prepared = prepare_distributions("harmful_middle", family_options=options)
            types = type_background(prepared, plot_geometry(prepared))
            figures.draw_types(ax, types, alpha=0.9)
            ax.plot(types["e_grid"], -types["e_grid"], color=figures.MECHANISM_COLORS["first_best"],
                    linestyle="-.", linewidth=1.0)
            ax.set_xlim(-3.2, 2.4)
            ax.set_ylim(0, 2.5)
            is_default = all(defaults[key] == value for key, value in options.items())
            title = ", ".join(f"{key.removeprefix('wiggle_')} {value:g}"
                              for key, value in options.items())
            ax.set_title(title + (" (default)" if is_default else ""), loc="left", fontsize=8)
        for ax in axes.flat:
            ax.set_xlabel("externality e" if ax in axes[-1] else "")
            ax.set_ylabel("valuation v" if ax in axes[:, 0] else "")
        fig.savefig(path, dpi=figures.DISTRIBUTION_PREVIEW_DPI, pil_kwargs={"quality": 80})
        plt.close(fig)
    return path


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for family in SYNTHETIC_FAMILIES:
        prepared = prepare_distributions(family)
        types = type_background(prepared, plot_geometry(prepared))
        path = figures.distribution_preview(types, family, FOLDER / f"{family}.webp")
        print(path.relative_to(FOLDER.parent.parent))
    path = harmful_middle_options_grid(FOLDER / "harmful_middle_options.webp")
    print(path.relative_to(FOLDER.parent.parent))


if __name__ == "__main__":
    main()
