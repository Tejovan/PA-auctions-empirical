"""
draw_distributions.py

Previews of the synthetic families for the README (section 6): one WebP per
family in docs/distributions/, drawn at ζ = 1 on the default grid with the
type-plane geometry of the run figures.

Usage:
    python draw_distributions.py
"""

from pathlib import Path

from audited_auctions import SYNTHETIC_FAMILIES, figures
from audited_auctions.experiments.compute import plot_geometry, type_background
from audited_auctions.pipeline import prepare_distributions

FOLDER = Path(__file__).resolve().parent / "docs" / "distributions"


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for family in SYNTHETIC_FAMILIES:
        prepared = prepare_distributions(family)
        types = type_background(prepared, plot_geometry(prepared))
        path = figures.distribution_preview(types, family, FOLDER / f"{family}.webp")
        print(path.relative_to(FOLDER.parent.parent))


if __name__ == "__main__":
    main()
