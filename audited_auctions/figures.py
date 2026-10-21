"""
figures.py

Figures of a run, drawn from the files that experiments/compute.py saves
(results.json, data.npz, types.npz); experiments/report.py calls them. Style
rules: CLAUDE_files/figure_style.md of the project repository (fixed,
color-vision-checked color and line style per mechanism; light,
semi-transparent type distribution in the background; legends outside busy
axes; one shared legend for panels with the same series).

Per combination:
  types_tau_penalties       types, τ, abstention region, first-best boundary v = −e,
                            penalties r(e), IVCG-PA pooled bids: (a) linear and
                            (b) logarithmic valuation axis
  expected_outcomes         (a) expected welfare split into bidder net utility,
                            revenue, penalties and externality part; (b) changes
                            over VCG with bootstrap intervals
  outcome_distributions     per-auction distributions (left) and paired
                            differences from VCG (right) of welfare, its
                            valuation and externality parts, revenue, penalties
  gain_scatter              per auction: captured vs achievable welfare gain;
                            revenue change vs welfare change
  ironing                   IVCG-PA design in quantile space (if ironing binds)
  marginal_contribution     maps of M^τ(e, v), penalty vs Pigouvian charge,
                            marginal contribution of threshold types
  search_history            tried thresholds; convergence of the search
  search_iterations         welfare and stopping diagnostics per iteration, one line per start
Across combinations:
  threshold_overlay         τ of several combinations on one type plot
  welfare_against_zeta      expected welfare and gains over VCG against ζ
  share_overview            share of the first-best gain of every combination
README gallery (draw_distributions.py):
  distribution_preview      type density of a synthetic family, WebP only

Paper mode (set_paper(True); python run.py --report RUN --paper): the same
functions export each panel as its own PDF in a paper/ folder next to the
figure, for multi-panel figures built in LaTeX. Panels are drawn at printed
size in cells of fixed size with fixed margins, so panels of one size align
side by side; there is no title and no note. Legends outside the axes and
colorbars go to their own PDFs; <figure>.caption.txt holds the caption draft,
and <figure>.tex a figure environment that includes the panels.

File size: rasterized artists (scatters, maps, the tried thresholds) are embedded
at 150 dpi in run figures and at 300 dpi in paper panels; every run figure also
gets a small WebP preview in thumbs/ next to it (thumbnail_path), shown in the
run's README files.
"""

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

MECHANISMS = ("vcg", "vcg_pa", "ivcg_pa", "first_best")
AUDITED = ("vcg_pa", "ivcg_pa", "first_best")
PARTICIPANT_AUDITS = ("vcg_pa", "ivcg_pa")
LABELS = {"vcg": "VCG (no audit)", "vcg_pa": "VCG-PA", "ivcg_pa": "IVCG-PA",
          "first_best": "First best (recipient audit)"}
SHORT = {"vcg": "VCG", "vcg_pa": "VCG-PA", "ivcg_pa": "IVCG-PA", "first_best": "First best"}
MECHANISM_COLORS = {"vcg": "#7a7974", "vcg_pa": "#2a78d6", "ivcg_pa": "#c0392b",
                    "first_best": "#19a170"}
MECHANISM_LINESTYLES = {"vcg": ":", "vcg_pa": "-", "ivcg_pa": "--", "first_best": "-."}
MECHANISM_HATCHES = {"vcg_pa": "", "ivcg_pa": "//", "first_best": ".."}
MECHANISM_MARKERS = {"vcg": "o", "vcg_pa": "s", "ivcg_pa": "D", "first_best": "^"}
# Components of welfare: W = bidder net utility + revenue + penalties + externality part.
COMPONENTS = ("bidder_utility", "revenue", "penalty_paid", "externality_welfare")
COMPONENT_COLORS = {"bidder_utility": "#bb7a03", "revenue": "#5e3c99",
                    "penalty_paid": "#be538e", "externality_welfare": "#3b3b3b"}
COMPONENT_HATCHES = {"bidder_utility": "", "revenue": "", "penalty_paid": "",
                     "externality_welfare": "xx"}
METRIC_LABELS = {"welfare": "welfare W", "valuation_welfare": "valuation part Σvx",
                 "externality_welfare": "externality part Σex", "revenue": "revenue Σp",
                 "penalty_paid": "penalties Σr", "bidder_utility": "bidder net utility Σu"}
# Categorical series that are not mechanisms (forms, optimizers, given thresholds).
# Non-mechanism series; mechanism hues are reserved (CLAUDE_files/figure_style.md).
# All eight colors (with MECHANISM_COLORS) pass check_palette.py: contrast >= 3:1 on white,
# CIEDE2000 >= 8.5 under simulated color-vision deficiency.
CATEGORY_COLORS = ("#bb7a03", "#5e3c99", "#be538e", "#3b3b3b")
CATEGORY_LINESTYLES = ("-", "--", "-.", ":", (0, (6, 2)), (0, (3, 1, 1, 1)), (0, (1, 1)))
THRESHOLD_COLOR = "#0b0b0b"
# Regions (figstyle.tex): abstention or exclusion light gray, pooled bids (a region that
# is a property of the auction, not a mechanism series) seriesB at 18%.
REGION_OUT = {"color": "#0b0b0b", "alpha": 0.10}
REGION_QTY = {"color": "#5e3c99", "alpha": 0.18}
# Reference lines (zero lines, 45° line): thin, dashed, secondary ink.
REFERENCE = {"color": "#52514e", "linewidth": 0.7, "linestyle": (0, (4, 2))}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"

STYLE = {
    "font.family": ["Latin Modern Roman", "Latin Modern Math", "DejaVu Sans"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "lines.linewidth": 1.8, "savefig.dpi": 150, "pdf.fonttype": 42,
}
def _register_latin_modern():
    """Latin Modern (text, and math for Greek letters) from the TeX installation, if present;
    glyphs missing from both fall back to DejaVu Sans."""
    import glob
    import logging

    from matplotlib import font_manager

    # The Latin Modern OpenType files have zero 'created' dates, which fontTools reports
    # on every PDF export.
    logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

    patterns = ("lm/lmroman10-*.otf", "lm-math/latinmodern-math.otf")  # math: Greek letters
    roots = ("/usr/local/texlive/*/texmf-dist/fonts/opentype/public/",
             "/usr/share/texlive/texmf-dist/fonts/opentype/public/")
    for path in [p for root in roots for pattern in patterns for p in glob.glob(root + pattern)]:
        font_manager.fontManager.addfont(path)


_register_latin_modern()
HALO = [path_effects.Stroke(linewidth=4.0, foreground="white", alpha=0.85), path_effects.Normal()]


# ─── Helpers ────────────────────────────────────────────────────────────────────


def compact(value, _position=None):
    """Tick and label format: 12.3k, 1.2M, or 4 significant digits."""
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value / 1e6:.3g}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:.3g}k"
    return f"{value:.4g}"


def exact_number(value):
    """Labels of exact values: all integer digits above 10⁴, else 4 significant digits."""
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def _compact_axes(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(compact))


def context_line(combination):
    """One-line description of a combination for figure titles."""
    parts = [combination["distribution"], f"ζ = {combination['zeta']:g}",
             f"n = {combination['num_bidders']}, k = {combination['num_goods']}"]
    if combination.get("objective"):
        parts += [f"objective {SHORT[combination['objective']]}",
                  f"form {combination['threshold_form']}",
                  f"optimizer {combination['optimizer']} ({combination['method']})"]
    elif combination.get("tau_label"):
        parts.append(f"τ = {combination['tau_label']}")
    return "  ·  ".join(parts)


# ─── Paper mode ─────────────────────────────────────────────────────────────────

# Cell of one panel at printed size (inches): half of a 6.5 in text width for panels of
# multi-panel figures, 0.6 of it for single panels; fixed margins around the axes box.
PAPER_CELL = {"multi": (3.1, 2.4), "single": (3.9, 2.9)}
PAPER_MARGINS = {"left": 0.62, "right": 0.12, "bottom": 0.50, "top": 0.12}
PAPER_DPI = 300  # rasterized artists in paper panels (run figures: STYLE, 150 dpi)
_PAPER = {"on": False}

# Previews of run figures in the README files: WebP, 80 pixels per inch, shown at
# THUMBNAIL_SCALE of their pixel width (sharp on high-density screens and when zoomed).
THUMBNAIL_DPI = 80
THUMBNAIL_SCALE = 0.75
# Previews of the synthetic families in the code README: WebP, about 70 pixels per inch.
DISTRIBUTION_PREVIEW_DPI = 70


def thumbnail_path(path):
    """Preview of the figure at path: thumbs/<stem>.webp in the figure's folder."""
    path = Path(path)
    return path.parent / "thumbs" / f"{path.stem}.webp"


def set_paper(on):
    """Switch paper mode on or off (module-wide; each worker process sets it)."""
    _PAPER["on"] = bool(on)


def _new_figure(width, height, rows=1, columns=1, **options):
    if not _PAPER["on"]:
        fig, axes = plt.subplots(rows, columns, figsize=(width, height), layout="constrained",
                                 squeeze=False, **options)
        return fig, axes
    cell_width, cell_height = PAPER_CELL["single" if rows * columns == 1 else "multi"]
    margin = PAPER_MARGINS
    fig, axes = plt.subplots(rows, columns, figsize=(columns * cell_width, rows * cell_height),
                             squeeze=False)
    figure_width, figure_height = columns * cell_width, rows * cell_height
    for row in range(rows):
        for column in range(columns):
            left = column * cell_width + margin["left"]
            bottom = (rows - 1 - row) * cell_height + margin["bottom"]
            axes[row, column].set_position([
                left / figure_width, bottom / figure_height,
                (cell_width - margin["left"] - margin["right"]) / figure_width,
                (cell_height - margin["bottom"] - margin["top"]) / figure_height])
    return fig, axes


def _colorbar(fig, mesh, axes, label):
    """Colorbar beside the axes; in paper mode, kept for its own PDF instead."""
    if _PAPER["on"]:
        fig.paper_colorbar = (mesh.norm, mesh.cmap, label)
        return
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.01)
    colorbar.set_label(label)


_LATEX_SYMBOLS = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon",
    "ζ": r"\zeta", "λ": r"\lambda", "μ": r"\mu", "π": r"\pi", "σ": r"\sigma", "τ": r"\tau",
    "ψ": r"\psi", "ω": r"\omega", "Σ": r"\Sigma", "Ψ": r"\Psi", "Δ": r"\Delta",
    "−": "-", "×": r"\times", "≥": r"\geq", "≤": r"\leq", "≈": r"\approx", "·": r"\cdot",
    "∫": r"\int", "⁺": "^{+}", "⁰": "^{0}", "°": r"^{\circ}", "|": "|"}


def latex_text(text):
    """Figure text (Unicode, matplotlib mathtext) as LaTeX text for captions and titles."""
    out = []
    for index, part in enumerate(str(text).split("$")):
        if index % 2:  # mathtext is LaTeX already
            out.append(f"${part}$")
            continue
        part = re.sub(r"(\w)\u0304", lambda m: f"$\\bar{{{m.group(1)}}}$", part)
        part = re.sub(r"([%&#_])", r"\\\1", part)
        part = "".join(f"${_LATEX_SYMBOLS[c]}$" if c in _LATEX_SYMBOLS and c != "−"
                       else ("-" if c == "−" else c) for c in part)
        out.append(part.replace("$$", ""))
    return "".join(out)


def _panel_title(ax):
    """Panel title without its letter, removed from the axes."""
    title = ax.get_title(loc="left") or ax.get_title()
    ax.set_title("", loc="left")
    ax.set_title("")
    return re.sub(r"^\([a-z]\)\s*", "", title)


def _save_only(fig, artist, file, bbox_inches):
    """Save the figure cropped to bbox_inches with only artist (an axes or a legend) drawn."""
    hidden = [a for a in fig.axes if a.get_visible()] + [a for a in fig.legends
                                                         if a.get_visible()]
    owner = artist if artist in fig.axes else None
    if owner is None and artist not in fig.legends:  # an axes legend: keep its axes, hide
        owner = artist.axes                             # everything else in that axes
        inner = [c for c in owner.get_children() if c is not artist and c.get_visible()]
    else:
        inner = []
    for item in hidden:
        if item is not owner and item is not artist:
            item.set_visible(False)
    for item in inner:
        item.set_visible(False)
    fig.savefig(file, bbox_inches=bbox_inches, dpi=PAPER_DPI)
    for item in hidden + inner:
        item.set_visible(True)


def _extent_inches(fig, artist, pad=0.04):
    renderer = fig.canvas.get_renderer()
    box = artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)


def _fit_legend(fig, legend, width):
    """Redraw a legend with as many columns as fit in width (inches); returns the legend."""
    handles, labels = legend.legend_handles, [text.get_text() for text in legend.texts]
    owner = None if legend in fig.legends else legend.axes
    columns = legend._ncols
    while True:
        options = {"handles": handles, "labels": labels, "ncol": columns,
                   "loc": "upper center", "bbox_to_anchor": (0.5, -0.1)}
        if owner is not None:
            new = owner.legend(**options, bbox_transform=fig.transFigure)
        else:
            legend.remove()
            new = fig.legend(**options, bbox_transform=fig.transFigure)
        fig.canvas.draw()
        if columns == 1 or _extent_inches(fig, new).width <= width:
            return new
        legend, columns = new, columns - 1


def _paper_export(fig, path, title, note):
    """Each panel, legend and colorbar of the figure as its own PDF in path.parent/paper/,
    with <stem>.caption.txt and <stem>.tex."""
    path = Path(path)
    folder = path.parent / "paper"
    folder.mkdir(exist_ok=True)
    stem = path.stem
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels, legend_axes = [], []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.axison:
            panels.append(ax)
        elif ax.get_legend() is not None:
            legend_axes.append(ax)
    single = len(panels) == 1
    cell_width, cell_height = PAPER_CELL["single" if single else "multi"]
    margin = PAPER_MARGINS
    files, titles, legends = [], [], []
    for index, ax in enumerate(panels):
        letter = chr(97 + index)
        titles.append(_panel_title(ax))
        legend = ax.get_legend()
        if legend is not None:
            inside = ax.bbox.contains(*legend.get_window_extent(renderer).p0) and \
                ax.bbox.contains(*legend.get_window_extent(renderer).p1)
            if not inside:
                legends.append((legend, f"{stem}_legend.pdf" if single
                                else f"{stem}_{letter}_legend.pdf"))
                legend.set_visible(False)
        box = ax.get_position()
        x0 = box.x0 * fig.get_figwidth() - margin["left"]
        y0 = box.y0 * fig.get_figheight() - margin["bottom"]
        file = f"{stem}.pdf" if single else f"{stem}_{letter}.pdf"
        _save_only(fig, ax, folder / file, Bbox.from_bounds(x0, y0, cell_width, cell_height))
        files.append(file)
    pending = getattr(fig, "pending_legend", None)
    shared = None
    if pending is not None:
        handles, columns = pending
        shared = fig.legend(handles=handles, loc="upper center", ncol=columns,
                            bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure)
    elif legend_axes:
        shared = legend_axes[0].get_legend()
    if shared is not None:
        legends.append((shared, f"{stem}_legend.pdf"))
    legend_files = []
    for legend, file in legends:
        legend.set_visible(True)
        legend = _fit_legend(fig, legend, 6.5 if file == f"{stem}_legend.pdf" and not single
                             else cell_width)
        _save_only(fig, legend, folder / file, _extent_inches(fig, legend))
        legend_files.append(file)
    colorbar = getattr(fig, "paper_colorbar", None)
    if colorbar is not None:
        norm, cmap, label = colorbar
        bar_figure = plt.figure(figsize=(cell_width, 0.62))
        bar_axes = bar_figure.add_axes([margin["left"] / cell_width, 0.62, 
                                        1 - (margin["left"] + margin["right"]) / cell_width,
                                        0.16])
        bar = bar_figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_axes,
                                  orientation="horizontal")
        bar.set_label(label)
        if hasattr(norm, "linthresh"):  # symmetric log: ±10^d, ±10^(d−2) and 0 only
            decade = np.floor(np.log10(norm.vmax))
            ticks = [10 ** decade, 10 ** (decade - 2)]
            ticks = sorted([0.0] + [t for t in ticks if t > norm.linthresh] +
                           [-t for t in ticks if t > norm.linthresh])
            bar.set_ticks(ticks)
        bar_figure.savefig(folder / f"{stem}_colorbar.pdf", bbox_inches="tight", pad_inches=0.02,
                           dpi=PAPER_DPI)
        plt.close(bar_figure)
        legend_files.append(f"{stem}_colorbar.pdf")
    plt.close(fig)
    _write_caption(folder, stem, title, note, titles, single)
    _write_snippet(folder, stem, files, titles, legend_files, note, single)
    return folder / files[0]


def _write_caption(folder, stem, title, note, titles, single):
    lines = [f"Settings: {title}", ""]
    if not single:
        lines += [f"({chr(97 + i)}) {t}" for i, t in enumerate(titles)] + [""]
    if note:
        lines += [textwrap.fill(note, 96)]
    (folder / f"{stem}.caption.txt").write_text("\n".join(lines) + "\n")


def _write_snippet(folder, stem, files, titles, legend_files, note, single):
    """A figure environment with the panels in rows of two, the legends below and a caption
    draft; graphics are referenced by file name (add the paper/ folder to \\graphicspath)."""
    label = re.sub(r"^\d+_", "", stem)
    lines = [f"% Drawn by run.py --report RUN --paper; caption draft from {stem}.caption.txt.",
             r"\begin{figure}[t]", r"\centering"]
    if single:
        lines.append(rf"\includegraphics{{{files[0]}}}")
    else:
        for index, (file, title) in enumerate(zip(files, titles)):
            letter = chr(97 + index)
            lines += [r"\begin{minipage}[t]{0.48\linewidth}\centering",
                      rf"{{\small ({letter}) {latex_text(title)}}}\\[2pt]",
                      rf"\includegraphics{{{file}}}" + (
                          rf"\\[2pt]\includegraphics{{{stem}_{letter}_legend.pdf}}"
                          if f"{stem}_{letter}_legend.pdf" in legend_files else ""),
                      r"\end{minipage}" + (r"\hfill" if index % 2 == 0 and index + 1 < len(files)
                                           else r"\par\medskip")]
    for file in legend_files:
        if not re.search(r"_[a-z]_legend\.pdf$", file):
            lines.append(rf"\includegraphics{{{file}}}\par")
    if single:
        caption = latex_text(titles[0]) + "." if titles[0] else ""
    else:
        caption = " ".join(f"({chr(97 + i)}) {latex_text(t)}." for i, t in enumerate(titles))
    if note:
        caption = (caption + " " + latex_text(note)).strip()
    lines += [rf"\caption{{{caption}}}", rf"\label{{fig:{label}}}", r"\end{figure}"]
    (folder / f"{stem}.tex").write_text("\n".join(lines) + "\n")


def _finish(fig, path, title, note=None):
    """Lay out the axes, then put the shared legend below them and the note below that.

    The axes are laid out by the constrained layout engine first; the layout is then
    frozen, so the legend and the note can be stacked under the lowest axes element.
    In paper mode, export the panels instead (_paper_export).
    """
    if _PAPER["on"]:
        return _paper_export(fig, path, title, note)
    fig.suptitle(_wrap_title(title, fig.get_figwidth()), x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    height = fig.bbox.height
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes if ax.get_visible()) / height
    pending = getattr(fig, "pending_legend", None)
    if pending is not None:
        handles, columns = pending
        legend = fig.legend(handles=handles, loc="upper center", ncol=columns,
                            bbox_to_anchor=(0.5, bottom - 0.004), bbox_transform=fig.transFigure)
        fig.canvas.draw()
        bottom = legend.get_window_extent(renderer).y0 / height
    if note:
        width = int(13 * fig.get_figwidth())  # characters per line at 7 pt
        fig.text(0.01, bottom - 0.008, textwrap.fill(note, width), ha="left", va="top",
                 fontsize=7, color=INK_SECONDARY, transform=fig.transFigure)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    thumbnail = thumbnail_path(path)
    thumbnail.parent.mkdir(exist_ok=True)
    fig.savefig(thumbnail, bbox_inches="tight", pad_inches=0.08, dpi=THUMBNAIL_DPI,
                pil_kwargs={"quality": 80})
    plt.close(fig)
    return path


def _wrap_title(title, width_inches, characters_per_inch=13.5):
    """Break a context line between its "  ·  " parts so that no line is wider than the
    figure (a wider title would widen the saved PDF with empty space)."""
    limit = int(characters_per_inch * width_inches)
    lines = []
    for part in title.split("  ·  "):
        if lines and len(lines[-1]) + 5 + len(part) <= limit:
            lines[-1] += "  ·  " + part
        else:
            lines.append(part)
    return "\n".join(lines)


def _shared_legend(fig, handles, columns=4):
    """One legend for all panels, placed below the axes by _finish."""
    fig.pending_legend = (handles, columns)


def mechanism_line(mechanism, **options):
    """Legend handle in the mechanism's color and line style (options override them)."""
    style = {"color": MECHANISM_COLORS[mechanism], "linestyle": MECHANISM_LINESTYLES[mechanism],
             "label": LABELS[mechanism], **options}
    return Line2D([], [], **style)


def _interval_text(interval):
    return "" if not interval else f" [{compact(interval[0])}, {compact(interval[1])}]"


def _percent(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.1f}%"


# ─── Type plane ─────────────────────────────────────────────────────────────────


def draw_types(ax, types, alpha=0.72):
    """Type distribution in semi-transparent gray (lowest layer)."""
    types = default_view(types)
    e_axis, v_axis = types["background_e"], types["background_v"]
    density = np.asarray(types["background_density"], dtype=float)
    if str(types["shading"]) == "log":
        masked = np.ma.masked_less_equal(density, density.max() * 1e-5)
        if masked.count():
            ax.pcolormesh(e_axis, v_axis, masked, cmap="Greys", shading="auto", alpha=alpha,
                          norm=LogNorm(vmin=masked.min(), vmax=masked.max()), rasterized=True,
                          zorder=0)
    else:
        levels = np.linspace(0, density.max(), 9)[1:]
        if density.max() > 0:
            ax.contourf(e_axis, v_axis, density, levels=np.append(levels, density.max() * 1.01),
                        cmap="Greys", alpha=alpha, zorder=0)
    type_plane_axes(ax, types)


def _valuation_quantile(types, level):
    """Quantile of the valuations of the type background."""
    v_axis = np.asarray(types["background_v"], dtype=float)
    weights = np.asarray(types["background_density"], dtype=float).sum(axis=1)
    if weights.sum() <= 0:
        return float(types["v_limits"][1])
    cumulative = np.cumsum(weights) / weights.sum()
    return float(v_axis[min(np.searchsorted(cumulative, level), len(v_axis) - 1)])


def _log_lower_limit(types):
    """Lower end of a logarithmic valuation axis: stored, or the 0.5% quantile of the
    positive valuations of the type background."""
    stored = float(types.get("v_log_lower", 0.0) or 0.0)
    if stored > 0:
        return stored
    v_axis = np.asarray(types["background_v"], dtype=float)
    weights = np.asarray(types["background_density"], dtype=float).sum(axis=1)
    positive = v_axis > 0
    if not positive.any() or weights[positive].sum() <= 0:
        return 1e-3 * float(types["v_limits"][1])
    cumulative = np.cumsum(weights[positive]) / weights[positive].sum()
    return float(v_axis[positive][min(np.searchsorted(cumulative, 0.005), positive.sum() - 1)])


def type_view(types, scale):
    """types with the valuation axis of one view: "linear" or "log" (plain logarithmic,
    from a small positive valuation to the top of the support)."""
    view = dict(types)
    view["scale"] = scale
    if scale == "log":
        view["v_limits"] = np.array([_log_lower_limit(types), float(types["v_limits"][1])])
    return view


def default_view(types):
    """The view of single-panel type plots: logarithmic for heavy-tailed valuations."""
    if "scale" in types:
        return types
    heavy = bool(types.get("v_log", float(types.get("v_linear_below", 0.0)) > 0))
    return type_view(types, "log" if heavy else "linear")


def type_plane_axes(ax, types):
    """Limits, labels and scale of a type-plane plot (plain logarithmic valuation axis in
    the log view)."""
    types = default_view(types)
    ax.set_xlim(types["e_limits"])
    ax.set_ylim(types["v_limits"])
    ax.set_xlabel("externality e")
    if types["scale"] == "log":
        ax.set_yscale("log")
        ax.set_ylabel("valuation v (log scale)")
    else:
        ax.set_ylabel("valuation v")
    _compact_axes(ax, y=types["scale"] != "log")


def draw_threshold(ax, e_grid, tau, types, color=THRESHOLD_COLOR, linestyle="-", label=None,
                   linewidth=2.0, zorder=5, halo=True):
    """τ(e) on the type plot; excluded externalities (τ = v̄⁺) as a thick bar at the top."""
    tau = np.asarray(tau, dtype=float)
    v_limits = default_view(types)["v_limits"]
    finite = np.isfinite(tau)
    shown = np.where(finite, np.clip(tau, v_limits[0], v_limits[1]), np.nan)
    line, = ax.plot(e_grid, shown, color=color, linestyle=linestyle, linewidth=linewidth,
                    zorder=zorder, label=label)
    if halo:
        line.set_path_effects(HALO)
    if (~finite).any():
        excluded = np.where(~finite, v_limits[1], np.nan)
        ax.plot(e_grid, excluded, color=color, linewidth=5, solid_capstyle="butt",
                zorder=zorder, alpha=0.8)
    return line


def _pooled_bid_intervals(data):
    """Bid intervals pooled by IVCG-PA ironing (constant score, varying ŵ^τ)."""
    if "iron_score_start" not in data:
        return []
    start, end = data["iron_score_start"], data["iron_score_end"]
    welfare_start, welfare_end = data["iron_welfare_start"], data["iron_welfare_end"]
    scale = max(1.0, float(np.nanmax(np.abs(welfare_end))))
    pooled = (np.abs(end - start) <= 1e-12 * scale) & (
        (np.abs(welfare_start - start) > 1e-9 * scale) | (np.abs(welfare_end - end) > 1e-9 * scale))
    intervals = []
    for index in np.flatnonzero(pooled):
        low, high = float(data["iron_bid_start"][index]), float(data["iron_bid_end"][index])
        if intervals and low <= intervals[-1][1] + 1e-12 * scale:
            intervals[-1][1] = max(intervals[-1][1], high)
        else:
            intervals.append([low, high])
    return [interval for interval in intervals if interval[1] > interval[0]]


def _visible_pooled_intervals(data, types, minimum_share=0.002):
    """Pooled bid intervals wider than minimum_share of the valuation axis of the view
    (measured in log10 v on a logarithmic axis); not merged, so that the many small
    pools at the thresholds of discrete externality nodes stay separate
    (5_ironing.pdf shows all of them)."""
    low_limit, high_limit = types["v_limits"]
    log = types.get("scale") == "log"
    def position(v):
        return np.log10(max(v, low_limit)) if log else v
    span = position(high_limit) - position(low_limit)
    intervals = []
    for low, high in _pooled_bid_intervals(data):
        if high <= low_limit or low >= high_limit:
            continue
        if position(min(high, high_limit)) - position(max(low, low_limit)) >= minimum_share * span:
            intervals.append((max(low, low_limit), min(high, high_limit)))
    return intervals


# ─── Per combination ────────────────────────────────────────────────────────────


def _types_panel(ax, data, types):
    """Types, abstention region, τ(e), the line v = −e and the penalties r(e) in one axes."""
    draw_types(ax, types)
    v_low, v_high = types["v_limits"]
    e_grid, tau = data["e_grid"], data["tau_grid"]
    shown = np.where(np.isfinite(tau), np.clip(tau, v_low, v_high), v_high)
    ax.fill_between(e_grid, v_low, shown, linewidth=0, zorder=-1, **REGION_OUT)
    pooled = _visible_pooled_intervals(data, types)
    for low, high in pooled:
        ax.axhspan(low, high, linewidth=0, **REGION_QTY, zorder=1)
    dense_e = np.linspace(*types["e_limits"], 800)
    boundary = np.where(-dense_e > 0, -dense_e, np.nan)
    ax.plot(dense_e, boundary, color=MECHANISM_COLORS["first_best"],
            linestyle=MECHANISM_LINESTYLES["first_best"], linewidth=1.4, zorder=3)
    for mechanism in PARTICIPANT_AUDITS:
        penalty = np.asarray(data[f"penalty_{mechanism}"], dtype=float)
        if types["scale"] == "log":
            penalty = np.where(penalty >= v_low, penalty, np.nan)
        ax.plot(e_grid, np.minimum(penalty, v_high), color=MECHANISM_COLORS[mechanism],
                linestyle=MECHANISM_LINESTYLES[mechanism], linewidth=1.2, zorder=4)
    draw_threshold(ax, e_grid, tau, types)
    ax.set_ylabel("valuation v, penalty r(e)" + (" (log scale)" if types["scale"] == "log"
                                                  else ""))
    return bool(pooled)


def types_tau_penalties(results, data, types, path):
    combination = results["combination"]
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 3.0, 1, 2)
        linear, logarithmic = type_view(types, "linear"), type_view(types, "log")
        heavy = default_view(types)["scale"] == "log"
        if heavy:  # heavy tail: the linear view stops at the 97.5% quantile of valuations
            upper = float(types.get("v_linear_upper", 0.0) or _valuation_quantile(types, 0.975))
            linear["v_limits"] = np.array([linear["v_limits"][0], upper])
        pooled = _types_panel(axes[0, 0], data, linear)
        axes[0, 0].set_title("(a) Types, threshold τ(e), penalties r(e)", loc="left")
        pooled = _types_panel(axes[0, 1], data, logarithmic) or pooled
        axes[0, 1].set_title("(b) The same, logarithmic valuation axis", loc="left")

        handles = [Line2D([], [], color=THRESHOLD_COLOR, linewidth=2, label="threshold τ(e)"),
                   Patch(**REGION_OUT, label="abstain: v < τ(e)"),
                   Line2D([], [], color=THRESHOLD_COLOR, linewidth=5,
                          label=r"excluded e ($\tau = \bar{v}^{+}$)"),
                   Line2D([], [], color=MECHANISM_COLORS["first_best"], linestyle="-.",
                          label="v = −e: first-best boundary and recipient penalty −e"),
                   mechanism_line("vcg_pa", label="penalty r(e) in VCG-PA"),
                   mechanism_line("ivcg_pa", label="penalty r(e) in IVCG-PA")]
        if pooled:
            handles.append(Patch(**REGION_QTY, label="bids pooled by IVCG-PA"))
        _shared_legend(fig, handles, columns=3)
        note = (f"{str(types['description'])}; penalties on the evaluation distribution, "
                "in valuation units (r(e) ≤ τ(e)); zero penalties are not drawn on the "
                "logarithmic axis" + ("; (a) shows valuations up to their 97.5% quantile."
                                      if heavy else "."))
        return _finish(fig, path, context_line(combination), note)


def _bar_stack(ax, position, components, width=0.6):
    positive, negative = 0.0, 0.0
    for component in COMPONENTS:
        value = components[component]
        bottom = positive if value >= 0 else negative
        ax.bar(position, value, width, bottom=bottom, color=COMPONENT_COLORS[component],
               hatch=COMPONENT_HATCHES[component], edgecolor="white", linewidth=1.0, zorder=2)
        if value >= 0:
            positive += value
        else:
            negative += value


def expected_outcomes(results, path):
    combination, exact = results["combination"], results["exact"]
    bootstrap = results.get("bootstrap")
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 3.2, 1, 2, gridspec_kw={"width_ratios": [0.85, 1.15]})
        ax = axes[0, 0]
        positions = np.arange(len(MECHANISMS))
        for position, mechanism in zip(positions, MECHANISMS):
            _bar_stack(ax, position, exact[mechanism])
            welfare = exact[mechanism]["welfare"]
            ax.plot(position, welfare, marker="D", color=INK, markersize=6, zorder=4,
                    markeredgecolor="white")
            interval = (bootstrap or {}).get("level_intervals", {}).get("welfare", {}).get(
                mechanism)
            if interval:
                ax.errorbar(position, welfare, yerr=[[max(welfare - interval[0], 0)],
                                                     [max(interval[1] - welfare, 0)]],
                            color=INK, capsize=3, linewidth=1, zorder=4)
            share = results["share_of_first_best_gain"].get(mechanism)
            # value and share of the first-best gain only (the note names them)
            text = compact(welfare) + (f"\n{_percent(share)}"
                                       if mechanism in ("vcg_pa", "ivcg_pa") else "")
            stack_top = sum(max(exact[mechanism][c], 0.0) for c in COMPONENTS)
            label_at = max(stack_top, interval[1] if interval else welfare)
            ax.annotate(text, (position, label_at), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK, zorder=5)
        ax.axhline(0, **REFERENCE, zorder=1)
        top = max(sum(max(exact[m][c], 0.0) for c in COMPONENTS) for m in MECHANISMS)
        ax.set_ylim(top=1.3 * top)
        ax.set_xticks(positions, ["VCG", "VCG-PA", "IVCG-PA", "First\nbest"])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("expected value per auction")
        _compact_axes(ax, x=False)
        ax.set_title("(a) Welfare W and its components", loc="left")
        ax.legend(handles=[Patch(facecolor=COMPONENT_COLORS[c], hatch=COMPONENT_HATCHES[c],
                                 edgecolor="white", label=METRIC_LABELS[c]) for c in COMPONENTS]
                  + [Line2D([], [], marker="D", color=INK, linestyle="none",
                            label="welfare W (sum)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

        ax = axes[0, 1]
        metrics = ("welfare", "valuation_welfare", "externality_welfare", "revenue",
                   "penalty_paid", "bidder_utility")
        width = 0.26
        for offset, mechanism in zip((-width, 0.0, width), AUDITED):
            values = [results["change_over_vcg"][metric][mechanism] for metric in metrics]
            positions = np.arange(len(metrics)) + offset
            ax.bar(positions, values, width, color=MECHANISM_COLORS[mechanism],
                   hatch=MECHANISM_HATCHES[mechanism], edgecolor="white", linewidth=1.0,
                   label=SHORT[mechanism], zorder=2)
            if bootstrap:
                intervals = [bootstrap["change_intervals"].get(metric, {}).get(mechanism)
                             for metric in metrics]
                for position, value, interval in zip(positions, values, intervals):
                    if interval:
                        ax.errorbar(position, value, yerr=[[max(value - interval[0], 0)],
                                                           [max(interval[1] - value, 0)]],
                                    color=INK, capsize=2, linewidth=0.9, zorder=3)
        ax.axhline(0, **REFERENCE, zorder=1)
        ax.set_xticks(np.arange(len(metrics)), ["W", "Σvx", "Σex", "Σp", "Σr", "Σu"])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("change over VCG per auction")
        _compact_axes(ax, x=False)
        ax.set_title("(b) Changes over VCG", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
        if bootstrap:
            note = (f"Exact expected values on the evaluation distribution (test half). Error "
                    f"bars: {100 * bootstrap['confidence_level']:g}% percentile intervals over "
                    f"{bootstrap['num_resamples']} bootstrap resamples of the test rows.")
        else:
            note = "Exact expected values; the distribution is known, so there is no sampling error."
        note += " W = Σu + Σp + Σr + Σex and Σvx = Σu + Σp + Σr."
        note += (" Numbers above the bars: expected welfare W and, for VCG-PA and IVCG-PA, "
                 "the share of the first-best gain over VCG. Σp: revenue; Σr: penalties; "
                 "Σu: bidder net utility; Σvx: valuation part; Σex: externality part.")
        return _finish(fig, path, context_line(combination), note)


def _density(ax, values, grid, color, linestyle, scale=1.0, fill=True):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.ptp(values) <= 1e-12 * max(1.0, np.abs(values).max()):
        if len(values):
            ax.axvline(values[0], color=color, linestyle=linestyle, linewidth=1.6, alpha=0.9)
        return
    density = scale * gaussian_kde(values)(grid)
    ax.plot(grid, density, color=color, linestyle=linestyle, alpha=0.9, zorder=3)
    if fill:
        ax.fill_between(grid, density, color=color, alpha=0.10, linewidth=0, zorder=2)


def _range(arrays, tail=0.005):
    arrays = [np.asarray(a, dtype=float) for a in arrays if len(a)]
    if not arrays:
        return -1.0, 1.0
    stacked = np.concatenate(arrays)
    low, high = np.quantile(stacked, [tail, 1 - tail])
    pad = 0.06 * (high - low) if high > low else max(abs(low), 1.0) * 0.1
    return low - pad, high + pad


def _mean_strip(ax, means, intervals, mechanisms):
    """Means (and intervals) as markers in a strip at the top of the axes."""
    transform = ax.get_xaxis_transform()
    for row, mechanism in enumerate(mechanisms):
        y = 0.97 - 0.055 * row
        interval = intervals.get(mechanism)
        if interval:
            ax.plot(interval, [y, y], color=MECHANISM_COLORS[mechanism], linewidth=1.4,
                    transform=transform, zorder=5, solid_capstyle="butt")
        ax.plot(means[mechanism], y, marker=MECHANISM_MARKERS[mechanism],
                color=MECHANISM_COLORS[mechanism], markersize=4.5, transform=transform,
                zorder=6, markeredgecolor="white", markeredgewidth=0.6)
        ax.axvline(means[mechanism], color=MECHANISM_COLORS[mechanism],
                   linestyle=MECHANISM_LINESTYLES[mechanism], linewidth=1.0, alpha=0.8,
                   zorder=4)


def _text_box(ax, lines):
    ax.text(0.99, 0.70, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=INK, zorder=7,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85,
                  "edgecolor": "none"})


def outcome_distributions(results, data, path):
    combination, exact = results["combination"], results["exact"]
    bootstrap = results.get("bootstrap") or {}
    metrics = ("welfare", "valuation_welfare", "externality_welfare", "revenue", "penalty_paid")
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 2.2 * len(metrics), len(metrics), 2)
        for row, metric in enumerate(metrics):
            mechanisms = [m for m in MECHANISMS if not (metric == "penalty_paid" and m == "vcg")]
            draws = {m: data[f"sim_{m}_{metric}"] for m in mechanisms}
            ax = axes[row, 0]
            grid = np.linspace(*_range(draws.values()), 200)
            for mechanism in mechanisms:
                _density(ax, draws[mechanism], grid, MECHANISM_COLORS[mechanism],
                         MECHANISM_LINESTYLES[mechanism], fill=mechanism != "vcg")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.45)
            levels = bootstrap.get("level_intervals", {}).get(metric, {})
            means = {m: exact[m][metric] for m in mechanisms}
            _mean_strip(ax, means, levels, mechanisms)
            _text_box(ax, [f"{SHORT[m]}: {exact_number(means[m])}{_interval_text(levels.get(m))}"
                           for m in mechanisms])
            ax.set_xlabel(f"{METRIC_LABELS[metric]} per auction")
            ax.set_ylabel("density")
            _compact_axes(ax)
            ax.set_title(f"({chr(97 + 2 * row)}) {METRIC_LABELS[metric]}", loc="left")

            ax = axes[row, 1]
            if metric == "penalty_paid":
                ax.axis("off")
                ax.text(0.5, 0.5, "VCG has no penalties: the left panel\nshows the paired "
                        "differences from VCG.", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=INK_SECONDARY)
                continue
            differences = {m: data[f"sim_{m}_{metric}"] - data[f"sim_vcg_{metric}"]
                           for m in AUDITED}
            tolerance = 1e-9 * max(1.0, float(np.abs(data[f"sim_vcg_{metric}"]).max()))
            nonzero = {m: d[np.abs(d) > tolerance] for m, d in differences.items()}
            grid = np.linspace(*_range([v for v in nonzero.values() if len(v)]), 200)
            lines = []
            for mechanism in AUDITED:
                share_zero = 1 - len(nonzero[mechanism]) / len(differences[mechanism])
                _density(ax, nonzero[mechanism], grid, MECHANISM_COLORS[mechanism],
                         MECHANISM_LINESTYLES[mechanism], scale=1 - share_zero)
                change = results["change_over_vcg"][metric][mechanism]
                interval = bootstrap.get("change_intervals", {}).get(metric, {}).get(mechanism)
                lines.append(f"{SHORT[mechanism]} − VCG: {compact(change)}"
                             f"{_interval_text(interval)}; 0 in {100 * share_zero:.0f}%")
            ax.axvline(0, **REFERENCE, zorder=1)
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1e-300) * 1.45)
            changes = {m: results["change_over_vcg"][metric][m] for m in AUDITED}
            _mean_strip(ax, changes, bootstrap.get("change_intervals", {}).get(metric, {}),
                        AUDITED)
            _text_box(ax, lines)
            ax.set_xlabel(f"change of {METRIC_LABELS[metric]} over VCG in the same auction")
            ax.set_ylabel("share × density (nonzero part)")
            _compact_axes(ax)
            ax.set_title(f"({chr(98 + 2 * row)}) paired difference from VCG", loc="left")
        _shared_legend(fig, [mechanism_line(m) for m in MECHANISMS], columns=4)
        draws_count = len(data["sim_vcg_welfare"])
        note = (f"Densities of {draws_count:,} simulated auctions (the same auctions for all "
                f"mechanisms). Markers and vertical lines: exact expected values")
        note += (f"; bars: {100 * bootstrap['confidence_level']:g}% bootstrap intervals."
                 if bootstrap else " (no sampling error).")
        note += " Right column: density of the nonzero differences, scaled by their share."
        return _finish(fig, path, context_line(combination), note)


def gain_scatter(results, data, path):
    combination = results["combination"]
    first_best_gain = data["sim_first_best_welfare"] - data["sim_vcg_welfare"]
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 5.6, 2, 2)
        for row, mechanism in enumerate(PARTICIPANT_AUDITS):
            gain = data[f"sim_{mechanism}_welfare"] - data["sim_vcg_welfare"]
            revenue = data[f"sim_{mechanism}_revenue"] - data["sim_vcg_revenue"]
            tolerance = 1e-9 * max(1.0, float(np.abs(data["sim_vcg_welfare"]).max()))
            for column, (x, y, x_label, y_label, title) in enumerate((
                    (first_best_gain, gain, "first best − VCG (achievable gain)",
                     f"{SHORT[mechanism]} − VCG (captured gain)", "captured vs achievable gain"),
                    (gain, revenue, f"welfare: {SHORT[mechanism]} − VCG",
                     f"revenue: {SHORT[mechanism]} − VCG", "revenue change vs welfare change"))):
                ax = axes[row, column]
                zero = (np.abs(x) <= tolerance) & (np.abs(y) <= tolerance)
                ax.scatter(x[~zero], y[~zero], s=6, color=MECHANISM_COLORS[mechanism],
                           alpha=0.25, linewidths=0, rasterized=True, zorder=3)
                ax.axhline(0, **REFERENCE, zorder=1)
                ax.axvline(0, **REFERENCE, zorder=1)
                if column == 0:
                    low = min(np.min(x), np.min(y))
                    high = max(np.max(x), np.max(y))
                    ax.plot([low, high], [low, high], zorder=2, **REFERENCE)
                ax.plot(np.mean(x), np.mean(y), marker="X", color=INK, markersize=9,
                        markeredgecolor="white", zorder=5)
                corner = (0.02, 0.98, "left", "top") if column == 0 else (0.98, 0.03, "right",
                                                                          "bottom")
                ax.text(corner[0], corner[1], f"{100 * zero.mean():.0f}% of auctions at (0, 0)",
                        transform=ax.transAxes, ha=corner[2], va=corner[3], fontsize=7,
                        color=INK, zorder=6,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85,
                              "edgecolor": "none"})
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                _compact_axes(ax)
                ax.set_title(f"({chr(97 + 2 * row + column)}) {SHORT[mechanism]}: {title}",
                             loc="left")
        for column in range(2):  # panels of the same quantity share their limits
            limits = [axes[row, column] for row in range(2)]
            for getter, setter in (("get_xlim", "set_xlim"), ("get_ylim", "set_ylim")):
                low = min(getattr(ax, getter)()[0] for ax in limits)
                high = max(getattr(ax, getter)()[1] for ax in limits)
                for ax in limits:
                    getattr(ax, setter)(low, high)
        handles = [mechanism_line("vcg_pa", marker="o", linestyle="none", label="VCG-PA auction"),
                   mechanism_line("ivcg_pa", marker="o", linestyle="none",
                                  label="IVCG-PA auction"),
                   Line2D([], [], **REFERENCE, label="45° line: all of the achievable gain"),
                   Line2D([], [], marker="X", color=INK, linestyle="none",
                          label="mean (ratio of means = share of first-best gain)")]
        _shared_legend(fig, handles, columns=2)
        note = (f"One point per simulated auction ({len(first_best_gain):,} auctions, the same "
                f"for all mechanisms); auctions where both coordinates are zero are not drawn.")
        return _finish(fig, path, context_line(combination), note)


def _integrated(q_start, q_end, start, end, points_per_segment=8):
    """Points (q, f(q), ∫_0^q f) of a function linear on contiguous segments of q."""
    qs, values, integrals = [], [], []
    total = 0.0
    for a, b, fa, fb in zip(q_start, q_end, start, end):
        if b <= a:
            continue
        t = np.linspace(0.0, 1.0, points_per_segment)
        q = a + t * (b - a)
        qs.append(q)
        values.append(fa + t * (fb - fa))
        integrals.append(total + (b - a) * (fa * t + 0.5 * (fb - fa) * t ** 2))
        total += (b - a) * 0.5 * (fa + fb)
    return np.concatenate(qs), np.concatenate(values), np.concatenate(integrals)


def ironing(results, data, path):
    combination = results["combination"]
    q_start, q_end = data["iron_quantile_start"], data["iron_quantile_end"]
    q, welfare, psi = _integrated(q_start, q_end, data["iron_welfare_start"],
                                  data["iron_welfare_end"])
    _, score, hull = _integrated(q_start, q_end, data["iron_score_start"], data["iron_score_end"])
    pooled = np.abs(welfare - score) > 1e-9 * max(1.0, float(np.abs(welfare).max()))
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(9.3, 5.6, 2, 3)

        def shade(ax, x):
            ax.fill_between(x, 0, 1, where=pooled, **REGION_QTY,
                            linewidth=0, transform=ax.get_xaxis_transform(),
                            zorder=0)

        ax = axes[0, 0]
        shade(ax, q)
        ax.plot(q, welfare, color=INK, linewidth=1.4, zorder=3)
        ax.plot(q, score, color=MECHANISM_COLORS["ivcg_pa"], linestyle="--", zorder=4)
        ax.set_xlabel("participating quantile q (ordered by bid)")
        ax.set_ylabel("welfare contribution")
        _compact_axes(ax, x=False)
        ax.set_title(r"(a) $\hat{w}^{\tau}(q)$ and ironed $\bar{\omega}^{\tau}(q)$", loc="left")

        ax = axes[0, 1]
        shade(ax, q)
        ax.plot(q, psi, color=INK, linewidth=1.4, zorder=3)
        ax.plot(q, hull, color=MECHANISM_COLORS["ivcg_pa"], linestyle="--", zorder=4)
        ax.set_xlabel("participating quantile q")
        ax.set_ylabel(r"$\Psi(q) = \int_0^q \hat{w}^{\tau}$")
        _compact_axes(ax, x=False)
        ax.set_title(r"(b) $\Psi$ and its convex hull", loc="left")

        ax = axes[0, 2]
        shade(ax, q)
        ax.plot(q, psi - hull, color=MECHANISM_COLORS["ivcg_pa"], zorder=3)
        ax.axhline(0, **REFERENCE)
        ax.set_xlabel("participating quantile q")
        ax.set_ylabel(r"$\Psi$ − convex hull of $\Psi$")
        ax.set_title("(c) Gap: positive exactly where bids are pooled", loc="left")

        ax = axes[1, 0]
        bids = np.concatenate([data["iron_bid_start"][:, None], data["iron_bid_end"][:, None]],
                              axis=1).ravel()
        order_welfare = np.concatenate([data["iron_welfare_start"][:, None],
                                        data["iron_welfare_end"][:, None]], axis=1).ravel()
        order_score = np.concatenate([data["iron_score_start"][:, None],
                                      data["iron_score_end"][:, None]], axis=1).ravel()
        for low, high in _pooled_bid_intervals(data):
            ax.axvspan(low, high, linewidth=0, **REGION_QTY)
        ax.plot(bids, order_welfare, color=INK, linewidth=1.4, zorder=3)
        ax.plot(bids, order_score, color=MECHANISM_COLORS["ivcg_pa"], linestyle="--", zorder=4)
        ax.set_xlabel("bid b")
        ax.set_ylabel("score")
        _compact_axes(ax)
        ax.set_title("(d) Scores as functions of the bid", loc="left")

        ax = axes[1, 1]
        for mechanism in PARTICIPANT_AUDITS:
            ax.plot(data["alloc_bids"], data[f"alloc_{mechanism}"],
                    color=MECHANISM_COLORS[mechanism], linestyle=MECHANISM_LINESTYLES[mechanism],
                    zorder=3)
        for low, high in _pooled_bid_intervals(data):
            ax.axvspan(low, high, linewidth=0, **REGION_QTY)
        ax.set_xlabel("bid b")
        ax.set_ylabel(r"interim allocation $X^{\tau}(b)$")
        _compact_axes(ax, y=False)
        ax.set_title("(e) Interim allocation", loc="left")
        axes[1, 2].axis("off")
        handles = [Line2D([], [], color=INK, linewidth=1.4,
                          label=r"expected welfare contribution $\hat{w}^{\tau}$ / $\Psi$"),
                   Line2D([], [], color=MECHANISM_COLORS["ivcg_pa"], linestyle="--",
                          label=r"ironed $\bar{\omega}^{\tau}$ (IVCG-PA score) / convex hull"),
                   Patch(**REGION_QTY, label="pooled bids"),
                   mechanism_line("vcg_pa", label=r"$X^{\tau}$ in VCG-PA"),
                   mechanism_line("ivcg_pa", label=r"$X^{\tau}$ in IVCG-PA")]
        axes[1, 2].legend(handles=handles, loc="center", fontsize=8)
        note = (f"IVCG-PA design (ironing) on the training distribution "
                f"({results['ivcg_design']}); interim allocations on the evaluation distribution.")
        return _finish(fig, path, context_line(combination), note)


def marginal_contribution(results, data, types, first_order, path):
    combination = results["combination"]
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 5.6, 2, 2)
        values = np.concatenate([np.abs(data[f"mc_{m}"]).ravel() for m in PARTICIPANT_AUDITS])
        limit = float(np.quantile(values[np.isfinite(values)], 0.98)) or 1.0
        # Symmetric logarithmic scale: M varies over orders of magnitude, and its sign
        # near the threshold matters most.
        norm = SymLogNorm(linthresh=0.01 * limit, vmin=-limit, vmax=limit, base=10)
        mesh = None
        for column, mechanism in enumerate(PARTICIPANT_AUDITS):
            ax = axes[0, column]
            mesh = ax.pcolormesh(data["mc_e"], data["mc_v"], data[f"mc_{mechanism}"].T,
                                 cmap="PuOr", norm=norm, shading="auto", rasterized=True,
                                 zorder=0)
            ax.contour(data["mc_e"], data["mc_v"], data[f"mc_{mechanism}"].T, levels=[0.0],
                       colors=INK, linewidths=1.0, linestyles="--", zorder=3)
            density = np.asarray(types["background_density"], dtype=float)
            if density.max() > 0:
                ax.contour(types["background_e"], types["background_v"], density,
                           levels=[density.max() * 0.02], colors=INK_SECONDARY, linewidths=0.6,
                           alpha=0.6, zorder=2)
            draw_threshold(ax, data["e_grid"], data["tau_grid"], types)
            type_plane_axes(ax, types)
            ax.set_title(f"({'ab'[column]}) " + r"$M^{\tau}(e, v)$" + f" in {SHORT[mechanism]}",
                         loc="left")
        _colorbar(fig, mesh, axes[0, :], r"marginal contribution $M^{\tau}(e, v)$")

        ax, ax_residual = axes[1, 0], axes[1, 1]
        low, high = types["e_limits"]
        shown = first_order[(first_order.mass > 0) & (first_order.status != "excluded")
                            & first_order.externality.between(low, high)]
        for mechanism in PARTICIPANT_AUDITS:
            rows = shown[shown.mechanism == mechanism]
            interior = rows[rows.status == "interior"]
            color = MECHANISM_COLORS[mechanism]
            ax.plot(rows.externality, rows.penalty, color=color, zorder=3)
            ax.plot(interior.externality, interior.pigouvian_charge, color=color, linestyle=":",
                    marker=MECHANISM_MARKERS[mechanism], markersize=3, linewidth=1.0, zorder=4)
            ax_residual.plot(interior.externality, interior.marginal_contribution, color=color,
                             linestyle=MECHANISM_LINESTYLES[mechanism],
                             marker=MECHANISM_MARKERS[mechanism], markersize=3, zorder=3)
        for axis in (ax, ax_residual):
            axis.axhline(0, **REFERENCE, zorder=1)
            axis.set_xlim(types["e_limits"])
            axis.set_xlabel("externality e")
            _compact_axes(axis)
        ax.set_ylabel("penalty")
        ax.set_title("(c) Penalty r(e) and Pigouvian charge", loc="left")
        ax_residual.set_ylabel(r"$M^{\tau}(e, \tau(e))$")
        ax_residual.set_title("(d) M of threshold types", loc="left")
        handles = [Line2D([], [], color=INK, linestyle="--", linewidth=1,
                          label=r"$M^{\tau} = 0$ (zero contour)"),
                   Line2D([], [], color=THRESHOLD_COLOR, linewidth=2, label="threshold τ(e)"),
                   Line2D([], [], color=INK_SECONDARY, linewidth=0.6,
                          label="edge of the type distribution"),
                   mechanism_line("vcg_pa", label="VCG-PA: penalty r(e)"),
                   mechanism_line("vcg_pa", linestyle=":", marker="s", markersize=3,
                                  label="VCG-PA: Pigouvian charge"),
                   mechanism_line("ivcg_pa", label=r"IVCG-PA: penalty / $M^{\tau}$"),
                   mechanism_line("ivcg_pa", linestyle=":", marker="D", markersize=3,
                                  label="IVCG-PA: Pigouvian charge")]
        _shared_legend(fig, handles, columns=3)
        note = ("Marginal contribution M(e, v) = (v + e) X(v) − L(v) of a participating type, "
                "for the auction and the rivals' thresholds of this τ (training distribution; "
                "IVCG-PA with its scores fixed). At an optimal interior threshold, M(e, τ(e)) = 0 "
                "and the penalty equals the Pigouvian charge −eX + L − P (paper: optimality "
                "conditions for thresholds). Panels (c), (d): at the externality nodes; charge and M at interior "
                "thresholds only; excluded externalities (penalty R(v̄)) are not drawn.")
        return _finish(fig, path, context_line(combination), note)


def search_history(results, data, types, path_tried, path_trace):
    combination, search = results["combination"], results["search"]
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(3.9, 3.3)
        ax = axes[0, 0]
        draw_types(ax, types, alpha=0.45)
        tried = data["search_tried"]
        for curve in tried:
            ax.plot(data["search_e"], np.clip(curve, *default_view(types)["v_limits"]),
                    color="#9a9a9a",
                    linewidth=0.5, alpha=0.18, zorder=2, rasterized=True,
                    antialiased=False)  # without antialiasing the image compresses ~2×
        draw_threshold(ax, data["e_grid"], data["tau_grid"], types, color=THRESHOLD_COLOR,
                       label="best τ found")
        ax.set_title("Tried thresholds", loc="left")
        shown = (f"all {len(tried):,} tried τ" if len(tried) >= search["num_tried"]
                 else f"{len(tried):,} of {search['num_tried']:,} tried τ (random subset)")
        handles = [Line2D([], [], color="#9a9a9a", linewidth=0.8, label=shown),
                   Line2D([], [], color=THRESHOLD_COLOR, linewidth=2, label="best τ found")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        _finish(fig, path_tried, context_line(combination))

        fig, axes = _new_figure(3.9, 3.0)
        ax = axes[0, 0]
        trace = search["trace"]
        best = max(point["welfare"] for point in trace)
        benchmarks = search.get("training_benchmarks")
        gain = benchmarks["first_best"] - benchmarks["vcg"] if benchmarks else 0.0
        scale = gain if gain > 0 else max(abs(best), 1e-300)
        groups = {}
        for point in trace:
            if point["stage"] == "end":
                continue
            label = (point["stage"] if point["optimizer"] == "best_response"
                     else f"{point['stage']} (run {point['run']})")
            groups.setdefault((point["optimizer"], label), []).append(point)
        for index, ((optimizer, label), points) in enumerate(groups.items()):
            seconds = np.array([p["seconds"] for p in points])
            welfare = np.maximum.accumulate(np.array([p["welfare"] for p in points]))
            gap = np.maximum((best - welfare) / scale, 1e-13)
            ax.step(seconds, gap, where="post", color=CATEGORY_COLORS[index % len(CATEGORY_COLORS)],
                    linestyle=CATEGORY_LINESTYLES[index % 7], label=f"{optimizer}: {label}")
        ax.set_yscale("log")
        ax.set_xlabel("seconds since the start of the search")
        ax.set_ylabel("(best W − W so far) / (FB − W_VCG)" if gain > 0
                      else "(best W − W so far) / |best W|")
        ax.set_title("Convergence of the search (training objective)", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        note = (f"Best training welfare {compact(best)} after {search['seconds']:.1f} s; values "
                f"below 1e-13 are drawn at 1e-13.")
        return _finish(fig, path_trace, context_line(combination), note)


STOP_LABELS = {"no_improving_step": "no improving step", "small_gain": "small gain",
               "max_iterations": "iteration cap", "time_limit": "time limit",
               "stalled": "stalled", "max_generations": "generation cap",
               "converged": "converged", "max_evaluations": "evaluation cap"}
FLOOR = 1e-12  # log panels: smaller values (and zeros) are drawn here


def _reference(ax, value, text, axis="y"):
    """Dashed reference line with its label at the right end, above the line."""
    ax.axhline(value, **REFERENCE, zorder=1)
    ax.annotate(text, (1.0, value), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                textcoords="offset points", ha="right", va="bottom", fontsize=7,
                color=INK_SECONDARY)


def search_iterations(results, path, tolerances):
    """Welfare and stopping diagnostics against the iteration, one line per start.

    (a) share of the first-best gain (W − W_VCG)/(FB − W_VCG) on the training
    distribution; (b) distance to the best welfare found, in units of the first-best
    gain; (c) stationarity (projected gradient of W in the form's parameters); (d)
    largest violation of the first-order conditions in M(e, τ(e))/v̄; (e) best-response
    gap. tolerances: target levels drawn in (b)–(e).
    """
    combination, search = results["combination"], results["search"]
    lines = search["history"]
    benchmarks = search["training_benchmarks"]
    gain = benchmarks["first_best"] - benchmarks["vcg"]
    unit = gain if gain > 0 else max(abs(benchmarks["first_best"]), 1e-300)
    best = max(point["welfare"] for line in lines for point in line["points"])
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 7.6, 3, 2, sharex=True)
        axes[2, 1].set_visible(False)
        handles = []
        for index, line in enumerate(lines):
            points = line["points"]
            style = {"color": CATEGORY_COLORS[index % len(CATEGORY_COLORS)],
                     "linestyle": CATEGORY_LINESTYLES[index % len(CATEGORY_LINESTYLES)],
                     "linewidth": 1.2}
            x = np.array([p["iteration"] for p in points], dtype=float)
            welfare = np.array([p["welfare"] for p in points])
            axes[0, 0].plot(x, (welfare - benchmarks["vcg"]) / unit, **style)
            axes[0, 1].plot(x, np.maximum((best - welfare) / unit, FLOOR), **style)
            for ax, key in ((axes[1, 0], "stationarity"), (axes[1, 1], "max_marginal"),
                            (axes[2, 0], "gap")):
                values = np.array([np.nan if p.get(key) is None else p[key] for p in points],
                                  dtype=float)
                shown = np.isfinite(values)
                if shown.any():
                    ax.plot(x[shown], np.maximum(values[shown], FLOOR), marker="o",
                            markersize=2.5, **style)
            polished = [i for i, p in enumerate(points) if p.get("stage") == "polish"]
            if polished:
                axes[0, 0].plot(x[polished], (welfare[polished] - benchmarks["vcg"]) / unit,
                                linestyle="none", marker="o", markersize=5,
                                markerfacecolor="white", color=style["color"], zorder=4)
            stop = STOP_LABELS.get(line["stop_reason"], line["stop_reason"])
            handles.append(Line2D([], [], **style,
                                  label=f"{line['optimizer']}: {line['start']} ({stop})"))
        ax = axes[0, 0]
        _reference(ax, 0.0, "VCG")
        _reference(ax, 1.0, "first best")
        ax.set_ylabel("(W − W_VCG) / (FB − W_VCG)")
        ax.set_title("(a) Share of the first-best gain", loc="left")
        ax = axes[0, 1]
        ax.set_yscale("log")
        _reference(ax, tolerances["welfare"], f"target {tolerances['welfare']:g}")
        ax.set_ylabel("(best W − W) / (FB − W_VCG)")
        ax.set_title("(b) Distance to the best welfare found", loc="left")
        diagnostics = any(p.get("max_marginal") is not None
                          for line in lines for p in line["points"])
        panels = ((axes[1, 0], "stationarity", "m Σ|∂W/∂θ| / (FB − W_VCG), m = median v",
                   "(c) Stationarity in the form's parameters θ"),
                  (axes[1, 1], "max_marginal", "first-order violation in M / v̄",
                   "(d) Zero marginal contribution"),
                  (axes[2, 0], "gap", "best-response gap / (FB − W_VCG)",
                   "(e) Gain of one bidder deviating alone"))
        for ax, key, label, title in panels:
            ax.set_title(title, loc="left")
            ax.set_xlabel("iteration (generation for the genetic stage)")
            ax.xaxis.set_tick_params(labelbottom=True)
            if not diagnostics:
                ax.set_yticks([])
                ax.text(0.5, 0.5, "best-response ascent only\n(forms knots, bins, nodes)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=7,
                        color=INK_SECONDARY)
                continue
            ax.set_yscale("log")
            _reference(ax, tolerances[key], f"target {tolerances[key]:g}")
            ax.set_ylabel(label)
        _shared_legend(fig, handles, columns=2)
        note = (f"Training distribution: W_VCG = {compact(benchmarks['vcg'])}, "
                f"FB = {compact(benchmarks['first_best'])}, best W found = {compact(best)}. "
                f"(c)–(e): best-response ascent, at the τ of each iteration; hollow circles in "
                f"(a): after a gradient polish. (c) is zero at a stationary point in the form, (d) "
                f"at a τ that meets the first-order conditions node by node (|M| at interior τ, "
                f"M(e, 0) < 0 at τ = 0, M(e, v̄) > 0 at exclusion; need not reach zero for "
                f"restricted forms). τ ≡ 0 meets both (M(e, 0) = 0) without being optimal; (e) "
                f"detects it, but (e) need not be zero at an optimal common τ. "
                f"Values below {FLOOR:g} are drawn at {FLOOR:g}. Legend: start (stop reason).")
        return _finish(fig, path, context_line(combination), note)


# ─── Across combinations ────────────────────────────────────────────────────────


def threshold_overlay(entries, types, title, path, note=None):
    """entries: dicts with e_grid, tau_grid, label, color, linestyle."""
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(3.9, 3.3)
        ax = axes[0, 0]
        draw_types(ax, types, alpha=0.45)
        ax.plot(types["e_grid"], -types["e_grid"], color=MECHANISM_COLORS["first_best"],
                linestyle="-.", linewidth=1.2, zorder=2, alpha=0.8)
        handles = []
        for order, entry in enumerate(entries):
            line = draw_threshold(ax, entry["e_grid"], entry["tau_grid"], types,
                                  color=entry["color"], linestyle=entry["linestyle"],
                                  label=entry["label"], linewidth=1.9, zorder=5 + order)
            handles.append(line)
        handles.append(Line2D([], [], color=MECHANISM_COLORS["first_best"], linestyle="-.",
                              label="v = −e (first best)"))
        _shared_legend(fig, handles, columns=1)
        return _finish(fig, path, title, note)


def distribution_preview(types, title, path):
    """Type density with the line v = −e (welfare contribution w = 0), saved as WebP only."""
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(3.1, 2.4)
        ax = axes[0, 0]
        draw_types(ax, types, alpha=0.9)
        ax.plot(types["e_grid"], -types["e_grid"], color=MECHANISM_COLORS["first_best"],
                linestyle="-.", linewidth=1.2, zorder=2)
        ax.set_title(title, loc="left")
        fig.savefig(path, dpi=DISTRIBUTION_PREVIEW_DPI, pil_kwargs={"quality": 80})
        plt.close(fig)
    return path


def welfare_against_zeta(rows, title, path, note=None):
    """rows: dicts with zeta, exact welfare per mechanism, gains, intervals (may be None)."""
    rows = sorted(rows, key=lambda r: r["zeta"])
    zetas = np.array([r["zeta"] for r in rows])
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 3.0, 1, 2)
        for column, (key, mechanisms, label) in enumerate((
                ("welfare", MECHANISMS, "expected welfare W"),
                ("gain", AUDITED, "expected welfare gain over VCG"))):
            ax = axes[0, column]
            for offset, mechanism in enumerate(mechanisms):
                values = np.array([r[f"{key}_{mechanism}"] for r in rows])
                intervals = [r.get(f"{key}_{mechanism}_interval") for r in rows]
                jitter = zetas * (1 + 0.012 * (offset - 1.5))
                ax.plot(jitter, values, color=MECHANISM_COLORS[mechanism],
                        linestyle=MECHANISM_LINESTYLES[mechanism],
                        marker=MECHANISM_MARKERS[mechanism], markersize=5, zorder=3)
                for x, value, interval in zip(jitter, values, intervals):
                    if interval:
                        ax.errorbar(x, value, yerr=[[max(value - interval[0], 0)],
                                                    [max(interval[1] - value, 0)]],
                                    color=MECHANISM_COLORS[mechanism], capsize=2.5,
                                    linewidth=1.0, zorder=4)
            if zetas.max() / max(zetas.min(), 1e-300) >= 10:
                ax.set_xscale("log")
            ax.set_xticks(zetas, [f"{z:g}" for z in zetas])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel("externality scale ζ")
            ax.set_ylabel(label)
            _compact_axes(ax, x=False)
            ax.set_title(f"({'ab'[column]}) {label}", loc="left")
        _shared_legend(fig, [mechanism_line(m, marker=MECHANISM_MARKERS[m]) for m in MECHANISMS],
                       columns=4)
        return _finish(fig, path, title, note)


def share_overview(groups, path, title, note=None):
    """groups: {panel title: [(row label, share VCG-PA, share IVCG-PA, interval VCG-PA)]}."""
    rows_total = sum(len(rows) for rows in groups.values())
    with plt.rc_context(STYLE):
        fig, axes = _new_figure(6.5, 1.2 + 0.19 * rows_total + 0.5 * len(groups), len(groups), 1,
                                gridspec_kw={"height_ratios": [len(r) for r in groups.values()]})
        for ax, (panel, rows) in zip(axes[:, 0], groups.items()):
            positions = np.arange(len(rows))[::-1]
            for position, (label, share_pa, share_ipa, interval) in zip(positions, rows):
                if interval:
                    ax.plot(interval, [position, position], color=MECHANISM_COLORS["vcg_pa"],
                            linewidth=1.2, zorder=2)
                ax.plot(share_pa, position, marker="s", color=MECHANISM_COLORS["vcg_pa"],
                        markersize=5, zorder=3, linestyle="none")
                ax.plot(share_ipa, position + 0.18, marker="D", color=MECHANISM_COLORS["ivcg_pa"],
                        markersize=4.5, zorder=3, linestyle="none")
            ax.set_yticks(positions, [row[0] for row in rows], fontsize=7)
            ax.axvline(1.0, color=MECHANISM_COLORS["first_best"], linestyle="-.", linewidth=1)
            ax.axvline(0.0, **REFERENCE)
            ax.grid(axis="y", visible=False)
            ax.set_title(panel, loc="left", fontsize=9)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        axes[-1, 0].set_xlabel("share of the first-best gain over VCG, (W − W_VCG)/(W_FB − W_VCG)")
        handles = [Line2D([], [], marker="s", color=MECHANISM_COLORS["vcg_pa"], linestyle="none",
                          label="VCG-PA"),
                   Line2D([], [], marker="D", color=MECHANISM_COLORS["ivcg_pa"], linestyle="none",
                          label="IVCG-PA"),
                   Line2D([], [], color=MECHANISM_COLORS["first_best"], linestyle="-.",
                          label="first best (100%)")]
        _shared_legend(fig, handles, columns=3)
        return _finish(fig, path, title, note)
