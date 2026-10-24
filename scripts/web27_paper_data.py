r"""Data for the empirical section and appendix of the WEB27 submission: pgfplots tables,
type-distribution background images and LaTeX table rows, from the paper run
(configs/paper_run.toml).

    python scripts/web27_paper_data.py RUN OUT_DIR

Type-plane panels use zoomed views (zoomed_view with VIEW_OPTIONS); the IVCG-PA type-plane
figure of harmful-middle (types_hm) shows the whole support. Writes to OUT_DIR:
  table_results.tex       tabular of the appendix table (TABLE_MARKETS of every distribution)
  share_k1.dat            share of the first-best gain (%) against n at k = 1, harmful-middle
  share_ratio_k<k>.dat    share of the first-best gain (%) against n/k for k goods, harmful-middle
  tau_<family>.dat        VCG-PA-optimized τ(e) for n ∈ SHAPE_BIDDERS, k = 1
  tau_markets.dat         VCG-PA-optimized τ(e) of every market with k < n, harmful-middle
  bg_<family>.png         type density of each distribution (axes limits in limits.tex)
  types_hm.dat            τ(e) and IVCG-PA penalty r(e), harmful-middle, n = 4, k = 1, IVCG-PA
  pooled_hm.tex           bands of the bids pooled by IVCG-PA there (pgfplots fill commands)
  bg_types_hm.png         its background (whole support; macros \eminTypes ... in limits.tex)
  shapes_<family>.dat/.tex      τ(e) and penalty r(e) of VCG-PA in the SHAPE_MARKETS and of IVCG-PA
                          in SHAPE_PAIR, and pgfplots commands drawing them (color = n/k as in
                          family_tau_*.tex; VCG-PA τ solid, r dashed; IVCG-PA τ dotted with
                          diamonds, r dash-dotted with circles)
  shapes_fill_<family>.tex      the band of bids pooled by IVCG-PA there, at the
                          participating types only (drawn below the type density)
  mc_<family>.png         marginal contribution M(e, v) of VCG-PA at its optimized τ, n = 8, k = 1
                          (PuOr, symmetric logarithmic scale, dashed M = 0, edge of the type density)
  tau_mc_<family>.dat     the optimized τ(e) drawn over mc_<family>.png (view: MAP_VIEW_OPTIONS,
                          else the panel view, widened by MAP_ZOOM_OUT; macros \eminMap<Family> ...)
  limits.tex              axis limits of every type-plane panel as macros, and the extent of
                          its background image (padded by IMAGE_PAD; \eminImg<Family> ...)
and everything family_figures_data.py writes (figures of shares and τ across distributions).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PA_auctions import figures  # noqa: E402

FAMILIES = {"independent": "Independent", "harmful-middle": "HarmfulMiddle",
            "heavy-tail": "HeavyTail", "harm-rises": "HarmRises"}
FAMILY_NAMES = {family: family for family in FAMILIES}
MAIN_FAMILY = "harmful-middle"
MAP_MARKET = (8, 1)
MAP_FAMILIES = ["harmful-middle", "heavy-tail", "harm-rises", "independent"]
TABLE_MARKETS = [(2, 1), (8, 1), (32, 1), (8, 2), (16, 4), (32, 8), (4, 3), (16, 14)]
SHAPE_BIDDERS = [2, 4, 8, 16, 32]
# Markets of the τ-and-penalty figure (n/k = 2, 8, 32: well separated colors, as in
# family_figures_data), with VCG-PA in each and also IVCG-PA in one per distribution (with the band
# of bids it pools): where the two differ most (harmful-middle) and where they coincide (others).
SHAPE_MARKETS = [(2, 1), (16, 2), (32, 1)]
SHAPE_PAIR = {"independent": (32, 1), "harmful-middle": (2, 1), "heavy-tail": (2, 1), "harm-rises": (16, 2)}
TABLE_HEADER = r"""\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & & & \multicolumn{6}{c}{VCG-PA with $\tau$} & IVCG-PA & part.\\
\cmidrule(lr){4-9}
distribution & $n$ & $k$ & $\tau^0$ & relax.\ $\tau$ & cutoff & affine & optimized & relax.\ bound & optimized & gap\\
\midrule
"""
FOOTER = "\\bottomrule\n\\end{tabular}\n"


def label(row):
    return row.tau_label if isinstance(row.tau_label, str) else f"{row.objective}_{row.threshold_form}"


def shares(run, family):
    """Share of the first-best gain by market and τ for one distribution: τ⁰, relaxation τ,
    cutoff, poly1, VCG-PA and IVCG-PA optimized, participation gap, and the relaxation bound
    (VCG-PA optimum + relaxation gap)."""
    summary = pd.read_csv(run / "summary.csv")
    summary = summary[summary.distribution == family].copy()
    summary["key"] = summary.apply(label, axis=1)
    gaps = pd.read_csv(run / "gap-decomposition.csv")
    gaps = gaps[gaps.distribution == family]
    table = {}
    for (n, k), group in summary.groupby(["num_bidders", "num_goods"]):
        row = {}
        for _, r in group.iterrows():
            if r.key == "zero-welfare":
                row["tau0"] = r.share_vcg_pa
            elif r.key == "relaxation":
                row["relaxtau"] = r.share_vcg_pa
            elif r.key == "vcg_pa_cutoff":
                row["cutoff"] = r.share_vcg_pa
            elif r.key == "vcg_pa_poly1":
                row["line"] = r.share_vcg_pa
            elif r.key == "vcg_pa_nodes":
                row["vcgpa"] = r.share_vcg_pa
                row["gap"] = r.participation_gap_share_vcg_pa
            elif r.key == "ivcg_pa_nodes":
                row["ivcgpa"] = r.share_ivcg_pa
        nodes = gaps[(gaps.num_bidders == n) & (gaps.num_goods == k) & (gaps.threshold_class == "nodes")]
        if len(nodes) and np.isfinite(nodes.relaxation_minus_opt_vcg_share.iloc[0]) and "vcgpa" in row:
            row["bound"] = row["vcgpa"] + nodes.relaxation_minus_opt_vcg_share.iloc[0]
        table[(n, k)] = row
    return table


def percent(value):
    if value is None or not np.isfinite(value):
        return "--"
    return f"{100 * value:.1f}".replace("-0.0", "0.0")


def load_types(run, family):
    with np.load(run / family / "z1" / "types.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_data(leaf):
    with np.load(leaf / "data.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# Views of the type plane chosen per distribution (zoomed_view options): independent zoomed to
# the 2.5%–97.5% externality quantiles; heavy-tail also cut at the 80% valuation quantile and
# shifted right by 10% of its width, so the density is nearer the center.
VIEW_OPTIONS = {"independent": {"e_levels": (0.025, 0.975)},
                "heavy-tail": {"e_levels": (0.025, 0.975), "v_top": 0.80, "e_shift": 0.10}}
# Views of the marginal-contribution maps where they differ from VIEW_OPTIONS (heavy-tail: most of
# the distribution).
MAP_VIEW_OPTIONS = {"heavy-tail": {"e_levels": (0.005, 0.995), "v_top": 0.95}}
# Map views are widened by MAP_ZOOM_OUT of their width on each side and of their height at the top.
MAP_ZOOM_OUT = 0.08


def map_view(types, family):
    view = zoomed_view(types, **MAP_VIEW_OPTIONS.get(family, VIEW_OPTIONS.get(family, {})))
    e_low, e_high = (float(x) for x in view["e_limits"])
    v_low, v_high = (float(x) for x in view["v_limits"])
    view["e_limits"] = np.array([e_low - MAP_ZOOM_OUT * (e_high - e_low), e_high + MAP_ZOOM_OUT * (e_high - e_low)])
    view["v_limits"] = np.array([v_low, v_high + MAP_ZOOM_OUT * (v_high - v_low)])
    return view
# Background images extend beyond the view by IMAGE_PAD of its width and height on each side, so a
# panel can widen its limits (\eminImg... macros give the image extent).
IMAGE_PAD = 0.25


def padded(view, pad=IMAGE_PAD):
    out = dict(view)
    for key in ("e_limits", "v_limits"):
        low, high = (float(x) for x in view[key])
        out[key] = np.array([low - pad * (high - low), high + pad * (high - low)])
    return out


def at_support_top(values, low, vbar):
    """Values for a type-plane panel: below low raised to low; not finite (τ = v̄⁺, the
    externality is excluded) at the top valuation v̄ of the distribution, so τ runs flat along v̄
    there, as τ = 0 runs along the bottom. Values above the panel are left for the axis to clip."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), np.maximum(values, low), vbar)


def _quantiles(axis, weights, levels):
    cumulative = np.cumsum(weights) / weights.sum()
    return [float(np.interp(level, cumulative, axis)) for level in levels]


def zoomed_view(types, scale="linear", e_levels=(0.005, 0.995), v_top=None, e_shift=0.0):
    """The view of a type-plane panel, zoomed to quantiles of the marginal distributions of e and
    v (from the background density, valuations × externalities), with a 3% margin; v_top
    overrides the upper valuation quantile of a linear view; e_shift moves the externality range
    right by that share of its width."""
    view = figures.type_view(types, scale)
    density = np.asarray(types["background_density"], dtype=float)
    e_low, e_high = _quantiles(np.asarray(types["background_e"]), density.sum(axis=0), e_levels)
    levels = (0.005, 0.995) if scale == "log" else (0.005, v_top or 0.99)
    v_low, v_high = _quantiles(np.asarray(types["background_v"]), density.sum(axis=1), levels)
    e_margin = 0.03 * (e_high - e_low)
    shift = e_shift * (e_high - e_low + 2 * e_margin)
    view["e_limits"] = np.array([e_low - e_margin + shift, e_high + e_margin + shift])
    if scale == "log":
        view["v_limits"] = np.array([max(v_low, 1e-3), v_high * 1.05])
    else:
        v_margin = 0.03 * (v_high - v_low)
        view["v_limits"] = np.array([v_low - v_margin, v_high + v_margin])
    return view


def linear_view(types, family=None):
    """The view of a type-plane panel: the zoomed view, with the VIEW_OPTIONS of the family."""
    return zoomed_view(types, **VIEW_OPTIONS.get(family, {}))


def background(types, path, family=None, view=None):
    """The type density alone, filling the image, with the limits of its view (default: the
    zoomed view of the family)."""
    view = linear_view(types, family) if view is None else view
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    figures.draw_types(ax, padded(view))
    ax.set_axis_off()
    fig.savefig(path, dpi=300, transparent=True)
    figures.plt.close(fig)
    return view


def marginal_background(data, view, path):
    """M(e, v) of VCG-PA on a view of the type plane, filling the image: PuOr on a symmetric
    logarithmic scale as in the run figures, the zero contour dashed and the edge of the type density."""
    values = np.asarray(data["mc_vcg_pa"], dtype=float)
    limit = float(np.quantile(np.abs(values[np.isfinite(values)]), 0.98)) or 1.0
    norm = figures.SymLogNorm(linthresh=0.01 * limit, vmin=-limit, vmax=limit, base=10)
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(data["mc_e"], data["mc_v"], values.T, cmap="PuOr", norm=norm, shading="auto")
    ax.contour(data["mc_e"], data["mc_v"], values.T, levels=[0.0], colors=figures.INK,
               linewidths=0.9, linestyles="--")
    density = np.asarray(view["background_density"], dtype=float)
    if density.max() > 0:
        ax.contour(view["background_e"], view["background_v"], density, levels=[density.max() * 0.02],
                   colors=figures.INK_SECONDARY, linewidths=0.6)
    ax.set_xlim(*view["e_limits"])
    ax.set_ylim(*view["v_limits"])
    ax.set_axis_off()
    fig.savefig(path, dpi=200)
    figures.plt.close(fig)
    return view


def limit_macros(macro, view, image=True):
    r"""Axis limits of a view (\emin<macro> ...) and, with image, the extent of its padded
    background image (\eminImg<macro> ...)."""
    text = ""
    for prefix, limits in [("", view)] + ([("Img", padded(view))] if image else []):
        e_low, e_high = (float(x) for x in limits["e_limits"])
        v_low, v_high = (float(x) for x in limits["v_limits"])
        text += (rf"\def\emin{prefix}{macro}{{{e_low:.4f}}}\def\emax{prefix}{macro}{{{e_high:.4f}}}"
                 rf"\def\vmin{prefix}{macro}{{{v_low:.5g}}}\def\vmax{prefix}{macro}{{{v_high:.4f}}}")
    return text


def shown_threshold(data, view, clip_above=False):
    """τ(e) for a panel of the view (at_support_top: an excluded externality at v̄);
    clip_above=True puts values above the view at its top (for filling the abstention region)."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    low, high = (float(x) for x in view["v_limits"])
    if clip_above:
        return np.where(np.isfinite(tau), np.clip(tau, low, high), high)
    return at_support_top(tau, low, float(view["maximum_valuation"]))


def write_table(run, out):
    with open(out / "table_results.tex", "w") as file:
        file.write(TABLE_HEADER)
        for family, name in FAMILY_NAMES.items():
            table = shares(run, family)
            for index, (n, k) in enumerate(TABLE_MARKETS):
                row = table[(n, k)]
                first = rf"\multirow{{{len(TABLE_MARKETS)}}}{{*}}{{{name}}}" if index == 0 else ""
                file.write(f"{first} & {n} & {k} & " + " & ".join(percent(row.get(c)) for c in
                           ("tau0", "relaxtau", "cutoff", "line", "vcgpa", "bound", "ivcgpa"))
                           + f" & {100 * row['gap']:.2f}" + r" \\" + "\n")
            file.write(r"\midrule" + "\n" if family != list(FAMILY_NAMES)[-1] else "")
        file.write(FOOTER)


def pooled_fill(data, view):
    """pgfplots fill commands for the bands of bids pooled by an IVCG-PA design, across the
    width of the image."""
    image = padded(view)
    e_low, e_high = (float(x) for x in image["e_limits"])
    return [rf"\fill[regionqty] (axis cs:{e_low:.4f},{low:.5g}) rectangle (axis cs:{e_high:.4f},{high:.5g});"
            for low, high in figures._visible_pooled_intervals(data, view)]


MARK_PANEL = (3.6, 1.9)  # panel width and height (cm) of the τ-and-penalty figure
MARK_STEP = 0.32         # distance (cm) between marks along a curve


def mark_points(e, values, view, phase=0.0):
    """Points at equal distances MARK_STEP along a curve as drawn in a MARK_PANEL panel of the
    view (inside the view only), starting phase·MARK_STEP from its first visible point."""
    e, values = np.asarray(e, dtype=float), np.asarray(values, dtype=float)
    e_low, e_high = (float(x) for x in view["e_limits"])
    v_low, v_high = (float(x) for x in view["v_limits"])
    inside = (e >= e_low) & (e <= e_high) & (values >= v_low) & (values <= v_high)
    x = (e - e_low) / (e_high - e_low) * MARK_PANEL[0]
    y = (values - v_low) / (v_high - v_low) * MARK_PANEL[1]
    chosen, travelled, target = [], 0.0, phase * MARK_STEP
    for i in range(len(e)):
        if i > 0 and inside[i] and inside[i - 1]:
            travelled += float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        if inside[i] and travelled >= target:
            chosen.append(i)
            target = travelled + MARK_STEP
    return e[chosen], values[chosen]


def write_shapes(run, family, view, out):
    """τ and penalty of the SHAPE_MARKETS of one distribution (VCG-PA; also IVCG-PA at
    SHAPE_PAIR), and pgfplots commands: shapes_<family>.tex draws VCG-PA τ (solid) first, then
    VCG-PA r (dashed), IVCG-PA r (thin dash-dotted, hollow circles) and IVCG-PA τ (thin dotted,
    hollow diamonds; marks evenly spaced along the drawn curve, shapes_marks_<family>.dat), so
    IVCG-PA stays visible where it coincides with VCG-PA; shapes_fill_<family>.tex
    the bands of bids pooled by IVCG-PA, drawn below the type density."""
    from family_figures_data import RATIO_COLORS
    low, vbar = float(view["v_limits"][0]), float(view["maximum_valuation"])
    columns, names, fills, e_grid = [], [], [], None
    layers = {"tau-vcgpa": [], "r-vcgpa": [], "r-ivcgpa": [], "tau-ivcgpa": []}
    styles = {"tau-vcgpa": "line width=1.4pt", "r-vcgpa": "line width=1.1pt,densely dashed",
              "r-ivcgpa": "line width=0.6pt,dashdotted", "tau-ivcgpa": "line width=0.6pt,densely dotted"}
    mark_styles = {"r": "only marks,mark=*,mark size=1.6pt,mark options={fill=white,line width=0.7pt}",
                   "tau": "only marks,mark=diamond*,mark size=2.4pt,mark options={fill=white,line width=0.8pt}"}
    marks, mark_names = [], []
    for n, k in SHAPE_MARKETS:
        color = RATIO_COLORS[min(RATIO_COLORS, key=lambda r: abs(r - n / k))]
        for objective in ["vcgpa"] + (["ivcgpa"] if (n, k) == SHAPE_PAIR[family] else []):
            data = load_data(run / family / "z1" / f"n{n}-k{k}" / f"{objective}_nodes_br")
            mechanism = "ivcg_pa" if objective == "ivcgpa" else "vcg_pa"
            name = f"{objective}n{n}k{k}"
            columns += [shown_threshold(data, view), at_support_top(data[f"penalty_{mechanism}"], low, vbar)]
            names += [f"tau{name}", f"r{name}"]
            source = rf"{{\familydata/shapes_{family}.dat}}"
            for kind, column in (("tau", f"tau{name}"), ("r", f"r{name}")):
                shade = color if kind == "tau" else f"{color}!70!black"
                layers[f"{kind}-{objective}"].append(
                    rf"\addplot[{shade},{styles[f'{kind}-{objective}']},forget plot] table[x=e,y={column}]{source};")
                if objective == "ivcgpa":
                    values = columns[-2] if kind == "tau" else columns[-1]
                    marks += list(mark_points(data["e_grid"], values, view, phase=0.0 if kind == "tau" else 0.5))
                    mark_names += [f"e{column}", column]
                    layers[f"{kind}-{objective}"].append(
                        rf"\addplot[{shade},{mark_styles[kind]},forget plot] table[x=e{column},y={column}]"
                        rf"{{\familydata/shapes_marks_{family}.dat}};")
            if objective == "ivcgpa":
                fills += pooled_fill(data, view)
            e_grid = data["e_grid"]
    np.savetxt(out / f"shapes_{family}.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(names), comments="", fmt="%.5g")
    length = max([len(m) for m in marks] + [1])
    padded_marks = [np.concatenate([m, np.full(length - len(m), np.nan)]) for m in marks]
    if padded_marks:
        np.savetxt(out / f"shapes_marks_{family}.dat", np.column_stack(padded_marks),
                   header=" ".join(mark_names), comments="", fmt="%.5g")
    (out / f"shapes_{family}.tex").write_text("\n".join(sum(layers.values(), [])) + "\n")
    (out / f"shapes_fill_{family}.tex").write_text("\n".join(fills) + "\n")


def main(run, out):
    out.mkdir(parents=True, exist_ok=True)
    limits = []
    write_table(run, out)

    main_table = shares(run, MAIN_FAMILY)
    with open(out / "share_k1.dat", "w") as file:
        file.write("n tau0 cutoff vcgpa ivcgpa bound\n")
        for (n, k), row in sorted(main_table.items()):
            if k == 1:
                file.write(" ".join([str(n)] + [percent(row.get(c)) for c in
                                                ("tau0", "cutoff", "vcgpa", "ivcgpa", "bound")]) + "\n")
    for k in (1, 2, 4, 8):
        with open(out / f"share_ratio_k{k}.dat", "w") as file:
            file.write("ratio tau0 cutoff vcgpa ivcgpa bound\n")
            for (n, kk), row in sorted(main_table.items()):
                if kk == k and n > k:
                    file.write(" ".join([f"{n // k}"] + [percent(row.get(c)) for c in
                                                       ("tau0", "cutoff", "vcgpa", "ivcgpa", "bound")]) + "\n")

    for family, macro in FAMILIES.items():
        types = load_types(run, family)
        view = background(types, out / f"bg_{family}.png", family)
        limits.append(limit_macros(macro, view))
        columns, e_grid = [], None
        for n in SHAPE_BIDDERS:
            data = load_data(run / family / "z1" / f"n{n}-k1" / "vcgpa_nodes_br")
            e_grid = data["e_grid"]
            columns.append(shown_threshold(data, view))
        np.savetxt(out / f"tau_{family}.dat", np.column_stack([e_grid] + columns),
                   header="e " + " ".join(f"n{n}" for n in SHAPE_BIDDERS), comments="", fmt="%.5g")
        write_shapes(run, family, view, out)
        if family in MAP_FAMILIES:
            n, k = MAP_MARKET
            data = load_data(run / family / "z1" / f"n{n}-k{k}" / "vcgpa_nodes_br")
            mview = map_view(types, family)
            limits.append(limit_macros("Map" + macro, mview, image=False))
            marginal_background(data, mview, out / f"mc_{family}.png")
            np.savetxt(out / f"tau_mc_{family}.dat", np.column_stack([data["e_grid"], shown_threshold(data, mview)]),
                       header="e tau", comments="", fmt="%.5g")
        if family == MAIN_FAMILY:
            columns, names = [], []
            for (n, k) in sorted(main_table, key=lambda m: (m[0] / m[1], m[1])):
                if k < n:
                    data = load_data(run / family / "z1" / f"n{n}-k{k}" / "vcgpa_nodes_br")
                    columns.append(shown_threshold(data, view))
                    names.append(f"n{n}k{k}")
            np.savetxt(out / "tau_markets.dat", np.column_stack([data["e_grid"]] + columns),
                       header="e " + " ".join(names), comments="", fmt="%.5g")

    # IVCG-PA type-plane figure, whole support: τ is drawn only where it is inside the view
    # (column tau, nan above it, for a line with jumps); taufill clips it to the top of the view
    # for the abstention region.
    leaf = run / MAIN_FAMILY / "z1" / "n4-k1" / "ivcgpa_nodes_br"
    data, types = load_data(leaf), load_types(run, MAIN_FAMILY)
    view = background(types, out / "bg_types_hm.png", view=figures.type_view(types, "linear"))
    limits.append(limit_macros("Types", view))
    tau_fill = shown_threshold(data, view, clip_above=True)
    raw = np.asarray(data["tau_grid"], dtype=float)
    tau = np.where(np.isfinite(raw) & (raw <= float(view["v_limits"][1])), tau_fill, np.nan)
    penalty = np.minimum(np.asarray(data["penalty_ivcg_pa"], dtype=float), float(view["v_limits"][1]))
    np.savetxt(out / "types_hm.dat", np.column_stack([data["e_grid"], tau, tau_fill, penalty]),
               header="e tau taufill penalty", comments="", fmt="%.5g")
    pooled = figures._visible_pooled_intervals(data, view)
    e_low, e_high = (float(x) for x in view["e_limits"])
    (out / "pooled_hm.tex").write_text("".join(
        rf"\fill[regionqty] (axis cs:{e_low:.4f},{low:.5g}) rectangle (axis cs:{e_high:.4f},{high:.5g});" + "\n"
        for low, high in pooled))
    (out / "limits.tex").write_text("\n".join(limits) + "\n")

    import family_figures_data
    family_figures_data.main(out, [run])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
