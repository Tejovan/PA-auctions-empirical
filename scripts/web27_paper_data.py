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
  shapes_<family>.dat/.tex      τ(e) and penalty r(e) of the SHAPE_CURVES designs and pgfplots
                          commands drawing them (color = n/k as in family_tau_*.tex; τ solid
                          below r dashed)
  shapes_fill_<family>.tex      the band of bids pooled by an IVCG-PA design there, at the
                          participating types only (drawn below the type density)
  mc_<family>.png         marginal contribution M(e, v) of VCG-PA at its optimized τ, n = 8, k = 1
                          (PuOr, symmetric logarithmic scale, dashed M = 0, edge of the type density)
  tau_mc_<family>.dat     the optimized τ(e) drawn over mc_<family>.png
  limits.tex              axis limits of every type-plane panel as macros
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
# Designs of the τ-and-penalty figure per distribution: (objective, n, k); an IVCG-PA design
# also gets the band of valuations it pools. Colors by n/k as in family_figures_data.
SHAPE_CURVES = {"independent": [("vcgpa", 8, 7), ("vcgpa", 2, 1), ("vcgpa", 32, 1)],
                "harmful-middle": [("vcgpa", 8, 7), ("vcgpa", 2, 1), ("ivcgpa", 32, 1)],
                "heavy-tail": [("vcgpa", 4, 3), ("vcgpa", 16, 4), ("vcgpa", 32, 1)],
                "harm-rises": [("vcgpa", 16, 14), ("vcgpa", 8, 2), ("vcgpa", 32, 1)]}
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
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    figures.draw_types(ax, view)
    ax.set_axis_off()
    fig.savefig(path, dpi=200, transparent=True)
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


def limit_macros(macro, view):
    e_low, e_high = (float(x) for x in view["e_limits"])
    v_low, v_high = (float(x) for x in view["v_limits"])
    return (rf"\def\emin{macro}{{{e_low:.4f}}}\def\emax{macro}{{{e_high:.4f}}}"
            rf"\def\vmin{macro}{{{v_low:.5g}}}\def\vmax{macro}{{{v_high:.4f}}}")


def shown_threshold(data, view, clip_above=False):
    """τ(e) inside the view: nan where τ is above the top of the view or excluded (τ = v̄⁺),
    so no line is drawn there; clip_above=True puts those values at the top instead (for
    filling the abstention region)."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    low, high = (float(x) for x in view["v_limits"])
    if clip_above:
        return np.where(np.isfinite(tau), np.clip(tau, low, high), high)
    return np.where(np.isfinite(tau) & (tau <= high), np.maximum(tau, low), np.nan)


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
    """pgfplots fill commands for the bands of bids pooled by an IVCG-PA design, only at the
    participating types (v ≥ τ(e)), as in the IVCG-PA type-plane figure."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    e_grid = np.asarray(data["e_grid"], dtype=float)
    e_low, e_high = (float(x) for x in view["e_limits"])
    inside = (e_grid >= e_low) & (e_grid <= e_high)
    commands = []
    for low, high in figures._visible_pooled_intervals(data, view):
        bottom = np.where(np.isfinite(tau), np.clip(tau, low, high), high)[inside]
        points = [(e, b) for e, b in zip(e_grid[inside], bottom)]
        points = [(e_low, points[0][1])] + points + [(e_high, points[-1][1]), (e_high, high), (e_low, high)]
        commands.append(r"\addplot[draw=none,fill=regionqty,forget plot] coordinates {"
                        + " ".join(f"({e:.4f},{b:.5g})" for e, b in points) + "} -- cycle;")
    return commands


def write_shapes(run, family, view, out):
    """τ and penalty of the SHAPE_CURVES designs of one distribution, and pgfplots commands:
    shapes_<family>.tex draws τ (solid) below the penalties (dashed); shapes_fill_<family>.tex
    the band of bids pooled by an IVCG-PA design, drawn below the type density."""
    from family_figures_data import RATIO_COLORS
    columns, names, fills, taus, penalties = [], [], [], [], []
    for objective, n, k in SHAPE_CURVES[family]:
        data = load_data(run / family / "z1" / f"n{n}-k{k}" / f"{objective}_nodes_br")
        mechanism = "ivcg_pa" if objective == "ivcgpa" else "vcg_pa"
        name = f"{objective}n{n}k{k}"
        tau = shown_threshold(data, view)
        penalty = np.asarray(data[f"penalty_{mechanism}"], dtype=float)
        columns += [tau, np.where(np.isfinite(tau), penalty, np.nan)]
        names += [f"tau{name}", f"r{name}"]
        color = RATIO_COLORS[min(RATIO_COLORS, key=lambda r: abs(r - n / k))]
        source = rf"{{\familydata/shapes_{family}.dat}}"
        taus.append(rf"\addplot[{color},line width=1.4pt,forget plot] table[x=e,y=tau{name}]{source};")
        penalties.append(rf"\addplot[{color},line width=1.0pt,densely dashed,forget plot] table[x=e,y=r{name}]{source};")
        if objective == "ivcgpa":
            fills += pooled_fill(data, view)
        e_grid = data["e_grid"]
    np.savetxt(out / f"shapes_{family}.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(names), comments="", fmt="%.5g")
    (out / f"shapes_{family}.tex").write_text("\n".join(taus + penalties) + "\n")
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
            marginal_background(data, view, out / f"mc_{family}.png")
            np.savetxt(out / f"tau_mc_{family}.dat", np.column_stack([data["e_grid"], shown_threshold(data, view)]),
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
