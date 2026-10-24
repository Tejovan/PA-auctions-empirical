"""Data for the empirical section of the WEB27 submission: pgfplots tables, type-distribution
background images and LaTeX table rows, from the synthetic run and the shapes run.

    python scripts/web27_paper_data.py SYNTHETIC_RUN SHAPES_RUN OUT_DIR

Writes to OUT_DIR:
  share_k1.dat            share of the first-best gain (%) against n at k = 1 (synthetic run)
  table_main.tex          tabular of the main table (selected markets, synthetic run)
  table_shapes.tex        tabular of the appendix table (all families and markets, shapes run)
  tau_<family>.dat        VCG-PA-optimized τ(e) for n ∈ {2, 4, 8, 16}, k = 1 (shapes run)
  bg_<family>.png         type density of each family, linear valuation axis (axes limits in limits.tex)
  types_hm.dat            τ(e) and IVCG-PA penalty r(e), harmful_middle, n = 4, k = 1, IVCG-PA
  pooled_hm.tex           bands of the bids pooled by IVCG-PA there (pgfplots fill commands)
  limits.tex              axis limits of every type-plane panel as macros
  share_ratio_k<k>.dat    share of the first-best gain (%) against n/k for k goods (synthetic run)
  tau_markets.dat         VCG-PA-optimized τ(e) of every market with k < n, harmful-middle (synthetic run)
  mc_<family>.png         marginal contribution M(e, v) of VCG-PA at its optimized τ, n = 8, k = 1
                          (PuOr, symmetric logarithmic scale, dashed M = 0, edge of the type density)
  tau_mc_<family>.dat     the optimized τ(e) drawn over mc_<family>.png
  forms_two-modes.dat     VCG-PA τ(e) of τ⁰, best cutoff, best line and optimized, n = 8, k = 1 (shapes run)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PA_auctions import figures  # noqa: E402

FAMILIES = {"independent": "Independent", "harmful-middle": "HarmfulMiddle",
            "heavy-tail": "HeavyTail", "two-modes": "TwoModes"}
FAMILY_NAMES = {"independent": "independent", "harmful-middle": "harmful-middle",
                "heavy-tail": "heavy-tail", "two-modes": "two-modes"}
MAP_LEAVES = {"harmful-middle": ("synthetic", "harmful-middle/z1/n8-k1/vcgpa_nodes_br"),
              "two-modes": ("shapes", "two-modes/z1/n8-k1/vcgpa_nodes_br")}
FORM_LEAVES = {"tau0": "tau-zero-welfare", "cutoff": "vcgpa_cutoff_ga", "line": "vcgpa_poly1_ga",
               "nodes": "vcgpa_nodes_br"}
MAIN_MARKETS = [(2, 1), (8, 1), (32, 1), (8, 2), (16, 4), (32, 8)]
SHAPE_BIDDERS = [2, 4, 8, 16]
MAIN_HEADER = r"""\begin{tabular}{@{}rrrrrrr@{}}
\toprule
 & & \multicolumn{4}{c}{VCG-PA with $\tau$} & IVCG-PA\\
\cmidrule(lr){3-6}
$n$ & $k$ & $\tau^0$ & best cutoff & optimized & relax.\ bound & optimized\\
\midrule
"""
SHAPES_HEADER = r"""\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & & & & \multicolumn{2}{c}{best} & \multicolumn{2}{c}{optimized}\\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
distribution & $n$ & $k$ & $\tau^0$ & cutoff & affine & share & gap\\
\midrule
"""
FOOTER = "\\bottomrule\n\\end{tabular}\n"


def label(row):
    return row.tau_label if isinstance(row.tau_label, str) else f"{row.objective}_{row.threshold_form}"


def shares(run):
    """Share of the first-best gain (%) by market and τ: τ⁰, cutoff, poly1, VCG-PA and IVCG-PA
    optimized, and the relaxation bound (VCG-PA optimum + relaxation gap)."""
    summary = pd.read_csv(run / "summary.csv")
    summary["key"] = summary.apply(label, axis=1)
    gaps = pd.read_csv(run / "gap-decomposition.csv")
    table = {}
    for (n, k), group in summary.groupby(["num_bidders", "num_goods"]):
        row = {}
        for _, r in group.iterrows():
            if r.key == "zero-welfare":
                row["tau0"] = r.share_vcg_pa
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
# the 2.5%–97.5% externality quantiles; heavy-tail also cut at the 80% valuation quantile.
VIEW_OPTIONS = {"independent": {"e_levels": (0.025, 0.975)},
                "heavy-tail": {"e_levels": (0.025, 0.975), "v_top": 0.80}}


def _quantiles(axis, weights, levels):
    cumulative = np.cumsum(weights) / weights.sum()
    return [float(np.interp(level, cumulative, axis)) for level in levels]


def zoomed_view(types, scale="linear", e_levels=(0.005, 0.995), v_top=None):
    """The view of a type-plane panel, zoomed to quantiles of the marginal distributions of e and
    v (from the background density, valuations × externalities), with a 3% margin; v_top
    overrides the upper valuation quantile of a linear view."""
    view = figures.type_view(types, scale)
    density = np.asarray(types["background_density"], dtype=float)
    e_low, e_high = _quantiles(np.asarray(types["background_e"]), density.sum(axis=0), e_levels)
    levels = (0.005, 0.995) if scale == "log" else (0.005, v_top or 0.99)
    v_low, v_high = _quantiles(np.asarray(types["background_v"]), density.sum(axis=1), levels)
    e_margin = 0.03 * (e_high - e_low)
    view["e_limits"] = np.array([e_low - e_margin, e_high + e_margin])
    if scale == "log":
        view["v_limits"] = np.array([max(v_low, 1e-3), v_high * 1.05])
    else:
        v_margin = 0.03 * (v_high - v_low)
        view["v_limits"] = np.array([v_low - v_margin, v_high + v_margin])
    return view


def linear_view(types, family=None):
    """The view of a type-plane panel: linear valuation axis over the whole support, or the
    zoomed view of VIEW_OPTIONS for that family."""
    if family in VIEW_OPTIONS:
        return zoomed_view(types, **VIEW_OPTIONS[family])
    return figures.type_view(types, "linear")


def background(types, path, family=None):
    """The type density alone, filling the image, with the limits of its view."""
    view = linear_view(types, family)
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    figures.draw_types(ax, view)
    ax.set_axis_off()
    fig.savefig(path, dpi=200, transparent=True)
    figures.plt.close(fig)
    return view


def marginal_background(data, types, path):
    """M(e, v) of VCG-PA on the view of the type plane, filling the image: PuOr on a symmetric
    logarithmic scale as in the run figures, the zero contour dashed and the edge of the type density."""
    view = linear_view(types)
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


def shown_threshold(data, view, clip_above=False):
    """τ(e) inside the view: nan where τ is above the top of the view or excluded (τ = v̄⁺),
    so no line is drawn there; clip_above=True puts those values at the top instead (for
    filling the abstention region)."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    low, high = (float(x) for x in view["v_limits"])
    if clip_above:
        return np.where(np.isfinite(tau), np.clip(tau, low, high), high)
    return np.where(np.isfinite(tau) & (tau <= high), np.maximum(tau, low), np.nan)


def main(synthetic, shapes, out):
    out.mkdir(parents=True, exist_ok=True)
    limits = []

    main_table = shares(synthetic)
    with open(out / "share_k1.dat", "w") as file:
        file.write("n tau0 cutoff vcgpa ivcgpa bound\n")
        for (n, k), row in sorted(main_table.items()):
            if k == 1:
                file.write(" ".join([str(n)] + [percent(row.get(c)) for c in
                                                ("tau0", "cutoff", "vcgpa", "ivcgpa", "bound")]) + "\n")
    with open(out / "table_main.tex", "w") as file:
        file.write(MAIN_HEADER)
        for n, k in MAIN_MARKETS:
            row = main_table[(n, k)]
            file.write(f"{n} & {k} & " + " & ".join(percent(row.get(c)) for c in
                       ("tau0", "cutoff", "vcgpa", "bound", "ivcgpa")) + r" \\" + "\n")
        file.write(FOOTER)

    with open(out / "table_shapes.tex", "w") as file:
        file.write(SHAPES_HEADER)
        summary = pd.read_csv(shapes / "summary.csv")
        summary["key"] = summary.apply(label, axis=1)
        for family, name in FAMILY_NAMES.items():
            part = summary[summary.distribution == family]
            markets = sorted(set(zip(part.num_bidders, part.num_goods)), key=lambda m: (m[1] > 1, m[1], m[0]))
            for index, (n, k) in enumerate(markets):
                cell = part[(part.num_bidders == n) & (part.num_goods == k)].set_index("key")
                values = [cell.share_vcg_pa.get(key) for key in
                          ("zero-welfare", "vcg_pa_cutoff", "vcg_pa_poly1", "vcg_pa_nodes")]
                gap = cell.participation_gap_share_vcg_pa.get("vcg_pa_nodes")
                first = rf"\multirow{{{len(markets)}}}{{*}}{{{name}}}" if index == 0 else ""
                file.write(f"{first} & {n} & {k} & " + " & ".join(percent(v) for v in values)
                           + f" & {100 * gap:.2f}" + r" \\" + "\n")
            file.write(r"\midrule" + "\n" if family != "two-modes" else "")
        file.write(FOOTER)

    for family, macro in FAMILIES.items():
        types = load_types(shapes, family)
        view = background(types, out / f"bg_{family}.png", family)
        e_low, e_high = (float(x) for x in view["e_limits"])
        v_low, v_high = (float(x) for x in view["v_limits"])
        limits.append(rf"\def\emin{macro}{{{e_low:.4f}}}\def\emax{macro}{{{e_high:.4f}}}"
                      rf"\def\vmin{macro}{{{v_low:.5g}}}\def\vmax{macro}{{{v_high:.4f}}}")
        columns, e_grid = [], None
        for n in SHAPE_BIDDERS:
            data = load_data(shapes / family / "z1" / f"n{n}-k1" / "vcgpa_nodes_br")
            e_grid = data["e_grid"]
            columns.append(shown_threshold(data, view))
        np.savetxt(out / f"tau_{family}.dat", np.column_stack([e_grid] + columns),
                   header="e " + " ".join(f"n{n}" for n in SHAPE_BIDDERS), comments="", fmt="%.5g")

    # Type-plane figure: τ is drawn only where it is inside the view (column tau, nan above it,
    # for a line with jumps); taufill clips it to the top of the view for the abstention region.
    leaf = synthetic / "harmful-middle" / "z1" / "n4-k1" / "ivcgpa_nodes_br"
    data, types = load_data(leaf), load_types(synthetic, "harmful-middle")
    view = linear_view(types)
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
    for k in (1, 2, 4, 8):
        with open(out / f"share_ratio_k{k}.dat", "w") as file:
            file.write("ratio tau0 cutoff vcgpa ivcgpa bound\n")
            for (n, kk), row in sorted(main_table.items()):
                if kk == k and n > k:
                    file.write(" ".join([f"{n // k}"] + [percent(row.get(c)) for c in
                                                       ("tau0", "cutoff", "vcgpa", "ivcgpa", "bound")]) + "\n")

    view = linear_view(types)
    columns, names = [], []
    for (n, k) in sorted(main_table, key=lambda m: (m[0] / m[1], m[1])):
        if k < n:
            data = load_data(synthetic / "harmful-middle" / "z1" / f"n{n}-k{k}" / "vcgpa_nodes_br")
            columns.append(shown_threshold(data, view))
            names.append(f"n{n}k{k}")
    np.savetxt(out / "tau_markets.dat", np.column_stack([data["e_grid"]] + columns),
               header="e " + " ".join(names), comments="", fmt="%.5g")

    runs = {"synthetic": synthetic, "shapes": shapes}
    for family, (run, leaf) in MAP_LEAVES.items():
        data = load_data(runs[run] / leaf)
        view = marginal_background(data, load_types(runs[run], family), out / f"mc_{family}.png")
        np.savetxt(out / f"tau_mc_{family}.dat", np.column_stack([data["e_grid"], shown_threshold(data, view)]),
                   header="e tau", comments="", fmt="%.5g")

    view = linear_view(load_types(shapes, "two-modes"))
    columns = []
    for leaf in FORM_LEAVES.values():
        data = load_data(shapes / "two-modes" / "z1" / "n8-k1" / leaf)
        columns.append(shown_threshold(data, view))
    np.savetxt(out / "forms_two-modes.dat", np.column_stack([data["e_grid"]] + columns),
               header="e " + " ".join(FORM_LEAVES), comments="", fmt="%.5g")
    (out / "limits.tex").write_text("\n".join(limits) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
