"""Data for the empirical section of the WEB27 submission: pgfplots tables, type-distribution
background images and LaTeX table rows, from the synthetic run and the shapes run.

    python scripts/web27_paper_data.py SYNTHETIC_RUN SHAPES_RUN OUT_DIR

Writes to OUT_DIR:
  share_k1.dat            share of the first-best gain (%) against n at k = 1 (synthetic run)
  table_main.tex          tabular of the main table (selected markets, synthetic run)
  table_shapes.tex        tabular of the appendix table (all families and markets, shapes run)
  tau_<family>.dat        VCG-PA-optimized τ(e) for n ∈ {2, 4, 8, 16}, k = 1 (shapes run)
  bg_<family>.png         type density of each family (axes limits in limits.tex)
  types_hm.dat            τ(e) and IVCG-PA penalty r(e), harmful_middle, n = 4, k = 1, IVCG-PA
  pooled_hm.tex           bands of the bids pooled by IVCG-PA there (pgfplots fill commands)
  limits.tex              axis limits of every type-plane panel as macros
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PA_auctions import figures  # noqa: E402

FAMILIES = {"independent": "Independent", "harmful-middle": "HarmfulMiddle",
            "heavy-tail": "HeavyTail", "two-modes": "TwoModes"}
FAMILY_NAMES = {"independent": "independent", "harmful-middle": "harmful middle",
                "heavy-tail": "heavy tail", "two-modes": "two modes"}
MAIN_MARKETS = [(2, 1), (8, 1), (32, 1), (8, 2), (16, 4), (32, 8)]
SHAPE_BIDDERS = [2, 4, 8, 16]
MAIN_HEADER = r"""\begin{tabular}{@{}rrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{VCG-PA with $\tau$} & IVCG-PA & relax.\\
\cmidrule(lr){3-5}
$n$ & $k$ & $\tau^0$ & cutoff & optimized & optimized & bound\\
\midrule
"""
SHAPES_HEADER = r"""\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
distribution & $n$ & $k$ & $\tau^0$ & cutoff & line & optimized & gap\\
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


def background(types, path):
    """The type density alone, filling the image, with the limits of its view."""
    view = figures.default_view(types)
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    figures.draw_types(ax, view)
    ax.set_axis_off()
    fig.savefig(path, dpi=200, transparent=True)
    figures.plt.close(fig)
    return view


def shown_threshold(data, view):
    """τ(e) clipped to the view; excluded externalities (τ = v̄⁺) at the top of the view."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    low, high = (float(x) for x in view["v_limits"])
    return np.where(np.isfinite(tau), np.clip(tau, low, high), high)


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
                       ("tau0", "cutoff", "vcgpa", "ivcgpa", "bound")) + r" \\" + "\n")
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
        view = background(types, out / f"bg_{family}.png")
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

    leaf = synthetic / "harmful-middle" / "z1" / "n4-k1" / "ivcgpa_nodes_br"
    data, types = load_data(leaf), load_types(synthetic, "harmful-middle")
    view = figures.default_view(types)
    tau = shown_threshold(data, view)
    penalty = np.minimum(np.asarray(data["penalty_ivcg_pa"], dtype=float), float(view["v_limits"][1]))
    np.savetxt(out / "types_hm.dat", np.column_stack([data["e_grid"], tau, penalty]),
               header="e tau penalty", comments="", fmt="%.5g")
    pooled = figures._visible_pooled_intervals(data, view)
    e_low, e_high = (float(x) for x in view["e_limits"])
    (out / "pooled_hm.tex").write_text("".join(
        rf"\fill[regionqty] (axis cs:{e_low:.4f},{low:.5g}) rectangle (axis cs:{e_high:.4f},{high:.5g});" + "\n"
        for low, high in pooled))
    (out / "limits.tex").write_text("\n".join(limits) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
