"""Tables, pgfplots data and copied run figures for the empirical supplement (additional
results of the WEB27 paper), from the synthetic run and the shapes run.

    python scripts/empirical_supplement_data.py SYNTHETIC_RUN SHAPES_RUN OUT_DIR

Writes OUT_DIR/data/ (tables *.tex, curves *.dat, type backgrounds bg_*.png, axis limits
limits.tex) and copies selected run figures to OUT_DIR/figures/. OUT_DIR/data/ also receives
everything web27_paper_data.py writes for the submission, so the supplement can show every
figure of the paper; the supplement's own tables replace the paper's of the same name.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import web27_paper_data  # noqa: E402
from web27_paper_data import (FAMILIES, background, label, load_data, load_types,  # noqa: E402
                              percent, shown_threshold)

DATA, FIGURES = Path("data"), Path("figures")  # set in main
SHAPE_BIDDERS = [2, 4, 8, 16]
FOOTER = "\\bottomrule\n\\end{tabular}\n"

# Run figures copied as they are: (run, path in the run, name in figures/).
COPIED = [
    ("synthetic", "harmful-middle/z1/n8-k1/vcgpa_nodes_br/2_expected-outcomes.pdf", "outcomes_n8k1.pdf"),
    ("synthetic", "harmful-middle/z1/n8-k1/vcgpa_nodes_br/3_outcome-distributions.pdf", "distributions_n8k1.pdf"),
    ("synthetic", "harmful-middle/z1/n8-k1/vcgpa_nodes_br/6_marginal-contribution.pdf", "marginal_n8k1.pdf"),
    ("shapes", "two-modes/z1/n8-k1/vcgpa_nodes_br/6_marginal-contribution.pdf", "marginal_two-modes_n8k1.pdf"),
    ("synthetic", "harmful-middle/z1/n4-k1/ivcgpa_nodes_br/5_ironing.pdf", "ironing_n4k1.pdf"),
    ("synthetic", "harmful-middle/z1/compare/tau-by-objective_n4-k1_nodes_br.pdf", "tau_by_objective_n4k1.pdf"),
    ("synthetic", "harmful-middle/z1/compare/tau-by-goods_n32_vcgpa_nodes_br.pdf", "tau_by_goods_n32.pdf"),
    ("synthetic", "harmful-middle/z1/compare/tau-by-goods-at-fixed-ratio_nk4_vcgpa_nodes_br.pdf", "tau_by_goods_nk4.pdf"),
    ("synthetic", "harmful-middle/z1/n32-k8/vcgpa_nodes_br/search/convergence.pdf", "convergence_n32k8.pdf"),
]


def keyed(run):
    summary = pd.read_csv(run / "summary.csv")
    summary["key"] = summary.apply(label, axis=1)
    return summary


def cell(summary, n, k, key, column):
    rows = summary[(summary.num_bidders == n) & (summary.num_goods == k) & (summary.key == key)]
    return rows[column].iloc[0] if len(rows) else None


def synthetic_tables(run):
    summary, gaps = keyed(run), pd.read_csv(run / "gap-decomposition.csv")
    markets = sorted(set(zip(summary.num_bidders, summary.num_goods)), key=lambda m: (m[1], m[0]))

    with open(DATA / "table_synthetic.tex", "w") as file:
        file.write(r"""\begin{tabular}{@{}rrrrrrrrrrr@{}}
\toprule
 & & \multicolumn{5}{c}{VCG-PA with $\tau$} & \multicolumn{2}{c}{IVCG-PA with $\tau$} & relax. & part.\\
\cmidrule(lr){3-7}\cmidrule(lr){8-9}
$n$ & $k$ & $\tau^0$ & $\tau^*$ & relax.\ $\tau$ & cutoff & optimized & cutoff & optimized & bound & gap\\
\midrule
""")
        for n, k in markets:
            nodes = gaps[(gaps.num_bidders == n) & (gaps.num_goods == k) & (gaps.threshold_class == "nodes")]
            vcgpa = cell(summary, n, k, "vcg_pa_nodes", "share_vcg_pa")
            bound = nodes.relaxation_minus_opt_vcg_share.iloc[0] if len(nodes) else np.nan
            values = [cell(summary, n, k, "zero-welfare", "share_vcg_pa"),
                      cell(summary, n, k, "ex-ante", "share_vcg_pa"),
                      cell(summary, n, k, "relaxation", "share_vcg_pa"),
                      cell(summary, n, k, "vcg_pa_cutoff", "share_vcg_pa"), vcgpa,
                      cell(summary, n, k, "ivcg_pa_cutoff", "share_ivcg_pa"),
                      cell(summary, n, k, "ivcg_pa_nodes", "share_ivcg_pa"),
                      vcgpa + bound if np.isfinite(bound) else None]
            gap = cell(summary, n, k, "vcg_pa_nodes", "participation_gap_share_vcg_pa")
            file.write(f"{n} & {k} & " + " & ".join(percent(v) for v in values)
                       + f" & {100 * gap:.2f}" + r" \\" + "\n")
        file.write(FOOTER)

    with open(DATA / "table_gaps.tex", "w") as file:
        file.write(r"""\begin{tabular}{@{}rrrrrrrrr@{}}
\toprule
$n$ & $k$ & $\mathrm{FB}-\mathrm{OPT}$ & $\mathrm{OPT}-\mathrm{OPT}^{\mathrm{VCG}}$ & cutoff & $\tau^0$ & $\tau^*$ & relax.\ $\tau$ & $J-\mathrm{OPT}^{\mathrm{VCG}}$\\
\midrule
""")
        for n, k in markets:
            part = gaps[(gaps.num_bidders == n) & (gaps.num_goods == k)].set_index("threshold_class")
            first = part.iloc[0]
            values = [first.fb_minus_opt_share, first.opt_minus_opt_vcg_share]
            values += [part.opt_vcg_minus_class_share.get(c) for c in
                       ("cutoff", "tau-zero-welfare", "tau-ex-ante", "tau-relaxation")]
            values.append(part.relaxation_minus_opt_vcg_share.get("nodes"))
            file.write(f"{n} & {k} & " + " & ".join(percent(v) for v in values) + r" \\" + "\n")
        file.write(FOOTER)

    with open(DATA / "table_revenue.tex", "w") as file:
        file.write(r"""\begin{tabular}{@{}rrrrrrrr@{}}
\toprule
 & & VCG & \multicolumn{5}{c}{VCG-PA with optimized $\tau$}\\
\cmidrule(lr){4-8}
$n$ & $k$ & revenue & revenue & penalties & rev.\,+\,pen. & net utility & participants\\
\midrule
""")
        for n, k in markets:
            if k >= n:
                continue
            rows = summary[(summary.num_bidders == n) & (summary.num_goods == k) & (summary.key == "vcg_pa_nodes")]
            r = rows.iloc[0]
            base = r.revenue_vcg
            revenue = base + r.revenue_change_vcg_pa
            values = [f"{base:.2f}", percent(revenue / base), percent(r.penalty_paid_vcg_pa / base),
                      percent((revenue + r.penalty_paid_vcg_pa) / base),
                      percent(r.bidder_utility_vcg_pa / r.bidder_utility_vcg),
                      percent(r.num_participants_vcg_pa / n)]
            file.write(f"{n} & {k} & " + " & ".join(values) + r" \\" + "\n")
        file.write(FOOTER)



def shapes_table(run):
    summary, gaps = keyed(run), pd.read_csv(run / "gap-decomposition.csv")
    with open(DATA / "table_shapes.tex", "w") as file:
        file.write(r"""\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
distribution & $n$ & $k$ & $\tau^0$ & cutoff & affine & optimized & relax.\ bound & IVCG-PA & gap\\
\midrule
""")
        for family, name in web27_paper_data.FAMILY_NAMES.items():
            part = summary[summary.distribution == family]
            markets = sorted(set(zip(part.num_bidders, part.num_goods)), key=lambda m: (m[1] > 1, m[1], m[0]))
            for index, (n, k) in enumerate(markets):
                c = part[(part.num_bidders == n) & (part.num_goods == k)].set_index("key")
                nodes = gaps[(gaps.distribution == family) & (gaps.num_bidders == n)
                             & (gaps.num_goods == k) & (gaps.threshold_class == "nodes")]
                optimized = c.share_vcg_pa.get("vcg_pa_nodes")
                bound = optimized + nodes.relaxation_minus_opt_vcg_share.iloc[0]
                values = [c.share_vcg_pa.get("zero-welfare"), c.share_vcg_pa.get("vcg_pa_cutoff"),
                          c.share_vcg_pa.get("vcg_pa_poly1"), optimized, bound,
                          c.share_ivcg_pa.get("vcg_pa_nodes")]
                gap = c.participation_gap_share_vcg_pa.get("vcg_pa_nodes")
                first = rf"\multirow{{{len(markets)}}}{{*}}{{{name}}}" if index == 0 else ""
                file.write(f"{first} & {n} & {k} & " + " & ".join(percent(v) for v in values)
                           + f" & {100 * gap:.2f}" + r" \\" + "\n")
            if family != "two-modes":
                file.write("\\midrule\n")
        file.write(FOOTER)
    for family in web27_paper_data.FAMILY_NAMES:
        part = summary[(summary.distribution == family) & (summary.num_goods == 1)]
        with open(DATA / f"share_family_{family}.dat", "w") as file:
            file.write("n tau0 cutoff line nodes\n")
            for n in SHAPE_BIDDERS:
                c = part[part.num_bidders == n].set_index("key").share_vcg_pa
                file.write(" ".join([str(n)] + [percent(c.get(key)) for key in
                                                ("zero-welfare", "vcg_pa_cutoff", "vcg_pa_poly1", "vcg_pa_nodes")]) + "\n")


def curves(synthetic, shapes):
    limits = []
    for run, families in ((synthetic, ["harmful-middle"]), (shapes, list(FAMILIES))):
        for family in families:
            types = load_types(run, family)
            view = background(types, DATA / f"bg_{family}.png", family)
            macro = FAMILIES[family]
            e_low, e_high = (float(x) for x in view["e_limits"])
            v_low, v_high = (float(x) for x in view["v_limits"])
            limits.append(rf"\def\emin{macro}{{{e_low:.4f}}}\def\emax{macro}{{{e_high:.4f}}}"
                          rf"\def\vmin{macro}{{{v_low:.5g}}}\def\vmax{macro}{{{v_high:.4f}}}")
            if run is shapes:
                write_family_curves(shapes, family, view)
    (DATA / "limits.tex").write_text("\n".join(limits) + "\n")

    view = load_types(synthetic, "harmful-middle")
    from PA_auctions import figures
    view = figures.default_view(view)
    leaves = {"tau0": "tau-zero-welfare", "exante": "tau-ex-ante", "relaxation": "tau-relaxation",
              "cutoff": "vcgpa_cutoff_ga", "vcgpa": "vcgpa_nodes_br", "ivcgpa": "ivcgpa_nodes_br"}
    columns, e_grid = [], None
    for leaf in leaves.values():
        data = load_data(synthetic / "harmful-middle" / "z1" / "n8-k1" / leaf)
        e_grid = data["e_grid"]
        columns.append(shown_threshold(data, view))
    np.savetxt(DATA / "tau_compare_n8k1.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(leaves), comments="", fmt="%.5g")


def write_family_curves(shapes, family, view):
    leaves = {"tau0": "tau-zero-welfare", "cutoff": "vcgpa_cutoff_ga", "line": "vcgpa_poly1_ga",
              "nodes": "vcgpa_nodes_br"}
    columns, e_grid = [], None
    for leaf in leaves.values():
        data = load_data(shapes / family / "z1" / "n8-k1" / leaf)
        e_grid = data["e_grid"]
        columns.append(shown_threshold(data, view))
    np.savetxt(DATA / f"forms_{family}.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(leaves), comments="", fmt="%.5g")
    columns = []
    for n in SHAPE_BIDDERS:
        data = load_data(shapes / family / "z1" / f"n{n}-k1" / "vcgpa_nodes_br")
        columns.append(np.asarray(data["penalty_vcg_pa"], dtype=float))
    np.savetxt(DATA / f"penalty_{family}.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(f"n{n}" for n in SHAPE_BIDDERS), comments="", fmt="%.5g")


def _harm_parts(summary, n, k, key):
    """Valuation lost and harm avoided relative to VCG, as % of the first-best gain, for VCG-PA
    with threshold function key, and for the first best."""
    row = summary[(summary.num_bidders == n) & (summary.num_goods == k) & (summary.key == key)].iloc[0]
    gain = row.welfare_first_best - row.welfare_vcg
    parts = {}
    for mechanism in ("vcg_pa", "first_best"):
        lost = 100 * (row.valuation_welfare_vcg - row[f"valuation_welfare_{mechanism}"]) / gain
        avoided = 100 * (row[f"externality_welfare_{mechanism}"] - row.externality_welfare_vcg) / gain
        parts[mechanism] = (lost, avoided)
    return parts


HARM_FORMS = [("zero-welfare", "o"), ("vcg_pa_cutoff", "triangle*"), ("vcg_pa_poly1", "square*"),
              ("vcg_pa_nodes", "*")]


def _harm_commands(summary, markets, colors):
    """pgfplots commands: per market, a thin path through the forms in HARM_FORMS (those present)
    and the first best, with one marker per form."""
    lines = []
    for (n, k), color in zip(markets, colors):
        points = []
        for key, mark in HARM_FORMS:
            if (summary[(summary.num_bidders == n) & (summary.num_goods == k)].key == key).any():
                points.append((_harm_parts(summary, n, k, key)["vcg_pa"], mark))
        best = _harm_parts(summary, n, k, "vcg_pa_nodes")["first_best"]
        path = " ".join(f"({x:.2f},{y:.2f})" for (x, y), _ in points) + f" ({best[0]:.2f},{best[1]:.2f})"
        lines.append(rf"\addplot[{color},thin,forget plot] coordinates {{{path}}};")
        for (x, y), mark in points:
            lines.append(rf"\addplot[{color},only marks,mark={mark},mark size=1.8pt,forget plot] coordinates {{({x:.2f},{y:.2f})}};")
        lines.append(rf"\addplot[{color},only marks,mark=star,mark size=2.6pt,forget plot] coordinates {{({best[0]:.2f},{best[1]:.2f})}};")
    return "\n".join(lines) + "\n"


def harm_figures(synthetic, shapes):
    summary = keyed(synthetic)
    markets = [(n, 1) for n in (2, 4, 8, 16, 32)]
    (DATA / "harm_markets.tex").write_text(
        _harm_commands(summary, markets, ["fiveA", "fiveB", "fiveC", "fiveD", "fiveE"]))
    summary = keyed(shapes)
    families = list(web27_paper_data.FAMILY_NAMES)
    parts = []
    for family, color in zip(families, ["seriesA", "seriesB", "seriesC", "seriesD"]):
        part = summary[summary.distribution == family]
        parts.append(_harm_commands(part, [(8, 1)], [color]))
    (DATA / "harm_families.tex").write_text("".join(parts))


def main(synthetic, shapes, out):
    global DATA, FIGURES
    DATA, FIGURES = out / "data", out / "figures"
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    web27_paper_data.main(synthetic, shapes, DATA)
    synthetic_tables(synthetic)
    shapes_table(shapes)
    curves(synthetic, shapes)
    harm_figures(synthetic, shapes)
    runs = {"synthetic": synthetic, "shapes": shapes}
    for run, path, name in COPIED:
        shutil.copyfile(runs[run] / path, FIGURES / name)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
