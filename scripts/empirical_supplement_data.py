"""Tables, pgfplots data and copied run figures for the empirical supplement (additional
results of the WEB27 paper), from the paper run (configs/paper_run.toml).

    python scripts/empirical_supplement_data.py RUN OUT_DIR

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
from web27_paper_data import label, load_data, load_types, percent, shown_threshold  # noqa: E402

DATA, FIGURES = Path("data"), Path("figures")  # set in main
SHAPE_BIDDERS = web27_paper_data.SHAPE_BIDDERS
MAIN = web27_paper_data.MAIN_FAMILY
FOOTER = "\\bottomrule\n\\end{tabular}\n"

# Run figures copied as they are: (run, path in the run, name in figures/).
COPIED = [
    ("harmful-middle/z1/n8-k1/vcgpa_nodes_br/2_expected-outcomes.pdf", "outcomes_n8k1.pdf"),
    ("harmful-middle/z1/n8-k1/vcgpa_nodes_br/3_outcome-distributions.pdf", "distributions_n8k1.pdf"),
    ("harmful-middle/z1/n8-k1/vcgpa_nodes_br/6_marginal-contribution.pdf", "marginal_n8k1.pdf"),
    ("harm-rises/z1/n8-k1/vcgpa_nodes_br/6_marginal-contribution.pdf", "marginal_harm-rises_n8k1.pdf"),
    ("harmful-middle/z1/n4-k1/ivcgpa_nodes_br/5_ironing.pdf", "ironing_n4k1.pdf"),
    ("harmful-middle/z1/compare/tau-by-objective_n4-k1_nodes_br.pdf", "tau_by_objective_n4k1.pdf"),
    ("harmful-middle/z1/compare/tau-by-goods_n32_vcgpa_nodes_br.pdf", "tau_by_goods_n32.pdf"),
    ("harmful-middle/z1/compare/tau-by-goods-at-fixed-ratio_nk4_vcgpa_nodes_br.pdf", "tau_by_goods_nk4.pdf"),
    ("harmful-middle/z1/n32-k8/vcgpa_nodes_br/search/convergence.pdf", "convergence_n32k8.pdf"),
]


def keyed(run, family):
    summary = pd.read_csv(run / "summary.csv")
    summary = summary[summary.distribution == family].copy()
    summary["key"] = summary.apply(label, axis=1)
    return summary


def family_gaps(run, family):
    gaps = pd.read_csv(run / "gap-decomposition.csv")
    return gaps[gaps.distribution == family]


def cell(summary, n, k, key, column):
    rows = summary[(summary.num_bidders == n) & (summary.num_goods == k) & (summary.key == key)]
    return rows[column].iloc[0] if len(rows) else None


def market_tables(run):
    """One table per distribution: shares of the first-best gain in every market."""
    for family in web27_paper_data.FAMILIES:
        summary, gaps = keyed(run, family), family_gaps(run, family)
        markets = sorted(set(zip(summary.num_bidders, summary.num_goods)), key=lambda m: (m[1], m[0]))
        with open(DATA / f"table_markets_{family}.tex", "w") as file:
            file.write(r"""\begin{tabular}{@{}rrrrrrrrrrrrr@{}}
\toprule
 & & \multicolumn{6}{c}{VCG-PA with $\tau$} & \multicolumn{3}{c}{IVCG-PA with $\tau$} & relax. & part.\\
\cmidrule(lr){3-8}\cmidrule(lr){9-11}
$n$ & $k$ & $\tau^0$ & $\tau^*$ & relax.\ $\tau$ & cutoff & affine & optimized & cutoff & affine & optimized & bound & gap\\
\midrule
""")
            for n, k in markets:
                nodes = gaps[(gaps.num_bidders == n) & (gaps.num_goods == k) & (gaps.threshold_class == "nodes")]
                vcgpa = cell(summary, n, k, "vcg_pa_nodes", "share_vcg_pa")
                bound = nodes.relaxation_minus_opt_vcg_share.iloc[0] if len(nodes) else np.nan
                values = [cell(summary, n, k, "zero-welfare", "share_vcg_pa"),
                          cell(summary, n, k, "ex-ante", "share_vcg_pa"),
                          cell(summary, n, k, "relaxation", "share_vcg_pa"),
                          cell(summary, n, k, "vcg_pa_cutoff", "share_vcg_pa"),
                          cell(summary, n, k, "vcg_pa_poly1", "share_vcg_pa"), vcgpa,
                          cell(summary, n, k, "ivcg_pa_cutoff", "share_ivcg_pa"),
                          cell(summary, n, k, "ivcg_pa_poly1", "share_ivcg_pa"),
                          cell(summary, n, k, "ivcg_pa_nodes", "share_ivcg_pa"),
                          vcgpa + bound if np.isfinite(bound) else None]
                gap = cell(summary, n, k, "vcg_pa_nodes", "participation_gap_share_vcg_pa")
                file.write(f"{n} & {k} & " + " & ".join(percent(v) for v in values)
                           + f" & {100 * gap:.2f}" + r" \\" + "\n")
            file.write(FOOTER)


def main_tables(run):
    summary, gaps = keyed(run, MAIN), family_gaps(run, MAIN)
    markets = sorted(set(zip(summary.num_bidders, summary.num_goods)), key=lambda m: (m[1], m[0]))

    with open(DATA / "table_gaps.tex", "w") as file:
        file.write(r"""\begin{tabular}{@{}rrrrrrrrrr@{}}
\toprule
$n$ & $k$ & $\mathrm{FB}-\mathrm{OPT}$ & $\mathrm{OPT}-\mathrm{OPT}^{\mathrm{VCG}}$ & cutoff & affine & $\tau^0$ & $\tau^*$ & relax.\ $\tau$ & $J-\mathrm{OPT}^{\mathrm{VCG}}$\\
\midrule
""")
        for n, k in markets:
            part = gaps[(gaps.num_bidders == n) & (gaps.num_goods == k)].set_index("threshold_class")
            first = part.iloc[0]
            values = [first.fb_minus_opt_share, first.opt_minus_opt_vcg_share]
            values += [part.opt_vcg_minus_class_share.get(c) for c in
                       ("cutoff", "poly1", "tau-zero-welfare", "tau-ex-ante", "tau-relaxation")]
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



def family_shares(run):
    for family in web27_paper_data.FAMILIES:
        part = keyed(run, family)
        part = part[part.num_goods == 1]
        with open(DATA / f"share_family_{family}.dat", "w") as file:
            file.write("n tau0 cutoff line nodes\n")
            for n in SHAPE_BIDDERS:
                c = part[part.num_bidders == n].set_index("key").share_vcg_pa
                file.write(" ".join([str(n)] + [percent(c.get(key)) for key in
                                                ("zero-welfare", "vcg_pa_cutoff", "vcg_pa_poly1", "vcg_pa_nodes")]) + "\n")


def curves(run):
    """Forms at n = 8, k = 1 and penalties by n for every distribution (views of
    web27_paper_data, whose limits.tex is used), and the threshold functions of harmful-middle at
    n = 8, k = 1."""
    for family in web27_paper_data.FAMILIES:
        view = web27_paper_data.linear_view(load_types(run, family), family)
        write_family_curves(run, family, view)
    view = web27_paper_data.linear_view(load_types(run, MAIN), MAIN)
    leaves = {"tau0": "tau-zero-welfare", "exante": "tau-ex-ante", "relaxation": "tau-relaxation",
              "cutoff": "vcgpa_cutoff_grid", "affine": "vcgpa_poly1_grid", "vcgpa": "vcgpa_nodes_br",
              "ivcgpa": "ivcgpa_nodes_br"}
    columns, e_grid = [], None
    for leaf in leaves.values():
        data = load_data(run / MAIN / "z1" / "n8-k1" / leaf)
        e_grid = data["e_grid"]
        columns.append(shown_threshold(data, view))
    np.savetxt(DATA / "tau_compare_n8k1.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(leaves), comments="", fmt="%.5g")


def write_family_curves(run, family, view):
    leaves = {"tau0": "tau-zero-welfare", "cutoff": "vcgpa_cutoff_grid", "line": "vcgpa_poly1_grid",
              "nodes": "vcgpa_nodes_br"}
    columns, e_grid = [], None
    for leaf in leaves.values():
        data = load_data(run / family / "z1" / "n8-k1" / leaf)
        e_grid = data["e_grid"]
        columns.append(shown_threshold(data, view))
    np.savetxt(DATA / f"forms_{family}.dat", np.column_stack([e_grid] + columns),
               header="e " + " ".join(leaves), comments="", fmt="%.5g")
    columns = []
    for n in SHAPE_BIDDERS:
        data = load_data(run / family / "z1" / f"n{n}-k1" / "vcgpa_nodes_br")
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


def harm_figures(run):
    summary = keyed(run, MAIN)
    markets = [(n, 1) for n in (2, 4, 8, 16, 32)]
    (DATA / "harm_markets.tex").write_text(
        _harm_commands(summary, markets, ["fiveA", "fiveB", "fiveC", "fiveD", "fiveE"]))
    parts = []
    for family, color in zip(web27_paper_data.FAMILIES, ["seriesA", "seriesB", "seriesC", "seriesD"]):
        parts.append(_harm_commands(keyed(run, family), [(8, 1)], [color]))
    (DATA / "harm_families.tex").write_text("".join(parts))


def main(run, out):
    global DATA, FIGURES
    DATA, FIGURES = out / "data", out / "figures"
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    web27_paper_data.main(run, DATA)
    market_tables(run)
    main_tables(run)
    family_shares(run)
    curves(run)
    harm_figures(run)
    for path, name in COPIED:
        shutil.copyfile(run / path, FIGURES / name)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
