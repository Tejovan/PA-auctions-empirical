"""Tables and pgfplots data for the empirical supplement (methods and additional results of the
WEB27 paper), from the paper run (configs/paper_run.toml).

    python scripts/empirical_supplement_data.py RUN OUT_DIR

Writes OUT_DIR/data/: everything web27_paper_data.py writes for the submission (type
backgrounds, axis limits, curves of the paper's figures), and the supplement's own files:

    table_all_<family>.tex     shares of the first-best gain and % of the first best, all markets
    table_money.tex            revenue, penalties, net utility and participation (VCG-PA, optimized τ)
    share_n_<family>.dat       k = 1: share of the first-best gain against n, by threshold function
    share_ratio_<family>_k<k>.dat  share of optimized VCG-PA and of τ⁰ against n/k
    curves_<family>.dat        VCG-PA-optimized τ and its penalty r in every market with k < n
    pigou_<family>.dat         (8, 1): penalty and Pigouvian charge at each externality node
    iron_*.dat, iron_*.tex     harmful-middle (2, 1): ironing of the IVCG-PA-optimized design
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import web27_paper_data  # noqa: E402
from PA_auctions import figures  # noqa: E402
from web27_paper_data import label, load_data, load_types, percent, shown_threshold  # noqa: E402

DATA = Path("data")  # set in main
FAMILIES = list(web27_paper_data.FAMILIES)
MAIN = web27_paper_data.MAIN_FAMILY
TABLE_MARKETS = web27_paper_data.TABLE_MARKETS
IRON_MARKET = (2, 1)
PIGOU_MARKET = (8, 1)
FOOTER = "\\bottomrule\n\\end{tabular}\n"


def keyed(run, family):
    summary = pd.read_csv(run / "summary.csv")
    summary = summary[summary.distribution == family].copy()
    summary["key"] = summary.apply(label, axis=1)
    return summary


def row(summary, n, k, key):
    rows = summary[(summary.num_bidders == n) & (summary.num_goods == k) & (summary.key == key)]
    return rows.iloc[0] if len(rows) else None


def guarantee(n, k):
    """max{ρ_{n,k}, 1 − 1/e} of the paper's Cor. 6.5(3), for k < n."""
    rho = 1 - math.comb(n, k) * (1 - k / n) ** (n + 1 - k) * (k / n) ** k
    return max(rho, 1 - 1 / math.e)


def markets(summary):
    return sorted(set(zip(summary.num_bidders, summary.num_goods)), key=lambda m: (m[1], m[0]))


def market_tables(run):
    """Per distribution, every market: % of the first best (VCG, guarantee, IVCG-PA with τ^EA,
    optimized VCG-PA) and shares of the first-best gain (VCG-PA: τ⁰, τ^EA, cutoff, affine,
    optimized; IVCG-PA: τ^EA, optimized)."""
    for family in FAMILIES:
        summary = keyed(run, family)
        with open(DATA / f"table_all_{family}.tex", "w") as file:
            file.write(r"""\begin{tabular}{@{}rrrrrrrrrrrrr@{}}
\toprule
 & & \multicolumn{4}{c}{welfare (\% of $\FB$)} & \multicolumn{7}{c}{share of the first-best gain (\%)}\\
\cmidrule(lr){3-6}\cmidrule(l){7-13}
 & & & & IVCG-PA & VCG-PA & \multicolumn{5}{c}{VCG-PA} & \multicolumn{2}{c}{IVCG-PA}\\
\cmidrule(lr){7-11}\cmidrule(l){12-13}
$n$ & $k$ & VCG & bound & $\tauea$ & opt. & $\tau^0$ & $\tauea$ & cutoff & affine & opt. & $\tauea$ & opt.\\
\midrule
""")
            for n, k in markets(summary):
                nodes, exante = row(summary, n, k, "vcg_pa_nodes"), row(summary, n, k, "ex-ante")
                fb = nodes.welfare_first_best
                values = [nodes.welfare_vcg / fb, guarantee(n, k) if k < n else None,
                          exante.welfare_ivcg_pa / fb, nodes.welfare_vcg_pa / fb,
                          row(summary, n, k, "zero-welfare").share_vcg_pa, exante.share_vcg_pa,
                          row(summary, n, k, "vcg_pa_cutoff").share_vcg_pa,
                          row(summary, n, k, "vcg_pa_poly1").share_vcg_pa, nodes.share_vcg_pa,
                          exante.share_ivcg_pa, row(summary, n, k, "ivcg_pa_nodes").share_ivcg_pa]
                file.write(f"{n} & {k} & " + " & ".join(percent(v) for v in values) + r" \\" + "\n")
            file.write(FOOTER)


def money_table(run):
    """VCG-PA with optimized τ in the markets of TABLE_MARKETS of each distribution."""
    with open(DATA / "table_money.tex", "w") as file:
        file.write(r"""\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & & & VCG & \multicolumn{5}{c}{VCG-PA with optimized $\tau$}\\
\cmidrule(l){5-9}
distribution & $n$ & $k$ & revenue & revenue & penalties & rev.\,+\,pen. & net utility & participants\\
\midrule
""")
        for index, family in enumerate(FAMILIES):
            summary = keyed(run, family)
            for position, (n, k) in enumerate(TABLE_MARKETS):
                r = row(summary, n, k, "vcg_pa_nodes")
                base = r.revenue_vcg
                revenue = base + r.revenue_change_vcg_pa
                values = [f"{base:.2f}", percent(revenue / base), percent(r.penalty_paid_vcg_pa / base),
                          percent((revenue + r.penalty_paid_vcg_pa) / base),
                          percent(r.bidder_utility_vcg_pa / r.bidder_utility_vcg),
                          percent(r.num_participants_vcg_pa / n)]
                first = rf"\multirow{{{len(TABLE_MARKETS)}}}{{*}}{{{family}}}" if position == 0 else ""
                file.write(f"{first} & {n} & {k} & " + " & ".join(values) + r" \\" + "\n")
            if index < len(FAMILIES) - 1:
                file.write(r"\midrule" + "\n")
        file.write(FOOTER)


SHARE_N = [("tau0", "zero-welfare", "share_vcg_pa"), ("exante", "ex-ante", "share_vcg_pa"),
           ("cutoff", "vcg_pa_cutoff", "share_vcg_pa"), ("affine", "vcg_pa_poly1", "share_vcg_pa"),
           ("vcgpa", "vcg_pa_nodes", "share_vcg_pa"), ("ivcgpa", "ivcg_pa_nodes", "share_ivcg_pa")]


def share_files(run):
    for family in FAMILIES:
        summary = keyed(run, family)
        with open(DATA / f"share_n_{family}.dat", "w") as file:
            file.write("n " + " ".join(name for name, _, _ in SHARE_N) + "\n")
            for n in web27_paper_data.SHAPE_BIDDERS:
                file.write(" ".join([str(n)] + [percent(row(summary, n, 1, key)[column])
                                                for _, key, column in SHARE_N]) + "\n")
        for k in (1, 2, 4, 8):
            with open(DATA / f"share_ratio_{family}_k{k}.dat", "w") as file:
                file.write("ratio vcgpa tau0\n")
                for n, kk in markets(summary):
                    if kk == k and n >= 2 * k:
                        file.write(f"{n / k:g} {percent(row(summary, n, k, 'vcg_pa_nodes').share_vcg_pa)} "
                                   f"{percent(row(summary, n, k, 'zero-welfare').share_vcg_pa)}\n")


def curve_files(run):
    """VCG-PA-optimized τ (shown in the paper's view) and its penalty r in every market with k < n."""
    for family in FAMILIES:
        view = web27_paper_data.linear_view(load_types(run, family), family)
        summary = keyed(run, family)
        columns, names, e_grid = [], [], None
        for n, k in markets(summary):
            if k >= n:
                continue
            data = load_data(run / family / "z1" / f"n{n}-k{k}" / "vcgpa_nodes_tr")
            e_grid = data["e_grid"]
            columns += [shown_threshold(data, view), np.asarray(data["penalty_vcg_pa"], dtype=float)]
            names += [f"tau{n}k{k}", f"r{n}k{k}"]
        np.savetxt(DATA / f"curves_{family}.dat", np.column_stack([e_grid] + columns),
                   header="e " + " ".join(names), comments="", fmt="%.5g")


def pigou_files(run):
    """Penalty and Pigouvian charge −eX + L − P at each externality node of the VCG-PA optimum
    (the charge only at interior thresholds)."""
    n, k = PIGOU_MARKET
    for family in FAMILIES:
        leaf = web27_paper_data.existing(run / family / "z1" / f"n{n}-k{k}" / "vcgpa_nodes_tr")
        table = pd.read_csv(leaf / "first-order.csv")
        table = table[table.mechanism == "vcg_pa"]
        table[["externality", "penalty", "pigouvian_charge"]].to_csv(
            DATA / f"pigou_{family}.dat", sep=" ", index=False, header=["e", "penalty", "charge"],
            na_rep="nan", float_format="%.5g")


def iron_files(run):
    """Harmful-middle (2, 1), IVCG-PA-optimized τ: the expected and ironed welfare contribution
    against the bid, the interim allocations of VCG-PA and IVCG-PA with this τ, the VCG-PA- and
    IVCG-PA-optimized τ over the type plane, and the pooled bids (fills for both axes)."""
    n, k = IRON_MARKET
    base = run / MAIN / "z1" / f"n{n}-k{k}"
    data, vcg = load_data(base / "ivcgpa_nodes_tr"), load_data(base / "vcgpa_nodes_tr")
    types = load_types(run, MAIN)
    view = web27_paper_data.linear_view(types, MAIN)
    rows = []
    for i in range(len(data["iron_bid_start"])):
        rows.append((data["iron_bid_start"][i], data["iron_welfare_start"][i], data["iron_score_start"][i]))
        rows.append((data["iron_bid_end"][i], data["iron_welfare_end"][i], data["iron_score_end"][i]))
    np.savetxt(DATA / "iron_scores.dat", np.array(rows), header="b what wbar", comments="", fmt="%.5g")
    np.savetxt(DATA / "iron_alloc.dat",
               np.column_stack([data["alloc_bids"], data["alloc_vcg_pa"], data["alloc_ivcg_pa"]]),
               header="b vcgpa ivcgpa", comments="", fmt="%.5g")
    np.savetxt(DATA / "iron_tau.dat",
               np.column_stack([data["e_grid"], shown_threshold(vcg, view), shown_threshold(data, view)]),
               header="e vcgpa ivcgpa", comments="", fmt="%.5g")
    pooled = figures._visible_pooled_intervals(data, view)
    e_low, e_high = (float(x) for x in web27_paper_data.padded(view)["e_limits"])
    (DATA / "iron_pool_plane.tex").write_text("".join(
        rf"\fill[regionqty] (axis cs:{e_low:.4f},{low:.5g}) rectangle (axis cs:{e_high:.4f},{high:.5g});" + "\n"
        for low, high in pooled))
    (DATA / "iron_pool_bid.tex").write_text("".join(
        rf"\fill[regionqty] ({{axis cs:{low:.5g},0}}|-{{rel axis cs:0,0}}) rectangle "
        rf"({{axis cs:{high:.5g},0}}|-{{rel axis cs:0,1}});" + "\n" for low, high in pooled))
    (DATA / "iron_values.tex").write_text(
        rf"\def\ironpoolstart{{{pooled[-1][0]:.2f}}}" + "\n" if pooled else "")


def main(run, out):
    global DATA
    DATA = out / "data"
    DATA.mkdir(parents=True, exist_ok=True)
    web27_paper_data.main(run, DATA)
    market_tables(run)
    money_table(run)
    share_files(run)
    curve_files(run)
    pigou_files(run)
    iron_files(run)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
