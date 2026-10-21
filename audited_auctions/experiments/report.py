"""
report.py

Figures and README.md of a run, drawn only from its saved files (no
recomputation): python run.py --report RUN_FOLDER redraws them.

Figures of each combination (in its folder):
  1_types-tau-penalties.pdf     types, τ, penalties r(e)
  2_expected-outcomes.pdf       expected welfare components; changes over VCG
  3_outcome-distributions.pdf   per-auction distributions and paired differences
  4_gain-scatter.pdf            per-auction captured gain; revenue vs welfare change
  5_ironing.pdf                 IVCG-PA ironing (only when ironing binds)
  6_marginal-contribution.pdf   marginal contribution maps; Pigouvian check
  search/tried-taus.pdf, search/convergence.pdf, search/iterations.pdf   (optimize runs)
Comparison figures: a figure that varies some inputs and fixes the others is
saved in the compare/ folder of the deepest level above every varied input:
  <distribution>/z<ζ>/compare/tau-by-<input>_<fixed inputs>.pdf
                                     input ∈ market, form, optimizer, objective, threshold
  <distribution>/compare/welfare-vs-zeta_<fixed inputs>.pdf
  compare/share-of-first-best-gain.pdf
Every figure has a WebP preview thumbs/<name>.webp next to it. README files (FIGURE_INFO and
COMPARE_INFO hold the descriptions):
  README.md                   inputs, checks, results with links to every figure, gap
                              decomposition, figure reference, comparison figures, files
  <combination>/README.md     results table, files, each figure under a heading with its
                              description and preview (linked to the PDF), figures not drawn
                              and why, links to paper/ folders
  .../compare/README.md       each comparison figure with its description and preview
write_readme rewrites them from the figures present (also after a paper export).

Paper mode (python run.py --report RUN --paper [NAMES]): one PDF per panel, legend
and colorbar PDFs, <figure>.caption.txt and <figure>.tex in a paper/ folder next to
each figure (figures.py, "Paper mode"); the run's figures and README stay as they
are. NAMES (comma separated) selects figures: the keys of LEAF_FIGURES and
tau-by, welfare-vs-zeta; default all. The share overview has no paper version.
"""

import math
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import figures
from .compute import read_json
from .config import OBJECTIVE_SHORT

LEAF_FIGURES = {
    "types": "1_types-tau-penalties.pdf",
    "outcomes": "2_expected-outcomes.pdf",
    "distributions": "3_outcome-distributions.pdf",
    "scatter": "4_gain-scatter.pdf",
    "ironing": "5_ironing.pdf",
    "marginal": "6_marginal-contribution.pdf",
    "tried": "search/tried-taus.pdf",
    "trace": "search/convergence.pdf",
    "iterations": "search/iterations.pdf",
}

PAPER_FIGURES = (*LEAF_FIGURES, "tau-by", "welfare-vs-zeta")

# Title, contents and condition of every figure, for the README files.
FIGURE_INFO = {
    "types": ("Types, τ and penalties",
              "(a) type distribution, τ(e), abstention region, excluded externalities, "
              "first-best boundary v = −e (also the recipient penalty −e), penalties r(e) of "
              "VCG-PA and IVCG-PA, bids pooled by IVCG-PA, linear valuation axis; (b) the "
              "same on a logarithmic valuation axis.", "always"),
    "outcomes": ("Expected outcomes",
                 "(a) expected welfare W = bidder net utility + revenue + penalties + "
                 "externality welfare, per mechanism, with the share of the first-best gain; "
                 "(b) change of each component over VCG, with bootstrap intervals (empirical "
                 "data).", "always"),
    "distributions": ("Per-auction distributions",
                      "Rows: W, valuation welfare Σvx, externality welfare Σex, revenue, "
                      "penalties. Left: distribution over the simulated auctions, per "
                      "mechanism. Right: paired difference from VCG in the same auction, with "
                      "the share of exact zeros.", "always"),
    "scatter": ("Per-auction gain scatter",
                "Per simulated auction, for VCG-PA and IVCG-PA: captured gain (mechanism − "
                "VCG) against achievable gain (first best − VCG); revenue change against "
                "welfare change.", "always"),
    "ironing": ("IVCG-PA ironing",
                "In quantile space: expected welfare contribution ŵ and ironed welfare "
                "contribution ω̄, Ψ and its convex hull, their gap; scores against bids; "
                "interim allocations.", "only when ironing binds"),
    "marginal": ("Marginal contribution",
                 "(a, b) maps of the marginal contribution M(e, v) for VCG-PA and IVCG-PA, "
                 "with the zero contour and τ; (c) penalty r(e) against the Pigouvian charge; "
                 "(d) M at the threshold types.", "always"),
    "tried": ("Tried thresholds", "Thresholds tried by the search, behind the best one found.",
              "optimize runs"),
    "trace": ("Convergence", "Training objective against time, per start and stage.",
              "optimize runs"),
    "iterations": ("Search iterations",
                   "Per iteration: (a) share of the first-best gain; (b) distance to the best "
                   "welfare found; (c) stationarity in the form's parameters; (d) violation of "
                   "zero marginal contribution; (e) gain of one bidder deviating alone.",
                   "optimize runs with a recorded search history"),
}
COMPARE_INFO = {
    "tau-by": ("τ across one varied input",
               "τ of the combinations that differ in one input (market, form, optimizer, "
               "objective or threshold), other inputs fixed; the legend gives each τ's welfare "
               "and share of the first-best gain.", "two or more such combinations"),
    "welfare-vs-zeta": ("Welfare against ζ",
                        "(a) expected welfare of the four mechanisms and (b) gains over VCG, "
                        "against the externality scale ζ.", "a distribution with several ζ"),
    "share-of-first-best-gain": ("Share of the first-best gain",
                                 "Share of the first-best gain of VCG-PA and IVCG-PA for every "
                                 "combination of the run.", "always"),
}

# Inputs that comparison figures vary; "market" stands for (n, k).
OVERLAY_INPUTS = {"optimize": ("market", "threshold_form", "optimizer", "objective"),
                  "evaluate": ("market", "tau_label")}
INPUT_NAMES = {"market": "market", "threshold_form": "form", "optimizer": "optimizer",
               "objective": "objective", "tau_label": "threshold"}


def _load_types(folder, results):
    combination = results["combination"]
    path = folder / combination["distribution"] / f"z{combination['zeta']:g}" / "types.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_data(leaf):
    with np.load(leaf / "data.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def leaf_figures(folder, path, select=None):
    """Draw the figures of one combination (all, or the keys in select); returns their paths
    relative to the run folder."""
    folder = Path(folder)
    wanted = lambda key: select is None or key in select  # noqa: E731
    leaf = folder / path
    results = read_json(leaf / "results.json")
    data, types = _load_data(leaf), _load_types(folder, results)
    drawn = []

    def target(key):
        file = leaf / LEAF_FIGURES[key]
        file.parent.mkdir(exist_ok=True)
        drawn.append(f"{path}/{LEAF_FIGURES[key]}")
        return file

    if wanted("types"):
        figures.types_tau_penalties(results, data, types, target("types"))
    if wanted("outcomes"):
        figures.expected_outcomes(results, target("outcomes"))
    if wanted("distributions"):
        figures.outcome_distributions(results, data, target("distributions"))
    if wanted("scatter"):
        figures.gain_scatter(results, data, target("scatter"))
    if results["ironing_binds"] and wanted("ironing"):
        figures.ironing(results, data, target("ironing"))
    first_order_file = leaf / "first-order.csv"
    if "mc_vcg_pa" in data and first_order_file.exists() and wanted("marginal"):
        figures.marginal_contribution(results, data, types, pd.read_csv(first_order_file),
                                      target("marginal"))
    if results.get("search") and (wanted("tried") or wanted("trace")):
        figures.search_history(results, data, types, target("tried"), target("trace"))
    if results.get("search") and results["search"].get("history") and wanted("iterations"):
        tolerance = next((c["tolerance"] for c in results["checks"] if c["check"] == "first_order"),
                         1e-4)
        figures.search_iterations(results, target("iterations"),
                                  {"welfare": 1e-4, "stationarity": 1e-4,
                                   "max_marginal": tolerance, "gap": 1e-4})
    return drawn


# ─── Comparison figures ─────────────────────────────────────────────────────────


def _fixed_label(row, varied):
    parts = []
    if varied != "market":
        parts.append(f"n{int(row['num_bidders'])}-k{int(row['num_goods'])}")
    if "objective" in row and isinstance(row.get("objective"), str) and varied != "objective":
        parts.append(OBJECTIVE_SHORT[row["objective"]])
    for key in ("threshold_form", "optimizer", "tau_label"):
        if key in row and isinstance(row.get(key), str) and varied != key:
            parts.append(str(row[key]))
    return "_".join(parts)


def _identity_columns(summary, kind):
    columns = ["distribution", "zeta", "num_bidders", "num_goods"]
    columns += ["objective", "threshold_form", "optimizer"] if kind == "optimize" else ["tau_label"]
    return [c for c in columns if c in summary]


def _share_label(row, mechanism):
    value = row.get(f"share_{mechanism}")
    return figures._percent(value if value is not None and not (
        isinstance(value, float) and math.isnan(value)) else None)


def comparison_jobs(folder, summary, kind):
    """(function name, arguments) of every comparison figure of the run."""
    jobs = []
    identity = _identity_columns(summary, kind)
    for varied in OVERLAY_INPUTS[kind]:
        varied_columns = ["num_bidders", "num_goods"] if varied == "market" else [varied]
        if any(c not in summary for c in varied_columns):
            continue
        fixed = [c for c in identity if c not in varied_columns]
        for _, group in summary.groupby(fixed, dropna=False):
            if len(group) < 2:
                continue
            first = group.iloc[0].to_dict()
            name = f"tau-by-{INPUT_NAMES[varied]}_{_fixed_label(first, varied)}.pdf"
            target = f"{first['distribution']}/z{first['zeta']:g}/compare/{name}"
            jobs.append(("overlay", (str(folder), varied, group.to_dict("records"), target)))
    if summary.groupby("distribution").zeta.nunique().max() > 1:
        fixed = [c for c in identity if c != "zeta"]
        for _, group in summary.groupby(fixed, dropna=False):
            if group.zeta.nunique() < 2:
                continue
            first = group.iloc[0].to_dict()
            target = (f"{first['distribution']}/compare/"
                      f"welfare-vs-zeta_{_fixed_label(first, None)}.pdf")
            jobs.append(("zeta", (str(folder), group.to_dict("records"), target)))
    jobs.append(("overview", (str(folder), summary.to_dict("records"), kind,
                              "compare/share-of-first-best-gain.pdf")))
    return jobs


def _overlay(folder, varied, rows, target):
    folder = Path(folder)
    first = read_json(folder / rows[0]["path"] / "results.json")
    types = _load_types(folder, first)
    entries = []
    if varied == "market":
        ratios = np.array([r["num_bidders"] / r["num_goods"] for r in rows])
        span = np.log(ratios.max() / ratios.min()) if ratios.max() > ratios.min() else 1.0
        colormap = figures.plt.get_cmap("viridis")
        goods = sorted({r["num_goods"] for r in rows})
    objective_of = lambda r: r.get("objective") if isinstance(r.get("objective"), str) else None  # noqa: E731
    for index, row in enumerate(sorted(rows, key=lambda r: (r["num_bidders"] / r["num_goods"],
                                                            str(r.get(varied, ""))))):
        data = _load_data(folder / row["path"])
        mechanism = objective_of(row) or "vcg_pa"
        welfare = f"W = {figures.compact(row[f'welfare_{mechanism}'])}, " \
                  f"{_share_label(row, mechanism)} of first-best gain"
        if varied == "market":
            ratio = row["num_bidders"] / row["num_goods"]
            color = colormap(0.1 + 0.8 * np.log(ratio / ratios.min()) / span)
            linestyle = figures.CATEGORY_LINESTYLES[goods.index(row["num_goods"]) % 7]
            label = (f"n = {row['num_bidders']}, k = {row['num_goods']} (n/k = {ratio:g}): "
                     f"{welfare}")
        elif varied == "objective":
            color = figures.MECHANISM_COLORS[row["objective"]]
            linestyle = figures.MECHANISM_LINESTYLES[row["objective"]]
            label = f"objective {figures.SHORT[row['objective']]}: {welfare}"
        else:
            color = figures.CATEGORY_COLORS[index % len(figures.CATEGORY_COLORS)]
            linestyle = figures.CATEGORY_LINESTYLES[index % 7]
            label = f"{row[varied]}: {welfare}"
            if varied == "optimizer":
                results = read_json(folder / row["path"] / "results.json")
                first_order = next((c["value"] for c in results["checks"]
                                    if c["check"] == "first_order"), None)
                label += (f"; {results['search']['seconds']:.1f} s"
                          + ("" if first_order is None else f"; max |M|/v̄ = {first_order:.1e}"))
        entries.append({"e_grid": data["e_grid"], "tau_grid": data["tau_grid"], "label": label,
                        "color": color, "linestyle": linestyle})
    title = (f"τ by {INPUT_NAMES[varied]}  ·  {first['combination']['distribution']}  ·  "
             f"ζ = {first['combination']['zeta']:g}  ·  {_fixed_label(rows[0], varied)}")
    note = ("W: exact expected welfare of the objective mechanism on the evaluation "
            "distribution. " + str(types["description"]) + ".")
    path = folder / target
    path.parent.mkdir(parents=True, exist_ok=True)
    figures.threshold_overlay(entries, types, title, path, note)
    return target


def _zeta(folder, rows, target):
    folder = Path(folder)
    prepared = []
    for row in rows:
        results = read_json(folder / row["path"] / "results.json")
        bootstrap = results.get("bootstrap") or {}
        entry = {"zeta": row["zeta"]}
        for mechanism in figures.MECHANISMS:
            entry[f"welfare_{mechanism}"] = results["exact"][mechanism]["welfare"]
            entry[f"welfare_{mechanism}_interval"] = bootstrap.get("level_intervals", {}).get(
                "welfare", {}).get(mechanism)
        for mechanism in figures.AUDITED:
            entry[f"gain_{mechanism}"] = results["change_over_vcg"]["welfare"][mechanism]
            entry[f"gain_{mechanism}_interval"] = bootstrap.get("change_intervals", {}).get(
                "welfare", {}).get(mechanism)
        prepared.append(entry)
    first = rows[0]
    title = f"Welfare against ζ  ·  {first['distribution']}  ·  {_fixed_label(first, None)}"
    note = ("Exact expected welfare on the evaluation distribution; error bars: bootstrap "
            "percentile intervals (empirical data). Markers are offset slightly in ζ.")
    path = folder / target
    path.parent.mkdir(parents=True, exist_ok=True)
    figures.welfare_against_zeta(prepared, title, path, note)
    return target


def _overview(folder, rows, kind, target):
    folder = Path(folder)
    groups = {}
    for row in sorted(rows, key=lambda r: r["path"]):
        panel = f"{row['distribution']}, ζ = {row['zeta']:g}"
        label = row["path"].split("/", 2)[2].replace("/", "  ")
        results = read_json(folder / row["path"] / "results.json")
        interval = ((results.get("bootstrap") or {}).get("share_intervals") or {}).get("vcg_pa")
        groups.setdefault(panel, []).append(
            (label, results["share_of_first_best_gain"]["vcg_pa"],
             results["share_of_first_best_gain"]["ivcg_pa"], interval))
    path = folder / target
    path.parent.mkdir(parents=True, exist_ok=True)
    figures.share_overview(groups, path, f"Share of the first-best gain  ·  {folder.name}",
                           "Lines: bootstrap percentile intervals of the VCG-PA share "
                           "(empirical data).")
    return target


def _run_job(job, paper=None):
    """paper: None (the run's figures) or the set of figure names to export as panels."""
    figures.set_paper(paper is not None)
    kind, arguments = job
    if kind == "leaf":
        return leaf_figures(*arguments, select=paper)
    function = {"overlay": _overlay, "zeta": _zeta, "overview": _overview}[kind]
    return [function(*arguments)]


# ─── Decomposition of the welfare gap ───────────────────────────────────────────


def gap_decomposition(summary, kind):
    """The paper's decomposition of the welfare gap (§10 [sec:empirical]), per distribution,
    ζ, market and class T of threshold functions (form, or given threshold):

        FB − OPT,   OPT − OPT^VCG,   OPT^VCG − OPT^VCG_T,

    with OPT = sup_τ W*(τ) (IVCG-PA), OPT^VCG = sup_τ W^VCG(τ) (VCG-PA), and OPT^VCG_T the
    supremum over T. The suprema are replaced by the best values found among the run's
    combinations (IVCG-PA and VCG-PA welfare of every τ of the market), so OPT and OPT^VCG
    are lower bounds. Columns *_share divide each gap by the first-best gain FB − W_VCG.
    With the symmetric relaxation (piecewise-uniform training data), relaxation_vcg is the
    welfare of the best symmetric participation rule for VCG-PA on the training distribution,
    an estimate of an upper bound on OPT^VCG over common τ, and relaxation_minus_opt_vcg its
    gap to the best VCG-PA welfare found. relaxation_basis says where both are measured:
    "evaluation" (training = evaluation: synthetic data without continuous evaluation), or
    "training grid" (both on the gridded training distribution; relaxation_vcg is then not
    comparable with the evaluation welfare in the other columns).
    """
    class_column = "threshold_form" if kind == "optimize" else "tau_label"
    market = ["distribution", "zeta", "num_bidders", "num_goods"]
    rows = []
    for keys, group in summary.groupby(market):
        first_best = float(group.welfare_first_best.iloc[0])
        vcg = float(group.welfare_vcg.iloc[0])
        opt = float(group.welfare_ivcg_pa.max())
        opt_vcg = float(group.welfare_vcg_pa.max())
        gain = first_best - vcg
        for label, members in group.groupby(class_column):
            opt_class = float(members.welfare_vcg_pa.max())
            gaps = {"fb_minus_opt": first_best - opt, "opt_minus_opt_vcg": opt - opt_vcg,
                    "opt_vcg_minus_class": opt_vcg - opt_class}
            row = dict(zip(market, keys), threshold_class=label, first_best=first_best,
                       vcg=vcg, opt_found=opt, opt_vcg_found=opt_vcg, opt_vcg_class=opt_class,
                       **gaps)
            if "relaxation_value" in group and group.relaxation_value.notna().any():
                relaxation = float(group.relaxation_value.max())
                row.update(relaxation_vcg=relaxation, relaxation_basis="evaluation")
                gaps["relaxation_minus_opt_vcg"] = relaxation - opt_vcg
            elif ("relaxation_value_training" in group
                  and group.relaxation_value_training.notna().any()):
                relaxation = float(group.relaxation_value_training.max())
                row.update(relaxation_vcg=relaxation, relaxation_basis="training grid")
                gaps["relaxation_minus_opt_vcg"] = relaxation - float(
                    group.training_welfare_vcg_pa.max())
            if "relaxation_minus_opt_vcg" in gaps:
                row["relaxation_minus_opt_vcg"] = gaps["relaxation_minus_opt_vcg"]
            row.update({f"{name}_share": (value / gain if abs(gain) > 1e-15 else np.nan)
                        for name, value in gaps.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# ─── README ─────────────────────────────────────────────────────────────────────


def _readme(folder, record, summary, checks, leaf_files, compare_files):
    run, config = record["run"], record["config"]
    lines = [f"# Run {run['folder']}", ""]
    if run.get("description"):
        lines += [run["description"], ""]
    seconds = run.get("seconds", {})
    lines += ["| | |", "|---|---|",
              f"| kind | {run['kind']} |",
              f"| started | {run['started']} |",
              f"| runtime | {seconds.get('total', 'running')} s (compute "
              f"{seconds.get('compute', '?')} s, figures {seconds.get('report', '?')} s) |",
              f"| combinations | {run['num_combinations']} ({run.get('num_failed', 0)} failed) |",
              f"| host | {record['host']['hostname']} ({record['host']['where']}), "
              f"{record['host']['workers']} workers |",
              f"| code | {record['code'].get('commit', '')[:10]} on {record['code'].get('branch')}"
              f"{' (uncommitted changes: code.patch)' if record['code'].get('dirty') else ''} |",
              f"| configuration | config.toml (as given), run.toml (resolved, with all defaults) |",
              ""]
    lines += ["## Inputs", ""]
    for distribution in config["distributions"]:
        lines.append(f"- **distribution {distribution['label']}** ({distribution['kind']}, "
                     f"source {distribution['source']}): ζ ∈ {distribution['externality_scales']},"
                     f" grid {distribution['num_externality_nodes']} externality nodes × "
                     f"{distribution['num_valuation_cells']} valuation cells")
    lines.append(f"- **markets:** n ∈ {config['markets']['num_bidders']}, "
                 f"k ∈ {config['markets']['num_goods']}")
    if run["kind"] == "optimize":
        lines.append(f"- **objectives:** {', '.join(config['design']['objectives'])}")
        for optimizer in config["optimizers"]:
            # Runs before forms were set per optimizer list them under [design].
            forms = optimizer.get("threshold_forms", config["design"].get("threshold_forms", []))
            lines.append(f"- **optimizer {optimizer['label']}:** {optimizer['method']}, "
                         f"forms {', '.join(forms)}")
    else:
        for threshold in config["thresholds"]:
            lines.append(f"- **threshold {threshold.get('label') or ''}:** "
                         f"{threshold['source']} = {threshold['value']}")
    evaluation = config["evaluation"]
    lines += [f"- **evaluation:** {evaluation['num_simulated_auctions']:,} simulated auctions; "
              f"{evaluation['num_bootstrap']} bootstrap resamples (empirical data)",
              f"- **seeds:** {config['seeds']}", ""]

    lines += ["## Checks", "",
              "| check | pass | fail | info | what it measures |", "|---|---|---|---|---|"]
    if len(checks):
        for name, group in checks.groupby("check"):
            counts = group.verdict.value_counts()
            note = str(group.note.iloc[0]).replace("|", "\\|")
            lines.append(f"| {name} | {counts.get('pass', 0)} | {counts.get('fail', 0)} | "
                         f"{counts.get('info', 0)} | {note} |")
        failed = checks[checks.verdict == "fail"]
        if len(failed):
            lines += ["", "Failed checks:", ""]
            lines += [f"- {row.path}: {row.check} = {row.value:.3g} (tolerance {row.tolerance})"
                      for row in failed.itertuples()]
    lines += ["", "All checks: checks.csv.", ""]

    lines += ["## Results", "",
              "Share of the first-best welfare gain over VCG, (W − W_VCG)/(W_FB − W_VCG), and "
              "the change of expected revenue over VCG. All rows: summary.csv.", "",
              "| combination | share VCG-PA | share IVCG-PA | Δrevenue VCG-PA | figures |",
              "|---|---|---|---|---|"]
    for row in summary.sort_values("path").to_dict("records"):
        links = " ".join(f"[{_figure_key(file)}]({file})"
                         for file in leaf_files.get(row["path"], []))
        _leaf_readme(folder, row, leaf_files.get(row["path"], []))
        lines.append(f"| [{row['path']}]({row['path']}/README.md) | {_share_label(row, 'vcg_pa')} | "
                     f"{_share_label(row, 'ivcg_pa')} | "
                     f"{figures.compact(row['revenue_change_vcg_pa'])} | {links} |")
    gaps_file = folder / "gap-decomposition.csv"
    if gaps_file.exists():
        gaps = pd.read_csv(gaps_file)
        lines += ["", "## Decomposition of the welfare gap", "",
                  "Gaps as shares of the first-best gain FB − W_VCG (paper §10): FB − OPT (not "
                  "observing externalities), OPT − OPT^VCG (ranking by bid instead of ironed "
                  "welfare contribution), OPT^VCG − OPT^VCG_T (restricting τ to the class T). "
                  "OPT and OPT^VCG are the best values found in this run. All values: "
                  "gap-decomposition.csv.", "",
                  "| distribution | ζ | n, k | class T | FB − OPT | OPT − OPT^VCG | "
                  "OPT^VCG − OPT^VCG_T |", "|---|---|---|---|---|---|---|"]
        for row in gaps.to_dict("records"):
            lines.append(f"| {row['distribution']} | {row['zeta']:g} | {row['num_bidders']}, "
                         f"{row['num_goods']} | {row['threshold_class']} | "
                         f"{figures._percent(row['fb_minus_opt_share'])} | "
                         f"{figures._percent(row['opt_minus_opt_vcg_share'])} | "
                         f"{figures._percent(row['opt_vcg_minus_class_share'])} |")
    lines += ["", "## Figures of each combination", "",
              "Previews and descriptions: the README.md of each combination. Panels for a paper "
              "(`python run.py --report RUN --paper`): `paper/` folders next to the figures.", "",
              "| link | file | shows | drawn |", "|---|---|---|---|"]
    lines += [f"| {key} | `{name}` | {FIGURE_INFO[key][1]} | {FIGURE_INFO[key][2]} |"
              for key, name in LEAF_FIGURES.items()]
    lines += ["", "## Comparison figures", ""]
    lines += ["| figure | shows | drawn when |", "|---|---|---|"]
    lines += [f"| `{key}` | {text} | {condition} |"
              for key, (_, text, condition) in COMPARE_INFO.items()]
    lines.append("")
    by_folder = {}
    for file in sorted(compare_files):
        by_folder.setdefault(str(Path(file).parent), []).append(file)
    lines += _gallery(by_folder.pop("compare", []), ".")
    for compare, files in by_folder.items():
        _compare_readme(folder, compare, files)
        lines.append(f"- [{compare}]({compare}/README.md): {_count(len(files), 'figure')}")
    lines += ["", "## Files", "",
              "- `run.toml`: resolved configuration, code version, software, host, seeds, timing",
              "- `summary.csv`: one row per combination; `checks.csv`: every check with verdict",
              "- `gap-decomposition.csv`: the paper's decomposition of the welfare gap per market "
              "and threshold class",
              "- `<distribution>/z<ζ>/types.npz`: type-distribution background of the figures",
              "- per combination: `tau.json` (τ; reusable in evaluate runs), `tau.csv` (τ and "
              "penalties at the externality nodes), `results.json` (all statistics), "
              "`first-order.csv`, `data.npz` (arrays behind the figures)",
              "- `log.txt`: progress messages",
              "- `thumbs/*.webp` next to every figure: the previews shown in the README files", ""]
    (folder / "README.md").write_text("\n".join(lines))


def _count(number, noun):
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _figure_key(file):
    """Key of LEAF_FIGURES or COMPARE_INFO for a figure path."""
    name = str(Path(file).name)
    for key, leaf in LEAF_FIGURES.items():
        if file.endswith(leaf):
            return key
    return next((key for key in COMPARE_INFO if name.startswith(key)), Path(file).stem)


def _gallery(files, base, info=None):
    """Markdown lines showing the preview of each figure (paths relative to the run
    folder), linked to the PDF, for a README in folder base (relative to the run); with
    info (key → title, text, condition), a heading and description above each preview."""
    lines = []
    for file in files:
        relative = Path(os.path.relpath(file, base))
        if info is not None and _figure_key(file) in info:
            title, text, _ = info[_figure_key(file)]
            lines += [f"### {title}: [{relative}]({relative})", "", text, ""]
        else:
            lines += [f"**[{relative}]({relative})**  "]
        lines += [f"[![{relative.stem}]({figures.thumbnail_path(relative)})]({relative})", ""]
    return lines


def _paper_folders(folder, base):
    """Links to the paper/ folders under folder/base (paper mode output)."""
    found = sorted(p for p in (folder / base).glob("**/paper") if p.is_dir())
    if not found:
        return []
    links = ", ".join(f"[{p.relative_to(folder / base)}/]({p.relative_to(folder / base)})"
                      for p in found)
    return ["## Panels for a paper", "", f"Exported by `run.py --report RUN --paper`: {links}.",
            ""]


def _leaf_readme(folder, row, files):
    path = row["path"]
    if not (folder / path).is_dir():
        return
    up = os.path.relpath(".", path)
    lines = [f"# {path}", "", f"Run: [README.md]({up}/README.md)", ""]
    compare = Path(path).parent.parent / "compare"
    if (folder / compare).is_dir():
        lines[-2:] = [f"Run: [README.md]({up}/README.md) · comparison figures of this market "
                      f"family: [{compare}/README.md]({os.path.relpath(compare, path)}/README.md)",
                      ""]
    lines += ["## Results", "",
              "Exact expected values on the evaluation distribution. Share: (W − W_VCG)/"
              "(W_FB − W_VCG).", "",
              "| mechanism | welfare W | share of first-best gain | revenue | penalties paid |",
              "|---|---|---|---|---|"]
    for mechanism in figures.MECHANISMS:
        share = _share_label(row, mechanism) if mechanism != "vcg" else "0%"
        lines.append(f"| {figures.SHORT[mechanism]} | {figures.compact(row[f'welfare_{mechanism}'])}"
                     f" | {share} | {figures.compact(row[f'revenue_{mechanism}'])} | "
                     f"{figures.compact(row[f'penalty_paid_{mechanism}'])} |")
    binds = row.get("ironing_binds")
    lines += ["", f"Ironing binds: {'yes' if binds else 'no'}. All statistics: results.json.", "",
              "## Files", "",
              "- `tau.json`: τ; reusable in evaluate runs (`tau_file`)",
              "- `tau.csv`: per externality node: mass, τ, exclusion, penalties of VCG-PA and "
              "IVCG-PA",
              "- `first-order.csv`: per node and mechanism: marginal contribution M, its terms, "
              "best-response gap, Pigouvian charge",
              "- `results.json`: every statistic; `data.npz`: arrays behind the figures", "",
              "## Figures", ""]
    lines += _gallery(files, path, FIGURE_INFO)
    drawn = {_figure_key(file) for file in files}
    optimized = isinstance(row.get("optimizer"), str)
    missing = [key for key in FIGURE_INFO if key not in drawn
               and (optimized or not FIGURE_INFO[key][2].startswith("optimize runs"))]
    if missing:
        lines += ["Not drawn: " + "; ".join(f"{FIGURE_INFO[key][0]} ({FIGURE_INFO[key][2]})"
                                            for key in missing) + ".", ""]
    lines += _paper_folders(folder, path)
    (folder / path / "README.md").write_text("\n".join(lines))


def _compare_readme(folder, compare, files):
    up = os.path.relpath(".", compare)
    lines = [f"# Comparison figures in {compare}", "", f"Run: [README.md]({up}/README.md)", ""]
    lines += _gallery(files, compare, COMPARE_INFO)
    lines += _paper_folders(folder, compare)
    (folder / compare / "README.md").write_text("\n".join(lines))


def make_report(folder, workers=None, log=print):
    """Draw every figure of a run folder and write its README.md."""
    folder = Path(folder)
    with open(folder / "run.toml", "rb") as file:
        record = tomllib.load(file)
    summary = pd.read_csv(folder / "summary.csv")
    checks_file = folder / "checks.csv"
    checks = pd.read_csv(checks_file) if checks_file.exists() else pd.DataFrame()
    kind = record["run"]["kind"]
    gap_decomposition(summary, kind).to_csv(folder / "gap-decomposition.csv", index=False)
    jobs = [("leaf", (str(folder), path)) for path in summary.path]
    jobs += comparison_jobs(folder, summary, kind)
    workers = workers or record["host"].get("workers", 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_run_job, jobs, chunksize=1))
    else:
        outputs = [_run_job(job) for job in jobs]
    leaf_files, compare_files = {}, []
    for job, files in zip(jobs, outputs):
        if job[0] == "leaf":
            leaf_files[job[1][1]] = files
        else:
            compare_files.extend(files)
    files = {"leaf": leaf_files, "compare": compare_files}
    write_readme(folder, files)
    log(f"Report: {sum(len(f) for f in leaf_files.values())} combination figures, "
        f"{len(compare_files)} comparison figures, README.md")
    return files


def make_paper_figures(folder, select=None, workers=None, log=print):
    """Export the figures of a run (all, or the names in select; see PAPER_FIGURES) as
    panel PDFs with caption drafts and LaTeX snippets in paper/ folders."""
    folder = Path(folder)
    select = set(select or PAPER_FIGURES)
    unknown = select - set(PAPER_FIGURES)
    if unknown:
        raise ValueError(f"unknown figure names {sorted(unknown)}; choose from "
                         f"{', '.join(PAPER_FIGURES)}")
    with open(folder / "run.toml", "rb") as file:
        record = tomllib.load(file)
    summary = pd.read_csv(folder / "summary.csv")
    jobs = [("leaf", (str(folder), path)) for path in summary.path]
    names = {"overlay": "tau-by", "zeta": "welfare-vs-zeta"}
    jobs += [job for job in comparison_jobs(folder, summary, record["run"]["kind"])
             if names.get(job[0]) in select]
    workers = workers or record["host"].get("workers", 1)
    papers = [select] * len(jobs)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_run_job, jobs, papers, chunksize=1))
    else:
        outputs = [_run_job(job, paper) for job, paper in zip(jobs, papers)]
    count = sum(len(files) for files in outputs)
    write_readme(folder)  # links to the paper/ folders
    log(f"Paper figures: {count} figures exported as panels into paper/ folders")
    return [file for files in outputs for file in files]


def _existing_figures(folder, summary):
    """The figure lists of make_report, from the figures present in the run folder."""
    leaf = {path: [f"{path}/{name}" for name in LEAF_FIGURES.values()
                   if (folder / path / name).exists()] for path in summary.path}
    compare = sorted(str(p.relative_to(folder)) for p in folder.glob("**/compare/*.pdf"))
    return {"leaf": leaf, "compare": compare}


def write_readme(folder, files=None):
    """README.md of a run from run.toml, summary.csv, checks.csv and the figure lists
    (default: the figures present in the folder)."""
    folder = Path(folder)
    with open(folder / "run.toml", "rb") as file:
        record = tomllib.load(file)
    summary = pd.read_csv(folder / "summary.csv")
    checks_file = folder / "checks.csv"
    checks = pd.read_csv(checks_file) if checks_file.exists() else pd.DataFrame()
    files = files or _existing_figures(folder, summary)
    _readme(folder, record, summary, checks, files["leaf"], files["compare"])
