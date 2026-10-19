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
  search/tried-taus.pdf, search/convergence.pdf   (optimize runs)
Comparison figures: a figure that varies some inputs and fixes the others is
saved in the compare/ folder of the deepest level above every varied input:
  <distribution>/z<ζ>/compare/tau-by-<input>_<fixed inputs>.pdf
                                     input ∈ market, form, optimizer, objective, threshold
  <distribution>/compare/welfare-vs-zeta_<fixed inputs>.pdf
  compare/share-of-first-best-gain.pdf
"""

import math
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


def leaf_figures(folder, path):
    """Draw the figures of one combination; returns their paths relative to the run folder."""
    folder = Path(folder)
    leaf = folder / path
    results = read_json(leaf / "results.json")
    data, types = _load_data(leaf), _load_types(folder, results)
    drawn = []

    def target(key):
        file = leaf / LEAF_FIGURES[key]
        file.parent.mkdir(exist_ok=True)
        drawn.append(f"{path}/{LEAF_FIGURES[key]}")
        return file

    figures.types_tau_penalties(results, data, types, target("types"))
    figures.expected_outcomes(results, target("outcomes"))
    figures.outcome_distributions(results, data, target("distributions"))
    figures.gain_scatter(results, data, target("scatter"))
    if results["ironing_binds"]:
        figures.ironing(results, data, target("ironing"))
    first_order_file = leaf / "first-order.csv"
    if "mc_vcg_pa" in data and first_order_file.exists():
        figures.marginal_contribution(results, data, types, pd.read_csv(first_order_file),
                                      target("marginal"))
    if results.get("search"):
        figures.search_history(results, data, types, target("tried"), target("trace"))
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


def _run_job(job):
    kind, arguments = job
    if kind == "leaf":
        return leaf_figures(*arguments)
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
        lines.append(f"- **objectives:** {', '.join(config['design']['objectives'])}; "
                     f"**forms:** {', '.join(config['design']['threshold_forms'])}")
        for optimizer in config["optimizers"]:
            lines.append(f"- **optimizer {optimizer['label']}:** {optimizer['method']}")
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
        links = " ".join(f"[{Path(file).name.split('_')[0] if '/search/' not in file else Path(file).stem}]"
                         f"({file})" for file in leaf_files.get(row["path"], []))
        lines.append(f"| [{row['path']}]({row['path']}) | {_share_label(row, 'vcg_pa')} | "
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
    lines += ["", "Figure numbers: 1 types, τ, penalties · 2 expected outcomes · 3 per-auction "
              "distributions · 4 per-auction gain scatter · 5 IVCG-PA ironing · 6 marginal "
              "contribution and Pigouvian check · tried-taus, convergence: the search.", ""]
    lines += ["## Comparison figures", ""]
    lines += [f"- [{file}]({file})" for file in sorted(compare_files)] or ["(none)"]
    lines += ["", "## Files", "",
              "- `run.toml`: resolved configuration, code version, software, host, seeds, timing",
              "- `summary.csv`: one row per combination; `checks.csv`: every check with verdict",
              "- `gap-decomposition.csv`: the paper's decomposition of the welfare gap per market "
              "and threshold class",
              "- `<distribution>/z<ζ>/types.npz`: type-distribution background of the figures",
              "- per combination: `tau.json` (τ; reusable in evaluate runs), `tau.csv` (τ and "
              "penalties at the externality nodes), `results.json` (all statistics), "
              "`first-order.csv`, `data.npz` (arrays behind the figures)",
              "- `log.txt`: progress messages", ""]
    (folder / "README.md").write_text("\n".join(lines))


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


def write_readme(folder, files):
    """README.md of a run from run.toml, summary.csv, checks.csv and the figure lists."""
    folder = Path(folder)
    with open(folder / "run.toml", "rb") as file:
        record = tomllib.load(file)
    summary = pd.read_csv(folder / "summary.csv")
    checks_file = folder / "checks.csv"
    checks = pd.read_csv(checks_file) if checks_file.exists() else pd.DataFrame()
    _readme(folder, record, summary, checks, files["leaf"], files["compare"])
