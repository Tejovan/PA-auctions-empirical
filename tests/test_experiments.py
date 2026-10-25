"""
Tests of the experiment runs (PA_auctions/experiments): configuration
validation and expansion, the files of one combination, evaluate runs that
reuse a saved τ, and a whole small run with figures and README.
"""

import json
import re
import tomllib

import numpy as np
import pandas as pd
import pytest

from PA_auctions.experiments import report
from PA_auctions.experiments.compute import compute_combination, load_threshold_file
from PA_auctions.experiments.config import ConfigError, expand, resolve_config
from PA_auctions.experiments.provenance import to_toml
from PA_auctions.experiments.runner import run

SMALL = {"num_externality_nodes": 31, "num_valuation_cells": 40}
SHORT_GENETIC = {"label": "ga", "method": "genetic", "threshold_forms": ["poly1", "knots3"],
                 "num_generations": 6, "population_size": 8, "num_parents_mating": 4,
                 "stall_generations": 4, "num_local_starts": 1, "local_max_evaluations": 30}


def optimize_config(**sections):
    raw = {"kind": "optimize", "name": "test",
           "distributions": [{"source": "harmful_middle", **SMALL}],
           "markets": {"num_bidders": [4], "num_goods": [1]},
           "design": {"objectives": ["vcg_pa"]},
           "optimizers": [SHORT_GENETIC, {"method": "threshold_response", "threshold_forms": ["knots3"],
                                          "max_iterations": 10}],
           "evaluation": {"num_simulated_auctions": 300, "num_bootstrap": 4}}
    raw.update(sections)
    return raw


def test_configuration_errors_name_the_problem():
    with pytest.raises(ConfigError, match="unknown keys"):
        resolve_config({**optimize_config(), "markets": {"num_bidder": [4]}})
    with pytest.raises(ConfigError, match="method"):
        resolve_config(optimize_config(optimizers=[{"method": "annealing"}]))
    with pytest.raises(ConfigError, match="neither a synthetic family"):
        resolve_config(optimize_config(distributions=[{"source": "no_such_family"}]))
    with pytest.raises(ConfigError, match="unknown options"):
        resolve_config(optimize_config(distributions=[{"source": "harmful_middle",
                                                       "options": {"wiggle": 0.3}}]))
    with pytest.raises(ConfigError, match="thresholds"):
        resolve_config({"kind": "evaluate", "distributions": [{"source": "harmful_middle"}]})
    with pytest.raises(ConfigError, match="cannot search form 'poly1'"):
        resolve_config(optimize_config(optimizers=[{"method": "threshold_response",
                                                    "threshold_forms": ["poly1"]}]))
    with pytest.raises(ConfigError, match="per optimizer"):
        resolve_config(optimize_config(design={"threshold_forms": ["knots3"]}))
    with pytest.raises(ConfigError, match="previews"):
        resolve_config({**optimize_config(), "output": {"previews": "svg"}})


def test_expansion_runs_every_valid_combination():
    config = resolve_config(optimize_config(
        markets={"num_bidders": [2, 4], "num_goods": [1, 2]},
        optimizers=[SHORT_GENETIC, {"method": "threshold_response", "threshold_forms": ["knots3"]},
                    {"method": "relaxation"}]))
    combinations, skipped = expand(config)
    # 4 markets × (ga: poly1, knots3; tr: knots3; relax: nodes), relax skipped for n2-k2.
    assert len(combinations) == 15
    assert any("k < n" in reason for _, reason in skipped)
    paths = {c.path for c in combinations}
    assert {"harmful-middle/z1/n4-k2/vcgpa_knots3_tr",
            "harmful-middle/z1/n4-k2/vcgpa_nodes_relax"} <= paths
    assert "harmful-middle/z1/n2-k2/vcgpa_nodes_relax" not in paths
    assert len(paths) == len(combinations)


def test_market_pairs_and_given_thresholds_in_optimize_runs():
    config = resolve_config(optimize_config(
        markets={"pairs": [[4, 1], [8, 2], [2, 2]]},
        optimizers=[{"method": "threshold_response", "threshold_forms": ["knots3"]}],
        thresholds=[{"label": "zero-welfare", "cutoff": 0.0},
                    {"label": "relaxation", "relaxation": True}]))
    combinations, skipped = expand(config)
    # 3 markets × (tr knots3 + τ⁰) + relaxation τ in the two markets with k < n
    assert len(combinations) == 3 * 2 + 2
    assert {(c.num_bidders, c.num_goods) for c in combinations} == {(4, 1), (8, 2), (2, 2)}
    assert any("relaxation" in d and "n2-k2" in d for d, _ in skipped)
    assert "harmful-middle/z1/n8-k2/tau-relaxation" in {c.path for c in combinations}
    with pytest.raises(ConfigError, match="either pairs"):
        resolve_config(optimize_config(markets={"pairs": [[4, 1]], "num_goods": [1]}))
    with pytest.raises(ConfigError, match="pairs"):
        resolve_config(optimize_config(markets={"pairs": [[4, 1, 2]]}))
    with pytest.raises(ConfigError, match="relaxation = true"):
        resolve_config(optimize_config(thresholds=[{"label": "r", "relaxation": 1}]))


def test_market_comparisons_one_direction_at_a_time():
    summary = pd.DataFrame([{"distribution": "d", "zeta": 1.0, "num_bidders": n, "num_goods": k,
                             "objective": "vcg_pa", "threshold_form": "nodes", "optimizer": "tr",
                             "tau_label": np.nan, "path": f"d/z1/n{n}-k{k}/vcgpa_nodes_tr"}
                            for n, k in [(4, 1), (8, 1), (8, 2), (4, 2), (16, 4)]])
    targets = {job[1][3].rsplit("/", 1)[1] for job in report.comparison_jobs("x", summary,
                                                                              "optimize")
               if job[0] == "overlay"}
    assert {"tau-by-bidders_k1_vcgpa_nodes_tr.pdf", "tau-by-bidders_k2_vcgpa_nodes_tr.pdf",
            "tau-by-goods_n8_vcgpa_nodes_tr.pdf", "tau-by-goods_n4_vcgpa_nodes_tr.pdf",
            "tau-by-goods-at-fixed-ratio_nk4_vcgpa_nodes_tr.pdf",
            "tau-by-market_vcgpa_nodes_tr.pdf"} <= targets
    # single-market groups get no figure
    assert not any(tag in t for t in targets for tag in ("_k4_", "_n16_", "_nk2_", "_nk8_"))


def test_toml_writer_round_trip():
    config = resolve_config(optimize_config())
    record = {"run": {"name": "x", "seconds": {"total": 1.5}}, "config": config}
    parsed = tomllib.loads(to_toml(record))
    assert resolve_config(parsed["config"]) == config


def test_synthetic_combination_files_and_accounting(tmp_path):
    config = resolve_config(optimize_config(checks={"grid_refinement": True}))
    combination = next(c for c in expand(config)[0] if c.optimizer["method"] == "threshold_response")
    row, checks, threshold = compute_combination(combination, config, tmp_path)
    folder = tmp_path / combination.path
    for name in ("tau.json", "tau.csv", "results.json", "first-order.csv", "data.npz",
                 "grid-refinement.csv"):
        assert (folder / name).exists(), name
    results = json.loads((folder / "results.json").read_text())
    assert results["share_of_first_best_gain"]["first_best"] == pytest.approx(1.0)
    assert "revenue" not in results["exact"]["first_best"]  # an allocation: no payments
    for mechanism, outcomes in results["exact"].items():
        if mechanism != "first_best":
            assert outcomes["valuation_welfare"] == pytest.approx(
                outcomes["bidder_utility"] + outcomes["revenue"] + outcomes["penalty_paid"])
        assert outcomes["welfare"] == pytest.approx(
            outcomes["valuation_welfare"] + outcomes["externality_welfare"])
    verdicts = {c["check"]: c["verdict"] for c in checks}
    assert verdicts["monte_carlo_vs_exact"] == "pass"
    assert verdicts["grid_refinement"] == "pass"
    reloaded = load_threshold_file(folder / "tau.json")
    externalities = np.linspace(-2, 0.5, 40)
    np.testing.assert_array_equal(reloaded(externalities), threshold(externalities))
    data = np.load(folder / "data.npz")
    assert data["search_tried"].shape[1] == len(data["search_e"])
    assert row["share_vcg_pa"] == pytest.approx(results["share_of_first_best_gain"]["vcg_pa"])


def test_empirical_combination_bootstrap_and_sensitivity(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "population.csv"
    pd.DataFrame({"e_score": rng.normal(-0.3, 0.6, 1500),
                  "v_score": rng.lognormal(0.0, 0.8, 1500)}).to_csv(path, index=False)
    config = resolve_config(optimize_config(
        distributions=[{"source": str(path), "label": "toy", **SMALL}],
        optimizers=[{"method": "threshold_response", "threshold_forms": ["knots3"],
                     "max_iterations": 10}],
        checks={"bandwidth_sensitivity": [0.5, 2.0]}))
    combination = expand(config)[0][0]
    row, checks, _ = compute_combination(combination, config, tmp_path / "run")
    folder = tmp_path / "run" / combination.path
    results = json.loads((folder / "results.json").read_text())
    assert results["ivcg_design"] == "training half"
    assert results["bootstrap"]["num_resamples"] == 4
    assert results["bootstrap"]["change_intervals"]["revenue"]["vcg_pa"] is not None
    assert pd.read_csv(folder / "bandwidth-sensitivity.csv").shape[0] == 3
    assert "bandwidth_sensitivity" in {c["check"] for c in checks}
    assert "welfare_vcg_pa_lower" in row


def test_whole_runs_with_figures_and_reused_thresholds(tmp_path):
    config_file = tmp_path / "optimize.toml"
    config_file.write_text(to_toml(optimize_config(
        markets={"num_bidders": [3, 4], "num_goods": [1]},
        output={"base": str(tmp_path / "runs"), "workers": 1})))
    folder = run(config_file, assume_yes=True)
    summary = pd.read_csv(folder / "summary.csv")
    assert len(summary) == 6
    leaf = folder / summary.path.iloc[0]
    for name in ("1_types-tau-penalties.pdf", "2_expected-outcomes.pdf",
                 "3_outcome-distributions.pdf", "4_gain-scatter.pdf",
                 "6_marginal-contribution.pdf", "search/convergence.pdf"):
        assert (leaf / name).exists(), name
    compare = folder / "harmful-middle" / "z1" / "compare"
    assert any(compare.glob("tau-by-market_*.pdf"))
    assert any(compare.glob("tau-by-form_*.pdf"))
    assert (folder / "compare" / "share-of-first-best-gain.pdf").exists()
    readme = (folder / "README.md").read_text()
    assert "## Checks" in readme and summary.path.iloc[0] in readme
    assert (leaf / "thumbs" / "2_expected-outcomes.webp").exists()
    assert (leaf / "search" / "thumbs" / "convergence.webp").exists()
    assert 'src="compare/thumbs/share-of-first-best-gain.webp"' in readme
    assert f"({summary.path.iloc[0]}/README.md)" in readme
    leaf_readme = (leaf / "README.md").read_text()
    assert ('<a href="2_expected-outcomes.pdf" title="2_expected-outcomes.pdf"><img '
            'src="thumbs/2_expected-outcomes.webp" alt="2_expected-outcomes" width="440"'
            in leaf_readme)
    assert 'src="search/thumbs/convergence.webp"' in leaf_readme
    assert "(../README.md)" in leaf_readme
    market = leaf.parent
    market_readme = (market / "README.md").read_text()
    assert f"({market.relative_to(folder)}/README.md)" in readme
    assert f"[{leaf.name}]({leaf.name}/README.md)" in market_readme
    assert (f'src="{leaf.name}/thumbs/1_types-tau-penalties.webp" alt="1_types-tau-penalties" '
            'width="300"') in market_readme
    record = tomllib.loads((folder / "run.toml").read_text())
    record["config"]["output"]["previews"] = "markdown"
    assert report.preview_settings(record)["previews"] == "markdown"
    report._PREVIEWS.update(report.preview_settings(record))
    assert report._preview(folder, f"{leaf.relative_to(folder)}/1_types-tau-penalties.pdf",
                           str(market.relative_to(folder))).startswith("[![1_types-tau-penalties]")
    report._PREVIEWS.update(report.preview_settings({}))
    assert "(harmful-middle/z1/compare/README.md)" in readme
    assert 'src="thumbs/' in (compare / "README.md").read_text()
    record = tomllib.loads((folder / "run.toml").read_text())
    assert record["run"]["num_combinations"] == 6 and "commit" in record["code"]

    # paper mode: one PDF per panel at the fixed cell size, legend, caption and snippet
    from PA_auctions.experiments.report import make_paper_figures
    from PA_auctions.figures import PAPER_CELL
    make_paper_figures(folder, ["outcomes", "tau-by"], workers=1)
    paper = leaf / "paper"
    for name in ("2_expected-outcomes_a.pdf", "2_expected-outcomes_b.pdf",
                 "2_expected-outcomes_a_legend.pdf", "2_expected-outcomes.tex",
                 "2_expected-outcomes.caption.txt"):
        assert (paper / name).exists(), name
    box = re.search(rb"/MediaBox \[ *0 0 ([\d.]+) ([\d.]+) *\]",
                    (paper / "2_expected-outcomes_a.pdf").read_bytes())
    assert (float(box[1]) / 72, float(box[2]) / 72) == pytest.approx(PAPER_CELL["multi"], abs=0.01)
    assert not (paper / "1_types-tau-penalties.tex").exists()
    assert r"\includegraphics{2_expected-outcomes_b.pdf}" in \
        (paper / "2_expected-outcomes.tex").read_text()
    assert any((compare / "paper").glob("tau-by-market_*.tex"))

    evaluate_file = tmp_path / "evaluate.toml"
    evaluate_file.write_text(to_toml({
        "kind": "evaluate", "name": "reuse",
        "distributions": [{"source": "harmful_middle", **SMALL}],
        "markets": {"num_bidders": [3], "num_goods": [1]},
        "thresholds": [{"from_run": str(folder)}, {"label": "minus-e", "polynomial": [0, -1]}],
        "evaluation": {"num_simulated_auctions": 200},
        "output": {"base": str(tmp_path / "runs"), "workers": 1, "report": False}}))
    evaluated = run(evaluate_file, assume_yes=True)
    rows = pd.read_csv(evaluated / "summary.csv")
    assert len(rows) == 4  # three optimized τ for n = 3, plus τ(e) = −e
    original = summary.set_index("path")
    for row in rows.itertuples():
        if row.tau_label == "minus-e":
            continue
        source = original.loc[f"harmful-middle/z1/n3-k1/{row.tau_label}"]
        assert row.welfare_vcg_pa == pytest.approx(source.welfare_vcg_pa, rel=1e-12)


def test_ex_ante_cutoff_meets_the_paper_guarantee_and_gaps_are_reported(tmp_path):
    from PA_auctions.threshold_forms import ex_ante_guarantee
    # β_{n,k} table of the paper (Thm 8.9 [thm:k-goods]); the bound is max{β, 1 − 1/e}.
    for (n, k), beta in {(2, 1): 0.750, (10, 5): 0.877, (20, 10): 0.912}.items():
        assert ex_ante_guarantee(n, k) == pytest.approx(beta, abs=5e-4)
    from PA_auctions.threshold_forms import vcg_guarantee
    # γ_{n,k} of the submission (Prop 6.6 [prop:vcg-guarantee])
    for (n, k), gamma in {(2, 1): 0.618, (3, 1): 0.570, (10, 1): 0.518, (10, 5): 0.745}.items():
        assert vcg_guarantee(n, k)[0] == pytest.approx(gamma, abs=5e-4)
    config_file = tmp_path / "evaluate.toml"
    config_file.write_text(to_toml({
        "kind": "evaluate", "name": "cutoffs",
        "distributions": [{"source": "harm_rises", **SMALL}],
        "markets": {"num_bidders": [4], "num_goods": [1, 2]},
        "thresholds": [{"label": "zero-welfare", "cutoff": 0.0},
                       {"label": "ex-ante", "cutoff": "ex_ante"},
                       {"label": "vcg-guarantee", "cutoff": "vcg_guarantee"}],
        "evaluation": {"num_simulated_auctions": 200},
        "output": {"base": str(tmp_path / "runs"), "workers": 1}}))
    folder = run(config_file, assume_yes=True)
    checks = pd.read_csv(folder / "checks.csv")
    guarantee = checks[checks.check == "ex_ante_guarantee"]
    assert len(guarantee) == 2 and (guarantee.verdict == "pass").all()
    guarantee = checks[checks.check == "vcg_guarantee"]
    assert len(guarantee) == 2 and (guarantee.verdict == "pass").all()
    gaps = pd.read_csv(folder / "gap-decomposition.csv")
    assert set(gaps.threshold_class) == {"zero-welfare", "ex-ante", "vcg-guarantee"}
    assert (gaps.fb_minus_opt >= -1e-12).all() and (gaps.opt_vcg_minus_class >= -1e-12).all()
    assert "Decomposition of the welfare gap" in (folder / "README.md").read_text()


def test_optimize_run_with_given_thresholds_reports_them_as_classes(tmp_path):
    config_file = tmp_path / "mixed.toml"
    config_file.write_text(to_toml({
        **optimize_config(
            markets={"pairs": [[4, 1]]},
            optimizers=[{"method": "threshold_response", "threshold_forms": ["nodes"],
                         "max_iterations": 5, "num_random_starts": 0,
                         "short_genetic_generations": 2, "polish_rounds": 1}],
            thresholds=[{"label": "zero-welfare", "cutoff": 0.0},
                        {"label": "ex-ante", "cutoff": "ex_ante"},
                        {"label": "relaxation", "relaxation": True}]),
        "output": {"base": str(tmp_path / "runs"), "workers": 1}}))
    folder = run(config_file, assume_yes=True)
    summary = pd.read_csv(folder / "summary.csv")
    assert len(summary) == 4
    gaps = pd.read_csv(folder / "gap-decomposition.csv")
    assert set(gaps.threshold_class) == {"nodes", "tau-zero-welfare", "tau-ex-ante",
                                         "tau-relaxation"}
    # The relaxation τ is the ascent's "relaxation" start: same VCG-PA welfare on the grid
    leaf = summary.set_index("path")
    ascent = json.loads((folder / "harmful-middle/z1/n4-k1/vcgpa_nodes_tr/results.json"
                         ).read_text())
    start = next(s for s in ascent["search"]["starts"] if s["start"] == "relaxation")
    given = leaf.loc["harmful-middle/z1/n4-k1/tau-relaxation"]
    assert given.training_welfare_vcg_pa == pytest.approx(start["start_welfare"], rel=1e-9)
    assert start["welfare"] >= start["start_welfare"] - 1e-12
    readme = (folder / "README.md").read_text()
    assert "given threshold relaxation" in readme and "(n, k) ∈ {(4, 1)}" in readme
    assert "given τ" in (folder / "harmful-middle/z1/n4-k1/README.md").read_text()


def test_inner_parallelism_matches_serial(tmp_path):
    """Search starts, genetic restarts and bootstrap resamples in 2 processes give the
    serial results."""
    rng = np.random.default_rng(3)
    path = tmp_path / "population.csv"
    pd.DataFrame({"e_score": rng.normal(-0.3, 0.6, 600),
                  "v_score": rng.lognormal(0.0, 0.8, 600)}).to_csv(path, index=False)
    config = resolve_config(optimize_config(
        distributions=[{"source": str(path), "label": "toy", **SMALL}],
        optimizers=[{"method": "genetic+threshold_response", "threshold_forms": ["knots3"],
                     "max_iterations": 3,
                     "num_generations": 3, "population_size": 6, "num_parents_mating": 3,
                     "num_restarts": 2, "num_local_starts": 2, "local_max_evaluations": 10,
                     "num_random_starts": 1, "short_genetic_generations": 2}]))
    combination = expand(config)[0][0]
    rows = []
    for workers in (1, 2):
        row, _, _ = compute_combination(combination, {**config, "inner_workers": workers},
                                        tmp_path / f"run{workers}")
        rows.append(row)
    for key in ("training_welfare", "welfare_vcg_pa", "welfare_vcg_pa_lower",
                "welfare_vcg_pa_upper"):
        assert rows[0][key] == pytest.approx(rows[1][key], rel=1e-12), key
