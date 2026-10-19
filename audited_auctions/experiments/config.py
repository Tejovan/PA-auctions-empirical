"""
config.py

Run configurations: a TOML file (configs/*.toml) is read, completed with the
defaults below, validated, and expanded into combinations. Every list in
[markets], [design], the distribution tables ([[distributions]]: externality
scales) and the method tables ([[optimizers]]) is a set of values; a run
computes every combination of them.

Two kinds of run:
  optimize  for each combination of distribution, ζ, n, k, objective, threshold
            form and optimizer: search for the best τ, then evaluate it
  evaluate  for each combination of distribution, ζ, n, k and given threshold
            function ([[thresholds]]): evaluate it

The folder of a combination inside the run folder is
  <distribution>/z<ζ>/n<n>-k<k>/<design>
with <design> = <objective>_<form>_<optimizer> (optimize) or tau-<label> (evaluate).
"""

import copy
import itertools
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..threshold_forms import BEST_RESPONSE_FORMS, GENETIC_FORMS, parse_form_label
from ..type_distributions import SYNTHETIC_FAMILIES

CODE_ROOT = Path(__file__).resolve().parents[2]

# Named empirical data sets (paths relative to the code folder).
DATA_SETS = {"tweets": "data/samples/g_full_tweets.csv"}

RUN_KINDS = ("optimize", "evaluate")
OBJECTIVES = ("vcg_pa", "ivcg_pa")
OBJECTIVE_SHORT = {"vcg_pa": "vcgpa", "ivcg_pa": "ivcgpa"}
METHODS = ("genetic", "best_response", "genetic+best_response")

SYNTHETIC_DEFAULTS = {
    "externality_scales": [1.0],
    "num_externality_nodes": 151,
    "num_valuation_cells": 150,
    "options": {},  # family parameters, e.g. {correlation = -0.6} for harm_rises
}
EMPIRICAL_DEFAULTS = {
    "externality_scales": [1.0],
    "valuation_scale": 1.0,
    "truncation_quantile": 0.995,
    "evaluation_truncation_quantile": None,  # None: same as truncation_quantile
    "train_fraction": 0.5,
    "smoothing": "kernel",  # "kernel" or "none" (training half)
    "bandwidth_scale": 1.0,
    "evaluation": "atomic",  # "atomic" or "smoothed" (test half)
    "num_externality_nodes": 241,
    "num_valuation_cells": 400,
}
GENETIC_DEFAULTS = {
    "num_generations": 500,
    "population_size": 50,
    "num_parents_mating": 15,
    "mutation_probability": 0.25,
    "range_multiplier": 3.0,
    "local_refinement": "nelder_mead",  # or "none"
    "stall_generations": 60,
    "num_local_starts": 5,
    "local_max_evaluations": 2000,
    "num_restarts": 1,
}
BEST_RESPONSE_DEFAULTS = {
    "starts": ["zero", "zero_welfare"],  # also "short_genetic"
    "max_iterations": 100,
    "relative_tolerance": 1e-10,
    "polish_rounds": 3,
    "short_genetic_generations": 40,
}
OPTIMIZER_OTHER_DEFAULTS = {
    "genetic_form": None,  # genetic+best_response: form of the genetic stage
    "warm_start_across_zeta": False,  # best response: also start from the previous ζ's optimum
}
SECTION_DEFAULTS = {
    "markets": {"num_bidders": [8], "num_goods": [1]},
    "design": {"threshold_forms": ["knots6"], "objectives": ["vcg_pa"]},
    "evaluation": {"num_simulated_auctions": 2000, "num_bootstrap": 200,
                   "confidence_level": 0.95, "bootstrap_all_outcomes": True},
    "checks": {"grid_refinement": False, "refinement_factor": 2,
               "bandwidth_sensitivity": [], "grid_relative_tolerance": 0.01,
               "first_order_tolerance": 1e-4, "monte_carlo_z": 4.0},
    "seeds": {"split": 0, "search": 0, "simulation": 1234, "bootstrap": 0},
    "output": {"base": "runs", "workers": 0, "search_history_curves": 400, "report": True},
}


class ConfigError(ValueError):
    pass


def _check_keys(table, allowed, where):
    unknown = set(table) - set(allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _merge(defaults, table, where):
    _check_keys(table, defaults, where)
    merged = copy.deepcopy(defaults)
    merged.update(table)
    return merged


# ─── Distributions ──────────────────────────────────────────────────────────────


def resolve_distribution(table):
    """A [[distributions]] table with defaults; source = family name, data set name or CSV."""
    table = dict(table)
    if "source" not in table:
        raise ConfigError("every [[distributions]] table needs a source")
    source = table.pop("source")
    label = table.pop("label", None)
    for derived in ("kind", "path"):  # present when re-reading a resolved configuration
        table.pop(derived, None)
    if source in SYNTHETIC_FAMILIES:
        resolved = _merge(SYNTHETIC_DEFAULTS, table, f"distribution {source}")
        resolved.update(kind="synthetic", source=source, path=None,
                        label=label or source.replace("_", "-"))
    else:
        path = DATA_SETS.get(source, source)
        if not (CODE_ROOT / path).exists() and not Path(path).exists():
            raise ConfigError(f"distribution source {source!r} is neither a synthetic family "
                              f"({sorted(SYNTHETIC_FAMILIES)}), a data set ({sorted(DATA_SETS)}) "
                              f"nor an existing CSV file")
        resolved = _merge(EMPIRICAL_DEFAULTS, table, f"distribution {source}")
        resolved.update(kind="empirical", source=source, path=str(path),
                        label=label or (source if source in DATA_SETS else Path(path).stem))
        if resolved["evaluation_truncation_quantile"] is None:
            resolved["evaluation_truncation_quantile"] = resolved["truncation_quantile"]
    resolved["externality_scales"] = [float(z) for z in _as_list(resolved["externality_scales"])]
    return resolved


# ─── Optimizers and thresholds ──────────────────────────────────────────────────


def resolve_optimizer(table):
    """An [[optimizers]] table: label, method and the method's parameters."""
    table = dict(table)
    for group in ("genetic", "best_response"):  # nested in a resolved configuration
        table.update(table.pop(group, {}))
    method = table.pop("method", None)
    if method not in METHODS:
        raise ConfigError(f"[[optimizers]] method must be one of {METHODS}, got {method!r}")
    label = table.pop("label", {"genetic": "ga", "best_response": "br",
                                "genetic+best_response": "ga-br"}[method])
    allowed = {**GENETIC_DEFAULTS, **BEST_RESPONSE_DEFAULTS, **OPTIMIZER_OTHER_DEFAULTS}
    _check_keys(table, allowed, f"optimizer {label}")
    genetic = {key: table.get(key, value) for key, value in GENETIC_DEFAULTS.items()}
    best_response = {key: table.get(key, value) for key, value in BEST_RESPONSE_DEFAULTS.items()}
    other = {key: table.get(key, value) for key, value in OPTIMIZER_OTHER_DEFAULTS.items()}
    return {"label": label, "method": method, "genetic": genetic,
            "best_response": best_response, **other}


def method_accepts_form(method, form_label):
    kind, _ = parse_form_label(form_label)
    if method == "genetic":
        return kind in GENETIC_FORMS
    return kind in BEST_RESPONSE_FORMS


THRESHOLD_SOURCES = ("polynomial", "constant", "tabulated", "cutoff", "tau_file", "from_run")


def resolve_threshold(table):
    """A [[thresholds]] table (evaluate runs): a label and exactly one source of τ."""
    table = dict(table)
    _check_keys(table, ("label",) + THRESHOLD_SOURCES, "[[thresholds]]")
    sources = [key for key in THRESHOLD_SOURCES if key in table]
    if len(sources) != 1:
        raise ConfigError(f"each [[thresholds]] table needs exactly one of {THRESHOLD_SOURCES}")
    source = sources[0]
    if source == "cutoff" and not (table["cutoff"] == "ex_ante"
                                   or isinstance(table["cutoff"], (int, float))):
        raise ConfigError('cutoff must be a number (welfare cutoff w) or "ex_ante"')
    if source != "from_run" and "label" not in table:
        raise ConfigError("[[thresholds]] tables need a label (used in folder names)")
    return {"label": table.get("label"), "source": source, "value": table[source]}


# ─── Whole configuration ────────────────────────────────────────────────────────


def resolve_config(raw):
    """Defaults, validation and normalization of a parsed TOML configuration."""
    raw = copy.deepcopy(raw)
    allowed = {"kind", "name", "description", "distributions", "optimizers", "thresholds",
               *SECTION_DEFAULTS}
    _check_keys(raw, allowed, "configuration")
    kind = raw.get("kind")
    if kind not in RUN_KINDS:
        raise ConfigError(f"kind must be one of {RUN_KINDS}")
    config = {"kind": kind, "name": raw.get("name", kind),
              "description": raw.get("description", "")}
    for section, defaults in SECTION_DEFAULTS.items():
        config[section] = _merge(defaults, raw.get(section, {}), f"[{section}]")
    for key in ("num_bidders", "num_goods"):
        config["markets"][key] = [int(v) for v in _as_list(config["markets"][key])]
    config["design"]["threshold_forms"] = _as_list(config["design"]["threshold_forms"])
    config["design"]["objectives"] = _as_list(config["design"]["objectives"])
    for objective in config["design"]["objectives"]:
        if objective not in OBJECTIVES:
            raise ConfigError(f"objectives must be in {OBJECTIVES}, got {objective!r}")
    for form in config["design"]["threshold_forms"]:
        parse_form_label(form)  # raises on an unknown label
    config["checks"]["bandwidth_sensitivity"] = [
        float(s) for s in _as_list(config["checks"]["bandwidth_sensitivity"])]

    if not raw.get("distributions"):
        raise ConfigError("at least one [[distributions]] table is needed")
    config["distributions"] = [resolve_distribution(t) for t in raw["distributions"]]
    labels = [d["label"] for d in config["distributions"]]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"distribution labels must differ (set label = ...): {labels}")

    if kind == "optimize":
        if raw.get("thresholds"):
            raise ConfigError("[[thresholds]] belong to evaluate runs")
        optimizers = raw.get("optimizers") or [{"method": "best_response"}]
        config["optimizers"] = [resolve_optimizer(t) for t in optimizers]
        labels = [o["label"] for o in config["optimizers"]]
        if len(set(labels)) != len(labels):
            raise ConfigError(f"optimizer labels must differ: {labels}")
    else:
        if not raw.get("thresholds"):
            raise ConfigError("evaluate runs need at least one [[thresholds]] table")
        config["thresholds"] = [resolve_threshold(t) for t in raw["thresholds"]]
    return config


def load_config(path):
    with open(path, "rb") as file:
        return resolve_config(tomllib.load(file))


# ─── Combinations ───────────────────────────────────────────────────────────────


@dataclass
class Combination:
    """One cell of a run: a distribution at one ζ, a market (n, k) and a design."""

    distribution: dict
    externality_scale: float
    num_bidders: int
    num_goods: int
    objective: str = None  # optimize runs
    threshold_form: str = None
    optimizer: dict = None
    threshold: dict = None  # evaluate runs: resolved [[thresholds]] entry
    extra: dict = field(default_factory=dict)

    @property
    def design_label(self):
        if self.threshold is not None:
            return f"tau-{self.threshold['label']}"
        return f"{OBJECTIVE_SHORT[self.objective]}_{self.threshold_form}_{self.optimizer['label']}"

    @property
    def distribution_folder(self):
        return f"{self.distribution['label']}/z{self.externality_scale:g}"

    @property
    def market_label(self):
        return f"n{self.num_bidders}-k{self.num_goods}"

    @property
    def path(self):
        return f"{self.distribution_folder}/{self.market_label}/{self.design_label}"

    def keys(self):
        """Identifying fields (for tables and comparison figures)."""
        return {"distribution": self.distribution["label"], "zeta": self.externality_scale,
                "num_bidders": self.num_bidders, "num_goods": self.num_goods,
                "objective": self.objective, "threshold_form": self.threshold_form,
                "optimizer": None if self.optimizer is None else self.optimizer["label"],
                "method": None if self.optimizer is None else self.optimizer["method"],
                "tau_label": None if self.threshold is None else self.threshold["label"],
                "path": self.path}


def _thresholds_from_run(run_folder, distribution_label, zeta, n, k):
    """τ of every combination of an earlier run with the same distribution, ζ, n, k."""
    folder = Path(run_folder)
    if not folder.is_absolute() and not folder.exists():
        folder = CODE_ROOT / folder
    leaf = folder / distribution_label / f"z{zeta:g}" / f"n{n}-k{k}"
    entries = []
    for tau_file in sorted(leaf.glob("*/tau.json")):
        entries.append({"label": tau_file.parent.name, "source": "tau_file",
                        "value": str(tau_file)})
    return entries


def expand(config):
    """(combinations, skipped): every combination of the configuration's sets.

    skipped lists (description, reason) for combinations that cannot run, e.g.
    a polynomial form with best-response ascent.
    """
    combinations, skipped = [], []
    markets = list(itertools.product(config["markets"]["num_bidders"],
                                     config["markets"]["num_goods"]))
    for distribution in config["distributions"]:
        for zeta in distribution["externality_scales"]:
            for n, k in markets:
                base = dict(distribution=distribution, externality_scale=zeta,
                            num_bidders=n, num_goods=k)
                if config["kind"] == "optimize":
                    for objective, form, optimizer in itertools.product(
                            config["design"]["objectives"], config["design"]["threshold_forms"],
                            config["optimizers"]):
                        if not method_accepts_form(optimizer["method"], form):
                            skipped.append((f"{form} with {optimizer['label']}",
                                            f"{optimizer['method']} cannot search {form}"))
                            continue
                        combinations.append(Combination(**base, objective=objective,
                                                        threshold_form=form,
                                                        optimizer=optimizer))
                else:
                    for entry in config["thresholds"]:
                        entries = ([entry] if entry["source"] != "from_run" else
                                   _thresholds_from_run(entry["value"], distribution["label"],
                                                        zeta, n, k))
                        if not entries:
                            skipped.append((f"from_run {entry['value']}",
                                            f"no τ for {distribution['label']} z{zeta:g} "
                                            f"n{n}-k{k}"))
                        combinations.extend(Combination(**base, threshold=e) for e in entries)
    unique_skipped = list(dict.fromkeys(skipped))
    return combinations, unique_skipped


def config_to_json(config):
    return json.loads(json.dumps(config, default=str))
