"""
config.py

Run configurations: a TOML file (configs/*.toml) is read, completed with the
defaults below, validated, and expanded into combinations. Every list in
[markets], [design], the distribution tables ([[distributions]]: externality
scales) and the method tables ([[optimizers]]: threshold forms) is a set of
values; a run computes every combination of them.

Two kinds of run:
  optimize  for each combination of distribution, ζ, n, k, objective, optimizer
            and one of that optimizer's threshold forms: search for the best τ,
            then evaluate it
  evaluate  for each combination of distribution, ζ, n, k and given threshold
            function ([[thresholds]]): evaluate it

The folder of a combination inside the run folder is
  <distribution>/z<ζ>/n<n>-k<k>/<design>
with <design> = <objective>_<form>_<optimizer> (optimize) or tau-<label> (evaluate).
"""

import copy
import inspect
import itertools
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..grid_search import GRID_FORMS
from ..threshold_forms import THRESHOLD_RESPONSE_FORMS, GENETIC_FORMS, parse_form_label
from ..type_distributions import SYNTHETIC_FAMILIES

CODE_ROOT = Path(__file__).resolve().parents[2]

# Named empirical data sets (paths relative to the code folder).
DATA_SETS = {"tweets": "data/samples/g_full_tweets.csv"}

RUN_KINDS = ("optimize", "evaluate")
OBJECTIVES = ("vcg_pa", "ivcg_pa")
OBJECTIVE_SHORT = {"vcg_pa": "vcgpa", "ivcg_pa": "ivcgpa"}
METHODS = ("genetic", "threshold_response", "genetic+threshold_response", "relaxation", "grid")
METHOD_LABELS = {"genetic": "ga", "threshold_response": "tr", "genetic+threshold_response": "ga-tr",
                 "relaxation": "relax", "grid": "grid"}
# Method names before best-response ascent was renamed threshold-response ascent (old configs and
# run.toml files).
LEGACY_METHODS = {"best_response": "threshold_response",
                  "genetic+best_response": "genetic+threshold_response"}
# Threshold forms of an [[optimizers]] table without threshold_forms.
DEFAULT_FORMS = {"genetic": ["knots6"], "threshold_response": ["knots6"],
                 "genetic+threshold_response": ["knots6"], "relaxation": ["nodes"],
                 "grid": ["poly1"]}

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
    "stall_tolerance": 1e-4,  # in units of the first-best gain FB − W_VCG (training)
    "num_local_starts": 5,
    "local_max_evaluations": 2000,
    "num_restarts": 4,
}
THRESHOLD_RESPONSE_DEFAULTS = {
    "starts": ["zero", "zero_welfare", "short_genetic", "relaxation"],
    "num_random_starts": 3,
    "max_iterations": 100,
    "relative_tolerance": 1e-10,
    "polish_rounds": 3,
    "short_genetic_generations": 40,
}
GRID_DEFAULTS = {  # grid search (grid_search.py): cutoff and poly1 forms only
    "grid_points": 121,  # per parameter: 121 cutoffs w, or 121 × 121 lines (angle, offset)
    "grid_local_starts": 5,  # Nelder–Mead polish starts (separated grid local maxima)
    "grid_local_max_evaluations": 500,
}
OPTIMIZER_OTHER_DEFAULTS = {
    "objectives": None,  # None: the [design] objectives; else a subset for this optimizer
    "genetic_form": None,  # genetic+threshold_response: form of the genetic stage
    "warm_start_across_zeta": False,  # threshold response: also start from the previous ζ's optimum
    "time_limit_minutes": 60,  # per combination's search; 0 = no limit
}
SECTION_DEFAULTS = {
    "markets": {"num_bidders": [8], "num_goods": [1], "pairs": []},
    "design": {"objectives": ["vcg_pa"]},
    "evaluation": {"num_simulated_auctions": 2000, "num_bootstrap": 200,
                   "confidence_level": 0.95, "bootstrap_all_outcomes": True,
                   "continuous": True},
    "checks": {"grid_refinement": False, "refinement_factor": 2,
               "bandwidth_sensitivity": [], "grid_relative_tolerance": 0.01,
               "first_order_tolerance": 1e-4, "monte_carlo_z": 4.0,
               "symmetric_relaxation": True},
    "seeds": {"split": 0, "search": 0, "simulation": 1234, "bootstrap": 0},
    "output": {"base": "runs", "workers": 0, "search_history_curves": 400, "report": True,
               "previews": "html", "preview_dpi": 120, "preview_width": 440,
               "preview_grid_width": 300},
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


GRID_KEYS = ("num_externality_nodes", "num_valuation_cells")


def family_options(family):
    """Parameters of a synthetic family settable by `options` (grid keys excluded), with defaults."""
    parameters = inspect.signature(SYNTHETIC_FAMILIES[family]).parameters
    return {name: parameter.default for name, parameter in parameters.items()
            if name not in GRID_KEYS}


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
        unknown = set(resolved["options"]) - set(family_options(source))
        if unknown:
            raise ConfigError(f"distribution {source}: unknown options {sorted(unknown)}; "
                              f"allowed: {sorted(family_options(source)) or 'none'}")
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


def method_accepts_form(method, form_label):
    kind, _ = parse_form_label(form_label)
    if method == "grid":
        return str(form_label).strip().lower() in GRID_FORMS
    if method == "genetic":
        return kind in GENETIC_FORMS
    return kind in THRESHOLD_RESPONSE_FORMS


def resolve_optimizer(table):
    """An [[optimizers]] table: label, method, threshold forms and the method's parameters."""
    table = dict(table)
    for group in ("genetic", "threshold_response", "best_response", "grid"):  # nested in a resolved configuration
        table.update(table.pop(group, {}))
    method = table.pop("method", None)
    method = LEGACY_METHODS.get(method, method)
    if method not in METHODS:
        raise ConfigError(f"[[optimizers]] method must be one of {METHODS}, got {method!r}")
    label = table.pop("label", METHOD_LABELS[method])
    forms = _as_list(table.pop("threshold_forms", DEFAULT_FORMS[method]))
    for form in forms:
        parse_form_label(form)  # raises on an unknown label
        if not method_accepts_form(method, form):
            accepted = {"genetic": GENETIC_FORMS, "grid": GRID_FORMS}.get(method,
                                                                         THRESHOLD_RESPONSE_FORMS)
            raise ConfigError(f"optimizer {label}: method {method} cannot search form {form!r} "
                              f"(form kinds: {', '.join(accepted)})")
    allowed = {**GENETIC_DEFAULTS, **THRESHOLD_RESPONSE_DEFAULTS, **GRID_DEFAULTS,
               **OPTIMIZER_OTHER_DEFAULTS}
    _check_keys(table, allowed, f"optimizer {label}")
    genetic = {key: table.get(key, value) for key, value in GENETIC_DEFAULTS.items()}
    threshold_response = {key: table.get(key, value) for key, value in THRESHOLD_RESPONSE_DEFAULTS.items()}
    grid = {key: table.get(key, value) for key, value in GRID_DEFAULTS.items()}
    other = {key: table.get(key, value) for key, value in OPTIMIZER_OTHER_DEFAULTS.items()}
    if other["objectives"] is not None:
        other["objectives"] = _as_list(other["objectives"])
        for objective in other["objectives"]:
            if objective not in OBJECTIVES:
                raise ConfigError(f"optimizer {label}: objectives must be in {OBJECTIVES}, "
                                  f"got {objective!r}")
    if not (isinstance(grid["grid_points"], int) and grid["grid_points"] >= 3):
        raise ConfigError(f"optimizer {label}: grid_points must be an integer >= 3")
    return {"label": label, "method": method, "threshold_forms": forms, "genetic": genetic,
            "threshold_response": threshold_response, "grid": grid, **other}


THRESHOLD_SOURCES = ("polynomial", "constant", "tabulated", "cutoff", "relaxation", "tau_file",
                     "from_run")


def resolve_threshold(table):
    """A [[thresholds]] table: a label and exactly one source of τ."""
    table = dict(table)
    _check_keys(table, ("label",) + THRESHOLD_SOURCES, "[[thresholds]]")
    sources = [key for key in THRESHOLD_SOURCES if key in table]
    if len(sources) != 1:
        raise ConfigError(f"each [[thresholds]] table needs exactly one of {THRESHOLD_SOURCES}")
    source = sources[0]
    if source == "cutoff" and not (table["cutoff"] == "ex_ante"
                                   or isinstance(table["cutoff"], (int, float))):
        raise ConfigError('cutoff must be a number (welfare cutoff w) or "ex_ante"')
    if source == "relaxation" and table["relaxation"] is not True:
        raise ConfigError("relaxation = true is the only value (τ read off the symmetric "
                          "relaxation)")
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
    if "threshold_forms" in raw.get("design", {}):
        raise ConfigError("[design] threshold_forms: the forms are now set per optimizer; move "
                          "them to threshold_forms = [...] in each [[optimizers]] table")
    for section, defaults in SECTION_DEFAULTS.items():
        config[section] = _merge(defaults, raw.get(section, {}), f"[{section}]")
    for key in ("num_bidders", "num_goods"):
        config["markets"][key] = [int(v) for v in _as_list(config["markets"][key])]
    pairs = config["markets"]["pairs"]
    if pairs:
        if {"num_bidders", "num_goods"} & set(raw.get("markets", {})):
            raise ConfigError("[markets]: give either pairs or num_bidders/num_goods")
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pairs):
            raise ConfigError("[markets] pairs must be a list of [n, k] pairs")
        config["markets"]["pairs"] = [[int(n), int(k)] for n, k in pairs]
        if any(n < 1 or k < 1 for n, k in config["markets"]["pairs"]):
            raise ConfigError("[markets] pairs: n and k must be at least 1")
        config["markets"]["num_bidders"] = sorted({n for n, _ in config["markets"]["pairs"]})
        config["markets"]["num_goods"] = sorted({k for _, k in config["markets"]["pairs"]})
    config["design"]["objectives"] = _as_list(config["design"]["objectives"])
    for objective in config["design"]["objectives"]:
        if objective not in OBJECTIVES:
            raise ConfigError(f"objectives must be in {OBJECTIVES}, got {objective!r}")
    if config["output"]["previews"] not in ("html", "markdown"):
        raise ConfigError('[output] previews must be "html" or "markdown"')
    for key in ("preview_dpi", "preview_width", "preview_grid_width"):
        if not (isinstance(config["output"][key], int) and config["output"][key] > 0):
            raise ConfigError(f"[output] {key} must be a positive integer")
    config["checks"]["bandwidth_sensitivity"] = [
        float(s) for s in _as_list(config["checks"]["bandwidth_sensitivity"])]

    if not raw.get("distributions"):
        raise ConfigError("at least one [[distributions]] table is needed")
    config["distributions"] = [resolve_distribution(t) for t in raw["distributions"]]
    labels = [d["label"] for d in config["distributions"]]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"distribution labels must differ (set label = ...): {labels}")

    if kind == "optimize":
        # Given τ are evaluated next to the optimized ones (benchmarks)
        config["thresholds"] = [resolve_threshold(t) for t in raw.get("thresholds", [])]
        optimizers = raw.get("optimizers") or [{"method": "threshold_response"}]
        config["optimizers"] = [resolve_optimizer(t) for t in optimizers]
        # Tables may share a label (e.g. one grid table per set of objectives) if they share
        # the method; expand() then checks that no two combinations share a folder.
        methods = {}
        for optimizer in config["optimizers"]:
            if methods.setdefault(optimizer["label"], optimizer["method"]) != optimizer["method"]:
                raise ConfigError(f"optimizers labeled {optimizer['label']!r} have different "
                                  f"methods; give them different labels")
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


def market_list(config):
    """The markets (n, k) of a configuration: [markets] pairs, else every n × k."""
    if config["markets"].get("pairs"):
        return [tuple(pair) for pair in config["markets"]["pairs"]]
    return list(itertools.product(config["markets"]["num_bidders"],
                                  config["markets"]["num_goods"]))


def markets_text(config):
    """The markets of a configuration for plans and READMEs."""
    if config["markets"].get("pairs"):
        return "(n, k) ∈ {" + ", ".join(f"({n}, {k})" for n, k in market_list(config)) + "}"
    return f"n ∈ {config['markets']['num_bidders']}, k ∈ {config['markets']['num_goods']}"


def _relaxation_undefined(distribution, n, k):
    atomic = distribution["kind"] == "empirical" and distribution["smoothing"] == "none"
    return k >= n or atomic


def expand(config):
    """(combinations, skipped): every combination of the configuration's sets.

    skipped lists (description, reason) for combinations that cannot run: the
    relaxation method or relaxation thresholds with k >= n or an atomic training
    distribution, and from_run thresholds without a match.
    """
    combinations, skipped = [], []
    markets = market_list(config)
    relaxation_reason = "the relaxation needs k < n and a piecewise-uniform training distribution"
    for distribution in config["distributions"]:
        for zeta in distribution["externality_scales"]:
            for n, k in markets:
                base = dict(distribution=distribution, externality_scale=zeta,
                            num_bidders=n, num_goods=k)
                if config["kind"] == "optimize":
                    pairs = [(objective, optimizer) for objective in OBJECTIVES
                             for optimizer in config["optimizers"]
                             if objective in (optimizer.get("objectives")
                                              or config["design"]["objectives"])]
                    for objective, optimizer in pairs:
                        if (optimizer["method"] == "relaxation"
                                and _relaxation_undefined(distribution, n, k)):
                            skipped.append((f"{optimizer['label']} for {distribution['label']} "
                                            f"n{n}-k{k}", relaxation_reason))
                            continue
                        for form in optimizer["threshold_forms"]:
                            combinations.append(Combination(**base, objective=objective,
                                                            threshold_form=form,
                                                            optimizer=optimizer))
                for entry in config.get("thresholds", []):
                    if (entry["source"] == "relaxation"
                            and _relaxation_undefined(distribution, n, k)):
                        skipped.append((f"thresholds {entry['label']} for {distribution['label']} "
                                        f"n{n}-k{k}", relaxation_reason))
                        continue
                    entries = ([entry] if entry["source"] != "from_run" else
                               _thresholds_from_run(entry["value"], distribution["label"],
                                                    zeta, n, k))
                    if not entries:
                        skipped.append((f"from_run {entry['value']}",
                                        f"no τ for {distribution['label']} z{zeta:g} n{n}-k{k}"))
                    combinations.extend(Combination(**base, threshold=e) for e in entries)
    paths = [c.path for c in combinations]
    duplicates = sorted({p for p in paths if paths.count(p) > 1})
    if duplicates:
        raise ConfigError(f"several [[optimizers]] tables give the same combination (same "
                          f"label, form and objective): {duplicates[:3]}")
    unique_skipped = list(dict.fromkeys(skipped))
    return combinations, unique_skipped


def config_to_json(config):
    return json.loads(json.dumps(config, default=str))
