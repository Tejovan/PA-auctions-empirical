"""
sweep_support.py

Glue between the audited_auctions package and the sweep scripts
(run_sweep_genetic2.py, post_optimization_analysis.py).

The sweep scripts store per-draw results as flat lists with key prefixes
  vcg_    VCG without audits
  vcga_   VCG-PA (participant audit)
  ivcga_  IVCG-PA (ironed VCG with participant audits)
  ra_     first best (optimal recipient audit)
and suffixes welfare, adv_welfare (Σ x v), ext_welfare (Σ x e), revenue,
penalty_paid, num_participants, num_allocated. The VCG/VCG-PA/RA welfare keys
are the ones used before IVCG-PA was added, so older plotting code still works.

Each sweep record also stores "evaluation", the full ThresholdEvaluation
(exact expectations, penalty tables that determine r(e) exactly, ŵ^τ and ω̄^τ,
and the simulated draws), "bootstrap" (empirical data) and "type_sample" (a
sample of types for scatter plots); see audited_auctions/pipeline.py.
"""

import numpy as np

from .threshold_evaluation import penalty_from_table
from .threshold_functions import polynomial_threshold

MECHANISM_KEY_PREFIX = {"vcg": "vcg", "vcg_pa": "vcga", "ivcg_pa": "ivcga", "first_best": "ra"}

METRIC_KEY_SUFFIX = {
    "welfare": "welfare",
    "valuation_welfare": "adv_welfare",
    "externality_welfare": "ext_welfare",
    "revenue": "revenue",
    "penalty_paid": "penalty_paid",
    "num_participants": "num_participants",
    "num_allocated": "num_allocated",
}

# Labels, colors (validated palette, see figures.py) and markers for sweep plots.
SWEEP_SERIES = [
    ("vcg", "VCG (no audit)", "#7a7974", "o"),
    ("vcga", "VCG-PA", "#2a78d6", "s"),
    ("ivcga", "IVCG-PA", "#c0392b", "D"),
    ("ra", "First best (recipient audit)", "#1baf7a", "^"),
]


def sweep_record_fields(evaluation):
    """Flat per-draw lists and exact results to merge into a sweep record."""
    fields = {}
    for mechanism, prefix in MECHANISM_KEY_PREFIX.items():
        for metric, suffix in METRIC_KEY_SUFFIX.items():
            fields[f"{prefix}_{suffix}"] = evaluation.simulated[mechanism][metric].tolist()
    fields["evaluation"] = evaluation
    return fields


def record_penalty_curves(record, externality_values):
    """Penalty functions of VCG-PA and IVCG-PA at the given externalities.

    Returns {"vcg_pa": array, "ivcg_pa": array}. Records written before the
    exact penalties existed have no "evaluation"; for them this raises, since
    their stored payments only support the biased estimator that the exact
    computation replaces.
    """
    if "evaluation" not in record:
        raise KeyError(
            "record has no exact penalty tables; rerun the sweep, or pass --data to "
            "post_optimization_analysis.py to recompute them"
        )
    threshold_values = polynomial_threshold(record["tau_coeffs"])(np.asarray(externality_values))
    return {
        mechanism: penalty_from_table(table, threshold_values)
        for mechanism, table in record["evaluation"].penalty_tables.items()
    }


def record_type_sample(record):
    """(externalities, valuations) of the record's type sample (for scatter plots)."""
    sample = record["type_sample"]
    return np.asarray(sample["externalities"]), np.asarray(sample["valuations"])


def record_exact_welfare(record, confidence_level=0.95):
    """prefix -> {"welfare", "lower", "upper"}: exact expected welfare of each
    mechanism on the evaluation distribution, with bootstrap percentile bounds
    when the record has a bootstrap (empirical data), else None."""
    evaluation = record["evaluation"]
    bootstrap = record.get("bootstrap")
    tail = (1 - confidence_level) / 2
    result = {}
    for mechanism, prefix in MECHANISM_KEY_PREFIX.items():
        value = {"welfare": evaluation.exact[mechanism]["welfare"], "lower": None, "upper": None}
        if bootstrap is not None and len(bootstrap.resamples):
            column = bootstrap.resamples[f"welfare_{mechanism}"]
            value["lower"], value["upper"] = (float(column.quantile(tail)),
                                              float(column.quantile(1 - tail)))
        result[prefix] = value
    return result
