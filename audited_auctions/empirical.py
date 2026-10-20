"""
empirical.py

Pipeline pieces for an empirical population (a CSV with columns e_score,
v_score), in the order they are used:

1. load_empirical_sample: scale (e = ζ e_score, v = action_cost × v_score) and
   keep positive valuations.
2. split_sample: a random train/test split of all rows. τ is optimized on the
   training half and evaluated on the test half, so the reported welfare is
   not inflated by fitting τ to the noise of the same rows. The split is made
   before truncation, so it does not change with the truncation level.
3. valuation_bound + truncate_sample: drop rows with v above a quantile of the
   full sample's valuations; that quantile is the valuation bound v̄ of the
   model. The training half is truncated at truncation_quantile (default
   0.995: the top rows are few and noisy and would dominate the design); the
   test half may use another level (1.0 keeps every row).
4. sample_distribution: the type distribution of a sample, kernel-smoothed
   (smoothing.py; used for the training half, where τ and the IVCG-PA scores
   are designed) or atomic (the rows as they are; used by default for the test
   half, so evaluation has no smoothing bias).
5. bootstrap_welfare: resample the test rows with replacement, rebuild the
   distribution the same way, and recompute expected welfare exactly for every
   resample, with τ and the IVCG-PA scoring rule held fixed at their training
   design. The spread is the sampling uncertainty of the test population (not
   Monte Carlo error, which exact computation removes). Gains over VCG are
   paired: all mechanisms are evaluated on the same resample.

IVCG-PA on atomic data must not be ironed on the data it is evaluated on: with
distinct valuations, each valuation is one row, so ŵ^τ(v) = v + e of that row
and the mechanism would rank bidders by their true welfare contribution. The
pipeline therefore designs the IVCG-PA scoring rule on the smoothed training
distribution and applies it to the test rows.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .parallel import parallel_map
from .smoothing import kernel_density_distribution
from .threshold_evaluation import (
    AUDITED_MECHANISMS,
    MECHANISMS,
    PARTICIPANT_AUDIT_MECHANISMS,
    OUTCOME_METRICS,
    TAU_INDEPENDENT_MECHANISMS,
    build_mechanism_model,
    exact_outcomes,
    tau_independent_outcomes,
)
from .threshold_functions import as_threshold_function
from .type_distributions import load_population_csv, population_distribution

SMOOTHING_CHOICES = ("kernel", "none")


@dataclass
class EmpiricalSample:
    """Population rows after scaling and truncation."""

    externalities: np.ndarray
    valuations: np.ndarray
    maximum_valuation: float
    source: str
    settings: dict
    # τ-independent bootstrap results (VCG and first-best welfare per resample),
    # reused by every τ bootstrapped on this sample; not copied by replace.
    cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def num_rows(self) -> int:
        return len(self.valuations)

    def subset(self, rows, label):
        return replace(self, externalities=self.externalities[rows],
                       valuations=self.valuations[rows],
                       settings={**self.settings, "subset": label})


def load_empirical_sample(csv_path, externality_scale=1.0, valuation_scale=1.0,
                          truncation_quantile=None):
    """Scaled population with positive valuations; optionally truncated at a valuation quantile."""
    externalities, valuations = load_population_csv(csv_path, externality_scale, valuation_scale)
    positive = valuations > 0
    sample = EmpiricalSample(
        externalities=externalities[positive], valuations=valuations[positive],
        maximum_valuation=float(valuations[positive].max()), source=str(csv_path),
        settings={"externality_scale": externality_scale, "valuation_scale": valuation_scale,
                  "num_rows_loaded": int(len(valuations)),
                  "num_rows_dropped_nonpositive": int((~positive).sum())},
    )
    if truncation_quantile is not None:
        sample = truncate_sample(sample, valuation_bound(sample, truncation_quantile))
    return sample


def valuation_bound(sample, truncation_quantile):
    """v̄ = the truncation_quantile of the sample's valuations (1.0: the maximum)."""
    return float(np.quantile(sample.valuations, truncation_quantile))


def truncate_sample(sample, maximum_valuation):
    """Rows with v <= maximum_valuation; the bound becomes the sample's v̄."""
    keep = sample.valuations <= maximum_valuation
    return replace(sample, externalities=sample.externalities[keep],
                   valuations=sample.valuations[keep], maximum_valuation=maximum_valuation,
                   settings={**sample.settings,
                             "num_rows_dropped_above_bound": int((~keep).sum())})


def split_sample(sample, train_fraction=0.5, seed=0):
    """Random train/test split of the rows (same split for every ζ, k, n, degree
    when the seed is the same)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(sample.num_rows)
    num_train = int(round(train_fraction * sample.num_rows))
    return sample.subset(order[:num_train], "train"), sample.subset(order[num_train:], "test")


def sample_distribution(sample, smoothing="kernel", bandwidth_scale=1.0,
                        num_externality_nodes=241, num_valuation_cells=400, name=None):
    """Type distribution of a sample: kernel-smoothed (default) or atomic."""
    label = name or f"{sample.source.split('/')[-1]} ({sample.settings.get('subset', 'all')})"
    if smoothing == "kernel":
        return kernel_density_distribution(
            sample.externalities, sample.valuations, sample.maximum_valuation,
            bandwidth_scale=bandwidth_scale, num_externality_nodes=num_externality_nodes,
            num_valuation_cells=num_valuation_cells, name=f"{label}, smoothed")
    if smoothing == "none":
        distribution = population_distribution(sample.externalities, sample.valuations,
                                               name=f"{label}, atomic")
        distribution.metadata = {"smoothing": "none",
                                 "maximum_valuation": sample.maximum_valuation}
        return distribution
    raise ValueError(f"smoothing must be one of {SMOOTHING_CHOICES}")


def exact_welfare_by_mechanism(type_distribution, threshold_function, num_bidders, num_goods,
                               ivcg_scoring_rule=None):
    """Exact expected welfare of the four mechanisms (no penalties, revenue or simulation).

    VCG and first-best welfare are cached on the distribution (they do not depend on τ).
    """
    threshold_function = as_threshold_function(threshold_function)
    welfare = {
        mechanism: build_mechanism_model(mechanism, type_distribution, threshold_function,
                                         num_bidders, num_goods,
                                         ivcg_scoring_rule).expected_welfare()
        for mechanism in PARTICIPANT_AUDIT_MECHANISMS
    }
    for mechanism in TAU_INDEPENDENT_MECHANISMS:
        welfare[mechanism] = tau_independent_outcomes(type_distribution, mechanism, num_bidders,
                                                      num_goods, welfare_only=True)
    return {mechanism: welfare[mechanism] for mechanism in MECHANISMS}


@dataclass
class BootstrapResult:
    """Per-resample exact welfare and gains over VCG, with percentile intervals."""

    resamples: pd.DataFrame
    confidence_level: float

    def summary(self) -> pd.DataFrame:
        """Mean, standard deviation and percentile interval of each column."""
        lower_tail = (1 - self.confidence_level) / 2
        return pd.DataFrame({
            "mean": self.resamples.mean(),
            "sd": self.resamples.std(ddof=1),
            f"lower_{100 * self.confidence_level:g}": self.resamples.quantile(lower_tail),
            f"upper_{100 * self.confidence_level:g}": self.resamples.quantile(1 - lower_tail),
        })


def _bootstrap_rows(job):
    """Rows of the resamples job["indices"] (module level, for worker processes)."""
    sample, threshold_function = job["sample"], job["threshold"]
    num_bidders, num_goods, all_outcomes = job["num_bidders"], job["num_goods"], job["all_outcomes"]
    baselines = job["baselines"]
    rows = []
    for index, rows_drawn in job["indices"]:
        resample = sample.subset(rows_drawn, "bootstrap")
        distribution = sample_distribution(resample, job["smoothing"], job["bandwidth_scale"],
                                           **job["grid_options"])
        if baselines is not None and index in baselines:
            distribution.cache.update(baselines[index])
        try:
            if all_outcomes:
                outcomes = exact_outcomes(distribution, threshold_function, num_bidders,
                                          num_goods, job["ivcg_scoring_rule"])
                welfare = {m: outcomes[m]["welfare"] for m in MECHANISMS}
            else:
                welfare = exact_welfare_by_mechanism(distribution, threshold_function,
                                                     num_bidders, num_goods,
                                                     job["ivcg_scoring_rule"])
        except ValueError:  # nobody participates in this resample
            continue
        if baselines is not None:
            baselines[index] = {key: value for key, value in distribution.cache.items()
                                if key[-1] in ("welfare", "outcomes")}
        row = {f"welfare_{m}": welfare[m] for m in MECHANISMS}
        if all_outcomes:
            row.update({f"{metric}_{m}": outcomes[m][metric]
                        for metric in OUTCOME_METRICS for m in MECHANISMS})
        first_best_gain = welfare["first_best"] - welfare["vcg"]
        for mechanism in AUDITED_MECHANISMS:
            gain = welfare[mechanism] - welfare["vcg"]
            row[f"gain_{mechanism}"] = gain
            row[f"share_{mechanism}"] = gain / first_best_gain if first_best_gain else np.nan
        rows.append(row)
    return rows


def bootstrap_welfare(sample, threshold_function, num_bidders, num_goods, num_resamples=200,
                      seed=0, smoothing="none", bandwidth_scale=1.0, confidence_level=0.95,
                      ivcg_scoring_rule=None, all_outcomes=False, workers=1, **grid_options):
    """Bootstrap of exact expected welfare over resamples of the rows of `sample`.

    smoothing: how each resample's distribution is built ("none": atomic, the
    default evaluation; "kernel": re-smoothed). ivcg_scoring_rule: the IVCG-PA
    design held fixed across resamples (None: iron each resample).
    all_outcomes: also every metric of OUTCOME_METRICS (revenue, penalties,
    welfare parts, bidder utility; about twice the time).
    workers: processes; the resamples are split into one chunk per process (same
    resamples and results as serial; serial runs reuse VCG and first-best welfare
    across τ bootstrapped on the same sample).

    Columns of the result: welfare_<mechanism>, gain_<mechanism> (paired gain
    over VCG on the same resample), share_<mechanism> (share of the first-best
    gain over VCG); with all_outcomes, <metric>_<mechanism> for every metric.
    """
    threshold_function = as_threshold_function(threshold_function)
    rng = np.random.default_rng(seed)
    drawn = [(index, rng.integers(0, sample.num_rows, sample.num_rows))
             for index in range(num_resamples)]
    job = {"sample": sample, "threshold": threshold_function, "num_bidders": num_bidders,
           "num_goods": num_goods, "all_outcomes": all_outcomes, "smoothing": smoothing,
           "bandwidth_scale": bandwidth_scale, "grid_options": grid_options,
           "ivcg_scoring_rule": ivcg_scoring_rule}
    if workers is None or workers <= 1:
        # VCG and first-best welfare of each resample do not depend on τ. The resamples
        # are determined by the seed, so these values are computed once per (seed,
        # distribution construction, n, k) and reused for every τ bootstrapped on this sample.
        baseline_key = (seed, num_resamples, smoothing, bandwidth_scale,
                        tuple(sorted(grid_options.items())), num_bidders, num_goods)
        rows = _bootstrap_rows({**job, "indices": drawn,
                                "baselines": sample.cache.setdefault(baseline_key, {})})
    else:
        bounds = np.linspace(0, num_resamples, min(workers, num_resamples) + 1).astype(int)
        chunks = [drawn[low:high] for low, high in zip(bounds[:-1], bounds[1:])]
        parts = parallel_map(_bootstrap_rows, [{**job, "indices": chunk, "baselines": None}
                                               for chunk in chunks], workers)
        rows = [row for part in parts for row in part]  # contiguous chunks: serial order
    return BootstrapResult(pd.DataFrame(rows), confidence_level)
