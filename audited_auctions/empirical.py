"""
empirical.py

Pipeline pieces for an empirical population (a CSV with columns e_score,
v_score), in the order they are used:

1. load_empirical_sample: scale (e = ζ e_score, v = action_cost × v_score),
   keep positive valuations, and truncate at a high valuation quantile, which
   becomes the valuation bound v̄ of the model.
2. split_sample: a random train/test split. τ is optimized on the training
   half and evaluated on the test half, so the reported welfare is not inflated
   by fitting τ to the noise of the same rows.
3. sample_distribution: the type distribution of a sample, either smoothed
   (kernel density, smoothing.py; the default, matching the model's continuous
   valuations) or atomic (the rows as they are; a robustness check, mainly for
   VCG and VCG-PA: with distinct valuations an atomic IVCG-PA ranks each row by
   its own v + e, which overfits).
4. bootstrap_welfare: resample the test rows with replacement, rebuild the
   distribution the same way (re-smoothing, with the bandwidth re-chosen by
   Scott's rule on each resample), and recompute expected welfare exactly for
   every resample. The spread is the sampling uncertainty of the population
   (not Monte Carlo error, which exact computation removes). Gains over VCG
   are paired: all mechanisms are evaluated on the same resample.

The truncation bound v̄ is fixed from the full sample and not re-estimated
inside the bootstrap.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .smoothing import kernel_density_distribution
from .threshold_evaluation import AUDITED_MECHANISMS, MECHANISMS, build_mechanism_models
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

    @property
    def num_rows(self) -> int:
        return len(self.valuations)

    def subset(self, rows, label):
        return replace(self, externalities=self.externalities[rows],
                       valuations=self.valuations[rows],
                       settings={**self.settings, "subset": label})


def load_empirical_sample(csv_path, externality_scale=1.0, valuation_scale=1.0,
                          truncation_quantile=0.995):
    """Scaled population, truncated to valuations in (0, v̄], v̄ = valuation quantile."""
    externalities, valuations = load_population_csv(csv_path, externality_scale, valuation_scale)
    positive = valuations > 0
    maximum_valuation = float(np.quantile(valuations[positive], truncation_quantile))
    keep = positive & (valuations <= maximum_valuation)
    return EmpiricalSample(
        externalities=externalities[keep], valuations=valuations[keep],
        maximum_valuation=maximum_valuation, source=str(csv_path),
        settings={"externality_scale": externality_scale, "valuation_scale": valuation_scale,
                  "truncation_quantile": truncation_quantile,
                  "num_rows_loaded": int(len(valuations)),
                  "num_rows_dropped_nonpositive": int((~positive).sum()),
                  "num_rows_dropped_above_bound": int((positive & ~keep).sum())},
    )


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


def exact_welfare_by_mechanism(type_distribution, threshold_function, num_bidders, num_goods):
    """Exact expected welfare of the four mechanisms (no penalties, revenue or simulation)."""
    models, _ = build_mechanism_models(type_distribution, threshold_function, num_bidders,
                                       num_goods)
    return {mechanism: models[mechanism].expected_welfare() for mechanism in MECHANISMS}


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


def bootstrap_welfare(sample, threshold_function, num_bidders, num_goods, num_resamples=200,
                      seed=0, smoothing="kernel", bandwidth_scale=1.0, confidence_level=0.95,
                      **grid_options):
    """Bootstrap of exact expected welfare over resamples of the rows of `sample`.

    Columns of the result: welfare_<mechanism>, gain_<mechanism> (paired gain
    over VCG on the same resample), share_<mechanism> (share of the first-best
    gain over VCG).
    """
    threshold_function = as_threshold_function(threshold_function)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_resamples):
        resample = sample.subset(rng.integers(0, sample.num_rows, sample.num_rows), "bootstrap")
        distribution = sample_distribution(resample, smoothing, bandwidth_scale, **grid_options)
        try:
            welfare = exact_welfare_by_mechanism(distribution, threshold_function, num_bidders,
                                                 num_goods)
        except ValueError:  # nobody participates in this resample
            continue
        row = {f"welfare_{m}": welfare[m] for m in MECHANISMS}
        first_best_gain = welfare["first_best"] - welfare["vcg"]
        for mechanism in AUDITED_MECHANISMS:
            gain = welfare[mechanism] - welfare["vcg"]
            row[f"gain_{mechanism}"] = gain
            row[f"share_{mechanism}"] = gain / first_best_gain if first_best_gain else np.nan
        rows.append(row)
    return BootstrapResult(pd.DataFrame(rows), confidence_level)
