"""
threshold_evaluation.py

Main entry point: given a type distribution H and a common threshold function
τ, compute (exactly) and optionally simulate the outcomes of four mechanisms:

  "vcg"         VCG auction without audits: every bidder participates and the
                k highest positive valuations win.
  "vcg_pa"      VCG-PA: types with v >= τ(e) participate; the k highest
                positive bids win.
  "ivcg_pa"     IVCG-PA: same participants; the k highest positive ironed
                welfare contributions ω̄^τ(b) win, ties broken uniformly.
  "first_best"  The optimal recipient audit (penalty -e on winners): every
                bidder participates and the k highest positive welfare
                contributions w = v + e win. This attains the first best.

Exact computation and the welfare objective used for optimizing τ
(expected_welfare) work on the type distribution itself; simulation is used
only for per-auction welfare distributions and as a check.

The penalties are not inputs. For VCG-PA and IVCG-PA the module computes the
penalty function r(e) = ∫_0^{τ(e)} X^τ(b) db that induces τ (X^τ is the interim
allocation of the respective auction), and reports it for plotting.

Usage
-----
    from audited_auctions import evaluate_threshold, synthetic_distribution
    from audited_auctions import polynomial_threshold

    evaluation = evaluate_threshold(
        type_distribution=synthetic_distribution("h"),
        threshold_function=polynomial_threshold([2.0, -1.0]),
        num_bidders=8,
        num_goods=1,
        num_simulated_draws=20000,
        random_seed=0,
    )
    print(evaluation.summary_table())
"""

import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .auction_simulation import simulate_score_auction
from .exact_computation import (
    VALUATION_BIDS,
    WELFARE_CONTRIBUTION_BIDS,
    build_score_auction_model,
)
from .scoring_rules import IdentityScore
from .threshold_functions import as_threshold_function, describe_threshold

MECHANISMS = ("vcg", "vcg_pa", "ivcg_pa", "first_best")

MECHANISM_LABELS = {
    "vcg": "VCG (no audit)",
    "vcg_pa": "VCG-PA",
    "ivcg_pa": "IVCG-PA",
    "first_best": "First best (recipient audit)",
}

PARTICIPANT_AUDIT_MECHANISMS = ("vcg_pa", "ivcg_pa")
AUDITED_MECHANISMS = ("vcg_pa", "ivcg_pa", "first_best")


@dataclass
class ThresholdEvaluation:
    """Results of evaluate_threshold. Plain arrays, dicts and small objects, so it pickles.

    exact          : mechanism -> {metric -> expected value}; metrics are those
                     of ScoreAuctionModel.expected_outcomes plus "penalty_paid"
    penalty_curves : {"externality_grid", "threshold", "vcg_pa", "ivcg_pa",
                      "recipient"} -> arrays over the externality grid;
                     "threshold" is τ(e) (may contain ±inf)
    welfare_contribution_curves :
                     IVCG-PA segments: bid_start/end, welfare_start/end (ŵ^τ),
                     score_start/end (ω̄^τ), quantile_start/end
    penalty_tables : mechanism ("vcg_pa", "ivcg_pa") -> {"allocation_integral":
                     callable R(b), "maximum_valuation": v̄}; evaluate the
                     penalty function exactly with penalty_from_table
    simulated      : mechanism -> {metric -> (num_draws,) array}, or None
    """

    distribution_name: str
    threshold_description: str
    num_bidders: int
    num_goods: int
    exact: dict
    penalty_curves: dict
    welfare_contribution_curves: dict
    penalty_tables: dict
    simulated: Optional[dict] = None
    settings: dict = field(default_factory=dict)

    def welfare_gains_over_vcg(self) -> dict:
        """Exact welfare gain of each audited mechanism over unaudited VCG, and the
        share of the first-best gain it captures:
            gain_m = W_m − W_VCG,  share_m = gain_m / (W_first_best − W_VCG)."""
        vcg_welfare = self.exact["vcg"]["welfare"]
        first_best_gain = self.exact["first_best"]["welfare"] - vcg_welfare
        gains = {}
        for mechanism in AUDITED_MECHANISMS:
            gain = self.exact[mechanism]["welfare"] - vcg_welfare
            share = gain / first_best_gain if abs(first_best_gain) > 1e-15 else float("nan")
            gains[mechanism] = {"gain": gain, "share_of_first_best_gain": share}
        return gains

    def summary_table(self) -> pd.DataFrame:
        """One row per mechanism: exact expectations, gains over VCG, simulated means."""
        gains = self.welfare_gains_over_vcg()
        rows = []
        for mechanism in MECHANISMS:
            row = {"mechanism": MECHANISM_LABELS[mechanism]}
            row.update(
                {f"exact_{metric}": value for metric, value in self.exact[mechanism].items()}
            )
            row["exact_welfare_gain_over_vcg"] = gains.get(mechanism, {}).get("gain", 0.0)
            row["share_of_first_best_gain"] = gains.get(mechanism, {}).get(
                "share_of_first_best_gain", 0.0)
            if self.simulated is not None:
                welfare = self.simulated[mechanism]["welfare"]
                row["simulated_welfare_mean"] = float(welfare.mean())
                row["simulated_welfare_ci95"] = float(
                    1.96 * welfare.std(ddof=1) / np.sqrt(len(welfare))
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("mechanism")

    def paired_welfare_differences(self) -> dict:
        """Per-draw welfare gains over unaudited VCG on the same draws.

        Every audited mechanism is measured against the baseline VCG (no audit):
        keys "vcg_pa - vcg", "ivcg_pa - vcg", "first_best - vcg".
        """
        if self.simulated is None:
            raise ValueError("no simulated draws; rerun with num_simulated_draws > 0")
        welfare = {m: self.simulated[m]["welfare"] for m in MECHANISMS}
        return {
            f"{mechanism} - vcg": welfare[mechanism] - welfare["vcg"]
            for mechanism in AUDITED_MECHANISMS
        }

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path) -> "ThresholdEvaluation":
        with open(path, "rb") as file:
            return pickle.load(file)


# ─── Building the four mechanisms ─────────────────────────────────────────────

# mechanism -> (participation, bid kind, ironed)
MECHANISM_SPECIFICATIONS = {
    "vcg": ("everyone", VALUATION_BIDS, False),
    "vcg_pa": ("threshold", VALUATION_BIDS, False),
    "ivcg_pa": ("threshold", VALUATION_BIDS, True),
    "first_best": ("everyone", WELFARE_CONTRIBUTION_BIDS, False),
}


def build_mechanism_model(mechanism, type_distribution, threshold_function, num_bidders,
                          num_goods):
    """Exact ScoreAuctionModel of one mechanism."""
    participation, bid_kind, ironed = MECHANISM_SPECIFICATIONS[mechanism]
    if participation == "everyone":
        cell_thresholds = np.full(type_distribution.num_cells, -np.inf)
    else:
        cell_thresholds = threshold_function(type_distribution.externalities)
    return build_score_auction_model(mechanism, type_distribution, cell_thresholds, bid_kind,
                                     ironed, num_bidders, num_goods)


def build_mechanism_models(type_distribution, threshold_function, num_bidders, num_goods):
    """Exact ScoreAuctionModel of each mechanism, plus the participating mass per cell."""
    threshold_function = as_threshold_function(threshold_function)
    models = {
        mechanism: build_mechanism_model(mechanism, type_distribution, threshold_function,
                                         num_bidders, num_goods)
        for mechanism in MECHANISMS
    }
    return models, models["vcg_pa"].cell_participation_mass


def expected_welfare(type_distribution, threshold_function, num_bidders, num_goods,
                     mechanism="vcg_pa"):
    """Exact expected welfare of one mechanism for τ (the optimization objective).

    Returns 0 if nobody participates (no allocation, no welfare).
    """
    threshold_function = as_threshold_function(threshold_function)
    try:
        model = build_mechanism_model(mechanism, type_distribution, threshold_function,
                                      num_bidders, num_goods)
    except ValueError:
        return 0.0
    return model.expected_welfare()


def penalty_from_table(penalty_table, threshold_values):
    """Penalty r(e) = R(clip(τ(e), 0, v̄)) from a stored penalty table.

    penalty_table: ThresholdEvaluation.penalty_tables[mechanism]. Thresholds are
    clipped to [0, v̄]: τ <= 0 needs no penalty, and τ >= v̄ (including
    τ = +inf, exclusion) takes the penalty R(v̄) = G(v̄).
    """
    clipped = np.clip(np.asarray(threshold_values, dtype=float), 0.0,
                      penalty_table["maximum_valuation"])
    return penalty_table["allocation_integral"](clipped)


def default_externality_grid(type_distribution, num_points=400, tail_probability=0.005):
    """Grid over the central 99% of the externality distribution."""
    lower = type_distribution.weighted_quantile(type_distribution.externalities, tail_probability)
    upper = type_distribution.weighted_quantile(type_distribution.externalities,
                                                1 - tail_probability)
    return np.linspace(lower, upper, num_points)


# ─── Main entry point ─────────────────────────────────────────────────────────


def evaluate_threshold(
    type_distribution,
    threshold_function,
    num_bidders,
    num_goods,
    num_simulated_draws=0,
    random_seed=None,
    simulation_draws=None,
    externality_grid=None,
):
    """Exact (and optionally simulated) outcomes of the four mechanisms for τ.

    type_distribution   : TypeDistribution
    threshold_function  : callable τ(e), coefficient list, or constant
    num_bidders         : n
    num_goods           : k
    num_simulated_draws : number of auctions to simulate (0 = exact only)
    random_seed         : seed for the simulated draws
    simulation_draws    : optional (externalities, valuations) arrays of shape
                          (num_draws, n) to simulate on instead of sampling
    externality_grid    : externalities at which to tabulate penalty curves
    """
    threshold_function = as_threshold_function(threshold_function)
    models, cell_participation_mass = build_mechanism_models(
        type_distribution, threshold_function, num_bidders, num_goods
    )
    maximum_valuation = type_distribution.maximum_valuation
    penalty_tables = {
        mechanism: {"allocation_integral": models[mechanism].allocation_integral_function,
                    "maximum_valuation": maximum_valuation}
        for mechanism in PARTICIPANT_AUDIT_MECHANISMS
    }

    def penalty_function(mechanism):
        return lambda externalities: penalty_from_table(
            penalty_tables[mechanism], threshold_function(externalities))

    # Exact expectations --------------------------------------------------------
    exact = {mechanism: models[mechanism].expected_outcomes() for mechanism in MECHANISMS}
    exact["vcg"]["penalty_paid"] = 0.0
    for mechanism in PARTICIPANT_AUDIT_MECHANISMS:
        cell_penalties = penalty_function(mechanism)(type_distribution.externalities)
        exact[mechanism]["penalty_paid"] = num_bidders * float(
            np.sum(cell_participation_mass * cell_penalties))
    # The recipient penalty −e is charged to winners: n E[x (−e)].
    exact["first_best"]["penalty_paid"] = -exact["first_best"]["externality_welfare"]

    # Curves for plotting -------------------------------------------------------
    if externality_grid is None:
        externality_grid = default_externality_grid(type_distribution)
    penalty_curves = {
        "externality_grid": externality_grid,
        "threshold": threshold_function(externality_grid),
        "vcg_pa": penalty_function("vcg_pa")(externality_grid),
        "ivcg_pa": penalty_function("ivcg_pa")(externality_grid),
        "recipient": -externality_grid,
    }

    # Simulation ----------------------------------------------------------------
    simulated = None
    if simulation_draws is not None or num_simulated_draws > 0:
        if simulation_draws is None:
            rng = np.random.default_rng(random_seed)
            externalities, valuations = type_distribution.sample_types(
                num_simulated_draws, num_bidders, rng
            )
        else:
            externalities, valuations = (np.asarray(a, dtype=float) for a in simulation_draws)
        simulated = simulate_mechanisms(
            models, threshold_function, penalty_function, externalities, valuations, num_goods
        )

    return ThresholdEvaluation(
        distribution_name=type_distribution.name,
        threshold_description=describe_threshold(threshold_function),
        num_bidders=num_bidders,
        num_goods=num_goods,
        exact=exact,
        penalty_curves=penalty_curves,
        welfare_contribution_curves=models["ivcg_pa"].welfare_contribution_curves(),
        penalty_tables=penalty_tables,
        simulated=simulated,
        settings={
            "num_simulated_draws": 0 if simulated is None else len(externalities),
            "random_seed": random_seed,
            "num_cells": type_distribution.num_cells,
            "distribution_kind": type_distribution.kind,
            "maximum_valuation": maximum_valuation,
            "distribution_metadata": dict(type_distribution.metadata),
        },
    )


def simulate_mechanisms(
    models, threshold_function, penalty_function, externalities, valuations, num_goods
):
    """Simulate all four mechanisms on the same draws (common random numbers)."""
    everyone = np.ones(externalities.shape, dtype=bool)
    participates = valuations >= threshold_function(externalities)
    no_penalty = np.zeros(externalities.shape)
    simulated = {
        "vcg": simulate_score_auction(
            externalities, valuations, everyone, valuations, IdentityScore(), num_goods, no_penalty
        ),
        "first_best": simulate_score_auction(
            externalities,
            valuations,
            everyone,
            valuations + externalities,
            IdentityScore(),
            num_goods,
            "recipient",
        ),
    }
    for mechanism in PARTICIPANT_AUDIT_MECHANISMS:
        simulated[mechanism] = simulate_score_auction(
            externalities,
            valuations,
            participates,
            valuations,
            models[mechanism].scoring_rule,
            num_goods,
            penalty_function(mechanism)(externalities),
        )
    return {mechanism: simulated[mechanism] for mechanism in MECHANISMS}


# ─── Grid-refinement check ────────────────────────────────────────────────────


def grid_refinement_check(family, threshold_function, num_bidders, num_goods,
                          externality_scale=1.0, refinement_factor=2, **options):
    """Exact welfare on the default grid and on a grid refined by refinement_factor.

    For a synthetic family, the gridded (piecewise-uniform) distribution
    approximates a continuous density; the change in welfare under refinement
    measures the discretization error. Returns a DataFrame with one row per
    mechanism: welfare on both grids, their difference, and the same for the
    welfare gain over VCG.
    """
    from .type_distributions import synthetic_distribution

    base_nodes = options.pop("num_externality_nodes", 301)
    base_cells = options.pop("num_valuation_cells", 600)
    results = {}
    for label, factor in (("base", 1), ("refined", refinement_factor)):
        grid = {"num_externality_nodes": (base_nodes - 1) * factor + 1,
                "num_valuation_cells": base_cells * factor}
        if family == "example_n":
            grid.pop("num_externality_nodes")
        distribution = synthetic_distribution(family, externality_scale=externality_scale,
                                              **grid, **options)
        models, _ = build_mechanism_models(distribution, threshold_function, num_bidders,
                                           num_goods)
        results[label] = {m: models[m].expected_welfare() for m in MECHANISMS}
    rows = []
    for mechanism in MECHANISMS:
        base, refined = results["base"][mechanism], results["refined"][mechanism]
        base_gain = base - results["base"]["vcg"]
        refined_gain = refined - results["refined"]["vcg"]
        rows.append({"mechanism": MECHANISM_LABELS[mechanism], "welfare_base": base,
                     "welfare_refined": refined, "welfare_change": refined - base,
                     "gain_over_vcg_base": base_gain, "gain_over_vcg_refined": refined_gain,
                     "gain_change": refined_gain - base_gain})
    return pd.DataFrame(rows).set_index("mechanism")
