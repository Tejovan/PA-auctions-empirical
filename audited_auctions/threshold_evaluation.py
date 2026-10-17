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
from .exact_computation import allocation_integral_from_table, build_score_auction_model
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


@dataclass
class ThresholdEvaluation:
    """Results of evaluate_threshold. Plain arrays and dicts only, so it pickles.

    exact          : mechanism -> {metric -> expected value}; metrics are those
                     of ScoreAuctionModel.expected_outcomes plus "penalty_paid"
    penalty_curves : {"externality_grid", "threshold", "vcg_pa", "ivcg_pa",
                      "recipient"} -> arrays over the externality grid;
                     "threshold" is τ(e) (may contain ±inf)
    welfare_contribution_curves :
                     {"bid_levels", "level_mass", "expected", "ironed"} for the
                     participating valuations (IVCG-PA scoring)
    penalty_tables : mechanism ("vcg_pa", "ivcg_pa") -> arrays determining the
                     penalty function exactly; evaluate with penalty_from_table
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

    def summary_table(self) -> pd.DataFrame:
        """One row per mechanism: exact expectations and simulated means."""
        rows = []
        for mechanism in MECHANISMS:
            row = {"mechanism": MECHANISM_LABELS[mechanism]}
            row.update(
                {f"exact_{metric}": value for metric, value in self.exact[mechanism].items()}
            )
            if self.simulated is not None:
                welfare = self.simulated[mechanism]["welfare"]
                row["simulated_welfare_mean"] = float(welfare.mean())
                row["simulated_welfare_ci95"] = float(
                    1.96 * welfare.std(ddof=1) / np.sqrt(len(welfare))
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("mechanism")

    def paired_welfare_differences(self) -> dict:
        """Per-draw welfare differences between mechanisms on the same draws."""
        if self.simulated is None:
            raise ValueError("no simulated draws; rerun with num_simulated_draws > 0")
        welfare = {m: self.simulated[m]["welfare"] for m in MECHANISMS}
        return {
            "ivcg_pa - vcg_pa": welfare["ivcg_pa"] - welfare["vcg_pa"],
            "first_best - ivcg_pa": welfare["first_best"] - welfare["ivcg_pa"],
            "first_best - vcg_pa": welfare["first_best"] - welfare["vcg_pa"],
            "vcg_pa - vcg": welfare["vcg_pa"] - welfare["vcg"],
        }

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path) -> "ThresholdEvaluation":
        with open(path, "rb") as file:
            return pickle.load(file)


# ─── Building the four mechanisms ─────────────────────────────────────────────


def build_mechanism_models(type_distribution, threshold_function, num_bidders, num_goods):
    """Exact ScoreAuctionModel of each mechanism, plus the atom participation mask."""
    everyone = np.ones(type_distribution.num_atoms, dtype=bool)
    participates = type_distribution.valuations >= threshold_function(
        type_distribution.externalities
    )
    valuations = type_distribution.valuations
    common = dict(
        distribution=type_distribution, num_bidders=num_bidders, num_goods=num_goods
    )
    models = {
        "vcg": build_score_auction_model(
            "vcg", atom_participates=everyone, atom_bids=valuations, ironed=False, **common
        ),
        "vcg_pa": build_score_auction_model(
            "vcg_pa", atom_participates=participates, atom_bids=valuations, ironed=False, **common
        ),
        "ivcg_pa": build_score_auction_model(
            "ivcg_pa", atom_participates=participates, atom_bids=valuations, ironed=True, **common
        ),
        "first_best": build_score_auction_model(
            "first_best",
            atom_participates=everyone,
            atom_bids=type_distribution.welfare_contributions,
            ironed=False,
            **common,
        ),
    }
    return models, participates


def inducing_penalty(model, threshold_values, maximum_valuation):
    """Penalty r(e) = R(τ(e)) inducing the thresholds in a normalized DSIC auction.

    Thresholds are clipped to [0, v̄]: τ <= 0 needs no penalty, and τ >= v̄
    (including τ = +inf, exclusion) takes the penalty R(v̄) = G(v̄).
    """
    return penalty_from_table(
        {**model.allocation_integral_table(), "maximum_valuation": maximum_valuation},
        threshold_values,
    )


def penalty_from_table(penalty_table, threshold_values):
    """Penalty r(e) = R(clip(τ(e), 0, v̄)) from a stored penalty table.

    penalty_table: ThresholdEvaluation.penalty_tables[mechanism], i.e. the
    allocation-integral arrays plus "maximum_valuation" (v̄).
    """
    clipped = np.clip(np.asarray(threshold_values, dtype=float), 0.0,
                      penalty_table["maximum_valuation"])
    return allocation_integral_from_table(penalty_table, clipped)


def default_externality_grid(type_distribution, num_points=400, tail_probability=0.005):
    """Grid over the central 99% of the externality distribution."""
    lower = type_distribution.weighted_quantile(
        type_distribution.externalities, tail_probability
    )
    upper = type_distribution.weighted_quantile(
        type_distribution.externalities, 1 - tail_probability
    )
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
                          (used by the sweep scripts to reuse their test draws)
    externality_grid    : externalities at which to tabulate penalty curves
    """
    threshold_function = as_threshold_function(threshold_function)
    models, atom_participates = build_mechanism_models(
        type_distribution, threshold_function, num_bidders, num_goods
    )
    maximum_valuation = float(type_distribution.valuations.max())

    def penalty_function(mechanism):
        return lambda externalities: inducing_penalty(
            models[mechanism], threshold_function(externalities), maximum_valuation
        )

    # Exact expectations --------------------------------------------------------
    exact = {mechanism: models[mechanism].expected_outcomes() for mechanism in MECHANISMS}
    atom_probabilities = type_distribution.probabilities
    exact["vcg"]["penalty_paid"] = 0.0
    for mechanism in PARTICIPANT_AUDIT_MECHANISMS:
        atom_penalties = penalty_function(mechanism)(type_distribution.externalities)
        exact[mechanism]["penalty_paid"] = num_bidders * float(
            np.sum(atom_probabilities * atom_participates * atom_penalties)
        )
    exact["first_best"]["penalty_paid"] = num_bidders * float(
        np.sum(
            atom_probabilities
            * models["first_best"].atom_allocation()
            * -type_distribution.externalities
        )
    )

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
    ivcg_model = models["ivcg_pa"]
    welfare_contribution_curves = {
        "bid_levels": ivcg_model.bid_levels,
        "level_mass": ivcg_model.level_mass,
        "expected": ivcg_model.expected_welfare_contribution(),
        "ironed": ivcg_model.level_scores,
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
        welfare_contribution_curves=welfare_contribution_curves,
        penalty_tables={
            mechanism: {**models[mechanism].allocation_integral_table(),
                        "maximum_valuation": maximum_valuation}
            for mechanism in PARTICIPANT_AUDIT_MECHANISMS
        },
        simulated=simulated,
        settings={
            "num_simulated_draws": 0 if simulated is None else len(externalities),
            "random_seed": random_seed,
            "num_atoms": type_distribution.num_atoms,
            "maximum_valuation": maximum_valuation,
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
