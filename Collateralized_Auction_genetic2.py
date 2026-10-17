#!/usr/bin/env python3
"""
Collateralized_Auction_genetic2.py

Single-cell command-line run of the τ optimization: genetic search for a
polynomial threshold τ(e) that maximizes the exact expected welfare of VCG-PA
(or IVCG-PA), then exact evaluation of VCG, VCG-PA, IVCG-PA and the first best,
one simulation for per-auction welfare distributions and, for empirical data,
a bootstrap of exact welfare over the test rows.

The optimizer itself is audited_auctions/genetic_search.py and the cell logic
is audited_auctions/pipeline.py (shared with run_sweep_genetic2.py, which runs
many cells). The objective is exact: it is computed on the type distribution,
not on simulated auction draws.

Usage:
    python Collateralized_Auction_genetic2.py --data data/samples/g_full_tweets.csv \\
        --k 1 --externality-cost 5 --polynomial-degree 1 --num-items 8 --seed 1234
    python Collateralized_Auction_genetic2.py --data harm_rises \\
        --k 2 --externality-cost 1 --polynomial-degree 2 --num-items 8

--data is a CSV with columns e_score, v_score (empirical) or a synthetic family
name (see audited_auctions/README.md, "Type distributions").

Output: output/results/genetic2_results_<id>.pkl, a one-element list holding
the record of audited_auctions.pipeline.optimize_and_evaluate plus the fields
read by the legacy plot_sweep.py (tau, tested_functions, advertisers,
w_vcg_tot, w_coll_tot, ...), filled from the simulated auctions.
"""

import argparse
import os
import pickle

import numpy as np

from audited_auctions.empirical import SMOOTHING_CHOICES
from audited_auctions.pipeline import optimize_and_evaluate, prepare_distributions


def legacy_fields(record, externality_cost, num_items):
    """Fields of the pre-WEB27 result format, from the simulated auctions (VCG-PA = "coll")."""
    simulated = record["evaluation"].simulated
    if simulated is None:
        return {}
    vcg, vcg_pa = simulated["vcg"], simulated["vcg_pa"]
    return {
        "externality_cost_per_impression": externality_cost,
        "num_advertisers": num_items,
        "num_auctions": len(vcg["welfare"]),
        "k": record["num_goods"],
        "tau": record["tau_coeffs"],
        "tested_functions": record["trajectory"],
        "advertisers": [list(zip(record["type_sample"]["valuations"].tolist(),
                                 record["type_sample"]["externalities"].tolist()))],
        "w_vcg_adv": vcg["valuation_welfare"].tolist(),
        "w_coll_adv": vcg_pa["valuation_welfare"].tolist(),
        "w_vcg_ext": vcg["externality_welfare"].tolist(),
        "w_coll_ext": vcg_pa["externality_welfare"].tolist(),
        "w_vcg_tot": vcg["welfare"].tolist(),
        "w_coll_tot": vcg_pa["welfare"].tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV path or synthetic family name")
    parser.add_argument("--k", type=int, required=True, help="number of goods k")
    parser.add_argument("--externality-cost", type=float, required=True,
                        help="ζ: externalities are multiplied by ζ")
    parser.add_argument("--polynomial-degree", type=int, required=True)
    parser.add_argument("--action-cost", type=float, default=1.0,
                        help="valuation scale for CSV data (v = action_cost × v_score)")
    parser.add_argument("--num-items", type=int, default=20, help="number of bidders n")
    parser.add_argument("--objective", choices=("vcg_pa", "ivcg_pa"), default="vcg_pa",
                        help="mechanism whose exact welfare the GA maximizes")
    parser.add_argument("--seed", type=int, default=None, help="GA seed")
    parser.add_argument("--num-generations", type=int, default=500)
    parser.add_argument("--sol-per-pop", type=int, default=50)
    parser.add_argument("--num-parents-mating", type=int, default=15)
    parser.add_argument("--mutation-probability", type=float, default=0.25)
    parser.add_argument("--range-multiplier", type=float, default=3.0)
    parser.add_argument("--num-restarts", type=int, default=1)
    parser.add_argument("--test-auctions", type=int, default=2000,
                        help="simulated auctions for welfare distributions")
    parser.add_argument("--truncation-quantile", type=float, default=0.995,
                        help="CSV data: valuations above this quantile are dropped; v̄ = quantile")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--smoothing", choices=SMOOTHING_CHOICES, default="kernel")
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--grid-refinement", action="store_true",
                        help="synthetic families: also evaluate τ on a twice finer grid")
    parser.add_argument("--id", default=None, help="run identifier for the output file name")
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = prepare_distributions(
        args.data, externality_scale=args.externality_cost, valuation_scale=args.action_cost,
        truncation_quantile=args.truncation_quantile, train_fraction=args.train_fraction,
        split_seed=args.split_seed, smoothing=args.smoothing,
        bandwidth_scale=args.bandwidth_scale)
    record = optimize_and_evaluate(
        prepared, args.polynomial_degree, args.num_items, args.k,
        objective_mechanism=args.objective,
        ga_options={"num_generations": args.num_generations, "sol_per_pop": args.sol_per_pop,
                    "num_parents_mating": args.num_parents_mating,
                    "mutation_probability": args.mutation_probability,
                    "range_multiplier": args.range_multiplier},
        ga_seed=args.seed, num_restarts=args.num_restarts,
        num_simulated_draws=args.test_auctions, simulation_seed=args.seed,
        num_bootstrap=args.bootstrap_samples, grid_refinement=args.grid_refinement)
    record.update(legacy_fields(record, args.externality_cost, args.num_items))

    print(record["evaluation"].summary_table().T.to_string(float_format=lambda x: f"{x:.5g}"))
    if record["bootstrap"] is not None:
        print("\nBootstrap of exact welfare over test rows:")
        print(record["bootstrap"].summary().to_string(float_format=lambda x: f"{x:.5g}"))
    if record["grid_refinement"] is not None:
        print("\nGrid refinement check:")
        print(record["grid_refinement"].to_string(float_format=lambda x: f"{x:.5g}"))

    run_id = args.id or str(np.random.default_rng().integers(1, 100_000_000))
    os.makedirs("./output/results", exist_ok=True)
    path = f"./output/results/genetic2_results_{run_id}.pkl"
    with open(path, "wb") as file:
        pickle.dump([record], file)
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
