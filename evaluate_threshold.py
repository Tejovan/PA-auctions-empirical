#!/usr/bin/env python3
"""
evaluate_threshold.py

Command-line front end of audited_auctions.evaluate_threshold: for one type
distribution and one common threshold function τ(e), compute exactly (and
simulate) the outcomes of VCG (no audit), VCG-PA, IVCG-PA and the first-best
recipient audit; compute the penalties inducing τ; write a summary table, the
pickled evaluation, and two figures.

The threshold comes from one of
  --tau-coefficients "c0,c1,..."   τ(e) = c0 + c1 e + c2 e² + ...
  --sweep-result PATH              a per-run pickle written by run_sweep_genetic2.py
                                   (uses its tau_coeffs, k, num_items, ζ unless overridden)

The distribution comes from --distribution:
  a, b, c, d, e, f, h              gridded parametric Gaussian mixture (alternate_distributions.ipynb)
  path/to/file.csv                 population with columns e_score, v_score; e = ζ e_score

Examples
--------
    python evaluate_threshold.py --distribution h --tau-coefficients 2,-1 --num-bidders 8 --num-goods 1
    python evaluate_threshold.py --distribution data/samples/g_full_tweets.csv \\
        --externality-scale 5 --tau-coefficients 1e5,-50 --num-bidders 8 --num-goods 1
    python evaluate_threshold.py --sweep-result output/sweep_x/results/run_k1_deg1_n8_ext0.pkl \\
        --distribution data/samples/h_single_modal_shifted_normal_distribution.csv
"""

import argparse
import os
import pickle

from audited_auctions import (
    evaluate_threshold,
    polynomial_threshold,
    population_distribution_from_csv,
    synthetic_distribution,
)
from audited_auctions.figures import plot_threshold_and_penalties, plot_welfare_distributions
from audited_auctions.type_distributions import SYNTHETIC_DISTRIBUTION_PARAMETERS


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--distribution", required=True,
                        help="letter of a synthetic distribution or a CSV path")
    parser.add_argument("--externality-scale", type=float, default=None,
                        help="ζ: e = ζ e_score for CSV populations (default 1, or the sweep's ζ)")
    parser.add_argument("--valuation-scale", type=float, default=None,
                        help="v = scale × v_score for CSV populations (default 1)")
    parser.add_argument("--tau-coefficients", default=None,
                        help='comma-separated polynomial coefficients of τ, lowest degree first')
    parser.add_argument("--sweep-result", default=None,
                        help="per-run pickle from run_sweep_genetic2.py supplying τ, k, n, ζ")
    parser.add_argument("--num-bidders", type=int, default=None, help="n")
    parser.add_argument("--num-goods", type=int, default=None, help="k")
    parser.add_argument("--num-draws", type=int, default=20000,
                        help="simulated auctions (0 = exact only; default 20000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="output/threshold_evaluations")
    parser.add_argument("--tag", default=None, help="file name stem (default: derived)")
    return parser.parse_args()


def main():
    args = parse_args()

    sweep_record = {}
    if args.sweep_result:
        with open(args.sweep_result, "rb") as file:
            sweep_record = pickle.load(file)
    coefficients = (
        [float(c) for c in args.tau_coefficients.split(",")]
        if args.tau_coefficients
        else sweep_record.get("tau_coeffs")
    )
    if coefficients is None:
        raise SystemExit("give --tau-coefficients or --sweep-result")
    num_bidders = args.num_bidders or sweep_record.get("num_items")
    num_goods = args.num_goods or sweep_record.get("k")
    if num_bidders is None or num_goods is None:
        raise SystemExit("give --num-bidders and --num-goods (or --sweep-result)")

    if args.distribution in SYNTHETIC_DISTRIBUTION_PARAMETERS:
        type_distribution = synthetic_distribution(args.distribution)
    else:
        externality_scale = args.externality_scale or sweep_record.get("externality_cost", 1.0)
        valuation_scale = args.valuation_scale or sweep_record.get("action_cost", 1.0)
        type_distribution = population_distribution_from_csv(
            args.distribution, externality_scale, valuation_scale,
            name=f"{os.path.basename(args.distribution)} (ζ = {externality_scale:g})",
        )

    evaluation = evaluate_threshold(
        type_distribution, polynomial_threshold(coefficients), num_bidders, num_goods,
        num_simulated_draws=args.num_draws, random_seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.tag or (
        f"{os.path.splitext(os.path.basename(args.distribution))[0][:12]}"
        f"_n{num_bidders}_k{num_goods}_deg{len(coefficients) - 1}"
    )
    summary = evaluation.summary_table()
    summary.to_csv(os.path.join(args.output_dir, f"{stem}_summary.csv"))
    evaluation.save(os.path.join(args.output_dir, f"{stem}_evaluation.pkl"))
    plot_threshold_and_penalties(
        evaluation, type_distribution,
        os.path.join(args.output_dir, f"{stem}_threshold_penalties.png"),
    )
    if evaluation.simulated is not None:
        plot_welfare_distributions(
            evaluation, os.path.join(args.output_dir, f"{stem}_welfare_distributions.png")
        )
    print(summary.T.to_string(float_format=lambda x: f"{x:.5g}"))
    print(f"Saved outputs with stem '{stem}' in {args.output_dir}")


if __name__ == "__main__":
    main()
