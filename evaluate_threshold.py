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
  a family name                    synthetic piecewise-uniform family (a–f, h, example_n,
                                   independent, harm_rises, harmful_middle, heavy_tail;
                                   see audited_auctions/README.md), externalities × ζ
  path/to/file.csv                 population with columns e_score, v_score; e = ζ e_score,
                                   truncated at a valuation quantile and kernel-smoothed
                                   (--smoothing none keeps the rows as atoms); no train/test
                                   split here (that is the sweep pipeline's job)

Examples
--------
    python evaluate_threshold.py --distribution harm_rises --tau-coefficients 0.5,-1 --num-bidders 8 --num-goods 1
    python evaluate_threshold.py --distribution data/samples/g_full_tweets.csv \\
        --externality-scale 5 --tau-coefficients 1e5,-50 --num-bidders 8 --num-goods 1
    python evaluate_threshold.py --sweep-result output/sweep_x/results/run_k1_deg1_n8_ext0.pkl \\
        --distribution data/samples/h_single_modal_shifted_normal_distribution.csv
"""

import argparse
import os
import pickle

from audited_auctions import SYNTHETIC_FAMILIES, evaluate_threshold, polynomial_threshold
from audited_auctions.empirical import SMOOTHING_CHOICES, load_empirical_sample, sample_distribution
from audited_auctions.figures import plot_threshold_and_penalties, plot_welfare_distributions
from audited_auctions.type_distributions import synthetic_distribution


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--distribution", required=True,
                        help="synthetic family name or a CSV path")
    parser.add_argument("--externality-scale", type=float, default=None,
                        help="ζ: externalities are multiplied by ζ (default 1, or the sweep's ζ)")
    parser.add_argument("--valuation-scale", type=float, default=None,
                        help="v = scale × v_score for CSV populations (default 1)")
    parser.add_argument("--tau-coefficients", default=None,
                        help='comma-separated polynomial coefficients of τ, lowest degree first')
    parser.add_argument("--sweep-result", default=None,
                        help="per-run pickle from run_sweep_genetic2.py supplying τ, k, n, ζ")
    parser.add_argument("--truncation-quantile", type=float, default=0.995,
                        help="CSV: valuations above this quantile are dropped; v̄ = the quantile")
    parser.add_argument("--smoothing", choices=SMOOTHING_CHOICES, default="kernel",
                        help="CSV: kernel smoothing (default) or atomic rows")
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
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

    externality_scale = args.externality_scale or sweep_record.get("externality_cost", 1.0)
    if args.distribution in SYNTHETIC_FAMILIES:
        type_distribution = synthetic_distribution(args.distribution,
                                                   externality_scale=externality_scale)
    else:
        valuation_scale = args.valuation_scale or sweep_record.get("action_cost", 1.0)
        sample = load_empirical_sample(args.distribution, externality_scale, valuation_scale,
                                       args.truncation_quantile)
        type_distribution = sample_distribution(
            sample, args.smoothing, args.bandwidth_scale,
            name=f"{os.path.basename(args.distribution)} (ζ = {externality_scale:g})")

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
