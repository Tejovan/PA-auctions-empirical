#!/usr/bin/env python3
"""
run_sweep_genetic2.py

Sweep runner for the τ optimization. Sweeps over the externality scale ζ,
polynomial degrees, k (goods) and n (bidders). Each (k, degree, n, ζ) cell runs
audited_auctions.pipeline:

  1. Distributions. --data is a synthetic family name (H known; τ optimized and
     evaluated on the same piecewise-uniform distribution) or a CSV with
     columns e_score, v_score (empirical: split into training and test halves
     with a seed common to all cells, truncated at a valuation quantile v̄;
     training half kernel-smoothed, test half atomic by default).
  2. Search (with restarts and optional warm start across degrees) for the
     polynomial τ maximizing the exact expected welfare of VCG-PA (or IVCG-PA,
     --objective) on the training distribution: genetic stage, then
     Nelder–Mead refinement (default; --local-refinement none to skip).
  3. Exact evaluation on the evaluation distribution of VCG, VCG-PA, IVCG-PA and
     the first best: welfare, gains over VCG and share of the first-best gain,
     revenue, penalty paid, participation, and the exact penalties r(e)
     inducing τ.
  4. One simulation (--test-auctions draws, common random numbers across
     mechanisms, k and degree) for per-auction welfare distributions.
  5. Empirical data: the IVCG-PA scoring rule is designed on the training half
     and applied to the test rows; bootstrap of exact welfare over the test
     rows (--bootstrap-samples); optional bandwidth sensitivity check
     (--sensitivity-bandwidths 0.5,2). Synthetic families: optional
     grid-refinement check (--grid-refinement).

Output goes to a timestamped subdirectory with a README describing the files.

Usage:
    python run_sweep_genetic2.py --data harm_rises --degrees 1,2 --k-values 1,2 \
        --num-items 8 --ext-min 0.5 --ext-max 2 --num-ext 4 --tag harm
    python run_sweep_genetic2.py --data data/samples/g_full_tweets.csv \
        --degrees 1,2,3 --k-values 1,2 --num-items 8,16 \
        --ext-min 1 --ext-max 100 --num-ext 8 --log-scale --num-restarts 3 \
        --warm-start --tag tweets

Arguments: see --help. ζ multiplies the externalities of every source.
"""

import argparse
import datetime
import math
import os
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from audited_auctions.empirical import SMOOTHING_CHOICES  # noqa: E402
from audited_auctions.figures import (  # noqa: E402
    plot_threshold_and_penalties,
    plot_welfare_distributions as plot_welfare_distribution_densities,
)
from audited_auctions.pipeline import (  # noqa: E402
    is_synthetic_source,
    optimize_and_evaluate,
    prepare_distributions,
)
from audited_auctions.sweep_support import (  # noqa: E402
    SWEEP_SERIES,
    record_exact_welfare,
    record_penalty_curves,
    record_type_sample,
    sweep_record_fields,
)


# ─── One sweep cell ───────────────────────────────────────────────────────────


def run_one(prepared, k, degree, ext_cost, action_cost, num_items, ga_options, ga_seed,
            num_restarts, initial_solution, simulation_seed, num_test_auctions,
            objective_mechanism, num_bootstrap, bootstrap_seed, grid_refinement,
            sensitivity_bandwidth_scales=()):
    """Optimize τ for one cell and evaluate it; returns the sweep record."""
    print(f"\n{'=' * 60}")
    print(f"  k={k}  degree={degree}  ζ={ext_cost:.6g}  n={num_items}  ga_seed={ga_seed}  "
          f"simulation_seed={simulation_seed}  restarts={num_restarts}")
    if initial_solution is not None:
        print(f"  warm start (restart 0): {[f'{c:.4g}' for c in initial_solution]}")
    print(f"{'=' * 60}")
    record = optimize_and_evaluate(
        prepared, degree, num_items, k, objective_mechanism=objective_mechanism,
        ga_options=ga_options, ga_seed=ga_seed, num_restarts=num_restarts,
        initial_solution=initial_solution, num_simulated_draws=num_test_auctions,
        simulation_seed=simulation_seed, num_bootstrap=num_bootstrap,
        bootstrap_seed=bootstrap_seed, grid_refinement=grid_refinement,
        sensitivity_bandwidth_scales=sensitivity_bandwidth_scales)
    print(record["evaluation"].summary_table()[
        ["exact_welfare", "exact_welfare_gain_over_vcg", "share_of_first_best_gain"]
    ].to_string(float_format=lambda x: f"{x:.5g}"))
    return {
        **record,
        # Keys used by the plotting code (names kept from earlier sweeps).
        "k": k,
        "num_items": num_items,
        "externality_cost": ext_cost,
        "action_cost": action_cost,
        "ga_welfare": record["training_welfare"],
        "num_test_auctions": num_test_auctions,
        # vcg_*, vcga_*, ivcga_*, ra_* per-draw lists from the simulation
        **sweep_record_fields(record["evaluation"]),
    }


# ─── Plot helpers ─────────────────────────────────────────────────────────────


def _eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def _scatter_sample(all_e, all_v, ax, rng, n=1000):
    idx = rng.choice(len(all_e), size=min(n, len(all_e)), replace=False)
    ax.scatter(all_e[idx], all_v[idx], s=4, alpha=0.25, color="steelblue", zorder=1)


def _e_range(all_e, percentile=95):
    return float(np.percentile(all_e, 100 - percentile)), float(
        np.percentile(all_e, percentile)
    )


def _grid_axes(n):
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    return ncols, nrows


def _hide_unused(axs, n, nrows, ncols):
    for idx in range(n, nrows * ncols):
        axs[idx // ncols][idx % ncols].set_visible(False)


# ─── Figures ──────────────────────────────────────────────────────────────────


def plot_tau_grid(records, out_dir, tag, max_fns=200, zoomed=False, percentile=99):
    n = len(records)
    ncols, nrows = _grid_axes(n)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    rng = np.random.default_rng(0)

    for idx, rec in enumerate(records):
        ax = axs[idx // ncols][idx % ncols]
        all_e, all_v = record_type_sample(rec)

        if zoomed:
            e_lo, e_hi = _e_range(all_e, percentile)
            v_hi = float(np.percentile(all_v, percentile))
        else:
            e_lo, e_hi = float(all_e.min()), float(all_e.max())
            v_hi = float(all_v.max())

        _scatter_sample(all_e, all_v, ax, rng)

        e_line = np.linspace(e_lo, e_hi, 300)
        fns = rec["trajectory"]
        if len(fns) > max_fns:
            chosen = rng.choice(len(fns), size=max_fns, replace=False)
            fns = [fns[i] for i in chosen]
        for coeffs in fns:
            ax.plot(
                e_line,
                [_eval_poly(coeffs, e) for e in e_line],
                color="gray",
                alpha=0.12,
                linewidth=0.5,
                zorder=2,
            )

        y_opt = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
        ax.plot(e_line, y_opt, color="tab:red", linewidth=2.0, zorder=3)

        ax.set_xlim(e_lo, e_hi)
        ax.set_ylim(0, v_hi * 1.05)
        ax.set_title(f"ζ={rec['externality_cost']:.4g}", fontsize=9)
        ax.set_xlabel("Externality e", fontsize=8)
        ax.set_ylabel("τ threshold", fontsize=8)
        ax.tick_params(labelsize=7)

    _hide_unused(axs, n, nrows, ncols)
    suffix = " (zoomed)" if zoomed else ""
    fig.suptitle(
        f"Tested & Optimal τ{suffix}  "
        f"[k={records[0]['k']}, degree={records[0]['polynomial_degree']}, n={records[0]['num_items']}]",
        fontsize=12,
    )
    fig.tight_layout()
    fname = f"tau_grid_zoomed_{tag}.png" if zoomed else f"tau_grid_{tag}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_penalty_grid(records, out_dir, tag, percentile=99):
    """Grid of penalty plots: tau + R(e) + -e line over joint distribution scatter."""
    n = len(records)
    ncols, nrows = _grid_axes(n)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    legend_done = False

    for idx, rec in enumerate(records):
        ax = axs[idx // ncols][idx % ncols]
        all_e, all_v = record_type_sample(rec)

        e_lo, e_hi = _e_range(all_e, percentile)
        v_hi = float(np.percentile(all_v, percentile))

        _scatter_sample(all_e, all_v, ax, rng)

        e_line = np.linspace(e_lo, e_hi, 300)

        y_tau = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
        penalties = record_penalty_curves(rec, e_line)  # exact r(e) inducing tau
        neg_e = -e_line

        (p1,) = ax.plot(e_line, y_tau, color="black", linewidth=2.0, zorder=3)
        (p2,) = ax.plot(e_line, penalties["vcg_pa"], color="#2a78d6", linewidth=1.8, zorder=4)
        (p2b,) = ax.plot(
            e_line, penalties["ivcg_pa"], color="#c0392b", linewidth=1.8, linestyle="--", zorder=5
        )
        (p3,) = ax.plot(
            e_line, neg_e, color="#1baf7a", linewidth=1.5, linestyle="-.", zorder=3
        )

        # Y limits: include all curves and the scatter body
        y_all = np.concatenate([y_tau, penalties["vcg_pa"], neg_e, all_v])
        y_lo_plot = min(float(np.percentile(y_all, 1)), 0.0)
        y_hi_plot = max(float(np.percentile(y_all, 99)), v_hi) * 1.05
        ax.set_xlim(e_lo, e_hi)
        ax.set_ylim(y_lo_plot, y_hi_plot)

        ax.set_title(f"ζ={rec['externality_cost']:.4g}", fontsize=9)
        ax.set_xlabel("Externality e", fontsize=8)
        ax.set_ylabel("Value / Penalty", fontsize=8)
        ax.tick_params(labelsize=7)

        if not legend_done:
            ax.legend(
                [p1, p2, p2b, p3],
                [
                    "τ (optimal threshold)",
                    "r(e) for VCG-PA",
                    "r(e) for IVCG-PA",
                    "−e (recipient penalty)",
                ],
                fontsize=6,
                loc="best",
            )
            legend_done = True

    _hide_unused(axs, n, nrows, ncols)
    fig.suptitle(
        f"Threshold & Penalty Curves  "
        f"[k={records[0]['k']}, degree={records[0]['polynomial_degree']}, n={records[0]['num_items']}]",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"penalty_grid_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_welfare_comparison(records, out_dir, tag):
    """Exact expected welfare of VCG, VCG-PA, IVCG-PA and the first best vs ζ,
    with bootstrap 95% intervals when the records have them (empirical data)."""
    ext_costs = [r["externality_cost"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, label, color, marker in SWEEP_SERIES:
        values = [record_exact_welfare(r)[prefix] for r in records]
        means = [value["welfare"] for value in values]
        ax.plot(ext_costs, means, marker=marker, linewidth=1.8, label=label, color=color)
        if all(value["lower"] is not None for value in values):
            ax.fill_between(ext_costs, [v["lower"] for v in values], [v["upper"] for v in values],
                            color=color, alpha=0.15, linewidth=0)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Externality scale ζ", fontsize=11)
    ax.set_ylabel("Expected welfare (exact)", fontsize=11)
    ax.set_title(
        f"Expected Welfare vs Externality Scale\n"
        f"k={records[0]['k']}, degree={records[0]['polynomial_degree']}, n={records[0]['num_items']}",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    if len(ext_costs) > 1:
        span = max(ext_costs) / max(min(ext_costs), 1e-12)
        if span > 100:
            ax.set_xscale("log")

    fig.tight_layout()
    path = os.path.join(out_dir, f"welfare_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cell_evaluation_figures(records, prepared_by_ext, out_dir, tag):
    """For each record: the threshold/penalty figure and the welfare-density figure."""
    for ext_idx, rec in enumerate(records):
        type_distribution = prepared_by_ext[ext_idx].evaluation
        evaluation = rec["evaluation"]
        path = os.path.join(out_dir, f"threshold_penalties_{tag}_ext{ext_idx}.png")
        plot_threshold_and_penalties(evaluation, type_distribution, path)
        print(f"Saved: {path}")
        path = os.path.join(out_dir, f"welfare_densities_{tag}_ext{ext_idx}.png")
        plot_welfare_distribution_densities(evaluation, path)
        print(f"Saved: {path}")


def plot_tau_degree_comparison(
    all_results, degrees_ordered, fig_dir, k_values, num_items_list, percentile=99
):
    """For each (k, n): one subplot per ext_cost, overlaying optimal tau per degree."""
    _palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:cyan",
    ]
    degree_colors = [_palette[i % len(_palette)] for i in range(len(degrees_ordered))]

    for k in k_values:
        for num_items in num_items_list:
            avail_degrees = [
                d for d in degrees_ordered if (k, d, num_items) in all_results
            ]
            if not avail_degrees:
                continue

            first_deg = avail_degrees[0]
            ext_recs = all_results[(k, first_deg, num_items)]
            num_ext = len(ext_recs)
            ncols, nrows = _grid_axes(num_ext)
            fig, axs = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
            )
            rng = np.random.default_rng(0)

            for ext_idx in range(num_ext):
                ax = axs[ext_idx // ncols][ext_idx % ncols]
                ref_rec = all_results[(k, first_deg, num_items)][ext_idx]
                all_e, all_v = record_type_sample(ref_rec)
                e_lo, e_hi = _e_range(all_e, percentile)
                v_hi = float(np.percentile(all_v, percentile))

                _scatter_sample(all_e, all_v, ax, rng)
                e_line = np.linspace(e_lo, e_hi, 300)

                for d_idx, degree in enumerate(avail_degrees):
                    rec = all_results[(k, degree, num_items)][ext_idx]
                    y = np.array([_eval_poly(rec["tau_coeffs"], e) for e in e_line])
                    ax.plot(
                        e_line,
                        y,
                        color=degree_colors[d_idx],
                        linewidth=2.0,
                        label=f"deg {degree}",
                        zorder=3,
                    )

                ax.set_xlim(e_lo, e_hi)
                ax.set_ylim(0, v_hi * 1.05)
                ax.set_title(f"ζ={ref_rec['externality_cost']:.4g}", fontsize=9)
                ax.set_xlabel("e", fontsize=8)
                ax.set_ylabel("τ", fontsize=8)
                ax.tick_params(labelsize=7)
                if ext_idx == 0:
                    ax.legend(fontsize=7, loc="best")

            _hide_unused(axs, num_ext, nrows, ncols)
            fig.suptitle(f"Optimal τ by Degree  [k={k}, n={num_items}]", fontsize=12)
            fig.tight_layout()
            path = os.path.join(fig_dir, f"tau_degree_comparison_k{k}_n{num_items}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved: {path}")


def plot_welfare_distributions(all_results, fig_dir):
    """For each (k, degree, n): grid of rows=ext_costs × cols=[total, adv, ext] histograms."""
    col_specs = [
        ("welfare", "Total Welfare"),
        ("adv_welfare", "Advertiser Welfare (v)"),
        ("ext_welfare", "Externality Welfare (e)"),
    ]

    for (k, degree, num_items), records in all_results.items():
        records_sorted = sorted(records, key=lambda r: r["externality_cost"])
        num_ext = len(records_sorted)

        fig, axs = plt.subplots(num_ext, 3, figsize=(12, 4 * num_ext), squeeze=False)

        for row_idx, rec in enumerate(records_sorted):
            ext_cost = rec["externality_cost"]
            for col_idx, (suffix, col_title) in enumerate(col_specs):
                ax = axs[row_idx][col_idx]
                series_values = [np.array(rec[f"{prefix}_{suffix}"]) for prefix, *_ in SWEEP_SERIES]

                all_vals = np.concatenate(series_values)
                vmin = float(np.percentile(all_vals, 1))
                vmax = float(np.percentile(all_vals, 99))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                bins = np.linspace(vmin, vmax, 40)

                for values, (_, label, color, _) in zip(series_values, SWEEP_SERIES):
                    ax.hist(values, bins=bins, histtype="step", linewidth=1.4, color=color, label=label)

                ax.set_title(f"ζ={ext_cost:.4g} — {col_title}", fontsize=8)
                ax.tick_params(labelsize=7)
                if col_idx == 0:
                    ax.set_ylabel("Count", fontsize=7)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

        fig.suptitle(
            f"Welfare Distributions  [k={k}, degree={degree}, n={num_items}]",
            fontsize=13,
        )
        fig.tight_layout()
        path = os.path.join(fig_dir, f"welfare_dist_k{k}_deg{degree}_n{num_items}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


# ─── README ───────────────────────────────────────────────────────────────────


def write_readme(out_dir, args, ext_values, timestamp):
    dir_name = os.path.basename(out_dir)
    ext_list = [f"{v:.4g}" for v in ext_values]
    empirical = not is_synthetic_source(args.data)
    data_rows = (
        f"""| Source | empirical CSV `{args.data}` |
| Valuation scale (action cost) | {args.action_cost} |
| Truncation quantile (v̄) | {args.truncation_quantile} |
| Train fraction / split seed | {args.train_fraction} / {args.split_seed} |
| Grid (externality nodes × valuation cells) | {args.num_externality_nodes or "default"} × {args.num_valuation_cells or "default"} |
| Training smoothing | {args.smoothing} (bandwidth scale {args.bandwidth_scale}) |
| Evaluation | {args.evaluation} test rows (truncation {args.evaluation_truncation_quantile or args.truncation_quantile}); IVCG-PA scoring rule designed on the training half |
| Bandwidth sensitivity scales | {args.sensitivity_bandwidths or "none"} |
| Bootstrap resamples | {args.bootstrap_samples} |"""
        if empirical else
        f"""| Source | synthetic family `{args.data}` (piecewise-uniform grid) |
| Grid-refinement check | {args.grid_refinement} |"""
    )
    content = f"""\
# τ Optimization Sweep

**Generated:** {timestamp}
**Tag:** `{args.tag}`

---

## Parameters

| Parameter | Value |
|-----------|-------|
{data_rows}
| Objective (exact expected welfare of) | `{args.objective}` |
| k values | `{args.k_values}` |
| Polynomial degrees | `{args.degrees}` |
| Bidders n | `{args.num_items}` |
| Externality scale ζ | [{args.ext_min}, {args.ext_max}] ({args.num_ext} values, {"log" if args.log_scale else "linear"} spacing): `{ext_list}` |
| Simulated auctions per cell | {args.test_auctions} |
| Base random seed | {args.seed} |
| GA generations / population / parents | {args.num_generations} / {args.sol_per_pop} / {args.num_parents_mating} |
| GA mutation probability / range multiplier | {args.mutation_probability} / {args.range_multiplier} |
| GA restarts / warm start | {args.num_restarts} / {args.warm_start} |
| Local refinement / stall generations / starts / max evaluations per start | {args.local_refinement} / {args.stall_generations} / {args.num_local_starts} / {args.local_max_evaluations} |

---

## Directory Structure

```
{dir_name}/
├── results/
│   └── run_k{{k}}_deg{{d}}_n{{n}}_ext{{i}}.pkl        # one record per (k, degree, n, ζ index)
├── figures/
│   ├── tau_grid_k{{k}}_deg{{d}}_n{{n}}.png             # GA candidates and optimal τ over a type sample
│   ├── tau_grid_zoomed_k{{k}}_deg{{d}}_n{{n}}.png      # same, central 98% of types
│   ├── penalty_grid_k{{k}}_deg{{d}}_n{{n}}.png         # τ, exact r(e) for VCG-PA and IVCG-PA, −e
│   ├── welfare_k{{k}}_deg{{d}}_n{{n}}.png              # exact expected welfare by mechanism vs ζ
│   ├── threshold_penalties_k{{k}}_deg{{d}}_n{{n}}_ext{{i}}.png  # types + τ, penalties, exact welfare
│   └── welfare_densities_k{{k}}_deg{{d}}_n{{n}}_ext{{i}}.png    # per-auction welfare, paired gains over VCG
├── sweep_data.pkl                                # all records, keyed by (k, degree, n)
└── README.md                                     # this file
```

---

## Mechanisms

All mechanisms allocate only to positive scores, break ties uniformly at random,
and charge critical-value payments.

| Prefix | Mechanism | Participants | Ranking score |
|--------|-----------|--------------|---------------|
| `vcg` | VCG without audits | all bidders | valuation v |
| `vcga` | VCG-PA (participant audit) | types with v ≥ τ(e) | bid = v |
| `ivcga` | IVCG-PA (ironed VCG, participant audit) | types with v ≥ τ(e) | ironed welfare contribution ω̄^τ(v) |
| `ra` | first best (recipient audit, penalty −e) | all bidders | welfare contribution w = v + e |

---

## Per-Run Result Files

Each `results/run_k{{k}}_deg{{d}}_n{{n}}_ext{{i}}.pkl` holds one `dict`:

| Field | Description |
|-------|-------------|
| `k`, `num_items`, `polynomial_degree`, `externality_cost`, `action_cost` | cell coordinates (k goods, n bidders, degree, ζ, valuation scale) |
| `objective_mechanism` | mechanism whose exact welfare the GA maximized |
| `tau_coeffs` | optimal τ coefficients [β₀, β₁, …] (τ(e) = Σ β_m e^m) |
| `training_welfare` (= `ga_welfare`) | exact objective at the optimum on the training distribution |
| `trajectory` | every coefficient vector in every GA generation |
| `search_summaries` | per restart: genetic optimum, generations run, distinct objective evaluations, and per Nelder–Mead start the starting and final welfare, evaluations and convergence flag |
| `evaluation` | `ThresholdEvaluation` on the evaluation distribution: exact outcomes (`exact`), `summary_table()`, `welfare_gains_over_vcg()`, exact penalty functions (`penalty_tables`, `penalty_curves`), ŵ^τ and ω̄^τ, simulated draws |
| `bootstrap` | empirical data: `BootstrapResult` (`resamples` DataFrame and `summary()` with percentile intervals) of exact welfare, paired gains over VCG and shares of the first-best gain over test-row resamples; `None` for synthetic families |
| `bandwidth_sensitivity` | empirical data with `--sensitivity-bandwidths`: DataFrame indexed by bandwidth scale (base included): τ coefficients, training objective, exact test welfare, gains over VCG and shares of the first-best gain of the design made with that bandwidth; else `None` |
| `grid_refinement` | synthetic families with `--grid-refinement`: DataFrame of exact welfare and gains on the base and twice finer grid; else `None` |
| `type_sample` | 4000 types drawn from the evaluation distribution (for scatter plots) |
| `distribution_settings` | how the distributions were built (family/CSV, ζ, truncation, split, smoothing, grid) |
| `{{prefix}}_welfare`, `_adv_welfare`, `_ext_welfare`, `_revenue`, `_penalty_paid`, `_num_participants`, `_num_allocated` | per-draw simulated outcomes (lists) on `num_test_auctions` auctions |
| `ga_seed`, `simulation_seed`, `bootstrap_seed`, `num_restarts` | seeds and restarts |

Welfare comparisons are gains over unaudited VCG. Allocations in simulated draws are
probabilities (the tie-breaking lottery is averaged out). Bidders are i.i.d. draws
from the evaluation distribution.

---

## Consolidated Data File (`sweep_data.pkl`)

A `dict` keyed by `(k, degree, num_items)`; each value is the list of records sorted by ζ.

```python
import pickle
with open('sweep_data.pkl', 'rb') as f:
    data = pickle.load(f)
record = data[(1, 2, 8)][0]
print(record["evaluation"].summary_table())        # exact outcomes per mechanism
if record["bootstrap"] is not None:
    print(record["bootstrap"].summary())           # bootstrap intervals (empirical data)
```

---

## Figures

- `tau_grid_*`: sample of types (blue), up to 200 τ candidates evaluated by the GA (gray), optimal τ (red).
- `penalty_grid_*`: optimal τ (black) and the exact penalty functions inducing it, r(e) = ∫_0^τ(e) X^τ(b) db (blue: VCG-PA; red dashed: IVCG-PA), and the recipient penalty −e (green).
- `welfare_*`: exact expected welfare vs ζ for the four mechanisms (with bootstrap 95% intervals for empirical data).
- `threshold_penalties_*`, `welfare_densities_*`: see `audited_auctions/README.md` (Figures).
"""
    path = os.path.join(out_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--data", required=True,
                   help="synthetic family name or CSV with columns e_score, v_score")
    p.add_argument("--action-cost", type=float, default=1.0,
                   help="CSV data: v = action_cost × v_score (default: 1)")
    p.add_argument("--truncation-quantile", type=float, default=0.995,
                   help="CSV data: valuations above this quantile are dropped; v̄ = the quantile")
    p.add_argument("--train-fraction", type=float, default=0.5,
                   help="CSV data: share of rows used to optimize τ (default: 0.5)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="CSV data: seed of the train/test split, common to all cells")
    p.add_argument("--num-externality-nodes", type=int, default=None,
                   help="number of externality nodes of the grid (default: 101 for synthetic families, 241 for smoothed CSV data)")
    p.add_argument("--num-valuation-cells", type=int, default=None,
                   help="number of valuation cells of the grid (default: 600 for synthetic families, 400 for smoothed CSV data)")
    p.add_argument("--smoothing", choices=SMOOTHING_CHOICES, default="kernel",
                   help="CSV data: kernel smoothing (default) or atomic rows")
    p.add_argument("--bandwidth-scale", type=float, default=1.0,
                   help="CSV data: multiplier on Scott's bandwidth")
    p.add_argument("--evaluation", choices=("atomic", "smoothed"), default="atomic",
                   help="CSV data: evaluate on the atomic test rows (default) or the smoothed test half")
    p.add_argument("--evaluation-truncation-quantile", type=float, default=None,
                   help="CSV data: truncation of the test rows (default: same as "
                        "--truncation-quantile; 1.0 keeps every row)")
    p.add_argument("--sensitivity-bandwidths", default="",
                   help='CSV data: extra bandwidth scales for the sensitivity check, e.g. "0.5,2" '
                        "(reruns the GA per scale; default: none)")
    p.add_argument("--bootstrap-samples", type=int, default=200,
                   help="CSV data: bootstrap resamples of the test rows (0 = none)")
    p.add_argument("--grid-refinement", action="store_true",
                   help="synthetic families: evaluate τ also on a twice finer grid")

    # Sweep dimensions
    p.add_argument("--degrees", default="1", help='polynomial degrees, e.g. "1,2,3"')
    p.add_argument("--k-values", default="1", help='numbers of goods k, e.g. "1,2"')
    p.add_argument("--num-items", default="20", help='numbers of bidders n, e.g. "8,16"')
    p.add_argument("--ext-min", type=float, default=1.0, help="smallest ζ (default: 1)")
    p.add_argument("--ext-max", type=float, default=1.0, help="largest ζ (default: 1)")
    p.add_argument("--num-ext", type=int, default=1, help="number of ζ values (default: 1)")
    p.add_argument("--log-scale", action="store_true", help="log spacing of ζ")

    # Optimization
    p.add_argument("--objective", choices=("vcg_pa", "ivcg_pa"), default="vcg_pa",
                   help="mechanism whose exact expected welfare the GA maximizes")
    p.add_argument("--num-generations", type=int, default=500)
    p.add_argument("--sol-per-pop", type=int, default=50)
    p.add_argument("--num-parents-mating", type=int, default=15)
    p.add_argument("--mutation-probability", type=float, default=0.25)
    p.add_argument("--range-multiplier", type=float, default=3.0)
    p.add_argument("--local-refinement", choices=("nelder_mead", "none"),
                   default="nelder_mead",
                   help="Nelder–Mead refinement after the genetic stage (default) or none")
    p.add_argument("--stall-generations", type=int, default=60,
                   help="with refinement: end the genetic stage after this many generations "
                        "without improvement (default: 60)")
    p.add_argument("--num-local-starts", type=int, default=5,
                   help="Nelder–Mead starts from the best distinct final candidates (default: 5)")
    p.add_argument("--local-max-evaluations", type=int, default=2000,
                   help="objective evaluations allowed per Nelder–Mead start (default: 2000)")
    p.add_argument("--num-restarts", type=int, default=1,
                   help="independent GA restarts per cell; the best is kept")
    p.add_argument("--warm-start", action="store_true",
                   help="seed the degree-d search with the degree-(d−1) optimum")

    # Evaluation
    p.add_argument("--test-auctions", type=int, default=2000,
                   help="simulated auctions per cell for welfare distributions")
    p.add_argument("--seed", type=int, default=1234, help="base random seed")

    # Output
    p.add_argument("--tag", default="sweep", help="label in the output directory name")
    p.add_argument("--output-base", default="output", help="parent output directory")
    p.add_argument("--no-plot", action="store_true", help="skip figures")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    t_start = time.time()

    degrees = [int(d.strip()) for d in args.degrees.split(",")]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    num_items_list = [int(n.strip()) for n in args.num_items.split(",")]
    if args.log_scale:
        ext_values = list(np.logspace(np.log10(args.ext_min), np.log10(args.ext_max), args.num_ext))
    else:
        ext_values = list(np.linspace(args.ext_min, args.ext_max, args.num_ext))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_base, f"sweep_genetic2_{args.tag}_{timestamp}")
    res_dir = os.path.join(out_dir, "results")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Output directory : {out_dir}")
    print(f"k values         : {k_values}")
    print(f"Degrees          : {degrees}")
    print(f"Bidders n        : {num_items_list}")
    print(f"ζ values         : {[f'{v:.4g}' for v in ext_values]}")
    write_readme(out_dir, args, ext_values, timestamp)

    # Distributions depend only on ζ: build them once per ζ.
    prepared_by_ext = [
        prepare_distributions(
            args.data, externality_scale=ext_cost, valuation_scale=args.action_cost,
            truncation_quantile=args.truncation_quantile, train_fraction=args.train_fraction,
            split_seed=args.split_seed, smoothing=args.smoothing,
            bandwidth_scale=args.bandwidth_scale, evaluation=args.evaluation,
            evaluation_truncation_quantile=args.evaluation_truncation_quantile,
            num_externality_nodes=args.num_externality_nodes,
            num_valuation_cells=args.num_valuation_cells)
        for ext_cost in ext_values
    ]
    print(f"Evaluation distribution: {prepared_by_ext[0].evaluation.name} "
          f"({prepared_by_ext[0].evaluation.num_cells:,} cells)")

    ga_options = {"num_generations": args.num_generations, "sol_per_pop": args.sol_per_pop,
                  "num_parents_mating": args.num_parents_mating,
                  "mutation_probability": args.mutation_probability,
                  "range_multiplier": args.range_multiplier,
                  "local_refinement": args.local_refinement,
                  "stall_generations": args.stall_generations,
                  "num_local_starts": args.num_local_starts,
                  "local_max_evaluations": args.local_max_evaluations}
    sensitivity_scales = [float(x) for x in args.sensitivity_bandwidths.split(",") if x.strip()]
    degrees_ordered = sorted(degrees) if args.warm_start else degrees
    valid_kn = [(k, n) for k in k_values for n in num_items_list if k < n]
    skipped_kn = [(k, n) for k in k_values for n in num_items_list if k >= n]
    if skipped_kn:
        print(f"Skipping (k, n) pairs with k >= n: {skipped_kn}")
    total_cells = len(valid_kn) * len(degrees_ordered) * len(ext_values)
    all_results = {}
    cell = 0

    for k in k_values:
        for n_idx, num_items in enumerate(num_items_list):
            if k >= num_items:
                continue
            for degree in degrees_ordered:
                all_results[(k, degree, num_items)] = []
            for ext_idx, ext_cost in enumerate(ext_values):
                previous_coefficients = None  # warm start carries across degrees within one ζ
                for degree in degrees_ordered:
                    cell += 1
                    ga_seed = args.seed + ext_idx + degree * 1000 + k * 100000 + n_idx * 10000000
                    # Same simulated auctions for all k and degrees at a given (n, ζ).
                    simulation_seed = args.seed + 1_000_000 + ext_idx + n_idx * 10000000
                    bootstrap_seed = args.seed + 2_000_000 + ext_idx
                    initial_solution = (previous_coefficients + [0.0]
                                        if args.warm_start and previous_coefficients else None)
                    print(f"\n[{cell}/{total_cells}]")
                    rec = run_one(
                        prepared_by_ext[ext_idx], k, degree, ext_cost, args.action_cost, num_items,
                        ga_options, ga_seed, args.num_restarts, initial_solution, simulation_seed,
                        args.test_auctions, args.objective, args.bootstrap_samples,
                        bootstrap_seed, args.grid_refinement, sensitivity_scales)
                    all_results[(k, degree, num_items)].append(rec)
                    previous_coefficients = rec["tau_coeffs"]
                    fname = f"run_k{k}_deg{degree}_n{num_items}_ext{ext_idx}.pkl"
                    with open(os.path.join(res_dir, fname), "wb") as f:
                        pickle.dump(rec, f)
                    print(f"  -> saved {fname}")

    consolidated = os.path.join(out_dir, "sweep_data.pkl")
    with open(consolidated, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nConsolidated data saved: {consolidated}")

    if not args.no_plot:
        print("\nGenerating figures...")
        for (k, degree, num_items), records in all_results.items():
            records_sorted = sorted(records, key=lambda r: r["externality_cost"])
            tag_kdn = f"k{k}_deg{degree}_n{num_items}"
            plot_tau_grid(records_sorted, fig_dir, tag_kdn, zoomed=False)
            plot_tau_grid(records_sorted, fig_dir, tag_kdn, zoomed=True)
            plot_penalty_grid(records_sorted, fig_dir, tag_kdn)
            plot_welfare_comparison(records_sorted, fig_dir, tag_kdn)
            plot_cell_evaluation_figures(records_sorted, prepared_by_ext, fig_dir, tag_kdn)
        plot_tau_degree_comparison(all_results, degrees_ordered, fig_dir, k_values,
                                   num_items_list)
        plot_welfare_distributions(all_results, fig_dir)

    elapsed = time.time() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal time: {h:02d}:{m:02d}:{s:02d}  ({elapsed:.1f}s)")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
