# Collateralized Auction Optimization

This directory contains code for simulating and optimizing a *collateralized* second-price auction mechanism, where a threshold function τ (tau) gates advertiser participation based on the externality their content produces. The goal is to find the optimal τ that maximizes social welfare relative to a standard VCG auction counterfactual.

Advertiser values (`v`) are proxied by tweet engagement (actions per month) and externalities (`e`) are derived from Community Notes ratings.

---

## WEB27 Branch (`web27`): Exact Computation, IVCG-PA, Model-Consistent Distributions

The `web27` branch updates the code to the WEB27 core draft (single-parameter model with a common participation threshold τ). Everything below is documented in detail in [audited_auctions/README.md](audited_auctions/README.md).

- **[audited_auctions/](audited_auctions/)** — package. Given a type distribution and τ(e), it computes **exactly** (and simulates) VCG (no audit), **VCG-PA**, **IVCG-PA** (ironed VCG with participant audits) and the **first best** (optimal recipient audit), and the penalty functions r(e) inducing τ. Reported per mechanism: expected welfare (valuation and externality parts), gain over VCG and share of the first-best gain, revenue, expected penalty paid, participation, number allocated; per-auction distributions from simulation.
- **Model-consistent distributions.** Valuations are non-negative and bounded (enforced). Continuous distributions are piecewise uniform (valuations uniform within grid cells, externalities on grid nodes), so valuation ties have probability zero and all quantities are exact for that distribution. Synthetic families: the mixtures a–f, h (valuation means shifted by +1, truncated at 0) and new families `example_n` (analytic benchmark), `independent`, `harm_rises`, `harmful_middle` (ironing binds), `heavy_tail`.
- **Empirical pipeline.** CSV rows are split into training and test halves and truncated at a valuation quantile (v̄; default 0.995, set separately for the test rows). τ and the IVCG-PA scoring rule are designed on the kernel-smoothed training half (variance-preserving, in asinh e and log v) and evaluated exactly on the atomic test rows, with a bootstrap of exact welfare over test rows and an optional bandwidth sensitivity check.
- **Exact objective.** The genetic algorithm (`audited_auctions/genetic_search.py`) maximizes the exact expected welfare of VCG-PA (or IVCG-PA) on the distribution, not an average over simulated auction draws.
- **Welfare comparisons** are gains over unaudited VCG (VCG-PA − VCG, IVCG-PA − VCG, first best − VCG).
- **Corrected participant penalty.** The old estimator `R(e) = A(e)(τ(e) − P(e))` from realized (k+1)-th prices was biased upward; the exact r(e) = ∫_0^{τ(e)} X^τ(b) db replaces it everywhere.
- **Scripts.** [evaluate_threshold.py](evaluate_threshold.py) (one distribution and one τ → table, pickle, two figures); [Collateralized_Auction_genetic2.py](Collateralized_Auction_genetic2.py) (optimize and evaluate one cell); [run_sweep_genetic2.py](run_sweep_genetic2.py) (sweeps over ζ, degree, k, n; writes a README describing its output); [post_optimization_analysis.py](post_optimization_analysis.py) (figures from a sweep).
- **Tests:** [tests/](tests/), run with `python -m pytest tests -q` (46 tests).

Typical runs:

```bash
# synthetic family, with a check of the discretization error
python run_sweep_genetic2.py --data harm_rises --degrees 1,2 --k-values 1,2 --num-items 8 \
    --ext-min 0.5 --ext-max 2 --num-ext 4 --grid-refinement --warm-start --tag harm
# empirical data: design on smoothed training half, exact evaluation on atomic test rows,
# bootstrap, bandwidth sensitivity
python run_sweep_genetic2.py --data data/samples/g_full_tweets.csv --degrees 1,2 --k-values 1 \
    --num-items 8 --ext-min 1 --ext-max 100 --num-ext 6 --log-scale \
    --sensitivity-bandwidths 0.5,2 --tag tweets
python post_optimization_analysis.py output/sweep_genetic2_tweets_
```

Not yet done (planned): a grid representation of τ and the best-response ascent based on the first-order condition of the core draft's Appendix E; running large sweeps on the BU SCC cluster.

The sections below describe the analysis workflow (data pipeline, optimization, figures, descriptive statistics). Material from the original EC 2025 / EAAMO 2026 workflow that the WEB27 code no longer uses is in [legacy/](legacy/).

### Setup

With [uv](https://docs.astral.sh/uv/): `uv sync`. Without uv:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib scipy seaborn tqdm pygad pytest
```

---

## Current Analysis Workflow

The final analysis for the paper uses the following scripts. Start here.

### 1. Data Pipeline

**[get_tweets.ipynb](get_tweets.ipynb)**
Fetches tweets from the Twitter/X API using Tweepy and stores them in a MySQL database. Reads tweet IDs from a pickle file, batches requests in groups of 100, and respects rate limits.

**[create_distribution.ipynb](create_distribution.ipynb)**
Queries the database for tweets, Community Notes, and note ratings. Computes an externality score for each tweet by combining note classifications with helpfulness ratings, normalizes by impression count, and exports `full_tweets.csv` — the primary data input for the optimizer.

**[full_tweets.csv](full_tweets.csv)**
Processed tweet dataset. Each row is a tweet with `v_score` (advertiser value, proportional to actions per month) and `e_score` (externality, normalized Community Notes signal per month).

### 2. Optimization Sweep

**[Collateralized_Auction_genetic2.py](Collateralized_Auction_genetic2.py)**
Single-cell run: genetic search (`audited_auctions/genetic_search.py`) for the polynomial τ maximizing exact expected welfare, then exact evaluation, simulation and (empirical data) bootstrap via `audited_auctions/pipeline.py`. `--data` is a CSV or a synthetic family name.

**[run_sweep_genetic2.py](run_sweep_genetic2.py)**
Sweep runner. Iterates over all combinations of `k` (goods), polynomial degree, number of bidders `n`, and externality scale ζ, running one pipeline cell each. Results are saved to `output/sweep_genetic2_<tag>_<timestamp>/sweep_data.pkl` along with a per-run `README.md`.

```bash
python run_sweep_genetic2.py --tag myrun --data data/samples/g_full_tweets.csv
```

### 3. Analysis and Figures

**[post_optimization_analysis.py](post_optimization_analysis.py)**
Loads a completed sweep and generates all publication-ready figures. Accepts a sweep directory or path prefix; when given a prefix, processes all matching directories.

Key flags:
- `--empirical` — empirical mode; zooms axes to central percentiles of the types
- `--no-ci` — omit bootstrap 95% interval error bars
- `--font-scale N` — scale all figure text (default 1.2)

Individual plots can be generated selectively to avoid regenerating everything:
```bash
python post_optimization_analysis.py output/sweep_myrun --welfare --penalty-by-degree
python post_optimization_analysis.py output/sweep_myrun --tau-by-k --empirical
```

Available plot flags: `--welfare-dist`, `--tau-degree`, `--tau-grid`, `--tau-grid-zoomed`, `--tau-by-degree`, `--tau-by-k`, `--tau-by-n`, `--penalty-grid`, `--penalty-by-degree`, `--welfare`, `--welfare-kn`.

### 4. Descriptive Statistics

**[create_descriptive_stats.py](create_descriptive_stats.py)**
Computes and saves summary statistics, correlation tables, and figures for `full_tweets.csv`. Outputs to `output/descriptive_stats/`.

**[csv_to_latex.py](csv_to_latex.py)**
Converts the CSV tables produced by `create_descriptive_stats.py` into booktabs-formatted LaTeX table environments. Writes a single `.tex` file that can be `\input{}`-ed directly.

```bash
python csv_to_latex.py --input-dir output/descriptive_stats/tables
```

### 5. Supplemental

**[legacy/supplemental/](legacy/supplemental/)**
LaTeX source for the EC 2025 supplemental document, including the methodology and data collection description. **[value_externality_definitions/](value_externality_definitions/)** defines the valuation and externality scores of the data.

---

## Complete Workflow

```
get_tweets.ipynb              → MySQL DB
create_distribution.ipynb     → full_tweets.csv
create_descriptive_stats.py   → output/descriptive_stats/

run_sweep_genetic2.py
  └─ audited_auctions/pipeline.py      → output/sweep_genetic2_<tag>_<timestamp>/sweep_data.pkl

post_optimization_analysis.py → output/sweep_<name>/figures/

csv_to_latex.py               → output/descriptive_stats/tables/tables.tex
```

---

---

## Legacy Material

Superseded scripts, notebooks, synthetic samples and sweep outputs are in [legacy/](legacy/), described in [legacy/README.md](legacy/README.md).
