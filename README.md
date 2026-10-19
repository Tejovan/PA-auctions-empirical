# Audited auctions: computation, optimization and experiments

Code for the WEB27 paper on auctions with participant audits. The model is in the paper (`WEB27/arxiv_WEB/` of the paper repository): n bidders, k identical goods, types (e, v), with externality e and valuation v.

For a type distribution and a participation threshold function τ(e) common to all bidders, the package computes four mechanisms exactly:

- VCG without audits;
- VCG-PA;
- IVCG-PA, the ironed VCG auction with participant audits;
- the first best, implemented by the optimal recipient audit.

It also computes:

- the penalties r(e) that induce τ;
- simulated per-auction outcomes;
- for empirical data, bootstrap intervals;
- the τ that maximizes exact expected welfare, found by genetic search or best-response ascent.

Methodology and output formats are documented in the supplement `WEB27/drafting/code_supplement/` of the paper repository. Module-level documentation is in [audited_auctions/README.md](audited_auctions/README.md), and every configuration key is in [configs/README.md](configs/README.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas matplotlib pygad pytest
```

With [uv](https://docs.astral.sh/uv/): `uv sync`. Python ≥ 3.12, which provides `tomllib` for the configuration files.

## Running experiments

Every experiment is a **run**, described by a TOML configuration file:

```bash
python run.py configs/standard_test.toml            # plan + time estimate, asks, runs, draws figures
python run.py configs/standard_test.toml --dry-run  # plan and time estimate only
python run.py CONFIG.toml --yes --workers 8         # no question (batch jobs), 8 processes
python run.py CONFIG.toml --no-report               # compute only
python run.py --report runs/<run folder>            # redraw every figure of a run from its saved data
python run.py --options                             # distributions, forms, optimizers, all defaults
```

### Kinds of run

| Kind | For each combination of inputs | Inputs that take sets of values |
|---|---|---|
| `optimize` | search for the best τ, then evaluate it | distributions, ζ per distribution, n, k, threshold forms, objectives (VCG-PA or IVCG-PA welfare), optimizers with their parameters |
| `evaluate` | evaluate given thresholds | distributions, ζ, n, k, thresholds (polynomial, constant, tabulated, a saved `tau.json`, or every optimum of an earlier run) |

**Evaluation of each combination:**

- exact expected outcomes of the four mechanisms;
- 2000 simulated auctions by default, the same auctions for every design of a market;
- for empirical data, a bootstrap over test rows;
- the checks listed below.

**Optional checks:**

- a grid-refinement check for synthetic families;
- a bandwidth-sensitivity check for empirical data.

A minimal configuration:

```toml
kind = "optimize"
name = "harm-n"

[[distributions]]
source = "harm_rises"

[markets]
num_bidders = [4, 8, 16]
num_goods = [1]

[design]
threshold_forms = ["knots6", "nodes"]
objectives = ["vcg_pa"]

[[optimizers]]
method = "best_response"
```

**Standard test run:** [configs/standard_test.toml](configs/standard_test.toml), about 1 minute with 8 workers.

- Two distributions: `harmful_middle`, where ironing binds, and the tweet data at ζ ∈ {5, 50}.
- n ∈ {4, 8} and k ∈ {1, 2}.
- Forms `poly1` and `knots4`; both objectives; both methods; short searches on coarse grids.
- 72 combinations. Every feature runs: bootstrap, grid refinement, bandwidth sensitivity, every figure.

Use it after code changes. For a targeted change, copy it and shrink the lists to make a smoke run.

### Outputs

Each run writes one folder, `runs/<YYYY-MM-DD_HHMM>_<kind>_<name>/`. Example:

```
runs/2026-10-19_1400_optimize_standard-test/
  README.md          index: inputs, check verdicts, results table with links to every figure
  config.toml        the configuration as given
  run.toml           resolved configuration (all defaults explicit), git commit and uncommitted-change
                     flag, package versions, host (local / BU SCC + job id), seeds, data SHA-256,
                     start/end times, runtime per stage
  code.patch         uncommitted changes of tracked code files (only when the tree was dirty)
  log.txt            progress messages
  summary.csv        one row per combination: inputs, exact outcomes of the four mechanisms,
                     gains and shares, revenue changes, bootstrap bounds, runtime
  checks.csv         one row per check and combination: value, tolerance, verdict, meaning
  compare/share-of-first-best-gain.pdf          overview of every combination
  harmful-middle/                                distribution
    z1/                                          externality scale ζ
      types.npz                                  type-distribution background of the figures
      compare/tau-by-market_vcgpa_knots4_br.pdf  τ for every (n, k), other inputs fixed
      compare/tau-by-form_n8-k2_vcgpa_ga.pdf     τ for every form
      compare/tau-by-optimizer_n8-k2_vcgpa_knots4.pdf
      compare/tau-by-objective_n8-k2_knots4_br.pdf
      n8-k2/                                     market: n bidders, k goods
        vcgpa_knots4_br/                         objective_form_optimizer (evaluate runs: tau-<label>)
          tau.json   tau.csv   results.json   first-order.csv   data.npz
          1_types-tau-penalties.pdf   2_expected-outcomes.pdf   3_outcome-distributions.pdf
          4_gain-scatter.pdf   5_ironing.pdf   6_marginal-contribution.pdf
          search/tried-taus.pdf   search/convergence.pdf
  tweets/compare/welfare-vs-zeta_n8-k2_vcgpa_knots4_br.pdf   (distributions with several ζ)
runs/index.csv       one line per run: folder, kind, name, start, seconds, combinations, failures,
                     failed checks, commit, uncommitted changes, host
```

A comparison figure goes in the `compare/` folder of the deepest level above every input it varies. `runs/` is not tracked by git.

| Figure | Shows |
|---|---|
| 1 types, τ, penalties | (a) type distribution, τ(e), abstention region, excluded externalities, first-best boundary v = −e, bids pooled by IVCG-PA; (b) penalties r(e) of VCG-PA and IVCG-PA, recipient penalty −e |
| 2 expected outcomes | (a) expected welfare split as W = bidder net utility + revenue + penalties + externality part, with the share of the first-best gain; (b) changes over VCG of every component, with bootstrap intervals |
| 3 outcome distributions | rows: welfare, valuation part Σvx, externality part Σex, revenue, penalties; left: per-auction distribution of each mechanism; right: paired difference from VCG in the same auction (share of exact zeros stated); means marked, with intervals |
| 4 gain scatter | per auction: captured gain (m − VCG) against achievable gain (first best − VCG), and revenue change against welfare change |
| 5 ironing | IVCG-PA design in quantile space: ŵ and ironed ω̄, Ψ and its convex hull, their gap, scores against bids, interim allocations (only when ironing binds) |
| 6 marginal contribution | maps of M(e, v) for VCG-PA and IVCG-PA with the zero contour and τ; penalty against the Pigouvian charge −eX + L − P; M at the threshold types |
| search | tried thresholds (subset) behind the best one; convergence trace of every start and stage |
| compare | τ overlays across markets (color: n/k on a perceptually uniform scale; line style: k), forms, optimizers (with time and first-order residual) and objectives; welfare against ζ; share of the first-best gain of every combination |

**Checks** (`checks.csv`; counts in the run's README):

| Check | Value | Verdict |
|---|---|---|
| `monte_carlo_vs_exact` | largest \|z\| of a simulated mean against the exact value (welfare, revenue, four mechanisms) | pass if < 4 |
| `grid_refinement` | largest relative change of a gain over VCG on a 2× finer grid, τ fixed | pass if < 1% |
| `first_order` | max \|M(e, τ(e))\| / v̄ at interior thresholds | pass if < 10⁻⁴ for the `nodes` form; info for restricted forms and given τ, where it measures the distance from the per-node optimum |
| `search_starts_spread` | relative welfare spread of the final points of the starts | info: a large value means local optima, and the best start is kept |
| `bandwidth_sensitivity` | spread of the gain over VCG across smoothing bandwidths | pass if ≤ the width of its bootstrap interval |

**Reusing results:**

- `tau.json` rebuilds τ (`audited_auctions.experiments.compute.load_threshold_file`) and can be evaluated in any later run (`tau_file`, `from_run`).
- `results.json` holds every statistic of a combination.
- `data.npz` holds the arrays behind its figures.

### On the BU SCC cluster

```bash
#!/bin/bash -l
#$ -pe omp 16
#$ -l h_rt=12:00:00
module load python3
cd $HOME/audited-auctions-empirical
.venv/bin/python run.py configs/optimize_example.toml --yes --workers $NSLOTS
```

`run.toml` records the host, the job id (`JOB_ID`) and the slots.

## Distributions

**Synthetic families:**

- the mixtures `a`–`f`, `h`;
- `example_n`, the analytic benchmark of the core draft;
- `independent`, `harm_rises`, `harmful_middle` (ironing binds), `heavy_tail`.

Every family is a piecewise-uniform distribution on a grid: externalities on nodes, valuations uniform within cells. All quantities are exact for that distribution. Family parameters and grid accuracy are in [audited_auctions/README.md](audited_auctions/README.md#synthetic-families).

**Empirical data:** any CSV with columns `e_score`, `v_score`. In the model, e = ζ · e_score and v = `valuation_scale` · v_score.

Processing of a CSV:

- rows are split ½/½ into training and test halves;
- each half is truncated at a valuation quantile (v̄; default 0.995);
- τ and the IVCG-PA scores are designed on the kernel-smoothed training half;
- outcomes are evaluated exactly on the atomic test rows;
- the bootstrap resamples the test rows.

### The tweet data set (`tweets` = `data/samples/g_full_tweets.csv`)

The set has 11,811 tweets created between 2022-12-16 and 2025-01-18, each with at least one Community Notes note. Engagement counts are as of 2025-02-02. It was built by [create_distribution.ipynb](create_distribution.ipynb) from the Twitter/X API ([get_tweets.ipynb](get_tweets.ipynb)) and the public Community Notes data. `full_tweets.csv` in the top folder is the same file.

**Construction (from the notebook):**

1. **Note score:** each note scores +1 if classified `NOT_MISLEADING`, −1 if `MISINFORMED_OR_POTENTIALLY_MISLEADING`.
2. **Rating score:** the sum over the note's ratings (HELPFUL = 1, SOMEWHAT_HELPFUL = 0.5, NOT_HELPFUL = 0), plus 1 for the note's author.
3. **`agg_post_rating`:** the sum over a tweet's notes of note score × rating score.
4. **Filters:** tweets without notes or without impressions are dropped, as are tweets with `agg_post_rating` or `action_count` outside their 0.1%–99.9% quantiles.
5. **Age:** `months_old` = `days_old` / 30.

| Column | Meaning |
|---|---|
| `id`, `tweetId` | tweet id (equal) |
| `text`, `created_at`, `created_at_dt`, `author_id` | tweet text, creation time (string, parsed), author id |
| `retweet_count`, `reply_count`, `like_count`, `quote_count`, `bookmark_count` | engagement counts |
| `action_count` | sum of the five engagement counts |
| `impression_count` | times the tweet was shown |
| `interaction_score` | `action_count` / `impression_count` |
| `agg_post_rating` | signed, rating-weighted sum of the tweet's notes (negative: notes say misleading) |
| `agg_post_rating_per_impression`, `agg_post_rating_per_action` | `agg_post_rating` divided by impressions, by actions |
| `days_old`, `months_old` | age at 2025-02-02 in days and in 30-day months |
| `impressions_per_month`, `actions_per_month`, `agg_rating_per_month` | the counts divided by `months_old` |
| `v_score` | `actions_per_month`: the valuation score (engagement per month) |
| `e_score` | `agg_rating_per_month`: the externality score (note signal per month; 9.7% positive) |

[value_externality_definitions/](value_externality_definitions/) documents an earlier definition of the scores, which differs from this file:

- rating levels −1/0/+1;
- normalization by the mean externality per impression;
- an outlier filter on actions per impression.

[create_descriptive_stats.py](create_descriptive_stats.py) writes summary tables and figures of the data to `output/descriptive_stats/`. [csv_to_latex.py](csv_to_latex.py) turns the tables into LaTeX.

## Python interface

| Function | Module | Does |
|---|---|---|
| `synthetic_distribution(name, externality_scale, **options)` | `type_distributions` | a synthetic family |
| `prepare_distributions(source, ...)` | `pipeline` | training and evaluation distributions of a family or CSV |
| `evaluate_threshold(distribution, τ, n, k, num_simulated_draws=...)` | `threshold_evaluation` | exact outcomes, penalties, simulated auctions |
| `exact_outcomes(distribution, τ, n, k)` | `threshold_evaluation` | exact outcomes only (welfare and its parts, revenue, penalties, bidder utility) |
| `expected_welfare(distribution, τ, n, k, mechanism)` | `threshold_evaluation` | the optimization objective |
| `optimize_threshold(training, form, n, k, objective, optimizer)` | `pipeline` | search for τ |
| `run_genetic_search`, `run_best_response_ascent` | `genetic_search`, `best_response_ascent` | the two optimizers |
| `first_order_check(distribution, τ, n, k, mechanism)` | `best_response_ascent` | marginal contribution and Pigouvian charge at each threshold |
| `build_threshold_form(label, distribution)`, `threshold_from_spec(spec)` | `threshold_forms` | forms of τ |
| `run(config_file)`, `make_report(run_folder)` | `experiments.runner`, `experiments.report` | a whole run; its figures |

## Code layout

| Path | Contents |
|---|---|
| [run.py](run.py) | command-line entry point for runs |
| [configs/](configs/) | run configurations and their reference |
| [audited_auctions/](audited_auctions/) | the package: distributions, exact computation, simulation, optimizers, figures |
| [audited_auctions/experiments/](audited_auctions/experiments/) | runs: configuration, computation of combinations, run records, report |
| [tests/](tests/) | `python -m pytest tests -q` (66 tests) |
| `data/samples/`, `full_tweets.csv` | the tweet data |
| [legacy/](legacy/) | superseded scripts, notebooks and outputs of the EC 2025 / EAAMO 2026 workflow ([legacy/README.md](legacy/README.md)) |

The sweep scripts `run_sweep_genetic2.py`, `Collateralized_Auction_genetic2.py`, `evaluate_threshold.py` and `post_optimization_analysis.py` are replaced by `run.py`; they remain in the git history.
