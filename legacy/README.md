# Legacy material

Files from the original EC 2025 / EAAMO 2026 workflow that the WEB27 code (`audited_auctions/`, `run_sweep_genetic2.py`, `post_optimization_analysis.py`) no longer uses. Kept for reference; paths inside these files assume they sit at the repository root, so they do not run from here without edits.

| Path | What it is | Why it is legacy |
|------|------------|------------------|
| `output/sweep_genetic2_*` | May 2026 sweeps (Community Notes data, synthetic samples a–f, h) | Simulated GA objective, biased penalty estimator, no IVCG-PA, records without `evaluation`/`type_sample`; `post_optimization_analysis.py` no longer reads them |
| `data/samples/*.csv` (a–f, h) | Synthetic samples of `alternate_distributions.ipynb` | Contain negative valuations; replaced by the parametric families in `audited_auctions/type_distributions.py` (shifted by +1 and truncated at 0) |
| `data/figures/` | Plots of those samples | As above |
| `alternate_distributions.ipynb` | Generates the samples a–f, h | As above (its parameters are reproduced in `type_distributions.py`) |
| `alternate_optimization_algorithms/` | Grid search, SGD, first-generation GA, SGE cluster scripts | Superseded before EC 2025 |
| `run_local.sh`, `run_sweep.sh` | Shell sweep runners for the single-cell optimizers | Superseded by `run_sweep_genetic2.py` |
| `plot_sweep.py`, `Create_Plots.ipynb` | Figures from pre-WEB27 result pickles | Read the old result format (`tau`, `advertisers`, `w_coll_tot`, …), no longer produced |
| `main.py` | Placeholder | Unused |
| `supplemental/` | EC 2025 supplemental document (LaTeX) | Describes the EC 2025 analysis; still the reference for data collection |

---

## Alternate Optimization Algorithms (from the original README)

The scripts in [`alternate_optimization_algorithms/`](alternate_optimization_algorithms/) are earlier iterations of the optimizer included as alternate possibilities for readers interested in comparing approaches. They were **not used in the final EC 2025 analysis** and produced significantly worse results than the final genetic algorithm (`Collateralized_Auction_genetic2.py`). They can be exercised via `run_local.sh` and `run_sweep.sh` using the `--genetic`, `--sgd`, `--sgd-bb`, and `--grid` flags.

### Grid Search

**[alternate_optimization_algorithms/Collateralized_Auction_grid_search.py](alternate_optimization_algorithms/Collateralized_Auction_grid_search.py)**
Exhaustive vectorized grid search over polynomial coefficient space. Computes welfare for all coefficient combinations using matrix multiplication (`B @ E^T`); relies on BLAS threading rather than joblib. Useful for low-degree polynomials where the search space is small enough to enumerate.

### SGD-Based Optimizers

**[alternate_optimization_algorithms/Collateralized_Auction_sgd.py](alternate_optimization_algorithms/Collateralized_Auction_sgd.py)**
Gradient-based optimizer using stochastic gradient descent on τ coefficients.

**[alternate_optimization_algorithms/Collateralized_Auction_sgd_bb.py](alternate_optimization_algorithms/Collateralized_Auction_sgd_bb.py)**
SGD variant with Barzilai–Borwein adaptive step size.

### First-Generation Genetic Algorithm

**[alternate_optimization_algorithms/Collateralized_Auction_genetic.py](alternate_optimization_algorithms/Collateralized_Auction_genetic.py)**
Exploratory, percent-cell-structured script for running the GA interactively. Uses log-space coefficient search (`ln_coeffs_to_coeffs`). Superseded by `Collateralized_Auction_genetic2.py`.

**[alternate_optimization_algorithms/Collateralized_Auction_genetic_script.py](alternate_optimization_algorithms/Collateralized_Auction_genetic_script.py)**
CLI version of the first-generation GA designed for HPC cluster submission. Uses joint `[-1, 1]` normalization of `v` and `e` before running the GA, then converts coefficients back to original space. Superseded by `run_sweep_genetic2.py` + `Collateralized_Auction_genetic2.py`.

### HPC Cluster Scripts (first-generation)

**[alternate_optimization_algorithms/cluster_run.sh](alternate_optimization_algorithms/cluster_run.sh)**
SGE array job that submits one distribution × one polynomial degree, running 7 parallel tasks (one per externality cost value). Used with `Collateralized_Auction_genetic_script.py`.

**[alternate_optimization_algorithms/cluster_run_all.sh](alternate_optimization_algorithms/cluster_run_all.sh)**
Submits all combinations of distributions × polynomial degrees by calling `cluster_run.sh` repeatedly.

**[run_sweep.sh](run_sweep.sh)**
Earlier local sweep runner, predating `run_sweep_genetic2.py`.

### Older Analysis

**[Create_Plots.ipynb](Create_Plots.ipynb)**
Jupyter notebook that loaded first-generation GA pickle files and generated figures. Superseded by `post_optimization_analysis.py`.

**[plot_sweep.py](plot_sweep.py)**
Earlier sweep plotting script.

**[alternate_distributions.ipynb](alternate_distributions.ipynb)**
Generates synthetic 2D (e, v) distributions for controlled experiments: single-modal normal, two-modal side-by-side, diagonal variants, and four-modal. Saved to `data/samples/`.
