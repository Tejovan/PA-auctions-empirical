# Run configurations

Every configuration key, its default and its meaning: [§3 of the top README](../README.md#3-configuration-files). `python run.py --options` prints the defaults.

| File | Purpose |
|---|---|
| [smoke_test.toml](smoke_test.toml) | smoke test for code changes: 6 combinations (three optimizers, two distributions) through every stage, about 15 s; results not meaningful |
| [standard_test.toml](standard_test.toml) | standard test: every feature on small grids, 120 combinations, about 3 min with 12 workers |
| [evaluate_example.toml](evaluate_example.toml) | evaluate run: the zero-welfare threshold τ⁰, the ex ante cutoff τ*, and the optima of an earlier run |
| [nodes_run.toml](nodes_run.toml) | nodes form, threshold-response ascent and symmetric relaxation: harmful_middle and tweets (ζ 1, 10, 100), n ∈ {4, 8}, k ∈ {1, 4} (k = n: no competition), both objectives; default grids, 56 combinations, about 5 min with 12 workers |
| [paper_run.toml](paper_run.toml) | WEB27 paper and supplement in one run (replaces synthetic_run.toml and shapes_run.toml): independent, harmful_middle, heavy_tail, harm_rises; 20 markets (k/n from 1/32 to 1: the 14 with n/k ≥ 2, (4,3), (8,6), (16,12), (8,7), (16,14), (4,4)); optimized τ (threshold-response ascent), best clipped affine τ and best cutoff (grid search; 121 points per parameter for VCG-PA, 81 for IVCG-PA), all for both objectives; given τ⁰, τ*, relaxation τ; 716 combinations |
| [synthetic_run.toml](synthetic_run.toml) | WEB27 synthetic results: harmful_middle, the 14 markets with k < n of n ∈ {2, 4, 8, 16, 32} × k ∈ {1, 2, 4, 8} plus n = k = 4 as a no-competition check (`pairs`; fixed n/k on diagonals), both objectives; nodes by threshold-response ascent (no genetic start), common welfare cutoff by genetic search; given τ⁰, ex ante cutoff τ* and relaxation τ evaluated in the same run; 104 combinations |
| [synthetic_benchmarks.toml](synthetic_benchmarks.toml) | the three given τ of synthetic_run.toml as a separate evaluate run (evaluation only); 44 combinations |
| [shapes_run.toml](shapes_run.toml) | shapes of VCG-PA-optimal τ and penalties: independent, harmful_middle, heavy_tail, two-modes (mixture d); k = 1 with n ∈ {2, 4, 8, 16}, plus (8, 2), (16, 4); nodes by threshold-response ascent, cutoff and linear τ by genetic search, τ⁰ given; 96 combinations |
| [optimize_example.toml](optimize_example.toml) | research-size optimize run on the tweet data (several ζ, per-node τ) |
