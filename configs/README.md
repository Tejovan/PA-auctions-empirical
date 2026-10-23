# Run configurations

Every configuration key, its default and its meaning: [§3 of the top README](../README.md#3-configuration-files). `python run.py --options` prints the defaults.

| File | Purpose |
|---|---|
| [smoke_test.toml](smoke_test.toml) | smoke test for code changes: 6 combinations (three optimizers, two distributions) through every stage, about 15 s; results not meaningful |
| [standard_test.toml](standard_test.toml) | standard test: every feature on small grids, 120 combinations, about 3 min with 12 workers |
| [evaluate_example.toml](evaluate_example.toml) | evaluate run: the zero-welfare threshold τ⁰, the ex ante cutoff τ*, and the optima of an earlier run |
| [nodes_run.toml](nodes_run.toml) | nodes form, best-response ascent and symmetric relaxation: harmful_middle and tweets (ζ 1, 10, 100), n ∈ {4, 8}, k ∈ {1, 4} (k = n: no competition), both objectives; default grids, 56 combinations, about 5 min with 12 workers |
| [synthetic_run.toml](synthetic_run.toml) | WEB27 synthetic results: harmful_middle, the 14 markets with k < n of n ∈ {2, 4, 8, 16, 32} × k ∈ {1, 2, 4, 8} (`pairs`; fixed n/k on diagonals), both objectives; nodes by best-response ascent, common welfare cutoff by genetic search; given τ⁰, ex ante cutoff τ* and relaxation τ evaluated in the same run; 98 combinations |
| [optimize_example.toml](optimize_example.toml) | research-size optimize run on the tweet data (several ζ, per-node τ) |
