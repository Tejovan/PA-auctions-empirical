# Run configurations

Every configuration key, its default and its meaning: [§3 of the top README](../README.md#3-configuration-files). `python run.py --options` prints the defaults.

| File | Purpose |
|---|---|
| [smoke_test.toml](smoke_test.toml) | smoke test for code changes: 4 combinations through every stage, about 10 s; results not meaningful |
| [standard_test.toml](standard_test.toml) | standard test: every feature on small grids, 96 combinations, about 2 min with 12 workers |
| [evaluate_example.toml](evaluate_example.toml) | evaluate run: the zero-welfare threshold τ⁰, the ex ante cutoff τ*, and the optima of an earlier run |
| [optimize_example.toml](optimize_example.toml) | research-size optimize run on the tweet data (several ζ, per-node τ) |
