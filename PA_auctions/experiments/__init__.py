"""
experiments: runs of the PA_auctions package from configuration files.

  config.py      read, complete and validate a TOML configuration; expand it
                 into combinations
  compute.py     compute one combination and save its data
  runner.py      a whole run: plan, time estimate, parallel computation,
                 run record (run.toml), summary and checks tables
  report.py      figures and README of a run, drawn from the saved data
  provenance.py  code version, software, host, data hashes; TOML writer

Entry point: run.py in the code folder.
"""
