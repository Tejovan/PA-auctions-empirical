#!/usr/bin/env python3
"""
run.py — run experiments of the audited_auctions package from configuration files.

    python run.py CONFIG.toml                 run (asks to proceed after the time estimate)
    python run.py CONFIG.toml --dry-run       print the plan and the time estimate only
    python run.py CONFIG.toml --yes           run without asking (batch jobs)
    python run.py CONFIG.toml --workers 4     number of parallel worker processes
    python run.py CONFIG.toml --no-report     compute only; draw figures later
    python run.py --report RUN_FOLDER         (re)draw all figures and README of a run
    python run.py --report RUN_FOLDER --paper [NAMES]
                                              figures for a paper: one PDF per panel, caption
                                              draft and LaTeX snippet in paper/ folders
    python run.py --options                   list distributions, forms, optimizers, ...

Configurations are TOML files; configs/README.md documents every key, and
configs/*.toml are examples (configs/standard_test.toml is the standard test run).
Outputs go to runs/<date>_<time>_<kind>_<name>/ (see audited_auctions/README.md).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def list_options():
    from audited_auctions.experiments.config import (
        BEST_RESPONSE_DEFAULTS,
        DATA_SETS,
        EMPIRICAL_DEFAULTS,
        GENETIC_DEFAULTS,
        METHODS,
        OBJECTIVES,
        SECTION_DEFAULTS,
        SYNTHETIC_DEFAULTS,
        THRESHOLD_SOURCES,
    )
    from audited_auctions.type_distributions import SYNTHETIC_FAMILIES

    print("Synthetic families:", ", ".join(sorted(SYNTHETIC_FAMILIES)))
    print("Data sets:", ", ".join(f"{name} = {path}" for name, path in DATA_SETS.items()),
          "(or any CSV with columns e_score, v_score)")
    print("Threshold forms: poly<d>, knots<K>, bins<K>, nodes")
    print("Objectives:", ", ".join(OBJECTIVES))
    print("Optimizer methods:", ", ".join(METHODS))
    print("Given thresholds (evaluate runs):", ", ".join(THRESHOLD_SOURCES))
    for title, defaults in (("synthetic distribution keys", SYNTHETIC_DEFAULTS),
                            ("empirical distribution keys", EMPIRICAL_DEFAULTS),
                            ("genetic keys", GENETIC_DEFAULTS),
                            ("best-response keys", BEST_RESPONSE_DEFAULTS),
                            *((f"[{s}]", d) for s, d in SECTION_DEFAULTS.items())):
        print(f"\n{title} (defaults):")
        for key, value in defaults.items():
            print(f"  {key} = {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="configuration file (TOML)")
    parser.add_argument("--dry-run", action="store_true", help="plan and time estimate only")
    parser.add_argument("--yes", action="store_true", help="do not ask before running")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: output.workers, else cores − 2)")
    parser.add_argument("--no-report", action="store_true", help="skip the figures")
    parser.add_argument("--report", metavar="RUN_FOLDER", help="redraw figures of a run")
    parser.add_argument("--paper", nargs="?", const="", metavar="NAMES",
                        help="with --report: export panels for a paper (comma-separated "
                             "figure names, default all)")
    parser.add_argument("--options", action="store_true", help="list the available options")
    args = parser.parse_args()

    if args.options:
        list_options()
        return
    if args.paper is not None and not args.report:
        parser.error("--paper needs --report RUN_FOLDER")
    if args.report:
        from audited_auctions.experiments.report import make_paper_figures, make_report

        if args.paper is not None:
            names = [n.strip() for n in args.paper.split(",") if n.strip()] or None
            make_paper_figures(Path(args.report), names, workers=args.workers)
        else:
            make_report(Path(args.report), workers=args.workers)
        return
    if not args.config:
        parser.error("give a configuration file, --report RUN_FOLDER or --options")
    from audited_auctions.experiments.runner import run

    run(args.config, workers=args.workers, dry_run=args.dry_run, assume_yes=args.yes,
        report=False if args.no_report else None)


if __name__ == "__main__":
    main()
