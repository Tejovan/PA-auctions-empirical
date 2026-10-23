"""
runner.py

A run: read a configuration, print the plan and a time estimate, compute every
combination (in parallel worker processes), write the run-level files, then
draw the figures (report.py).

Run folder <output.base>/<YYYY-MM-DD_HHMM>_<kind>_<name>/:
  config.toml   the configuration file as given
  run.toml      the resolved configuration (all defaults explicit), code version,
                software, host, seeds, start and end times, runtime per stage
  code.patch    uncommitted changes of tracked files (only if the code was dirty)
  log.txt       progress messages
  summary.csv   one row per combination: inputs and main results
  checks.csv    one row per check and combination, with a verdict
  README.md     index: inputs, check verdicts, results table, links to figures
  <distribution>/z<ζ>/types.npz              type-distribution background
  <distribution>/z<ζ>/n<n>-k<k>/<design>/    files of one combination (compute.py)
  compare/ and <distribution>[/z<ζ>]/compare/   comparison figures (report.py)
<output.base>/index.csv gets one line per run.
"""

import csv
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..parallel import split_workers
from ..threshold_evaluation import expected_welfare
from .compute import (
    compute_combination,
    plot_geometry,
    continuous_evaluation,
    prepared_distributions,
    rescaled_threshold,
    type_background,
)
from .config import CODE_ROOT, expand, load_config, markets_text
from .provenance import (
    code_version,
    file_sha256,
    host_information,
    now,
    software_versions,
    to_toml,
    uncommitted_diff,
)

FIGURE_SECONDS = 0.6  # rough cost of one figure, for the time estimate


class Log:
    """Messages to the terminal and to log.txt."""

    def __init__(self):
        self.lines, self.file = [], None

    def open(self, path):
        self.file = open(path, "a")
        for line in self.lines:
            self.file.write(line + "\n")

    def __call__(self, message=""):
        print(message, flush=True)
        self.lines.append(message)
        if self.file is not None:
            self.file.write(message + "\n")
            self.file.flush()


def default_workers():
    """All cores but two (one for the system, one for the main process)."""
    return max(1, (os.cpu_count() or 3) - 2)


# ─── Tasks ──────────────────────────────────────────────────────────────────────


def build_tasks(combinations):
    """Lists of combinations computed in order by one worker.

    Optimizers with warm_start_across_zeta chain the ζ values of a distribution
    (each start uses the previous optimum); every other combination is its own task.
    """
    tasks, chains = [], {}
    for combination in combinations:
        optimizer = combination.optimizer
        if optimizer is not None and optimizer["warm_start_across_zeta"]:
            key = (combination.distribution["label"], combination.num_bidders,
                   combination.num_goods, combination.design_label)
            chains.setdefault(key, []).append(combination)
        else:
            tasks.append([combination])
    for chain in chains.values():
        tasks.append(sorted(chain, key=lambda c: c.externality_scale))
    return tasks


def run_task(task, config, run_folder):
    """Worker entry point: compute the combinations of a task in order."""
    outputs, previous = [], None
    for combination in task:
        warm_start = None
        if previous is not None:
            warm_start = rescaled_threshold(previous[1],
                                            previous[0] / combination.externality_scale)
        try:
            row, checks, threshold = compute_combination(combination, config, run_folder,
                                                         warm_start)
        except Exception as error:  # report and continue with the other combinations
            import traceback

            outputs.append((combination.path, None, None, traceback.format_exc()))
            previous = None
            continue
        outputs.append((combination.path, row, checks, None))
        previous = (combination.externality_scale, threshold)
    return outputs


# ─── Time estimate ──────────────────────────────────────────────────────────────


def _evaluations_per_search(optimizer):
    """Rough number of exact welfare evaluations of one search (genetic: an upper bound)."""
    genetic = optimizer["genetic"]
    genetic_count = genetic["num_restarts"] * (
        genetic["population_size"] * genetic["num_generations"]
        + (genetic["num_local_starts"] * genetic["local_max_evaluations"]
           if genetic["local_refinement"] == "nelder_mead" else 0))
    best_response = optimizer["best_response"]
    iterations = min(best_response["max_iterations"], 15)
    num_starts = len(best_response["starts"]) + best_response["num_random_starts"]
    best_response_count = (30 * best_response["short_genetic_generations"]
                           if "short_genetic" in best_response["starts"] else 0) + num_starts * (
        iterations * 4 * (1 + best_response["polish_rounds"]) + 40 * best_response["polish_rounds"])
    relaxation_count = iterations * 4 * (1 + best_response["polish_rounds"]) + 40 * best_response[
        "polish_rounds"]  # one start (the linear program is timed as RELAXATION_SECONDS)
    return {"genetic": genetic_count, "best_response": best_response_count,
            "genetic+best_response": genetic_count + best_response_count,
            "relaxation": relaxation_count}[optimizer["method"]]


RELAXATION_SECONDS = 5.0  # one symmetric relaxation (linear program), default size


def estimate_seconds(config, combinations, workers):
    """(estimate in seconds, per-distribution calibration) from timing a few evaluations."""
    calibration = {}
    for combination in combinations:
        key = (combination.distribution["label"], combination.externality_scale,
               combination.num_bidders, combination.num_goods)
        if key in calibration:
            continue
        prepared = prepared_distributions(combination.distribution, combination.externality_scale,
                                          config["seeds"]["split"], continuous_evaluation(config))
        threshold = lambda e: np.maximum(-np.asarray(e, dtype=float), 0.0)  # noqa: E731
        timings = {}
        for mechanism in ("vcg_pa", "ivcg_pa"):
            begin = time.time()
            for _ in range(3):
                expected_welfare(prepared.training, threshold, combination.num_bidders,
                                 combination.num_goods, mechanism)
            timings[mechanism] = (time.time() - begin) / 3
        begin = time.time()
        expected_welfare(prepared.evaluation, threshold, combination.num_bidders,
                         combination.num_goods, "ivcg_pa")
        timings["evaluation"] = time.time() - begin
        calibration[key] = timings
    evaluation = config["evaluation"]
    total = 0.0
    for combination in combinations:
        timings = calibration[(combination.distribution["label"], combination.externality_scale,
                               combination.num_bidders, combination.num_goods)]
        seconds = 0.0
        if combination.optimizer is not None:
            per_evaluation = timings[combination.objective]
            searches = 1 + (len(config["checks"]["bandwidth_sensitivity"])
                            if combination.distribution["kind"] == "empirical" else 0)
            seconds += searches * per_evaluation * _evaluations_per_search(combination.optimizer)
            if combination.optimizer["method"] == "relaxation":
                seconds += RELAXATION_SECONDS
        seconds += 30 * timings["evaluation"]  # outcomes, first-order checks, maps
        seconds += evaluation["num_simulated_auctions"] * combination.num_bidders * 2e-6 * 4
        if combination.distribution["kind"] == "empirical":
            seconds += evaluation["num_bootstrap"] * 8 * timings["evaluation"]
        elif config["checks"]["grid_refinement"]:
            seconds += 4 * config["checks"]["refinement_factor"] ** 2 * timings["vcg_pa"] * 4
        total += seconds
    figures = 8 * len(combinations) * FIGURE_SECONDS
    return total / workers + figures / workers + 2.0, calibration


# ─── Plan ───────────────────────────────────────────────────────────────────────


def describe_plan(config, combinations, skipped):
    lines = [f"Run kind: {config['kind']}   name: {config['name']}",
             f"Combinations: {len(combinations)}"]
    for distribution in config["distributions"]:
        lines.append(f"  distribution {distribution['label']} ({distribution['kind']}, "
                     f"source {distribution['source']}): ζ ∈ {distribution['externality_scales']}, "
                     f"grid {distribution['num_externality_nodes']} × "
                     f"{distribution['num_valuation_cells']}")
    lines.append(f"  {markets_text(config)}")
    if config["kind"] == "optimize":
        lines.append(f"  objectives {config['design']['objectives']}, optimizers "
                     f"{[o['label'] + ' (' + o['method'] + ': ' + ', '.join(o['threshold_forms']) + ')' for o in config['optimizers']]}")
    if config.get("thresholds"):
        lines.append(f"  {'given ' if config['kind'] == 'optimize' else ''}thresholds "
                     f"{[t['label'] or t['value'] for t in config['thresholds']]}")
    for description, reason in skipped:
        lines.append(f"  skipped: {description}: {reason}")
    return lines


def run_folder_name(config, start):
    return f"{start:%Y-%m-%d_%H%M}_{config['kind']}_{config['name']}"


# ─── Main entry ─────────────────────────────────────────────────────────────────


def run(config_path, workers=None, dry_run=False, assume_yes=False, report=None):
    """Run a configuration file; returns the run folder (None for a dry run)."""
    log = Log()
    config = load_config(config_path)
    workers = workers or config["output"]["workers"] or default_workers()
    combinations, skipped = expand(config)
    if not combinations:
        raise SystemExit("no combinations to run")
    for line in describe_plan(config, combinations, skipped):
        log(line)
    estimate, _ = estimate_seconds(config, combinations, workers)
    log(f"Estimated time: about {_format_seconds(estimate)} with {workers} workers "
        f"(genetic searches: upper bound)")
    if dry_run:
        return None
    if not assume_yes and sys.stdin.isatty():
        answer = input("Proceed? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            log("Cancelled.")
            return None

    start = now()
    base = Path(config["output"]["base"])
    base = base if base.is_absolute() else CODE_ROOT / base
    folder = base / run_folder_name(config, start)
    suffix = 1
    while folder.exists():
        suffix += 1
        folder = base / f"{run_folder_name(config, start)}-{suffix}"
    folder.mkdir(parents=True)
    log.open(folder / "log.txt")
    log(f"Run folder: {folder}")
    shutil.copy(config_path, folder / "config.toml")

    version = code_version()
    if version["dirty"]:
        (folder / "code.patch").write_text(uncommitted_diff())
    data_hashes = {}
    for distribution in config["distributions"]:
        if distribution["kind"] == "empirical":
            path = Path(distribution["path"])
            path = path if path.is_absolute() or path.exists() else CODE_ROOT / path
            data_hashes[distribution["label"]] = {"path": str(distribution["path"]),
                                                  "sha256": file_sha256(path)}
    record = {
        "run": {"folder": folder.name, "kind": config["kind"], "name": config["name"],
                "description": config["description"], "config_file": str(config_path),
                "command": " ".join(sys.argv), "started": start.isoformat(timespec="seconds"),
                "num_combinations": len(combinations), "estimated_seconds": round(estimate, 1)},
        "code": version, "software": software_versions(), "host": host_information(workers),
        "data": data_hashes, "config": config,
    }
    (folder / "run.toml").write_text(to_toml(record) + "\n")

    # Type backgrounds, one per distribution and ζ ---------------------------------
    for distribution, zeta in {(c.distribution["label"], c.externality_scale): (
            c.distribution, c.externality_scale) for c in combinations}.values():
        prepared = prepared_distributions(distribution, zeta, config["seeds"]["split"],
                                          continuous_evaluation(config))
        background = type_background(prepared, plot_geometry(prepared))
        target = folder / distribution["label"] / f"z{zeta:g}"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "types.npz", **background)

    # Combinations ----------------------------------------------------------------
    compute_begin = time.time()
    tasks = build_tasks(combinations)
    rows, checks, failures, done = [], [], [], 0
    log(f"Computing {len(combinations)} combinations in {len(tasks)} tasks "
        f"with {workers} workers")

    def collect(outputs):
        nonlocal done
        for path, row, task_checks, error in outputs:
            done += 1
            if error is not None:
                failures.append((path, error))
                log(f"[{done}/{len(combinations)}] FAILED {path}\n{error}")
                continue
            rows.append(row)
            checks.extend(task_checks)
            objective = row.get("objective") or "vcg_pa"
            log(f"[{done}/{len(combinations)}] {path}  share of first-best gain "
                f"{objective}: {_format_share(row.get(f'share_{objective}'))}  "
                f"({row['seconds']:.1f} s)")

    # Workers beyond one per task go to the tasks' own parallel stages (search starts and
    # restarts, Nelder–Mead starts, bootstrap): processes, inner = split_workers(...).
    processes, inner = split_workers(workers, len(tasks))
    task_config = {**config, "inner_workers": inner}
    if processes == 1:
        for task in tasks:
            collect(run_task(task, task_config, folder))
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            futures = [pool.submit(run_task, task, task_config, folder) for task in tasks]
            for future in as_completed(futures):
                collect(future.result())
    compute_seconds = time.time() - compute_begin

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values("path")
    summary.to_csv(folder / "summary.csv", index=False)
    check_table = pd.DataFrame(checks, columns=["path", "check", "value", "tolerance",
                                                "verdict", "note"])
    check_table.sort_values(["check", "path"]).to_csv(folder / "checks.csv", index=False)
    if failures:
        with open(folder / "failures.txt", "w") as file:
            for path, error in failures:
                file.write(f"{path}\n{error}\n\n")
    log(f"Computation: {_format_seconds(compute_seconds)}; {len(failures)} failed")

    report_seconds, files = 0.0, None
    if (config["output"]["report"] if report is None else report) and len(summary):
        from .report import make_report

        report_begin = time.time()
        files = make_report(folder, workers=workers, log=log)
        report_seconds = time.time() - report_begin
        log(f"Figures: {_format_seconds(report_seconds)}")

    end = now()
    record["run"].update(finished=end.isoformat(timespec="seconds"),
                         seconds={"compute": round(compute_seconds, 1),
                                  "report": round(report_seconds, 1),
                                  "total": round((end - start).total_seconds(), 1)},
                         num_failed=len(failures))
    (folder / "run.toml").write_text(to_toml(record) + "\n")
    if files is not None:
        from .report import write_readme

        write_readme(folder, files)  # again, with the final runtime
    _append_index(base, record, check_table)
    log(f"Done in {_format_seconds((end - start).total_seconds())}: {folder}")
    return folder


def _append_index(base, record, check_table):
    path = base / "index.csv"
    new = not path.exists()
    verdicts = check_table.verdict.value_counts().to_dict() if len(check_table) else {}
    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(["folder", "kind", "name", "started", "seconds", "combinations",
                             "failed", "checks_failed", "commit", "dirty", "where"])
        writer.writerow([record["run"]["folder"], record["run"]["kind"], record["run"]["name"],
                         record["run"]["started"], record["run"]["seconds"]["total"],
                         record["run"]["num_combinations"], record["run"]["num_failed"],
                         verdicts.get("fail", 0), (record["code"]["commit"] or "")[:10],
                         record["code"]["dirty"], record["host"]["where"]])


def _format_seconds(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _format_share(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.1f}%"
