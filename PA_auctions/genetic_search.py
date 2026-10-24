"""
genetic_search.py

Search for a threshold function τ(e) in a parametric form (threshold_forms.py:
a polynomial τ(e) = β_0 + β_1 e + ... + β_d e^d by default, or knots / bins)
that maximizes the exact expected welfare of a participant-audit mechanism
(VCG-PA by default, or IVCG-PA) for a type distribution.

Two stages (default):

1. Genetic algorithm (global exploration). pygad with tournament selection,
   single-point crossover and Gaussian mutation scaled to each coefficient's
   search range, as in the original genetic search of this repository; warm
   start optional. It stops after num_generations, or earlier once the best
   welfare has risen by at most stall_tolerance × welfare_scale (the first-best
   gain FB − W_VCG when given, else |best welfare|) over the last
   stall_generations generations. (A rule of exact equality would let gains of
   1e-12 restart the count, so the stage would run out its generations.)
2. Nelder–Mead refinement (local convergence). Started from the
   num_local_starts best distinct candidates of the final population, in
   coordinates scaled to the search ranges (each coefficient's range maps to
   [0, 1], initial simplex steps 5% of the range), with tight tolerances. The
   result replaces the genetic optimum only if its welfare is higher.

A genetic algorithm finds the right region quickly but converges slowly inside
it; Nelder–Mead converges quickly from a good start but is local. With 2–4
coefficients and an exact, deterministic objective, the combination is cheap
and precise. Neither stage guarantees the global optimum of a non-convex
objective; restarts (pipeline num_restarts) and several local starts reduce the
risk. local_refinement="none" runs the genetic algorithm alone, for all
num_generations.

Time limit. With a deadline (a time.time() value), the genetic stage stops after
the generation in progress and Nelder–Mead after its current iteration; the
best point found so far is kept, and stop_reason records "time_limit".

The objective is exact (threshold_evaluation.expected_welfare), computed on the
distribution itself, so it has no Monte Carlo noise.
"""

import time

from dataclasses import dataclass, field

import numpy as np
import pygad
from scipy.optimize import minimize

from .parallel import parallel_map
from .threshold_evaluation import expected_welfare


def compute_gene_space(type_distribution, polynomial_degree, range_multiplier=3.0):
    """Search range of each coefficient, from moments of the distribution.

    β_0 ∈ [v_min − margin, v_max + margin], margin = half a valuation standard
    deviation, so the threshold can start below or above every valuation.
    β_m (m >= 1) ∈ ±range_multiplier · sd(v) / E|e|^m, so that at a typical
    externality the term β_m e^m can move τ across the spread of valuations.
    """
    probabilities = type_distribution.probabilities
    valuations = type_distribution.valuation_midpoints
    mean_valuation = np.sum(probabilities * valuations)
    valuation_sd = float(np.sqrt(np.sum(probabilities * (valuations - mean_valuation) ** 2))) or 1.0
    margin = 0.5 * valuation_sd
    gene_space = [{"low": float(type_distribution.valuation_lower.min()) - margin,
                   "high": type_distribution.maximum_valuation + margin}]
    for degree in range(1, polynomial_degree + 1):
        mean_power = max(float(np.sum(probabilities * np.abs(type_distribution.externalities)
                                      ** degree)), 1e-12)
        half_width = range_multiplier * valuation_sd / mean_power
        gene_space.append({"low": -half_width, "high": half_width})
    return gene_space


def make_gaussian_mutation(gene_space, mutation_probability, mutation_sigma_fraction, rng):
    """Mutation that perturbs each selected gene by N(0, (σ_frac · range)²), clipped to range."""
    lows = np.array([gene["low"] for gene in gene_space])
    highs = np.array([gene["high"] for gene in gene_space])
    sigmas = mutation_sigma_fraction * (highs - lows)

    def mutation(offspring, ga_instance):  # noqa: ARG001 (pygad signature)
        for row in range(offspring.shape[0]):
            mask = rng.random(offspring.shape[1]) < mutation_probability
            if not mask.any():  # always mutate at least one gene
                mask[rng.integers(offspring.shape[1])] = True
            noise = rng.standard_normal(offspring.shape[1]) * sigmas
            offspring[row] = np.clip(offspring[row] + mask * noise, lows, highs)
        return offspring

    return mutation


LOCAL_REFINEMENTS = ("nelder_mead", "none")

# Defaults lean toward reliable convergence over speed.
DEFAULT_STALL_GENERATIONS = 60
DEFAULT_STALL_TOLERANCE = 1e-4      # gain over stall_generations generations, in welfare_scale
DEFAULT_NUM_LOCAL_STARTS = 5
DEFAULT_LOCAL_MAX_EVALUATIONS = 2000
LOCAL_COORDINATE_TOLERANCE = 1e-7   # in range-scaled coordinates
LOCAL_RELATIVE_WELFARE_TOLERANCE = 1e-10
INITIAL_SIMPLEX_STEP = 0.05         # fraction of each coefficient's range


@dataclass
class SearchResult:
    """Outcome of run_genetic_search.

    coefficients, welfare : best τ coefficients and their exact welfare
    trajectory            : every coefficient vector in each generation's population
    summary               : genetic optimum, generations run, stop_reason ("stalled",
                            "max_generations", "time_limit"), history (best welfare
                            after each generation), convergence trace (best welfare
                            against evaluations and time), number of distinct objective
                            evaluations, and per local start the starting and final
                            welfare, evaluations, stop_reason ("converged",
                            "max_evaluations", "time_limit") and history (welfare
                            after each iteration)
    """

    coefficients: list
    welfare: float
    trajectory: list
    summary: dict = field(default_factory=dict)


def _past(deadline):
    return deadline is not None and time.time() >= deadline


def refine_with_nelder_mead(welfare_of, starts, lows, highs, max_evaluations, deadline=None,
                            simplex_step=INITIAL_SIMPLEX_STEP):
    """Nelder–Mead from each start, in coordinates where each coefficient's range is [0, 1]
    (initial simplex steps simplex_step in these coordinates).

    Returns a list of dicts (start, coefficients, welfare, start_welfare,
    evaluations, converged, stop_reason, history), one per start; starts not
    begun before the deadline are left out.
    """
    span = highs - lows

    def to_coefficients(scaled):
        return lows + scaled * span

    results = []
    for start in starts:
        if _past(deadline):
            break
        start = np.asarray(start, dtype=float)
        start_welfare = welfare_of(start)
        scaled_start = (start - lows) / span
        simplex = np.vstack([scaled_start,
                             scaled_start + simplex_step * np.eye(len(start))])
        history = [{"iteration": 0, "welfare": float(start_welfare), "time": time.time()}]

        def callback(intermediate_result):
            history.append({"iteration": len(history), "welfare": float(-intermediate_result.fun),
                            "time": time.time()})
            if _past(deadline):
                raise StopIteration

        outcome = minimize(
            lambda scaled: -welfare_of(to_coefficients(scaled)), scaled_start,
            method="Nelder-Mead", callback=callback,
            options={"initial_simplex": simplex, "xatol": LOCAL_COORDINATE_TOLERANCE,
                     "fatol": LOCAL_RELATIVE_WELFARE_TOLERANCE * max(abs(start_welfare), 1.0),
                     "maxfev": max_evaluations, "adaptive": True})
        stop_reason = ("converged" if outcome.success
                       else "time_limit" if _past(deadline) else "max_evaluations")
        results.append({"start": start.tolist(),
                        "coefficients": to_coefficients(outcome.x).tolist(),
                        "welfare": float(-outcome.fun), "start_welfare": float(start_welfare),
                        "evaluations": int(outcome.nfev), "converged": bool(outcome.success),
                        "stop_reason": stop_reason, "history": history})
    return results


class ExactWelfare:
    """Exact expected welfare of a form's parameters (picklable objective)."""

    def __init__(self, type_distribution, form, num_bidders, num_goods, mechanism):
        self.arguments = (type_distribution, form, num_bidders, num_goods, mechanism)

    def __call__(self, coefficients):
        d, form, n, k, mechanism = self.arguments
        return expected_welfare(d, form.threshold(coefficients), n, k, mechanism)


def _nelder_mead_job(job):
    """One Nelder–Mead start in a worker process (None if the deadline passed first)."""
    results = refine_with_nelder_mead(ExactWelfare(*job["objective"]), [job["start"]],
                                      job["lows"], job["highs"], job["max_evaluations"],
                                      job["deadline"])
    return results[0] if results else None


def _best_distinct(population, fitness, count):
    """The `count` fittest rows of the population with distinct coefficients."""
    chosen, seen = [], set()
    for index in np.argsort(-np.asarray(fitness)):
        key = tuple(np.round(population[index], 9))
        if key not in seen:
            seen.add(key)
            chosen.append(np.array(population[index], dtype=float))
        if len(chosen) == count:
            break
    return chosen


def run_genetic_search(type_distribution, polynomial_degree, num_bidders, num_goods,
                       objective_mechanism="vcg_pa", num_generations=500, sol_per_pop=50,
                       num_parents_mating=15, mutation_probability=0.25,
                       mutation_sigma_fraction=0.15, range_multiplier=3.0, seed=0,
                       initial_solution=None, local_refinement="nelder_mead",
                       stall_generations=DEFAULT_STALL_GENERATIONS,
                       stall_tolerance=DEFAULT_STALL_TOLERANCE,
                       num_local_starts=DEFAULT_NUM_LOCAL_STARTS,
                       local_max_evaluations=DEFAULT_LOCAL_MAX_EVALUATIONS, verbose=True,
                       threshold_form=None, deadline=None, welfare_scale=None, workers=1):
    """Maximize exact expected welfare of objective_mechanism over τ in a form.

    threshold_form        : a ThresholdForm or label ("poly2", "knots6", "bins10");
                            default: the polynomial of degree polynomial_degree

    local_refinement      : "nelder_mead" (default) or "none" (genetic algorithm only)
    stall_generations     : with refinement, stop the genetic stage once the best welfare
                            has risen by at most stall_tolerance × welfare_scale over
                            this many generations (None: never early)
    welfare_scale         : welfare unit of the stall rule (pipeline: the first-best gain
                            FB − W_VCG); None: |best welfare|
    workers               : processes for the Nelder–Mead starts (run simultaneously)
    deadline              : time.time() value after which the search stops gracefully
    num_local_starts      : Nelder–Mead starts (best distinct final candidates)
    local_max_evaluations : objective evaluations allowed per Nelder–Mead start

    Returns a SearchResult.
    """
    if local_refinement not in LOCAL_REFINEMENTS:
        raise ValueError(f"local_refinement must be one of {LOCAL_REFINEMENTS}")
    from .threshold_forms import GENETIC_FORMS, build_threshold_form

    if threshold_form is None:
        threshold_form = f"poly{polynomial_degree}"
    form = (build_threshold_form(threshold_form, type_distribution)
            if isinstance(threshold_form, (str, int)) else threshold_form)
    if form.kind not in GENETIC_FORMS:
        raise ValueError(f"the genetic search needs a form in {GENETIC_FORMS}")
    num_genes = form.num_parameters
    rng = np.random.default_rng(seed)
    lows, highs = form.parameter_bounds(type_distribution, range_multiplier)
    gene_space = [{"low": float(low), "high": float(high)} for low, high in zip(lows, highs)]

    initial_population = rng.uniform(lows, highs, size=(sol_per_pop, num_genes))
    if initial_solution is not None:
        seed_row = np.zeros(num_genes)
        shared = min(len(initial_solution), num_genes)
        seed_row[:shared] = initial_solution[:shared]
        initial_population[0] = np.clip(seed_row, lows, highs)

    cache = {}
    # Convergence trace: best welfare so far after each new objective evaluation
    # that improves it (stage, number of evaluations, wall-clock time stamp).
    trace, stage = [], {"name": "genetic", "best": -np.inf}

    def welfare_of(coefficients):
        key = tuple(np.round(coefficients, 12))
        if key not in cache:
            cache[key] = expected_welfare(type_distribution, form.threshold(coefficients),
                                          num_bidders, num_goods, objective_mechanism)
            if cache[key] > stage["best"]:
                stage["best"] = cache[key]
                trace.append({"stage": stage["name"], "evaluations": len(cache),
                              "time": time.time(), "welfare": float(cache[key])})
        return cache[key]

    def fitness_batch(ga_instance, solutions, solution_index):  # noqa: ARG001
        return np.array([welfare_of(solution) for solution in np.atleast_2d(solutions)])

    trajectory, history = [], []
    stall = local_refinement != "none" and stall_generations
    stop = {"reason": "max_generations"}

    def on_generation(ga_instance):
        trajectory.extend(solution.tolist() for solution in ga_instance.population)
        history.append({"iteration": ga_instance.generations_completed,
                        "welfare": float(np.max(ga_instance.last_generation_fitness)),
                        "time": time.time()})
        if len(history) > 1:
            history[-1]["welfare"] = max(history[-1]["welfare"], history[-2]["welfare"])
        if verbose and (ga_instance.generations_completed % 25 == 0
                        or ga_instance.generations_completed == num_generations):
            best, fitness, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness)
            print(f"  generation {ga_instance.generations_completed:4d}  welfare={fitness:.6g}  "
                  f"β={[f'{c:.4g}' for c in best]}")
        if _past(deadline):
            stop["reason"] = "time_limit"
            return "stop"
        if stall and len(history) > stall_generations:
            now, before = history[-1]["welfare"], history[-1 - stall_generations]["welfare"]
            if now - before <= stall_tolerance * (welfare_scale or max(abs(now), 1e-300)):
                stop["reason"] = "stalled"
                return "stop"
        return None

    if verbose:
        ranges = "  ".join(f"β{i} ∈ [{g['low']:.4g}, {g['high']:.4g}]"
                           for i, g in enumerate(gene_space))
        print(f"Genetic search ({objective_mechanism}, {form.label}, exact objective): {ranges}")

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_batch,
        fitness_batch_size=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        initial_population=initial_population,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type=make_gaussian_mutation(gene_space, mutation_probability,
                                             mutation_sigma_fraction, rng),
        on_generation=on_generation,
        random_seed=seed,
        suppress_warnings=True,
    )
    start = time.time()
    ga.run()
    genetic_solution, genetic_fitness, _ = ga.best_solution()
    best_coefficients = np.array(genetic_solution, dtype=float)
    best_welfare = float(genetic_fitness)
    summary = {"optimizer": "genetic", "genetic_coefficients": best_coefficients.tolist(),
               "genetic_welfare": best_welfare,
               "generations_run": int(ga.generations_completed),
               "stop_reason": stop["reason"], "history": history,
               "local_refinement": local_refinement, "local_starts": []}
    if verbose:
        print(f"Genetic stage done in {time.time() - start:.1f}s after "
              f"{ga.generations_completed} generations: welfare {best_welfare:.8g}")

    if local_refinement == "nelder_mead" and not _past(deadline):
        stage["name"] = "nelder_mead"
        fitness = [welfare_of(solution) for solution in ga.population]
        starts = _best_distinct(ga.population, fitness, num_local_starts)
        if not any(np.allclose(start, best_coefficients) for start in starts):
            starts = [best_coefficients] + starts[:-1]
        local_start = time.time()
        if workers > 1 and len(starts) > 1:
            jobs = [{"objective": (type_distribution, form, num_bidders, num_goods,
                                   objective_mechanism), "start": start, "lows": lows,
                     "highs": highs, "max_evaluations": local_max_evaluations,
                     "deadline": deadline} for start in starts]
            local_results = [r for r in parallel_map(_nelder_mead_job, jobs, workers) if r]
            for result in local_results:  # convergence trace of the parallel starts
                for point in result["history"]:
                    if point["welfare"] > stage["best"]:
                        stage["best"] = point["welfare"]
                        trace.append({"stage": "nelder_mead", "evaluations": None,
                                      "time": point["time"], "welfare": point["welfare"]})
            trace.sort(key=lambda entry: entry["time"])
        else:
            local_results = refine_with_nelder_mead(welfare_of, starts, lows, highs,
                                                    local_max_evaluations, deadline)
        summary["local_starts"] = local_results
        best_local = max(local_results, key=lambda result: result["welfare"],
                         default={"welfare": -np.inf})
        if best_local["welfare"] > best_welfare:
            best_coefficients = np.array(best_local["coefficients"])
            best_welfare = best_local["welfare"]
        if verbose:
            print(f"Nelder–Mead from {len(starts)} starts in {time.time() - local_start:.1f}s: "
                  f"welfare {best_welfare:.8g} (genetic {summary['genetic_welfare']:.8g}); "
                  f"converged {sum(r['converged'] for r in local_results)}/{len(local_results)}")

    summary["num_objective_evaluations"] = len(cache) + (
        sum(r["evaluations"] for r in summary["local_starts"]) if workers > 1 else 0)
    trace.append({"stage": "end", "evaluations": len(cache), "time": time.time(),
                  "welfare": float(best_welfare)})
    summary["trace"] = trace
    summary["form"] = form.spec(best_coefficients)
    if verbose:
        print(f"Best τ: β = {[f'{c:.6g}' for c in best_coefficients]}, welfare {best_welfare:.8g}")
    return SearchResult([float(c) for c in best_coefficients], best_welfare, trajectory, summary)
