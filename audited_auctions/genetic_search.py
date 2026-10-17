"""
genetic_search.py

Genetic-algorithm search for a polynomial threshold function
      τ(e) = β_0 + β_1 e + ... + β_d e^d
that maximizes the exact expected welfare of a participant-audit mechanism
(VCG-PA by default, or IVCG-PA) for a type distribution.

The fitness of a coefficient vector is the exact expected welfare
(threshold_evaluation.expected_welfare), computed on the distribution itself,
so it has no Monte Carlo noise and does not fit a particular set of simulated
auctions. The GA mechanics (pygad, tournament selection, single-point
crossover, Gaussian mutation scaled to each gene's range, optional warm start
and restarts) are those of the original Collateralized_Auction_genetic2.py.
"""

import time

import numpy as np
import pygad

from .threshold_evaluation import expected_welfare
from .threshold_functions import polynomial_threshold


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


def run_genetic_search(type_distribution, polynomial_degree, num_bidders, num_goods,
                       objective_mechanism="vcg_pa", num_generations=500, sol_per_pop=50,
                       num_parents_mating=15, mutation_probability=0.25,
                       mutation_sigma_fraction=0.15, range_multiplier=3.0, seed=None,
                       initial_solution=None, verbose=True):
    """Maximize exact expected welfare of objective_mechanism over polynomial τ.

    Returns (best_coefficients, best_welfare, trajectory), where trajectory
    lists every coefficient vector in each generation's population.
    """
    num_genes = polynomial_degree + 1
    rng = np.random.default_rng(seed)
    gene_space = compute_gene_space(type_distribution, polynomial_degree, range_multiplier)
    lows = np.array([gene["low"] for gene in gene_space])
    highs = np.array([gene["high"] for gene in gene_space])

    initial_population = rng.uniform(lows, highs, size=(sol_per_pop, num_genes))
    if initial_solution is not None:
        seed_row = np.zeros(num_genes)
        shared = min(len(initial_solution), num_genes)
        seed_row[:shared] = initial_solution[:shared]
        initial_population[0] = np.clip(seed_row, lows, highs)

    cache = {}

    def welfare_of(coefficients):
        key = tuple(np.round(coefficients, 12))
        if key not in cache:
            cache[key] = expected_welfare(type_distribution, polynomial_threshold(coefficients),
                                          num_bidders, num_goods, objective_mechanism)
        return cache[key]

    def fitness_batch(ga_instance, solutions, solution_index):  # noqa: ARG001
        return np.array([welfare_of(solution) for solution in np.atleast_2d(solutions)])

    trajectory = []

    def on_generation(ga_instance):
        trajectory.extend(solution.tolist() for solution in ga_instance.population)
        if verbose and (ga_instance.generations_completed % 25 == 0
                        or ga_instance.generations_completed == num_generations):
            best, fitness, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness)
            print(f"  generation {ga_instance.generations_completed:4d}  welfare={fitness:.6g}  "
                  f"β={[f'{c:.4g}' for c in best]}")

    if verbose:
        ranges = "  ".join(f"β{i} ∈ [{g['low']:.4g}, {g['high']:.4g}]"
                           for i, g in enumerate(gene_space))
        print(f"Genetic search ({objective_mechanism}, exact objective): {ranges}")

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
    best_solution, best_fitness, _ = ga.best_solution()
    if verbose:
        print(f"Genetic search done in {time.time() - start:.1f}s: welfare {best_fitness:.6g}, "
              f"β = {[f'{c:.6g}' for c in best_solution]}")
    return [float(c) for c in best_solution], float(best_fitness), trajectory
