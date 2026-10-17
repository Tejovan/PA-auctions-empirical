"""
audited_auctions: exact computation and simulation of auctions with participant
audits (VCG-PA, IVCG-PA) and the first-best recipient audit, for a common
participation threshold function τ(e). See audited_auctions/README.md.
"""

from .threshold_evaluation import (
    MECHANISM_LABELS,
    MECHANISMS,
    ThresholdEvaluation,
    build_mechanism_model,
    build_mechanism_models,
    evaluate_threshold,
    expected_welfare,
    grid_refinement_check,
    penalty_from_table,
)
from .threshold_functions import (
    as_threshold_function,
    constant_threshold,
    polynomial_threshold,
    tabulated_threshold,
)
from .type_distributions import (
    SYNTHETIC_FAMILIES,
    TypeDistribution,
    discrete_externality_distribution,
    gaussian_mixture_distribution,
    gridded_density_distribution,
    load_population_csv,
    population_distribution,
    population_distribution_from_csv,
    synthetic_distribution,
)
