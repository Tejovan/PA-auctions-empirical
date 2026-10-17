"""
audited_auctions: exact computation and simulation of auctions with participant
audits (VCG-PA, IVCG-PA) and the first-best recipient audit, for a common
participation threshold function τ(e). See audited_auctions/README.md.
"""

from .threshold_evaluation import (
    MECHANISM_LABELS,
    MECHANISMS,
    ThresholdEvaluation,
    build_mechanism_models,
    evaluate_threshold,
)
from .threshold_functions import (
    as_threshold_function,
    constant_threshold,
    polynomial_threshold,
    tabulated_threshold,
)
from .type_distributions import (
    TypeDistribution,
    gaussian_mixture_distribution,
    gridded_density_distribution,
    population_distribution,
    population_distribution_from_csv,
    synthetic_distribution,
)
