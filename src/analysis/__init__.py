"""Analysis modules for generating tables and figures."""

from .generate_tables import (
    generate_performance_table,
    generate_convergence_table,
    generate_computational_table,
    generate_hyperparameter_table,
)
from .generate_figures import (
    generate_all_figures,
    generate_convergence_figures,
    generate_trajectory_figures,
)

__all__ = [
    'generate_performance_table',
    'generate_convergence_table',
    'generate_computational_table',
    'generate_hyperparameter_table',
    'generate_all_figures',
    'generate_convergence_figures',
    'generate_trajectory_figures',
]
