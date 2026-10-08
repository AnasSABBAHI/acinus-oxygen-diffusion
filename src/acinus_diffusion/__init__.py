"""
Acinus Oxygen Diffusion Model

A computational model for simulating oxygen diffusion in the pulmonary acinus
using finite difference methods with various boundary conditions.
"""

__version__ = "0.1.0"

from .boundary_conditions import DirichletBC, NeumannBC, RobinBC
from .constants import PhysicalConstants
from .geometry import (create_copd_domain, create_deformed_domain,
                       create_rectangular_domain)
from .quasistationary import (animate_solution, breathing_boundary_value,
                              solve_quasistationary_diffusion)
from .stationary import (analytical_oxygen_flux,
                         analytical_stationary_solution, assemble_system,
                         calculate_oxygen_flux, grid_spacing,
                         solve_stationary_diffusion)
from .visualization import (plot_comparison, plot_concentration_field,
                            plot_oxygen_flux)

__all__ = [
    'solve_stationary_diffusion',
    'analytical_stationary_solution',
    'analytical_oxygen_flux',
    'calculate_oxygen_flux',
    'assemble_system',
    'grid_spacing',
    'solve_quasistationary_diffusion',
    'breathing_boundary_value',
    'animate_solution',
    'DirichletBC',
    'NeumannBC',
    'RobinBC',
    'create_rectangular_domain',
    'create_deformed_domain',
    'create_copd_domain',
    'PhysicalConstants',
    'plot_concentration_field',
    'plot_oxygen_flux',
    'plot_comparison',
]
