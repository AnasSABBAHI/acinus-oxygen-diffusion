"""
Tests for the stationary diffusion solver.
"""

import numpy as np
import pytest

from acinus_diffusion.constants import PhysicalConstants
from acinus_diffusion.geometry import create_deformed_domain
from acinus_diffusion.stationary import (analytical_oxygen_flux,
                                         analytical_stationary_solution,
                                         calculate_oxygen_flux, grid_spacing,
                                         solve_stationary_diffusion)


def test_stationary_solution_shape():
    """Solution has one row per y grid point and one column per x point."""
    N, M = 50, 40
    C = solve_stationary_diffusion(N, M, 0.01)
    assert C.shape == (M, N)


def test_boundary_conditions():
    """Dirichlet (top) and Robin (bottom) conditions are satisfied."""
    N, M = 30, 30
    L = 0.01
    C_a, C_b, lambda_param = 8.4, 5.1e-4, 0.28

    C = solve_stationary_diffusion(N, M, L, C_a, C_b, lambda_param)

    # Dirichlet boundary (top)
    np.testing.assert_allclose(C[-1, :], C_a, rtol=1e-10)

    # Robin boundary (bottom): (u1 - u0) / dy = u0 / Λ with u = C - C_b
    _, dy = grid_spacing(N, M, L)
    expected_robin = (C[1, :] - C_b) / (1 + dy / lambda_param) + C_b
    np.testing.assert_allclose(C[0, :], expected_robin, rtol=1e-10)

    # Neumann boundaries (sides): no variation across the side walls
    np.testing.assert_allclose(C[:, 0], C[:, 1], rtol=1e-10)
    np.testing.assert_allclose(C[:, -1], C[:, -2], rtol=1e-10)


@pytest.mark.parametrize("N, M", [(10, 10), (40, 25), (60, 60)])
@pytest.mark.parametrize("lambda_param", [0.001, 0.01, 0.28])
def test_matches_analytical_solution(N, M, lambda_param):
    """The scheme reproduces the exact linear profile on the full domain."""
    L = 0.01
    C = solve_stationary_diffusion(N, M, L, lambda_param=lambda_param)
    y = np.linspace(0, L, M)
    expected = analytical_stationary_solution(y, L, lambda_param=lambda_param)
    np.testing.assert_allclose(C, np.broadcast_to(expected[:, None], (M, N)),
                               rtol=1e-9)


def test_flux_conservation():
    """In steady state the flux entering at the top leaves at the bottom."""
    N, M = 40, 40
    L = 0.01
    lambda_param = 0.01
    D = PhysicalConstants.D_O2_AIR
    C = solve_stationary_diffusion(N, M, L, lambda_param=lambda_param)
    dx, dy = grid_spacing(N, M, L)

    bottom_flux = calculate_oxygen_flux(C, dx, lambda_param)
    top_density = D * (C[-1, :] - C[-2, :]) / dy
    top_flux = dx * (np.sum(top_density) - 0.5 * (top_density[0] + top_density[-1]))

    np.testing.assert_allclose(top_flux, bottom_flux, rtol=1e-9)


@pytest.mark.parametrize("lambda_param", [0.001, 0.01, 0.28, 2.0])
def test_flux_matches_analytical(lambda_param):
    """Numerical flux equals D (C_a - C_b) L / (L + Λ)."""
    N, M, L = 30, 30, 0.01
    C = solve_stationary_diffusion(N, M, L, lambda_param=lambda_param)
    dx, _ = grid_spacing(N, M, L)
    flux = calculate_oxygen_flux(C, dx, lambda_param)
    expected = analytical_oxygen_flux(L, lambda_param=lambda_param)
    np.testing.assert_allclose(flux, expected, rtol=1e-9)


def test_flux_decreases_with_screening_length():
    """A larger screening length (less permeable membrane) reduces the flux."""
    lambdas = np.array([0.001, 0.01, 0.1, 1.0])
    fluxes = analytical_oxygen_flux(0.01, lambda_param=lambdas)
    assert np.all(np.diff(fluxes) < 0)


def test_masked_domain():
    """Destroyed tissue is excluded and reduces the absorbed flux."""
    N, M, L = 40, 40, 0.01
    lambda_param = 0.01
    _, _, mask = create_deformed_domain(N, M, L, L, deformation_factor=0.6)

    C_normal = solve_stationary_diffusion(N, M, L, lambda_param=lambda_param)
    C_copd = solve_stationary_diffusion(N, M, L, lambda_param=lambda_param,
                                        mask=mask)

    assert np.all(np.isnan(C_copd[~mask]))
    assert not np.any(np.isnan(C_copd[mask]))
    np.testing.assert_allclose(C_copd[-1, :], PhysicalConstants.C_AIR)

    dx, _ = grid_spacing(N, M, L)
    assert (calculate_oxygen_flux(C_copd, dx, lambda_param)
            < calculate_oxygen_flux(C_normal, dx, lambda_param))


def test_full_mask_matches_unmasked():
    """A mask covering the whole domain gives the unmasked solution."""
    N, M, L = 20, 20, 0.01
    mask = np.ones((M, N), dtype=bool)
    np.testing.assert_allclose(solve_stationary_diffusion(N, M, L, mask=mask),
                               solve_stationary_diffusion(N, M, L))


def test_invalid_grid():
    with pytest.raises(ValueError):
        solve_stationary_diffusion(2, 10, 0.01)
