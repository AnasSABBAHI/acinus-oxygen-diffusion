"""
Tests for the quasi-stationary diffusion solver.
"""

import numpy as np

from acinus_diffusion.geometry import create_copd_domain
from acinus_diffusion.quasistationary import solve_quasistationary_diffusion
from acinus_diffusion.stationary import solve_stationary_diffusion


def test_quasistationary_time_dependence():
    """Solutions at different times differ."""
    N, M = 30, 30
    L = 0.01
    solutions = [solve_quasistationary_diffusion(N, M, L, t)[0] for t in (0, 1, 2)]

    assert not np.allclose(solutions[0], solutions[1])
    assert not np.allclose(solutions[0], solutions[2])


def test_periodic_behavior():
    """Solutions one breathing period apart are identical."""
    N, M = 20, 20
    L = 0.01
    frequency = 0.3
    omega = 2 * np.pi * frequency

    C1, _ = solve_quasistationary_diffusion(N, M, L, 0, omega=omega)
    C2, _ = solve_quasistationary_diffusion(N, M, L, 1 / frequency, omega=omega)

    np.testing.assert_allclose(C1, C2, rtol=1e-10)


def test_boundary_amplitude():
    """The breathing boundary value oscillates between C_a - C_b and C_a - C_b - 2 C_1."""
    N, M = 25, 25
    L = 0.01
    C_a, C_b, C_1 = 8.4, 5.1e-4, 4.2
    omega = 1.0

    C_max, C_top_max = solve_quasistationary_diffusion(N, M, L, 0, C_a, C_b, C_1, omega)
    C_min, C_top_min = solve_quasistationary_diffusion(N, M, L, np.pi / omega,
                                                       C_a, C_b, C_1, omega)

    np.testing.assert_allclose(C_top_max, C_a - C_b, rtol=1e-10)
    np.testing.assert_allclose(C_top_min, C_a - C_b - 2 * C_1, rtol=1e-10)
    np.testing.assert_allclose(C_max[-1, :], C_top_max + C_b, rtol=1e-10)
    np.testing.assert_allclose(C_min[-1, :], C_top_min + C_b, rtol=1e-10)


def test_matches_stationary_solver():
    """At each time the field is the stationary solution for the current boundary value."""
    N, M, L = 30, 30, 0.01
    _, _, mask = create_copd_domain(N, M, L, destruction_factor=0.3, seed=0)
    for t in (0.0, 0.7, 2.1):
        for m in (None, mask):
            C, C_top = solve_quasistationary_diffusion(N, M, L, t, mask=m)
            C_b = 5.1e-4
            expected = solve_stationary_diffusion(N, M, L, C_a=C_top + C_b, mask=m)
            np.testing.assert_allclose(C, expected, rtol=1e-10, equal_nan=True)
