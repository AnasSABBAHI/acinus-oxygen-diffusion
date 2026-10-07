"""
Tests for geometry creation functions.
"""

import numpy as np
import pytest

from acinus_diffusion.geometry import (create_copd_domain, create_deformed_domain,
                                       create_rectangular_domain)


def test_rectangular_domain():
    """Rectangular domain has the right shape and coordinates."""
    N, M = 50, 60
    L_x, L_y = 0.01, 0.02
    X, Y = create_rectangular_domain(N, M, L_x, L_y)

    assert X.shape == (M, N)
    assert Y.shape == (M, N)
    assert np.allclose(X[0, :], np.linspace(0, L_x, N))
    assert np.allclose(Y[:, 0], np.linspace(0, L_y, M))


def test_deformed_domain():
    """Deformed domain contains both tissue and lesion points."""
    N, M = 40, 40
    L_x, L_y = 0.01, 0.01
    X, Y, mask = create_deformed_domain(N, M, L_x, L_y, deformation_factor=0.3)

    assert X.shape == (M, N)
    assert Y.shape == (M, N)
    assert mask.shape == (M, N)
    assert mask.dtype == bool
    assert np.any(mask) and not np.all(mask)


def test_deformation_factor_range():
    """More deformation destroys more tissue; no deformation destroys none."""
    N, M = 20, 20
    L_x, L_y = 0.01, 0.01

    _, _, mask_zero = create_deformed_domain(N, M, L_x, L_y, 0)
    _, _, mask_max = create_deformed_domain(N, M, L_x, L_y, 1)

    assert np.all(mask_zero)
    assert np.sum(~mask_max) > np.sum(~mask_zero)

    with pytest.raises(ValueError):
        create_deformed_domain(N, M, L_x, L_y, 1.5)


def test_copd_domain_reproducible():
    """A fixed seed gives the same geometry; boundaries stay intact."""
    N, M, L = 50, 50, 0.01
    _, _, mask1 = create_copd_domain(N, M, L, 0.4, seed=42)
    _, _, mask2 = create_copd_domain(N, M, L, 0.4, seed=42)

    np.testing.assert_array_equal(mask1, mask2)
    assert np.all(mask1[:2, :]) and np.all(mask1[-2:, :])
    assert not np.all(mask1)


def test_copd_domain_severity():
    """Higher severity destroys more tissue."""
    N, M, L = 60, 60, 0.01
    destroyed = [np.sum(~create_copd_domain(N, M, L, s, seed=1)[2])
                 for s in (0.0, 0.2, 0.5)]
    assert destroyed[0] == 0
    assert destroyed[0] < destroyed[1] < destroyed[2]
