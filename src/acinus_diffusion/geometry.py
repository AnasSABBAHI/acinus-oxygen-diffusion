"""
Geometry creation for different domain types.

Masks follow a single convention throughout the package: True marks points
where tissue is present (part of the computational domain) and False marks
destroyed tissue.
"""

import numpy as np


def create_rectangular_domain(N, M, L_x, L_y):
    """
    Create a rectangular computational domain.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L_x, L_y : float
        Domain dimensions in x and y directions (m)

    Returns
    -------
    X, Y : ndarray, shape (M, N)
        2D coordinate arrays
    """
    x = np.linspace(0, L_x, N)
    y = np.linspace(0, L_y, M)
    X, Y = np.meshgrid(x, y)
    return X, Y


def create_deformed_domain(N, M, L_x, L_y, deformation_factor=0.3):
    """
    Create a domain with a central elliptical lesion to simulate COPD.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L_x, L_y : float
        Domain dimensions (m)
    deformation_factor : float
        Relative size of the lesion (0 = no deformation, 1 = ellipse
        inscribed in the domain)

    Returns
    -------
    X, Y : ndarray, shape (M, N)
        Coordinate arrays
    mask : ndarray of bool, shape (M, N)
        True where tissue is present, False inside the lesion
    """
    if not 0 <= deformation_factor <= 1:
        raise ValueError("deformation_factor must be between 0 and 1")

    X, Y = create_rectangular_domain(N, M, L_x, L_y)
    if deformation_factor == 0:
        return X, Y, np.ones((M, N), dtype=bool)

    center_x, center_y = L_x / 2, L_y / 2
    rx = L_x * deformation_factor / 2
    ry = L_y * deformation_factor / 2

    lesion = ((X - center_x)**2 / rx**2 + (Y - center_y)**2 / ry**2) <= 1
    return X, Y, ~lesion


def create_copd_domain(N, M, L, destruction_factor=0.3, seed=None):
    """
    Create a domain with randomly distributed lesions simulating emphysema.

    Lesions are kept away from the alveolar (top) and capillary (bottom)
    boundaries so that the boundary conditions stay well defined.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L : float
        Domain side length (m)
    destruction_factor : float
        Severity of the tissue destruction (0 = healthy, 1 = maximal)
    seed : int or numpy.random.Generator, optional
        Random seed for reproducible geometries

    Returns
    -------
    X, Y : ndarray, shape (M, N)
        Coordinate arrays
    mask : ndarray of bool, shape (M, N)
        True where tissue is present
    """
    if not 0 <= destruction_factor <= 1:
        raise ValueError("destruction_factor must be between 0 and 1")

    rng = np.random.default_rng(seed)
    X, Y = create_rectangular_domain(N, M, L, L)
    mask = np.ones((M, N), dtype=bool)

    # Elliptical lesions: more severe disease means more and larger holes
    n_lesions = int(round(5 * destruction_factor))
    for _ in range(n_lesions):
        center_x = rng.uniform(0.2 * L, 0.8 * L)
        center_y = rng.uniform(0.2 * L, 0.8 * L)
        rx = L * destruction_factor * rng.uniform(0.1, 0.3)
        ry = L * destruction_factor * rng.uniform(0.1, 0.3)
        mask[((X - center_x)**2 / rx**2 + (Y - center_y)**2 / ry**2) <= 1] = False

    # Scattered small lesions simulating irregular tissue remodeling
    mask[rng.random((M, N)) < 0.1 * destruction_factor] = False

    # Keep the exchange surfaces and the rows next to them intact, so that no
    # boundary point is cut off from the rest of the domain
    mask[:2, :] = True
    mask[-2:, :] = True
    return X, Y, mask
