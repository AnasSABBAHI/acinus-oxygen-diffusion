"""
Stationary regime oxygen diffusion solver.

Solves the Laplace equation ΔC = 0 on a square domain [0, L] x [0, L] for the
shifted concentration u = C - C_b, with

* top boundary    (y = L, alveolar air) : Dirichlet  u = C_a - C_b
* bottom boundary (y = 0, capillaries)  : Robin      ∂u/∂n = -u / Λ
* side boundaries (x = 0, x = L)        : Neumann    ∂u/∂n = 0

The grid has N points along x and M points along y, including the boundaries,
so that the spacings are dx = L / (N - 1) and dy = L / (M - 1).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .constants import PhysicalConstants


def grid_spacing(N, M, L):
    """Return the grid spacings (dx, dy) of an N x M grid on [0, L]²."""
    if N < 3 or M < 3:
        raise ValueError("N and M must be at least 3")
    return L / (N - 1), L / (M - 1)


def assemble_system(N, M, L, top_value, lambda_param, mask=None):
    """
    Assemble the finite-difference system A u = b for the shifted
    concentration u = C - C_b.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L : float
        Domain side length (m)
    top_value : float
        Dirichlet value of u on the top boundary (mol/m³)
    lambda_param : float
        Screening length Λ (m)
    mask : ndarray of bool, shape (M, N), optional
        True where tissue/air is present. Masked-out points (e.g. destroyed
        tissue in COPD) are excluded from the domain and act as impermeable
        (no-flux) obstacles. Defaults to the full domain.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        System matrix of shape (N*M, N*M)
    b : ndarray
        Right-hand side vector
    """
    dx, dy = grid_spacing(N, M, L)
    if mask is None:
        mask = np.ones((M, N), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (M, N):
            raise ValueError(f"mask must have shape {(M, N)}, got {mask.shape}")

    def index(i, j):
        """Convert 2D grid indices (i along x, j along y) to a 1D index."""
        return j * N + i

    def active(i, j):
        return 0 <= i < N and 0 <= j < M and mask[j, i]

    rows, cols, vals = [], [], []
    b = np.zeros(N * M)

    def add(k, col, value):
        rows.append(k)
        cols.append(col)
        vals.append(value)

    for j in range(M):
        for i in range(N):
            k = index(i, j)

            # Points outside the domain: trivial equation u = 0
            if not mask[j, i]:
                add(k, k, 1.0)
                continue

            # Top boundary (Dirichlet): u = C_a - C_b
            if j == M - 1:
                add(k, k, 1.0)
                b[k] = top_value
                continue

            # Bottom boundary (Robin): (u[1] - u[0]) / dy = u[0] / Λ
            if j == 0:
                add(k, k, 1.0 + dy / lambda_param)
                if active(i, 1):
                    add(k, index(i, 1), -1.0)
                continue

            # Interior and side points: 5-point Laplacian. A missing neighbour
            # (domain side or destroyed tissue) is replaced by a mirror ghost
            # point, which enforces the no-flux condition ∂u/∂n = 0.
            diag = 0.0
            for (di, dj), h in (((1, 0), dx), ((-1, 0), dx),
                                ((0, 1), dy), ((0, -1), dy)):
                coef = 1.0 / h**2
                diag -= coef
                if active(i + di, j + dj):
                    add(k, index(i + di, j + dj), coef)
                elif active(i - di, j - dj):
                    add(k, index(i - di, j - dj), coef)
                else:
                    # Isolated in this direction: the term vanishes
                    diag += coef
            if diag == 0.0:
                # Completely isolated point: no diffusion possible, set u = 0
                diag = 1.0
            add(k, k, diag)

    A = csr_matrix((vals, (rows, cols)), shape=(N * M, N * M))
    return A, b


def solve_stationary_diffusion(N, M, L, C_a=None, C_b=None, lambda_param=None,
                               mask=None):
    """
    Solve the stationary diffusion equation ΔC = 0 with mixed boundary
    conditions.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L : float
        Domain side length (m)
    C_a : float, optional
        Alveolar oxygen concentration (mol/m³). Defaults to PhysicalConstants.C_AIR
    C_b : float, optional
        Blood oxygen concentration (mol/m³). Defaults to PhysicalConstants.C_BLOOD
    lambda_param : float, optional
        Screening length Λ (m). Defaults to PhysicalConstants.LAMBDA_TYPICAL
    mask : ndarray of bool, shape (M, N), optional
        True where tissue is present (see :func:`assemble_system`). Points
        outside the mask are returned as NaN.

    Returns
    -------
    concentration : ndarray, shape (M, N)
        Oxygen concentration (mol/m³); row j corresponds to y = j * dy.
    """
    if C_a is None:
        C_a = PhysicalConstants.C_AIR
    if C_b is None:
        C_b = PhysicalConstants.C_BLOOD
    if lambda_param is None:
        lambda_param = PhysicalConstants.LAMBDA_TYPICAL

    A, b = assemble_system(N, M, L, C_a - C_b, lambda_param, mask)
    solution = spsolve(A, b)

    # Reshape and add the blood concentration baseline
    concentration = solution.reshape((M, N)) + C_b
    if mask is not None:
        concentration = np.where(mask, concentration, np.nan)
    return concentration


def analytical_stationary_solution(y, L, C_a=None, C_b=None, lambda_param=None):
    """
    Exact solution of the stationary problem on the full (undeformed) domain.

    With no-flux side walls the problem is one-dimensional and

        C(y) = C_b + (C_a - C_b) * (y + Λ) / (L + Λ)

    Parameters
    ----------
    y : array-like
        Vertical coordinate(s) (m)
    L : float
        Domain side length (m)
    C_a, C_b, lambda_param : float, optional
        Same as in :func:`solve_stationary_diffusion`

    Returns
    -------
    ndarray
        Concentration at the given heights (mol/m³)
    """
    if C_a is None:
        C_a = PhysicalConstants.C_AIR
    if C_b is None:
        C_b = PhysicalConstants.C_BLOOD
    if lambda_param is None:
        lambda_param = PhysicalConstants.LAMBDA_TYPICAL
    y = np.asarray(y, dtype=float)
    return C_b + (C_a - C_b) * (y + lambda_param) / (L + lambda_param)


def calculate_oxygen_flux(concentration, dx, lambda_param, C_b=None, D=None):
    """
    Total oxygen flux absorbed by the capillaries (bottom Robin boundary).

    The local flux density is D * ∂C/∂y = D * (C - C_b) / Λ = W * (C - C_b),
    integrated along the boundary with the trapezoidal rule. Destroyed tissue
    (NaN values) does not contribute.

    Parameters
    ----------
    concentration : ndarray, shape (M, N)
        Concentration field (mol/m³)
    dx : float
        Grid spacing along x (m)
    lambda_param : float
        Screening length Λ (m)
    C_b : float, optional
        Blood oxygen concentration (mol/m³). Defaults to PhysicalConstants.C_BLOOD
    D : float, optional
        Oxygen diffusion coefficient (m²/s). Defaults to PhysicalConstants.D_O2_AIR

    Returns
    -------
    flux : float
        Oxygen flux per unit depth of the 2D domain (mol/(m·s))
    """
    if C_b is None:
        C_b = PhysicalConstants.C_BLOOD
    if D is None:
        D = PhysicalConstants.D_O2_AIR
    density = np.nan_to_num(D * (concentration[0, :] - C_b) / lambda_param)
    return dx * (np.sum(density) - 0.5 * (density[0] + density[-1]))


def analytical_oxygen_flux(L, C_a=None, C_b=None, lambda_param=None, D=None):
    """
    Exact oxygen flux on the full domain: D * (C_a - C_b) * L / (L + Λ).

    For L << Λ the flux is limited by the membrane (≈ W (C_a - C_b) L) and for
    L >> Λ by diffusion through the acinus (≈ D (C_a - C_b)).
    """
    if C_a is None:
        C_a = PhysicalConstants.C_AIR
    if C_b is None:
        C_b = PhysicalConstants.C_BLOOD
    if lambda_param is None:
        lambda_param = PhysicalConstants.LAMBDA_TYPICAL
    if D is None:
        D = PhysicalConstants.D_O2_AIR
    return D * (C_a - C_b) * L / (L + lambda_param)
