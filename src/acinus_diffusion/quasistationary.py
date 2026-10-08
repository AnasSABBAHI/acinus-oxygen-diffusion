"""
Quasi-stationary regime oxygen diffusion solver.

Breathing modulates the alveolar concentration periodically. When the
diffusion time L² / D is short compared with the breathing period, the field
follows the boundary instantaneously and ΔC = 0 holds at every time t with a
time-dependent Dirichlet condition on the top boundary:

    u(x, y = L, t) = C_a - C_b + C_1 (cos(ω t) - 1)
"""

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import spsolve

from .constants import PhysicalConstants
from .stationary import assemble_system, calculate_oxygen_flux, grid_spacing


def breathing_boundary_value(time, C_a=None, C_b=None, C_1=None, omega=None):
    """
    Time-dependent Dirichlet value of u = C - C_b on the alveolar boundary.

    Returns C_a - C_b + C_1 (cos(ω t) - 1) (mol/m³).
    """
    if C_a is None:
        C_a = PhysicalConstants.C_AIR
    if C_b is None:
        C_b = PhysicalConstants.C_BLOOD
    if C_1 is None:
        C_1 = PhysicalConstants.C_REST
    if omega is None:
        omega = PhysicalConstants.OMEGA_REST
    return C_a - C_b + C_1 * (np.cos(omega * time) - 1)


@lru_cache(maxsize=32)
def _unit_solution(N, M, L, lambda_param, mask_bytes):
    """Solution u for a unit top boundary value (cached per geometry)."""
    mask = None
    if mask_bytes is not None:
        mask = np.frombuffer(mask_bytes, dtype=bool).reshape((M, N))
    A, b = assemble_system(N, M, L, 1.0, lambda_param, mask)
    solution = spsolve(A, b).reshape((M, N))
    solution.setflags(write=False)
    return solution


def solve_quasistationary_diffusion(N, M, L, time, C_a=None, C_b=None,
                                    C_1=None, omega=None, lambda_param=None,
                                    mask=None):
    """
    Solve quasi-stationary diffusion with a time-dependent Dirichlet boundary.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L : float
        Domain side length (m)
    time : float
        Current time (s)
    C_a, C_b, C_1 : float, optional
        Alveolar concentration, blood concentration and breathing amplitude
        (mol/m³)
    omega : float, optional
        Breathing angular frequency (rad/s)
    lambda_param : float, optional
        Screening length Λ (m)
    mask : ndarray of bool, shape (M, N), optional
        True where tissue is present (see :func:`assemble_system`). Points
        outside the mask are returned as NaN.

    Returns
    -------
    concentration : ndarray, shape (M, N)
        Concentration field at the given time (mol/m³)
    C_top : float
        Value of u = C - C_b imposed on the top boundary (mol/m³)
    """
    if C_b is None:
        C_b = PhysicalConstants.C_BLOOD
    if lambda_param is None:
        lambda_param = PhysicalConstants.LAMBDA_TYPICAL

    C_top = breathing_boundary_value(time, C_a, C_b, C_1, omega)

    # The problem is linear and homogeneous except for the top boundary, so
    # the solution is C_top times the solution for a unit boundary value.
    mask_bytes = None
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        mask_bytes = np.ascontiguousarray(mask).tobytes()
    unit = _unit_solution(N, M, float(L), float(lambda_param), mask_bytes)

    concentration = C_top * unit + C_b
    if mask is not None:
        concentration = np.where(mask, concentration, np.nan)
    return concentration, C_top


def animate_solution(N=50, M=50, L=0.01, duration=10, fps=10):
    """
    Create an animation of the quasi-stationary solution.

    Parameters
    ----------
    N, M : int
        Number of grid points in x and y directions
    L : float
        Domain side length (m)
    duration : float
        Animation duration (s)
    fps : int
        Frames per second

    Returns
    -------
    animation : FuncAnimation
        Matplotlib animation object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    dx, _ = grid_spacing(N, M, L)
    lambda_param = PhysicalConstants.LAMBDA_TYPICAL

    C, _ = solve_quasistationary_diffusion(N, M, L, 0)

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, M)
    X, Y = np.meshgrid(x, y)

    # Plot 1: Concentration field (fixed colour scale over the cycle)
    im = ax1.pcolormesh(X, Y, C, shading='auto', cmap='viridis',
                        vmin=PhysicalConstants.C_BLOOD,
                        vmax=PhysicalConstants.C_AIR)
    fig.colorbar(im, ax=ax1, label='Concentration (mol/m³)')
    ax1.set_title('Oxygen Concentration Field')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')

    # Plot 2: Time series of flux; the flux is largest when cos(ωt) = 1
    max_flux = calculate_oxygen_flux(C, dx, lambda_param)
    times = []
    fluxes = []
    line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1.1 * max_flux)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Oxygen Flux (mol/(m·s))')
    ax2.set_title('Oxygen Flux vs Time')
    ax2.grid(True)

    def update(frame):
        time = frame / fps
        C, _ = solve_quasistationary_diffusion(N, M, L, time)
        im.set_array(C.ravel())

        times.append(time)
        fluxes.append(calculate_oxygen_flux(C, dx, lambda_param))
        line.set_data(times, fluxes)
        return im, line

    frames = int(duration * fps)
    animation = FuncAnimation(fig, update, frames=frames,
                              interval=1000 / fps, blit=True)

    fig.tight_layout()
    return animation
