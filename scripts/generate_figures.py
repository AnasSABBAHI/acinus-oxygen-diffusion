"""
Generate the figures used in docs/report.md and docs/presentation.md.

Usage (from the repository root):

    python scripts/generate_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from acinus_diffusion import (PhysicalConstants as P, analytical_oxygen_flux,  # noqa: E402
                              calculate_oxygen_flux, create_copd_domain, grid_spacing,
                              solve_quasistationary_diffusion, solve_stationary_diffusion)

OUT = os.path.join(ROOT, 'docs', 'figures')
N = M = 100
L = 0.01
SEED = 0


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved docs/figures/{name}")


def stationary_figure(X, Y):
    C = solve_stationary_diffusion(N, M, L)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, C, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Concentration (mol/m³)')
    ax.set(xlabel='x (m)', ylabel='y (m)', title=f'Stationary regime, Λ = {P.LAMBDA_TYPICAL} m')
    ax.set_aspect('equal')
    save(fig, 'stationary.png')


def flux_lambda_figure():
    dx, _ = grid_spacing(N, M, L)
    lambdas = np.logspace(-4, np.log10(2.0), 25)
    fluxes = [calculate_oxygen_flux(solve_stationary_diffusion(N, M, L, lambda_param=lam), dx, lam)
              for lam in lambdas]
    lam_fine = np.logspace(-4, np.log10(2.0), 200)
    delta_C = P.C_AIR - P.C_BLOOD

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(lambdas, fluxes, 'o', label='Finite differences')
    ax.loglog(lam_fine, analytical_oxygen_flux(L, lambda_param=lam_fine), 'k-',
              label='Exact: D ΔC L / (L + Λ)')
    ax.loglog(lam_fine, np.full_like(lam_fine, P.D_O2_AIR * delta_C), 'k--', alpha=0.6,
              label='Diffusion-limited')
    ax.loglog(lam_fine, P.D_O2_AIR / lam_fine * delta_C * L, 'k:', alpha=0.6,
              label='Membrane-limited')
    ax.axvline(P.LAMBDA_TYPICAL, color='r', alpha=0.5, label='Physiological Λ')
    ax.set_ylim(min(fluxes) / 2, max(fluxes) * 2)
    ax.set(xlabel='Screening length Λ (m)', ylabel='Oxygen flux (mol/(m·s))',
           title='Oxygen flux vs screening length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, 'flux_lambda.png')


def breathing_animation(X, Y):
    period = 1 / P.BREATHING_RATE_REST
    times = np.linspace(0, period, 30, endpoint=False)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    C, _ = solve_quasistationary_diffusion(N, M, L, 0)
    im = ax.pcolormesh(X, Y, C, shading='auto', cmap='viridis', vmin=P.C_BLOOD, vmax=P.C_AIR)
    fig.colorbar(im, ax=ax, label='Concentration (mol/m³)')
    ax.set(xlabel='x (m)', ylabel='y (m)')
    ax.set_aspect('equal')

    def update(frame):
        C, _ = solve_quasistationary_diffusion(N, M, L, times[frame])
        im.set_array(C.ravel())
        ax.set_title(f'Breathing cycle, t = {times[frame]:.2f} s')
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(times))
    anim.save(os.path.join(OUT, 'breathing.gif'), writer=PillowWriter(fps=10), dpi=70)
    plt.close(fig)
    print("Saved docs/figures/breathing.gif")


def copd_figure(X, Y):
    _, _, mask = create_copd_domain(N, M, L, 0.3, seed=SEED)
    C = solve_stationary_diffusion(N, M, L, mask=mask)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pcolormesh(X, Y, mask, shading='auto', cmap='Greys_r')
    axes[0].set_title(f'COPD domain (white = tissue), {(1 - mask.mean()) * 100:.1f}% destroyed')
    im = axes[1].pcolormesh(X, Y, C, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=axes[1], label='Concentration (mol/m³)')
    axes[1].set_title('Concentration in the COPD domain')
    for ax in axes:
        ax.set(xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal')
    save(fig, 'copd_domain.png')


def main():
    os.makedirs(OUT, exist_ok=True)
    x = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, x)
    stationary_figure(X, Y)
    flux_lambda_figure()
    breathing_animation(X, Y)
    copd_figure(X, Y)


if __name__ == '__main__':
    main()
