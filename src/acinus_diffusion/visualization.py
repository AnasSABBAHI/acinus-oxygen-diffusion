"""
Visualization utilities for concentration fields and results.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_concentration_field(X, Y, concentration, title="Oxygen Concentration",
                             cmap='viridis', figsize=(10, 8)):
    """
    Plot 2D concentration field.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate arrays
    concentration : ndarray
        2D concentration field
    title : str
        Plot title
    cmap : str
        Colormap
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(X, Y, concentration, shading='auto', cmap=cmap)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Concentration (mol/m³)')

    return fig, ax


def plot_oxygen_flux(lambda_values, flux_values, figsize=(10, 6)):
    """
    Plot oxygen flux vs screening length.

    Parameters
    ----------
    lambda_values : array-like
        Screening length values
    flux_values : array-like
        Corresponding flux values
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lambda_values, flux_values, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Screening Length Λ (m)')
    ax.set_ylabel('Oxygen Flux (mol/(m·s))')
    ax.set_title('Oxygen Flux vs Screening Length')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_comparison(concentration_normal, concentration_copd, X, Y,
                    titles=('Normal', 'COPD'), figsize=(12, 5)):
    """
    Plot comparison between normal and COPD cases.

    Parameters
    ----------
    concentration_normal : ndarray
        Concentration field for normal case
    concentration_copd : ndarray
        Concentration field for COPD case
    X, Y : ndarray
        Coordinate arrays
    titles : tuple
        Titles for subplots
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Normal case
    im1 = axes[0].pcolormesh(X, Y, concentration_normal, shading='auto', cmap='viridis')
    axes[0].set_title(f'{titles[0]} - Oxygen Concentration')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes[0], label='Concentration (mol/m³)')

    # COPD case
    im2 = axes[1].pcolormesh(X, Y, concentration_copd, shading='auto', cmap='viridis')
    axes[1].set_title(f'{titles[1]} - Oxygen Concentration')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    plt.colorbar(im2, ax=axes[1], label='Concentration (mol/m³)')

    fig.tight_layout()
    return fig, axes
