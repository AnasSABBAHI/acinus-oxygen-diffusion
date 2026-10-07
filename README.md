# Acinus Oxygen Diffusion Model

A computational model of oxygen diffusion in the pulmonary acinus, solved with finite differences under physiological boundary conditions.

## Project Overview

The pulmonary acinus is the functional unit of the lung where gas exchange occurs. This project models oxygen transport from the alveolar air to the capillaries in a 2D domain. It solves the diffusion equation in different physiological regimes and in a simplified model of Chronic Obstructive Pulmonary Disease (COPD).

### Key Features

- **Stationary regime**: steady-state oxygen diffusion, validated against the exact solution
- **Quasi-stationary regime**: breathing modulation of the alveolar concentration
- **COPD pathology**: destroyed tissue (masked domain) and reduced membrane permeability
- **Mixed boundary conditions**: Dirichlet (air), Robin (membrane/capillaries), Neumann (sides)
- **Parameter studies**: screening length effect and oxygen flux analysis

## Model

With $u = C - C_b$ on the square $[0, L]^2$:

| Boundary | Physiology | Condition |
|---|---|---|
| top, $y = L$ | alveolar air | $u = C_a - C_b$ (or $C_a - C_b + C_1(\cos\omega t - 1)$ with breathing) |
| bottom, $y = 0$ | membrane / capillaries | $\partial u/\partial n = -u/\Lambda$ |
| sides | symmetry | $\partial u/\partial n = 0$ |

The screening length $\Lambda = D/W$ compares diffusion in air with the membrane permeability $W$. On the full domain the absorbed flux is $\Phi = D (C_a - C_b) L/(L + \Lambda)$. See [docs/report.md](docs/report.md) for details.

## Repository Structure

```
├── src/acinus_diffusion/        # Python package
│   ├── constants.py             # Physical parameters
│   ├── stationary.py            # Stationary solver, exact solution, flux
│   ├── quasistationary.py       # Breathing (quasi-stationary) solver and animation
│   ├── geometry.py              # Rectangular and COPD (deformed) domains
│   ├── boundary_conditions.py   # Dirichlet / Neumann / Robin descriptions
│   └── visualization.py         # Plotting helpers
├── notebooks/
│   ├── 01_stationary_regime.ipynb
│   ├── 02_quasistationary_regime.ipynb
│   └── 03_copd_pathology.ipynb
├── tests/                       # pytest test suite
├── scripts/generate_figures.py  # Regenerates the figures in docs/figures
└── docs/                        # Report, presentation and figures
```

## Getting Started

With conda:

```bash
conda env create -f environment.yaml
conda activate acinus-diffusion
```

Or with pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
import numpy as np
from acinus_diffusion import (PhysicalConstants, calculate_oxygen_flux, create_copd_domain,
                              grid_spacing, solve_quasistationary_diffusion,
                              solve_stationary_diffusion)

N, M, L = 100, 100, 0.01  # grid points and domain size (m)
dx, _ = grid_spacing(N, M, L)

# Stationary regime
C = solve_stationary_diffusion(N, M, L)
flux = calculate_oxygen_flux(C, dx, PhysicalConstants.LAMBDA_TYPICAL)

# Breathing at time t = 1 s
C_t, boundary_value = solve_quasistationary_diffusion(N, M, L, time=1.0)

# COPD domain (True = tissue present)
_, _, mask = create_copd_domain(N, M, L, destruction_factor=0.3, seed=0)
C_copd = solve_stationary_diffusion(N, M, L, mask=mask)
```

The notebooks in `notebooks/` walk through each regime. They import the package from `../src`, so they run without installation.

## Running the Tests

```bash
pytest
```

To regenerate the documentation figures:

```bash
python scripts/generate_figures.py
```

## Authors

Aya Kamouni, Kpankpan Edouard Kambire, H'nia Harras, Anas Sabbahi. Supervised by Marcel Filoche.

## References

1. Sapoval, B., Filoche, M., & Weibel, E. R. (2002). Smaller is better—but not too small: A physical scale for the design of the mammalian pulmonary acinus. *PNAS*, 99(16), 10411-10416.
2. Felici, M., Filoche, M., & Sapoval, B. (2003). Diffusional screening in the human pulmonary acinus. *Journal of Applied Physiology*, 94(5), 2010-2016.
