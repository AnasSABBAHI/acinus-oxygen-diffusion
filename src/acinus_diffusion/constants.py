"""
Physical constants and parameters for oxygen diffusion in the pulmonary acinus.
"""

import numpy as np


class PhysicalConstants:
    """Physical constants for oxygen diffusion modeling (SI units)."""

    # Oxygen diffusion coefficient in alveolar air (m²/s). Transport inside the
    # acinus happens in the gas phase, so this is the coefficient used by the
    # model (and the one the screening length below is defined with).
    D_O2_AIR = 2.0e-5

    # Oxygen diffusion coefficients in tissue and in water (m²/s), for reference
    D_O2 = 1.8e-9
    D_O2_WATER = 2.1e-9

    # Physiological concentrations (mol/m³)
    C_AIR = 8.4          # Oxygen concentration in alveolar air
    C_BLOOD = 5.1e-4     # Oxygen concentration in venous blood
    C_REST = 4.2         # Amplitude of the breathing modulation

    # Breathing parameters
    BREATHING_RATE_REST = 0.3      # Hz (18 breaths/min, normal range 12-20)
    BREATHING_RATE_EXERCISE = 0.5  # Hz (30 breaths/min)
    OMEGA_REST = 2 * np.pi * BREATHING_RATE_REST          # rad/s
    OMEGA_EXERCISE = 2 * np.pi * BREATHING_RATE_EXERCISE  # rad/s

    # Typical acinus dimensions (m)
    ACINUS_DIAMETER = 7e-3       # 7 mm
    ACINUS_LENGTH = 10e-3        # 10 mm
    MEMBRANE_THICKNESS = 0.5e-6  # 0.5 μm

    # Screening length Λ = D / W (m), W being the membrane permeability
    LAMBDA_TYPICAL = 0.28        # 28 cm (Sapoval et al., 2002)
    LAMBDA_RANGE = (0.01, 2.0)   # Range for parameter studies

    # Membrane permeability W = D_O2_AIR / Λ (m/s)
    W_MEMBRANE = D_O2_AIR / LAMBDA_TYPICAL

    # Membrane transport parameters
    BETA = 1.0  # Partition coefficient
    TAU = 1.0   # Tortuosity factor
