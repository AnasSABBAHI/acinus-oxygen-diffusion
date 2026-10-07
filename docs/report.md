# Oxygen Diffusion in Pulmonary Acinus: Computational Modeling and Analysis

**Project Report**
Aya Kamouni, Kpankpan Edouard Kambire, H'nia Harras, Anas Sabbahi
*Supervised by: Marcel Filoche*

## Abstract

This project develops a computational model of oxygen diffusion in the pulmonary acinus, the functional unit of the lung where gas exchange occurs. We solve the diffusion equation with finite differences in a 2D domain, in the stationary and quasi-stationary (breathing) regimes, with boundary conditions that represent the alveolar air, the alveolar membrane and the capillaries. The solver is validated against an exact solution and used to study the screening effect and a simplified model of Chronic Obstructive Pulmonary Disease (COPD).

## 1 Introduction

### 1.1 Physiological Context

The pulmonary acinus is the gas exchange unit of mammalian lungs. It consists of respiratory bronchioles, alveolar ducts and alveoli. In the acinus, oxygen is transported from the alveolar air to the capillaries by diffusion in the gas phase and then across the alveolar membrane. Understanding this process is important for respiratory physiology and pathology.

### 1.2 Problem Statement

We model oxygen diffusion in a 2D representation of the acinus, solving the diffusion equation with mixed boundary conditions that reflect physiological constraints. The model addresses:

- Stationary diffusion under constant conditions
- Quasi-stationary diffusion with breathing dynamics
- Pathological modifications in COPD

## 2 Mathematical Model

### 2.1 Governing Equations

The general diffusion equation is:

$$
\frac{\partial C}{\partial t} = D \nabla^2 C
$$

where $C$ is the oxygen concentration (mol/m³), $D$ the diffusion coefficient of oxygen in air (m²/s) and $t$ the time (s).

### 2.2 Stationary Regime

For steady-state conditions, with $u = C - C_b$ on the square $[0, L]^2$:

$$
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
$$

### 2.3 Boundary Conditions

Dirichlet condition (alveolar air, top boundary):

$$
u(x, y = L) = C_a - C_b
$$

Robin condition (alveolar membrane and capillaries, bottom boundary):

$$
\frac{\partial u}{\partial n} = -\frac{u}{\Lambda}, \qquad \Lambda = \frac{D}{W}
$$

where $W$ is the membrane permeability and $\Lambda$ the screening length.

Neumann conditions (lateral boundaries, symmetry):

$$
\frac{\partial u}{\partial n} = 0
$$

### 2.4 Exact Solution

On the undeformed domain the problem is one-dimensional and has the exact solution

$$
C(y) = C_b + (C_a - C_b)\,\frac{y + \Lambda}{L + \Lambda}
$$

and the oxygen flux absorbed by the capillaries (per unit depth) is

$$
\Phi = \int_0^L W\,u(x, 0)\,dx = D\,(C_a - C_b)\,\frac{L}{L + \Lambda}
$$

### 2.5 Physical Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Alveolar O₂ concentration | $C_a$ | 8.4 | mol/m³ |
| Blood O₂ concentration | $C_b$ | 5.1×10⁻⁴ | mol/m³ |
| Breathing amplitude | $C_1$ | 4.2 | mol/m³ |
| Screening length | $\Lambda$ | 0.28 | m |
| O₂ diffusion coefficient in air | $D$ | 2.0×10⁻⁵ | m²/s |
| Membrane permeability | $W = D/\Lambda$ | 7.1×10⁻⁵ | m/s |
| Breathing frequency (rest) | $\omega$ | 1.88 (0.3 Hz) | rad/s |
| Domain size | $L$ | 0.01 | m |

## 3 Numerical Methods

### 3.1 Finite Difference Discretization

The domain is discretized on a regular grid of $N \times M$ points including the boundaries ($h_x = L/(N-1)$, $h_y = L/(M-1)$). The Laplace operator is discretized with central differences:

$$
\nabla^2 u_{i,j} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h_x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h_y^2}
$$

The Robin condition is discretized with a one-sided difference, $(u_{i,1} - u_{i,0})/h_y = u_{i,0}/\Lambda$, and the Neumann conditions with mirror (ghost) points.

### 3.2 Linear System Formulation

The discretized equations form a sparse linear system

$$
A \mathbf{u} = \mathbf{b}
$$

where $A$ is the coefficient matrix incorporating the boundary conditions, $\mathbf{u}$ the vector of unknown concentrations and $\mathbf{b}$ contains the boundary data. It is solved with a sparse direct solver (`scipy.sparse.linalg.spsolve`). A typical resolution is 100×100 points.

### 3.3 Validation

The exact solution of section 2.4 is linear in $y$, so the scheme reproduces it to machine precision on every grid (errors below 10⁻¹² mol/m³); the computed flux matches the exact flux with a relative error below 10⁻¹². These checks are part of the automated test suite.

## 4 Results and Analysis

### 4.1 Stationary Regime

The concentration decreases linearly from the alveolar air towards the capillaries. At the physiological screening length ($\Lambda \gg L$), the drop across a 1 cm domain is small (8.40 to 8.11 mol/m³): most of the resistance to oxygen transfer is in the membrane.

![Stationary concentration field](figures/stationary.png)

### 4.2 Screening Effect Analysis

The flux decreases monotonically with $\Lambda$, between two regimes:

- $\Lambda \ll L$, **diffusion-limited**: $\Phi \to D(C_a - C_b)$; oxygen is absorbed close to the source and the rest of the surface is screened.
- $\Lambda \gg L$, **membrane-limited**: $\Phi \approx W (C_a - C_b) L$; the whole exchange surface is used.

The crossover occurs at $\Lambda \approx L$. At $\Lambda = 0.28$ m a 1 cm domain is in the membrane-limited regime.

![Flux vs screening length](figures/flux_lambda.png)

### 4.3 Quasi-Stationary Regime

With the boundary condition $u(x, L, t) = C_a - C_b + C_1(\cos\omega t - 1)$, the field and the flux are proportional to the instantaneous boundary value. The flux oscillates at the breathing frequency, in phase with the alveolar concentration, and its cycle average does not depend on the breathing rate. The approximation requires the diffusion time $L^2/D$ (5 s for $L = 1$ cm) to be small compared with the breathing period (3.3 s at rest); it is therefore only approximate for a 1 cm domain and well justified for smaller acinar units.

![Breathing cycle](figures/breathing.gif)

### 4.4 COPD Pathology Modeling

COPD is modeled by (i) randomly placed lesions where oxygen cannot diffuse (impermeable obstacles), and (ii) a reduced membrane permeability, i.e. a larger $\Lambda$. With a destruction factor of 0.3 (about 4.5 % of the domain destroyed):

- tissue destruction alone reduces the flux by less than 1 % at the physiological $\Lambda$, but by about 9-14 % when $\Lambda$ is comparable to or smaller than $L$;
- halving the membrane permeability reduces the flux by about 50 %.

![COPD domain](figures/copd_domain.png)

## 5 Discussion

### 5.1 Physiological Relevance

The model captures the competition between diffusion in the gas phase and transfer across the membrane, summarized by the screening length. In the membrane-limited regime, structural changes in the gas phase have little effect, whereas any reduction of membrane permeability or exchange surface directly reduces the oxygen uptake.

### 5.2 Clinical Implications

The COPD simulations illustrate qualitatively how structural changes impair gas exchange efficiency, and that the impact of tissue destruction depends on the diffusion regime of the acinus. The results are not validated against clinical data.

### 5.3 Model Limitations

- 2D approximation of a 3D branched geometry
- Homogeneous tissue and membrane properties
- Quasi-stationary approximation for breathing
- Randomly placed lesions rather than patient-specific anatomy

## 6 Conclusion

We developed and validated a finite-difference model of oxygen diffusion in the pulmonary acinus. It reproduces the screening effect, the breathing modulation of the oxygen flux and a simplified COPD scenario, and provides a basis for more realistic geometries.

## 7 References

1. Sapoval, B., Filoche, M., & Weibel, E. R. (2002). Smaller is better—but not too small: A physical scale for the design of the mammalian pulmonary acinus. *Proceedings of the National Academy of Sciences*, 99(16), 10411-10416.

2. Felici, M., Filoche, M., & Sapoval, B. (2003). Diffusional screening in the human pulmonary acinus. *Journal of Applied Physiology*, 94(5), 2010-2016.

3. Weibel, E. R. (1984). *The pathway for oxygen: structure and function in the mammalian respiratory system*. Harvard University Press.
