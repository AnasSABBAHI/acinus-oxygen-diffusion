"""
Boundary condition descriptions for the diffusion equation.

Each condition writes the first-order finite-difference equation of a
boundary point into the rows of a (dense or LIL sparse) matrix. ``indices``
are the boundary points and ``neighbor_indices`` the adjacent interior points
along the inward normal, at distance ``h``.
"""


class BoundaryCondition:
    """Base class for boundary conditions."""

    def __init__(self, value=0.0):
        self.value = value

    def apply(self, matrix, rhs, indices, neighbor_indices=None, h=None):
        raise NotImplementedError("Subclasses must implement apply method")


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition: u = value."""

    def apply(self, matrix, rhs, indices, neighbor_indices=None, h=None):
        for idx in indices:
            matrix[idx, idx] = 1
            rhs[idx] = self.value


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition: ∂u/∂n = value (outward normal)."""

    def apply(self, matrix, rhs, indices, neighbor_indices=None, h=None):
        for idx, neighbor_idx in zip(indices, neighbor_indices):
            matrix[idx, idx] = 1
            matrix[idx, neighbor_idx] = -1
            rhs[idx] = self.value * h


class RobinBC(BoundaryCondition):
    """
    Robin boundary condition: ∂u/∂n + alpha * u = value (outward normal).

    The capillary condition ∂u/∂n = -u / Λ corresponds to alpha = 1 / Λ and
    value = 0.
    """

    def __init__(self, alpha, value=0.0):
        super().__init__(value)
        self.alpha = alpha

    def apply(self, matrix, rhs, indices, neighbor_indices=None, h=None):
        for idx, neighbor_idx in zip(indices, neighbor_indices):
            matrix[idx, idx] = 1 + self.alpha * h
            matrix[idx, neighbor_idx] = -1
            rhs[idx] = self.value * h
