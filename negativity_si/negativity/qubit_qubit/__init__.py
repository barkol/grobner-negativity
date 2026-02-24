"""
Negativity estimation for 2×2 (qubit-qubit) systems.

Uses partial transpose moments μ₂, μ₃, μ₄ to reconstruct
the 4 PT eigenvalues via Newton-Girard identities.

KEY IDENTITY: μ₂ = I₂ (purity) for ALL bipartite states.

Two-stage ML estimation:
- Stage 1: Calibrate f₂, f₃, f₄, p using known θ (oracle)
- Stage 2: Fit θ blindly using calibrated parameters
"""

from .analysis import (
    compute_negativity_newton_girard,
    compute_negativity_from_moments,
    theoretical_moments,
    theoretical_negativity,
    pt_eigenvalues_pure,
    pt_eigenvalues_mixed,
)
from .maxlik import (
    NegativityMaxLikEstimator,
    run_negativity_maxlik,
)

__all__ = [
    # Analysis
    "compute_negativity_newton_girard",
    "compute_negativity_from_moments",
    "theoretical_moments",
    "theoretical_negativity",
    "pt_eigenvalues_pure",
    "pt_eigenvalues_mixed",
    # MaxLik
    "NegativityMaxLikEstimator",
    "run_negativity_maxlik",
]
