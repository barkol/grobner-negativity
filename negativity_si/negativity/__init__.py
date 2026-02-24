"""
Negativity Estimation Branch
============================

Estimates entanglement negativity from partial transpose moments μₖ.

Key formula: N = Σ|λᵢ| for λᵢ < 0, where λᵢ are PT eigenvalues.

Supported systems:
- qubit_qubit (2×2): 4 PT eigenvalues, moments μ₂, μ₃, μ₄
- qubit_qutrit (2×3): 6 PT eigenvalues, moments μ₂-μ₆

KEY IDENTITY: μ₂ = I₂ for ALL bipartite states.
This means Tr[(ρ^{T_A})²] = Tr[ρ²] (purity).
"""

from . import qubit_qubit
from . import qubit_qutrit

__all__ = [
    "qubit_qubit",
    "qubit_qutrit",
]
