"""
Negativity estimation for qubit-qutrit (2×3) systems.

For 2×3 systems:
- 6 PT eigenvalues
- Moments μ₂, μ₃, μ₄, μ₅, μ₆ for full reconstruction
- Two-stage ML estimation for calibrated negativity

Key identity: μ₂ = I₂ for ALL bipartite states
"""

from .analysis import (
    newton_girard_elementary_6,
    reconstruct_eigenvalues_6,
    compute_negativity_qubit_qutrit,
    compute_negativity_from_moments,
    theoretical_moments,
    theoretical_negativity,
    pt_eigenvalues_pure,
    pt_eigenvalues_mixed,
    moments_from_eigenvalues,
    negativity_from_eigenvalues,
)

from .maxlik import (
    NegativityMaxLikEstimator,
    StateData,
    run_negativity_maxlik,
)

from .circuits import (
    create_mu2_circuit,
    create_mu3_circuit,
    create_mu4_circuit,
    create_mu5_circuit,
    create_mu6_circuit,
    create_purity_circuit,
    create_moment_circuits,
    CIRCUIT_QUBITS,
    CIRCUIT_CSWAPS,
)

from .states import (
    create_state_preparation,
    get_theoretical_values,
    get_pt_eigenvalues,
    PRODUCT_STATES,
    ENTANGLED_STATES,
    SUPPORTED_STATES,
    QUBITS_PER_COPY,
)

__all__ = [
    # Analysis
    "newton_girard_elementary_6",
    "reconstruct_eigenvalues_6",
    "compute_negativity_qubit_qutrit",
    "compute_negativity_from_moments",
    "theoretical_moments",
    "theoretical_negativity",
    "pt_eigenvalues_pure",
    "pt_eigenvalues_mixed",
    "moments_from_eigenvalues",
    "negativity_from_eigenvalues",
    # MaxLik
    "NegativityMaxLikEstimator",
    "StateData",
    "run_negativity_maxlik",
    # Circuits
    "create_mu2_circuit",
    "create_mu3_circuit",
    "create_mu4_circuit",
    "create_mu5_circuit",
    "create_mu6_circuit",
    "create_purity_circuit",
    "create_moment_circuits",
    "CIRCUIT_QUBITS",
    "CIRCUIT_CSWAPS",
    # States
    "create_state_preparation",
    "get_theoretical_values",
    "get_pt_eigenvalues",
    "PRODUCT_STATES",
    "ENTANGLED_STATES",
    "SUPPORTED_STATES",
    "QUBITS_PER_COPY",
]
