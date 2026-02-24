"""
Chirality witness estimation for bipartite quantum systems.

Chirality witness: Q = I₂² - M₂ (= R₂² - M₂)
KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

Properties:
- Q = 0 for separable states
- Q > 0 for entangled states
- Q_max = 0.75 for maximally entangled (Bell) states

Supported systems:
- Qubit-qubit (2×2): 5 qubits for I₂, 9 qubits for M₂
- Qubit-qutrit (2×3): Extended circuits
"""

from .qubit_qubit import (
    # Analysis
    I2_model_depolarized,
    M2_model_depolarized,
    Q_model_depolarized,
    compute_theoretical_values,
    compute_M2_from_terms,
    # MaxLik
    ChiralityMaxLikEstimator,
    run_chirality_maxlik,
    # Circuits
    create_I2_circuit,
    create_M2_circuits,
    create_M2_circuit_SS,
    create_M2_circuit_SY,
    create_M2_circuit_YS,
    create_M2_circuit_YY,
    # States
    create_state_vector,
    CIRCUIT_QUBITS,
)

__all__ = [
    # Analysis
    "I2_model_depolarized",
    "M2_model_depolarized",
    "Q_model_depolarized",
    "compute_theoretical_values",
    "compute_M2_from_terms",
    # MaxLik
    "ChiralityMaxLikEstimator",
    "run_chirality_maxlik",
    # Circuits
    "create_I2_circuit",
    "create_M2_circuits",
    "create_M2_circuit_SS",
    "create_M2_circuit_SY",
    "create_M2_circuit_YS",
    "create_M2_circuit_YY",
    # States
    "create_state_vector",
    "CIRCUIT_QUBITS",
]
