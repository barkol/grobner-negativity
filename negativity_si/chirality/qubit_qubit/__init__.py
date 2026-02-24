"""
Chirality witness estimation for qubit-qubit (2×2) systems.

Chirality witness: Q = I₂² - M₂
KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

Circuit depths:
- I₂: 5 qubits (2 copies × 2 + 1 ancilla)
- M₂: 9 qubits (4 copies × 2 + 1 ancilla), 4 terms (SS, SY, YS, YY)
"""

from .analysis import (
    I2_model_depolarized,
    M2_model_depolarized,
    Q_model_depolarized,
    compute_theoretical_values,
    compute_M2_from_terms,
)

from .maxlik import (
    ChiralityMaxLikEstimator,
    StateData,
    run_chirality_maxlik,
)

from .circuits import (
    create_I2_circuit,
    create_M2_circuits,
    create_M2_circuit_SS,
    create_M2_circuit_SY,
    create_M2_circuit_YS,
    create_M2_circuit_YY,
    CIRCUIT_QUBITS,
)

from .states import (
    create_state_vector,
    create_state_preparation,
    get_theoretical_values,
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
    "StateData",
    "run_chirality_maxlik",
    # Circuits
    "create_I2_circuit",
    "create_M2_circuits",
    "create_M2_circuit_SS",
    "create_M2_circuit_SY",
    "create_M2_circuit_YS",
    "create_M2_circuit_YY",
    "CIRCUIT_QUBITS",
    # States
    "create_state_vector",
    "create_state_preparation",
    "get_theoretical_values",
]
