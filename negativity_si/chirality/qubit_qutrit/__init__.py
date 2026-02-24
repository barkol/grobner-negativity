"""
Chirality witness estimation for qubit-qutrit (2×3) systems.

Chirality witness: Q = I₂² - M₂
KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

For 2×3 systems:
- Qutrit encoded in 2 qubits
- I₂ circuit: 7 qubits (2 copies × 3 + 1 ancilla)
- M₂ circuits: More complex due to qutrit encoding
"""

from .analysis import (
    I2_model_depolarized,
    M2_model_depolarized,
    Q_model_depolarized,
    compute_theoretical_values,
)

from .maxlik import (
    ChiralityMaxLikEstimator,
    StateData,
    run_chirality_maxlik,
)

from .circuits import (
    create_I2_circuit,
    CIRCUIT_QUBITS,
)

from .states import (
    create_state_preparation,
    get_theoretical_values,
    QUBITS_PER_COPY,
)

__all__ = [
    # Analysis
    "I2_model_depolarized",
    "M2_model_depolarized",
    "Q_model_depolarized",
    "compute_theoretical_values",
    # MaxLik
    "ChiralityMaxLikEstimator",
    "StateData",
    "run_chirality_maxlik",
    # Circuits
    "create_I2_circuit",
    "CIRCUIT_QUBITS",
    # States
    "create_state_preparation",
    "get_theoretical_values",
    "QUBITS_PER_COPY",
]
