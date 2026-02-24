"""
Quantum circuits for qubit-qutrit (2×3) chirality witness measurements.

KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

For 2×3 systems:
- Qutrit encoded in 2 qubits
- I₂ circuit: 7 qubits (2 copies × 3 + 1 ancilla)
- M₂ circuits: Larger due to qutrit encoding
"""

from typing import Optional, List
import numpy as np
from qiskit import QuantumCircuit

from .states import create_state_preparation, QUBITS_PER_COPY


def create_I2_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create I₂ = Tr[ρ²] measurement circuit for qubit-qutrit (7 qubits).

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.

    Layout:
        q0: A₁ (qubit)
        q1, q2: B₁ (qutrit)
        q3: A₂ (qubit)
        q4, q5: B₂ (qutrit)
        q6: Ancilla

    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states

    Returns:
        QuantumCircuit for I₂ measurement
    """
    n_copies = 2
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"I2_23_{state_type}")

    # Prepare state copies
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_state_preparation(state_type, theta)
    prepare(qc, state_qubits)

    ancilla = n_qubits - 1
    qc.h(ancilla)

    # Full SWAP on both subsystems (I₂ = purity)
    qc.cswap(ancilla, 0, 3)   # A₁ ↔ A₂
    qc.cswap(ancilla, 1, 4)   # B₁[0] ↔ B₂[0]
    qc.cswap(ancilla, 2, 5)   # B₁[1] ↔ B₂[1]

    qc.h(ancilla)
    qc.measure(ancilla, 0)

    return qc


# Circuit resource requirements
CIRCUIT_QUBITS = {
    "I2": 7,   # 2 copies × 3 + 1 ancilla
}
