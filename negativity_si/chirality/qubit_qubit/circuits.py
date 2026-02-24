"""
Quantum circuits for qubit-qubit (2×2) chirality witness measurements.

KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states
This means we only need to run the I₂ circuit once.

Circuit structure:
- I₂ circuit: 5 qubits (2 copies × 2 + 1 ancilla)
- M₂ circuits: 9 qubits (4 copies × 2 + 1 ancilla), 4 terms (SS, SY, YS, YY)

M₂ = (1/4)(SS - SY - YS + YY)
"""

from typing import Optional, Dict, List
import numpy as np
from numpy import kron
from qiskit import QuantumCircuit

from .states import create_state_vector


def create_I2_circuit(state_vector: np.ndarray) -> QuantumCircuit:
    """
    Create I₂ = Tr[ρ²] measurement circuit (5 qubits).

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
    This circuit measures the purity, which equals:
    - I₂ = Tr[ρ²] (purity)
    - R₂ = Tr[(ρ^R)²] (realignment purity)
    - μ₂ = Tr[(ρ^{T_A})²] (partial transpose second moment)

    Result: I₂ = 2*P(0) - 1

    Args:
        state_vector: 4-dimensional state vector

    Returns:
        QuantumCircuit for I₂ measurement
    """
    psi_psi = kron(state_vector, state_vector)

    qc = QuantumCircuit(5, 1, name="I2")
    qc.initialize(psi_psi, range(4))
    qc.barrier()
    qc.h(4)
    qc.cswap(4, 0, 2)  # A1 <-> A2
    qc.cswap(4, 1, 3)  # B1 <-> B2
    qc.h(4)
    qc.measure(4, 0)

    return qc


def create_M2_circuit_SS(state_vector: np.ndarray) -> QuantumCircuit:
    """Create M₂^SS circuit (CSWAP on both cross-subsystem pairs)."""
    psi4 = kron(kron(kron(state_vector, state_vector), state_vector), state_vector)

    qc = QuantumCircuit(9, 1, name="M2_SS")
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cswap(8, 1, 2)  # B1 <-> A2
    qc.cswap(8, 5, 6)  # B3 <-> A4
    # Cyclic permutation
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)

    return qc


def create_M2_circuit_SY(state_vector: np.ndarray) -> QuantumCircuit:
    """Create M₂^SY circuit (CSWAP on first, CY on second)."""
    psi4 = kron(kron(kron(state_vector, state_vector), state_vector), state_vector)

    qc = QuantumCircuit(9, 1, name="M2_SY")
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cswap(8, 1, 2)
    qc.cy(8, 5)
    qc.cy(8, 6)
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)

    return qc


def create_M2_circuit_YS(state_vector: np.ndarray) -> QuantumCircuit:
    """Create M₂^YS circuit (CY on first, CSWAP on second)."""
    psi4 = kron(kron(kron(state_vector, state_vector), state_vector), state_vector)

    qc = QuantumCircuit(9, 1, name="M2_YS")
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cy(8, 1)
    qc.cy(8, 2)
    qc.cswap(8, 5, 6)
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)

    return qc


def create_M2_circuit_YY(state_vector: np.ndarray) -> QuantumCircuit:
    """Create M₂^YY circuit (CY on both)."""
    psi4 = kron(kron(kron(state_vector, state_vector), state_vector), state_vector)

    qc = QuantumCircuit(9, 1, name="M2_YY")
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cy(8, 1)
    qc.cy(8, 2)
    qc.cy(8, 5)
    qc.cy(8, 6)
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)

    return qc


def create_M2_circuits(state_vector: np.ndarray) -> Dict[str, QuantumCircuit]:
    """
    Create all 4 M₂ circuit variants.

    M₂ = (1/4)(SS - SY - YS + YY)

    Args:
        state_vector: 4-dimensional state vector

    Returns:
        Dictionary with keys 'SS', 'SY', 'YS', 'YY'
    """
    return {
        'SS': create_M2_circuit_SS(state_vector),
        'SY': create_M2_circuit_SY(state_vector),
        'YS': create_M2_circuit_YS(state_vector),
        'YY': create_M2_circuit_YY(state_vector),
    }


# Circuit resource requirements
CIRCUIT_QUBITS = {
    "I2": 5,   # 2 copies × 2 + 1 ancilla
    "M2_SS": 9,  # 4 copies × 2 + 1 ancilla
    "M2_SY": 9,
    "M2_YS": 9,
    "M2_YY": 9,
}
