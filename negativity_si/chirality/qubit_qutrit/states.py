"""
State preparation for qubit-qutrit (2×3) chirality witness experiments.

Qutrit encoding:
    |0⟩_qutrit → |00⟩
    |1⟩_qutrit → |01⟩
    |2⟩_qutrit → |10⟩
    (|11⟩ is invalid and must be avoided)

Parametrized state: |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
"""

import numpy as np
from numpy import cos, sin
from typing import Dict, Optional, Callable, List
from qiskit import QuantumCircuit


# Qubits per subsystem
QUBITS_PER_QUBIT = 1   # A subsystem (qubit)
QUBITS_PER_QUTRIT = 2  # B subsystem (qutrit encoded in 2 qubits)
QUBITS_PER_COPY = QUBITS_PER_QUBIT + QUBITS_PER_QUTRIT  # = 3


def create_state_preparation(
    state_type: str,
    theta: Optional[float] = None,
) -> Callable[[QuantumCircuit, List[int]], None]:
    """
    Create state preparation function for qubit-qutrit states.

    Args:
        state_type: Type of state to prepare:
            - "param_theta": |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
            - "maximally_entangled": (|0,0⟩ + |1,1⟩)/√2
            - "product_00", "product_01", etc.
        theta: Rotation angle for parameterized states

    Returns:
        Function that applies state preparation to circuit
    """

    def prepare_param_theta(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩"""
        if theta is None:
            raise ValueError("theta required for param_theta state")

        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]      # Qubit A
            b1 = qubits[base + 2] # Qutrit B, second qubit

            qc.ry(theta, a)
            qc.cx(a, b1)  # |1⟩_A → |01⟩_B (qutrit |1⟩)

    def prepare_maximally_entangled(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare (|0,0⟩ + |1,1⟩)/√2"""
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]
            b1 = qubits[base + 2]

            qc.h(a)
            qc.cx(a, b1)

    def prepare_product(qc: QuantumCircuit, qubits: List[int], a_val: int, b_val: int) -> None:
        """Prepare product state |a_val⟩|b_val⟩"""
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]
            b0 = qubits[base + 1]
            b1 = qubits[base + 2]

            if a_val == 1:
                qc.x(a)

            # Qutrit encoding: 0→00, 1→01, 2→10
            if b_val == 1:
                qc.x(b1)
            elif b_val == 2:
                qc.x(b0)

    if state_type == "param_theta":
        return prepare_param_theta
    elif state_type == "maximally_entangled":
        return prepare_maximally_entangled
    elif state_type.startswith("product_"):
        a_val = int(state_type[8])
        b_val = int(state_type[9])
        return lambda qc, qubits: prepare_product(qc, qubits, a_val, b_val)
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def get_theoretical_values(
    state_type: str,
    theta: Optional[float] = None,
) -> Dict[str, float]:
    """
    Get theoretical values for qubit-qutrit states.

    Args:
        state_type: Type of state
        theta: Rotation angle for parameterized states

    Returns:
        Dictionary with theoretical values
    """
    if state_type in ["param_theta", "maximally_entangled"]:
        if state_type == "maximally_entangled":
            theta = np.pi / 2
        elif theta is None:
            raise ValueError("theta required")

        c = cos(theta / 2)
        s = sin(theta / 2)

        I2 = 1.0
        M2 = (c**4 + s**4)**2
        Q = I2**2 - M2
        N = abs(sin(theta)) / 2

        return {
            'theta_deg': np.degrees(theta),
            'theta_rad': theta,
            'I2': I2,
            'M2': M2,
            'Q': Q,
            'N': N,
        }

    elif state_type.startswith("product_"):
        return {
            'theta_deg': 0.0,
            'theta_rad': 0.0,
            'I2': 1.0,
            'M2': 1.0,
            'Q': 0.0,
            'N': 0.0,
        }

    else:
        raise ValueError(f"Unknown state type: {state_type}")
