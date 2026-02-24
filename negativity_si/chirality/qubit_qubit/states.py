"""
State preparation for qubit-qubit (2×2) chirality witness experiments.

Parametrized state: |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩

Properties:
- θ = 0°: |00⟩ (separable), N=0, Q=0
- θ = 90°: Bell state, N=0.5, Q=0.75
- θ = 180°: |11⟩ (separable), N=0, Q=0
"""

import numpy as np
from numpy import cos, sin
from typing import Dict, Optional, Callable, List
from qiskit import QuantumCircuit


def create_state_vector(theta_deg: float) -> np.ndarray:
    """
    Create parametrized entangled state vector.

    |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        4-dimensional state vector
    """
    theta = np.radians(theta_deg)
    return np.array([cos(theta/2), 0, 0, sin(theta/2)])


def create_state_preparation(
    theta_deg: float,
) -> Callable[[QuantumCircuit, List[int]], None]:
    """
    Create state preparation function for circuit-based preparation.

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        Function that applies state preparation to circuit
    """
    def prepare(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare parametrized state on given qubits."""
        theta = np.radians(theta_deg)
        n_copies = len(qubits) // 2

        for i in range(n_copies):
            a = qubits[2*i]      # Qubit A
            b = qubits[2*i + 1]  # Qubit B

            # Create entangled state
            qc.ry(theta, a)
            qc.cx(a, b)

    return prepare


def get_theoretical_values(theta_deg: float) -> Dict[str, float]:
    """
    Get theoretical values for the parametrized state.

    For |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩:
    - I₂ = 1 (pure state)
    - M₂ = (c⁴ + s⁴)²
    - Q = 1 - M₂
    - N = sin(θ)/2

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        Dictionary with theoretical values
    """
    theta = np.radians(theta_deg)
    c = cos(theta/2)
    s = sin(theta/2)

    # Purity (always 1 for pure state)
    I2 = 1.0

    # M₂ for parametrized state
    M2 = (c**4 + s**4)**2

    # Chirality witness
    Q = I2**2 - M2

    # Negativity
    N = abs(sin(theta)) / 2

    return {
        'theta_deg': theta_deg,
        'theta_rad': theta,
        'I2': I2,
        'M2': M2,
        'Q': Q,
        'N': N,
    }
