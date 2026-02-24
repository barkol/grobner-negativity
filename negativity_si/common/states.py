"""
Parametrized quantum state creation for entanglement experiments.

State parametrization: |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
- θ=0°: |00⟩ (separable)
- θ=90°: (|00⟩+|11⟩)/√2 (Bell state, maximally entangled)
- θ=180°: |11⟩ (separable)
"""

import numpy as np
from numpy import cos, sin


def create_parametrized_state(theta_deg: float) -> np.ndarray:
    """
    Create parametrized entangled state vector for 2×2 system.

    |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩

    Args:
        theta_deg: Angle in degrees

    Returns:
        4-element state vector [cos(θ/2), 0, 0, sin(θ/2)]
    """
    theta = np.radians(theta_deg)
    return np.array([cos(theta / 2), 0, 0, sin(theta / 2)])


def create_state_vector(theta_deg: float) -> np.ndarray:
    """Alias for create_parametrized_state."""
    return create_parametrized_state(theta_deg)


def create_parametrized_state_2x3(theta_deg: float) -> np.ndarray:
    """
    Create parametrized entangled state vector for 2×3 (qubit-qutrit) system.

    |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|12⟩

    Basis ordering: |00⟩, |01⟩, |02⟩, |10⟩, |11⟩, |12⟩

    Args:
        theta_deg: Angle in degrees

    Returns:
        6-element state vector
    """
    theta = np.radians(theta_deg)
    psi = np.zeros(6, dtype=complex)
    psi[0] = cos(theta / 2)  # |00⟩
    psi[5] = sin(theta / 2)  # |12⟩
    return psi
