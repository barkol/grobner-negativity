"""
Analysis functions for qubit-qutrit (2×3) chirality witness.

Chirality witness: Q = I₂² - M₂
KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

For 2×3 systems:
- Depolarization model: ρ = (1-p)|ψ⟩⟨ψ| + p·I/6
- I₂(I/6) = 1/6 instead of 1/4 for 2×2
"""

import numpy as np
from numpy import cos, sin, sqrt


def I2_model_depolarized(p: float) -> float:
    """
    I₂ for depolarized qubit-qutrit state.

    Model: ρ = (1-p)|ψ⟩⟨ψ| + p·I/6

    For any pure state |ψ⟩, I₂(pure) = 1.
    For I/6, I₂(I/6) = Tr[(I/6)²] = 1/6.

    Args:
        p: Depolarization parameter (0 ≤ p ≤ 1)

    Returns:
        I₂ value
    """
    return (1 - p)**2 + p*(1-p)/3 + p**2/6


def M2_model_depolarized(theta_rad: float, p: float) -> float:
    """
    M₂ for depolarized parametrized qubit-qutrit state.

    Model: ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/6

    For pure state: M₂_pure = (c⁴ + s⁴)² where c=cos(θ/2), s=sin(θ/2)
    For I/6: M₂_mixed = 1/36

    Args:
        theta_rad: Rotation angle in radians
        p: Depolarization parameter (0 ≤ p ≤ 1)

    Returns:
        M₂ value
    """
    c = cos(theta_rad/2)
    s = sin(theta_rad/2)
    M2_pure = (c**4 + s**4)**2
    M2_mixed = 1/36
    return (1 - p)**2 * M2_pure + 2*p*(1-p) * sqrt(M2_pure * M2_mixed) + p**2 * M2_mixed


def Q_model_depolarized(theta_rad: float, p: float) -> float:
    """
    Chirality witness Q = I₂² - M₂ for depolarized qubit-qutrit state.

    Args:
        theta_rad: Rotation angle in radians
        p: Depolarization parameter

    Returns:
        Q value
    """
    I2 = I2_model_depolarized(p)
    M2 = M2_model_depolarized(theta_rad, p)
    return I2**2 - M2


def compute_theoretical_values(theta_deg: float) -> dict:
    """
    Compute theoretical I₂, M₂, Q, and N for parametrized qubit-qutrit state.

    State: |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        Dictionary with I₂, M₂, Q, N values
    """
    theta = np.radians(theta_deg)
    c = cos(theta/2)
    s = sin(theta/2)

    # For pure states, purity is always 1
    I2 = 1.0

    # M₂ = (c⁴ + s⁴)² (same formula as 2×2)
    M2 = (c**4 + s**4)**2

    # Chirality witness
    Q = I2**2 - M2

    # Negativity: N = |sin(θ)|/2
    N = abs(sin(theta)) / 2

    return {
        'I2': I2,
        'M2': M2,
        'Q': Q,
        'N': N,
    }
