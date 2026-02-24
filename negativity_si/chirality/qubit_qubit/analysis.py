"""
Analysis functions for qubit-qubit (2×2) chirality witness.

Chirality witness: Q = I₂² - M₂
KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

For the parametrized state |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩:
- I₂ = 1 (pure state)
- M₂ = (c⁴ + s⁴)² where c=cos(θ/2), s=sin(θ/2)
- Q = 1 - M₂
"""

import numpy as np
from numpy import cos, sin, sqrt


def I2_model_depolarized(p: float) -> float:
    """
    I₂ for depolarized two-qubit state.

    Model: ρ = (1-p)|ψ⟩⟨ψ| + p·I/4

    For any pure state |ψ⟩, I₂(pure) = 1.
    For I/4, I₂(I/4) = Tr[(I/4)²] = 1/4.

    I₂(ρ) = (1-p)² × 1 + 2p(1-p) × ⟨ψ|I/4|ψ⟩ + p² × 1/4
          = (1-p)² + p(1-p)/2 + p²/4

    Args:
        p: Depolarization parameter (0 ≤ p ≤ 1)

    Returns:
        I₂ value
    """
    return (1 - p)**2 + p*(1-p)/2 + p**2/4


def M2_model_depolarized(theta_rad: float, p: float) -> float:
    """
    M₂ for depolarized parametrized state.

    Model: ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

    For pure state: M₂_pure = (c⁴ + s⁴)² where c=cos(θ/2), s=sin(θ/2)
    For I/4: M₂_mixed = 1/16

    For depolarized state (approximate interpolation):
    M₂ ≈ (1-p)² × M₂_pure + 2p(1-p) × √(M₂_pure × M₂_mixed) + p² × M₂_mixed

    Args:
        theta_rad: Rotation angle in radians
        p: Depolarization parameter (0 ≤ p ≤ 1)

    Returns:
        M₂ value
    """
    c = cos(theta_rad/2)
    s = sin(theta_rad/2)
    M2_pure = (c**4 + s**4)**2
    M2_mixed = 1/16
    return (1 - p)**2 * M2_pure + 2*p*(1-p) * sqrt(M2_pure * M2_mixed) + p**2 * M2_mixed


def Q_model_depolarized(theta_rad: float, p: float) -> float:
    """
    Chirality witness Q = I₂² - M₂ for depolarized state.

    Args:
        theta_rad: Rotation angle in radians
        p: Depolarization parameter

    Returns:
        Q value
    """
    I2 = I2_model_depolarized(p)
    M2 = M2_model_depolarized(theta_rad, p)
    return I2**2 - M2


def compute_M2_from_terms(terms: dict) -> float:
    """
    Compute M₂ = (1/4)(SS - SY - YS + YY).

    Args:
        terms: Dictionary with keys 'SS', 'SY', 'YS', 'YY'

    Returns:
        M₂ value
    """
    return 0.25 * (terms['SS'] - terms['SY'] - terms['YS'] + terms['YY'])


def compute_theoretical_values(theta_deg: float) -> dict:
    """
    Compute theoretical I₂, M₂, Q, and N for parametrized state.

    State: |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
    This is fundamental: Tr[(ρ^{T_A})²] = Tr[(ρ^R)²] = Tr[ρ²].

    Chirality witness: Q = I₂² - M₂ (since R₂ = I₂)
    - Q = 0 for separable states
    - Q > 0 for entangled states
    - Q_max = 0.75 for maximally entangled (Bell) states

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        Dictionary with I₂, M₂, Q, N values
    """
    theta = np.radians(theta_deg)
    c = cos(theta/2)
    s = sin(theta/2)

    # For pure states, purity is always 1
    # KEY: μ₂ = R₂ = I₂ (this is exact for all states)
    I2 = 1.0

    # M₂ = Tr[(ρ × ρ^R)²]
    # For this specific state, M₂ = (c⁴ + s⁴)²
    c2 = c**2
    s2 = s**2
    M2 = (c2**2 + s2**2)**2

    # Chirality witness: Q = I₂² - M₂ (using R₂ = I₂)
    Q = I2 * I2 - M2

    # Negativity: N = |sin(θ)|/2
    N = abs(sin(theta)) / 2

    return {
        'I2': I2,
        'M2': M2,
        'Q': Q,
        'N': N,
    }
