"""
Analysis functions for negativity estimation from moments (2×2 system).

Implements Newton-Girard identities to reconstruct 4 PT eigenvalues
from power-sum moments μ₂, μ₃, μ₄.

Key formulas:
    e₁ = μ₁ = 1 (trace normalization)
    e₂ = (μ₁² - μ₂) / 2
    e₃ = (μ₁³ - 3μ₁μ₂ + 2μ₃) / 6
    e₄ = (μ₁⁴ - 6μ₁²μ₂ + 8μ₁μ₃ + 3μ₂² - 6μ₄) / 24

where eₖ are elementary symmetric polynomials of the eigenvalues.
"""

import numpy as np
from typing import Tuple


def pt_eigenvalues_pure(theta: float) -> np.ndarray:
    """
    PT eigenvalues for pure state |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩.

    Args:
        theta: Entanglement parameter in radians

    Returns:
        Array of 4 PT eigenvalues: [-sc, sc, c², s²]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([-s * c, s * c, c**2, s**2])


def pt_eigenvalues_mixed(theta: float, p: float) -> np.ndarray:
    """
    PT eigenvalues for depolarized state.

    ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

    The PT of I/4 is I/4 (maximally mixed is PPT).
    λᵢ = (1-p)·λᵢ_pure + p/4

    Args:
        theta: Entanglement parameter in radians
        p: Depolarization parameter [0, 1]

    Returns:
        Array of 4 PT eigenvalues
    """
    eigs_pure = pt_eigenvalues_pure(theta)
    return (1 - p) * eigs_pure + p / 4


def moments_from_eigenvalues(eigs: np.ndarray) -> Tuple[float, float, float]:
    """Compute μ₂, μ₃, μ₄ from eigenvalues."""
    return np.sum(eigs**2), np.sum(eigs**3), np.sum(eigs**4)


def negativity_from_eigenvalues(eigs: np.ndarray) -> float:
    """Negativity = sum of |negative eigenvalues|."""
    return -np.sum(eigs[eigs < 0])


def newton_girard_elementary(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_1: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Compute elementary symmetric polynomials from power-sum moments.

    Newton-Girard identities for 4 eigenvalues:
        e₁ = μ₁
        e₂ = (μ₁² - μ₂) / 2
        e₃ = (μ₁³ - 3μ₁μ₂ + 2μ₃) / 6
        e₄ = (μ₁⁴ - 6μ₁²μ₂ + 8μ₁μ₃ + 3μ₂² - 6μ₄) / 24

    Args:
        mu_2: Second moment Tr[(ρ^{T_A})²]
        mu_3: Third moment Tr[(ρ^{T_A})³]
        mu_4: Fourth moment Tr[(ρ^{T_A})⁴]
        mu_1: First moment Tr[ρ^{T_A}] = 1 (trace normalization)

    Returns:
        Tuple of elementary symmetric polynomials (e₁, e₂, e₃, e₄)
    """
    e1 = mu_1
    e2 = (mu_1**2 - mu_2) / 2
    e3 = (mu_1**3 - 3 * mu_1 * mu_2 + 2 * mu_3) / 6
    e4 = (mu_1**4 - 6 * mu_1**2 * mu_2 + 8 * mu_1 * mu_3 + 3 * mu_2**2 - 6 * mu_4) / 24
    return e1, e2, e3, e4


def reconstruct_eigenvalues(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_1: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct PT eigenvalues from moments using Newton-Girard.

    The characteristic polynomial is:
        λ⁴ - e₁λ³ + e₂λ² - e₃λ + e₄ = 0

    Args:
        mu_2, mu_3, mu_4: Power-sum moments
        mu_1: First moment (default 1.0)

    Returns:
        Array of 4 eigenvalues (sorted)
    """
    e1, e2, e3, e4 = newton_girard_elementary(mu_2, mu_3, mu_4, mu_1)

    # Characteristic polynomial coefficients
    coeffs = [1, -e1, e2, -e3, e4]
    eigenvalues = np.roots(coeffs)

    # Return real parts, sorted
    return np.sort(np.real(eigenvalues))


def compute_negativity_newton_girard(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_1: float = 1.0,
) -> float:
    """
    Compute negativity from moments using Newton-Girard reconstruction.

    Args:
        mu_2, mu_3, mu_4: Power-sum moments
        mu_1: First moment (default 1.0)

    Returns:
        Negativity = Σ|λᵢ| for λᵢ < 0
    """
    eigenvalues = reconstruct_eigenvalues(mu_2, mu_3, mu_4, mu_1)
    return -np.sum(eigenvalues[eigenvalues < -1e-10])


def compute_negativity_from_moments(
    mu_2: float,
    mu_3: float,
    mu_4: float,
) -> float:
    """Alias for compute_negativity_newton_girard."""
    return compute_negativity_newton_girard(mu_2, mu_3, mu_4)


def theoretical_moments(theta_deg: float) -> Tuple[float, float, float]:
    """
    Compute theoretical moments for pure parametrized state.

    |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩

    Args:
        theta_deg: Angle in degrees

    Returns:
        Tuple (μ₂, μ₃, μ₄)
    """
    theta = np.radians(theta_deg)
    eigs = pt_eigenvalues_pure(theta)
    return moments_from_eigenvalues(eigs)


def theoretical_negativity(theta_deg: float) -> float:
    """
    Compute theoretical negativity for pure parametrized state.

    N = |sin(θ)|/2

    Args:
        theta_deg: Angle in degrees

    Returns:
        Negativity value
    """
    theta = np.radians(theta_deg)
    return abs(np.sin(theta)) / 2
