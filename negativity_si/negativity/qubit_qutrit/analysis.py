"""
Analysis functions for qubit-qutrit (2×3) negativity estimation.

For 2×3 systems:
- 6 eigenvalues of ρ^{T_A}
- Need moments μ₂, μ₃, μ₄, μ₅, μ₆ for full reconstruction
- Extended Newton-Girard identities

Newton-Girard for 6 eigenvalues:
    e₁ = μ₁ = 1
    e₂ = (μ₁² - μ₂) / 2
    e₃ = (μ₁³ - 3μ₁μ₂ + 2μ₃) / 6
    e₄ = (μ₁⁴ - 6μ₁²μ₂ + 8μ₁μ₃ + 3μ₂² - 6μ₄) / 24
    e₅ = (μ₁⁵ - 10μ₁³μ₂ + 20μ₁²μ₃ + 15μ₁μ₂² - 30μ₁μ₄ - 20μ₂μ₃ + 24μ₅) / 120
    e₆ = ... (6th elementary symmetric polynomial)
"""

import numpy as np
from typing import Tuple, Dict, Optional

EIGENVALUE_TOLERANCE = 1e-10


def newton_girard_elementary_6(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_5: float,
    mu_6: float,
    mu_1: float = 1.0,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute elementary symmetric polynomials from power-sum moments for 6 eigenvalues.

    Uses Newton-Girard identities recursively:
        k * eₖ = Σᵢ₌₁ᵏ (-1)^(i-1) * eₖ₋ᵢ * μᵢ

    Args:
        mu_2 through mu_6: Power-sum moments
        mu_1: First moment (trace = 1)

    Returns:
        Tuple of elementary symmetric polynomials (e₁, e₂, e₃, e₄, e₅, e₆)
    """
    e1 = mu_1
    e2 = (e1 * mu_1 - mu_2) / 2
    e3 = (e2 * mu_1 - e1 * mu_2 + mu_3) / 3
    e4 = (e3 * mu_1 - e2 * mu_2 + e1 * mu_3 - mu_4) / 4
    e5 = (e4 * mu_1 - e3 * mu_2 + e2 * mu_3 - e1 * mu_4 + mu_5) / 5
    e6 = (e5 * mu_1 - e4 * mu_2 + e3 * mu_3 - e2 * mu_4 + e1 * mu_5 - mu_6) / 6

    return e1, e2, e3, e4, e5, e6


def reconstruct_eigenvalues_6(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_5: float,
    mu_6: float,
    mu_1: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct 6 partial transpose eigenvalues from moments.

    Solves the characteristic polynomial:
        λ⁶ - e₁λ⁵ + e₂λ⁴ - e₃λ³ + e₄λ² - e₅λ + e₆ = 0

    Args:
        mu_2 through mu_6: Power-sum moments
        mu_1: First moment

    Returns:
        Array of 6 eigenvalues (sorted)
    """
    e1, e2, e3, e4, e5, e6 = newton_girard_elementary_6(mu_2, mu_3, mu_4, mu_5, mu_6, mu_1)

    # Characteristic polynomial coefficients
    coefficients = [1, -e1, e2, -e3, e4, -e5, e6]

    # Find roots
    eigenvalues = np.roots(coefficients)

    # Take real parts (ρ^{T_A} is Hermitian)
    eigenvalues = np.real(eigenvalues)

    return np.sort(eigenvalues)


def pt_eigenvalues_pure(theta_rad: float) -> np.ndarray:
    """
    Compute PT eigenvalues for pure parameterized qubit-qutrit state.

    State: |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
    PT eigenvalues: [c², s², -cs, cs, 0, 0] where c=cos(θ/2), s=sin(θ/2)

    Args:
        theta_rad: Rotation angle in radians

    Returns:
        Array of 6 PT eigenvalues (sorted)
    """
    c = np.cos(theta_rad / 2)
    s = np.sin(theta_rad / 2)
    eigs = np.array([c**2, s**2, -c*s, c*s, 0, 0])
    return np.sort(eigs)


def pt_eigenvalues_mixed(theta_rad: float, p: float) -> np.ndarray:
    """
    Compute PT eigenvalues for depolarized qubit-qutrit state.

    Model: ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/6

    For depolarizing noise, PT eigenvalues transform as:
    λ_mixed = (1-p)λ_pure + p/6

    Args:
        theta_rad: Rotation angle in radians
        p: Depolarization parameter (0 ≤ p ≤ 1)

    Returns:
        Array of 6 PT eigenvalues (sorted)
    """
    eigs_pure = pt_eigenvalues_pure(theta_rad)
    eigs_mixed = (1 - p) * eigs_pure + p / 6
    return np.sort(eigs_mixed)


def moments_from_eigenvalues(eigs: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Compute moments μ₂-μ₆ from PT eigenvalues.

    Args:
        eigs: Array of 6 PT eigenvalues

    Returns:
        Tuple of (μ₂, μ₃, μ₄, μ₅, μ₆)
    """
    return (
        np.sum(eigs**2),
        np.sum(eigs**3),
        np.sum(eigs**4),
        np.sum(eigs**5),
        np.sum(eigs**6),
    )


def negativity_from_eigenvalues(eigs: np.ndarray) -> float:
    """
    Compute negativity from PT eigenvalues.

    N(ρ) = Σᵢ max(0, -λᵢ)

    Args:
        eigs: Array of PT eigenvalues

    Returns:
        Negativity value ≥ 0
    """
    return -np.sum(eigs[eigs < -EIGENVALUE_TOLERANCE])


def compute_negativity_qubit_qutrit(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_5: float,
    mu_6: float,
    mu_1: float = 1.0,
) -> float:
    """
    Compute negativity from moments using Newton-Girard reconstruction.

    Negativity N(ρ) = Σᵢ max(0, -λᵢ)

    Args:
        mu_2 through mu_6: Power-sum moments
        mu_1: First moment

    Returns:
        Negativity value ≥ 0
    """
    eigenvalues = reconstruct_eigenvalues_6(mu_2, mu_3, mu_4, mu_5, mu_6, mu_1)
    return negativity_from_eigenvalues(eigenvalues)


def compute_negativity_from_moments(moments: Dict[str, float]) -> Dict[str, float]:
    """
    Compute negativity and related quantities from moments dictionary.

    Args:
        moments: Dictionary with keys "mu_2" through "mu_6"

    Returns:
        Dictionary with negativity, eigenvalues, elementary symmetric polynomials
    """
    mu_2 = moments["mu_2"]
    mu_3 = moments["mu_3"]
    mu_4 = moments["mu_4"]
    mu_5 = moments["mu_5"]
    mu_6 = moments["mu_6"]

    eigenvalues = reconstruct_eigenvalues_6(mu_2, mu_3, mu_4, mu_5, mu_6)
    negativity = compute_negativity_qubit_qutrit(mu_2, mu_3, mu_4, mu_5, mu_6)
    e1, e2, e3, e4, e5, e6 = newton_girard_elementary_6(mu_2, mu_3, mu_4, mu_5, mu_6)

    return {
        "negativity": negativity,
        "eigenvalues": eigenvalues.tolist(),
        "elementary_symmetric": (e1, e2, e3, e4, e5, e6),
    }


def theoretical_moments(theta_deg: float) -> Tuple[float, float, float, float, float]:
    """
    Compute theoretical moments for parameterized qubit-qutrit state.

    State: |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
    PT eigenvalues: [c², s², -cs, cs, 0, 0] where c=cos(θ/2), s=sin(θ/2)

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        Tuple of (μ₂, μ₃, μ₄, μ₅, μ₆)
    """
    theta = np.radians(theta_deg)
    eigs = pt_eigenvalues_pure(theta)
    return moments_from_eigenvalues(eigs)


def theoretical_negativity(theta_deg: float) -> float:
    """
    Compute theoretical negativity for parameterized qubit-qutrit state.

    N(θ) = cos(θ/2)sin(θ/2) = sin(θ)/2

    Args:
        theta_deg: Rotation angle in degrees

    Returns:
        Negativity value
    """
    theta = np.radians(theta_deg)
    return np.sin(theta) / 2
