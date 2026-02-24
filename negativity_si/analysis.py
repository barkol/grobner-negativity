"""
Analysis functions for negativity estimation from moments.

Implements Newton-Girard identities to reconstruct eigenvalues from power-sum moments,
and computes negativity from the reconstructed spectrum.

Key formulas:
    e₁ = μ₁ = 1 (trace normalization)
    e₂ = (μ₁² - μ₂) / 2
    e₃ = (μ₁³ - 3μ₁μ₂ + 2μ₃) / 6
    e₄ = (μ₁⁴ - 6μ₁²μ₂ + 8μ₁μ₃ + 3μ₂² - 6μ₄) / 24

where eₖ are elementary symmetric polynomials of the eigenvalues.

References:
    - Newton-Girard identities: Cox, Little, O'Shea, "Ideals, Varieties, and Algorithms"
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .config import EIGENVALUE_TOLERANCE, MOMENT_TOLERANCE


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
    Reconstruct partial transpose eigenvalues from moments.
    
    Solves the characteristic polynomial:
        λ⁴ - e₁λ³ + e₂λ² - e₃λ + e₄ = 0
    
    where eₖ are computed from moments via Newton-Girard.
    
    Args:
        mu_2: Second moment
        mu_3: Third moment
        mu_4: Fourth moment
        mu_1: First moment (default 1.0)
        
    Returns:
        Array of 4 eigenvalues (real parts only, since ρ^{T_A} is Hermitian)
    """
    e1, e2, e3, e4 = newton_girard_elementary(mu_2, mu_3, mu_4, mu_1)
    
    # Characteristic polynomial: λ⁴ - e₁λ³ + e₂λ² - e₃λ + e₄ = 0
    coefficients = [1, -e1, e2, -e3, e4]
    
    # Find roots
    eigenvalues = np.roots(coefficients)
    
    # Take real parts (eigenvalues of Hermitian matrix are real)
    # Small imaginary parts are numerical artifacts
    eigenvalues = np.real(eigenvalues)
    
    return np.sort(eigenvalues)


def compute_negativity_newton_girard(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    mu_1: float = 1.0,
) -> float:
    """
    Compute negativity from moments using Newton-Girard reconstruction.
    
    Negativity N(ρ) = Σᵢ max(0, -λᵢ) where λᵢ are eigenvalues of ρ^{T_A}.
    Equivalently, N = (||ρ^{T_A}||₁ - 1) / 2.
    
    Args:
        mu_2: Second moment
        mu_3: Third moment
        mu_4: Fourth moment
        mu_1: First moment (default 1.0)
        
    Returns:
        Negativity value ≥ 0
    """
    eigenvalues = reconstruct_eigenvalues(mu_2, mu_3, mu_4, mu_1)
    
    # Negativity = sum of absolute values of negative eigenvalues
    negativity = -np.sum(eigenvalues[eigenvalues < -EIGENVALUE_TOLERANCE])
    
    return max(0.0, negativity)


def compute_negativity_from_moments(
    moments: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute negativity and related quantities from a dictionary of moments.
    
    Args:
        moments: Dictionary with keys "mu_2", "mu_3", "mu_4"
                 and optionally "mu_2_std", "mu_3_std", "mu_4_std"
        
    Returns:
        Dictionary with:
            - negativity: Computed negativity
            - eigenvalues: Reconstructed eigenvalues
            - elementary_symmetric: (e₁, e₂, e₃, e₄)
    """
    mu_2 = moments["mu_2"]
    mu_3 = moments["mu_3"]
    mu_4 = moments["mu_4"]
    
    eigenvalues = reconstruct_eigenvalues(mu_2, mu_3, mu_4)
    negativity = compute_negativity_newton_girard(mu_2, mu_3, mu_4)
    e1, e2, e3, e4 = newton_girard_elementary(mu_2, mu_3, mu_4)
    
    return {
        "negativity": negativity,
        "eigenvalues": eigenvalues.tolist(),
        "elementary_symmetric": (e1, e2, e3, e4),
    }


def theoretical_moments(theta: float) -> Tuple[float, float, float]:
    """
    Compute theoretical moments for parameterized state |ψ(θ)⟩.
    
    State: |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
    PT eigenvalues: [-sin(θ/2)cos(θ/2), sin(θ/2)cos(θ/2), cos²(θ/2), sin²(θ/2)]
    
    For ALL pure states: μ₂ = 1 (since Σλᵢ² = (Σλᵢ)² when properly normalized)
    
    Args:
        theta: Rotation angle
        
    Returns:
        Tuple of (μ₂, μ₃, μ₄)
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # PT eigenvalues
    eigs = np.array([-s * c, s * c, c**2, s**2])
    
    # μ₂ = 1 for ALL pure states (mathematical identity)
    # Proof: μ₂ = 2s²c² + c⁴ + s⁴ = (s² + c²)² = 1
    mu_2 = 1.0
    
    return (
        mu_2,
        np.sum(eigs**3),
        np.sum(eigs**4),
    )


def theoretical_negativity(theta: float) -> float:
    """
    Compute theoretical negativity for parameterized state |ψ(θ)⟩.

    State: |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>
    PT eigenvalues: [-cs, cs, c^2, s^2] where c=cos(theta/2), s=sin(theta/2)

    Negativity = sum of absolute values of negative eigenvalues
    N(theta) = |−sin(theta/2)cos(theta/2)| = sin(theta)/2

    Args:
        theta: Rotation angle

    Returns:
        Negativity value (sum of |negative eigenvalues|)
    """
    return np.abs(np.sin(theta / 2) * np.cos(theta / 2))


def check_grobner_conditions(
    mu_2: float,
    mu_3: float,
    tolerance: float = 1e-6,
) -> Dict[str, bool]:
    """
    Check Gröbner basis degeneracy conditions.
    
    G₁ = 6μ₂ - 8μ₃ - 1 = 0 (two-pair: α,α,β,β)
    G₂ = 16μ₂³ - 39μ₂² + 72μ₂μ₃ + 12μ₂ - 48μ₃² - 12μ₃ - 1 = 0 (triple: α,α,α,β)
    
    Args:
        mu_2: Second moment
        mu_3: Third moment
        tolerance: Numerical tolerance for checking conditions
        
    Returns:
        Dictionary with condition values and whether they are satisfied
    """
    G1 = 6 * mu_2 - 8 * mu_3 - 1
    G2 = (16 * mu_2**3 - 39 * mu_2**2 + 72 * mu_2 * mu_3 
          + 12 * mu_2 - 48 * mu_3**2 - 12 * mu_3 - 1)
    
    return {
        "G1_value": G1,
        "G1_satisfied": abs(G1) < tolerance,
        "G2_value": G2,
        "G2_satisfied": abs(G2) < tolerance,
    }


def compute_discriminant(
    mu_2: float,
    mu_3: float,
    mu_4: float,
) -> float:
    """
    Compute discriminant of characteristic polynomial.
    
    The discriminant Δ determines eigenvalue degeneracy:
    - Δ > 0: All eigenvalues distinct
    - Δ = 0: At least one repeated eigenvalue
    - Δ < 0: Complex eigenvalues (shouldn't happen for Hermitian)
    
    Args:
        mu_2: Second moment
        mu_3: Third moment
        mu_4: Fourth moment
        
    Returns:
        Discriminant value
    """
    e1, e2, e3, e4 = newton_girard_elementary(mu_2, mu_3, mu_4)
    
    # Discriminant of quartic x⁴ + px² + qx + r (after depressing)
    # Using the standard formula for quartic discriminant
    # Δ = 256e₄³ - 128e₂²e₄² + 144e₂e₃²e₄ - 27e₃⁴ + 16e₂⁴e₄ 
    #     - 4e₂³e₃² - 192e₁e₃e₄² + 144e₁e₂e₃e₄ - 27e₁²e₄² 
    #     - 4e₁e₂²e₃² + 18e₁²e₂e₃e₄ - 4e₁³e₃² + e₁²e₂²e₃² - 4e₁⁴e₄
    
    # Simplified for monic quartic with e₁ = 1
    discriminant = (
        256 * e4**3 
        - 128 * e2**2 * e4**2 
        + 144 * e2 * e3**2 * e4 
        - 27 * e3**4 
        + 16 * e2**4 * e4 
        - 4 * e2**3 * e3**2
        - 192 * e3 * e4**2 
        + 144 * e2 * e3 * e4 
        - 27 * e4**2
        - 4 * e2**2 * e3**2 
        + 18 * e2 * e3 * e4 
        - 4 * e3**2 
        + e2**2 * e3**2 
        - 4 * e4
    )
    
    return discriminant


def classify_state(
    mu_2: float,
    mu_3: float,
    mu_4: float,
    tolerance: float = 1e-6,
) -> str:
    """
    Classify state based on eigenvalue degeneracy pattern.
    
    Decision tree branches:
    1. If G₁ = 0: Two-pair degeneracy (α,α,β,β)
    2. If G₂ = 0: Triple degeneracy (α,α,α,β)
    3. Else: Generic (all distinct or simple pair)
    
    Args:
        mu_2: Second moment
        mu_3: Third moment
        mu_4: Fourth moment
        tolerance: Numerical tolerance
        
    Returns:
        Classification string: "two_pair", "triple", "generic"
    """
    grobner = check_grobner_conditions(mu_2, mu_3, tolerance)
    
    if grobner["G1_satisfied"]:
        return "two_pair"
    elif grobner["G2_satisfied"]:
        return "triple"
    else:
        return "generic"


def analyze_results(
    measured_moments: Dict[str, float],
    theoretical_values: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Comprehensive analysis of measured moments.
    
    Args:
        measured_moments: Dictionary with measured μ₂, μ₃, μ₄
        theoretical_values: Optional dictionary with theoretical values for comparison
        
    Returns:
        Dictionary with analysis results
    """
    mu_2 = measured_moments["mu_2"]
    mu_3 = measured_moments["mu_3"]
    mu_4 = measured_moments["mu_4"]
    
    # Reconstruct eigenvalues and compute negativity
    eigenvalues = reconstruct_eigenvalues(mu_2, mu_3, mu_4)
    negativity = compute_negativity_newton_girard(mu_2, mu_3, mu_4)
    
    # Classification
    state_class = classify_state(mu_2, mu_3, mu_4)
    grobner = check_grobner_conditions(mu_2, mu_3)
    
    result = {
        "measured": {
            "mu_2": mu_2,
            "mu_3": mu_3,
            "mu_4": mu_4,
        },
        "reconstructed": {
            "eigenvalues": eigenvalues.tolist(),
            "negativity": negativity,
            "trace_norm": np.sum(np.abs(eigenvalues)),
        },
        "classification": {
            "state_class": state_class,
            "G1": grobner["G1_value"],
            "G2": grobner["G2_value"],
        },
        "is_entangled": negativity > EIGENVALUE_TOLERANCE,
    }
    
    # Add comparison with theory if provided
    if theoretical_values is not None:
        result["theoretical"] = theoretical_values
        result["errors"] = {
            "negativity_error": abs(negativity - theoretical_values.get("negativity", 0)),
            "mu_2_error": abs(mu_2 - theoretical_values.get("mu_2", 0)),
            "mu_3_error": abs(mu_3 - theoretical_values.get("mu_3", 0)),
            "mu_4_error": abs(mu_4 - theoretical_values.get("mu_4", 0)),
        }
    
    return result
