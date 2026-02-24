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
from typing import Tuple, Dict, Optional, List

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
    
    # Negativity = sum of absolute values of negative eigenvalues
    negativity = -np.sum(eigenvalues[eigenvalues < -EIGENVALUE_TOLERANCE])
    
    return max(0.0, negativity)


def compute_negativity_from_moments_qubit_qutrit(
    moments: Dict[str, float],
) -> Dict[str, float]:
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


def theoretical_moments_qubit_qutrit(theta: float) -> Tuple[float, float, float, float, float]:
    """
    Compute theoretical moments for parameterized qubit-qutrit state.
    
    State: |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
    PT eigenvalues: [c², s², -cs, cs, 0, 0] where c=cos(θ/2), s=sin(θ/2)
    
    Args:
        theta: Rotation angle
        
    Returns:
        Tuple of (μ₂, μ₃, μ₄, μ₅, μ₆)
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # PT eigenvalues
    eigs = np.array([c**2, s**2, -c*s, c*s, 0, 0])
    
    return (
        np.sum(eigs**2),
        np.sum(eigs**3),
        np.sum(eigs**4),
        np.sum(eigs**5),
        np.sum(eigs**6),
    )


def theoretical_negativity_qubit_qutrit(theta: float) -> float:
    """
    Compute theoretical negativity for parameterized qubit-qutrit state.
    
    N(θ) = cos(θ/2)sin(θ/2) = sin(θ)/2
    
    Args:
        theta: Rotation angle
        
    Returns:
        Negativity value
    """
    return np.sin(theta) / 2


def analyze_results_qubit_qutrit(
    measured_moments: Dict[str, float],
    theoretical_values: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Comprehensive analysis of measured qubit-qutrit moments.
    
    Args:
        measured_moments: Dictionary with measured μ₂ through μ₆
        theoretical_values: Optional dictionary with theoretical values
        
    Returns:
        Dictionary with analysis results
    """
    mu_2 = measured_moments["mu_2"]
    mu_3 = measured_moments["mu_3"]
    mu_4 = measured_moments["mu_4"]
    mu_5 = measured_moments["mu_5"]
    mu_6 = measured_moments["mu_6"]
    
    eigenvalues = reconstruct_eigenvalues_6(mu_2, mu_3, mu_4, mu_5, mu_6)
    negativity = compute_negativity_qubit_qutrit(mu_2, mu_3, mu_4, mu_5, mu_6)
    
    result = {
        "measured": {
            "mu_2": mu_2,
            "mu_3": mu_3,
            "mu_4": mu_4,
            "mu_5": mu_5,
            "mu_6": mu_6,
        },
        "reconstructed": {
            "eigenvalues": eigenvalues.tolist(),
            "negativity": negativity,
            "trace_norm": np.sum(np.abs(eigenvalues)),
        },
        "is_entangled": negativity > EIGENVALUE_TOLERANCE,
    }
    
    if theoretical_values is not None:
        result["theoretical"] = theoretical_values
        result["errors"] = {
            "negativity_error": abs(negativity - theoretical_values.get("negativity", 0)),
        }
        for k in range(2, 7):
            key = f"mu_{k}"
            if key in theoretical_values:
                result["errors"][f"{key}_error"] = abs(
                    measured_moments[key] - theoretical_values[key]
                )
    
    return result


def validate_qubit_qutrit_formulas(n_tests: int = 100, verbose: bool = False) -> bool:
    """
    Validate Newton-Girard reconstruction for qubit-qutrit system.
    
    Returns:
        True if all tests pass
    """
    np.random.seed(42)
    max_error = 0.0
    
    for _ in range(n_tests):
        # Generate random theta
        theta = np.random.uniform(0, np.pi)
        
        # Theoretical values
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        eigs_true = np.sort([c**2, s**2, -c*s, c*s, 0, 0])
        neg_true = abs(-c*s)
        
        # Compute moments
        mu_2, mu_3, mu_4, mu_5, mu_6 = theoretical_moments_qubit_qutrit(theta)
        
        # Reconstruct
        eigs_recon = reconstruct_eigenvalues_6(mu_2, mu_3, mu_4, mu_5, mu_6)
        neg_recon = compute_negativity_qubit_qutrit(mu_2, mu_3, mu_4, mu_5, mu_6)
        
        # Check
        eig_error = np.max(np.abs(eigs_true - eigs_recon))
        neg_error = abs(neg_true - neg_recon)
        
        max_error = max(max_error, eig_error, neg_error)
        
        if verbose and (eig_error > 1e-8 or neg_error > 1e-8):
            print(f"θ={theta:.4f}: eig_err={eig_error:.2e}, neg_err={neg_error:.2e}")
    
    if verbose:
        print(f"Max error: {max_error:.2e}")
    
    return max_error < 1e-4
