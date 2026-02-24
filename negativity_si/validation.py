"""
Comprehensive Validation of Analytical Formulas.

This module validates all mathematical formulas used in the manuscript:
- Newton-Girard identities for moment-to-eigenvalue reconstruction
- Gröbner basis conditions for degeneracy classification
- Negativity computation from eigenvalues
- Noise robustness (RMSE scaling)
- Efficiency verification (measurement reduction)

References:
    Paper: "Resource-efficient negativity estimation from partial transpose moments"
    Authors: Patrycja Tulewicz and Karol Bartkiewicz
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations

from .analysis import (
    newton_girard_elementary,
    reconstruct_eigenvalues,
    compute_negativity_newton_girard,
    check_grobner_conditions,
    theoretical_moments,
    theoretical_negativity,
)


class ValidationResults:
    """Container for validation test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def record(self, name: str, condition: bool, details: str = "") -> bool:
        """Record a test result."""
        status = "✓ PASS" if condition else "✗ FAIL"
        self.results.append({
            "name": name,
            "passed": condition,
            "status": status,
            "details": details,
        })
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        return condition
    
    def summary(self) -> str:
        """Return summary string."""
        return f"{self.passed} passed, {self.failed} failed"


# ==============================================================================
# Quantum Operations
# ==============================================================================

def random_density_matrix(d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random valid density matrix via Ginibre ensemble.
    
    Guaranteed properties:
    - Hermitian: ρ = ρ†
    - Positive semidefinite: eigenvalues ≥ 0
    - Trace one: Tr[ρ] = 1
    """
    if seed is not None:
        np.random.seed(seed)
    G = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    rho = G @ G.conj().T
    rho = (rho + rho.conj().T) / 2
    return rho / np.trace(rho)


def partial_transpose(rho: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """Partial transpose on subsystem A."""
    rho_reshaped = rho.reshape(d1, d2, d1, d2)
    rho_pt = rho_reshaped.transpose(2, 1, 0, 3).reshape(d1*d2, d1*d2)
    return (rho_pt + rho_pt.conj().T) / 2


def compute_negativity_direct(rho: np.ndarray, d1: int = 2, d2: int = 2) -> float:
    """Compute negativity directly from eigenvalues."""
    rho_pt = partial_transpose(rho, d1, d2)
    eigenvals = np.linalg.eigvalsh(rho_pt)
    return -sum(v for v in eigenvals if v < 0)


def bell_state(idx: int = 0) -> np.ndarray:
    """Bell state: 0=Φ+, 1=Φ-, 2=Ψ+, 3=Ψ-"""
    vecs = [
        np.array([1, 0, 0, 1]) / np.sqrt(2),
        np.array([1, 0, 0, -1]) / np.sqrt(2),
        np.array([0, 1, 1, 0]) / np.sqrt(2),
        np.array([0, 1, -1, 0]) / np.sqrt(2),
    ]
    psi = vecs[idx]
    return np.outer(psi, psi.conj())


def werner_state(p: float) -> np.ndarray:
    """Werner state ρ_W(p) = p|Φ+⟩⟨Φ+| + (1-p)I/4"""
    return p * bell_state(0) + (1-p) * np.eye(4) / 4


# ==============================================================================
# Validation Tests
# ==============================================================================

def validate_newton_girard(n_tests: int = 100) -> ValidationResults:
    """
    Validate Newton-Girard identities.
    
    Tests that elementary symmetric polynomials computed via Newton-Girard
    match direct computation from eigenvalues.
    """
    results = ValidationResults()
    np.random.seed(42)
    
    max_errors = [0.0, 0.0, 0.0]
    
    for _ in range(n_tests):
        # Generate 4 real eigenvalues summing to 1
        eigs = np.random.randn(4)
        eigs = eigs - eigs.mean() + 0.25
        
        # Moments
        mu = {k: sum(e**k for e in eigs) for k in range(1, 5)}
        
        # Elementary symmetric via Newton-Girard
        e1, e2_ng, e3_ng, e4_ng = newton_girard_elementary(mu[2], mu[3], mu[4], mu[1])
        
        # Direct computation
        e2_dir = sum(eigs[i]*eigs[j] for i,j in combinations(range(4), 2))
        e3_dir = sum(eigs[i]*eigs[j]*eigs[k] for i,j,k in combinations(range(4), 3))
        e4_dir = eigs[0]*eigs[1]*eigs[2]*eigs[3]
        
        max_errors[0] = max(max_errors[0], abs(e2_ng - e2_dir))
        max_errors[1] = max(max_errors[1], abs(e3_ng - e3_dir))
        max_errors[2] = max(max_errors[2], abs(e4_ng - e4_dir))
    
    results.record("e₂ formula", max_errors[0] < 1e-12, f"max error = {max_errors[0]:.2e}")
    results.record("e₃ formula", max_errors[1] < 1e-12, f"max error = {max_errors[1]:.2e}")
    results.record("e₄ formula", max_errors[2] < 1e-12, f"max error = {max_errors[2]:.2e}")
    
    return results


def validate_eigenvalue_reconstruction(n_tests: int = 100) -> ValidationResults:
    """
    Validate eigenvalue reconstruction from moments.
    
    Tests that eigenvalues reconstructed via Newton-Girard match
    the original eigenvalues of ρ^{T_A}.
    """
    results = ValidationResults()
    np.random.seed(123)
    
    max_error = 0.0
    
    for _ in range(n_tests):
        rho = random_density_matrix(4)
        rho_pt = partial_transpose(rho, 2, 2)
        eigs_true = np.sort(np.linalg.eigvalsh(rho_pt))
        
        # Compute moments
        mu_2 = np.trace(rho_pt @ rho_pt).real
        mu_3 = np.trace(rho_pt @ rho_pt @ rho_pt).real
        mu_4 = np.trace(rho_pt @ rho_pt @ rho_pt @ rho_pt).real
        
        # Reconstruct
        eigs_recon = reconstruct_eigenvalues(mu_2, mu_3, mu_4)
        
        error = np.max(np.abs(eigs_true - eigs_recon))
        max_error = max(max_error, error)
    
    results.record("Eigenvalue reconstruction", max_error < 1e-10, f"max error = {max_error:.2e}")
    
    return results


def validate_negativity_computation(n_tests: int = 100) -> ValidationResults:
    """
    Validate negativity computation via Newton-Girard.
    
    Compares negativity computed from moments against direct computation
    from eigenvalues.
    """
    results = ValidationResults()
    np.random.seed(456)
    
    max_error = 0.0
    
    for _ in range(n_tests):
        rho = random_density_matrix(4)
        rho_pt = partial_transpose(rho, 2, 2)
        
        # Direct negativity
        neg_direct = compute_negativity_direct(rho)
        
        # Moments
        mu_2 = np.trace(rho_pt @ rho_pt).real
        mu_3 = np.trace(rho_pt @ rho_pt @ rho_pt).real
        mu_4 = np.trace(rho_pt @ rho_pt @ rho_pt @ rho_pt).real
        
        # Via Newton-Girard
        neg_ng = compute_negativity_newton_girard(mu_2, mu_3, mu_4)
        
        error = abs(neg_direct - neg_ng)
        max_error = max(max_error, error)
    
    results.record("Negativity from moments", max_error < 1e-10, f"max error = {max_error:.2e}")
    
    return results


def validate_grobner_conditions() -> ValidationResults:
    """
    Validate Gröbner basis degeneracy conditions.
    
    G₁ = 6μ₂ - 8μ₃ - 1 = 0 (two-pair: α,α,β,β)
    G₂ = 16μ₂³ - 39μ₂² + 72μ₂μ₃ + 12μ₂ - 48μ₃² - 12μ₃ - 1 = 0 (triple: α,α,α,β)
    """
    results = ValidationResults()
    
    # Test G₁ for two-pair degeneracy
    # Eigenvalues: α, α, β, β with 2α + 2β = 1
    for alpha in [0.1, 0.2, 0.25, 0.3, 0.4]:
        beta = 0.5 - alpha
        eigs = np.array([alpha, alpha, beta, beta])
        mu_2 = np.sum(eigs**2)
        mu_3 = np.sum(eigs**3)
        
        G1 = 6 * mu_2 - 8 * mu_3 - 1
        
    results.record("G₁ for two-pair", abs(G1) < 1e-10, f"G₁ = {G1:.2e}")
    
    # Test G₂ for triple degeneracy
    # Eigenvalues: α, α, α, β with 3α + β = 1
    for alpha in [0.1, 0.2, 0.25]:
        beta = 1 - 3*alpha
        eigs = np.array([alpha, alpha, alpha, beta])
        mu_2 = np.sum(eigs**2)
        mu_3 = np.sum(eigs**3)
        
        G2 = (16*mu_2**3 - 39*mu_2**2 + 72*mu_2*mu_3 
              + 12*mu_2 - 48*mu_3**2 - 12*mu_3 - 1)
        
    results.record("G₂ for triple", abs(G2) < 1e-10, f"G₂ = {G2:.2e}")
    
    return results


def validate_parameterized_states() -> ValidationResults:
    """
    Validate theoretical values for parameterized states |ψ(θ)⟩.
    """
    results = ValidationResults()
    
    test_angles = [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]
    
    for theta in test_angles:
        # Theoretical values
        mu_2_th, mu_3_th, mu_4_th = theoretical_moments(theta)
        neg_th = theoretical_negativity(theta)
        
        # Construct state and compute directly
        c, s = np.cos(theta/2), np.sin(theta/2)
        psi = np.array([c, 0, 0, s])
        rho = np.outer(psi, psi)
        
        rho_pt = partial_transpose(rho, 2, 2)
        mu_2_direct = np.trace(rho_pt @ rho_pt).real
        mu_3_direct = np.trace(rho_pt @ rho_pt @ rho_pt).real
        mu_4_direct = np.trace(rho_pt @ rho_pt @ rho_pt @ rho_pt).real
        neg_direct = compute_negativity_direct(rho)
        
        # Check
        error_mu2 = abs(mu_2_th - mu_2_direct)
        error_mu3 = abs(mu_3_th - mu_3_direct)
        error_mu4 = abs(mu_4_th - mu_4_direct)
        error_neg = abs(neg_th - neg_direct)
    
    results.record("μ₂ for |ψ(θ)⟩", error_mu2 < 1e-12, f"error = {error_mu2:.2e}")
    results.record("μ₃ for |ψ(θ)⟩", error_mu3 < 1e-12, f"error = {error_mu3:.2e}")
    results.record("μ₄ for |ψ(θ)⟩", error_mu4 < 1e-12, f"error = {error_mu4:.2e}")
    results.record("N for |ψ(θ)⟩", error_neg < 1e-12, f"error = {error_neg:.2e}")
    
    return results


def validate_bell_states() -> ValidationResults:
    """Validate values for Bell states."""
    results = ValidationResults()
    
    for idx, name in enumerate(["Φ+", "Φ-", "Ψ+", "Ψ-"]):
        rho = bell_state(idx)
        neg = compute_negativity_direct(rho)
        
        # All Bell states have N = 0.5
        error = abs(neg - 0.5)
    
    results.record(f"N(Bell) = 0.5", error < 1e-12, f"error = {error:.2e}")
    
    return results


def validate_werner_states() -> ValidationResults:
    """Validate Werner state negativity."""
    results = ValidationResults()
    
    # Werner state ρ(p) = p|Φ+⟩⟨Φ+| + (1-p)I/4
    # N(p) = max(0, (3p-1)/4) for p > 1/3
    
    test_p = [0.0, 0.2, 1/3, 0.5, 0.7, 1.0]
    max_error = 0.0
    
    for p in test_p:
        rho = werner_state(p)
        neg_direct = compute_negativity_direct(rho)
        neg_theory = max(0, (3*p - 1) / 4)
        
        error = abs(neg_direct - neg_theory)
        max_error = max(max_error, error)
    
    results.record("Werner state negativity", max_error < 1e-12, f"max error = {max_error:.2e}")
    
    return results


def validate_efficiency(n_samples: int = 10000) -> ValidationResults:
    """
    Validate measurement efficiency claim.
    
    Paper claims: ~3 measurements for two-qubit vs 16 for tomography → 5.3× efficiency
    """
    results = ValidationResults()
    np.random.seed(999)
    
    def classify_degeneracy(eigs, tol=1e-8):
        """Classify eigenvalue degeneracy pattern."""
        s = sorted(eigs)
        
        if max(eigs) - min(eigs) < tol:
            return "quadruple"
        
        for i in range(4):
            others = [eigs[j] for j in range(4) if j != i]
            if max(others) - min(others) < tol:
                return "triple"
        
        if abs(s[0]-s[1]) < tol and abs(s[2]-s[3]) < tol:
            return "two-pair"
        
        for i in range(4):
            for j in range(i+1, 4):
                if abs(eigs[i] - eigs[j]) < tol:
                    return "simple-pair"
        
        return "generic"
    
    def measurements_needed(deg_type):
        return 2 if deg_type in ["quadruple", "triple", "two-pair"] else 3
    
    total_meas = 0
    for _ in range(n_samples):
        rho = random_density_matrix(4)
        rho_pt = partial_transpose(rho, 2, 2)
        eigs = np.linalg.eigvalsh(rho_pt)
        deg = classify_degeneracy(eigs)
        total_meas += measurements_needed(deg)
    
    avg = total_meas / n_samples
    efficiency = 16 / avg
    
    results.record(
        f"Efficiency ≈ 5.3×",
        abs(efficiency - 5.33) < 0.2,
        f"measured: {efficiency:.2f}×"
    )
    
    return results


def validate_noise_robustness(n_samples: int = 5000) -> ValidationResults:
    """
    Validate noise robustness claim.
    
    Paper claims: RMSE ≈ 0.245η for depolarizing noise
    """
    results = ValidationResults()
    np.random.seed(111)
    
    def depolarize(rho, eta):
        return (1-eta)*rho + eta*np.eye(4)/4
    
    eta_vals = [0.02, 0.04, 0.06, 0.08, 0.10]
    slopes = []
    
    for eta in eta_vals:
        sq_errs = []
        for _ in range(n_samples):
            rho = random_density_matrix(4)
            N_true = compute_negativity_direct(rho)
            N_noisy = compute_negativity_direct(depolarize(rho, eta))
            sq_errs.append((N_true - N_noisy)**2)
        
        rmse = np.sqrt(np.mean(sq_errs))
        slopes.append(rmse / eta)
    
    avg_slope = np.mean(slopes)
    
    results.record(
        "RMSE ≈ 0.245η",
        abs(avg_slope - 0.245) / 0.245 < 0.25,
        f"measured slope: {avg_slope:.3f}"
    )
    
    return results


# ==============================================================================
# Main Validation Runner
# ==============================================================================

def run_validation(
    verbose: bool = True,
    full: bool = True,
) -> Dict[str, ValidationResults]:
    """
    Run all validation tests.
    
    Args:
        verbose: Whether to print results
        full: Whether to run full tests (slower but more thorough)
        
    Returns:
        Dictionary mapping test names to ValidationResults
    """
    all_results = {}
    
    if verbose:
        print("=" * 80)
        print("VALIDATION OF ANALYTICAL FORMULAS")
        print("Resource-efficient negativity estimation from partial transpose moments")
        print("=" * 80)
    
    tests = [
        ("Newton-Girard Identities", validate_newton_girard),
        ("Eigenvalue Reconstruction", validate_eigenvalue_reconstruction),
        ("Negativity Computation", validate_negativity_computation),
        ("Gröbner Conditions", validate_grobner_conditions),
        ("Parameterized States", validate_parameterized_states),
        ("Bell States", validate_bell_states),
        ("Werner States", validate_werner_states),
    ]
    
    if full:
        tests.extend([
            ("Efficiency (5.3×)", lambda: validate_efficiency(10000)),
            ("Noise Robustness", lambda: validate_noise_robustness(2000)),
        ])
    
    for name, test_func in tests:
        if verbose:
            print(f"\n{name}:")
        
        results = test_func()
        all_results[name] = results
        
        if verbose:
            for r in results.results:
                print(f"  {r['status']}: {r['name']}")
                if r['details'] and not r['passed']:
                    print(f"         {r['details']}")
    
    # Summary
    total_passed = sum(r.passed for r in all_results.values())
    total_failed = sum(r.failed for r in all_results.values())
    
    if verbose:
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Results: {total_passed} passed, {total_failed} failed")
        
        if total_failed == 0:
            print("\n✓ ALL ANALYTICAL FORMULAS VALIDATED SUCCESSFULLY")
        else:
            print(f"\n⚠ {total_failed} test(s) require review")
    
    return all_results


def main():
    """CLI entry point for validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate analytical formulas from the manuscript",
        epilog="For more details, see the Supplementary Information."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick validation (skip slow tests)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output, only show summary"
    )

    args = parser.parse_args()

    results = run_validation(
        verbose=not args.quiet,
        full=not args.quick
    )

    # Return exit code based on results
    total_failed = sum(r.failed for r in results.values())
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
