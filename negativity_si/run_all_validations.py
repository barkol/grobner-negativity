#!/usr/bin/env python3
"""
Run All Validations
===================

Comprehensive validation script for the negativity estimation package.
Verifies all claims from the manuscript and SI.

Usage:
    python run_all_validations.py           # Run all tests
    python run_all_validations.py --quick   # Skip slow tests
    python run_all_validations.py --circuits # Include circuit simulations
    python run_all_validations.py --simulations # Include chirality witness simulations

Authors: Patrycja Tulewicz and Karol Bartkiewicz
"""

import sys
import time
import argparse
import numpy as np

# Add package to path if running from this directory
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def run_analytical_validation(verbose=True, full=True):
    """Run analytical formula validation using local implementations."""
    from numpy import sqrt, trace
    from numpy.linalg import eigvalsh, matrix_power
    from itertools import combinations

    print("\n" + "=" * 80)
    print("PART 1: ANALYTICAL FORMULAS")
    print("=" * 80)

    def partial_transpose(rho):
        rho_reshaped = rho.reshape(2, 2, 2, 2)
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
        return rho_pt.reshape(4, 4)

    def random_density_matrix(d=4, seed=None):
        if seed is not None:
            np.random.seed(seed)
        G = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / sqrt(2)
        rho = G @ G.conj().T
        rho = (rho + rho.conj().T) / 2
        return rho / trace(rho)

    def newton_girard_elementary(mu_2, mu_3, mu_4, mu_1=1.0):
        e1 = mu_1
        e2 = (mu_1**2 - mu_2) / 2
        e3 = (mu_1**3 - 3 * mu_1 * mu_2 + 2 * mu_3) / 6
        e4 = (mu_1**4 - 6 * mu_1**2 * mu_2 + 8 * mu_1 * mu_3 + 3 * mu_2**2 - 6 * mu_4) / 24
        return e1, e2, e3, e4

    def reconstruct_eigenvalues(mu_2, mu_3, mu_4):
        e1, e2, e3, e4 = newton_girard_elementary(mu_2, mu_3, mu_4)
        coefficients = [1, -e1, e2, -e3, e4]
        eigenvalues = np.roots(coefficients)
        return np.sort(np.real(eigenvalues))

    def compute_negativity_newton_girard(mu_2, mu_3, mu_4):
        eigenvalues = reconstruct_eigenvalues(mu_2, mu_3, mu_4)
        return -np.sum(eigenvalues[eigenvalues < -1e-10])

    passed = 0
    failed = 0

    # Test 1: Newton-Girard identities
    print("\nNewton-Girard Identities:")
    np.random.seed(42)
    max_errors = [0.0, 0.0, 0.0]

    for _ in range(100):
        eigs = np.random.randn(4)
        eigs = eigs - eigs.mean() + 0.25
        mu = {k: sum(e**k for e in eigs) for k in range(1, 5)}
        e1, e2_ng, e3_ng, e4_ng = newton_girard_elementary(mu[2], mu[3], mu[4], mu[1])

        e2_dir = sum(eigs[i]*eigs[j] for i,j in combinations(range(4), 2))
        e3_dir = sum(eigs[i]*eigs[j]*eigs[k] for i,j,k in combinations(range(4), 3))
        e4_dir = eigs[0]*eigs[1]*eigs[2]*eigs[3]

        max_errors[0] = max(max_errors[0], abs(e2_ng - e2_dir))
        max_errors[1] = max(max_errors[1], abs(e3_ng - e3_dir))
        max_errors[2] = max(max_errors[2], abs(e4_ng - e4_dir))

    for i, name in enumerate(["e2", "e3", "e4"]):
        if max_errors[i] < 1e-10:
            passed += 1
            print(f"  [PASS] PASS: {name} formula, max error = {max_errors[i]:.2e}")
        else:
            failed += 1
            print(f"  [FAIL] FAIL: {name} formula, max error = {max_errors[i]:.2e}")

    # Test 2: Eigenvalue reconstruction
    print("\nEigenvalue Reconstruction:")
    np.random.seed(123)
    max_error = 0.0

    for i in range(50):
        rho = random_density_matrix(4, seed=i)
        rho_pt = partial_transpose(rho)
        eigs_true = np.sort(eigvalsh(rho_pt))

        mu_2 = np.real(trace(rho_pt @ rho_pt))
        mu_3 = np.real(trace(matrix_power(rho_pt, 3)))
        mu_4 = np.real(trace(matrix_power(rho_pt, 4)))

        eigs_recon = reconstruct_eigenvalues(mu_2, mu_3, mu_4)
        max_error = max(max_error, np.max(np.abs(eigs_true - eigs_recon)))

    if max_error < 1e-8:
        passed += 1
        print(f"  [PASS] PASS: eigenvalue reconstruction, max error = {max_error:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: eigenvalue reconstruction, max error = {max_error:.2e}")

    # Test 3: Grobner conditions
    print("\nGrobner Conditions:")

    # G1 for two-pair degeneracy
    alpha = 0.2
    beta = 0.5 - alpha
    eigs = np.array([alpha, alpha, beta, beta])
    mu_2 = np.sum(eigs**2)
    mu_3 = np.sum(eigs**3)
    G1 = 6 * mu_2 - 8 * mu_3 - 1

    if abs(G1) < 1e-10:
        passed += 1
        print(f"  [PASS] PASS: G1 two-pair, G1 = {G1:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: G1 two-pair, G1 = {G1:.2e}")

    # G2 for triple degeneracy
    alpha = 0.2
    beta = 1 - 3*alpha
    eigs = np.array([alpha, alpha, alpha, beta])
    mu_2 = np.sum(eigs**2)
    mu_3 = np.sum(eigs**3)
    G2 = 16*mu_2**3 - 39*mu_2**2 + 72*mu_2*mu_3 + 12*mu_2 - 48*mu_3**2 - 12*mu_3 - 1

    if abs(G2) < 1e-10:
        passed += 1
        print(f"  [PASS] PASS: G2 triple, G2 = {G2:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: G2 triple, G2 = {G2:.2e}")

    return passed, failed


def run_circuit_validation():
    """Run circuit simulation validation."""
    print("\n" + "=" * 80)
    print("PART 2: CIRCUIT SIMULATIONS")
    print("=" * 80)

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        print("Qiskit not available. Skipping circuit tests.")
        return 0, 0

    from numpy import cos, sin

    def partial_transpose(rho):
        rho_reshaped = rho.reshape(2, 2, 2, 2)
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
        return rho_pt.reshape(4, 4)

    def compute_mu_k(rho, k):
        rho_pt = partial_transpose(rho)
        return np.real(np.trace(np.linalg.matrix_power(rho_pt, k)))

    def prepare_state(qc, qubits, theta):
        n_copies = len(qubits) // 2
        for i in range(n_copies):
            a = qubits[2 * i]
            b = qubits[2 * i + 1]
            qc.ry(theta, a)
            qc.cx(a, b)

    def create_mu_circuit(k, theta):
        if k == 2:
            qc = QuantumCircuit(5, 1)
            prepare_state(qc, [0, 1, 2, 3], theta)
            qc.h(4)
            qc.cswap(4, 0, 2)
            qc.cswap(4, 1, 3)
            qc.h(4)
            qc.measure(4, 0)
        elif k == 3:
            qc = QuantumCircuit(7, 1)
            prepare_state(qc, [0, 1, 2, 3, 4, 5], theta)
            qc.h(6)
            qc.cswap(6, 0, 2)
            qc.cswap(6, 2, 4)
            qc.cswap(6, 5, 3)
            qc.cswap(6, 3, 1)
            qc.h(6)
            qc.measure(6, 0)
        elif k == 4:
            qc = QuantumCircuit(9, 1)
            prepare_state(qc, [0, 1, 2, 3, 4, 5, 6, 7], theta)
            qc.h(8)
            qc.cswap(8, 0, 2)
            qc.cswap(8, 2, 4)
            qc.cswap(8, 4, 6)
            qc.cswap(8, 7, 5)
            qc.cswap(8, 5, 3)
            qc.cswap(8, 3, 1)
            qc.h(8)
            qc.measure(8, 0)
        return qc

    def run_circuit(qc, shots=50000):
        sim = AerSimulator()
        compiled = transpile(qc, sim)
        result = sim.run(compiled, shots=shots).result()
        counts = result.get_counts()
        p0 = counts.get('0', 0) / shots
        return 2 * p0 - 1

    passed = 0
    failed = 0

    test_angles = [15, 30, 45, 60, 90]

    print(f"\nTesting mu circuits for theta = {test_angles}")
    print("-" * 60)

    for theta_deg in test_angles:
        theta = np.radians(theta_deg)
        c, s = cos(theta / 2), sin(theta / 2)
        psi = np.array([c, 0, 0, s])
        rho = np.outer(psi, psi.conj())

        for k in [2, 3, 4]:
            mu_theory = compute_mu_k(rho, k)
            qc = create_mu_circuit(k, theta)
            mu_circuit = run_circuit(qc)

            error = abs(mu_circuit - mu_theory)
            if error < 0.03:  # 3% tolerance for statistical error
                passed += 1
                status = "[PASS] PASS"
            else:
                failed += 1
                status = "[FAIL] FAIL"

            print(f"  mu_{k} at theta={theta_deg:>2}°: circuit={mu_circuit:>7.4f}, "
                  f"theory={mu_theory:>7.4f}, error={error:.4f} {status}")

    return passed, failed


def run_key_claims_validation():
    """Validate the key claims from the manuscript."""
    print("\n" + "=" * 80)
    print("PART 3: KEY MANUSCRIPT CLAIMS")
    print("=" * 80)

    from numpy import cos, sin, sqrt
    from numpy.linalg import eigvalsh, matrix_power
    from numpy import trace

    def partial_transpose(rho):
        rho_reshaped = rho.reshape(2, 2, 2, 2)
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
        return rho_pt.reshape(4, 4)

    def compute_negativity(rho):
        rho_pt = partial_transpose(rho)
        eigenvalues = eigvalsh(rho_pt)
        return sum(abs(ev) for ev in eigenvalues if ev < -1e-10)

    def create_state(theta):
        psi = np.array([cos(theta / 2), 0, 0, sin(theta / 2)])
        return np.outer(psi, psi.conj())

    def bell_state():
        psi = np.array([1, 0, 0, 1]) / sqrt(2)
        return np.outer(psi, psi.conj())

    passed = 0
    failed = 0

    # Claim 1: N(theta) = sin(theta)/2
    print("\n--- Claim 1: N(theta) = sin(theta)/2 ---")
    max_error = 0
    for theta in np.linspace(0, np.pi, 10):
        rho = create_state(theta)
        N_computed = compute_negativity(rho)
        N_theory = sin(theta) / 2
        max_error = max(max_error, abs(N_computed - N_theory))

    if max_error < 1e-10:
        passed += 1
        print(f"  [PASS] PASS: max error = {max_error:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: max error = {max_error:.2e}")

    # Claim 2: All Bell states have N = 0.5
    print("\n--- Claim 2: N(Bell) = 0.5 ---")
    bell_vecs = [
        np.array([1, 0, 0, 1]) / sqrt(2),
        np.array([1, 0, 0, -1]) / sqrt(2),
        np.array([0, 1, 1, 0]) / sqrt(2),
        np.array([0, 1, -1, 0]) / sqrt(2),
    ]
    max_error = 0
    for psi in bell_vecs:
        rho = np.outer(psi, psi.conj())
        N = compute_negativity(rho)
        max_error = max(max_error, abs(N - 0.5))

    if max_error < 1e-10:
        passed += 1
        print(f"  [PASS] PASS: max error = {max_error:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: max error = {max_error:.2e}")

    # Claim 3: mu_2 = I_2 for all states
    print("\n--- Claim 3: mu_2 = I_2 for all states ---")
    max_error = 0
    for _ in range(20):
        # Random density matrix
        G = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / sqrt(2)
        rho = G @ G.conj().T
        rho = rho / trace(rho)

        rho_pt = partial_transpose(rho)
        mu_2 = np.real(trace(rho_pt @ rho_pt))
        I_2 = np.real(trace(rho @ rho))
        max_error = max(max_error, abs(mu_2 - I_2))

    if max_error < 1e-10:
        passed += 1
        print(f"  [PASS] PASS: max error = {max_error:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: max error = {max_error:.2e}")

    # Claim 4: Efficiency ~5.3x
    print("\n--- Claim 4: Measurement efficiency ~5.3x ---")

    def random_density_matrix():
        G = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / sqrt(2)
        rho = G @ G.conj().T
        rho = (rho + rho.conj().T) / 2
        return rho / trace(rho)

    def classify_degeneracy(eigs, tol=1e-8):
        s = sorted(eigs)
        if max(eigs) - min(eigs) < tol:
            return "quadruple"
        for i in range(4):
            others = [eigs[j] for j in range(4) if j != i]
            if max(others) - min(others) < tol:
                return "triple"
        if abs(s[0] - s[1]) < tol and abs(s[2] - s[3]) < tol:
            return "two_pair"
        return "generic"

    total_meas = 0
    n_samples = 5000
    for _ in range(n_samples):
        rho = random_density_matrix()
        rho_pt = partial_transpose(rho)
        eigs = eigvalsh(rho_pt)
        deg = classify_degeneracy(eigs)
        total_meas += 2 if deg in ["quadruple", "triple", "two_pair"] else 3

    avg_meas = total_meas / n_samples
    efficiency = 16 / avg_meas

    if 4.5 < efficiency < 6.5:
        passed += 1
        print(f"  [PASS] PASS: efficiency = {efficiency:.2f}x (expected ~5.3x)")
    else:
        failed += 1
        print(f"  [FAIL] FAIL: efficiency = {efficiency:.2f}x (expected ~5.3x)")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Run all validations for the negativity estimation package"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip slow tests (efficiency, noise robustness)"
    )
    parser.add_argument(
        "--circuits", action="store_true",
        help="Include circuit simulation tests (requires qiskit)"
    )
    parser.add_argument(
        "--simulations", action="store_true",
        help="Include chirality witness simulations (requires qiskit-aer)"
    )
    parser.add_argument(
        "--claims-only", action="store_true",
        help="Only run key manuscript claims validation"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("NEGATIVITY ESTIMATION PACKAGE - VALIDATION SUITE")
    print("=" * 80)
    print("Validating claims from:")
    print("  'Resource-efficient negativity estimation from partial transpose moments'")
    print("  Tulewicz & Bartkiewicz, npj Quantum Information (2026)")
    print("=" * 80)

    start_time = time.time()

    total_passed = 0
    total_failed = 0

    if args.claims_only:
        p, f = run_key_claims_validation()
        total_passed += p
        total_failed += f
    else:
        # Part 1: Analytical formulas
        p, f = run_analytical_validation(verbose=True, full=not args.quick)
        total_passed += p
        total_failed += f

        # Part 2: Circuit simulations (optional)
        if args.circuits:
            p, f = run_circuit_validation()
            total_passed += p
            total_failed += f

        # Part 3: Chirality witness simulations (optional)
        if args.simulations:
            try:
                from simulations import run_simulation_validation
                p, f = run_simulation_validation(verbose=True)
                total_passed += p
                total_failed += f
            except ImportError as e:
                print(f"\n[SKIP] Simulation validation skipped: {e}")

        # Part 4: Key claims
        p, f = run_key_claims_validation()
        total_passed += p
        total_failed += f

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print(f"Time: {elapsed:.1f} seconds")

    if total_failed == 0:
        print("\n[PASS] ALL VALIDATIONS PASSED")
        return 0
    else:
        print(f"\n[!] {total_failed} validation(s) require review")
        return 1


if __name__ == "__main__":
    sys.exit(main())
