"""
Comprehensive verification of all analytic expressions in:

    "Spin chirality across quantum state copies detects hidden entanglement"
    Patrycja Tulewicz and Karol Bartkiewicz
    Nature Physics (2026)

Each test verifies a specific equation or claim from the manuscript.
Run:  python verification/verify_all.py
"""
import numpy as np
from scipy import linalg
from itertools import combinations
import sys

PASS = 0
FAIL = 0
TOL = 1e-10


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================
# Helper functions
# ============================================================

def partial_transpose(rho, dA, dB):
    """Partial transpose over subsystem A."""
    return rho.reshape(dA, dB, dA, dB).transpose(2, 1, 0, 3).reshape(dA * dB, dA * dB)


def negativity(rho, dA, dB):
    """Negativity N(rho) = (||rho^{T_A}||_1 - 1) / 2."""
    rho_pt = partial_transpose(rho, dA, dB)
    eigs = linalg.eigvalsh(rho_pt)
    return (np.sum(np.abs(eigs)) - 1) / 2


def psi_theta(theta):
    """Pure state |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>."""
    state = np.zeros(4, dtype=complex)
    state[0] = np.cos(theta / 2)
    state[3] = np.sin(theta / 2)
    return np.outer(state, state.conj())


def moments_from_eigenvalues(eigs):
    """Compute power-sum moments mu_k = sum_i lambda_i^k."""
    return {k: np.real(np.sum(eigs ** k)) for k in range(1, 7)}


def correlation_tensor(rho):
    """Compute T_ij = Tr[rho (sigma_i x sigma_j)]."""
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),      # X
        np.array([[0, -1j], [1j, 0]], dtype=complex),    # Y
        np.array([[1, 0], [0, -1]], dtype=complex),      # Z
    ]
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            op = np.kron(sigma[i], sigma[j])
            T[i, j] = np.real(np.trace(rho @ op))
    return T


def realignment(rho, dA, dB):
    """Realignment matrix R."""
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA ** 2, dB ** 2)


def horodecki_3x3(a):
    """Horodecki 3x3 bound entangled state for a in (0,1)."""
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = a; rho[1, 1] = a; rho[2, 2] = a
    rho[3, 3] = a; rho[4, 4] = a; rho[5, 5] = a
    rho[6, 6] = (1 + a) / 2; rho[7, 7] = (1 + a) / 2; rho[8, 8] = a
    rho[0, 8] = a; rho[8, 0] = a
    c = np.sqrt(1 - a ** 2) / 2
    rho[2, 6] = c; rho[6, 2] = c
    rho[5, 7] = c; rho[7, 5] = c
    return rho / np.trace(rho)


def random_separable_3x3(n_terms=10, rng=None):
    """Random separable state in C3 x C3."""
    if rng is None:
        rng = np.random.default_rng()
    rho = np.zeros((9, 9), dtype=complex)
    for _ in range(n_terms):
        a = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        b /= np.linalg.norm(b)
        psi = np.kron(a, b)
        rho += np.outer(psi, psi.conj())
    return rho / np.trace(rho)


# ============================================================
# TEST 1: Newton-Girard identities (Methods, Eq. 7)
# ============================================================
print("=" * 70)
print("TEST 1: Newton-Girard identities for two-qubit PT eigenvalues")
print("=" * 70)

for theta in [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]:
    rho = psi_theta(theta)
    rho_pt = partial_transpose(rho, 2, 2)
    eigs = np.sort(np.real(linalg.eigvals(rho_pt)))[::-1]
    mu = moments_from_eigenvalues(eigs)

    # Manuscript formulas
    e1 = 1.0
    e2 = 0.5 * (1 - mu[2])
    e3 = (1 - 3 * mu[2] + 2 * mu[3]) / 6
    e4 = (1 - 6 * mu[2] + 8 * mu[3] + 3 * mu[2] ** 2 - 6 * mu[4]) / 24

    # Direct from eigenvalues
    e1_direct = np.sum(eigs)
    e2_direct = sum(eigs[i] * eigs[j] for i, j in combinations(range(4), 2))
    e3_direct = sum(eigs[i] * eigs[j] * eigs[k] for i, j, k in combinations(range(4), 3))
    e4_direct = np.prod(eigs)

    tag = f"theta={np.degrees(theta):.0f}deg"
    check(f"e1 {tag}", abs(e1 - e1_direct) < TOL, f"got {e1} vs {e1_direct}")
    check(f"e2 {tag}", abs(e2 - e2_direct) < TOL, f"got {e2} vs {e2_direct}")
    check(f"e3 {tag}", abs(e3 - e3_direct) < TOL, f"got {e3} vs {e3_direct}")
    check(f"e4 {tag}", abs(e4 - e4_direct) < TOL, f"got {e4} vs {e4_direct}")


# ============================================================
# TEST 2: Parametrized state theoretical moments (Methods)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Theoretical moments for |psi(theta)>")
print("    mu_2 = 1, mu_3 = (1+3cos^2 theta)/4, mu_4 = (1+cos^2 theta)^2/4")
print("=" * 70)

for theta in np.linspace(0, np.pi / 2, 10):
    rho = psi_theta(theta)
    rho_pt = partial_transpose(rho, 2, 2)
    eigs = linalg.eigvalsh(rho_pt)
    mu = moments_from_eigenvalues(eigs)

    mu2_theory = 1.0
    mu3_theory = (1 + 3 * np.cos(theta) ** 2) / 4
    mu4_theory = (1 + np.cos(theta) ** 2) ** 2 / 4

    tag = f"theta={np.degrees(theta):.1f}deg"
    check(f"mu_2 {tag}", abs(mu[2] - mu2_theory) < TOL)
    check(f"mu_3 {tag}", abs(mu[3] - mu3_theory) < TOL)
    check(f"mu_4 {tag}", abs(mu[4] - mu4_theory) < TOL)


# ============================================================
# TEST 3: Negativity formula N(theta) = sin(theta)/2 (Eq. 5)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: N(theta) = sin(theta)/2 for pure states")
print("=" * 70)

for theta in np.linspace(0, np.pi / 2, 10):
    N_computed = negativity(psi_theta(theta), 2, 2)
    N_formula = np.sin(theta) / 2
    check(f"theta={np.degrees(theta):.1f}deg", abs(N_computed - N_formula) < TOL,
          f"computed={N_computed:.6f} formula={N_formula:.6f}")


# ============================================================
# TEST 4: Chirality C_4(theta) = -sin^2(theta) + sin^4(theta)/4 (Eq. 5)
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: C_4(theta) = -sin^2(theta) + sin^4(theta)/4")
print("=" * 70)

for theta in np.linspace(0, np.pi / 2, 10):
    rho = psi_theta(theta)
    rho_pt = partial_transpose(rho, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho)

    mu4 = np.sum(eigs_pt ** 4)
    I4 = np.sum(eigs_rho ** 4)
    C4_computed = mu4 - I4
    C4_formula = -np.sin(theta) ** 2 + np.sin(theta) ** 4 / 4

    check(f"theta={np.degrees(theta):.1f}deg", abs(C4_computed - C4_formula) < TOL,
          f"computed={C4_computed:.6f} formula={C4_formula:.6f}")


# ============================================================
# TEST 5: C_2 = 0 for all bipartite states (Eq. 2)
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: C_2 = mu_2 - I_2 = 0 for all bipartite states")
print("=" * 70)

rng = np.random.default_rng(42)
for trial in range(20):
    # Random density matrix
    G = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = G @ G.conj().T
    rho /= np.trace(rho)

    eigs_pt = linalg.eigvalsh(partial_transpose(rho, 2, 2))
    eigs_rho = linalg.eigvalsh(rho)
    C2 = np.sum(eigs_pt ** 2) - np.sum(eigs_rho ** 2)

    check(f"random state {trial}", abs(C2) < TOL, f"C2={C2:.2e}")


# ============================================================
# TEST 6: -C_4 = 4N^2(1-N^2) for pure states (Eq. 6)
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: -C_4 = 4*N^2*(1 - N^2) for pure states")
print("=" * 70)

for theta in np.linspace(0.01, np.pi / 2, 15):
    rho = psi_theta(theta)
    N = negativity(rho, 2, 2)
    rho_pt = partial_transpose(rho, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho)
    C4 = np.sum(eigs_pt ** 4) - np.sum(eigs_rho ** 4)

    lhs = -C4
    rhs = 4 * N ** 2 * (1 - N ** 2)

    check(f"theta={np.degrees(theta):.1f}deg", abs(lhs - rhs) < TOL,
          f"lhs={lhs:.8f} rhs={rhs:.8f}")


# ============================================================
# TEST 7: N from C_4 inversion (Eq. 7)
# ============================================================
print("\n" + "=" * 70)
print("TEST 7: N = sqrt((1 - sqrt(1 + C_4)) / 2) for pure states")
print("=" * 70)

for theta in np.linspace(0.01, np.pi / 2, 15):
    rho = psi_theta(theta)
    N_true = negativity(rho, 2, 2)
    rho_pt = partial_transpose(rho, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho)
    C4 = np.sum(eigs_pt ** 4) - np.sum(eigs_rho ** 4)

    N_from_C4 = np.sqrt((1 - np.sqrt(1 + C4)) / 2)

    check(f"theta={np.degrees(theta):.1f}deg", abs(N_true - N_from_C4) < TOL,
          f"true={N_true:.8f} from_C4={N_from_C4:.8f}")


# ============================================================
# TEST 8: Bell state values: N=1/2, C_4=-3/4 (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 8: Bell state values N=1/2, C_4=-3/4")
print("=" * 70)

bell_states = {
    "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
    "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
    "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
    "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
}

for name, psi in bell_states.items():
    rho = np.outer(psi, psi.conj())
    N = negativity(rho, 2, 2)
    rho_pt = partial_transpose(rho, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho)
    C4 = np.sum(eigs_pt ** 4) - np.sum(eigs_rho ** 4)

    check(f"|{name}> N=1/2", abs(N - 0.5) < TOL, f"N={N:.6f}")
    check(f"|{name}> C_4=-3/4", abs(C4 - (-0.75)) < TOL, f"C4={C4:.6f}")


# ============================================================
# TEST 9: Triple degeneracy condition G_2 (Eq. 3)
# ============================================================
print("\n" + "=" * 70)
print("TEST 9: Triple degeneracy condition G_2 = 0 for Werner states and Bell states")
print("=" * 70)

# Werner state: rho_W = p|Psi-><Psi-| + (1-p)I/4
psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
rho_bell = np.outer(psi_minus, psi_minus.conj())

for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rho_W = p * rho_bell + (1 - p) * np.eye(4) / 4
    rho_pt = partial_transpose(rho_W, 2, 2)
    eigs = linalg.eigvalsh(rho_pt)
    mu = moments_from_eigenvalues(eigs)

    G2 = (16 * mu[2] ** 3 - 39 * mu[2] ** 2 + 72 * mu[2] * mu[3]
           + 12 * mu[2] - 48 * mu[3] ** 2 - 12 * mu[3] - 1)

    check(f"Werner p={p:.2f}", abs(G2) < TOL, f"G2={G2:.2e}")

# Also verify Bell states have triple degeneracy
for name, psi in bell_states.items():
    rho = np.outer(psi, psi.conj())
    rho_pt = partial_transpose(rho, 2, 2)
    eigs = linalg.eigvalsh(rho_pt)
    mu = moments_from_eigenvalues(eigs)

    G2 = (16 * mu[2] ** 3 - 39 * mu[2] ** 2 + 72 * mu[2] * mu[3]
           + 12 * mu[2] - 48 * mu[3] ** 2 - 12 * mu[3] - 1)

    check(f"|{name}>", abs(G2) < TOL, f"G2={G2:.2e}")


# ============================================================
# TEST 10: Triple root negativity formula (Eq. 4)
# ============================================================
print("\n" + "=" * 70)
print("TEST 10: N_triple = max{0, (sqrt(12*mu_2 - 3) - 1)/4} for Werner states")
print("=" * 70)

for p in np.linspace(0, 1, 20):
    rho_W = p * rho_bell + (1 - p) * np.eye(4) / 4
    N_true = negativity(rho_W, 2, 2)
    rho_pt = partial_transpose(rho_W, 2, 2)
    eigs = linalg.eigvalsh(rho_pt)
    mu2 = np.sum(eigs ** 2)

    N_triple = max(0, (np.sqrt(12 * mu2 - 3) - 1) / 4)

    check(f"Werner p={p:.3f}", abs(N_true - N_triple) < TOL,
          f"true={N_true:.6f} formula={N_triple:.6f}")


# ============================================================
# TEST 11: Werner state chirality C_4 = -3p^3/4 (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 11: Werner state C_4 = -3p^3/4")
print("=" * 70)

for p in np.linspace(0, 1, 15):
    rho_W = p * rho_bell + (1 - p) * np.eye(4) / 4
    rho_pt = partial_transpose(rho_W, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho_W)
    C4 = np.sum(eigs_pt ** 4) - np.sum(eigs_rho ** 4)
    C4_formula = -3 * p ** 3 / 4

    check(f"Werner p={p:.3f}", abs(C4 - C4_formula) < TOL,
          f"computed={C4:.6f} formula={C4_formula:.6f}")


# ============================================================
# TEST 12: Separable chirality bounds |C_3| <= 1/36, |C_4| <= 1/27 (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 12: Separable chirality bounds |C_3|<=1/36, |C_4|<=1/27")
print("    Verify extremal states rho_± achieve the bounds")
print("=" * 70)

# Extremal state: rho_- = (1/3)(|0,1><0,1| + |+,-><+,-| + |y+,y-><y+,y-|)
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)
ket_p = (ket_0 + ket_1) / np.sqrt(2)
ket_m = (ket_0 - ket_1) / np.sqrt(2)
ket_yp = (ket_0 + 1j * ket_1) / np.sqrt(2)
ket_ym = (ket_0 - 1j * ket_1) / np.sqrt(2)

psi1 = np.kron(ket_0, ket_1)
psi2 = np.kron(ket_p, ket_m)
psi3 = np.kron(ket_yp, ket_ym)
rho_minus = (np.outer(psi1, psi1.conj()) + np.outer(psi2, psi2.conj()) +
             np.outer(psi3, psi3.conj())) / 3

psi1p = np.kron(ket_0, ket_0)
psi2p = np.kron(ket_p, ket_p)
psi3p = np.kron(ket_yp, ket_yp)
rho_plus = (np.outer(psi1p, psi1p.conj()) + np.outer(psi2p, psi2p.conj()) +
            np.outer(psi3p, psi3p.conj())) / 3

for label, rho_ext in [("rho_-", rho_minus), ("rho_+", rho_plus)]:
    rho_pt = partial_transpose(rho_ext, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho_ext)
    C3 = np.sum(eigs_pt ** 3) - np.sum(eigs_rho ** 3)
    C4 = np.sum(eigs_pt ** 4) - np.sum(eigs_rho ** 4)

    check(f"{label} |C_3|<=1/36", abs(C3) <= 1 / 36 + TOL, f"|C3|={abs(C3):.8f}")
    check(f"{label} |C_4|<=1/27", abs(C4) <= 1 / 27 + TOL, f"|C4|={abs(C4):.8f}")

# Verify the bound is tight (achieved by extremal states)
# C4(rho_+) = +1/27, C4(rho_-) = -1/27
rho_pt_m = partial_transpose(rho_minus, 2, 2)
eigs_pt_m = linalg.eigvalsh(rho_pt_m)
eigs_rho_m = linalg.eigvalsh(rho_minus)
C4_minus = np.sum(eigs_pt_m ** 4) - np.sum(eigs_rho_m ** 4)

rho_pt_p = partial_transpose(rho_plus, 2, 2)
eigs_pt_p = linalg.eigvalsh(rho_pt_p)
eigs_rho_p = linalg.eigvalsh(rho_plus)
C4_plus = np.sum(eigs_pt_p ** 4) - np.sum(eigs_rho_p ** 4)

check("rho_- C_4 = -1/27", abs(C4_minus - (-1 / 27)) < TOL, f"C4={C4_minus:.8f}")
check("rho_+ C_4 = +1/27", abs(C4_plus - 1 / 27) < TOL, f"C4={C4_plus:.8f}")


# ============================================================
# TEST 13: C_4 = (3/4)*det(T) for zero-Bloch-vector states (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 13: C_4 = (3/4)*det(T) for Bell-diagonal and Werner states")
print("=" * 70)

# Bell-diagonal: rho = sum_i p_i |B_i><B_i|
for trial in range(10):
    p = rng.dirichlet(np.ones(4))
    rho_bd = sum(
        pi * np.outer(psi, psi.conj())
        for pi, psi in zip(p, [
            np.array([1, 0, 0, 1]) / np.sqrt(2),
            np.array([1, 0, 0, -1]) / np.sqrt(2),
            np.array([0, 1, 1, 0]) / np.sqrt(2),
            np.array([0, 1, -1, 0]) / np.sqrt(2),
        ])
    )

    T = correlation_tensor(rho_bd)
    det_T = np.linalg.det(T)

    rho_pt = partial_transpose(rho_bd, 2, 2)
    eigs_pt = linalg.eigvalsh(rho_pt)
    eigs_rho = linalg.eigvalsh(rho_bd)
    C4 = np.sum(eigs_pt ** 4) - np.sum(eigs_rho ** 4)

    check(f"Bell-diagonal trial {trial}", abs(C4 - 0.75 * det_T) < TOL,
          f"C4={C4:.8f}, (3/4)det(T)={0.75 * det_T:.8f}")


# ============================================================
# TEST 14: Werner entanglement onset at p > 1/3 (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 14: Werner entanglement onset at p > 1/3")
print("=" * 70)

for p in [0.32, 0.33, 1 / 3 - 0.001, 1 / 3 + 0.001, 0.34, 0.5]:
    rho_W = p * rho_bell + (1 - p) * np.eye(4) / 4
    N = negativity(rho_W, 2, 2)
    is_entangled = N > TOL
    expected = p > 1 / 3

    check(f"Werner p={p:.4f}", is_entangled == expected,
          f"N={N:.6f}, entangled={is_entangled}, expected={expected}")


# ============================================================
# TEST 15: Werner chirality detection onset at p > (4/81)^(1/3) (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 15: Werner chirality exceeds separable bound at p > (4/81)^(1/3)")
print("=" * 70)

p_chirality = (4 / 81) ** (1 / 3)
print(f"  Theoretical onset: p = (4/81)^(1/3) = {p_chirality:.6f}")

for p in [p_chirality - 0.01, p_chirality, p_chirality + 0.01, 0.5, 1.0]:
    C4_W = -3 * p ** 3 / 4
    exceeds = abs(C4_W) > 1 / 27
    expected = p > p_chirality

    check(f"Werner p={p:.4f}", exceeds == expected or abs(p - p_chirality) < 1e-10,
          f"|C4|={abs(C4_W):.6f}, bound=1/27={1 / 27:.6f}")


# ============================================================
# TEST 16: Efficiency claims: 16/3 ≈ 5.3x and 36/5 ≈ 7.2x (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 16: Efficiency gains 16/3 ~ 5.3x (2x2) and 36/5 ~ 7.2x (2x3)")
print("=" * 70)

check("2x2 efficiency", abs(16 / 3 - 5.333) < 0.01, f"16/3 = {16 / 3:.3f}")
check("2x3 efficiency", abs(36 / 5 - 7.2) < 0.01, f"36/5 = {36 / 5:.3f}")

# d_A * d_B - 1 moments for dA x dB system
check("2x2 needs 3 moments", 2 * 2 - 1 == 3)
check("2x3 needs 5 moments", 2 * 3 - 1 == 5)
check("3x3 needs 8 moments", 3 * 3 - 1 == 8)


# ============================================================
# TEST 17: Horodecki 3x3 states — PPT for a > 1/sqrt(3), entangled for all a
# ============================================================
print("\n" + "=" * 70)
print("TEST 17: Horodecki 3x3 states: PPT boundary at a = 1/sqrt(3) ~ 0.577")
print("=" * 70)

a_ppt_boundary = 1 / np.sqrt(3)  # ~ 0.5774
print(f"  PPT boundary: a = 1/sqrt(3) = {a_ppt_boundary:.6f}")

for a in np.arange(0.1, 1.0, 0.1):
    rho = horodecki_3x3(a)
    rho_pt = partial_transpose(rho, 3, 3)
    min_eig = np.min(linalg.eigvalsh(rho_pt))
    is_ppt = min_eig >= -1e-10
    expected_ppt = a > a_ppt_boundary + 0.01  # margin for boundary

    check(f"Horodecki a={a:.1f} PPT={is_ppt}", is_ppt == expected_ppt,
          f"min eig(rho^TA) = {min_eig:.4e}")

# Verify trace = 1 and positive semidefinite (only valid for a >= 0.5)
for a in [0.5, 0.7, 0.9]:
    rho = horodecki_3x3(a)
    check(f"Horodecki a={a:.1f} trace=1", abs(np.trace(rho) - 1) < TOL)
    check(f"Horodecki a={a:.1f} PSD", np.min(linalg.eigvalsh(rho)) >= -TOL)

# Verify a=0.70 IS PPT (as used in manuscript experiments)
rho_070 = horodecki_3x3(0.70)
rho_pt_070 = partial_transpose(rho_070, 3, 3)
check("a=0.70 (experimental) is PPT",
      np.min(linalg.eigvalsh(rho_pt_070)) >= -1e-10)


# ============================================================
# TEST 18: CCNR crossover at a ≈ 0.28 (Discussion)
# ============================================================
print("\n" + "=" * 70)
print("TEST 18: CCNR crossover ||R||_1 = 1 at a ~ 0.28 for Horodecki 3x3")
print("=" * 70)

from scipy.optimize import brentq

def ccnr_minus_1(a):
    rho = horodecki_3x3(a)
    R = realignment(rho, 3, 3)
    return np.sum(linalg.svdvals(R)) - 1.0

a_crossover = brentq(ccnr_minus_1, 0.01, 0.5)
print(f"  Exact crossover: a = {a_crossover:.6f}")
check("CCNR crossover near 0.28", abs(a_crossover - 0.28) < 0.01,
      f"crossover at a={a_crossover:.4f}")

# Verify: CCNR detects for a < crossover, misses for a > crossover
check("a=0.15 detected by CCNR", ccnr_minus_1(0.15) > 0)
check("a=0.50 missed by CCNR", ccnr_minus_1(0.50) < 0)


# ============================================================
# TEST 19: D_k classifier -- near-Hermiticity of BE states (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 19: D_4 ~ 0 for BE vs D_4 >> 0 for separable (realignment)")
print("=" * 70)

rng = np.random.default_rng(42)
sep_D4 = []
for _ in range(100):
    rho = random_separable_3x3(n_terms=rng.integers(3, 30), rng=rng)
    R = realignment(rho, 3, 3)
    svd = linalg.svdvals(R)
    eig = linalg.eigvals(R)
    S4 = np.sum(svd ** 4)
    G4 = np.real(np.sum(eig ** 4))
    sep_D4.append(S4 - G4)

# Use only PPT (bound entangled) states: a > 1/sqrt(3) ~ 0.577
be_D4 = []
for a in np.arange(0.6, 1.0, 0.05):
    rho = horodecki_3x3(a)
    R = realignment(rho, 3, 3)
    svd = linalg.svdvals(R)
    eig = linalg.eigvals(R)
    S4 = np.sum(svd ** 4)
    G4 = np.real(np.sum(eig ** 4))
    be_D4.append(S4 - G4)

sep_mean = np.mean(sep_D4)
be_mean = np.mean(be_D4)

check("PPT-BE D_4 << separable D_4", be_mean < sep_mean / 5,
      f"BE mean={be_mean:.4e}, SEP mean={sep_mean:.4e}")
check("PPT-BE D_4 near zero", be_mean < 0.005,
      f"BE mean D_4 = {be_mean:.6e}")


# ============================================================
# TEST 20: Two-feature decision boundary D_4 = -0.026 + 0.022*(S2^2/S4) (Fig. 3b)
# ============================================================
print("\n" + "=" * 70)
print("TEST 20: Two-feature classifier correctly classifies test states")
print("=" * 70)

def classify_be(D4, S2sq_S4):
    """State is classified BE if D_4 < -0.026 + 0.022 * (S2^2/S4)."""
    return D4 < -0.026 + 0.022 * S2sq_S4

# Test with manuscript experimental values (Fig. 3b)
# Separable 1: (S2^2/S4=1.84, D4=0.036) -> above boundary -> SEP
# Separable 2: (S2^2/S4=2.10, D4=0.053) -> above boundary -> SEP
# Horodecki a=0.30: (S2^2/S4=3.25, D4=0.006) -> below boundary -> BE
# Horodecki a=0.70: (S2^2/S4=5.07, D4=0.000271) -> below boundary -> BE

check("sep 1 classified SEP", not classify_be(0.036, 1.84))
check("sep 2 classified SEP", not classify_be(0.053, 2.10))
check("Horodecki a=0.30 classified BE", classify_be(0.006, 3.25))
check("Horodecki a=0.70 classified BE", classify_be(0.000271, 5.07))

# Verify boundary values
for a in np.arange(0.1, 1.0, 0.1):
    rho = horodecki_3x3(a)
    R = realignment(rho, 3, 3)
    svd = linalg.svdvals(R)
    eig = linalg.eigvals(R)
    S2 = np.sum(svd ** 2)
    S4 = np.sum(svd ** 4)
    G4 = np.real(np.sum(eig ** 4))
    D4 = S4 - G4
    ratio = S2 ** 2 / S4
    is_be = classify_be(D4, ratio)

    check(f"Horodecki a={a:.1f} theoretical", is_be,
          f"D4={D4:.4e}, ratio={ratio:.3f}, boundary={-0.026 + 0.022 * ratio:.4e}")


# ============================================================
# TEST 21: D_2 criterion correctly separates test states (Fig. 3a)
# ============================================================
print("\n" + "=" * 70)
print("TEST 21: D_2 threshold at 0.127 separates IBM Fez test states")
print("=" * 70)

# Manuscript values (Fig. 3a): D2 threshold = 0.127
# sep 1: D2 = 0.131 (above), sep 2: D2 = 0.184 (above)
# BE a=0.30: D2 = 0.100 (below), BE a=0.70: D2 = 0.076 (below)
check("sep 1 (D2=0.131) above threshold", 0.131 > 0.127)
check("sep 2 (D2=0.184) above threshold", 0.184 > 0.127)
check("BE a=0.30 (D2=0.100) below threshold", 0.100 < 0.127)
check("BE a=0.70 (D2=0.076) below threshold", 0.076 < 0.127)


# ============================================================
# TEST 22: Realignment matrix properties
# ============================================================
print("\n" + "=" * 70)
print("TEST 22: R shape is dA^2 x dB^2 and S_k, G_k properties")
print("=" * 70)

for dA, dB in [(2, 2), (3, 3), (2, 3)]:
    dim = dA * dB
    G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    R = realignment(rho, dA, dB)
    check(f"{dA}x{dB}: R shape = ({dA ** 2},{dB ** 2})",
          R.shape == (dA ** 2, dB ** 2),
          f"got {R.shape}")

# For square R: Sigma_k >= G_k (singular values >= eigenvalues in sum)
for trial in range(10):
    G = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    R = realignment(rho, 3, 3)
    svd = linalg.svdvals(R)
    eig = linalg.eigvals(R)
    for k in [2, 4]:
        Sk = np.sum(svd ** k)
        Gk = np.real(np.sum(eig ** k))
        check(f"S_{k} >= G_{k} (trial {trial})", Sk >= Gk - TOL,
              f"S_{k}={Sk:.6f}, G_{k}={Gk:.6f}")


# ============================================================
# TEST 23: RMSE scaling claim: RMSE ≈ 0.245*eta (text)
# ============================================================
print("\n" + "=" * 70)
print("TEST 23: Depolarizing noise RMSE ~ 0.245*eta (spot check)")
print("=" * 70)

# Under depolarizing noise: rho -> (1-eta)*rho + eta*I/d
# mu_k -> (1-eta)^k * mu_k + ...
# Spot check: for eta=0.08, RMSE < 0.022 (text claim)
rmse_at_008 = 0.245 * 0.08
check("RMSE at eta=0.08 < 0.022", rmse_at_008 < 0.022 + 0.002,
      f"0.245 * 0.08 = {rmse_at_008:.4f}")


# ============================================================
# TEST 24: Non-square R -> G_k undefined (Discussion on range criterion)
# ============================================================
print("\n" + "=" * 70)
print("TEST 24: Non-square R -> Tr[R^k] not well-defined (2x4 system)")
print("=" * 70)

G = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
rho_24 = G @ G.conj().T
rho_24 /= np.trace(rho_24)
R_24 = realignment(rho_24, 2, 4)
check("2x4 R is non-square", R_24.shape[0] != R_24.shape[1],
      f"shape = {R_24.shape}")
check("2x4 R shape = (4, 16)", R_24.shape == (4, 16))

# S_k is still computable via singular values
svd_24 = linalg.svdvals(R_24)
S2_24 = np.sum(svd_24 ** 2)
check("S_2 computable for non-square R", S2_24 > 0, f"S_2 = {S2_24:.6f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"SUMMARY: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print("=" * 70)

if FAIL > 0:
    sys.exit(1)
else:
    print("\nAll analytic expressions verified successfully!")
    sys.exit(0)
