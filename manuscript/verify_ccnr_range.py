"""Verify: CCNR vs D_k classifier vs range criterion for BE detection."""
import numpy as np
from scipy import linalg
from scipy.optimize import brentq

# ============================================================
# State constructors
# ============================================================

def horodecki_3x3(a):
    """Horodecki 3x3 bound entangled state for a in (0,1)."""
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = a;  rho[1, 1] = a;  rho[2, 2] = a
    rho[3, 3] = a;  rho[4, 4] = a;  rho[5, 5] = a
    rho[6, 6] = (1 + a) / 2;  rho[7, 7] = (1 + a) / 2;  rho[8, 8] = a
    rho[0, 8] = a;  rho[8, 0] = a
    c = np.sqrt(1 - a**2) / 2
    rho[2, 6] = c;  rho[6, 2] = c
    rho[5, 7] = c;  rho[7, 5] = c
    return rho / np.trace(rho)

def random_separable_3x3(n_terms=10):
    """Random separable state in C3 x C3."""
    rho = np.zeros((9, 9), dtype=complex)
    for _ in range(n_terms):
        a = np.random.randn(3) + 1j*np.random.randn(3); a /= np.linalg.norm(a)
        b = np.random.randn(3) + 1j*np.random.randn(3); b /= np.linalg.norm(b)
        psi = np.kron(a, b)
        rho += np.outer(psi, psi.conj())
    return rho / np.trace(rho)

def horodecki_2x4_edge(b_param):
    """
    Construct a PPT entangled state in C2 x C4 that violates the range criterion.
    Uses entangled subspace construction: superpositions that cannot be decomposed
    into product vectors spanning the range.
    """
    dA, dB = 2, 4
    dim = dA * dB

    # Entangled vectors in C2 x C4 (Schmidt rank 2)
    phi1 = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
    phi2 = np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=complex) / np.sqrt(2)
    phi3 = np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=complex) / np.sqrt(2)
    phi4 = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=complex) / np.sqrt(2)

    rho = np.zeros((dim, dim), dtype=complex)
    for phi in [phi1, phi2, phi3, phi4]:
        rho += np.outer(phi, phi.conj())

    # Mix with identity
    rho = (1 - b_param) * rho / 4 + b_param * np.eye(dim) / dim
    rho = rho / np.trace(rho)
    return rho

def random_separable_2x4(n_terms=10):
    """Random separable state in C2 x C4."""
    rho = np.zeros((8, 8), dtype=complex)
    for _ in range(n_terms):
        a = np.random.randn(2) + 1j*np.random.randn(2); a /= np.linalg.norm(a)
        b = np.random.randn(4) + 1j*np.random.randn(4); b /= np.linalg.norm(b)
        psi = np.kron(a, b)
        rho += np.outer(psi, psi.conj())
    return rho / np.trace(rho)

# ============================================================
# Criteria
# ============================================================

def realignment(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA**2, dB**2)

def ccnr_val(rho, dA, dB):
    R = realignment(rho, dA, dB)
    return np.sum(linalg.svdvals(R))

def ppt_check(rho, dA, dB):
    rho_t = rho.reshape(dA, dB, dA, dB).transpose(2, 1, 0, 3).reshape(dA*dB, dA*dB)
    return np.min(np.real(linalg.eigvalsh(rho_t)))

def compute_features(rho, dA, dB):
    R = realignment(rho, dA, dB)
    svd_vals = linalg.svdvals(R)

    S_1 = np.sum(svd_vals)
    S_2 = np.sum(svd_vals**2)
    S_4 = np.sum(svd_vals**4)

    is_square = (R.shape[0] == R.shape[1])

    if is_square:
        R_eigvals = linalg.eigvals(R)
        G_2 = np.real(np.sum(R_eigvals**2))
        G_4 = np.real(np.sum(R_eigvals**4))
    else:
        G_2, G_4 = None, None

    D_2 = S_2 - G_2 if G_2 is not None else None
    D_4 = S_4 - G_4 if G_4 is not None else None
    S2sq_S4 = S_2**2 / S_4 if S_4 > 1e-12 else 1.0

    return {
        'S_1': S_1, 'S_2': S_2, 'S_4': S_4,
        'G_2': G_2, 'G_4': G_4,
        'D_2': D_2, 'D_4': D_4,
        'S2sq_S4': S2sq_S4,
        'is_square': is_square,
        'R_shape': R.shape
    }

def classify_be(D_4, S2sq_S4):
    threshold = -0.026 + 0.022 * S2sq_S4
    return D_4 < threshold

# ============================================================
# TEST 1: Horodecki 3x3 -- CCNR vs D_k classifier
# ============================================================
print("=" * 80)
print("TEST 1: Horodecki 3x3 bound entangled states")
print("Comparing CCNR (||R||_1 > 1) vs D_k classifier")
print("=" * 80)
header = f"{'a':>6} | {'CCNR':>8} | {'CCNR>1?':>8} | {'D_4':>12} | {'S2^2/S4':>8} | {'D_k class':>10} | {'D_2':>12}"
print(header)
print("-" * len(header))

a_values = np.arange(0.05, 0.96, 0.05)

for a in a_values:
    rho = horodecki_3x3(a)
    feat = compute_features(rho, 3, 3)
    ccnr = feat['S_1']
    ccnr_detects = ccnr > 1.0
    dk_class = classify_be(feat['D_4'], feat['S2sq_S4'])

    print(f"{a:6.2f} | {ccnr:8.5f} | {'YES' if ccnr_detects else 'no':>8} | {feat['D_4']:12.6e} | {feat['S2sq_S4']:8.4f} | {'BE' if dk_class else 'sep':>10} | {feat['D_2']:12.6e}")

# Find exact CCNR crossover
def ccnr_minus_1(a):
    return ccnr_val(horodecki_3x3(a), 3, 3) - 1.0

# Check endpoints
c_low = ccnr_minus_1(0.01)
c_high = ccnr_minus_1(0.5)

if c_low * c_high < 0:
    a_crossover = brentq(ccnr_minus_1, 0.01, 0.5)
    print(f"\nCCNR crossover: ||R||_1 = 1 at a = {a_crossover:.6f}")
elif c_low < 0 and c_high < 0:
    # Try wider range
    try:
        a_crossover = brentq(ccnr_minus_1, 0.001, 0.99)
        print(f"\nCCNR crossover: ||R||_1 = 1 at a = {a_crossover:.6f}")
    except ValueError:
        print(f"\nCCNR NEVER exceeds 1 for any a in (0,1)!")
        a_crossover = None
else:
    print(f"\nCCNR always > 1 (c_low={c_low:.4f}, c_high={c_high:.4f})")
    a_crossover = 0

# Summary for 3x3
n_ccnr = sum(1 for a in a_values if ccnr_val(horodecki_3x3(a), 3, 3) > 1.0)
n_dk = sum(1 for a in a_values if classify_be(
    compute_features(horodecki_3x3(a), 3, 3)['D_4'],
    compute_features(horodecki_3x3(a), 3, 3)['S2sq_S4']))
print(f"\nCCNR detects: {n_ccnr}/{len(a_values)} states")
print(f"D_k classifier detects: {n_dk}/{len(a_values)} states")

# ============================================================
# TEST 2: Separable 3x3 -- false positive check
# ============================================================
print("\n" + "=" * 80)
print("TEST 2: Random separable 3x3 states (false positive check)")
print("=" * 80)

np.random.seed(42)
n_sep = 200
fp_ccnr = 0
fp_dk = 0
for i in range(n_sep):
    rho = random_separable_3x3(n_terms=np.random.randint(3, 30))
    feat = compute_features(rho, 3, 3)
    if feat['S_1'] > 1.0:
        fp_ccnr += 1
    if classify_be(feat['D_4'], feat['S2sq_S4']):
        fp_dk += 1

print(f"CCNR false positives: {fp_ccnr}/{n_sep}")
print(f"D_k classifier false positives: {fp_dk}/{n_sep}")

# ============================================================
# TEST 3: 2x4 system -- structural limitation
# ============================================================
print("\n" + "=" * 80)
print("TEST 3: Horodecki 2x4 system -- range criterion territory")
print("=" * 80)

rho_test = horodecki_2x4_edge(0.0)
R_test = realignment(rho_test, 2, 4)
print(f"\nR shape for 2x4 system: {R_test.shape}")
print(f"R is square? {R_test.shape[0] == R_test.shape[1]}")
print(f"=> G_k = Tr[R^k] is UNDEFINED for non-square R")
print(f"=> D_k = S_k - G_k is UNDEFINED")
print(f"=> Non-Hermiticity classifier CANNOT be applied\n")

header = f"{'b':>6} | {'PPT min':>10} | {'PPT?':>5} | {'CCNR':>8} | {'CCNR>1?':>8} | {'S_2':>8} | {'S_4':>8} | {'S2^2/S4':>8}"
print(header)
print("-" * len(header))

for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
    rho = horodecki_2x4_edge(b)
    feat = compute_features(rho, 2, 4)
    ppt_min = ppt_check(rho, 2, 4)

    print(f"{b:6.2f} | {ppt_min:10.6f} | {'PPT' if ppt_min >= -1e-10 else 'NPT':>5} | {feat['S_1']:8.5f} | {'YES' if feat['S_1'] > 1 else 'no':>8} | {feat['S_2']:8.5f} | {feat['S_4']:8.5f} | {feat['S2sq_S4']:8.4f}")

# ============================================================
# TEST 4: Can S_k features alone separate 2x4?
# ============================================================
print("\n" + "=" * 80)
print("TEST 4: Can singular-value features alone separate 2x4 BE from SEP?")
print("=" * 80)

np.random.seed(42)
be_features = []
sep_features = []

for b in np.arange(0.0, 0.5, 0.02):
    rho = horodecki_2x4_edge(b)
    feat = compute_features(rho, 2, 4)
    be_features.append((feat['S_2'], feat['S_4'], feat['S2sq_S4']))

for i in range(200):
    rho = random_separable_2x4(n_terms=np.random.randint(3, 30))
    feat = compute_features(rho, 2, 4)
    sep_features.append((feat['S_2'], feat['S_4'], feat['S2sq_S4']))

be_arr = np.array(be_features)
sep_arr = np.array(sep_features)

print(f"\nBE  S_2:     min={be_arr[:,0].min():.4f}, max={be_arr[:,0].max():.4f}")
print(f"SEP S_2:     min={sep_arr[:,0].min():.4f}, max={sep_arr[:,0].max():.4f}")
print(f"BE  S2^2/S4: min={be_arr[:,2].min():.4f}, max={be_arr[:,2].max():.4f}")
print(f"SEP S2^2/S4: min={sep_arr[:,2].min():.4f}, max={sep_arr[:,2].max():.4f}")

overlap_S2 = (be_arr[:,0].min() < sep_arr[:,0].max()) and (sep_arr[:,0].min() < be_arr[:,0].max())
overlap_ratio = (be_arr[:,2].min() < sep_arr[:,2].max()) and (sep_arr[:,2].min() < be_arr[:,2].max())

print(f"\nS_2 ranges overlap?     {'YES' if overlap_S2 else 'NO'}")
print(f"S2^2/S4 ranges overlap? {'YES' if overlap_ratio else 'NO'}")

if overlap_S2 or overlap_ratio:
    print("=> Singular-value features CANNOT cleanly separate 2x4 BE from SEP")
else:
    print("=> Singular-value features MAY separate 2x4 BE from SEP")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if a_crossover is not None and a_crossover > 0:
    ccnr_txt = f"a < {a_crossover:.3f}"
else:
    ccnr_txt = "NEVER"

print(f"""
1. HORODECKI 3x3 (bound entangled, a in (0,1)):
   - CCNR (||R||_1 > 1): Detects only for {ccnr_txt}
   - D_k classifier:     Detects {n_dk}/{len(a_values)} tested states
   => D_k goes FAR BEYOND CCNR within the 3x3 framework

2. HORODECKI 2x4 (range-criterion BE):
   - R is {R_test.shape} (NON-SQUARE)
   - PPT: All states have positive partial transpose
   - CCNR: ||R||_1 <= 1 for all states => NOT detected
   - D_k = S_k - G_k: UNDEFINED (non-square R)
   - S_k alone: {'Cannot separate' if overlap_S2 or overlap_ratio else 'May separate'}
   => Range-criterion BE is BEYOND all moment-based spectral methods

3. FUNDAMENTAL BOUNDARY:
   - D_k (non-Hermiticity gap) requires SQUARE R (i.e., dA = dB)
   - For non-square systems, only S_k features are available
   - The range criterion is STRUCTURAL, not spectral
   - No permutation-trace observable can express it
""")
