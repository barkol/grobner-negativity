"""
Test whether (RR^T)^{k/2} can detect BE in non-square systems (2x4).

Key insight: For square R, the D_k gap compares eigenvalues vs singular values.
For non-square R, eigenvalues of R are undefined, but:
  - RR^T is always square (dA^2 x dA^2), Hermitian PSD
  - R^T R is always square (dB^2 x dB^2), Hermitian PSD

Question: Does Tr[(RR^T)^{k/2}] provide information beyond S_k = sum sigma_i^k?

We also explore alternative constructions:
  1. Block dilation: M = [[0, R], [R^T, 0]] (NOT Hermitian if R is complex)
  2. Asymmetric block: M = [[0, R], [conj(R), 0]] - Hermitian
  3. Compare RR^T with R^T R eigenvalue distributions
  4. Cross-moments: Tr[(RR^T)^a (R^T R)^b] type quantities
"""
import numpy as np
from scipy import linalg

# ============================================================
# State constructors (same as before)
# ============================================================

def horodecki_3x3(a):
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = a;  rho[1, 1] = a;  rho[2, 2] = a
    rho[3, 3] = a;  rho[4, 4] = a;  rho[5, 5] = a
    rho[6, 6] = (1 + a) / 2;  rho[7, 7] = (1 + a) / 2;  rho[8, 8] = a
    rho[0, 8] = a;  rho[8, 0] = a
    c = np.sqrt(1 - a**2) / 2
    rho[2, 6] = c;  rho[6, 2] = c
    rho[5, 7] = c;  rho[7, 5] = c
    return rho / np.trace(rho)

def horodecki_2x4_edge(b_param):
    """PPT entangled state in C2 x C4."""
    dA, dB = 2, 4
    dim = dA * dB
    phi1 = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
    phi2 = np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=complex) / np.sqrt(2)
    phi3 = np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=complex) / np.sqrt(2)
    phi4 = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=complex) / np.sqrt(2)
    rho = np.zeros((dim, dim), dtype=complex)
    for phi in [phi1, phi2, phi3, phi4]:
        rho += np.outer(phi, phi.conj())
    rho = (1 - b_param) * rho / 4 + b_param * np.eye(dim) / dim
    return rho / np.trace(rho)

def random_separable(dA, dB, n_terms=10):
    dim = dA * dB
    rho = np.zeros((dim, dim), dtype=complex)
    for _ in range(n_terms):
        a = np.random.randn(dA) + 1j*np.random.randn(dA); a /= np.linalg.norm(a)
        b = np.random.randn(dB) + 1j*np.random.randn(dB); b /= np.linalg.norm(b)
        psi = np.kron(a, b)
        rho += np.outer(psi, psi.conj())
    return rho / np.trace(rho)

def realignment(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA**2, dB**2)

# ============================================================
# TEST 0: Verify that Tr[(RR^T)^{k/2}] = S_k
# ============================================================
print("=" * 80)
print("TEST 0: Does Tr[(RR^T)^{k/2}] = S_k?")
print("=" * 80)

rho = horodecki_2x4_edge(0.1)
R = realignment(rho, 2, 4)
print(f"R shape: {R.shape}")

svd_vals = linalg.svdvals(R)
RRT = R @ R.conj().T  # 4x4 Hermitian PSD
RTR = R.conj().T @ R  # 16x16 Hermitian PSD

print(f"\nRR^T shape: {RRT.shape}, eigenvalues: {np.sort(np.real(linalg.eigvalsh(RRT)))[::-1][:6]}")
print(f"R^T R shape: {RTR.shape}, non-zero eigenvalues: {np.sort(np.real(linalg.eigvalsh(RTR)))[::-1][:6]}")
print(f"Singular values of R: {np.sort(svd_vals)[::-1][:6]}")

for k in [2, 4, 6]:
    S_k = np.sum(svd_vals**k)
    RRT_k = np.real(np.trace(np.linalg.matrix_power(RRT, k//2)))
    print(f"\nk={k}: S_k = {S_k:.10f},  Tr[(RR^T)^{k//2}] = {RRT_k:.10f},  equal? {np.isclose(S_k, RRT_k)}")

print("\n>>> CONCLUSION: Tr[(RR^T)^{k/2}] = S_k exactly.")
print(">>> (RR^T)^{k/2} provides NO NEW information beyond singular value moments.\n")

# ============================================================
# TEST 1: Explore alternative non-Hermiticity measures for non-square R
# ============================================================
print("=" * 80)
print("TEST 1: Alternative non-Hermiticity measures for non-square R")
print("=" * 80)

def compute_alt_features(rho, dA, dB):
    """Compute alternative spectral features for non-square R."""
    R = realignment(rho, dA, dB)
    svd_vals = linalg.svdvals(R)

    S_2 = np.sum(svd_vals**2)
    S_4 = np.sum(svd_vals**4)
    S2sq_S4 = S_2**2 / S_4 if S_4 > 1e-12 else 1.0

    # RR^T is dA^2 x dA^2 (always square, Hermitian PSD)
    RRT = R @ R.conj().T
    # R^T R is dB^2 x dB^2 (always square, Hermitian PSD)
    RTR = R.conj().T @ R

    # Alternative 1: Block dilation M = [[0, R], [R^T, 0]]
    # This is NOT Hermitian for complex R; R^T != R^dagger
    m, n = R.shape
    M_block = np.zeros((m + n, m + n), dtype=complex)
    M_block[:m, m:] = R
    M_block[m:, :m] = R.T  # transpose, NOT conjugate transpose

    M_block_eigs = linalg.eigvals(M_block)
    M_block_svds = linalg.svdvals(M_block)

    # Non-Hermiticity of block dilation
    G2_block = np.real(np.sum(M_block_eigs**2))
    G4_block = np.real(np.sum(M_block_eigs**4))
    S2_block = np.sum(M_block_svds**2)
    S4_block = np.sum(M_block_svds**4)
    D2_block = S2_block - G2_block
    D4_block = S4_block - G4_block

    # Alternative 2: Hermitian block M_H = [[0, R], [R^dagger, 0]]
    M_herm = np.zeros((m + n, m + n), dtype=complex)
    M_herm[:m, m:] = R
    M_herm[m:, :m] = R.conj().T
    # This IS Hermitian, eigenvalues = ±sigma_i
    M_herm_eigs = np.sort(np.real(linalg.eigvalsh(M_herm)))[::-1]

    # Alternative 3: Compare R with R^T in terms of how "symmetric" R is
    # For square R: ||R - R^T||_F measures asymmetry
    # For non-square R: can compare R with its "best symmetric approximation"
    # via SVD: R = U Sigma V^T, "symmetry" ~ alignment of U and V
    U, s, Vt = linalg.svd(R, full_matrices=False)
    # Overlap between left and right singular vectors (requires same dimension)
    # For non-square, U is m x min(m,n), V is n x min(m,n)
    # Can compare via: how close is R to being "factorizable" as A A^T?

    # Alternative 4: Frobenius norm asymmetry of RR^T vs R^T R
    # These share non-zero eigenvalues, but differ in size
    # Spectral entropy of RR^T vs R^T R
    eigs_RRT = np.real(linalg.eigvalsh(RRT))
    eigs_RRT = eigs_RRT[eigs_RRT > 1e-12]
    eigs_RTR = np.real(linalg.eigvalsh(RTR))
    eigs_RTR = eigs_RTR[eigs_RTR > 1e-12]

    # Entropy of singular value distribution
    p = svd_vals**2 / S_2
    sv_entropy = -np.sum(p * np.log(p + 1e-15))

    # Alternative 5: R - R^dagger for square subsystems
    # Consider the "partial Hermiticity" via Schur complement or similar

    return {
        'S_2': S_2, 'S_4': S_4, 'S2sq_S4': S2sq_S4,
        'D2_block': D2_block, 'D4_block': D4_block,
        'sv_entropy': sv_entropy,
        'n_nonzero_sv': len(eigs_RRT),
    }

# Test on 2x4 states
print("\nBlock dilation M = [[0,R],[R^T,0]] non-Hermiticity gap:")
print(f"{'Type':>8} | {'param':>6} | {'D2_block':>12} | {'D4_block':>12} | {'SV entropy':>10} | {'S2^2/S4':>8} | {'#sv>0':>6}")
print("-" * 80)

# Horodecki 2x4 BE states
for b in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
    rho = horodecki_2x4_edge(b)
    feat = compute_alt_features(rho, 2, 4)
    print(f"{'BE':>8} | {b:6.2f} | {feat['D2_block']:12.6e} | {feat['D4_block']:12.6e} | {feat['sv_entropy']:10.6f} | {feat['S2sq_S4']:8.4f} | {feat['n_nonzero_sv']:>6}")

print("-" * 80)

# Separable 2x4 states
np.random.seed(42)
sep_d2 = []
sep_d4 = []
for i in range(10):
    rho = random_separable(2, 4, n_terms=np.random.randint(3, 30))
    feat = compute_alt_features(rho, 2, 4)
    sep_d2.append(feat['D2_block'])
    sep_d4.append(feat['D4_block'])
    if i < 6:
        print(f"{'SEP':>8} | {i:6d} | {feat['D2_block']:12.6e} | {feat['D4_block']:12.6e} | {feat['sv_entropy']:10.6f} | {feat['S2sq_S4']:8.4f} | {feat['n_nonzero_sv']:>6}")

# Extended test: 200 separable states
for i in range(190):
    rho = random_separable(2, 4, n_terms=np.random.randint(3, 30))
    feat = compute_alt_features(rho, 2, 4)
    sep_d2.append(feat['D2_block'])
    sep_d4.append(feat['D4_block'])

print(f"\nSeparable D2_block: min={min(sep_d2):.6e}, max={max(sep_d2):.6e}")
print(f"Separable D4_block: min={min(sep_d4):.6e}, max={max(sep_d4):.6e}")

# Check for each BE state if its block D_k falls outside separable range
print("\nDoes block dilation D_k separate BE from SEP?")
for b in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
    rho = horodecki_2x4_edge(b)
    feat = compute_alt_features(rho, 2, 4)
    d2_sep = feat['D2_block'] >= min(sep_d2) and feat['D2_block'] <= max(sep_d2)
    d4_sep = feat['D4_block'] >= min(sep_d4) and feat['D4_block'] <= max(sep_d4)
    print(f"  b={b:.2f}: D2_block={feat['D2_block']:.6e} (in SEP range: {d2_sep}), D4_block={feat['D4_block']:.6e} (in SEP range: {d4_sep})")


# ============================================================
# TEST 2: The real question - does R^T differ from R^dagger for R non-square?
# ============================================================
print("\n" + "=" * 80)
print("TEST 2: Key insight - R^T vs R^dagger for realignment matrix")
print("=" * 80)

print("""
For the block dilation M = [[0, R], [R^T, 0]]:
- If R is real: M is symmetric, eigenvalues are real, D_k = 0
- If R is complex: M is NOT Hermitian (R^T != R^dagger)
  => eigenvalues can be complex => D_k != 0

Key: The non-Hermiticity of M comes from Im(R), i.e., the imaginary part
of the realignment matrix. This is a new type of asymmetry measure!
""")

# Check: how much imaginary content does R have?
for label, rho_fn, params in [
    ("BE b=0.0", lambda: horodecki_2x4_edge(0.0), {}),
    ("BE b=0.1", lambda: horodecki_2x4_edge(0.1), {}),
    ("BE b=0.3", lambda: horodecki_2x4_edge(0.3), {}),
]:
    rho = rho_fn()
    R = realignment(rho, 2, 4)
    im_frac = np.linalg.norm(np.imag(R)) / np.linalg.norm(R)
    print(f"  {label}: ||Im(R)||/||R|| = {im_frac:.6f}")

np.random.seed(42)
for i in range(3):
    rho = random_separable(2, 4, n_terms=np.random.randint(5, 20))
    R = realignment(rho, 2, 4)
    im_frac = np.linalg.norm(np.imag(R)) / np.linalg.norm(R)
    print(f"  SEP {i}: ||Im(R)||/||R|| = {im_frac:.6f}")

# ============================================================
# TEST 3: Back to basics - what about the 3x3 case? Verify D_k works
# ============================================================
print("\n" + "=" * 80)
print("TEST 3: Block dilation D_k for 3x3 Horodecki (where standard D_k works)")
print("=" * 80)

print(f"{'a':>6} | {'D4_standard':>12} | {'D4_block':>12} | {'D2_standard':>12} | {'D2_block':>12}")
print("-" * 70)

for a in [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
    rho = horodecki_3x3(a)
    R = realignment(rho, 3, 3)

    # Standard D_k
    svd_vals = linalg.svdvals(R)
    R_eigs = linalg.eigvals(R)
    S_2 = np.sum(svd_vals**2); S_4 = np.sum(svd_vals**4)
    G_2 = np.real(np.sum(R_eigs**2)); G_4 = np.real(np.sum(R_eigs**4))
    D2_std = S_2 - G_2; D4_std = S_4 - G_4

    # Block dilation D_k
    feat = compute_alt_features(rho, 3, 3)

    print(f"{a:6.2f} | {D4_std:12.6e} | {feat['D4_block']:12.6e} | {D2_std:12.6e} | {feat['D2_block']:12.6e}")


# ============================================================
# TEST 4: Comprehensive 2x4 test with larger state set
# ============================================================
print("\n" + "=" * 80)
print("TEST 4: Block dilation features - comprehensive 2x4 analysis")
print("=" * 80)

np.random.seed(42)

# Collect features
be_feats = []
for b in np.arange(0.0, 0.60, 0.02):
    rho = horodecki_2x4_edge(b)
    feat = compute_alt_features(rho, 2, 4)
    be_feats.append(feat)

sep_feats = []
for i in range(500):
    rho = random_separable(2, 4, n_terms=np.random.randint(2, 40))
    feat = compute_alt_features(rho, 2, 4)
    sep_feats.append(feat)

# Check each feature for separation power
for key in ['D2_block', 'D4_block', 'sv_entropy', 'S2sq_S4']:
    be_vals = np.array([f[key] for f in be_feats])
    sep_vals = np.array([f[key] for f in sep_feats])

    overlap = (be_vals.min() < sep_vals.max()) and (sep_vals.min() < be_vals.max())

    # Compute overlap fraction
    if overlap:
        lo = max(be_vals.min(), sep_vals.min())
        hi = min(be_vals.max(), sep_vals.max())
        be_in_overlap = np.sum((be_vals >= lo) & (be_vals <= hi)) / len(be_vals)
        sep_in_overlap = np.sum((sep_vals >= lo) & (sep_vals <= hi)) / len(sep_vals)
        print(f"{key:>12}: BE=[{be_vals.min():.4e}, {be_vals.max():.4e}], SEP=[{sep_vals.min():.4e}, {sep_vals.max():.4e}]")
        print(f"             Overlap: {be_in_overlap*100:.0f}% of BE, {sep_in_overlap*100:.0f}% of SEP in overlap region")
    else:
        print(f"{key:>12}: BE=[{be_vals.min():.4e}, {be_vals.max():.4e}], SEP=[{sep_vals.min():.4e}, {sep_vals.max():.4e}]")
        print(f"             NO OVERLAP - separation possible!")


# ============================================================
# FINAL ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

print("""
1. Tr[(RR^T)^{k/2}] = S_k (singular value moments) exactly.
   => NO new information from (RR^T)^{k/2}. Already tested and insufficient.

2. Block dilation M = [[0, R], [R^T, 0]]:
   - For REAL R: M is symmetric => D_k = 0 (useless)
   - For COMPLEX R: M is non-Hermitian => D_k != 0 (potentially useful)
   - The non-Hermiticity comes from Im(R), not from structural BE properties

3. For Horodecki 2x4 states constructed here:
   - R is nearly REAL (small Im(R)) because the state construction uses
     real coefficients => block dilation D_k is very small
   - Separation from separable states is weak or absent

4. FUNDAMENTAL ISSUE:
   The D_k gap for 3x3 square systems works because R being nearly Hermitian
   (R ~ R^dagger) is a PPT-specific structural property.
   For non-square R, "Hermiticity" is undefined, and no block construction
   recovers this information.

   The range criterion tests whether range(rho) is spanned by PRODUCT vectors
   - this is a question about the GEOMETRY of the range, not about spectral
   properties of any derived matrix.
""")
