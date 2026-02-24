"""
Two-feature SVM classifier for bound entanglement detection.
Features: D_4 = S_4 - G_4 and S_2^2/S_4
"""

import numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

def realignment(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)

def partial_transpose(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)

def is_ppt(rho, dA, dB):
    rho_pt = partial_transpose(rho, dA, dB)
    eigenvalues = linalg.eigvalsh(rho_pt)
    return np.all(eigenvalues >= -1e-10)

def realignment_criterion(rho, dA, dB):
    R = realignment(rho, dA, dB)
    trace_norm = np.sum(linalg.svdvals(R))
    return trace_norm <= 1 + 1e-10

def compute_features(rho, dA, dB):
    R = realignment(rho, dA, dB)
    svd_vals = linalg.svdvals(R)
    S_2 = np.sum(svd_vals ** 2)
    S_4 = np.sum(svd_vals ** 4)
    if R.shape[0] == R.shape[1]:
        R_eigvals = linalg.eigvals(R)
        G_4 = np.real(np.sum(R_eigvals ** 4))
    else:
        G_4 = S_4
    D_4 = S_4 - G_4
    S2sq_S4 = S_2**2 / S_4 if S_4 > 1e-10 else 1.0
    return {'D_4': D_4, 'S2sq_S4': S2sq_S4}

def random_density_matrix(dim, rng):
    G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = G @ G.conj().T
    return rho / np.trace(rho)

def random_separable_3x3(rng, n_terms=None):
    if n_terms is None:
        n_terms = rng.integers(2, 8)
    weights = rng.random(n_terms)
    weights /= weights.sum()
    rho = np.zeros((9, 9), dtype=complex)
    for w in weights:
        rhoA = random_density_matrix(3, rng)
        rhoB = random_density_matrix(3, rng)
        rho += w * np.kron(rhoA, rhoB)
    return rho

def horodecki_3x3(a):
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = a; rho[0, 8] = a; rho[1, 1] = a; rho[2, 2] = a
    rho[3, 3] = a; rho[4, 4] = a; rho[5, 5] = a
    rho[6, 6] = (1 + a) / 2; rho[7, 7] = (1 + a) / 2
    rho[8, 0] = a; rho[8, 8] = a
    sqrt_term = np.sqrt(1 - a**2) / 2
    rho[2, 6] = sqrt_term; rho[6, 2] = sqrt_term
    rho[5, 7] = sqrt_term; rho[7, 5] = sqrt_term
    return rho / np.trace(rho)

def tiles_3x3_upb():
    dim = 9
    upb_states = []
    psi1 = np.zeros(dim, dtype=complex); psi1[0] = 1/np.sqrt(2); psi1[1] = -1/np.sqrt(2)
    upb_states.append(psi1)
    psi2 = np.zeros(dim, dtype=complex); psi2[7] = 1/np.sqrt(2); psi2[8] = -1/np.sqrt(2)
    upb_states.append(psi2)
    psi3 = np.zeros(dim, dtype=complex); psi3[2] = 1/np.sqrt(2); psi3[5] = -1/np.sqrt(2)
    upb_states.append(psi3)
    psi4 = np.zeros(dim, dtype=complex); psi4[3] = 1/np.sqrt(2); psi4[6] = -1/np.sqrt(2)
    upb_states.append(psi4)
    psi5 = np.ones(dim, dtype=complex) / 3.0
    upb_states.append(psi5)
    return upb_states

def tiles_bound_entangled_3x3(rng=None, noise=0.0):
    dim = 9
    upb_states = tiles_3x3_upb()
    P_upb = np.zeros((dim, dim), dtype=complex)
    for psi in upb_states:
        P_upb += np.outer(psi, psi.conj())
    P_complement = np.eye(dim, dtype=complex) - P_upb
    if noise > 0 and rng is not None:
        P_complement = (1 - noise) * P_complement + noise * np.eye(dim) / dim
    eigvals, eigvecs = np.linalg.eigh(P_complement)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho / np.trace(rho)

if __name__ == '__main__':
    print("=" * 60)
    print("TWO-FEATURE SVM CLASSIFIER FOR BOUND ENTANGLEMENT")
    print("=" * 60)

    rng = np.random.default_rng(42)
    dA, dB = 3, 3

    print("\nGenerating dataset...")
    sep_data = []
    be_data = []

    for _ in range(3000):
        rho = random_separable_3x3(rng)
        if is_ppt(rho, dA, dB) and realignment_criterion(rho, dA, dB):
            sep_data.append(compute_features(rho, dA, dB))

    for a in np.linspace(0.01, 0.99, 500):
        try:
            rho = horodecki_3x3(a)
            if is_ppt(rho, dA, dB) and realignment_criterion(rho, dA, dB):
                be_data.append(compute_features(rho, dA, dB))
        except:
            pass

    for _ in range(1000):
        a = rng.uniform(0.01, 0.99)
        rho = horodecki_3x3(a)
        noise = rng.uniform(0.01, 0.25)
        rho = (1 - noise) * rho + noise * np.eye(9) / 9
        rho = rho / np.trace(rho)
        if is_ppt(rho, dA, dB) and realignment_criterion(rho, dA, dB):
            be_data.append(compute_features(rho, dA, dB))

    for _ in range(1000):
        noise = rng.random() * 0.25
        rho = tiles_bound_entangled_3x3(rng, noise=noise)
        if is_ppt(rho, dA, dB) and realignment_criterion(rho, dA, dB):
            be_data.append(compute_features(rho, dA, dB))

    print(f"  Separable states: {len(sep_data)}")
    print(f"  Bound entangled:  {len(be_data)}")

    sep_D4 = np.array([d['D_4'] for d in sep_data])
    be_D4 = np.array([d['D_4'] for d in be_data])
    sep_ratio = np.array([d['S2sq_S4'] for d in sep_data])
    be_ratio = np.array([d['S2sq_S4'] for d in be_data])

    all_labels = np.array([0] * len(sep_D4) + [1] * len(be_D4))

    X = np.column_stack([
        np.concatenate([sep_D4, be_D4]),
        np.concatenate([sep_ratio, be_ratio])
    ])
    y = all_labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel='linear', C=10.0)
    model.fit(X_scaled, y)

    # Convert to original scale
    w = model.coef_[0] / scaler.scale_
    b = model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_)

    print("\n" + "=" * 60)
    print("LINEAR SVM DECISION BOUNDARY")
    print("=" * 60)
    print(f"\nFeatures: D_4, S_2^2/S_4")
    print(f"\nDecision boundary (original scale):")
    print(f"  {w[0]:.6f} * D_4 + {w[1]:.6f} * (S_2^2/S_4) + {b:.6f} > 0  -->  BE")

    print(f"\nSimplified criterion:")
    print(f"  Classify as BOUND ENTANGLED if:")
    if w[0] < 0:
        print(f"    D_4 < {-b/w[0]:.8f} + {-w[1]/w[0]:.6f} * (S_2^2/S_4)")
    else:
        print(f"    D_4 > {-b/w[0]:.8f} + {-w[1]/w[0]:.6f} * (S_2^2/S_4)")

    y_pred = model.predict(X_scaled)
    print(f"\nPerformance:")
    print(f"  Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y, y_pred):.4f}")

    print(f"\nFeature statistics:")
    print(f"  Separable - D_4:       [{sep_D4.min():.8f}, {sep_D4.max():.8f}]")
    print(f"  BE        - D_4:       [{be_D4.min():.8f}, {be_D4.max():.8f}]")
    print(f"  Separable - S_2^2/S_4: [{sep_ratio.min():.6f}, {sep_ratio.max():.6f}]")
    print(f"  BE        - S_2^2/S_4: [{be_ratio.min():.6f}, {be_ratio.max():.6f}]")

    print("\n" + "=" * 60)
    print("FINAL TWO-FEATURE CRITERION")
    print("=" * 60)
    print(f"""
For a 3x3 PPT state rho with realignment matrix R(rho):

  D_4 = S_4 - G_4 = sum_i(sigma_i^4) - Re(sum_i(lambda_i^4))

where sigma_i are singular values and lambda_i are eigenvalues of R(rho).

CRITERION: A PPT state is BOUND ENTANGLED if:
""")
    if w[0] < 0:
        print(f"    D_4 < {-b/w[0]:.6f} + {-w[1]/w[0]:.4f} * (S_2^2/S_4)")
    else:
        print(f"    D_4 > {-b/w[0]:.6f} + {-w[1]/w[0]:.4f} * (S_2^2/S_4)")
