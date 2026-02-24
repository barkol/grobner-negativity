"""Check if the 4-feature SVM catches small-a Horodecki states."""
import numpy as np
from scipy import linalg

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

def realignment(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA**2, dB**2)

def compute_all_features(rho, dA=3, dB=3):
    R = realignment(rho, dA, dB)
    svd_vals = linalg.svdvals(R)
    R_eigvals = linalg.eigvals(R)

    S_2 = np.sum(svd_vals**2)
    S_4 = np.sum(svd_vals**4)
    G_2 = np.real(np.sum(R_eigvals**2))
    G_4 = np.real(np.sum(R_eigvals**4))

    D_2 = S_2 - G_2
    D_4 = S_4 - G_4
    S2sq_S4 = S_2**2 / S_4

    return {'D_4': D_4, 'D_2': D_2, 'G_2': G_2, 'S2sq_S4': S2sq_S4,
            'S_2': S_2, 'S_4': S_4, 'G_4': G_4}

# Print full feature vectors for small-a states
print("Full 4-feature vectors for Horodecki 3x3 at small a:")
print(f"{'a':>6} | {'D_4':>12} | {'D_2':>12} | {'G_2':>8} | {'S2^2/S4':>8} | {'2-feat':>8}")
print("-" * 72)

for a in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
    feat = compute_all_features(horodecki_3x3(a))
    thresh = -0.026 + 0.022 * feat['S2sq_S4']
    is_be_2feat = feat['D_4'] < thresh
    print(f"{a:6.2f} | {feat['D_4']:12.6e} | {feat['D_2']:12.6e} | {feat['G_2']:8.5f} | {feat['S2sq_S4']:8.4f} | {'BE' if is_be_2feat else 'SEP':>8}")

# Now compare with typical separable state features
print("\nTypical separable 3x3 features (for reference):")
np.random.seed(42)
sep_feats = []
for i in range(500):
    rho = np.zeros((9, 9), dtype=complex)
    for _ in range(np.random.randint(3, 30)):
        a = np.random.randn(3) + 1j*np.random.randn(3); a /= np.linalg.norm(a)
        b = np.random.randn(3) + 1j*np.random.randn(3); b /= np.linalg.norm(b)
        psi = np.kron(a, b)
        rho += np.outer(psi, psi.conj())
    rho = rho / np.trace(rho)
    sep_feats.append(compute_all_features(rho))

sep_D4 = [f['D_4'] for f in sep_feats]
sep_D2 = [f['D_2'] for f in sep_feats]
sep_G2 = [f['G_2'] for f in sep_feats]
sep_ratio = [f['S2sq_S4'] for f in sep_feats]

print(f"D_4:     min={min(sep_D4):.6e}, max={max(sep_D4):.6e}, mean={np.mean(sep_D4):.6e}")
print(f"D_2:     min={min(sep_D2):.6e}, max={max(sep_D2):.6e}, mean={np.mean(sep_D2):.6e}")
print(f"G_2:     min={min(sep_G2):.5f}, max={max(sep_G2):.5f}, mean={np.mean(sep_G2):.5f}")
print(f"S2^2/S4: min={min(sep_ratio):.4f}, max={max(sep_ratio):.4f}, mean={np.mean(sep_ratio):.4f}")

# Check: can D_2 alone separate a=0.05?
print(f"\na=0.05 state: D_2 = {compute_all_features(horodecki_3x3(0.05))['D_2']:.6e}")
print(f"Separable D_2 range: [{min(sep_D2):.6e}, {max(sep_D2):.6e}]")
a05_D2 = compute_all_features(horodecki_3x3(0.05))['D_2']
in_sep_range = min(sep_D2) <= a05_D2 <= max(sep_D2)
print(f"a=0.05 D_2 falls within separable range? {in_sep_range}")

# For comparison: the SVM uses D_4, D_2, G_2, S2^2/S4
# Let's check if a=0.05 is separable-looking in ALL features
print("\nFull comparison of a=0.05 with separable statistics:")
f05 = compute_all_features(horodecki_3x3(0.05))
for key in ['D_4', 'D_2', 'G_2', 'S2sq_S4']:
    vals = [f[key] for f in sep_feats]
    v = f05[key]
    within = min(vals) <= v <= max(vals)
    pctile = sum(1 for x in vals if x <= v) / len(vals) * 100
    print(f"  {key:>8}: a=0.05={v:.6e}, sep range=[{min(vals):.4e}, {max(vals):.4e}], within={within}, percentile={pctile:.1f}%")

# Check which a values the full SVM training set uses
print("\n\nHorodecki parameters in the standard training set: a in {0.10, 0.15, ..., 0.90}")
print("The a=0.05 state was NOT in the training/test set.")
print("\nFor a=0.05, the state is very close to maximally mixed (trace distance to I/9):")
rho05 = horodecki_3x3(0.05)
td = 0.5 * np.sum(np.abs(linalg.eigvalsh(rho05 - np.eye(9)/9)))
print(f"  ||rho(0.05) - I/9||_1 / 2 = {td:.4f}")
rho10 = horodecki_3x3(0.10)
td10 = 0.5 * np.sum(np.abs(linalg.eigvalsh(rho10 - np.eye(9)/9)))
print(f"  ||rho(0.10) - I/9||_1 / 2 = {td10:.4f}")
print(f"  ||I/9 - I/9||_1 / 2 = 0.0000 (maximally mixed = separable)")
