"""
Compute negativity values from raw, corrected, and physical moments for LaTeX tables.
"""
import sys
sys.path.insert(0, '.')

from negativity_si.analysis import compute_negativity_newton_girard
from negativity_si.analysis_qubit_qutrit import compute_negativity_qubit_qutrit

# ============================================================================
# 2x3 (Qubit-Qutrit) System Data
# ============================================================================

# Raw moments for 2x3 states
raw_23 = {
    'param_theta_0.000': [0.757, 0.836, 0.627, 0.459, 0.373],
    'param_theta_0.524': [0.785, 0.733, 0.526, 0.395, 0.357],
    'param_theta_0.785': [0.757, 0.730, 0.368, 0.243, 0.224],
    'param_theta_1.047': [0.813, 0.711, 0.209, 0.137, 0.108],
    'param_theta_1.571': [0.798, 0.749, 0.153, 0.050, 0.045],
    'param_theta_12_0.000': [0.738, 0.805, 0.628, 0.437, 0.338],
    'param_theta_12_0.524': [0.771, 0.770, 0.460, 0.389, 0.341],
    'param_theta_12_0.785': [0.823, 0.758, 0.396, 0.214, 0.240],
    'param_theta_12_1.047': [0.802, 0.768, 0.187, 0.190, 0.105],
    'param_theta_12_1.571': [0.799, 0.760, 0.144, 0.103, 0.071],
    'product_00': [0.781, 0.803, 0.677, 0.575, 0.475],
    'product_01': [0.737, 0.843, 0.668, 0.564, 0.455],
    'product_02': [0.788, 0.843, 0.661, 0.537, 0.409],
    'product_10': [0.776, 0.848, 0.672, 0.493, 0.406],
    'product_11': [0.733, 0.796, 0.630, 0.458, 0.396],
    'product_12': [0.730, 0.806, 0.735, 0.457, 0.391],
}

# Corrected moments for 2x3 states
corrected_23 = {
    'param_theta_0.000': [0.946, 0.994, 0.996, 0.900, 0.982],
    'param_theta_0.524': [0.981, 0.872, 0.835, 0.775, 0.940],
    'param_theta_0.785': [0.946, 0.869, 0.584, 0.476, 0.590],
    'param_theta_1.047': [1.016, 0.846, 0.332, 0.269, 0.284],
    'param_theta_1.571': [0.997, 0.891, 0.243, 0.098, 0.118],
    'param_theta_12_0.000': [0.922, 0.958, 0.997, 0.857, 0.890],
    'param_theta_12_0.524': [0.964, 0.916, 0.730, 0.763, 0.898],
    'param_theta_12_0.785': [1.029, 0.902, 0.629, 0.420, 0.632],
    'param_theta_12_1.047': [1.002, 0.914, 0.297, 0.373, 0.276],
    'param_theta_12_1.571': [0.999, 0.904, 0.229, 0.202, 0.187],
    'product_00': [0.976, 0.955, 1.075, 1.128, 1.251],
    'product_01': [0.921, 1.003, 1.061, 1.106, 1.198],
    'product_02': [0.985, 1.003, 1.050, 1.053, 1.077],
    'product_10': [0.970, 1.009, 1.067, 0.967, 1.069],
    'product_11': [0.916, 0.947, 1.000, 0.898, 1.043],
    'product_12': [0.913, 0.959, 1.167, 0.896, 1.030],
}

# Physical moments for 2x3 states
physical_23 = {
    'param_theta_0.000': [0.945, 0.993, 0.985, 0.929, 0.874],
    'param_theta_0.524': [0.981, 0.872, 0.835, 0.775, 0.718],
    'param_theta_0.785': [0.946, 0.869, 0.584, 0.476, 0.389],
    'param_theta_1.047': [1.000, 0.846, 0.332, 0.269, 0.218],
    'param_theta_1.571': [0.997, 0.891, 0.243, 0.098, 0.039],
    'param_theta_12_0.000': [0.939, 0.967, 0.975, 0.908, 0.843],
    'param_theta_12_0.524': [0.964, 0.916, 0.730, 0.649, 0.578],
    'param_theta_12_0.785': [1.000, 0.902, 0.629, 0.420, 0.280],
    'param_theta_12_1.047': [1.000, 0.914, 0.297, 0.228, 0.175],
    'param_theta_12_1.571': [0.999, 0.904, 0.229, 0.092, 0.037],
    'product_00': [0.976, 0.955, 0.935, 0.914, 0.893],
    'product_01': [0.972, 0.974, 0.946, 0.919, 0.892],
    'product_02': [0.979, 0.982, 0.963, 0.944, 0.925],
    'product_10': [0.963, 0.986, 0.952, 0.918, 0.885],
    'product_11': [0.945, 0.963, 0.929, 0.896, 0.863],
    'product_12': [0.942, 0.955, 0.916, 0.877, 0.839],
}

# ============================================================================
# 2x2 (Qubit-Qubit) System Data
# ============================================================================

# Raw moments for 2x2 states
raw_22 = {
    'bell_phi_minus': [0.820, 0.785, 0.191],
    'bell_phi_plus': [0.806, 0.795, 0.537],
    'bell_psi_minus': [0.807, 0.859, 0.311],
    'param_theta_0.000': [0.823, 0.898, 0.690],
    'param_theta_0.524': [0.823, 0.802, 0.521],
    'param_theta_0.785': [0.842, 0.790, 0.544],
    'param_theta_1.047': [0.828, 0.875, 0.315],
    'product_00': [0.720, 0.857, 0.618],
    'product_01': [0.728, 0.816, 0.667],
    'product_10': [0.711, 0.847, 0.619],
    'product_11': [0.809, 0.905, 0.664],
}

# Corrected moments for 2x2 states
corrected_22 = {
    'bell_phi_minus': [0.990, 0.936, 0.304],
    'bell_phi_plus': [0.973, 0.948, 0.855],
    'bell_psi_minus': [0.974, 1.024, 0.495],
    'param_theta_0.000': [0.994, 1.071, 1.099],
    'param_theta_0.524': [0.994, 0.956, 0.830],
    'param_theta_0.785': [1.017, 0.942, 0.866],
    'param_theta_1.047': [1.000, 1.044, 0.501],
    'product_00': [0.870, 1.022, 0.984],
    'product_01': [0.879, 0.973, 1.062],
    'product_10': [0.859, 1.010, 0.986],
    'product_11': [0.977, 1.079, 1.058],
}

# Physical moments for 2x2 states
physical_22 = {
    'bell_phi_minus': [0.985, 0.955, 0.895],
    'bell_phi_plus': [0.973, 0.948, 0.898],
    'bell_psi_minus': [0.974, 0.950, 0.900],
    'param_theta_0.000': [0.995, 0.990, 0.980],
    'param_theta_0.524': [0.994, 0.956, 0.830],
    'param_theta_0.785': [1.000, 0.942, 0.828],
    'param_theta_1.047': [1.000, 0.913, 0.750],
    'product_00': [0.926, 0.878, 0.813],
    'product_01': [0.940, 0.898, 0.843],
    'product_10': [0.918, 0.862, 0.791],
    'product_11': [0.943, 0.902, 0.852],
}

# ============================================================================
# Compute Negativity Values
# ============================================================================

print("=" * 80)
print("2×3 (Qubit-Qutrit) System")
print("=" * 80)

state_order_23 = [
    'param_theta_0.000', 'param_theta_0.524', 'param_theta_0.785',
    'param_theta_1.047', 'param_theta_1.571',
    'param_theta_12_0.000', 'param_theta_12_0.524', 'param_theta_12_0.785',
    'param_theta_12_1.047', 'param_theta_12_1.571',
    'product_00', 'product_01', 'product_02', 'product_10', 'product_11', 'product_12'
]

for state in state_order_23:
    mu_raw = raw_23[state]
    mu_corr = corrected_23[state]
    mu_phys = physical_23[state]

    N_raw = compute_negativity_qubit_qutrit(mu_raw[0], mu_raw[1], mu_raw[2], mu_raw[3], mu_raw[4])
    N_corr = compute_negativity_qubit_qutrit(mu_corr[0], mu_corr[1], mu_corr[2], mu_corr[3], mu_corr[4])
    N_phys = compute_negativity_qubit_qutrit(mu_phys[0], mu_phys[1], mu_phys[2], mu_phys[3], mu_phys[4])

    print(f"{state:25s} & {N_raw:.3f} & {N_corr:.3f} & {N_phys:.3f} \\\\")

print()
print("=" * 80)
print("2×2 (Qubit-Qubit) System")
print("=" * 80)

state_order_22 = [
    'bell_phi_minus', 'bell_phi_plus', 'bell_psi_minus',
    'param_theta_0.000', 'param_theta_0.524', 'param_theta_0.785', 'param_theta_1.047',
    'product_00', 'product_01', 'product_10', 'product_11'
]

for state in state_order_22:
    mu_raw = raw_22[state]
    mu_corr = corrected_22[state]
    mu_phys = physical_22[state]

    N_raw = compute_negativity_newton_girard(mu_raw[0], mu_raw[1], mu_raw[2])
    N_corr = compute_negativity_newton_girard(mu_corr[0], mu_corr[1], mu_corr[2])
    N_phys = compute_negativity_newton_girard(mu_phys[0], mu_phys[1], mu_phys[2])

    print(f"{state:25s} & {N_raw:.3f} & {N_corr:.3f} & {N_phys:.3f} \\\\")
