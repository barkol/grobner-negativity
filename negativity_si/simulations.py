#!/usr/bin/env python3
"""
Simulation Module for Negativity and Chirality Witness Experiments
===================================================================

Provides noise model creation from calibration data and simulation functions
for I2, R2, M2 (chirality witness), and mu_k (negativity) measurements.

State parametrization: |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>
- theta=0: |00> (separable)
- theta=90: Bell state (maximally entangled)
- theta=180: |11> (separable)
"""

import numpy as np
from numpy import sqrt, cos, sin, pi
from pathlib import Path
import csv
from typing import Optional, Dict, List, Tuple, Union

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# =============================================================================
# CALIBRATION DATA PARSING
# =============================================================================

def parse_calibration_csv(filepath: Union[str, Path]) -> Tuple[dict, list]:
    """
    Parse IBM calibration CSV file.

    Args:
        filepath: Path to calibration CSV file

    Returns:
        Tuple of (calibrations dict, coupling_edges list)
    """
    filepath = Path(filepath)
    calibrations = {}
    coupling_edges = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            qubit_id = int(row['Qubit'])

            def safe_float(val, default=0.0):
                try:
                    return float(val) if val else default
                except (ValueError, TypeError):
                    return default

            t1 = safe_float(row.get('T1 (us)', ''))
            t2 = safe_float(row.get('T2 (us)', ''))
            readout_error = safe_float(row.get('Readout assignment error', ''))
            prob_m0_p1 = safe_float(row.get('Prob meas0 prep1', ''))
            prob_m1_p0 = safe_float(row.get('Prob meas1 prep0', ''))
            sx_error = safe_float(row.get('\u221ax (sx) error', ''))
            x_error = safe_float(row.get('Pauli-X error', ''))
            operational = row.get('Operational', 'Yes').lower() == 'yes'

            # Parse CZ errors
            cz_errors = {}
            cz_error_str = row.get('CZ error', '')
            if cz_error_str and ':' in cz_error_str:
                for part in cz_error_str.split(';'):
                    if ':' in part:
                        try:
                            target_str, error_str = part.split(':')
                            target = int(target_str)
                            cz_errors[target] = float(error_str)
                            coupling_edges.add((min(qubit_id, target), max(qubit_id, target)))
                        except (ValueError, IndexError):
                            pass

            calibrations[qubit_id] = {
                'qubit_id': qubit_id,
                't1_us': t1 if t1 > 0 else 100.0,
                't2_us': t2 if t2 > 0 else 50.0,
                'readout_error': readout_error,
                'prob_m0_p1': prob_m0_p1,
                'prob_m1_p0': prob_m1_p0,
                'sx_error': sx_error,
                'x_error': x_error,
                'operational': operational,
                'cz_errors': cz_errors,
            }

    return calibrations, list(coupling_edges)


def create_noise_model_from_calibration(
    calibrations: dict,
    num_qubits: int = 10
) -> 'NoiseModel':
    """
    Create noise model from calibration data.

    Args:
        calibrations: Dictionary of qubit calibration data
        num_qubits: Number of qubits to include in noise model

    Returns:
        NoiseModel for AerSimulator
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for noise model creation")

    # Select best qubits by readout error
    sorted_qubits = sorted(
        [q for q, c in calibrations.items() if c['operational'] and c['t1_us'] > 10],
        key=lambda q: calibrations[q]['readout_error']
    )[:num_qubits]

    noise_model = NoiseModel()
    gate_time_1q = 32e-9  # 32 ns
    gate_time_2q = 68e-9  # 68 ns

    for i, qubit_id in enumerate(sorted_qubits):
        cal = calibrations[qubit_id]

        # Thermal relaxation
        t1 = cal['t1_us'] * 1e-6
        t2 = min(cal['t2_us'] * 1e-6, 2 * t1)

        # Single qubit thermal relaxation
        thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_quantum_error(thermal_1q, ['sx', 'x'], [i])

        # Depolarizing errors (1 qubit)
        sx_err = cal['sx_error'] if cal['sx_error'] > 0 else 0.001
        x_err = cal['x_error'] if cal['x_error'] > 0 else 0.001
        noise_model.add_quantum_error(depolarizing_error(sx_err, 1), 'sx', [i])
        noise_model.add_quantum_error(depolarizing_error(x_err, 1), 'x', [i])

        # Readout error
        p0g1 = cal['prob_m0_p1'] if cal['prob_m0_p1'] > 0 else cal['readout_error']
        p1g0 = cal['prob_m1_p0'] if cal['prob_m1_p0'] > 0 else cal['readout_error']
        if p0g1 > 0 or p1g0 > 0:
            from qiskit_aer.noise import ReadoutError
            read_err = ReadoutError([[1 - p1g0, p1g0], [p0g1, 1 - p0g1]])
            noise_model.add_readout_error(read_err, [i])

    # Two-qubit errors (linear connectivity)
    for i in range(min(num_qubits - 1, len(sorted_qubits) - 1)):
        qubit_id = sorted_qubits[i]
        cal = calibrations[qubit_id]

        t1 = cal['t1_us'] * 1e-6
        t2 = min(cal['t2_us'] * 1e-6, 2 * t1)
        thermal_2q = thermal_relaxation_error(t1, t2, gate_time_2q)
        thermal_2q = thermal_2q.tensor(thermal_2q)

        cz_err = min(cal.get('cz_errors', {}).values(), default=0.01)
        if cz_err <= 0:
            cz_err = 0.01
        depol_2q = depolarizing_error(cz_err, 2)

        combined = thermal_2q.compose(depol_2q)
        noise_model.add_quantum_error(combined, ['cx', 'cz', 'swap'], [i, i + 1])
        noise_model.add_quantum_error(combined, ['cx', 'cz', 'swap'], [i + 1, i])

    return noise_model


# =============================================================================
# STATE PREPARATION
# =============================================================================

def create_parametrized_state(theta_deg: float) -> np.ndarray:
    """
    Create parametrized entangled state |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>.

    Args:
        theta_deg: Angle in degrees

    Returns:
        State vector as numpy array

    Note:
        - theta=0: |00> (separable)
        - theta=90: Bell state (maximally entangled)
        - theta=180: |11> (separable)
    """
    theta = np.radians(theta_deg)
    return np.array([cos(theta/2), 0, 0, sin(theta/2)])


def compute_theoretical_values(theta_deg: float) -> dict:
    """
    Compute theoretical I2, M2, Q, and N for parametrized state.

    State: |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
    This is a fundamental property: Tr[(ρ^{T_A})²] = Tr[(ρ^R)²] = Tr[ρ²].
    Therefore we only need to measure I₂ once.

    Chirality witness: Q = I₂² - M₂ (since R₂ = I₂)
    - Q = 0 for separable states
    - Q > 0 for entangled states
    - Q_max = 0.75 for maximally entangled (Bell) states

    Args:
        theta_deg: Angle in degrees

    Returns:
        Dictionary with I2, M2, Q (chirality witness), N (negativity)
    """
    theta = np.radians(theta_deg)
    c = cos(theta/2)
    s = sin(theta/2)

    # For pure states, purity is always 1
    # KEY: μ₂ = R₂ = I₂ (this is exact for all states)
    I2 = 1.0

    # M2 = (c^4 + s^4)^2 for this state
    c2 = c**2
    s2 = s**2
    M2 = (c2**2 + s2**2)**2

    # Chirality witness: Q = I₂² - M₂ (using R₂ = I₂)
    # Q >= 0 indicates entanglement (Q > 0 for entangled, Q = 0 for separable)
    Q = I2 * I2 - M2

    # Negativity: N = |sin(theta)|/2
    N = abs(sin(theta)) / 2

    return {
        'I2': I2,
        'M2': M2,
        'Q': Q,
        'N': N,
    }


# =============================================================================
# CIRCUIT CREATION
# =============================================================================

def create_I2_circuit(state_vector: np.ndarray) -> 'QuantumCircuit':
    """
    Create I₂ = Tr[ρ²] measurement circuit (5 qubits).

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
    This circuit measures the purity, which equals:
    - I₂ = Tr[ρ²] (purity)
    - R₂ = Tr[(ρ^R)²] (realignment purity)
    - μ₂ = Tr[(ρ^{T_A})²] (partial transpose second moment)

    Layout:
        q0: Ancilla
        q1, q2: Copy 1 (A1, B1)
        q3, q4: Copy 2 (A2, B2)

    Result: I₂ = 2*P(0) - 1
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for circuit creation")

    qc = QuantumCircuit(5, 1, name="I2")

    # Prepare two copies
    qc.initialize(state_vector, [1, 2])
    qc.initialize(state_vector, [3, 4])

    # Hadamard test with SWAP on both subsystems
    qc.h(0)
    qc.cswap(0, 1, 3)  # A1 <-> A2
    qc.cswap(0, 2, 4)  # B1 <-> B2
    qc.h(0)
    qc.measure(0, 0)

    return qc


def create_M2_circuits(state_vector: np.ndarray) -> Dict[str, 'QuantumCircuit']:
    """
    Create M2 measurement circuits (4 variants: SS, SY, YS, YY).

    M2 = (1/4)(SS - SY - YS + YY)

    Layout (9 qubits):
        q0: Ancilla
        q1, q2: Copy 1 (A1, B1)
        q3, q4: Copy 2 (A2, B2)
        q5, q6: Copy 3 (A3, B3)
        q7, q8: Copy 4 (A4, B4)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for circuit creation")

    circuits = {}

    for variant in ['SS', 'SY', 'YS', 'YY']:
        qc = QuantumCircuit(9, 1, name=f"M2_{variant}")

        # Prepare 4 copies
        qc.initialize(state_vector, [1, 2])
        qc.initialize(state_vector, [3, 4])
        qc.initialize(state_vector, [5, 6])
        qc.initialize(state_vector, [7, 8])

        # Hadamard test
        qc.h(0)

        # Cross-subsystem operations based on variant
        if variant[0] == 'S':
            qc.cswap(0, 1, 5)  # A1 <-> A3
            qc.cswap(0, 3, 7)  # A2 <-> A4
        else:  # Y
            # CY gates for Y variant
            qc.cy(0, 1)
            qc.cswap(0, 1, 5)
            qc.cy(0, 1)
            qc.cy(0, 3)
            qc.cswap(0, 3, 7)
            qc.cy(0, 3)

        if variant[1] == 'S':
            qc.cswap(0, 2, 6)  # B1 <-> B3
            qc.cswap(0, 4, 8)  # B2 <-> B4
        else:  # Y
            qc.cy(0, 2)
            qc.cswap(0, 2, 6)
            qc.cy(0, 2)
            qc.cy(0, 4)
            qc.cswap(0, 4, 8)
            qc.cy(0, 4)

        # Cyclic permutation on copies
        qc.cswap(0, 1, 3)
        qc.cswap(0, 3, 5)
        qc.cswap(0, 5, 7)
        qc.cswap(0, 2, 4)
        qc.cswap(0, 4, 6)
        qc.cswap(0, 6, 8)

        qc.h(0)
        qc.measure(0, 0)

        circuits[variant] = qc

    return circuits


def create_mu_circuits(state_vector: np.ndarray) -> Dict[str, 'QuantumCircuit']:
    """
    Create μₖ measurement circuits for k=3,4.

    NOTE: μ₂ = I₂ for ALL bipartite states, so we don't need a separate circuit.
    Use create_I2_circuit() to get μ₂.

    Returns:
        Dictionary with 'mu3', 'mu4' circuits
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for circuit creation")

    circuits = {}

    # NOTE: mu2 is NOT included because μ₂ = I₂ (use create_I2_circuit instead)

    # mu3: 7 qubits
    qc3 = QuantumCircuit(7, 1, name="mu3")
    qc3.initialize(state_vector, [0, 1])
    qc3.initialize(state_vector, [2, 3])
    qc3.initialize(state_vector, [4, 5])
    qc3.h(6)
    # Forward cycle on A: A₁ → A₂ → A₃ → A₁
    qc3.cswap(6, 0, 2)
    qc3.cswap(6, 2, 4)
    # Backward cycle on B: B₃ → B₂ → B₁ → B₃
    qc3.cswap(6, 5, 3)
    qc3.cswap(6, 3, 1)
    qc3.h(6)
    qc3.measure(6, 0)
    circuits['mu3'] = qc3

    # mu4: 9 qubits
    qc4 = QuantumCircuit(9, 1, name="mu4")
    qc4.initialize(state_vector, [0, 1])
    qc4.initialize(state_vector, [2, 3])
    qc4.initialize(state_vector, [4, 5])
    qc4.initialize(state_vector, [6, 7])
    qc4.h(8)
    # Forward cycle on A: A₁ → A₂ → A₃ → A₄ → A₁
    qc4.cswap(8, 0, 2)
    qc4.cswap(8, 2, 4)
    qc4.cswap(8, 4, 6)
    # Backward cycle on B: B₄ → B₃ → B₂ → B₁ → B₄
    qc4.cswap(8, 7, 5)
    qc4.cswap(8, 5, 3)
    qc4.cswap(8, 3, 1)
    qc4.h(8)
    qc4.measure(8, 0)
    circuits['mu4'] = qc4

    return circuits


# =============================================================================
# SIMULATION EXECUTION
# =============================================================================

def run_circuit(
    circuit: 'QuantumCircuit',
    backend: 'AerSimulator',
    shots: int = 8192
) -> Tuple[float, float, int, int]:
    """
    Run circuit and extract Hadamard test result.

    Args:
        circuit: QuantumCircuit to run
        backend: AerSimulator backend
        shots: Number of shots

    Returns:
        Tuple of (expectation_value, std_dev, count_0, circuit_depth)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for simulation")

    compiled = transpile(circuit, backend, optimization_level=3)
    job = backend.run(compiled, shots=shots)
    counts = job.result().get_counts()

    n0 = counts.get('0', counts.get('0x0', 0))
    n1 = counts.get('1', counts.get('0x1', 0))
    total = n0 + n1

    p0 = n0 / total
    expectation = 2 * p0 - 1

    # Binomial error
    p0_std = sqrt(p0 * (1 - p0) / total)
    std = 2 * p0_std

    depth = compiled.depth()

    return expectation, std, n0, depth


def correct_measurement(measured: float, degradation_factor: float) -> float:
    """
    Correct measured value for degradation.

    O_corrected = O_meas / f

    Args:
        measured: Measured value
        degradation_factor: Degradation factor (0 < f <= 1)

    Returns:
        Corrected value (clipped to [-1, 1])
    """
    if degradation_factor <= 0:
        return measured
    corrected = measured / degradation_factor
    return np.clip(corrected, -1.0, 1.0)


def compute_M2_from_terms(terms: dict) -> float:
    """Compute M2 = (1/4)(SS - SY - YS + YY)."""
    return 0.25 * (terms['SS'] - terms['SY'] - terms['YS'] + terms['YY'])


def compute_negativity_from_moments(mu2: float, mu3: float, mu4: float, mu1: float = 1.0) -> float:
    """
    Compute negativity from partial transpose moments using Newton-Girard.

    Args:
        mu2, mu3, mu4: Partial transpose moments
        mu1: First moment (always 1 for density matrices)

    Returns:
        Negativity (sum of absolute values of negative eigenvalues)
    """
    # Elementary symmetric polynomials via Newton-Girard
    e1 = mu1
    e2 = (mu1**2 - mu2) / 2
    e3 = (mu1**3 - 3*mu1*mu2 + 2*mu3) / 6
    e4 = (mu1**4 - 6*mu1**2*mu2 + 8*mu1*mu3 + 3*mu2**2 - 6*mu4) / 24

    # Characteristic polynomial coefficients
    coeffs = [1, -e1, e2, -e3, e4]
    eigenvalues = np.roots(coeffs)
    eigenvalues = np.real(eigenvalues)

    # Negativity is sum of absolute values of negative eigenvalues
    return -sum(e for e in eigenvalues if e < -1e-10)


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_chirality_simulation(
    calibration_file: Optional[Union[str, Path]] = None,
    thetas: List[float] = None,
    shots: int = 8192,
    verbose: bool = True
) -> List[dict]:
    """
    Run chirality witness and negativity simulation.

    Args:
        calibration_file: Path to calibration CSV (uses default if None)
        thetas: List of theta angles in degrees (default: [0, 15, 30, 45, 60, 90])
        shots: Number of shots per circuit
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit and qiskit-aer are required for simulation")

    if thetas is None:
        thetas = [0, 15, 30, 45, 60, 90]

    # Find calibration file
    if calibration_file is None:
        data_dir = Path(__file__).parent / "data" / "IBM Torino"
        cal_files = list(data_dir.glob("ibm_torino_calibrations_*.csv"))
        if cal_files:
            calibration_file = max(cal_files)  # Most recent
        else:
            raise FileNotFoundError(f"No calibration files found in {data_dir}")

    if verbose:
        print("=" * 80)
        print("CHIRALITY WITNESS SIMULATION")
        print("=" * 80)
        print(f"\nLoading calibration: {Path(calibration_file).name}")

    # Load calibration and create noise model
    calibrations, _ = parse_calibration_csv(calibration_file)
    if verbose:
        print(f"  Found {len(calibrations)} qubits")

    noise_model = create_noise_model_from_calibration(calibrations, num_qubits=10)
    backend = AerSimulator(noise_model=noise_model)

    if verbose:
        print(f"\nSimulation parameters:")
        print(f"  Shots per circuit: {shots:,}")
        print(f"  Test angles: {thetas}")

    # Calibration using Bell state (theta=90)
    # KEY IDENTITY: μ₂ = R₂ = I₂ for ALL states, so we only run I₂ once
    if verbose:
        print("\n--- Calibration using theta=90 (Bell state) ---")
        print("  NOTE: μ₂ = R₂ = I₂ (identity), running only I₂ circuit")

    psi_bell = create_parametrized_state(90)

    # Only run I2 (since μ₂ = R₂ = I₂)
    I2_cal, _, _, _ = run_circuit(create_I2_circuit(psi_bell), backend, shots)
    M2_circuits_cal = create_M2_circuits(psi_bell)
    M2_SS_cal, _, _, _ = run_circuit(M2_circuits_cal['SS'], backend, shots)

    f_I2 = I2_cal  # This is also f_R2 and f_mu2
    f_M2 = M2_SS_cal

    if verbose:
        print(f"  Degradation factors:")
        print(f"    f_I2 (= f_R2 = f_μ2) = {f_I2:.4f}")
        print(f"    f_M2_SS = {f_M2:.4f}")

    # Run simulations
    results = []

    for theta in thetas:
        if verbose:
            print(f"\ntheta = {theta} deg:")

        psi = create_parametrized_state(theta)
        theory = compute_theoretical_values(theta)

        # I2 (also equals R2 and μ2)
        I2_val, I2_std, _, _ = run_circuit(create_I2_circuit(psi), backend, shots)
        I2_corr = correct_measurement(I2_val, f_I2)

        # M2
        M2_circuits = create_M2_circuits(psi)
        M2_terms = {}
        M2_terms_corr = {}
        for name, qc in M2_circuits.items():
            val, _, _, _ = run_circuit(qc, backend, shots)
            M2_terms[name] = val
            M2_terms_corr[name] = correct_measurement(val, f_M2)

        M2_val = compute_M2_from_terms(M2_terms)
        M2_corr = compute_M2_from_terms(M2_terms_corr)

        # Chirality witness Q = I₂² - M₂ (since R₂ = I₂)
        Q_val = I2_val * I2_val - M2_val
        Q_corr = I2_corr * I2_corr - M2_corr

        # mu3 and mu4 circuits (mu2 = I2, no separate circuit needed)
        mu_circuits = create_mu_circuits(psi)
        mu_vals = {'mu2': I2_val}  # μ₂ = I₂
        mu_vals_corr = {'mu2': I2_corr}
        for name, qc in mu_circuits.items():
            val, _, _, _ = run_circuit(qc, backend, shots)
            mu_vals[name] = val
            mu_vals_corr[name] = correct_measurement(val, f_I2)

        # Negativity from moments (μ₂ = I₂)
        try:
            N_val = compute_negativity_from_moments(
                mu_vals['mu2'], mu_vals['mu3'], mu_vals['mu4']
            )
        except:
            N_val = 0.0

        try:
            N_corr = compute_negativity_from_moments(
                mu_vals_corr['mu2'], mu_vals_corr['mu3'], mu_vals_corr['mu4']
            )
        except:
            N_corr = 0.0

        if verbose:
            print(f"  I2 (= R2 = μ2): raw={I2_val:.4f}, corr={I2_corr:.4f} (theory: {theory['I2']:.4f})")
            print(f"  M2: raw={M2_val:.4f}, corr={M2_corr:.4f} (theory: {theory['M2']:.4f})")
            print(f"  Q:  raw={Q_val:.4f}, corr={Q_corr:.4f} (theory: {theory['Q']:.4f})")
            print(f"  N:  raw={N_val:.4f}, corr={N_corr:.4f} (theory: {theory['N']:.4f})")

        results.append({
            'theta': theta,
            'I2_raw': I2_val, 'I2_corr': I2_corr, 'I2_theory': theory['I2'],
            'M2_raw': M2_val, 'M2_corr': M2_corr, 'M2_theory': theory['M2'],
            'M2_SS': M2_terms['SS'], 'M2_SY': M2_terms['SY'],
            'M2_YS': M2_terms['YS'], 'M2_YY': M2_terms['YY'],
            'Q_raw': Q_val, 'Q_corr': Q_corr, 'Q_theory': theory['Q'],
            'mu2_raw': mu_vals['mu2'], 'mu2_corr': mu_vals_corr['mu2'],
            'mu3_raw': mu_vals['mu3'], 'mu3_corr': mu_vals_corr['mu3'],
            'mu4_raw': mu_vals['mu4'], 'mu4_corr': mu_vals_corr['mu4'],
            'N_raw': N_val, 'N_corr': N_corr, 'N_theory': theory['N'],
        })

    if verbose:
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)

    return results


def save_results_to_csv(results: List[dict], filepath: Union[str, Path]) -> None:
    """Save simulation results to CSV file."""
    filepath = Path(filepath)

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def run_simulation_validation(verbose: bool = True) -> Tuple[int, int]:
    """
    Run simulation validation tests.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    if not QISKIT_AVAILABLE:
        if verbose:
            print("\n[SKIP] Simulation validation skipped (qiskit not available)")
        return 0, 0

    print("\n" + "=" * 80)
    print("SIMULATION VALIDATION")
    print("=" * 80)

    passed = 0
    failed = 0

    # Test 1: Theoretical values
    # NOTE: R2 removed since μ₂ = R₂ = I₂ (identity)
    print("\n--- Test 1: Theoretical value computation ---")
    print("  (NOTE: μ₂ = R₂ = I₂ identity used)")

    test_cases = [
        (0, {'I2': 1.0, 'M2': 1.0, 'Q': 0.0, 'N': 0.0}),
        (90, {'I2': 1.0, 'M2': 0.25, 'Q': 0.75, 'N': 0.5}),
        (180, {'I2': 1.0, 'M2': 1.0, 'Q': 0.0, 'N': 0.0}),
    ]

    for theta, expected in test_cases:
        theory = compute_theoretical_values(theta)
        max_error = max(abs(theory[k] - expected[k]) for k in expected)

        if max_error < 1e-10:
            passed += 1
            if verbose:
                print(f"  [PASS] theta={theta}: values match")
        else:
            failed += 1
            if verbose:
                print(f"  [FAIL] theta={theta}: max error = {max_error:.2e}")

    # Test 2: State preparation
    print("\n--- Test 2: State parametrization ---")

    psi_0 = create_parametrized_state(0)
    psi_90 = create_parametrized_state(90)
    psi_180 = create_parametrized_state(180)

    # theta=0 should be |00>
    if abs(psi_0[0] - 1.0) < 1e-10 and np.sum(np.abs(psi_0[1:])**2) < 1e-10:
        passed += 1
        if verbose:
            print("  [PASS] theta=0 gives |00>")
    else:
        failed += 1
        if verbose:
            print("  [FAIL] theta=0 does not give |00>")

    # theta=90 should be Bell state
    expected_bell = np.array([1, 0, 0, 1]) / sqrt(2)
    if np.max(np.abs(psi_90 - expected_bell)) < 1e-10:
        passed += 1
        if verbose:
            print("  [PASS] theta=90 gives Bell state")
    else:
        failed += 1
        if verbose:
            print("  [FAIL] theta=90 does not give Bell state")

    # theta=180 should be |11>
    if abs(psi_180[3] - 1.0) < 1e-10 and np.sum(np.abs(psi_180[:3])**2) < 1e-10:
        passed += 1
        if verbose:
            print("  [PASS] theta=180 gives |11>")
    else:
        failed += 1
        if verbose:
            print("  [FAIL] theta=180 does not give |11>")

    # Test 3: Newton-Girard negativity
    print("\n--- Test 3: Newton-Girard negativity computation ---")

    # For Bell state, mu2=mu3=mu4=1 for pure state, N=0.5
    # Actually for Bell state: mu2=1, mu3=0.25, mu4=0.125 (from PT eigenvalues)
    # PT eigenvalues for Bell: [0.5, 0.5, -0.5, 0.5] -> no wait, that's wrong
    # Let me use known values
    # For |psi> = (|00>+|11>)/sqrt(2), rho^PT has eigenvalues [0.5, 0.5, 0.5, -0.5]
    # So N = 0.5

    # Test with known moments for Bell state
    # mu_k = sum(lambda_i^k) for k=2,3,4
    # lambdas = [0.5, 0.5, 0.5, -0.5]
    lambdas = [0.5, 0.5, 0.5, -0.5]
    mu2_bell = sum(l**2 for l in lambdas)  # 4 * 0.25 = 1.0
    mu3_bell = sum(l**3 for l in lambdas)  # 3*0.125 - 0.125 = 0.25
    mu4_bell = sum(l**4 for l in lambdas)  # 4 * 0.0625 = 0.25

    N_computed = compute_negativity_from_moments(mu2_bell, mu3_bell, mu4_bell)

    if abs(N_computed - 0.5) < 0.01:
        passed += 1
        if verbose:
            print(f"  [PASS] Bell state negativity = {N_computed:.4f} (expected 0.5)")
    else:
        failed += 1
        if verbose:
            print(f"  [FAIL] Bell state negativity = {N_computed:.4f} (expected 0.5)")

    print(f"\nSimulation validation: {passed} passed, {failed} failed")

    return passed, failed


if __name__ == "__main__":
    # Run validation when executed directly
    run_simulation_validation(verbose=True)

    # Optionally run full simulation
    import sys
    if "--full" in sys.argv:
        results = run_chirality_simulation(verbose=True)

        # Save results
        output_file = Path(__file__).parent / "data" / "simulation_results.csv"
        save_results_to_csv(results, output_file)
        print(f"\nResults saved to: {output_file}")
