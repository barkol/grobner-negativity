#!/usr/bin/env python3
"""
Simulate Chirality Witness Experiments on Fake IBM Torino

Generates simulated experimental data for the chirality witness Q = I₂² - M₂
using a noise model derived from IBM Torino calibration data.

KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
This means we only need to run the I₂ circuit once, not separate I₂, R₂, μ₂ circuits.

TWO-STAGE MAXIMUM LIKELIHOOD ESTIMATION:
    Stage 1 (Oracle Calibration): Use KNOWN θ values to fit f_I2, f_M2, p
                                  These are hardware/noise parameters
    Stage 2 (Blind Estimation): Use calibrated f_I2, f_M2, p to fit θ
                                This gives M₂^phys for unknown states

This script produces realistic simulation results to populate the SI document
while waiting for real experimental data.
"""

import numpy as np
from numpy import sqrt, kron, cos, sin, pi
from pathlib import Path
import csv
import sys

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


# =============================================================================
# CALIBRATION DATA PARSING
# =============================================================================

def parse_torino_calibration(filepath: Path) -> dict:
    """Parse IBM Torino calibration CSV file."""
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
            sx_error = safe_float(row.get('\u221ax (sx) error', ''))  # √x
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


def create_torino_noise_model(calibrations: dict, num_qubits: int = 10) -> NoiseModel:
    """Create noise model from Torino calibration for the best qubits."""

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

        thermal_error = thermal_relaxation_error(t1, t2, gate_time_1q)
        for gate in ['id', 'rz', 'sx', 'x']:
            noise_model.add_quantum_error(thermal_error, gate, [i])

        # Depolarizing error
        if cal['sx_error'] > 0:
            dep_error = depolarizing_error(cal['sx_error'], 1)
            noise_model.add_quantum_error(dep_error, 'sx', [i])

        if cal['x_error'] > 0:
            dep_error = depolarizing_error(cal['x_error'], 1)
            noise_model.add_quantum_error(dep_error, 'x', [i])

        # Readout error
        if cal['prob_m0_p1'] > 0 or cal['prob_m1_p0'] > 0:
            p0_given_1 = cal['prob_m0_p1']
            p1_given_0 = cal['prob_m1_p0']
            readout_error_matrix = [
                [1 - p1_given_0, p1_given_0],
                [p0_given_1, 1 - p0_given_1],
            ]
            noise_model.add_readout_error(readout_error_matrix, [i])

    # Two-qubit errors for linear connectivity
    for i in range(min(num_qubits - 1, len(sorted_qubits) - 1)):
        cal1 = calibrations[sorted_qubits[i]]
        cal2 = calibrations[sorted_qubits[i + 1]]

        # Average CZ error
        avg_cz = 0.003  # Default 0.3% error

        dep_error_2q = depolarizing_error(avg_cz, 2)
        for gate in ['cx', 'cz', 'swap']:
            noise_model.add_quantum_error(dep_error_2q, gate, [i, i + 1])
            noise_model.add_quantum_error(dep_error_2q, gate, [i + 1, i])

        # Thermal for 2Q gates
        t1_1 = cal1['t1_us'] * 1e-6
        t2_1 = min(cal1['t2_us'] * 1e-6, 2 * t1_1)
        t1_2 = cal2['t1_us'] * 1e-6
        t2_2 = min(cal2['t2_us'] * 1e-6, 2 * t1_2)

        thermal_1 = thermal_relaxation_error(t1_1, t2_1, gate_time_2q)
        thermal_2 = thermal_relaxation_error(t1_2, t2_2, gate_time_2q)
        thermal_2q = thermal_1.tensor(thermal_2)

        for gate in ['cx', 'cz', 'swap']:
            noise_model.add_quantum_error(thermal_2q, gate, [i, i + 1])
            noise_model.add_quantum_error(thermal_2q, gate, [i + 1, i])

    return noise_model


# =============================================================================
# STATE PREPARATION
# =============================================================================

def create_state_vector(theta_deg: float) -> np.ndarray:
    """Create parametrized entangled state |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>.

    This parametrization gives:
    - theta=0: |00> (separable)
    - theta=90: (|00>+|11>)/sqrt(2) (Bell state, maximally entangled)
    - theta=180: |11> (separable)
    """
    theta = np.radians(theta_deg)
    return np.array([cos(theta/2), 0, 0, sin(theta/2)])


# =============================================================================
# CIRCUIT CREATION
# =============================================================================

def create_I2_circuit(state_vector: np.ndarray) -> QuantumCircuit:
    """
    Create I₂ = Tr[ρ²] measurement circuit (5 qubits).

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
    This circuit measures the purity, which equals:
    - I₂ = Tr[ρ²] (purity)
    - R₂ = Tr[(ρ^R)²] (realignment purity)
    - μ₂ = Tr[(ρ^{T_A})²] (partial transpose second moment)

    Result: I₂ = 2*P(0) - 1
    """
    psi_psi = kron(state_vector, state_vector)

    qc = QuantumCircuit(5, 1)
    qc.initialize(psi_psi, range(4))
    qc.barrier()
    qc.h(4)
    qc.cswap(4, 0, 2)  # A1 <-> A2
    qc.cswap(4, 1, 3)  # B1 <-> B2
    qc.h(4)
    qc.measure(4, 0)

    return qc


def create_M2_circuits(state_vector: np.ndarray) -> dict:
    """Create all 4 M2 circuit variants (9 qubits each)."""
    psi4 = kron(kron(kron(state_vector, state_vector), state_vector), state_vector)

    circuits = {}

    # M2^SS: CSWAP on both cross-subsystem pairs
    qc = QuantumCircuit(9, 1)
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cswap(8, 1, 2)  # B1 <-> A2
    qc.cswap(8, 5, 6)  # B3 <-> A4
    # Cyclic permutation
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)
    circuits['SS'] = qc

    # M2^SY: CSWAP on first, CY on second
    qc = QuantumCircuit(9, 1)
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cswap(8, 1, 2)
    qc.cy(8, 5)
    qc.cy(8, 6)
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)
    circuits['SY'] = qc

    # M2^YS: CY on first, CSWAP on second
    qc = QuantumCircuit(9, 1)
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cy(8, 1)
    qc.cy(8, 2)
    qc.cswap(8, 5, 6)
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)
    circuits['YS'] = qc

    # M2^YY: CY on both
    qc = QuantumCircuit(9, 1)
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    qc.cy(8, 1)
    qc.cy(8, 2)
    qc.cy(8, 5)
    qc.cy(8, 6)
    qc.cswap(8, 0, 6)
    qc.cswap(8, 1, 7)
    qc.cswap(8, 6, 2)
    qc.cswap(8, 7, 3)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 3, 5)
    qc.h(8)
    qc.measure(8, 0)
    circuits['YY'] = qc

    return circuits


def create_mu_circuits(state_vector: np.ndarray) -> dict:
    """
    Create μ₃ and μ₄ circuits for negativity estimation.

    NOTE: μ₂ = I₂ for ALL bipartite states, so we don't create a separate circuit.
    Use create_I2_circuit() to get μ₂.
    """
    circuits = {}

    # NOTE: mu2 is NOT included because μ₂ = I₂ (use create_I2_circuit instead)

    # mu3 (7 qubits)
    psi2 = kron(state_vector, state_vector)
    psi3 = kron(psi2, state_vector)
    qc = QuantumCircuit(7, 1)
    qc.initialize(psi3, range(6))
    qc.barrier()
    qc.h(6)
    # Cycle on A: A₁ → A₂ → A₃ → A₁
    qc.cswap(6, 0, 2)
    qc.cswap(6, 2, 4)
    # Anticycle on B: B₃ → B₂ → B₁ → B₃ (reversed order)
    qc.cswap(6, 5, 3)
    qc.cswap(6, 3, 1)
    qc.h(6)
    qc.measure(6, 0)
    circuits['mu3'] = qc

    # mu4 (9 qubits)
    psi4 = kron(psi3, state_vector)
    qc = QuantumCircuit(9, 1)
    qc.initialize(psi4, range(8))
    qc.barrier()
    qc.h(8)
    # Cycle on A: A₁ → A₂ → A₃ → A₄ → A₁
    qc.cswap(8, 0, 2)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 4, 6)
    # Anticycle on B: B₄ → B₃ → B₂ → B₁ → B₄ (reversed order)
    qc.cswap(8, 7, 5)
    qc.cswap(8, 5, 3)
    qc.cswap(8, 3, 1)
    qc.h(8)
    qc.measure(8, 0)
    circuits['mu4'] = qc

    return circuits


# =============================================================================
# SIMULATION
# =============================================================================

def run_circuit(qc: QuantumCircuit, backend: AerSimulator, shots: int) -> tuple:
    """Run circuit and return expectation value and std."""
    compiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    n0 = counts.get('0', 0)
    n1 = counts.get('1', 0)
    total = n0 + n1

    p0 = n0 / total
    expectation = 2 * p0 - 1

    # Binomial error
    p0_std = sqrt(p0 * (1 - p0) / total)
    std = 2 * p0_std

    # Get circuit depth for degradation estimation
    depth = compiled.depth()

    return expectation, std, n0, depth


def estimate_degradation_factor(depth: int, base_fidelity: float = 0.995) -> float:
    """
    Estimate degradation factor based on circuit depth.

    For depolarizing noise, the fidelity decays exponentially with depth:
    f = base_fidelity^depth

    Args:
        depth: Circuit depth
        base_fidelity: Per-layer fidelity (default 0.995)

    Returns:
        Estimated degradation factor
    """
    return base_fidelity ** depth


def correct_measurement(measured: float, degradation_factor: float) -> float:
    """
    Correct measured value for degradation.

    For observable O, measured value is:
        O_meas = f * O_ideal + (1-f) * O_noise

    For Hadamard test, O_noise = 0 (random outcomes give 0.5 probability),
    so: O_meas = f * O_ideal
    Thus: O_corrected = O_meas / f

    Args:
        measured: Measured value
        degradation_factor: Estimated degradation factor

    Returns:
        Corrected value (clipped to valid range)
    """
    if degradation_factor <= 0:
        return measured
    corrected = measured / degradation_factor
    # Clip to valid range for physical quantities
    return np.clip(corrected, -1.0, 1.0)


def compute_M2_from_terms(terms: dict) -> float:
    """Compute M2 = (1/4)(SS - SY - YS + YY)."""
    return 0.25 * (terms['SS'] - terms['SY'] - terms['YS'] + terms['YY'])


def compute_negativity_from_moments(mu2, mu3, mu4, mu1=1.0):
    """Compute negativity from partial transpose moments using Newton-Girard."""
    # Elementary symmetric polynomials
    e1 = mu1
    e2 = (mu1**2 - mu2) / 2
    e3 = (mu1**3 - 3*mu1*mu2 + 2*mu3) / 6
    e4 = (mu1**4 - 6*mu1**2*mu2 + 8*mu1*mu3 + 3*mu2**2 - 6*mu4) / 24

    # Characteristic polynomial coefficients
    coeffs = [1, -e1, e2, -e3, e4]
    eigenvalues = np.roots(coeffs)
    eigenvalues = np.real(eigenvalues)

    # Negativity = sum of absolute values of negative eigenvalues
    return -sum(e for e in eigenvalues if e < -1e-10)


# =============================================================================
# MAXIMUM LIKELIHOOD FOR I2 AND M2
# =============================================================================

def I2_model_depolarized(p: float) -> float:
    """
    I₂ for depolarized two-qubit state.
    ρ = (1-p)|ψ⟩⟨ψ| + p·I/4

    For any pure state |ψ⟩, I₂(pure) = 1.
    For I/4, I₂(I/4) = Tr[(I/4)²] = 1/4.

    I₂(ρ) = (1-p)² × 1 + 2p(1-p) × ⟨ψ|I/4|ψ⟩ + p² × 1/4
          = (1-p)² + p(1-p)/2 + p²/4
    """
    return (1 - p)**2 + p*(1-p)/2 + p**2/4


def M2_model_depolarized(theta_rad: float, p: float) -> float:
    """
    M₂ for depolarized parametrized state.
    ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

    For pure state: M₂_pure = (c⁴ + s⁴)² where c=cos(θ/2), s=sin(θ/2)
    For I/4: M₂_mixed = 1/16

    For depolarized state (approximate interpolation):
    M₂ ≈ (1-p)² × M₂_pure + 2p(1-p) × √(M₂_pure × M₂_mixed) + p² × M₂_mixed
    """
    c = cos(theta_rad/2)
    s = sin(theta_rad/2)
    M2_pure = (c**4 + s**4)**2
    M2_mixed = 1/16
    return (1 - p)**2 * M2_pure + 2*p*(1-p) * sqrt(M2_pure * M2_mixed) + p**2 * M2_mixed


def Q_model_depolarized(theta_rad: float, p: float) -> float:
    """Q = I₂² - M₂ for depolarized state."""
    I2 = I2_model_depolarized(p)
    M2 = M2_model_depolarized(theta_rad, p)
    return I2**2 - M2


class ChiralityMaxLikEstimator:
    """
    Two-Stage Maximum Likelihood Estimator for chirality witness.

    Physical model with depolarization:
        ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

    Circuit degradation model:
        I₂_measured = f_I2 × I₂_model(p)
        M₂_measured = f_M2 × M₂_model(θ, p)

    Two-stage approach:
        Stage 1 (Oracle Calibration): Use KNOWN θ values to fit f_I2, f_M2, p
                                      These are hardware/noise parameters
        Stage 2 (Blind Estimation): Use calibrated f_I2, f_M2, p to fit θ
                                    This gives M₂^phys for unknown states

    This is more realistic: calibrate hardware first, then use for state estimation.
    """

    def __init__(self):
        self.states = []  # List of state data
        self.f_I2 = None
        self.f_M2 = None
        self.p_cal = None  # Calibrated depolarization parameter
        self.calibrated = False

    def add_state(self, name: str, theta_deg: float, I2_meas: float, M2_meas: float):
        """Add a state's experimental data."""
        theta_rad = np.radians(theta_deg)
        c = cos(theta_rad/2)
        s = sin(theta_rad/2)
        M2_theory = (c**4 + s**4)**2
        Q_theory = 1.0 - M2_theory  # For pure state, I₂ = 1

        self.states.append({
            'name': name,
            'theta_deg': theta_deg,
            'theta_rad': theta_rad,
            'I2_meas': I2_meas,
            'M2_meas': M2_meas,
            'M2_theory': M2_theory,
            'Q_theory': Q_theory,
        })

    def _calibration_nll(self, params: np.ndarray) -> float:
        """
        Stage 1: Calibration NLL using KNOWN theta values.

        params = [f_I2, f_M2, p]  (shared p for all states)
        """
        f_I2, f_M2, p = params

        # Bounds check
        if f_I2 <= 0 or f_M2 <= 0 or f_I2 > 1.5 or f_M2 > 1.0:
            return 1e10
        if p < 0 or p > 0.5:
            return 1e10

        nll = 0.0
        sigma_I2 = 0.02
        sigma_M2 = 0.03

        for state in self.states:
            # Use KNOWN theta (oracle)
            theta = state['theta_rad']

            # Model predictions with known theta
            I2_model = I2_model_depolarized(p)
            M2_model = M2_model_depolarized(theta, p)

            # Apply degradation
            I2_pred = f_I2 * I2_model
            M2_pred = f_M2 * M2_model

            # NLL
            nll += 0.5 * ((state['I2_meas'] - I2_pred)**2 / sigma_I2**2)
            nll += 0.5 * ((state['M2_meas'] - M2_pred)**2 / sigma_M2**2)

        return nll

    def calibrate(self) -> dict:
        """
        Stage 1: Calibrate f_I2, f_M2, p using KNOWN theta values (oracle).

        This establishes hardware/noise parameters.
        """
        from scipy.optimize import differential_evolution

        if not self.states:
            raise ValueError("No states added")

        # Bounds: [f_I2, f_M2, p]
        bounds = [
            (0.8, 1.1),   # f_I2 (shallow circuit, ~no degradation)
            (0.1, 0.5),   # f_M2 (deep circuit, significant degradation)
            (0.0, 0.3),   # p (depolarization)
        ]

        result = differential_evolution(
            self._calibration_nll,
            bounds,
            seed=42,
            maxiter=1000,
            tol=1e-10,
            polish=True,
        )

        self.f_I2, self.f_M2, self.p_cal = result.x
        self.calibrated = True

        return {
            'f_I2': self.f_I2,
            'f_M2': self.f_M2,
            'p': self.p_cal,
            'success': result.success,
            'nll': result.fun,
        }

    def _fit_theta_nll(self, theta_rad: float, I2_meas: float, M2_meas: float) -> float:
        """NLL for fitting theta given fixed f_I2, f_M2, p."""
        if theta_rad < 0.001 or theta_rad > np.pi - 0.001:
            return 1e10

        I2_model = I2_model_depolarized(self.p_cal)
        M2_model = M2_model_depolarized(theta_rad, self.p_cal)

        I2_pred = self.f_I2 * I2_model
        M2_pred = self.f_M2 * M2_model

        sigma_I2 = 0.02
        sigma_M2 = 0.03

        nll = 0.5 * ((I2_meas - I2_pred)**2 / sigma_I2**2)
        nll += 0.5 * ((M2_meas - M2_pred)**2 / sigma_M2**2)

        return nll

    def fit_theta(self, I2_meas: float, M2_meas: float) -> dict:
        """
        Stage 2: Fit theta using calibrated f_I2, f_M2, p (blind estimation).

        Args:
            I2_meas: Measured I₂ (raw)
            M2_meas: Measured M₂ (raw)

        Returns:
            Dictionary with fitted theta and derived M₂^phys, Q^phys
        """
        from scipy.optimize import minimize_scalar

        if not self.calibrated:
            raise ValueError("Must call calibrate() first")

        # Optimize theta
        result = minimize_scalar(
            lambda t: self._fit_theta_nll(t, I2_meas, M2_meas),
            bounds=(0.001, np.pi - 0.001),
            method='bounded',
        )

        theta_fit = result.x

        # Compute physical values using fitted theta and calibrated p
        I2_phys = I2_model_depolarized(self.p_cal)
        M2_phys = M2_model_depolarized(theta_fit, self.p_cal)
        Q_phys = I2_phys**2 - M2_phys

        return {
            'theta_fit_rad': theta_fit,
            'theta_fit_deg': np.degrees(theta_fit),
            'I2_phys': I2_phys,
            'M2_phys': M2_phys,
            'Q_phys': Q_phys,
            'nll': result.fun,
        }

    def fit_all(self) -> dict:
        """
        Run both stages: calibrate then fit all states.

        Returns complete results dictionary.
        """
        # Stage 1: Calibration
        cal_results = self.calibrate()

        # Stage 2: Fit theta for each state
        results = {
            'f_I2': self.f_I2,
            'f_M2': self.f_M2,
            'p': self.p_cal,
            'calibration_success': cal_results['success'],
            'calibration_nll': cal_results['nll'],
            'states': {},
        }

        for state in self.states:
            fit_result = self.fit_theta(state['I2_meas'], state['M2_meas'])

            # Compute Q error
            Q_error = abs(fit_result['Q_phys'] - state['Q_theory'])

            results['states'][state['name']] = {
                'theta_true_deg': state['theta_deg'],
                'theta_fit_deg': fit_result['theta_fit_deg'],
                'I2_meas': state['I2_meas'],
                'M2_meas': state['M2_meas'],
                'I2_phys': fit_result['I2_phys'],
                'M2_phys': fit_result['M2_phys'],
                'Q_phys': fit_result['Q_phys'],
                'Q_theory': state['Q_theory'],
                'Q_error': Q_error,
            }

        # Summary
        errors = [d['Q_error'] for d in results['states'].values()]
        results['mean_error'] = np.mean(errors)

        return results


def compute_M2_physical(I2_meas: float, theta_deg: float) -> tuple:
    """
    Compute the physical M₂ value for a given measured I₂ and state angle.

    NOTE: This function uses the KNOWN theta value, which means it essentially
    replaces the noisy M₂ measurement with the theoretical value adjusted for
    depolarization. This is useful for understanding the noise model but is
    NOT a fair comparison for practical entanglement detection where theta
    is unknown.

    For the parametrized state |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
    with depolarization: ρ = (1-p)|ψ⟩⟨ψ| + p·I/4

    The measured I₂ determines the depolarization parameter p.

    Args:
        I2_meas: Measured purity (should be ≤ 1)
        theta_deg: State angle in degrees (KNOWN - this is the "oracle" aspect)

    Returns:
        Tuple of (M₂_phys, p_fit)
    """
    from scipy.optimize import brentq

    theta = np.radians(theta_deg)
    c = cos(theta/2)
    s = sin(theta/2)

    def I2_model(p):
        """I₂ for depolarized two-qubit state."""
        return (1 - p)**2 + p**2/4 + p*(1-p)/2

    # Clip I2_meas to valid range
    I2_meas = np.clip(I2_meas, 0.25, 1.0)

    # Find p that gives measured I₂
    try:
        if I2_meas >= 0.9999:
            p_fit = 0.0
        else:
            p_fit = brentq(lambda p: I2_model(p) - I2_meas, 0, 1)
    except:
        p_fit = 0.0

    # M₂ for pure state
    M2_pure = (c**4 + s**4)**2
    M2_max_mixed = 1/16  # M₂ for I/4

    # M₂ for depolarized state (approximate interpolation)
    M2_phys = (1 - p_fit)**2 * M2_pure + 2*p_fit*(1-p_fit) * sqrt(M2_pure * M2_max_mixed) + p_fit**2 * M2_max_mixed

    return M2_phys, p_fit


def compute_M2_physical_blind(I2_meas: float, M2_meas: float) -> tuple:
    """
    Compute the physical M₂ value WITHOUT knowing theta (blind estimation).

    This uses a joint ML fit to find (theta, p) that best explains both
    measured I₂ and M₂. This is a FAIR comparison since it doesn't use
    oracle knowledge of theta.

    For the parametrized state |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
    with depolarization: ρ = (1-p)|ψ⟩⟨ψ| + p·I/4

    Args:
        I2_meas: Measured purity
        M2_meas: Measured M₂ (from 4-term circuit)

    Returns:
        Tuple of (M₂_phys, theta_fit, p_fit)
    """
    from scipy.optimize import minimize

    def I2_model(p):
        """I₂ for depolarized two-qubit state."""
        return (1 - p)**2 + p**2/4 + p*(1-p)/2

    def M2_model(theta_rad, p):
        """M₂ for depolarized parametrized state."""
        c = cos(theta_rad/2)
        s = sin(theta_rad/2)
        M2_pure = (c**4 + s**4)**2
        M2_max_mixed = 1/16
        return (1 - p)**2 * M2_pure + 2*p*(1-p) * sqrt(M2_pure * M2_max_mixed) + p**2 * M2_max_mixed

    def neg_log_likelihood(params):
        theta_rad, p = params
        if p < 0 or p > 1 or theta_rad < 0 or theta_rad > np.pi:
            return 1e10

        I2_pred = I2_model(p)
        M2_pred = M2_model(theta_rad, p)

        # Weighted least squares (assuming similar uncertainties)
        return (I2_meas - I2_pred)**2 + (M2_meas - M2_pred)**2

    # Try multiple starting points
    best_result = None
    best_nll = float('inf')

    for theta_init in [0.1, 0.5, 1.0, 1.5]:
        for p_init in [0.0, 0.1, 0.3]:
            result = minimize(
                neg_log_likelihood,
                [theta_init, p_init],
                method='L-BFGS-B',
                bounds=[(0.01, np.pi - 0.01), (0, 0.99)],
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result

    theta_fit, p_fit = best_result.x
    M2_phys = M2_model(theta_fit, p_fit)

    return M2_phys, np.degrees(theta_fit), p_fit


def compute_chirality_maxlik(
    I2_meas: float,
    M2_meas: float,
    I2_std: float = 0.01,
    M2_std: float = 0.01,
    theta_init: float = 45.0,
) -> dict:
    """
    Maximum likelihood estimation for chirality witness.

    Fits a depolarized state model to measured I₂ and M₂:
        ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

    The physical M₂ is the M₂ value for the fitted depolarized state.

    Args:
        I2_meas: Measured I₂
        M2_meas: Measured M₂
        I2_std: Standard deviation of I₂
        M2_std: Standard deviation of M₂
        theta_init: Initial guess for θ (degrees)

    Returns:
        Dictionary with fitted parameters and physical values
    """
    from scipy.optimize import minimize

    def I2_model(theta_deg, p):
        """I₂ for depolarized two-qubit state."""
        # For depolarized state: I₂ = (1-p)² + p²/4 + p(1-p)/2
        # This is independent of theta for pure state targets
        return (1 - p)**2 + p**2/4 + p*(1-p)/2

    def M2_model(theta_deg, p):
        """M₂ for depolarized state."""
        theta = np.radians(theta_deg)
        c = cos(theta/2)
        s = sin(theta/2)

        M2_pure = (c**4 + s**4)**2
        M2_max_mixed = 1/16

        # Approximate interpolation
        return (1 - p)**2 * M2_pure + 2*p*(1-p) * sqrt(M2_pure * M2_max_mixed) + p**2 * M2_max_mixed

    def neg_log_likelihood(params):
        theta_deg, p = params
        if p < 0 or p > 1 or theta_deg < 0 or theta_deg > 180:
            return 1e10

        I2_pred = I2_model(theta_deg, p)
        M2_pred = M2_model(theta_deg, p)

        nll = 0.5 * ((I2_meas - I2_pred)**2 / max(I2_std**2, 1e-10))
        nll += 0.5 * ((M2_meas - M2_pred)**2 / max(M2_std**2, 1e-10))

        return nll

    # Optimize
    result = minimize(
        neg_log_likelihood,
        [theta_init, 0.1],
        method='L-BFGS-B',
        bounds=[(0, 180), (0, 0.99)],
    )

    theta_fit, p_fit = result.x

    # Compute physical values
    I2_phys = I2_model(theta_fit, p_fit)
    M2_phys = M2_model(theta_fit, p_fit)
    Q_phys = I2_phys**2 - M2_phys

    # Theoretical values for this theta
    theta_rad = np.radians(theta_fit)
    c = cos(theta_rad/2)
    s = sin(theta_rad/2)
    M2_theory = (c**4 + s**4)**2
    Q_theory = 1.0 - M2_theory
    N_theory = abs(sin(theta_rad)) / 2

    return {
        'theta_fit': theta_fit,
        'p_fit': p_fit,
        'I2_phys': I2_phys,
        'M2_phys': M2_phys,
        'Q_phys': Q_phys,
        'M2_theory': M2_theory,
        'Q_theory': Q_theory,
        'N_theory': N_theory,
        'nll': result.fun,
        'success': result.success,
    }


# =============================================================================
# THEORETICAL VALUES
# =============================================================================

def compute_theoretical_values(theta_deg: float) -> dict:
    """Compute theoretical I2, M2, Q, and N for parametrized state.

    State: |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>
    where theta is given in degrees.

    KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states.
    This is a fundamental property: Tr[(ρ^{T_A})²] = Tr[(ρ^R)²] = Tr[ρ²].

    Chirality witness: Q = I₂² - M₂ (since R₂ = I₂)
    - Q = 0 for separable states
    - Q > 0 for entangled states
    - Q_max = 0.75 for maximally entangled (Bell) states

    This parametrization gives:
    - theta=0: |00> (separable), N=0, Q=0
    - theta=90: Bell state, N=0.5, Q=0.75
    - theta=180: |11> (separable), N=0, Q=0
    """
    theta = np.radians(theta_deg)
    c = cos(theta/2)
    s = sin(theta/2)

    # For pure states, purity is always 1
    # KEY: μ₂ = R₂ = I₂ (this is exact for all states)
    I2 = 1.0

    # M2 = Tr[(rho * rho^R)^2]
    # For this specific state, M2 = (c^4 + s^4)^2
    c2 = c**2
    s2 = s**2
    M2 = (c2**2 + s2**2)**2

    # Chirality witness: Q = I₂² - M₂ (using R₂ = I₂)
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
# MAIN SIMULATION
# =============================================================================

def main():
    print("=" * 80)
    print("CHIRALITY WITNESS SIMULATION - FAKE IBM TORINO")
    print("=" * 80)

    # Load Torino calibration
    script_dir = Path(__file__).parent
    calib_file = script_dir / "data" / "IBM Torino" / "ibm_torino_calibrations_2026-01-10T06_45_42Z.csv"

    if not calib_file.exists():
        calib_file = script_dir / "data" / "torino_calibration.csv"

    if not calib_file.exists():
        print(f"[!] Calibration file not found: {calib_file}")
        print("    Using ideal simulator instead.")
        backend = AerSimulator()
        noise_model = None
    else:
        print(f"\nLoading calibration: {calib_file.name}")
        calibrations, coupling = parse_torino_calibration(calib_file)
        print(f"  Found {len(calibrations)} qubits")

        # Create noise model
        noise_model = create_torino_noise_model(calibrations, num_qubits=10)
        backend = AerSimulator(noise_model=noise_model)
        print("  Created noise model with thermal relaxation, depolarizing, and readout errors")

    # Simulation parameters
    shots = 8192
    thetas = [0, 15, 30, 45, 60, 90]

    print(f"\nSimulation parameters:")
    print(f"  Shots per circuit: {shots:,}")
    print(f"  Test angles: {thetas}")

    # Results storage
    results = []

    print("\n" + "-" * 80)
    print("RUNNING SIMULATIONS")
    print("-" * 80)

    # Calibration: Use theta=90 (Bell state) to estimate degradation factors
    # KEY IDENTITY: μ₂ = R₂ = I₂ for ALL states, so we only run I₂ once
    print("\n--- Calibration using theta=90 (Bell state) ---")
    print("  NOTE: mu2 = R2 = I2 (identity), running only I2 circuit")
    psi_bell = create_state_vector(90)

    # Only run I2 (since μ₂ = R₂ = I₂)
    I2_qc_cal = create_I2_circuit(psi_bell)
    I2_cal, _, _, I2_depth = run_circuit(I2_qc_cal, backend, shots)

    M2_circuits_cal = create_M2_circuits(psi_bell)
    M2_SS_cal, _, _, M2_depth = run_circuit(M2_circuits_cal['SS'], backend, shots)

    f_I2 = I2_cal  # This is also f_R2 and f_mu2
    f_M2 = M2_SS_cal

    print(f"  Degradation factors from Bell state:")
    print(f"    f_I2 (= f_R2 = f_mu2) = {f_I2:.4f} (depth={I2_depth})")
    print(f"    f_M2_SS = {f_M2:.4f} (depth={M2_depth})")

    for theta in thetas:
        print(f"\ntheta = {theta} deg:")

        # Create state
        psi = create_state_vector(theta)

        # Theoretical values
        theory = compute_theoretical_values(theta)

        # I2 circuit (also equals R2 and mu2)
        I2_qc = create_I2_circuit(psi)
        I2_val, I2_std, I2_n0, I2_d = run_circuit(I2_qc, backend, shots)
        I2_corr = correct_measurement(I2_val, f_I2)
        print(f"  I2 (= R2 = mu2): raw={I2_val:.4f}, corr={I2_corr:.4f} (theory: {theory['I2']:.4f})")

        # M2 circuits (4 terms)
        M2_circuits = create_M2_circuits(psi)
        M2_terms = {}
        M2_terms_corr = {}
        M2_stds = {}
        M2_n0s = {}
        M2_depths = {}
        for name, qc in M2_circuits.items():
            val, std, n0, depth = run_circuit(qc, backend, shots)
            M2_terms[name] = val
            M2_terms_corr[name] = correct_measurement(val, f_M2)
            M2_stds[name] = std
            M2_n0s[name] = n0
            M2_depths[name] = depth

        M2_val = compute_M2_from_terms(M2_terms)
        M2_corr = compute_M2_from_terms(M2_terms_corr)
        # Propagate error for M2
        M2_std = 0.25 * sqrt(sum(M2_stds[k]**2 for k in M2_stds))
        print(f"  M2_raw = {M2_val:.4f}, M2_corr = {M2_corr:.4f} (theory: {theory['M2']:.4f})")
        print(f"    M2^SS = {M2_terms['SS']:.4f}, M2^SY = {M2_terms['SY']:.4f}")
        print(f"    M2^YS = {M2_terms['YS']:.4f}, M2^YY = {M2_terms['YY']:.4f}")

        # Chirality witness Q = I₂² - M₂ (since R₂ = I₂)
        Q_val = I2_val * I2_val - M2_val
        Q_std = sqrt(4 * I2_val**2 * I2_std**2 + M2_std**2)
        Q_corr = I2_corr * I2_corr - M2_corr
        Q_theory = theory['Q']
        print(f"  Q_raw = {Q_val:.4f}, Q_corr = {Q_corr:.4f} (theory: {Q_theory:.4f})")

        # mu3 and mu4 circuits (mu2 = I2, no separate circuit needed)
        mu_circuits = create_mu_circuits(psi)
        mu_vals = {'mu2': I2_val}  # μ₂ = I₂
        mu_vals_corr = {'mu2': I2_corr}
        mu_stds = {'mu2': I2_std}
        mu_n0s = {'mu2': I2_n0}
        mu_depths = {'mu2': I2_d}
        for name, qc in mu_circuits.items():
            val, std, n0, depth = run_circuit(qc, backend, shots)
            mu_vals[name] = val
            mu_vals_corr[name] = correct_measurement(val, f_I2)
            mu_stds[name] = std
            mu_n0s[name] = n0
            mu_depths[name] = depth

        print(f"  mu2 (=I2): {mu_vals['mu2']:.4f}, mu3_raw = {mu_vals['mu3']:.4f}, mu4_raw = {mu_vals['mu4']:.4f}")
        print(f"  mu2_corr: {mu_vals_corr['mu2']:.4f}, mu3_corr = {mu_vals_corr['mu3']:.4f}, mu4_corr = {mu_vals_corr['mu4']:.4f}")

        # Compute negativity from moments (μ₂ = I₂)
        try:
            N_val = compute_negativity_from_moments(mu_vals['mu2'], mu_vals['mu3'], mu_vals['mu4'])
        except:
            N_val = 0.0

        try:
            N_corr = compute_negativity_from_moments(mu_vals_corr['mu2'], mu_vals_corr['mu3'], mu_vals_corr['mu4'])
        except:
            N_corr = 0.0

        print(f"  N_raw = {N_val:.4f}, N_corr = {N_corr:.4f} (theory: {theory['N']:.4f})")

        # Per-state M2_phys calculations (for comparison)
        # Oracle: uses known theta (not fair)
        M2_phys_oracle, p_fit_oracle = compute_M2_physical(I2_corr, theta)
        Q_phys_oracle = I2_corr**2 - M2_phys_oracle

        # Blind: fits theta from (I2, M2) measurements
        M2_phys_blind, theta_fit_blind, p_fit_blind = compute_M2_physical_blind(I2_corr, M2_corr)
        Q_phys_blind = I2_corr**2 - M2_phys_blind

        print(f"  M2_phys_oracle = {M2_phys_oracle:.4f} (using known theta)")
        print(f"  M2_phys_blind = {M2_phys_blind:.4f} (theta_fit={theta_fit_blind:.1f})")

        # Store results
        results.append({
            'theta': theta,
            'I2_meas': I2_val,
            'I2_corr': I2_corr,
            'I2_std': I2_std,
            'I2_n0': I2_n0,
            'M2_meas': M2_val,
            'M2_corr': M2_corr,
            'M2_std': M2_std,
            'M2_SS': M2_terms['SS'],
            'M2_SY': M2_terms['SY'],
            'M2_YS': M2_terms['YS'],
            'M2_YY': M2_terms['YY'],
            'M2_SS_n0': M2_n0s['SS'],
            'M2_SY_n0': M2_n0s['SY'],
            'M2_YS_n0': M2_n0s['YS'],
            'M2_YY_n0': M2_n0s['YY'],
            'Q_meas': Q_val,
            'Q_corr': Q_corr,
            'Q_std': Q_std,
            'Q_theory': Q_theory,
            'mu2': mu_vals['mu2'],
            'mu3': mu_vals['mu3'],
            'mu4': mu_vals['mu4'],
            'mu2_corr': mu_vals_corr['mu2'],
            'mu3_corr': mu_vals_corr['mu3'],
            'mu4_corr': mu_vals_corr['mu4'],
            'mu2_n0': mu_n0s['mu2'],
            'mu3_n0': mu_n0s['mu3'],
            'mu4_n0': mu_n0s['mu4'],
            'N_meas': N_val,
            'N_corr': N_corr,
            'N_theory': theory['N'],
            'I2_theory': theory['I2'],
            'M2_theory': theory['M2'],
            # Per-state ML estimations
            'M2_phys_oracle': M2_phys_oracle,
            'Q_phys_oracle': Q_phys_oracle,
            'p_fit_oracle': p_fit_oracle,
            'M2_phys_blind': M2_phys_blind,
            'Q_phys_blind': Q_phys_blind,
            'theta_fit': theta_fit_blind,
            'p_fit_blind': p_fit_blind,
        })

    # ==========================================================================
    # TWO-STAGE ML ESTIMATION
    # ==========================================================================
    print("\n" + "-" * 80)
    print("TWO-STAGE MAXIMUM LIKELIHOOD ESTIMATION")
    print("-" * 80)
    print("Stage 1: Calibrate f_I2, f_M2, p using KNOWN theta (oracle)")
    print("Stage 2: Fit theta for each state using calibrated parameters")

    # Create estimator and add all states with RAW measurements
    ml_estimator = ChiralityMaxLikEstimator()
    for r in results:
        ml_estimator.add_state(
            name=f"theta_{r['theta']}",
            theta_deg=r['theta'],
            I2_meas=r['I2_meas'],  # Use RAW measurements, not corrected
            M2_meas=r['M2_meas'],  # Use RAW measurements, not corrected
        )

    # Perform two-stage ML fit
    ml_results = ml_estimator.fit_all()

    print(f"\nStage 1 - Calibrated parameters (using known theta):")
    print(f"  f_I2 = {ml_results['f_I2']:.4f}")
    print(f"  f_M2 = {ml_results['f_M2']:.4f}")
    print(f"  p    = {ml_results['p']:.4f}")
    print(f"  Calibration success: {ml_results['calibration_success']}, NLL: {ml_results['calibration_nll']:.4f}")

    print(f"\nStage 2 - Fitted theta (using calibrated f, p):")
    print(f"{'State':>12} {'theta_true':>10} {'theta_fit':>10} {'M2_phys':>9} {'Q_phys':>9} {'Q_theory':>9} {'Error':>8}")
    print("-" * 75)
    for name, state_result in ml_results['states'].items():
        print(f"{name:>12} {state_result['theta_true_deg']:>10.1f} {state_result['theta_fit_deg']:>10.1f} "
              f"{state_result['M2_phys']:>9.4f} {state_result['Q_phys']:>9.4f} "
              f"{state_result['Q_theory']:>9.4f} {state_result['Q_error']:>8.4f}")

    print(f"\nTwo-stage ML mean error: {ml_results['mean_error']:.4f}")

    # Add ML results to each result entry
    for r in results:
        state_name = f"theta_{r['theta']}"
        if state_name in ml_results['states']:
            ml_state = ml_results['states'][state_name]
            r['theta_fit_ml'] = ml_state['theta_fit_deg']
            r['p_fit_ml'] = ml_results['p']  # Shared calibrated p
            r['I2_fit_ml'] = ml_state['I2_phys']
            r['M2_fit_ml'] = ml_state['M2_phys']
            r['Q_fit_ml'] = ml_state['Q_phys']
            r['Q_error_ml'] = ml_state['Q_error']

    # Print summary tables
    print("\n" + "=" * 80)
    print("SUMMARY TABLES FOR SI DOCUMENT")
    print("=" * 80)

    # Table 1: Chirality witness - RAW measurements
    # NOTE: Q = I2^2 - M2 (since mu2 = R2 = I2)
    print("\n--- Table: Chirality Witness RAW Measurements ---")
    print("  NOTE: Q = I2^2 - M2 (using mu2 = R2 = I2 identity)")
    print(f"{'theta':>6} {'I2_raw':>8} {'M2_raw':>8} {'Q_raw':>8} {'Q_theory':>9} {'Error':>8}")
    print("-" * 55)
    for r in results:
        error = abs(r['Q_meas'] - r['Q_theory'])
        print(f"{r['theta']:>6} {r['I2_meas']:>8.4f} {r['M2_meas']:>8.4f} "
              f"{r['Q_meas']:>8.4f} {r['Q_theory']:>9.4f} {error:>8.4f}")

    mean_error_raw = np.mean([abs(r['Q_meas'] - r['Q_theory']) for r in results])
    print(f"{'Mean':>6} {'---':>8} {'---':>8} {'---':>8} {'---':>9} {mean_error_raw:>8.4f}")

    # Table 2: Chirality witness - CORRECTED measurements
    print("\n--- Table: Chirality Witness CORRECTED Measurements ---")
    print(f"{'theta':>6} {'I2_corr':>8} {'M2_corr':>8} {'Q_corr':>8} {'Q_theory':>9} {'Error':>8}")
    print("-" * 55)
    for r in results:
        error = abs(r['Q_corr'] - r['Q_theory'])
        print(f"{r['theta']:>6} {r['I2_corr']:>8.4f} {r['M2_corr']:>8.4f} "
              f"{r['Q_corr']:>8.4f} {r['Q_theory']:>9.4f} {error:>8.4f}")

    mean_error_corr = np.mean([abs(r['Q_corr'] - r['Q_theory']) for r in results])
    print(f"{'Mean':>6} {'---':>8} {'---':>8} {'---':>8} {'---':>9} {mean_error_corr:>8.4f}")

    # Table 2b: Chirality witness with M2_phys ORACLE (uses known theta)
    print("\n--- Table: M2_phys ORACLE (uses known theta - not fair comparison) ---")
    print("  WARNING: Uses known theta, so essentially replaces M2 with theory!")
    print(f"{'theta':>6} {'M2_corr':>8} {'M2_oracle':>9} {'Q_oracle':>9} {'Q_theory':>9} {'Error':>8}")
    print("-" * 60)
    for r in results:
        error = abs(r['Q_phys_oracle'] - r['Q_theory'])
        print(f"{r['theta']:>6} {r['M2_corr']:>8.4f} {r['M2_phys_oracle']:>9.4f} "
              f"{r['Q_phys_oracle']:>9.4f} {r['Q_theory']:>9.4f} {error:>8.4f}")

    mean_error_oracle = np.mean([abs(r['Q_phys_oracle'] - r['Q_theory']) for r in results])
    print(f"{'Mean':>6} {'---':>8} {'---':>9} {'---':>9} {'---':>9} {mean_error_oracle:>8.4f}")

    # Table 2c: Chirality witness with M2_phys BLIND (fits theta from data)
    print("\n--- Table: M2_phys BLIND (fits theta from I2, M2 - fair comparison) ---")
    print(f"{'theta':>6} {'theta_fit':>9} {'M2_corr':>8} {'M2_blind':>9} {'Q_blind':>9} {'Q_theory':>9} {'Error':>8}")
    print("-" * 70)
    for r in results:
        error = abs(r['Q_phys_blind'] - r['Q_theory'])
        print(f"{r['theta']:>6} {r['theta_fit']:>9.1f} {r['M2_corr']:>8.4f} {r['M2_phys_blind']:>9.4f} "
              f"{r['Q_phys_blind']:>9.4f} {r['Q_theory']:>9.4f} {error:>8.4f}")

    mean_error_blind = np.mean([abs(r['Q_phys_blind'] - r['Q_theory']) for r in results])
    print(f"{'Mean':>6} {'---':>9} {'---':>8} {'---':>9} {'---':>9} {'---':>9} {mean_error_blind:>8.4f}")

    # Table 2d: Two-stage ML (oracle calibration + blind theta fit)
    print("\n--- Table: TWO-STAGE ML (oracle f,p calibration + blind theta fit) ---")
    print(f"  Calibrated: f_I2 = {ml_results['f_I2']:.4f}, f_M2 = {ml_results['f_M2']:.4f}, p = {ml_results['p']:.4f}")
    print(f"{'theta':>6} {'theta_ML':>9} {'M2_phys':>9} {'Q_phys':>9} {'Q_theory':>9} {'Error':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['theta']:>6} {r['theta_fit_ml']:>9.1f} {r['M2_fit_ml']:>9.4f} "
              f"{r['Q_fit_ml']:>9.4f} {r['Q_theory']:>9.4f} {r['Q_error_ml']:>8.4f}")

    mean_error_ml = ml_results['mean_error']
    print(f"{'Mean':>6} {'---':>9} {'---':>9} {'---':>9} {'---':>9} {mean_error_ml:>8.4f}")

    # Summary comparison of all methods
    print("\n" + "-" * 80)
    print("COMPARISON OF ALL CORRECTION METHODS")
    print("-" * 80)
    print(f"{'Method':<45} {'Mean Q Error':>15}")
    print("-" * 60)
    print(f"{'Raw (no correction)':<45} {mean_error_raw:>15.4f}")
    print(f"{'Simple correction (divide by f)':<45} {mean_error_corr:>15.4f}")
    print(f"{'Oracle ML (uses known theta for M2)':<45} {mean_error_oracle:>15.4f}")
    print(f"{'Blind ML (per-state theta fit, no calib)':<45} {mean_error_blind:>15.4f}")
    print(f"{'Two-stage ML (oracle f,p + blind theta)':<45} {mean_error_ml:>15.4f}")

    # Table 3: Comparison with negativity
    print("\n--- Table: Comparison with Negativity (Corrected) ---")
    print(f"{'theta':>6} {'N_theory':>9} {'N_corr':>9} {'Q_theory':>9} {'Q_corr':>9}")
    print("-" * 50)
    for r in results:
        print(f"{r['theta']:>6} {r['N_theory']:>9.4f} {r['N_corr']:>9.4f} "
              f"{r['Q_theory']:>9.4f} {r['Q_corr']:>9.4f}")

    # Table 3: Raw measurement data
    print("\n--- Table: Raw Measurement Data ---")
    print("  NOTE: mu2 = I2 (same circuit)")
    print(f"{'theta':>6} {'Circuit':>10} {'P(0)':>8} {'Shots':>8} {'Value':>10}")
    print("-" * 50)
    for r in results:
        p0_I2 = r['I2_n0'] / shots
        print(f"{r['theta']:>6} {'I2(=mu2)':>10} {p0_I2:>8.4f} {shots:>8} {r['I2_meas']:>10.4f}")
        for m2_name in ['SS', 'SY', 'YS', 'YY']:
            p0 = r[f'M2_{m2_name}_n0'] / shots
            val = r[f'M2_{m2_name}']
            print(f"{'':>6} {f'M2_{m2_name}':>10} {p0:>8.4f} {shots:>8} {val:>10.4f}")
        for mu_name in ['mu3', 'mu4']:  # mu2 = I2, already printed
            p0 = r[f'{mu_name}_n0'] / shots
            val = r[mu_name]
            print(f"{'':>6} {mu_name:>10} {p0:>8.4f} {shots:>8} {val:>10.4f}")
        print("-" * 50)

    # Output LaTeX table format
    print("\n" + "=" * 80)
    print("LATEX TABLE FORMAT FOR SI")
    print("=" * 80)

    # NOTE: Q = I₂² - M₂ (using μ₂ = R₂ = I₂ identity)
    print("\n% Chirality witness CORRECTED experimental results table")
    print("% NOTE: Q = I_2^2 - M_2 (using mu_2 = R_2 = I_2 identity)")
    print("\\begin{tabular}{@{}rccccc@{}}")
    print("\\toprule")
    print("$\\theta$ & $I_2^{\\text{corr}}$ & $M_2^{\\text{corr}}$ & "
          "$\\mathcal{Q}^{\\text{corr}}$ & $\\mathcal{Q}^{\\text{theory}}$ & Error \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\textit{Separable states}} \\\\")
    r = results[0]  # theta=0
    error = abs(r['Q_corr'] - r['Q_theory'])
    print(f"${r['theta']}^\\circ$ & {r['I2_corr']:.4f} & {r['M2_corr']:.4f} & "
          f"{r['Q_corr']:.4f} & {r['Q_theory']:.4f} & {error:.4f} \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\textit{Entangled states}} \\\\")
    for r in results[1:]:
        error = abs(r['Q_corr'] - r['Q_theory'])
        print(f"${r['theta']}^\\circ$ & {r['I2_corr']:.4f} & {r['M2_corr']:.4f} & "
              f"{r['Q_corr']:.4f} & {r['Q_theory']:.4f} & {error:.4f} \\\\")
    print("\\midrule")
    print(f"\\textbf{{Mean error}} & --- & --- & --- & --- & \\textbf{{{mean_error_corr:.4f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # LaTeX table with BLIND ML fit (fair comparison)
    print("\n% Chirality witness with BLIND ML fit (theta fitted from data)")
    print("% This is a FAIR comparison - does not use oracle knowledge of theta")
    print("\\begin{tabular}{@{}rccccccc@{}}")
    print("\\toprule")
    print("$\\theta$ & $\\theta_{\\text{fit}}$ & $M_2^{\\text{corr}}$ & $M_2^{\\text{blind}}$ & "
          "$\\mathcal{Q}^{\\text{corr}}$ & $\\mathcal{Q}^{\\text{blind}}$ & $\\mathcal{Q}^{\\text{theory}}$ & Error \\\\")
    print("\\midrule")
    print("\\multicolumn{8}{l}{\\textit{Separable states}} \\\\")
    r = results[0]  # theta=0
    error = abs(r['Q_phys_blind'] - r['Q_theory'])
    print(f"${r['theta']}^\\circ$ & {r['theta_fit']:.1f}$^\\circ$ & {r['M2_corr']:.4f} & {r['M2_phys_blind']:.4f} & "
          f"{r['Q_corr']:.4f} & {r['Q_phys_blind']:.4f} & {r['Q_theory']:.4f} & {error:.4f} \\\\")
    print("\\midrule")
    print("\\multicolumn{8}{l}{\\textit{Entangled states}} \\\\")
    for r in results[1:]:
        error = abs(r['Q_phys_blind'] - r['Q_theory'])
        print(f"${r['theta']}^\\circ$ & {r['theta_fit']:.1f}$^\\circ$ & {r['M2_corr']:.4f} & {r['M2_phys_blind']:.4f} & "
              f"{r['Q_corr']:.4f} & {r['Q_phys_blind']:.4f} & {r['Q_theory']:.4f} & {error:.4f} \\\\")
    print("\\midrule")
    print(f"\\textbf{{Mean error}} & --- & --- & --- & --- & --- & --- & \\textbf{{{mean_error_blind:.4f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # LaTeX table with JOINT ML fit (proper maxlik approach)
    mean_error_ml = ml_results['mean_error']
    print("\n% Chirality witness with TWO-STAGE ML fit")
    print("% Stage 1: Oracle calibration of f_I2, f_M2, p using known theta")
    print("% Stage 2: Blind theta fit using calibrated parameters")
    print(f"% Calibrated: f_I2 = {ml_results['f_I2']:.4f}, f_M2 = {ml_results['f_M2']:.4f}, p = {ml_results['p']:.4f}")
    print("\\begin{tabular}{@{}rccccc@{}}")
    print("\\toprule")
    print("$\\theta$ & $\\theta_{\\text{fit}}$ & $M_2^{\\text{phys}}$ & "
          "$\\mathcal{Q}^{\\text{phys}}$ & $\\mathcal{Q}^{\\text{theory}}$ & Error \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\textit{Separable states}} \\\\")
    r = results[0]  # theta=0
    print(f"${r['theta']}^\\circ$ & {r['theta_fit_ml']:.1f}$^\\circ$ & "
          f"{r['M2_fit_ml']:.4f} & {r['Q_fit_ml']:.4f} & {r['Q_theory']:.4f} & {r['Q_error_ml']:.4f} \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\textit{Entangled states}} \\\\")
    for r in results[1:]:
        print(f"${r['theta']}^\\circ$ & {r['theta_fit_ml']:.1f}$^\\circ$ & "
              f"{r['M2_fit_ml']:.4f} & {r['Q_fit_ml']:.4f} & {r['Q_theory']:.4f} & {r['Q_error_ml']:.4f} \\\\")
    print("\\midrule")
    print(f"\\textbf{{Mean error}} & --- & --- & --- & --- & \\textbf{{{mean_error_ml:.4f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # Summary comparison table
    print("\n% Summary comparison of all correction methods")
    print("\\begin{tabular}{@{}lc@{}}")
    print("\\toprule")
    print("Method & Mean Q Error \\\\")
    print("\\midrule")
    print(f"Raw (no correction) & {mean_error_raw:.4f} \\\\")
    print(f"Simple correction (divide by f) & {mean_error_corr:.4f} \\\\")
    print(f"Oracle ML (uses known $\\theta$ for $M_2$) & {mean_error_oracle:.4f} \\\\")
    print(f"Blind ML (per-state $\\theta$ fit, no calibration) & {mean_error_blind:.4f} \\\\")
    print(f"Two-stage ML (oracle $f,p$ + blind $\\theta$) & {mean_error_ml:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    print("\n% Comparison with negativity table (corrected values)")
    print("\\begin{tabular}{@{}rcccc@{}}")
    print("\\toprule")
    print("$\\theta$ & $\\mathcal{N}^{\\text{theory}}$ & $\\mathcal{N}^{\\text{corr}}$ & "
          "$\\mathcal{Q}^{\\text{theory}}$ & $\\mathcal{Q}^{\\text{corr}}$ \\\\")
    print("\\midrule")
    for r in results:
        print(f"${r['theta']}^\\circ$ & {r['N_theory']:.4f} & {r['N_corr']:.4f} & "
              f"{r['Q_theory']:.4f} & {r['Q_corr']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # Also output RAW measurements table
    print("\n% Chirality witness RAW experimental results table")
    print("% NOTE: Q = I_2^2 - M_2 (using mu_2 = R_2 = I_2 identity)")
    print("\\begin{tabular}{@{}rccccc@{}}")
    print("\\toprule")
    print("$\\theta$ & $I_2^{\\text{raw}}$ & $M_2^{\\text{raw}}$ & "
          "$\\mathcal{Q}^{\\text{raw}}$ & $\\mathcal{Q}^{\\text{theory}}$ & Error \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\textit{Separable states}} \\\\")
    r = results[0]  # theta=0
    error = abs(r['Q_meas'] - r['Q_theory'])
    print(f"${r['theta']}^\\circ$ & {r['I2_meas']:.4f} & {r['M2_meas']:.4f} & "
          f"{r['Q_meas']:.4f} & {r['Q_theory']:.4f} & {error:.4f} \\\\")
    print("\\midrule")
    print("\\multicolumn{6}{l}{\\textit{Entangled states}} \\\\")
    for r in results[1:]:
        error = abs(r['Q_meas'] - r['Q_theory'])
        print(f"${r['theta']}^\\circ$ & {r['I2_meas']:.4f} & {r['M2_meas']:.4f} & "
              f"{r['Q_meas']:.4f} & {r['Q_theory']:.4f} & {error:.4f} \\\\")
    print("\\midrule")
    print(f"\\textbf{{Mean error}} & --- & --- & --- & --- & \\textbf{{{mean_error_raw:.4f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
