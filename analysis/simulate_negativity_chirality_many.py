#!/usr/bin/env python3
"""
Simulation of negativity and chirality for many states (ideal + noisy).

Includes:
  - Pure parametrized states: |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>
  - Mixed Werner states: rho_W(p) = p|Psi^-><Psi^-| + (1-p)I/4

For pure states, chirality -C_4 = I_2^2 - mu_4 (since I_4 = 1 for pure).
For mixed states, -C_4 = I_4 - mu_4 (general definition).

Mixed state moments are measured via spectral decomposition (same approach
as SpectralMeasurer in ibmq_two_feature_batched.py):
  mu_k = sum_{i1...ik} lambda_{i1}...lambda_{ik} * hadamard_test(e_{i1},...,e_{ik})

Usage:
    python simulate_negativity_chirality_many.py [--shots 8192] [--output results.json]
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product as iterproduct
from math import sqrt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add grobner-negativity-chirality package to path
PACKAGE_DIR = Path(__file__).resolve().parent.parent / "grobner-negativity-chirality"
sys.path.insert(0, str(PACKAGE_DIR))

from simulations import (
    create_parametrized_state,
    compute_theoretical_values,
    create_I2_circuit,
    create_mu_circuits,
    run_circuit,
    correct_measurement,
    compute_negativity_from_moments,
    parse_calibration_csv,
)
from scipy.optimize import minimize_scalar, differential_evolution
from negativity.qubit_qubit.analysis import (
    pt_eigenvalues_pure, moments_from_eigenvalues, negativity_from_eigenvalues,
)
from negativity.qubit_qubit.maxlik import NegativityMaxLikEstimator
from chirality.qubit_qubit.maxlik import ChiralityMaxLikEstimator
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Publication style matching pgfplots template
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': 'gray',
    'legend.fontsize': 9,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Unified colour palette (matches pgfplots blue!70!black, red!70!black, green!60!black)
_BLUE = '#0000B3'
_RED = '#B30000'
_GREEN = '#009900'


# ========================================================================
# Checkpointing
# ========================================================================

CHECKPOINT_FILE = Path(__file__).parent / "sim_negativity_chirality_checkpoint.json"


def save_checkpoint(data, stage):
    """Save checkpoint after completing a stage."""
    data['completed_stage'] = stage
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [Checkpoint saved: {stage}]")


def load_checkpoint(shots, thetas, p_values):
    """Load checkpoint if it exists and config matches."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            cp = json.load(f)
        cfg = cp.get('config', {})
        if (cfg.get('shots') == shots
                and cfg.get('thetas') == thetas
                and cfg.get('p_values') == p_values):
            stage = cp.get('completed_stage', '')
            print(f"  Loaded checkpoint (last stage: {stage})")
            return cp
        else:
            print("  Checkpoint config mismatch — starting fresh")
            return None
    except Exception as e:
        print(f"  Checkpoint load error: {e} — starting fresh")
        return None


# ========================================================================
# Noisy backend from calibration CSV (real topology)
# ========================================================================

def create_noisy_backend_from_csv(cal_file, num_qubits=10):
    """Create AerSimulator with noise model and coupling map from CSV.

    Selects a connected subgraph of the best qubits (by readout error)
    via BFS, then builds the noise model with 2-qubit errors placed on
    the *actual* CZ edges from the calibration data.
    """
    calibrations, coupling_edges = parse_calibration_csv(cal_file)

    # Adjacency for operational qubits
    operational = {q for q, c in calibrations.items()
                   if c['operational'] and c['t1_us'] > 10}
    adj = {q: set() for q in operational}
    for q1, q2 in coupling_edges:
        if q1 in operational and q2 in operational:
            adj[q1].add(q2)
            adj[q2].add(q1)

    # BFS from best qubit to get a connected subgraph
    best_by_readout = sorted(operational,
                             key=lambda q: calibrations[q]['readout_error'])
    seed = best_by_readout[0]
    selected = []
    visited = {seed}
    queue = [seed]
    while queue and len(selected) < num_qubits:
        q = queue.pop(0)
        selected.append(q)
        for n in sorted(adj[q] - visited,
                        key=lambda n: calibrations[n]['readout_error']):
            visited.add(n)
            queue.append(n)

    qubit_map = {q: i for i, q in enumerate(selected)}
    qubit_set = set(selected)

    # Edges between selected qubits (mapped indices)
    edges = set()
    for q1, q2 in coupling_edges:
        if q1 in qubit_set and q2 in qubit_set:
            edges.add((qubit_map[q1], qubit_map[q2]))

    # Coupling map
    cmap = CouplingMap()
    for i in range(len(selected)):
        cmap.add_physical_qubit(i)
    for q1, q2 in edges:
        cmap.add_edge(q1, q2)
        cmap.add_edge(q2, q1)

    # Noise model
    noise_model = NoiseModel()
    gate_time_1q = 32e-9   # seconds
    gate_time_2q = 68e-9

    for i, qid in enumerate(selected):
        cal = calibrations[qid]
        t1 = cal['t1_us'] * 1e-6
        t2 = min(cal['t2_us'] * 1e-6, 2 * t1)

        # Single-qubit thermal relaxation + depolarising
        th1 = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_quantum_error(th1, ['sx', 'x'], [i])
        noise_model.add_quantum_error(
            depolarizing_error(max(cal['sx_error'], 1e-4), 1), 'sx', [i])
        noise_model.add_quantum_error(
            depolarizing_error(max(cal['x_error'], 1e-4), 1), 'x', [i])

        # Readout error
        p0g1 = cal['prob_m0_p1'] or cal['readout_error']
        p1g0 = cal['prob_m1_p0'] or cal['readout_error']
        if p0g1 > 0 or p1g0 > 0:
            noise_model.add_readout_error(
                ReadoutError([[1 - p1g0, p1g0], [p0g1, 1 - p0g1]]), [i])

    # Two-qubit errors on actual edges
    for q1m, q2m in edges:
        q1_orig = selected[q1m]
        q2_orig = selected[q2m]
        cal = calibrations[q1_orig]

        t1 = cal['t1_us'] * 1e-6
        t2 = min(cal['t2_us'] * 1e-6, 2 * t1)
        th2 = thermal_relaxation_error(t1, t2, gate_time_2q)
        th2 = th2.tensor(th2)

        cz_err = cal.get('cz_errors', {}).get(q2_orig, 0.01)
        if cz_err <= 0:
            cz_err = 0.01
        combined = th2.compose(depolarizing_error(cz_err, 2))

        noise_model.add_quantum_error(
            combined, ['cx', 'cz', 'swap'], [q1m, q2m])
        noise_model.add_quantum_error(
            combined, ['cx', 'cz', 'swap'], [q2m, q1m])

    print(f"  Selected qubits: {selected}")
    print(f"  Edges: {len(edges)}, connected: {cmap.is_connected()}")

    return AerSimulator(noise_model=noise_model, coupling_map=cmap)


# ========================================================================
# Mixed state utilities
# ========================================================================

def create_werner_state(p):
    """Create Werner state rho_W(p) = p|Psi^-><Psi^-| + (1-p)I/4."""
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    rho = p * np.outer(psi_minus, psi_minus.conj()) + (1 - p) / 4 * np.eye(4)
    return rho


def werner_pt_moments(p):
    """Exact PT moments mu2, mu3, mu4 for Werner state rho_W(p).

    PT eigenvalues: [(1+p)/4, (1+p)/4, (1+p)/4, (1-3p)/4]
    """
    a = (1 + p) / 4   # triply degenerate
    b = (1 - 3*p) / 4
    return 3*a**2 + b**2, 3*a**3 + b**3, 3*a**4 + b**4


def create_horodecki_state(p):
    """Create Horodecki state rho_H(p) = p|Psi^-><Psi^-| + (1-p)|00><00|."""
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    rho = (p * np.outer(psi_minus, psi_minus.conj())
           + (1 - p) * np.outer(ket00, ket00.conj()))
    return rho


def horodecki_pt_moments(p):
    """Exact PT moments mu2, mu3, mu4 for Horodecki state rho_H(p).

    PT eigenvalues: p/2 (x2), [(1-p) +/- sqrt(1-2p+2p^2)] / 2
    """
    disc = np.sqrt(1 - 2*p + 2*p**2)
    a = p / 2            # doubly degenerate
    b = ((1 - p) + disc) / 2
    c = ((1 - p) - disc) / 2
    return (2*a**2 + b**2 + c**2,
            2*a**3 + b**3 + c**3,
            2*a**4 + b**4 + c**4)


def partial_transpose_A(rho, dA=2, dB=2):
    """Partial transpose over subsystem A."""
    rho_pt = np.zeros_like(rho)
    for iA in range(dA):
        for jA in range(dA):
            for iB in range(dB):
                for jB in range(dB):
                    rho_pt[jA * dB + iB, iA * dB + jB] = rho[iA * dB + iB, jA * dB + jB]
    return rho_pt


def compute_theory_mixed(rho):
    """Compute theoretical N and C4 from density matrix."""
    rho_pt = partial_transpose_A(rho)
    eigvals_pt = np.linalg.eigvalsh(rho_pt)

    N = float(-sum(e for e in eigvals_pt if e < -1e-12))
    mu2 = float(sum(e**2 for e in eigvals_pt))
    mu3 = float(sum(e**3 for e in eigvals_pt))
    mu4 = float(sum(e**4 for e in eigvals_pt))

    eigvals_rho = np.linalg.eigvalsh(rho)
    I2 = float(sum(e**2 for e in eigvals_rho))
    I4 = float(sum(e**4 for e in eigvals_rho))

    # -C4 = I4 - mu4 (positive for entangled states)
    neg_C4 = I4 - mu4

    return {
        'N': N, 'neg_C4': neg_C4,
        'mu2': mu2, 'mu3': mu3, 'mu4': mu4,
        'I2': I2, 'I4': I4,
    }


def eigendecompose(rho, max_eigvals=4):
    """Eigendecompose rho, keep top eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    active = [(float(eigvals[i]), eigvecs[:, i])
              for i in np.argsort(-eigvals) if eigvals[i] > 1e-10]
    return active[:max_eigvals]


# ========================================================================
# Mixed state circuit creation (different states per copy)
# ========================================================================

def create_mu3_circuit_mixed(psi_1, psi_2, psi_3):
    """mu3 Hadamard test with different states in 3 copies.
    Layout: copies (0,1), (2,3), (4,5); ancilla 6."""
    qc = QuantumCircuit(7, 1, name="mu3_m")
    qc.initialize(psi_1, [0, 1])
    qc.initialize(psi_2, [2, 3])
    qc.initialize(psi_3, [4, 5])
    qc.h(6)
    # Forward cycle on A: A1->A2->A3->A1
    qc.cswap(6, 0, 2)
    qc.cswap(6, 2, 4)
    # Backward cycle on B: B3->B2->B1->B3
    qc.cswap(6, 5, 3)
    qc.cswap(6, 3, 1)
    qc.h(6)
    qc.measure(6, 0)
    return qc


def create_mu4_circuit_mixed(psi_1, psi_2, psi_3, psi_4):
    """mu4 Hadamard test with different states in 4 copies.
    Layout: copies (0,1), (2,3), (4,5), (6,7); ancilla 8."""
    qc = QuantumCircuit(9, 1, name="mu4_m")
    qc.initialize(psi_1, [0, 1])
    qc.initialize(psi_2, [2, 3])
    qc.initialize(psi_3, [4, 5])
    qc.initialize(psi_4, [6, 7])
    qc.h(8)
    # Forward cycle on A: A1->A2->A3->A4->A1
    qc.cswap(8, 0, 2)
    qc.cswap(8, 2, 4)
    qc.cswap(8, 4, 6)
    # Backward cycle on B: B4->B3->B2->B1->B4
    qc.cswap(8, 7, 5)
    qc.cswap(8, 5, 3)
    qc.cswap(8, 3, 1)
    qc.h(8)
    qc.measure(8, 0)
    return qc


# ========================================================================
# Spectral decomposition measurement (like SpectralMeasurer for BE)
# ========================================================================

def _prepare_mixed_circuits(args):
    """Create all circuits for a single mixed state (worker function).

    Args:
        args: tuple (p, state_fn) where state_fn(p) returns rho.

    Returns metadata dict with circuits, weights, moment types, and exact
    eigenvalue-derived quantities (I2, I4, mu2).
    """
    p, state_fn = args
    rho = state_fn(p)
    theory = compute_theory_mixed(rho)
    active = eigendecompose(rho)
    r = len(active)

    I2 = sum(l**2 for l, _ in active)
    I4 = sum(l**4 for l, _ in active)
    mu2 = I2

    circuits = []
    weights = []
    moment_types = []

    for idx_tuple in iterproduct(range(r), repeat=3):
        w = 1.0
        states = []
        for idx in idx_tuple:
            w *= active[idx][0]
            states.append(active[idx][1])
        if abs(w) < 1e-15:
            continue
        circuits.append(create_mu3_circuit_mixed(*states))
        weights.append(w)
        moment_types.append('mu3')

    for idx_tuple in iterproduct(range(r), repeat=4):
        w = 1.0
        states = []
        for idx in idx_tuple:
            w *= active[idx][0]
            states.append(active[idx][1])
        if abs(w) < 1e-15:
            continue
        circuits.append(create_mu4_circuit_mixed(*states))
        weights.append(w)
        moment_types.append('mu4')

    return {
        'p': p, 'theory': theory, 'circuits': circuits,
        'weights': weights, 'moment_types': moment_types,
        'mu2': mu2, 'I2': I2, 'I4': I4,
    }


# ========================================================================
# Physical state projection (ML moment matching before Newton-Girard)
# ========================================================================

def fit_physical_pure_state(mu2, mu3, mu4):
    """Map measured moments to nearest physical pure state |psi(theta)>.

    Fits theta, computes physical moments on the state manifold,
    then applies Newton-Girard to the physical moments.
    This removes shot-noise artefacts that would otherwise be amplified
    by the polynomial solver.
    """
    def cost(theta_rad):
        eigs = pt_eigenvalues_pure(theta_rad)
        m2, m3, m4 = moments_from_eigenvalues(eigs)
        return (mu2 - m2)**2 + (mu3 - m3)**2 + (mu4 - m4)**2

    res = minimize_scalar(cost, bounds=(0.001, np.pi - 0.001), method='bounded')
    theta_fit = res.x

    eigs = pt_eigenvalues_pure(theta_fit)
    mu2_p, mu3_p, mu4_p = moments_from_eigenvalues(eigs)
    N = compute_negativity_from_moments(float(mu2_p), float(mu3_p), float(mu4_p))
    C = 1.0 - float(mu4_p)  # -C4 = I4 - mu4, I4 = 1 for pure states

    return N, C, float(np.degrees(theta_fit))


def fit_physical_werner_state(mu2, mu3, mu4, I4):
    """Map measured moments to nearest physical Werner state rho_W(p).

    Fits p, computes physical moments, then Newton-Girard.
    """
    def cost(p):
        m2, m3, m4 = werner_pt_moments(p)
        return (mu2 - m2)**2 + (mu3 - m3)**2 + (mu4 - m4)**2

    res = minimize_scalar(cost, bounds=(0.0, 1.0), method='bounded')
    p_fit = res.x

    mu2_p, mu3_p, mu4_p = werner_pt_moments(p_fit)
    N = compute_negativity_from_moments(float(mu2_p), float(mu3_p), float(mu4_p))
    neg_C4 = I4 - float(mu4_p)

    return N, neg_C4, p_fit


def fit_physical_horodecki_state(mu2, mu3, mu4, I4):
    """Map measured moments to nearest physical Horodecki state rho_H(p).

    Fits p, computes physical moments, then Newton-Girard.
    """
    def cost(p):
        m2, m3, m4 = horodecki_pt_moments(p)
        return (mu2 - m2)**2 + (mu3 - m3)**2 + (mu4 - m4)**2

    res = minimize_scalar(cost, bounds=(0.0, 1.0), method='bounded')
    p_fit = res.x

    mu2_p, mu3_p, mu4_p = horodecki_pt_moments(p_fit)
    N = compute_negativity_from_moments(float(mu2_p), float(mu3_p), float(mu4_p))
    neg_C4 = I4 - float(mu4_p)

    return N, neg_C4, p_fit


# ========================================================================
# Pure state pipeline
# ========================================================================

def _create_pure_circuits(theta):
    """Create I2, mu3, mu4 circuits for a single theta (worker function)."""
    psi = create_parametrized_state(theta)
    I2_c = create_I2_circuit(psi)
    mu_cs = create_mu_circuits(psi)
    return I2_c, mu_cs['mu3'], mu_cs['mu4']


def _batch_run(compiled, backend, shots, desc="  Running", verbose=True):
    """Execute pre-compiled circuits in batches, return expectation values."""
    BATCH_SIZE = 300
    exp_vals = []
    n_batches = (len(compiled) + BATCH_SIZE - 1) // BATCH_SIZE
    for start in tqdm(range(0, len(compiled), BATCH_SIZE),
                      total=n_batches, desc=desc,
                      unit="batch", disable=not verbose):
        batch = compiled[start:start + BATCH_SIZE]
        job = backend.run(batch, shots=shots)
        result = job.result()
        for j in range(len(batch)):
            counts = result.get_counts(j)
            n0 = counts.get('0', counts.get('0x0', 0))
            n1 = counts.get('1', counts.get('0x1', 0))
            total = n0 + n1
            exp_vals.append(2 * n0 / total - 1)
    return exp_vals


def run_pure_mode(thetas, backend, shots, calibrate=True, verbose=True,
                  n_workers=4):
    """Run measurement pipeline for pure parametrized states.

    Optimised: parallel circuit creation → batch transpile → batch run.
    """
    if calibrate:
        psi_bell = create_parametrized_state(90)
        cal_circs = [create_I2_circuit(psi_bell), create_mu_circuits(psi_bell)['mu4']]
        cal_compiled = transpile(cal_circs, backend, optimization_level=1)
        cal_job = backend.run(cal_compiled, shots=shots)
        cal_result = cal_job.result()
        for idx, tag in enumerate(['I2_cal', 'mu4_cal']):
            counts = cal_result.get_counts(idx)
            n0 = counts.get('0', counts.get('0x0', 0))
            n1 = counts.get('1', counts.get('0x1', 0))
            if tag == 'I2_cal':
                f_I2 = 2 * n0 / (n0 + n1) - 1
            else:
                f_mu4 = (2 * n0 / (n0 + n1) - 1) / 0.25
        if verbose:
            print(f"  Calibration: f_I2={f_I2:.4f}, f_mu4={f_mu4:.4f}")
    else:
        f_I2 = 1.0
        f_mu4 = 1.0

    # Step 1: Create circuits in parallel
    if verbose:
        print(f"  Creating {len(thetas)*3} circuits ({n_workers} workers)...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        circuit_sets = list(pool.map(_create_pure_circuits, thetas))

    all_circuits = []
    for I2_c, mu3_c, mu4_c in circuit_sets:
        all_circuits.extend([I2_c, mu3_c, mu4_c])

    # Step 2: Batch transpile
    if verbose:
        print(f"  Transpiling {len(all_circuits)} circuits...")
    compiled = transpile(all_circuits, backend, optimization_level=1)

    # Step 3: Batch run
    exp_vals = _batch_run(compiled, backend, shots, verbose=verbose)

    # Step 4: Parse results + post-process
    results = []
    for i, theta in enumerate(thetas):
        theory = compute_theoretical_values(theta)
        I2_val = exp_vals[3 * i]
        mu3_val = exp_vals[3 * i + 1]
        mu4_val = exp_vals[3 * i + 2]

        I2_corr = correct_measurement(I2_val, f_I2)
        mu3_corr = correct_measurement(mu3_val, f_I2)
        mu4_corr = correct_measurement(mu4_val, f_mu4)

        C_raw = I2_val**2 - mu4_val
        try:
            N_raw = compute_negativity_from_moments(I2_val, mu3_val, mu4_val)
        except Exception:
            N_raw = 0.0

        # Physical projection for NEGATIVITY only (removes Newton-Girard
        # shot-noise amplification).  Chirality uses direct formula since
        # pure-state projection worsens noisy chirality estimates.
        try:
            N_corr, _, _ = fit_physical_pure_state(
                I2_corr, mu3_corr, mu4_corr)
        except Exception:
            N_corr = 0.0
        C_corr = I2_corr**2 - mu4_corr

        results.append({
            'theta': theta,
            'I2_raw': I2_val, 'I2_corr': I2_corr,
            'mu3_raw': mu3_val, 'mu3_corr': mu3_corr,
            'mu4_raw': mu4_val, 'mu4_corr': mu4_corr,
            'C_raw': C_raw, 'C_corr': C_corr, 'C_theory': theory['Q'],
            'N_raw': N_raw, 'N_corr': N_corr, 'N_theory': theory['N'],
        })
    return results, f_I2, f_mu4


# ========================================================================
# Werner state pipeline (spectral decomposition)
# ========================================================================

def run_mixed_mode(p_values, state_fn, fit_fn, backend, shots,
                   f_I2=1.0, f_mu4=1.0, verbose=True, n_workers=4,
                   label="Mixed"):
    """Run mixed-state pipeline with parallel circuit creation + batched execution.

    Generic driver for any 1-parameter mixed state family (Werner, Horodecki, ...).

    Args:
        state_fn: callable(p) -> rho (density matrix)
        fit_fn:   callable(mu2, mu3, mu4, I4) -> (N, neg_C4, p_fit)
    """
    # Step 1: Create circuits for all states in parallel
    if verbose:
        print(f"  Creating circuits for {len(p_values)} {label} states "
              f"({n_workers} workers)...")
    args_list = [(p, state_fn) for p in p_values]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        mixed_data = list(pool.map(_prepare_mixed_circuits, args_list))

    total_circuits = sum(len(md['circuits']) for md in mixed_data)
    if verbose:
        for md in mixed_data:
            n3 = sum(1 for t in md['moment_types'] if t == 'mu3')
            n4 = sum(1 for t in md['moment_types'] if t == 'mu4')
            print(f"    p={md['p']:.3f}: {n3} mu3 + {n4} mu4 "
                  f"= {len(md['circuits'])} circuits")
        print(f"  Total: {total_circuits} circuits")

    # Step 2: Flatten all circuits, transpile once, run in batches
    all_circuits = []
    offsets = []
    for md in mixed_data:
        offsets.append((len(all_circuits), len(md['circuits'])))
        all_circuits.extend(md['circuits'])

    if verbose:
        print(f"  Transpiling {len(all_circuits)} circuits...")
    compiled = transpile(all_circuits, backend, optimization_level=1)

    exp_vals = _batch_run(compiled, backend, shots, verbose=verbose)

    # Step 3: Parse results + post-process
    results = []
    for md, (offset, count) in zip(mixed_data, offsets):
        state_exp_vals = exp_vals[offset:offset + count]

        mu3 = 0.0
        mu4 = 0.0
        for w, mt, ev in zip(md['weights'], md['moment_types'], state_exp_vals):
            if mt == 'mu3':
                mu3 += w * ev
            else:
                mu4 += w * ev

        theory = md['theory']
        mu3_corr = correct_measurement(mu3, f_I2)
        mu4_corr = correct_measurement(mu4, f_mu4)

        neg_C4_raw = md['I4'] - mu4
        try:
            N_raw = compute_negativity_from_moments(md['mu2'], mu3, mu4)
        except Exception:
            N_raw = 0.0

        # Physical projection for NEGATIVITY only
        try:
            N_corr, _, _ = fit_fn(
                md['mu2'], mu3_corr, mu4_corr, md['I4'])
        except Exception:
            N_corr = 0.0
        # Direct chirality (better for noisy data)
        neg_C4_corr = md['I4'] - mu4_corr

        results.append({
            'p': md['p'],
            'mu2': md['mu2'],
            'mu3_raw': mu3, 'mu3_corr': mu3_corr,
            'mu4_raw': mu4, 'mu4_corr': mu4_corr,
            'I2': md['I2'], 'I4': md['I4'],
            'neg_C4_raw': neg_C4_raw, 'neg_C4_corr': neg_C4_corr,
            'neg_C4_theory': theory['neg_C4'],
            'N_raw': N_raw, 'N_corr': N_corr, 'N_theory': theory['N'],
        })
    return results


# ========================================================================
# ML calibration post-processing (pure states only)
# ========================================================================

def apply_ml_calibration(results, verbose=True):
    """Apply two-stage ML calibration to pure state measurements.

    Stage 1: Jointly fit f2, f3, f4, p across ALL states using known theta
    Stage 2: For each state, fit theta -> N_phys, Q_phys

    Adds 'N_ml' and 'C_ml' keys to each result entry.

    Returns:
        (results, ml_params) where ml_params = {'f2', 'f3', 'f4', 'p'}
    """
    # Negativity ML
    neg_est = NegativityMaxLikEstimator()
    for r in results:
        neg_est.add_state(
            name=f"theta_{r['theta']:.1f}",
            theta_deg=r['theta'],
            mu2_meas=r['I2_raw'],   # mu2 = I2 for all bipartite states
            mu3_meas=r['mu3_raw'],
            mu4_meas=r['mu4_raw'],
        )
    neg_results = neg_est.fit_all()

    # Chirality ML (uses I2 and mu4 as M2)
    chi_est = ChiralityMaxLikEstimator()
    for r in results:
        chi_est.add_state(
            name=f"theta_{r['theta']:.1f}",
            theta_deg=r['theta'],
            I2_meas=r['I2_raw'],
            M2_meas=r['mu4_raw'],   # M2 = mu4 for this family
        )
    chi_results = chi_est.fit_all()

    # Add ML results to each entry
    for r in results:
        name = f"theta_{r['theta']:.1f}"
        r['N_ml'] = neg_results['states'][name]['N_phys']
        r['C_ml'] = chi_results['states'][name]['Q_phys']

    ml_params = {
        'f2': neg_results['f2'],
        'f3': neg_results['f3'],
        'f4': neg_results['f4'],
        'p': neg_results['p'],
    }

    if verbose:
        print(f"  ML Negativity: f2={ml_params['f2']:.4f}, "
              f"f3={ml_params['f3']:.4f}, f4={ml_params['f4']:.4f}, "
              f"p={ml_params['p']:.4f}")
        print(f"  ML Chirality:  f_I2={chi_results['f_I2']:.4f}, "
              f"f_M2={chi_results['f_M2']:.4f}, p={chi_results['p']:.4f}")

    return results, ml_params


def apply_ml_correction_werner(results, ml_params=None, verbose=True):
    """Werner-specific two-stage ML calibration.

    Stage 1 (calibration): Jointly fit f3^W, f4^W, p_hw across ALL
    Werner states using known p values.  Model:
        mu_k_meas = f_k * mu_k_werner(p_eff)
        p_eff = (1 - p_hw) * p_true
    where p_hw captures additional hardware depolarisation on top of
    the intended Werner mixing.

    Stage 2 (estimation): For each state, correct with fitted f_k,
    map to nearest physical Werner state, then Newton-Girard.

    This calibrates on the spectral-decomposition circuits themselves
    rather than borrowing factors from different (pure-state) circuits.
    """
    # ------ Stage 1: joint calibration on Werner data ------
    def calibration_nll(params):
        f3, f4, p_hw = params
        if f3 <= 0 or f4 <= 0 or f3 > 2 or f4 > 2:
            return 1e10
        if p_hw < 0 or p_hw > 0.5:
            return 1e10
        cost = 0.0
        for r in results:
            p_eff = r['p'] * (1 - p_hw)
            _, mu3_model, mu4_model = werner_pt_moments(p_eff)
            cost += (r['mu3_raw'] - f3 * mu3_model)**2
            cost += (r['mu4_raw'] - f4 * mu4_model)**2
        return cost

    bounds = [
        (0.3, 1.5),   # f3
        (0.3, 1.5),   # f4
        (0.0, 0.4),   # p_hw (extra depolarisation)
    ]
    opt = differential_evolution(
        calibration_nll, bounds, seed=42, maxiter=1000, tol=1e-12, polish=True)
    f3, f4, p_hw = opt.x

    if verbose:
        print(f"  Werner ML (Stage 1): f3={f3:.4f}, f4={f4:.4f}, "
              f"p_hw={p_hw:.4f}")

    # ------ Stage 2: per-state correction + physical projection ------
    for r in results:
        # Correct raw moments
        mu3_corr = r['mu3_raw'] / f3 if f3 > 0 else r['mu3_raw']
        mu4_corr = r['mu4_raw'] / f4 if f4 > 0 else r['mu4_raw']

        # Fit p_eff from corrected moments
        def cost(p, _mu3=mu3_corr, _mu4=mu4_corr):
            _, m3, m4 = werner_pt_moments(p)
            return (_mu3 - m3)**2 + (_mu4 - m4)**2

        res = minimize_scalar(cost, bounds=(0.0, 1.0), method='bounded')
        p_eff = res.x

        # Physical moments from fitted Werner model
        mu2_phys, mu3_phys, mu4_phys = werner_pt_moments(p_eff)

        # Negativity via Newton-Girard on physical moments
        try:
            N_ml = compute_negativity_from_moments(mu2_phys, mu3_phys, mu4_phys)
        except Exception:
            N_ml = 0.0

        r['N_ml'] = N_ml
        r['p_eff'] = p_eff
        r['mu2_phys'] = mu2_phys
        r['mu3_phys'] = mu3_phys
        r['mu4_phys'] = mu4_phys

        if verbose:
            print(f"    p={r['p']:.2f}: p_eff={p_eff:.4f}, "
                  f"N_ml={N_ml:.4f} (theory={r['N_theory']:.4f})")

    return results


def apply_ml_correction_horodecki(results, verbose=True):
    """Horodecki-specific two-stage ML calibration.

    Same structure as Werner but uses Horodecki model:
      rho_H(p) = p|Psi^-><Psi^-| + (1-p)|00><00|

    Hardware depolarisation mixes towards I/4, so the noisy state is
    (1-p_hw)*rho_H(p) + p_hw*I/4, which is NOT another Horodecki state.
    We compute PT moments of the depolarised state numerically.
    """
    # ------ Stage 1: joint calibration on Horodecki data ------
    def calibration_nll(params):
        f3, f4, p_hw = params
        if f3 <= 0 or f4 <= 0 or f3 > 2 or f4 > 2:
            return 1e10
        if p_hw < 0 or p_hw > 0.5:
            return 1e10
        cost = 0.0
        for r in results:
            # Depolarised state: (1-p_hw)*rho_H(p) + p_hw*I/4
            rho = ((1 - p_hw) * create_horodecki_state(r['p'])
                   + p_hw / 4 * np.eye(4))
            rho_pt = partial_transpose_A(rho)
            eigs = np.linalg.eigvalsh(rho_pt)
            mu3_model = float(np.sum(eigs**3))
            mu4_model = float(np.sum(eigs**4))
            cost += (r['mu3_raw'] - f3 * mu3_model)**2
            cost += (r['mu4_raw'] - f4 * mu4_model)**2
        return cost

    bounds = [
        (0.3, 1.5),   # f3
        (0.3, 1.5),   # f4
        (0.0, 0.4),   # p_hw
    ]
    opt = differential_evolution(
        calibration_nll, bounds, seed=42, maxiter=1000, tol=1e-12, polish=True)
    f3, f4, p_hw = opt.x

    if verbose:
        print(f"  Horodecki ML (Stage 1): f3={f3:.4f}, f4={f4:.4f}, "
              f"p_hw={p_hw:.4f}")

    # ------ Stage 2: per-state correction + physical projection ------
    for r in results:
        mu3_corr = r['mu3_raw'] / f3 if f3 > 0 else r['mu3_raw']
        mu4_corr = r['mu4_raw'] / f4 if f4 > 0 else r['mu4_raw']

        def cost(p, _mu3=mu3_corr, _mu4=mu4_corr):
            _, m3, m4 = horodecki_pt_moments(p)
            return (_mu3 - m3)**2 + (_mu4 - m4)**2

        res = minimize_scalar(cost, bounds=(0.0, 1.0), method='bounded')
        p_eff = res.x

        mu2_phys, mu3_phys, mu4_phys = horodecki_pt_moments(p_eff)

        try:
            N_ml = compute_negativity_from_moments(mu2_phys, mu3_phys, mu4_phys)
        except Exception:
            N_ml = 0.0

        r['N_ml'] = N_ml
        r['p_eff'] = p_eff

        if verbose:
            print(f"    p={r['p']:.2f}: p_eff={p_eff:.4f}, "
                  f"N_ml={N_ml:.4f} (theory={r['N_theory']:.4f})")

    return results


# ========================================================================
# Figure generation
# ========================================================================

def make_figure(pure_ideal, pure_noisy, werner_ideal, werner_noisy,
                horodecki_ideal, horodecki_noisy, output_path):
    """Generate 3x2 figure: pure (top) + Werner (mid) + Horodecki (bottom)."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 12.5))
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = axes

    # ----- Row 1: Pure states -----
    thetas_i = [r['theta'] for r in pure_ideal]
    thetas_n = [r['theta'] for r in pure_noisy]
    theta_fine = np.linspace(0, 90, 200)

    N_theory = np.abs(np.sin(np.radians(theta_fine))) / 2
    c = np.cos(np.radians(theta_fine) / 2)
    s = np.sin(np.radians(theta_fine) / 2)
    C_theory = 1.0 - (c**4 + s**4)**2

    has_ml = 'N_ml' in pure_noisy[0]
    N_key = 'N_ml' if has_ml else 'N_corr'
    C_key = 'C_corr'
    noisy_N_label = 'Noisy + ML cal.' if has_ml else 'Noisy simulator (Torino)'
    noisy_C_label = 'Noisy simulator (Torino)'

    ax1.plot(theta_fine, N_theory, 'k-', linewidth=1.5, label='Theory', zorder=1)
    ax1.scatter(thetas_i, [r['N_corr'] for r in pure_ideal],
                marker='o', facecolors='none', edgecolors=_BLUE, s=40,
                linewidths=1.2, label='Ideal simulator', zorder=3)
    ax1.scatter(thetas_n, [r[N_key] for r in pure_noisy],
                marker='o', color=_RED, s=30, alpha=0.8,
                label=noisy_N_label, zorder=2)
    ax1.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax1.set_ylabel(r'Negativity $\mathcal{N}$', fontsize=12)
    ax1.set_xlim(-2, 92)
    ax1.set_ylim(-0.05, 0.6)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_title('Pure states', fontsize=11)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=13,
             fontweight='bold', va='top')

    ax2.plot(theta_fine, C_theory, 'k-', linewidth=1.5, label='Theory', zorder=1)
    ax2.scatter(thetas_i, [r['C_corr'] for r in pure_ideal],
                marker='o', facecolors='none', edgecolors=_BLUE, s=40,
                linewidths=1.2, label='Ideal simulator', zorder=3)
    ax2.scatter(thetas_n, [r[C_key] for r in pure_noisy],
                marker='o', color=_RED, s=30, alpha=0.8,
                label=noisy_C_label, zorder=2)
    ax2.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax2.set_ylabel(r'$-C_4 = I_4 - \mu_4$', fontsize=12)
    ax2.set_xlim(-2, 92)
    ax2.set_ylim(-0.05, 0.85)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_title('Pure states', fontsize=11)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=13,
             fontweight='bold', va='top')

    # ----- Helper for mixed-state rows (Werner / Horodecki) -----
    def _plot_mixed_row(ax_N, ax_C, panel_N, panel_C, title,
                        ideal, noisy, N_theory_arr, C4_theory_arr,
                        sep_line=None):
        p_i = [r['p'] for r in ideal]
        p_n = [r['p'] for r in noisy]

        has_mixed_ml = 'N_ml' in noisy[0]
        mk = 'N_ml' if has_mixed_ml else 'N_corr'
        ml_label = 'Noisy + ML cal.' if has_mixed_ml else 'Noisy simulator (Torino)'

        ax_N.plot(p_fine, N_theory_arr, 'k-', linewidth=1.5, label='Theory', zorder=1)
        ax_N.scatter(p_i, [r['N_corr'] for r in ideal],
                     marker='o', facecolors='none', edgecolors=_BLUE, s=50,
                     linewidths=1.2, label='Ideal simulator', zorder=3)
        ax_N.scatter(p_n, [r[mk] for r in noisy],
                     marker='o', color=_RED, s=40, alpha=0.8,
                     label=ml_label, zorder=2)
        if sep_line is not None:
            ax_N.axvline(x=sep_line, color='gray', linestyle=':', linewidth=1,
                         label=f'$p = {sep_line:.2f}$ (sep.)', zorder=0)
        ax_N.set_xlabel(r'$p$', fontsize=12)
        ax_N.set_ylabel(r'Negativity $\mathcal{N}$', fontsize=12)
        ax_N.set_xlim(-0.05, 1.05)
        ax_N.set_ylim(-0.05, 0.55)
        ax_N.legend(fontsize=8, loc='upper left')
        ax_N.set_title(title, fontsize=11)
        ax_N.text(0.02, 0.95, f'({panel_N})', transform=ax_N.transAxes,
                  fontsize=13, fontweight='bold', va='top')

        ax_C.plot(p_fine, C4_theory_arr, 'k-', linewidth=1.5, label='Theory', zorder=1)
        ax_C.scatter(p_i, [r['neg_C4_corr'] for r in ideal],
                     marker='o', facecolors='none', edgecolors=_BLUE, s=50,
                     linewidths=1.2, label='Ideal simulator', zorder=3)
        ax_C.scatter(p_n, [r['neg_C4_corr'] for r in noisy],
                     marker='o', color=_RED, s=40, alpha=0.8,
                     label='Noisy simulator (Torino)', zorder=2)
        if sep_line is not None:
            ax_C.axvline(x=sep_line, color='gray', linestyle=':', linewidth=1,
                         label=f'$p = {sep_line:.2f}$ (sep.)', zorder=0)
        ax_C.set_xlabel(r'$p$', fontsize=12)
        ax_C.set_ylabel(r'$-C_4 = I_4 - \mu_4$', fontsize=12)
        ax_C.set_xlim(-0.05, 1.05)
        ax_C.set_ylim(-0.05, 0.85)
        ax_C.legend(fontsize=8, loc='upper left')
        ax_C.set_title(title, fontsize=11)
        ax_C.text(0.02, 0.95, f'({panel_C})', transform=ax_C.transAxes,
                  fontsize=13, fontweight='bold', va='top')

    p_fine = np.linspace(0, 1, 200)

    # ----- Row 2: Werner states -----
    N_werner_theory = np.maximum(0, (3 * p_fine - 1) / 4)

    def werner_neg_C4(p):
        lam_pt = np.array([(1 + p) / 4] * 3 + [(1 - 3 * p) / 4])
        lam_rho = np.array([(1 + 3 * p) / 4] + [(1 - p) / 4] * 3)
        return np.sum(lam_rho**4) - np.sum(lam_pt**4)
    C4_werner_theory = np.array([werner_neg_C4(p) for p in p_fine])

    _plot_mixed_row(ax3, ax4, 'c', 'd',
                    r'Werner: $p|\Psi^-\rangle\!\langle\Psi^-| + (1-p)\,I/4$',
                    werner_ideal, werner_noisy,
                    N_werner_theory, C4_werner_theory, sep_line=1.0/3)

    # ----- Row 3: Horodecki states -----
    def horodecki_negativity(p):
        if p <= 0:
            return 0.0
        disc = np.sqrt(1 - 2*p + 2*p**2)
        return max(0, (disc - (1 - p)) / 2)

    def horodecki_neg_C4(p):
        rho = create_horodecki_state(p)
        rho_pt = partial_transpose_A(rho)
        eigs_pt = np.linalg.eigvalsh(rho_pt)
        eigs_rho = np.linalg.eigvalsh(rho)
        return float(np.sum(eigs_rho**4) - np.sum(eigs_pt**4))

    N_horo_theory = np.array([horodecki_negativity(p) for p in p_fine])
    C4_horo_theory = np.array([horodecki_neg_C4(p) for p in p_fine])

    _plot_mixed_row(ax5, ax6, 'e', 'f',
                    r'Horodecki: $p|\Psi^-\rangle\!\langle\Psi^-| + (1-p)\,|00\rangle\!\langle 00|$',
                    horodecki_ideal, horodecki_noisy,
                    N_horo_theory, C4_horo_theory, sep_line=None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}")


# ========================================================================
# Main
# ========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Simulate negativity and chirality for many states"
    )
    parser.add_argument('--shots', type=int, default=8192)
    parser.add_argument('--step', type=float, default=2.5,
                        help='Theta step size in degrees (default: 2.5)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--workers', type=int,
                        default=min(multiprocessing.cpu_count(), 8),
                        help='Number of parallel workers (default: cpu_count)')
    args = parser.parse_args()

    thetas = list(np.arange(0, 90 + args.step / 2, args.step))
    p_values = [0.0, 0.2, 1.0 / 3, 0.5, 0.6, 0.8, 1.0]
    shots = args.shots
    n_workers = args.workers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("NEGATIVITY & CHIRALITY SIMULATION - PURE + MIXED STATES")
    print("=" * 70)
    print(f"Pure: {len(thetas)} angles from {thetas[0]:.1f} to {thetas[-1]:.1f} deg")
    print(f"Mixed: {len(p_values)} Werner + {len(p_values)} Horodecki states")
    print(f"Shots per circuit: {shots:,}")
    print(f"Workers: {n_workers}")

    # === Load checkpoint ===
    backend_ideal = None
    backend_noisy = None
    cp = load_checkpoint(shots, thetas, p_values)
    cp_stage = cp['completed_stage'] if cp else ''
    cp_data = cp or {
        'config': {
            'shots': shots, 'thetas': thetas, 'p_values': p_values,
            'n_pure': len(thetas), 'n_werner': len(p_values),
        },
        'timestamp': timestamp,
    }

    # Checkpoint stage ordering
    ALL_STAGES = ('ideal_pure', 'ideal_werner', 'ideal_horodecki',
                  'noisy_pure', 'noisy_werner', 'noisy_horodecki', 'ml')

    def past_stage(stage):
        """True if checkpoint is at or past the given stage."""
        if not cp_stage:
            return False
        return ALL_STAGES.index(cp_stage) >= ALL_STAGES.index(stage)

    # === MODE 1: IDEAL (no noise) ===
    print("\n" + "-" * 70)
    print("MODE 1: IDEAL SIMULATOR (no noise)")
    print("-" * 70)

    if past_stage('ideal_pure'):
        pure_ideal = cp_data['pure_ideal']
        print(f"  [Pure states] loaded from checkpoint ({len(pure_ideal)} states)")
    else:
        backend_ideal = AerSimulator()
        print("  [Pure states]")
        pure_ideal, _, _ = run_pure_mode(thetas, backend_ideal, shots,
                                         calibrate=False, verbose=True,
                                         n_workers=n_workers)
        cp_data['pure_ideal'] = pure_ideal
        save_checkpoint(cp_data, 'ideal_pure')

    if past_stage('ideal_werner'):
        werner_ideal = cp_data['werner_ideal']
        print(f"  [Werner states] loaded from checkpoint ({len(werner_ideal)} states)")
    else:
        if backend_ideal is None:
            backend_ideal = AerSimulator()
        print("  [Werner states]")
        werner_ideal = run_mixed_mode(p_values, create_werner_state,
                                      fit_physical_werner_state,
                                      backend_ideal, shots,
                                      f_I2=1.0, f_mu4=1.0, verbose=True,
                                      n_workers=n_workers, label="Werner")
        cp_data['werner_ideal'] = werner_ideal
        save_checkpoint(cp_data, 'ideal_werner')

    if past_stage('ideal_horodecki'):
        horodecki_ideal = cp_data['horodecki_ideal']
        print(f"  [Horodecki states] loaded from checkpoint ({len(horodecki_ideal)} states)")
    else:
        if backend_ideal is None:
            backend_ideal = AerSimulator()
        print("  [Horodecki states]")
        horodecki_ideal = run_mixed_mode(p_values, create_horodecki_state,
                                         fit_physical_horodecki_state,
                                         backend_ideal, shots,
                                         f_I2=1.0, f_mu4=1.0, verbose=True,
                                         n_workers=n_workers, label="Horodecki")
        cp_data['horodecki_ideal'] = horodecki_ideal
        save_checkpoint(cp_data, 'ideal_horodecki')

    # === MODE 2: NOISY (Torino calibration) ===
    print("\n" + "-" * 70)
    print("MODE 2: NOISY SIMULATOR (IBM Torino calibration)")
    print("-" * 70)
    cal_dir = PACKAGE_DIR / "data" / "IBM Torino"
    cal_files = sorted(cal_dir.glob("ibm_torino_calibrations_*.csv"))
    if not cal_files:
        print(f"ERROR: No calibration files in {cal_dir}")
        return
    cal_file = cal_files[-1]
    print(f"Calibration file: {cal_file.name}")

    if past_stage('noisy_pure'):
        pure_noisy = cp_data['pure_noisy']
        f_I2 = cp_data.get('f_I2', 1.0)
        f_mu4 = cp_data.get('f_mu4', 1.0)
        print(f"  [Pure states] loaded from checkpoint ({len(pure_noisy)} states)")
        print(f"  Calibration: f_I2={f_I2:.4f}, f_mu4={f_mu4:.4f}")
    else:
        backend_noisy = create_noisy_backend_from_csv(cal_file, num_qubits=10)
        print("  [Pure states]")
        pure_noisy, f_I2, f_mu4 = run_pure_mode(thetas, backend_noisy, shots,
                                                 calibrate=True, verbose=True,
                                                 n_workers=n_workers)
        cp_data['pure_noisy'] = pure_noisy
        cp_data['f_I2'] = f_I2
        cp_data['f_mu4'] = f_mu4
        save_checkpoint(cp_data, 'noisy_pure')

    if past_stage('noisy_werner'):
        werner_noisy = cp_data['werner_noisy']
        print(f"  [Werner states] loaded from checkpoint ({len(werner_noisy)} states)")
    else:
        if backend_noisy is None:
            backend_noisy = create_noisy_backend_from_csv(cal_file, num_qubits=10)
        print("  [Werner states]")
        werner_noisy = run_mixed_mode(p_values, create_werner_state,
                                      fit_physical_werner_state,
                                      backend_noisy, shots,
                                      f_I2=f_I2, f_mu4=f_mu4, verbose=True,
                                      n_workers=n_workers, label="Werner")
        cp_data['werner_noisy'] = werner_noisy
        save_checkpoint(cp_data, 'noisy_werner')

    if past_stage('noisy_horodecki'):
        horodecki_noisy = cp_data['horodecki_noisy']
        print(f"  [Horodecki states] loaded from checkpoint ({len(horodecki_noisy)} states)")
    else:
        if backend_noisy is None:
            backend_noisy = create_noisy_backend_from_csv(cal_file, num_qubits=10)
        print("  [Horodecki states]")
        horodecki_noisy = run_mixed_mode(p_values, create_horodecki_state,
                                         fit_physical_horodecki_state,
                                         backend_noisy, shots,
                                         f_I2=f_I2, f_mu4=f_mu4, verbose=True,
                                         n_workers=n_workers, label="Horodecki")
        cp_data['horodecki_noisy'] = horodecki_noisy
        save_checkpoint(cp_data, 'noisy_horodecki')

    # === ML calibration (pure + Werner + Horodecki noisy states) ===
    ml_params = None
    if (cp_stage == 'ml'
            and 'N_ml' in cp_data['pure_noisy'][0]
            and 'N_ml' in cp_data['werner_noisy'][0]
            and 'N_ml' in cp_data['horodecki_noisy'][0]):
        pure_noisy = cp_data['pure_noisy']
        werner_noisy = cp_data['werner_noisy']
        horodecki_noisy = cp_data['horodecki_noisy']
        ml_params = cp_data.get('ml_params')
        print("\n  [ML calibration] loaded from checkpoint")
        if ml_params:
            print(f"  ML factors: f2={ml_params['f2']:.4f}, "
                  f"f3={ml_params['f3']:.4f}, f4={ml_params['f4']:.4f}")
    else:
        print("\n  [ML calibration — two-stage estimator]")
        try:
            pure_noisy, ml_params = apply_ml_calibration(pure_noisy, verbose=True)
            print("  [Werner-specific ML calibration]")
            werner_noisy = apply_ml_correction_werner(
                werner_noisy, verbose=True)
            print("  [Horodecki-specific ML calibration]")
            horodecki_noisy = apply_ml_correction_horodecki(
                horodecki_noisy, verbose=True)
            cp_data['pure_noisy'] = pure_noisy
            cp_data['werner_noisy'] = werner_noisy
            cp_data['horodecki_noisy'] = horodecki_noisy
            cp_data['ml_params'] = ml_params
            save_checkpoint(cp_data, 'ml')
        except Exception as e:
            print(f"  ML calibration failed: {e}")
            print("  (Falling back to simple correction for figure)")

    # === Save results ===
    output_json = args.output or f"sim_negativity_chirality_{timestamp}.json"
    save_data = {
        'timestamp': timestamp,
        'config': {
            'shots': shots, 'thetas': thetas, 'p_values': p_values,
            'n_pure': len(thetas), 'n_werner': len(p_values),
        },
        'pure_ideal': pure_ideal,
        'pure_noisy': pure_noisy,
        'werner_ideal': werner_ideal,
        'werner_noisy': werner_noisy,
        'horodecki_ideal': horodecki_ideal,
        'horodecki_noisy': horodecki_noisy,
    }
    output_path = Path(__file__).parent / output_json
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # === Generate figure ===
    fig_path = Path(__file__).parent / "fig_simulation_negativity_chirality.pdf"
    make_figure(pure_ideal, pure_noisy, werner_ideal, werner_noisy,
                horodecki_ideal, horodecki_noisy, fig_path)

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def rmse(errors):
        return np.sqrt(np.mean(np.array(errors)**2))

    for mode_name, res in [("Ideal (pure)", pure_ideal),
                            ("Noisy (pure, simple corr.)", pure_noisy)]:
        N_errors = [abs(r['N_corr'] - r['N_theory']) for r in res]
        C_errors = [abs(r['C_corr'] - r['C_theory']) for r in res]
        print(f"\n{mode_name}:")
        print(f"  N RMSE: {rmse(N_errors):.4f}")
        print(f"  -C4 RMSE: {rmse(C_errors):.4f}")

    # ML RMSE (if available)
    has_ml = 'N_ml' in pure_noisy[0]
    if has_ml:
        N_ml_errors = [abs(r['N_ml'] - r['N_theory']) for r in pure_noisy]
        C_ml_errors = [abs(r['C_ml'] - r['C_theory']) for r in pure_noisy]
        print(f"\nNoisy (pure, ML calibration):")
        print(f"  N RMSE: {rmse(N_ml_errors):.4f}")
        print(f"  -C4 RMSE: {rmse(C_ml_errors):.4f}")

    for family, ideal, noisy in [("Werner", werner_ideal, werner_noisy),
                                  ("Horodecki", horodecki_ideal, horodecki_noisy)]:
        for mode_name, res in [(f"Ideal ({family})", ideal),
                                (f"Noisy ({family}, simple corr.)", noisy)]:
            N_errors = [abs(r['N_corr'] - r['N_theory']) for r in res]
            C_errors = [abs(r['neg_C4_corr'] - r['neg_C4_theory']) for r in res]
            print(f"\n{mode_name}:")
            print(f"  N RMSE: {rmse(N_errors):.4f}")
            print(f"  -C4 RMSE: {rmse(C_errors):.4f}")
        has_mixed_ml = 'N_ml' in noisy[0]
        if has_mixed_ml:
            N_mml = [abs(r['N_ml'] - r['N_theory']) for r in noisy]
            print(f"\nNoisy ({family}, ML calibration):")
            print(f"  N RMSE: {rmse(N_mml):.4f}")


if __name__ == '__main__':
    main()
