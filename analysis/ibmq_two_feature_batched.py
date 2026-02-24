"""
IBM Quantum Experiment: Two-Feature Classifier (D2 + D4)
=========================================================

Measures D_2 and D_4 for the two-feature bound entanglement classifier:

    Classify as BOUND ENTANGLED if:
        D_4 < -0.026 + 0.022 * (S_2^2 / S_4)

Measurement approach:
  S_2 (classical): S_2 = sum_i lambda_i^2  (= Tr[rho^2], exact from spectrum)
  G_2 (quantum):   SWAP tests between |e_i> and SWAP_AB|e_j>
  G_4 (quantum):   SWAP tests using F_ij = E_i @ E_j
  S_4 (quantum):   SWAP tests using M_ij = E_i^dag @ E_j
  D_2 = S_2 - G_2
  D_4 computed two ways:
    (A) D4_mixed = S_4^classical - G_4^measured
    (B) D4_both  = S_4^measured  - G_4^measured

Usage:
  python ibmq_two_feature_batched.py [--shots 4000] [--states 4] [--dry-run]
  python ibmq_two_feature_batched.py --simulator    # noisy simulator (FakeMarrakesh)
"""

import numpy as np
from scipy import linalg
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    print("WARNING: qiskit-ibm-runtime not installed")

try:
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    FAKE_AVAILABLE = True
except ImportError:
    FAKE_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SHOTS = 4000
MAX_EIGVALS = 4  # Top eigenvalues to keep (controls circuit count)

# Two-feature classifier boundary (SVM-optimal, includes a=0 boundary)
CLASSIFIER_INTERCEPT = -0.068
CLASSIFIER_SLOPE = 0.053

# =============================================================================
# State Definitions
# =============================================================================

def horodecki_3x3(a):
    """Horodecki 3x3 bound entangled state."""
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = a; rho[0, 8] = a; rho[1, 1] = a; rho[2, 2] = a
    rho[3, 3] = a; rho[4, 4] = a; rho[5, 5] = a
    rho[6, 6] = (1 + a) / 2; rho[7, 7] = (1 + a) / 2
    rho[8, 0] = a; rho[8, 8] = a
    sqrt_term = np.sqrt(1 - a**2) / 2
    rho[2, 6] = sqrt_term; rho[6, 2] = sqrt_term
    rho[5, 7] = sqrt_term; rho[7, 5] = sqrt_term
    return rho / np.trace(rho)

def random_separable_3x3(rng, n_terms=4):
    """Random separable state."""
    weights = rng.random(n_terms)
    weights /= weights.sum()
    rho = np.zeros((9, 9), dtype=complex)
    for w in weights:
        psi_A = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        psi_A /= np.linalg.norm(psi_A)
        psi_B = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        psi_B /= np.linalg.norm(psi_B)
        rho += w * np.kron(np.outer(psi_A, psi_A.conj()),
                           np.outer(psi_B, psi_B.conj()))
    return rho

def tiles_bound_entangled_3x3():
    """Tiles UPB bound entangled state."""
    dim = 9
    upb_states = []
    psi1 = np.zeros(dim, dtype=complex)
    psi1[0] = 1/np.sqrt(2); psi1[1] = -1/np.sqrt(2)
    upb_states.append(psi1)
    psi2 = np.zeros(dim, dtype=complex)
    psi2[7] = 1/np.sqrt(2); psi2[8] = -1/np.sqrt(2)
    upb_states.append(psi2)
    psi3 = np.zeros(dim, dtype=complex)
    psi3[2] = 1/np.sqrt(2); psi3[5] = -1/np.sqrt(2)
    upb_states.append(psi3)
    psi4 = np.zeros(dim, dtype=complex)
    psi4[3] = 1/np.sqrt(2); psi4[6] = -1/np.sqrt(2)
    upb_states.append(psi4)
    psi5 = np.ones(dim, dtype=complex) / 3.0
    upb_states.append(psi5)
    P_upb = sum(np.outer(psi, psi.conj()) for psi in upb_states)
    P_complement = np.eye(dim) - P_upb
    eigvals, eigvecs = np.linalg.eigh(P_complement)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho / np.trace(rho)

def chessboard_3x3(a, b, c, d, m, n):
    """Bruss-Peres chessboard bound entangled state (3x3).

    Constructed from four range vectors with a checkerboard pattern of
    non-zero entries.  For real parameters with c*m*n != a*b*c the state
    is PPT entangled (bound entangled).

    Reference: Bruss & Peres, Phys. Rev. A 61, 030301(R) (2000).
    """
    s = a * np.conj(c) / np.conj(n)
    t = a * d / m
    v1 = np.array([m, 0, s, 0, n, 0, 0, 0, 0], dtype=complex)
    v2 = np.array([0, a, 0, b, 0, c, 0, 0, 0], dtype=complex)
    v3 = np.array([np.conj(n), 0, 0, 0, -np.conj(m), 0, t, 0, 0],
                  dtype=complex)
    v4 = np.array([0, np.conj(b), 0, -np.conj(a), 0, 0, 0, d, 0],
                  dtype=complex)
    rho = (np.outer(v1, v1.conj()) + np.outer(v2, v2.conj())
           + np.outer(v3, v3.conj()) + np.outer(v4, v4.conj()))
    return rho / np.trace(rho)

# =============================================================================
# Theoretical Computation
# =============================================================================

def realignment(rho, dA=3, dB=3):
    """Compute realignment matrix."""
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA**2, dB**2)

def compute_features_theory(rho, dA=3, dB=3):
    """Compute all classifier features theoretically."""
    R = realignment(rho, dA, dB)
    svd_vals = linalg.svdvals(R)
    S_2 = np.sum(svd_vals ** 2)
    S_4 = np.sum(svd_vals ** 4)
    R_eigvals = linalg.eigvals(R)
    G_2 = np.real(np.sum(R_eigvals ** 2))
    G_4 = np.real(np.sum(R_eigvals ** 4))
    return {
        'S_2': float(S_2), 'S_4': float(S_4),
        'G_2': float(G_2), 'G_4': float(G_4),
        'D_2': float(S_2 - G_2), 'D_4': float(S_4 - G_4),
        'S2sq_S4': float(S_2**2 / S_4)
    }

# =============================================================================
# Qutrit Encoding & SWAP_AB
# =============================================================================

def encode_qutrit_to_4qubit(psi_9, dA=3, dB=3):
    """Encode 9-dim qutrit state into 16-dim (4-qubit) space.
    Mapping: |a,b> (index a*dB+b) -> 4-qubit |a,b> (index a*4+b)."""
    psi_16 = np.zeros(16, dtype=complex)
    for a in range(dA):
        for b in range(dB):
            psi_16[a * 4 + b] = psi_9[a * dB + b]
    norm = np.linalg.norm(psi_16)
    if norm > 0:
        psi_16 /= norm
    return psi_16

def apply_swap_ab(psi_9, dA=3, dB=3):
    """Apply SWAP_AB: |a,b> -> |b,a>.
    Component at |a,b> in result = component at |b,a> in input."""
    result = np.zeros_like(psi_9)
    for a in range(dA):
        for b in range(dB):
            result[a * dB + b] = psi_9[b * dA + a]
    return result

def eigenstate_to_matrix(psi_9, dA=3, dB=3):
    """Reshape eigenstate vector to dA x dB matrix: E[a][b] = psi[a*dB+b]."""
    return psi_9.reshape(dA, dB)

# =============================================================================
# Circuit Construction
# =============================================================================

def create_swap_test_circuit(psi1_16, psi2_16):
    """SWAP test circuit: 9 qubits (1 ancilla + 4 + 4).
    Measures |<psi1|psi2>|^2 via P(0) = (1 + |<psi1|psi2>|^2) / 2."""
    ancilla = QuantumRegister(1, 'anc')
    reg1 = QuantumRegister(4, 'r1')
    reg2 = QuantumRegister(4, 'r2')
    meas = ClassicalRegister(1, 'm')
    qc = QuantumCircuit(ancilla, reg1, reg2, meas)

    psi1_16 = psi1_16 / np.linalg.norm(psi1_16)
    psi2_16 = psi2_16 / np.linalg.norm(psi2_16)

    qc.initialize(psi1_16, reg1[:])
    qc.initialize(psi2_16, reg2[:])
    qc.h(ancilla[0])
    for i in range(4):
        qc.cswap(ancilla[0], reg1[i], reg2[i])
    qc.h(ancilla[0])
    qc.measure(ancilla[0], meas[0])
    return qc

# =============================================================================
# Spectral Measurer: G2 + G4
# =============================================================================

class SpectralMeasurer:
    """Measures G_2, G_4, and S_4 using batched SWAP tests on IBM Quantum."""

    def __init__(self, backend, shots=DEFAULT_SHOTS, max_eigvals=MAX_EIGVALS):
        self.backend = backend
        self.shots = shots
        self.max_eigvals = max_eigvals
        self.pass_manager = None
        if backend is not None:
            self.pass_manager = generate_preset_pass_manager(
                optimization_level=3, backend=backend
            )

    def _get_active_eigenstates(self, rho):
        """Get top eigenvalues/eigenvectors of rho, truncated to max_eigvals."""
        eigvals, eigvecs = np.linalg.eigh(rho)
        sorted_idx = np.argsort(-eigvals)
        active = [(eigvals[i], eigvecs[:, i])
                  for i in sorted_idx if eigvals[i] > 1e-10]
        active = active[:self.max_eigvals]
        return active

    def prepare_G2_circuits(self, rho, dA=3, dB=3):
        """Prepare SWAP test circuits for G_2.

        G_2 = sum_{ij} lambda_i lambda_j |<e_i|SWAP_AB|e_j>|^2

        SWAP test between |e_i> and SWAP_AB|e_j> for each pair (i,j).
        """
        active = self._get_active_eigenstates(rho)
        n = len(active)
        print(f"  G2: {n} eigenvalues, {n**2} circuits")

        circuits = []
        circuit_info = []
        for li, ei in active:
            ei_16 = encode_qutrit_to_4qubit(ei, dA, dB)
            for lj, ej in active:
                # Apply SWAP_AB to e_j, then encode
                ej_swapped = apply_swap_ab(ej, dA, dB)
                ej_swap_16 = encode_qutrit_to_4qubit(ej_swapped, dA, dB)

                qc = create_swap_test_circuit(ei_16, ej_swap_16)
                circuits.append(qc)

                # Theory value for validation
                Ei = eigenstate_to_matrix(ei, dA, dB)
                Ej = eigenstate_to_matrix(ej, dA, dB)
                M_ij = np.vdot(ei, apply_swap_ab(ej, dA, dB))  # <e_i|SWAP_AB|e_j>
                circuit_info.append({
                    'type': 'G2',
                    'lambda_i': float(li),
                    'lambda_j': float(lj),
                    'overlap_sq_theory': float(np.abs(M_ij)**2),
                })
        return circuits, circuit_info

    def prepare_G4_circuits(self, rho, dA=3, dB=3):
        """Prepare SWAP test circuits for G_4.

        G_4 = sum_{ijkl} lambda_i lambda_j lambda_k lambda_l |Tr[F_ij F_kl]|^2
        where F_ij = E_i @ E_j (eigenstate matrix product).

        Uses identity: Tr[R_i R_j R_k R_l] = |Tr[E_i E_j E_k E_l]|^2
        and Tr[E_i E_j E_k E_l] = Tr[F_ij F_kl].

        Measures |Tr[F_ij F_kl]|^2 via SWAP test between
        vec(conj(F_ij)) and vec(F_kl^T).
        """
        active = self._get_active_eigenstates(rho)
        n = len(active)

        # Build all F_ij = E_i @ E_j and their norms
        E_list = [(l, eigenstate_to_matrix(e, dA, dB)) for l, e in active]
        F_pairs = []  # (weight_ij, F_ij, norm_ij, i, j)
        for i, (li, Ei) in enumerate(E_list):
            for j, (lj, Ej) in enumerate(E_list):
                F = Ei @ Ej
                norm_sq = np.real(np.sum(np.abs(F)**2))  # Frobenius norm squared
                if norm_sq < 1e-20:
                    continue
                F_pairs.append({
                    'w': li * lj,
                    'F': F,
                    'norm_sq': norm_sq,
                    'i': i, 'j': j,
                })

        n_pairs = len(F_pairs)
        # Use symmetry: overlap(ij,kl) = overlap(kl,ij)
        n_circuits = n_pairs * (n_pairs + 1) // 2
        print(f"  G4: {n} eigenvalues, {n_pairs} F-pairs, {n_circuits} circuits")

        circuits = []
        circuit_info = []
        for p_idx, p in enumerate(F_pairs):
            for q_idx in range(p_idx, len(F_pairs)):
                q = F_pairs[q_idx]

                # alpha = vec(conj(F_ij)), beta = vec(F_kl^T)
                alpha_9 = p['F'].conj().flatten()
                beta_9 = q['F'].T.flatten()

                alpha_16 = encode_qutrit_to_4qubit(alpha_9, dA, dB)
                beta_16 = encode_qutrit_to_4qubit(beta_9, dA, dB)

                qc = create_swap_test_circuit(alpha_16, beta_16)
                circuits.append(qc)

                # Theory: |Tr[F_ij F_kl]|^2 / (norm_ij^2 * norm_kl^2)
                trace_val = np.trace(p['F'] @ q['F'])
                theory_overlap_sq = float(np.abs(trace_val)**2 /
                                          (p['norm_sq'] * q['norm_sq']))

                multiplier = 1 if p_idx == q_idx else 2  # symmetry factor
                circuit_info.append({
                    'type': 'G4',
                    'p_idx': p_idx,
                    'q_idx': q_idx,
                    'w_p': float(p['w']),
                    'w_q': float(q['w']),
                    'norm_sq_p': float(p['norm_sq']),
                    'norm_sq_q': float(q['norm_sq']),
                    'multiplier': multiplier,
                    'overlap_sq_theory': theory_overlap_sq,
                })

        return circuits, circuit_info

    def prepare_S2_circuits(self, rho, dA=3, dB=3):
        """Prepare SWAP test circuits for S_2 verification.

        S_2 = Tr[R^dag R] = Tr[rho^2] = sum_{ij} lambda_i lambda_j |<e_i|e_j>|^2.
        For orthonormal eigenstates |<e_i|e_j>|^2 = delta_{ij}, so S_2 = sum_i lambda_i^2
        exactly.  Hardware measurement verifies state preparation fidelity.

        SWAP test between |e_i> and |e_j> (NO SWAP_AB).
        """
        active = self._get_active_eigenstates(rho)
        n = len(active)
        print(f"  S2: {n} eigenvalues, {n**2} circuits")

        circuits = []
        circuit_info = []
        for li, ei in active:
            ei_16 = encode_qutrit_to_4qubit(ei, dA, dB)
            for lj, ej in active:
                ej_16 = encode_qutrit_to_4qubit(ej, dA, dB)
                qc = create_swap_test_circuit(ei_16, ej_16)
                circuits.append(qc)
                circuit_info.append({
                    'type': 'S2',
                    'lambda_i': float(li),
                    'lambda_j': float(lj),
                    'overlap_sq_theory': float(np.abs(np.vdot(ei, ej))**2),
                })
        return circuits, circuit_info

    def prepare_S4_circuits(self, rho, dA=3, dB=3):
        """Prepare SWAP test circuits for S_4.

        S_4 = Tr[(R^dag R)^2] = sum_{ijkl} lambda_i lambda_j lambda_k lambda_l
              |Tr[M_ij M_kl]|^2
        where M_ij = E_i^dag E_j  (note: E_i^dag, not E_i as in G_4).

        Uses identity: Tr[R_i^dag R_j R_k^dag R_l] = |Tr[M_ij M_kl]|^2
        Measured via SWAP test between vec(conj(M_ij)) and vec(M_kl^T).
        """
        active = self._get_active_eigenstates(rho)
        n = len(active)

        E_list = [(l, eigenstate_to_matrix(e, dA, dB)) for l, e in active]
        M_pairs = []
        for i, (li, Ei) in enumerate(E_list):
            for j, (lj, Ej) in enumerate(E_list):
                M = Ei.conj().T @ Ej   # E_i^dag @ E_j
                norm_sq = np.real(np.sum(np.abs(M)**2))
                if norm_sq < 1e-20:
                    continue
                M_pairs.append({
                    'w': li * lj,
                    'M': M,
                    'norm_sq': norm_sq,
                    'i': i, 'j': j,
                })

        n_pairs = len(M_pairs)
        n_circuits = n_pairs * (n_pairs + 1) // 2
        print(f"  S4: {n} eigenvalues, {n_pairs} M-pairs, {n_circuits} circuits")

        circuits = []
        circuit_info = []
        for p_idx, p in enumerate(M_pairs):
            for q_idx in range(p_idx, len(M_pairs)):
                q = M_pairs[q_idx]

                alpha_9 = p['M'].conj().flatten()
                beta_9 = q['M'].T.flatten()

                alpha_16 = encode_qutrit_to_4qubit(alpha_9, dA, dB)
                beta_16 = encode_qutrit_to_4qubit(beta_9, dA, dB)

                qc = create_swap_test_circuit(alpha_16, beta_16)
                circuits.append(qc)

                trace_val = np.trace(p['M'] @ q['M'])
                theory_overlap_sq = float(np.abs(trace_val)**2 /
                                          (p['norm_sq'] * q['norm_sq']))
                multiplier = 1 if p_idx == q_idx else 2
                circuit_info.append({
                    'type': 'S4',
                    'p_idx': p_idx,
                    'q_idx': q_idx,
                    'w_p': float(p['w']),
                    'w_q': float(q['w']),
                    'norm_sq_p': float(p['norm_sq']),
                    'norm_sq_q': float(q['norm_sq']),
                    'multiplier': multiplier,
                    'overlap_sq_theory': theory_overlap_sq,
                })

        return circuits, circuit_info

    def compute_classical_S(self, rho, dA=3, dB=3):
        """Compute S_2 and S_4 classically from singular values of R."""
        R = realignment(rho, dA, dB)
        svd_vals = linalg.svdvals(R)
        S_2 = float(np.sum(svd_vals ** 2))
        S_4 = float(np.sum(svd_vals ** 4))
        return S_2, S_4

# =============================================================================
# Calibration Data
# =============================================================================

def download_calibration_data(backend):
    """Download full calibration data from the selected backend."""
    cal_data = {
        'backend_name': backend.name,
        'num_qubits': backend.num_qubits,
        'timestamp': datetime.now().isoformat(),
    }

    # Backend configuration
    try:
        config = backend.configuration()
        cal_data['configuration'] = {
            'basis_gates': config.basis_gates if hasattr(config, 'basis_gates') else None,
            'coupling_map': config.coupling_map if hasattr(config, 'coupling_map') else None,
            'dt': config.dt if hasattr(config, 'dt') else None,
            'max_circuits': config.max_experiments if hasattr(config, 'max_experiments') else None,
        }
    except Exception:
        cal_data['configuration'] = None

    # Backend properties (gate errors, T1, T2, readout errors)
    try:
        props = backend.properties()
        if props is not None:
            qubit_props = []
            for q in range(backend.num_qubits):
                qp = {}
                try:
                    qp['T1'] = props.t1(q)
                except Exception:
                    pass
                try:
                    qp['T2'] = props.t2(q)
                except Exception:
                    pass
                try:
                    qp['frequency'] = props.frequency(q)
                except Exception:
                    pass
                try:
                    qp['readout_error'] = props.readout_error(q)
                except Exception:
                    pass
                qubit_props.append(qp)
            cal_data['qubit_properties'] = qubit_props
            cal_data['properties_last_update'] = str(props.last_update_date)

            gate_errors = {}
            for gate in props.gates:
                key = f"{gate.gate}_{gate.qubits}"
                for param in gate.parameters:
                    if param.name == 'gate_error':
                        gate_errors[key] = param.value
            cal_data['gate_errors'] = gate_errors
    except Exception:
        pass

    # Target-based calibration (newer API)
    try:
        target = backend.target
        if target is not None:
            cal_data['target_basis_gates'] = list(target.operation_names)
            cal_data['target_num_qubits'] = target.num_qubits
            cal_data['target_dt'] = target.dt

            # Qubit properties from target
            target_qubit_props = []
            if target.qubit_properties is not None:
                for q_props in target.qubit_properties:
                    if q_props is not None:
                        target_qubit_props.append({
                            'frequency': q_props.frequency,
                            't1': q_props.t1,
                            't2': q_props.t2,
                        })
                    else:
                        target_qubit_props.append(None)
            cal_data['target_qubit_properties'] = target_qubit_props

            # Two-qubit gate errors from target
            twoq_errors = {}
            for op_name in target.operation_names:
                props_map = target[op_name]
                if props_map is None:
                    continue
                for qargs, inst_props in props_map.items():
                    if inst_props is not None and len(qargs) == 2:
                        twoq_errors[f"{op_name}_{list(qargs)}"] = {
                            'error': inst_props.error,
                            'duration': inst_props.duration,
                        }
            cal_data['two_qubit_gate_properties'] = twoq_errors
    except Exception:
        pass

    return cal_data

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(n_states=4, shots=DEFAULT_SHOTS, dry_run=False,
                   max_eigvals=MAX_EIGVALS, simulator=False):
    """Run the two-feature classifier experiment on IBM Quantum."""
    print("=" * 70)
    print("IBM QUANTUM EXPERIMENT: TWO-FEATURE CLASSIFIER (G_k + S_k)")
    print("=" * 70)
    print(f"\nClassifier: D_4 < {CLASSIFIER_INTERCEPT} + {CLASSIFIER_SLOPE} * (S2^2/S4)")
    print(f"\nConfiguration:")
    print(f"  States to test: {n_states}")
    print(f"  Shots per circuit: {shots}")
    print(f"  Max eigenvalues: {max_eigvals}")
    if simulator:
        print(f"  Mode: NOISY SIMULATOR (FakeMarrakesh)")

    # Connect to IBM Quantum or use simulator
    backend = None
    cal_data = None
    if simulator:
        if not FAKE_AVAILABLE:
            print("\nERROR: qiskit_ibm_runtime.fake_provider not available!")
            return
        backend = FakeMarrakesh()
        print(f"\nUsing noisy simulator: FakeMarrakesh ({backend.num_qubits} qubits)")
        cal_data = {'backend_name': 'FakeMarrakesh', 'simulator': True}
    elif not dry_run:
        if not IBM_AVAILABLE:
            print("\nERROR: qiskit-ibm-runtime not installed!")
            return

        print("\nConnecting to IBM Quantum...")
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
        except Exception as e:
            print(f"Connection failed: {e}")
            return

        print("\nFinding available backend...")
        backends = service.backends(
            filters=lambda x: x.status().operational
                             and not x.status().status_msg == "internal"
                             and x.num_qubits >= 9
        )
        if not backends:
            print("No available backends with 9+ qubits!")
            return

        backends_info = []
        for b in backends:
            try:
                status = b.status()
                backends_info.append((b, status.pending_jobs))
            except Exception:
                pass
        backends_info.sort(key=lambda x: x[1])

        print("\nAvailable backends:")
        for b, jobs in backends_info[:5]:
            print(f"  {b.name}: {jobs} pending jobs")

        backend = backends_info[0][0]
        print(f"\nSelected: {backend.name}")

        # Download calibration data
        print("\nDownloading calibration data...")
        cal_data = download_calibration_data(backend)
        n_twoq = len(cal_data.get('two_qubit_gate_properties', {}))
        n_qprops = len(cal_data.get('target_qubit_properties', []))
        print(f"  Qubit properties: {n_qprops} qubits")
        print(f"  Two-qubit gate entries: {n_twoq}")
    else:
        print("\nDRY RUN: Using theoretical calculations")

    measurer = SpectralMeasurer(backend, shots=shots, max_eigvals=max_eigvals)

    # Generate test states
    print("\n" + "=" * 70)
    print("GENERATING TEST STATES")
    print("=" * 70)

    rng = np.random.default_rng(42)
    states = []

    n_sep = n_states // 2
    for i in range(n_sep):
        rho = random_separable_3x3(rng)
        states.append((f'separable_{i+1}', rho, False))
        print(f"  Generated separable state {i+1}")

    n_be = n_states - n_sep
    # Tiles UPB bound entangled state (rank 4, exact at r=4)
    rho_tiles = tiles_bound_entangled_3x3()
    states.append(('tiles_BE', rho_tiles, True))
    print(f"  Generated Tiles UPB bound entangled state (rank 4)")
    # Fill remaining BE slots with Horodecki states
    for i in range(1, n_be):
        a = 0.3 + 0.4 * i / max(n_be - 1, 1)
        rho = horodecki_3x3(a)
        states.append((f'horodecki(a={a:.2f})', rho, True))
        print(f"  Generated Horodecki state (a={a:.2f})")

    # Prepare all circuits
    print("\n" + "=" * 70)
    print("PREPARING CIRCUITS")
    print("=" * 70)

    all_circuits = []
    all_info = []
    state_boundaries = [0]  # [start_g2, end_g2/start_g4, end_g4] per state

    for name, rho, is_be in states:
        print(f"\n{name}:")
        g2_circuits, g2_info = measurer.prepare_G2_circuits(rho)
        g4_circuits, g4_info = measurer.prepare_G4_circuits(rho)
        s4_circuits, s4_info = measurer.prepare_S4_circuits(rho)

        start = len(all_circuits)
        all_circuits.extend(g2_circuits)
        g2_end = len(all_circuits)
        all_circuits.extend(g4_circuits)
        g4_end = len(all_circuits)
        all_circuits.extend(s4_circuits)
        s4_end = len(all_circuits)

        all_info.extend(g2_info)
        all_info.extend(g4_info)
        all_info.extend(s4_info)
        state_boundaries.append((start, g2_end, g4_end, s4_end))

    print(f"\nTotal circuits: {len(all_circuits)}")

    # Run on hardware or compute theoretically
    results = []

    if not dry_run:
        print("\n" + "-" * 50)
        print("SUBMITTING JOBS")
        print("-" * 50)

        print(f"Transpiling {len(all_circuits)} circuits...")
        transpiled = measurer.pass_manager.run(all_circuits)
        if not isinstance(transpiled, list):
            transpiled = [transpiled]
        depths = [c.depth() for c in transpiled]
        print(f"Circuit depths: mean={np.mean(depths):.0f}, "
              f"max={np.max(depths)}, min={np.min(depths)}")

        # Split into batches to avoid IBM internal errors on large submissions
        BATCH_SIZE = 300
        all_results = [None] * len(transpiled)
        job_ids = []
        n_batches = (len(transpiled) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Splitting into {n_batches} batches of up to {BATCH_SIZE} circuits")

        sampler = Sampler(mode=backend)
        for batch_idx in range(n_batches):
            start_b = batch_idx * BATCH_SIZE
            end_b = min(start_b + BATCH_SIZE, len(transpiled))
            batch = transpiled[start_b:end_b]

            MAX_RETRIES = 3
            for attempt in range(MAX_RETRIES):
                try:
                    print(f"\nBatch {batch_idx+1}/{n_batches} "
                          f"(circuits {start_b}-{end_b-1})...")
                    job = sampler.run(batch, shots=shots)
                    jid = job.job_id()
                    job_ids.append(jid)
                    print(f"  Job ID: {jid}")
                    print(f"  Waiting for results...")
                    batch_result = job.result()
                    print(f"  Batch {batch_idx+1} completed!")
                    for i, idx in enumerate(range(start_b, end_b)):
                        all_results[idx] = batch_result[i]
                    break
                except Exception as e:
                    print(f"  Attempt {attempt+1} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        import time
                        wait = 30 * (attempt + 1)
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Batch {batch_idx+1} failed after {MAX_RETRIES} "
                            f"attempts: {e}")

        result = all_results
        job_id = job_ids
        print(f"\nAll {n_batches} batches completed!")

        # Process results per state
        for s_idx, (name, rho, is_be) in enumerate(states):
            start, g2_end, g4_end, s4_end = state_boundaries[s_idx + 1]
            theory = compute_features_theory(rho)

            # G2 from measurements
            G2_measured = 0.0
            for idx in range(start, g2_end):
                info = all_info[idx]
                pub_result = result[idx]
                counts = pub_result.data.m.get_counts()
                total = sum(counts.values())
                p0 = counts.get('0', 0) / total
                overlap_sq = max(0, 2 * p0 - 1)
                G2_measured += info['lambda_i'] * info['lambda_j'] * overlap_sq

            # G4 from measurements
            G4_measured = 0.0
            for idx in range(g2_end, g4_end):
                info = all_info[idx]
                pub_result = result[idx]
                counts = pub_result.data.m.get_counts()
                total = sum(counts.values())
                p0 = counts.get('0', 0) / total
                overlap_sq = max(0, 2 * p0 - 1)
                contribution = (info['w_p'] * info['w_q']
                                * info['norm_sq_p'] * info['norm_sq_q']
                                * overlap_sq * info['multiplier'])
                G4_measured += contribution

            # S4 from measurements
            S4_measured = 0.0
            for idx in range(g4_end, s4_end):
                info = all_info[idx]
                pub_result = result[idx]
                counts = pub_result.data.m.get_counts()
                total = sum(counts.values())
                p0 = counts.get('0', 0) / total
                overlap_sq = max(0, 2 * p0 - 1)
                contribution = (info['w_p'] * info['w_q']
                                * info['norm_sq_p'] * info['norm_sq_q']
                                * overlap_sq * info['multiplier'])
                S4_measured += contribution

            S2, S4_classical = measurer.compute_classical_S(rho)
            D2_m = S2 - G2_measured
            D4_mixed = S4_classical - G4_measured    # approach A
            D4_both = S4_measured - G4_measured       # approach B
            ratio_classical = S2**2 / S4_classical
            ratio_measured = S2**2 / S4_measured if S4_measured > 0 else ratio_classical

            results.append({
                'name': name, 'is_be': is_be,
                'S2': S2,
                'S4_classical': S4_classical, 'S4_measured': S4_measured,
                'S2sq_S4_classical': ratio_classical,
                'S2sq_S4_measured': ratio_measured,
                'G2_theory': theory['G_2'], 'G2_measured': G2_measured,
                'G4_theory': theory['G_4'], 'G4_measured': G4_measured,
                'S4_theory': theory['S_4'],
                'D2_theory': theory['D_2'], 'D2_measured': D2_m,
                'D4_theory': theory['D_4'],
                'D4_mixed': D4_mixed,    # S4 classical - G4 measured
                'D4_both': D4_both,      # S4 measured - G4 measured
            })
    else:
        for name, rho, is_be in states:
            theory = compute_features_theory(rho)
            S2, S4 = measurer.compute_classical_S(rho)
            results.append({
                'name': name, 'is_be': is_be,
                'S2': S2,
                'S4_classical': S4, 'S4_measured': S4,
                'S2sq_S4_classical': theory['S2sq_S4'],
                'S2sq_S4_measured': theory['S2sq_S4'],
                'G2_theory': theory['G_2'], 'G2_measured': theory['G_2'],
                'G4_theory': theory['G_4'], 'G4_measured': theory['G_4'],
                'S4_theory': theory['S_4'],
                'D2_theory': theory['D_2'], 'D2_measured': theory['D_2'],
                'D4_theory': theory['D_4'],
                'D4_mixed': theory['D_4'],
                'D4_both': theory['D_4'],
            })

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # S4 comparison
    print("\n--- S_4 Classical vs Measured ---")
    header_s4 = (f"{'State':<22} {'Type':<5} {'S4_class':<10} {'S4_meas':<10} "
                 f"{'ratio':<8}")
    print(header_s4)
    print("-" * len(header_s4))
    for r in results:
        st = "BE" if r['is_be'] else "SEP"
        s4_ratio = r['S4_measured'] / r['S4_classical'] if r['S4_classical'] > 0 else 0
        print(f"{r['name']:<22} {st:<5} {r['S4_classical']:<10.6f} "
              f"{r['S4_measured']:<10.6f} {s4_ratio:<8.4f}")

    # D4 comparison (both approaches)
    print("\n--- D_4: Mixed vs Both-Measured ---")
    header = (f"{'State':<22} {'Type':<5} {'D4_thy':<10} {'D4_mixed':<10} "
              f"{'D4_both':<10}")
    print(header)
    print("-" * len(header))
    for r in results:
        st = "BE" if r['is_be'] else "SEP"
        print(f"{r['name']:<22} {st:<5} {r['D4_theory']:<10.6f} "
              f"{r['D4_mixed']:<10.6f} {r['D4_both']:<10.6f}")

    # D2 results
    print("\n--- D_2 ---")
    header_d2 = f"{'State':<22} {'Type':<5} {'D2_thy':<10} {'D2_meas':<10}"
    print(header_d2)
    print("-" * len(header_d2))
    for r in results:
        st = "BE" if r['is_be'] else "SEP"
        print(f"{r['name']:<22} {st:<5} {r['D2_theory']:<10.6f} "
              f"{r['D2_measured']:<10.6f}")

    # Two-feature classifier: approach A (S4 classical)
    print("\n" + "=" * 70)
    print("TWO-FEATURE CLASSIFIER")
    print(f"  D_4 < {CLASSIFIER_INTERCEPT} + {CLASSIFIER_SLOPE} * (S2^2/S4)")
    print("=" * 70)

    print("\n--- Approach A: D4 = S4_classical - G4_measured ---")
    header2 = (f"{'State':<22} {'True':<6} {'D4_mixed':<10} {'Threshold':<10} "
               f"{'Pred':<6} {'Result':<6}")
    print(header2)
    print("-" * len(header2))

    correct_A = 0
    for r in results:
        threshold = CLASSIFIER_INTERCEPT + CLASSIFIER_SLOPE * r['S2sq_S4_classical']
        pred_be = r['D4_mixed'] < threshold
        true_be = r['is_be']
        ok = pred_be == true_be
        if ok:
            correct_A += 1
        pred_label = "BE" if pred_be else "SEP"
        true_label = "BE" if true_be else "SEP"
        status = "OK" if ok else "WRONG"
        print(f"{r['name']:<22} {true_label:<6} {r['D4_mixed']:<10.6f} "
              f"{threshold:<10.6f} {pred_label:<6} {status:<6}")
    print(f"  Accuracy: {correct_A}/{len(results)}")

    # Two-feature classifier: approach B (S4 measured)
    print(f"\n--- Approach B: D4 = S4_measured - G4_measured ---")
    print(header2.replace('D4_mixed', 'D4_both '))
    print("-" * len(header2))

    correct_B = 0
    for r in results:
        threshold = CLASSIFIER_INTERCEPT + CLASSIFIER_SLOPE * r['S2sq_S4_measured']
        pred_be = r['D4_both'] < threshold
        true_be = r['is_be']
        ok = pred_be == true_be
        if ok:
            correct_B += 1
        pred_label = "BE" if pred_be else "SEP"
        true_label = "BE" if true_be else "SEP"
        status = "OK" if ok else "WRONG"
        print(f"{r['name']:<22} {true_label:<6} {r['D4_both']:<10.6f} "
              f"{threshold:<10.6f} {pred_label:<6} {status:<6}")
    print(f"  Accuracy: {correct_B}/{len(results)}")

    print(f"\nOverall: Approach A = {correct_A}/{len(results)}, "
          f"Approach B = {correct_B}/{len(results)}")

    # Feature distribution summary
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 70)

    sep_D2 = [r['D2_measured'] for r in results if not r['is_be']]
    be_D2 = [r['D2_measured'] for r in results if r['is_be']]
    sep_D4m = [r['D4_mixed'] for r in results if not r['is_be']]
    be_D4m = [r['D4_mixed'] for r in results if r['is_be']]
    sep_D4b = [r['D4_both'] for r in results if not r['is_be']]
    be_D4b = [r['D4_both'] for r in results if r['is_be']]

    if sep_D2 and be_D2:
        print(f"\nD_2:")
        print(f"  SEP: mean={np.mean(sep_D2):.6f}, range=[{min(sep_D2):.6f}, {max(sep_D2):.6f}]")
        print(f"  BE:  mean={np.mean(be_D2):.6f}, range=[{min(be_D2):.6f}, {max(be_D2):.6f}]")
    if sep_D4m and be_D4m:
        print(f"\nD_4 (mixed = S4_class - G4_meas):")
        print(f"  SEP: mean={np.mean(sep_D4m):.6f}, range=[{min(sep_D4m):.6f}, {max(sep_D4m):.6f}]")
        print(f"  BE:  mean={np.mean(be_D4m):.6f}, range=[{min(be_D4m):.6f}, {max(be_D4m):.6f}]")
    if sep_D4b and be_D4b:
        print(f"\nD_4 (both = S4_meas - G4_meas):")
        print(f"  SEP: mean={np.mean(sep_D4b):.6f}, range=[{min(sep_D4b):.6f}, {max(sep_D4b):.6f}]")
        print(f"  BE:  mean={np.mean(be_D4b):.6f}, range=[{min(be_D4b):.6f}, {max(be_D4b):.6f}]")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ibmq_two_feature_{timestamp}.json"

    save_data = {
        'timestamp': timestamp,
        'classifier': {
            'formula': 'D_4 < intercept + slope * (S2^2/S4)',
            'intercept': CLASSIFIER_INTERCEPT,
            'slope': CLASSIFIER_SLOPE,
            'note': 'D4_mixed uses S4_classical; D4_both uses S4_measured',
        },
        'config': {
            'n_states': n_states,
            'shots': shots,
            'max_eigvals': max_eigvals,
            'dry_run': dry_run,
        },
        'accuracy': {
            'approach_A_mixed': f"{correct_A}/{len(results)}",
            'approach_B_both': f"{correct_B}/{len(results)}",
        },
        'results': results,
    }
    if not dry_run:
        save_data['job_ids'] = job_id
        save_data['backend'] = backend.name
        save_data['calibration'] = cal_data
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {filename}")

    return results

# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IBM Quantum two-feature classifier experiment (D2 + D4)"
    )
    parser.add_argument('--shots', type=int, default=DEFAULT_SHOTS,
                         help=f'Shots per circuit (default: {DEFAULT_SHOTS})')
    parser.add_argument('--states', type=int, default=4,
                         help='Number of states to test (default: 4)')
    parser.add_argument('--max-eigvals', type=int, default=MAX_EIGVALS,
                         help=f'Max eigenvalues to use (default: {MAX_EIGVALS})')
    parser.add_argument('--dry-run', action='store_true',
                         help='Run with theoretical values only (no IBM Q)')
    parser.add_argument('--simulator', action='store_true',
                         help='Run on noisy simulator (FakeMarrakesh)')
    args = parser.parse_args()
    run_experiment(
        n_states=args.states,
        shots=args.shots,
        dry_run=args.dry_run,
        max_eigvals=args.max_eigvals,
        simulator=args.simulator,
    )

if __name__ == "__main__":
    main()
