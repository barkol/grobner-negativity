#!/usr/bin/env python3
"""
Simulation of bound entanglement classification for many states (ideal + noisy).

Runs the same spectral measurement pipeline as the IBM Quantum experiment,
but on FakeMarrakesh simulator and as dry-run for theoretical values.

Usage:
    python simulate_be_many.py [--states 20] [--shots 4000] [--max-eigvals 4]
"""

import sys
import json
import time
import numpy as np
from scipy import linalg
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import psutil
import msvcrt  # Windows non-blocking keyboard input

# Import from the existing BE experiment pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ibmq_two_feature_batched import (
    horodecki_3x3,
    random_separable_3x3,
    tiles_bound_entangled_3x3,
    chessboard_3x3,
    compute_features_theory,
    SpectralMeasurer,
    realignment,
)

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    FAKE_AVAILABLE = True
except ImportError:
    FAKE_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False

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

# Unified colour palette (matches pgfplots blue!70!black, red!70!black)
_BLUE = '#0000B3'
_RED = '#B30000'

DEFAULT_SHOTS = 4000
MAX_EIGVALS = 4


# ---------------------------------------------------------------------------
# Entanglement verification helpers
# ---------------------------------------------------------------------------

def partial_transpose_3x3(rho):
    """Partial transpose w.r.t. subsystem B for a 3x3 bipartite system."""
    d = 3
    rho_pt = np.zeros_like(rho)
    for i in range(d):
        for j in range(d):
            rho_pt[i*d:(i+1)*d, j*d:(j+1)*d] = rho[i*d:(i+1)*d, j*d:(j+1)*d].T
    return rho_pt


def is_ppt(rho):
    """Check if state has positive partial transpose."""
    rho_pt = partial_transpose_3x3(rho)
    return np.min(np.linalg.eigvalsh(rho_pt)) >= -1e-10


def separability_gap(rho, n_random=3000, max_iter=500, n_restarts=15,
                     tol=1e-12, seed=123):
    """Hilbert-Schmidt distance from rho to the convex set of separable states.

    Fully-corrective Frank-Wolfe (NNLS column generation):
      Phase 1 — Random + structural seeds: NNLS warm-start.
      Phase 2 — Column generation: light alternating-eigenproblem oracle,
        many iterations, active-set pruning.
      Phase 3 — Active-set coordinate descent: re-optimise each active
        product state against the per-state residual.
    """
    from scipy.optimize import nnls as _nnls
    rng = np.random.default_rng(seed)
    dA = dB = 3
    d = dA * dB  # 9

    # --- helpers -----------------------------------------------------------
    def _proj_col(a, b):
        psi = np.kron(a, b)
        proj = np.outer(psi, psi.conj())
        return np.concatenate([proj.real.ravel(), proj.imag.ravel()])

    def _alt_eig(M_r, b_init):
        """Alternating eigenproblem from b-seed. Returns (a, b, val)."""
        b = b_init.copy()
        prev = -np.inf
        a = None
        for _ in range(60):
            M_A = np.einsum('ijmk,j,k->im', M_r, b.conj(), b)
            a = np.linalg.eigh(M_A)[1][:, -1]
            M_B = np.einsum('ijmk,i,m->jk', M_r, a.conj(), a)
            ev, evc = np.linalg.eigh(M_B)
            b = evc[:, -1]
            val = np.real(ev[-1])
            if abs(val - prev) < 1e-14:
                break
            prev = val
        return a, b, prev

    def _alt_eig_a(M_r, a_init):
        """Alternating eigenproblem from a-seed (reversed)."""
        a = a_init.copy()
        prev = -np.inf
        b = None
        for _ in range(60):
            M_B = np.einsum('ijmk,i,m->jk', M_r, a.conj(), a)
            b = np.linalg.eigh(M_B)[1][:, -1]
            M_A = np.einsum('ijmk,j,k->im', M_r, b.conj(), b)
            ev, evc = np.linalg.eigh(M_A)
            a = evc[:, -1]
            val = np.real(ev[-1])
            if abs(val - prev) < 1e-14:
                break
            prev = val
        return a, b, prev

    def _best_product(M, a_extra=None, b_extra=None):
        """Find product state maximising <a⊗b|M|a⊗b> — light oracle."""
        Mr = M.reshape(dA, dB, dA, dB)
        best_val = -np.inf
        best_ab = (None, None)
        # Random b-seeds
        for _ in range(n_restarts):
            b0 = rng.standard_normal(dB) + 1j * rng.standard_normal(dB)
            b0 /= np.linalg.norm(b0)
            a, b, val = _alt_eig(Mr, b0)
            if val > best_val:
                best_val = val; best_ab = (a, b)
        # Random a-seeds
        for _ in range(n_restarts):
            a0 = rng.standard_normal(dA) + 1j * rng.standard_normal(dA)
            a0 /= np.linalg.norm(a0)
            a, b, val = _alt_eig_a(Mr, a0)
            if val > best_val:
                best_val = val; best_ab = (a, b)
        # Structural seeds (capped to avoid O(active_set) scaling)
        _MAX_SEEDS = 20
        if b_extra:
            for b0 in b_extra[:_MAX_SEEDS]:
                a, b, val = _alt_eig(Mr, b0)
                if val > best_val:
                    best_val = val; best_ab = (a, b)
        if a_extra:
            for a0 in a_extra[:_MAX_SEEDS]:
                a, b, val = _alt_eig_a(Mr, a0)
                if val > best_val:
                    best_val = val; best_ab = (a, b)
        return best_ab, best_val

    def _solve(cols_list):
        """Unconstrained NNLS — fast, no trace penalty."""
        A = np.column_stack(cols_list)
        w, _ = _nnls(A, rho_vec)
        sv = A @ w
        sig = sv[:d*d].reshape(d, d) + 1j * sv[d*d:].reshape(d, d)
        return sig, np.linalg.norm(rho - sig, 'fro'), w

    # --- Structural seeds --------------------------------------------------
    eigvals, eigvecs = np.linalg.eigh(rho)
    a_seeds = []
    b_seeds = []
    cols = []
    ab_list = []

    # Eigenvector SVD seeds
    for k in range(d):
        if eigvals[k] < 1e-12:
            continue
        V = eigvecs[:, k].reshape(dA, dB)
        U_svd, s, Vh = np.linalg.svd(V, full_matrices=False)
        for si in range(len(s)):
            if s[si] < 1e-8:
                break
            a, b = U_svd[:, si], Vh[si, :].conj()
            a /= np.linalg.norm(a); b /= np.linalg.norm(b)
            ab_list.append((a, b))
            cols.append(_proj_col(a, b))
            a_seeds.append(a.copy())
            b_seeds.append(b.copy())

    # Partial trace eigenvectors — pairwise products
    rho_A = np.einsum('iajb->ij', rho.reshape(dA, dB, dA, dB))
    rho_B = np.einsum('iajb->ab', rho.reshape(dA, dB, dA, dB))
    _, evA = np.linalg.eigh(rho_A)
    _, evB = np.linalg.eigh(rho_B)
    for i in range(dA):
        a_seeds.append(evA[:, i].copy())
        for j in range(dB):
            if j == 0:
                b_seeds.append(evB[:, j].copy())
            ab_list.append((evA[:, i], evB[:, j]))
            cols.append(_proj_col(evA[:, i], evB[:, j]))

    # Partial transpose eigenvector seeds
    rho_pt = partial_transpose_3x3(rho)
    pt_ev, pt_evc = np.linalg.eigh(rho_pt)
    for k in range(d):
        if abs(pt_ev[k]) < 1e-12:
            continue
        V = pt_evc[:, k].reshape(dA, dB)
        U_svd, s, Vh = np.linalg.svd(V, full_matrices=False)
        if s[0] > 1e-8:
            a, b = U_svd[:, 0], Vh[0, :].conj()
            a /= np.linalg.norm(a); b /= np.linalg.norm(b)
            ab_list.append((a, b))
            cols.append(_proj_col(a, b))
            a_seeds.append(a.copy()); b_seeds.append(b.copy())

    # --- Phase 1: random product states + NNLS -----------------------------
    for _ in range(n_random):
        a = rng.standard_normal(dA) + 1j * rng.standard_normal(dA)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(dB) + 1j * rng.standard_normal(dB)
        b /= np.linalg.norm(b)
        ab_list.append((a, b))
        cols.append(_proj_col(a, b))

    rho_vec = np.concatenate([rho.real.ravel(), rho.imag.ravel()])
    sigma, gap, w = _solve(cols)

    # Early exit if already converged (high-rank SEP states)
    if gap < _GAP_THRESHOLD / 3:
        return gap

    # --- Phase 2: column generation ----------------------------------------
    stall = 0
    for it in range(max_iter):
        residual = rho - sigma
        # Active-set seeds
        active_idx = [i for i, wi in enumerate(w) if wi > 1e-15]
        a_ex = a_seeds + [ab_list[i][0] for i in active_idx]
        b_ex = b_seeds + [ab_list[i][1] for i in active_idx]

        (a_new, b_new), ov = _best_product(residual, a_ex, b_ex)
        if ov <= tol:
            break

        ab_list.append((a_new, b_new))
        cols.append(_proj_col(a_new, b_new))
        sigma, new_gap, w = _solve(cols)

        if gap - new_gap < tol:
            stall += 1
            if stall >= 15:
                break
        else:
            stall = 0
        gap = new_gap

        # Early exit once clearly below threshold
        if gap < _GAP_THRESHOLD / 3:
            return gap

        # Prune every 25 iterations
        if it % 25 == 24:
            keep = [i for i, wi in enumerate(w) if wi > 1e-15]
            if len(keep) < len(cols):
                ab_list = [ab_list[i] for i in keep]
                cols = [cols[i] for i in keep]
                w = w[np.array(keep)]

    # --- Phase 3: active-set coordinate descent ----------------------------
    for _round in range(5):
        improved = False
        active_idx = [i for i, wi in enumerate(w) if wi > 1e-15]

        for idx in active_idx:
            a_old, b_old = ab_list[idx]
            psi_old = np.kron(a_old, b_old)
            P_old = np.outer(psi_old, psi_old.conj())
            M_i = rho - sigma + w[idx] * P_old

            a_ex = a_seeds + [ab_list[j][0] for j in active_idx]
            b_ex = b_seeds + [ab_list[j][1] for j in active_idx]
            (a_new, b_new), val = _best_product(M_i, a_ex, b_ex)

            if val > tol:
                psi_new = np.kron(a_new, b_new)
                if np.real(psi_new.conj() @ M_i @ psi_new) > \
                   np.real(psi_old.conj() @ M_i @ psi_old) + 1e-14:
                    ab_list[idx] = (a_new, b_new)
                    cols[idx] = _proj_col(a_new, b_new)
                    improved = True

        if not improved:
            break
        sigma, new_gap, w = _solve(cols)
        if gap - new_gap < tol:
            break
        gap = new_gap
        # Prune
        keep = [i for i, wi in enumerate(w) if wi > 1e-15]
        if len(keep) < len(cols):
            ab_list = [ab_list[i] for i in keep]
            cols = [cols[i] for i in keep]
            w = w[np.array(keep)]

    return gap


def gilbert_gap(rho, n_trials=100000, n_corrections=10000, seed=123):
    """Wieśniak's original Gilbert algorithm (CSSFinder) for benchmarking.

    Pure Gilbert: no NNLS, just convex combination updates with exact
    line search.  Slower convergence but simple and well-tested.

    Reference: Wieśniak et al., Sci. Rep. 13, 18850 (2023);
               github.com/wiesnim9/CSSFinder
    """
    rng = np.random.default_rng(seed)
    dA = dB = 3
    d = dA * dB

    def _random_product():
        a = rng.standard_normal(dA) + 1j * rng.standard_normal(dA)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(dB) + 1j * rng.standard_normal(dB)
        b /= np.linalg.norm(b)
        psi = np.kron(a, b)
        return np.outer(psi, psi.conj())

    def _hs(A, B):
        return np.real(np.trace(A.conj().T @ B))

    def _optimize_product(v, residual):
        """OptimizeBS: refine product state via small-angle unitary rotations."""
        best_overlap = _hs(v, residual)
        for _ in range(5 * dA * dB):
            # Random small-angle unitary on subsystem A or B
            sub = rng.integers(2)
            ds = dA if sub == 0 else dB
            do = dB if sub == 0 else dA
            rand_proj = rng.standard_normal(ds) + 1j * rng.standard_normal(ds)
            rand_proj /= np.linalg.norm(rand_proj)
            rand_proj = np.outer(rand_proj, rand_proj.conj())
            angle = 0.01 * np.pi
            U_sub = (np.exp(1j * angle) - 1) * rand_proj + np.eye(ds)
            if sub == 0:
                U = np.kron(U_sub, np.eye(do))
            else:
                U = np.kron(np.eye(do), U_sub)
            v_rot = U @ v @ U.conj().T
            if _hs(v_rot, residual) < best_overlap:
                U = U.conj().T
                v_rot = U @ v @ U.conj().T
            while _hs(v_rot, residual) > best_overlap:
                v = v_rot
                best_overlap = _hs(v, residual)
                v_rot = U @ v_rot @ U.conj().T
        return v

    # Initialise sigma as maximally mixed (a valid separable state)
    sigma = np.eye(d, dtype=complex) / d
    residual = rho - sigma
    dd1 = _hs(sigma, residual)

    counter = 0
    for trial in range(n_trials):
        if counter >= n_corrections:
            break
        v = _random_product()
        # Check if this product state improves overlap with residual
        if _hs(v, residual) > dd1:
            v = _optimize_product(v, residual)
            # Exact line search: min ||rho - (cc*sigma + (1-cc)*v)||^2
            aa3 = _hs(v, v)
            aa2 = 2 * _hs(rho, v)
            aa4 = 2 * _hs(rho, sigma)
            aa5 = 2 * _hs(sigma, v)
            aa6 = _hs(sigma, sigma)
            bb2 = -aa4 + aa2 + aa5 - 2 * aa3
            bb3 = aa6 - aa5 + aa3
            if abs(bb3) > 1e-15:
                cc = -bb2 / (2 * bb3)
            else:
                cc = 0.5
            if 0 <= cc <= 1:
                sigma = cc * sigma + (1 - cc) * v
                residual = rho - sigma
                dd1 = _hs(sigma, residual)
                counter += 1
                if _hs(residual, residual) < 1e-14:
                    break

    return np.sqrt(max(0, _hs(residual, residual)))


_GAP_THRESHOLD = 1e-3   # HS-distance above which state is entangled


def is_verified_be(rho):
    """Verify bound entanglement: PPT + entangled (Caratheodory gap)."""
    if not is_ppt(rho):
        return False
    # Fast check: realignment criterion
    R = realignment(rho)
    if np.linalg.norm(R, ord='nuc') > 1 + 1e-10:
        return True
    # Slower but more powerful: iterative Caratheodory separability gap
    gap = separability_gap(rho)
    detected = gap > _GAP_THRESHOLD
    tag = "BE" if detected else "SEP?"
    print(f"    Caratheodory gap = {gap:.6f} -> {tag}")
    return detected


def generate_states(n_states, rng):
    """Generate ~100 BE states plus separable controls.

    n_states controls the number of random separable states (n_states // 4).
    BE states: dense Horodecki sweep, Tiles, Chessboard parameter grid,
    noise admixtures, and cross-family mixtures.  Analytically known BE
    states are included directly; derived states are verified via
    PPT + iterative Caratheodory gap.

    Returns:
        List of (name, rho, is_be) tuples.
    """
    states = []
    n_sep = max(n_states // 4, 27)

    # --- Separable states: varied rank (low, medium, high, full) ---
    rank_schedule = (
        [2] * 3 + [3] * 3 + [4] * 3          # low-rank  (9 states)
        + [5, 6, 7, 8] * 2                    # medium    (8 states)
        + [12, 20, 30, 50, 81] * 2            # high/full (10 states)
    )
    # pad or trim to n_sep
    while len(rank_schedule) < n_sep:
        rank_schedule.append(int(rng.integers(2, 82)))
    rank_schedule = rank_schedule[:n_sep]
    for i, n_terms in enumerate(rank_schedule):
        rho = random_separable_3x3(rng, n_terms=n_terms)
        states.append((f'sep_{i+1}(r{n_terms})', rho, False))

    # === Analytically proven BE families (no verification needed) ========

    # --- Horodecki family: BE for a in (0, 1) — dense sweep ---
    hor_a_values = np.linspace(0.02, 0.98, 25)
    for a in hor_a_values:
        states.append((f'hor({a:.2f})', horodecki_3x3(float(a)), True))

    # --- Tiles UPB state ---
    rho_tiles = tiles_bound_entangled_3x3()
    states.append(('tiles', rho_tiles, True))

    # --- Chessboard (Bruss-Peres) family: 12 parameter sets ---
    # BE when a*b != m*n (Phys. Rev. A 61, 030301(R))
    chess_configs = [
        ((1, 2, 1, 1, 1, 1), 'A'),    # ab=2, mn=1
        ((2, 1, 1, 1, 1, 1), 'B'),    # ab=2, mn=1
        ((1, 1, 1, 1, 2, 1), 'C'),    # ab=1, mn=2
        ((1, 1, 1, 1, 1, 2), 'D'),    # ab=1, mn=2
        ((3, 1, 1, 1, 1, 1), 'E'),    # ab=3, mn=1
        ((1, 3, 1, 1, 1, 1), 'F'),    # ab=3, mn=1
        ((1, 1, 1, 1, 3, 1), 'G'),    # ab=1, mn=3
        ((1, 1, 1, 1, 1, 3), 'H'),    # ab=1, mn=3
        ((2, 2, 1, 1, 1, 1), 'I'),    # ab=4, mn=1
        ((1, 1, 1, 1, 2, 2), 'J'),    # ab=1, mn=4
        ((3, 2, 1, 1, 1, 1), 'K'),    # ab=6, mn=1
        ((2, 1, 2, 1, 1, 1), 'L'),    # ab=2, mn=1, c=2
    ]
    chess_rhos = {}
    for params, label in chess_configs:
        rho = chessboard_3x3(*params)
        states.append((f'chess_{label}', rho, True))
        chess_rhos[label] = rho

    # === Derived BE candidates (verified by Caratheodory gap) ============

    n_verified = 0
    n_skipped = 0

    def _try_add(name, rho_candidate):
        nonlocal n_verified, n_skipped
        rho_candidate = rho_candidate / np.trace(rho_candidate).real
        if is_verified_be(rho_candidate):
            states.append((name, rho_candidate, True))
            n_verified += 1
        else:
            print(f"  [skip] {name}: gap < threshold")
            n_skipped += 1

    I9 = np.eye(9) / 9

    # --- Noise admixtures: p*rho_BE + (1-p)*I/9 ---
    noise_bases = [
        ('tiles', rho_tiles),
        ('hor(0.10)', horodecki_3x3(0.1)),
        ('hor(0.25)', horodecki_3x3(0.25)),
        ('hor(0.50)', horodecki_3x3(0.5)),
        ('hor(0.75)', horodecki_3x3(0.75)),
        ('hor(0.90)', horodecki_3x3(0.9)),
        ('chess_A', chess_rhos['A']),
        ('chess_C', chess_rhos['C']),
        ('chess_E', chess_rhos['E']),
    ]
    for base_name, rho_base in noise_bases:
        for p in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
            _try_add(f'{base_name}+I({1-p:.0%})',
                     p * rho_base + (1 - p) * I9)

    # --- Cross-family mixtures ---
    mix_defs = [
        # hor + tiles  (various a, ratios)
        ('mix(hor.10+tiles)', horodecki_3x3(0.1), rho_tiles, 0.5),
        ('mix(hor.25+tiles)', horodecki_3x3(0.25), rho_tiles, 0.5),
        ('mix(hor.50+tiles)', horodecki_3x3(0.5), rho_tiles, 0.5),
        ('mix(hor.75+tiles)', horodecki_3x3(0.75), rho_tiles, 0.5),
        ('mix(hor.50+tiles,0.7)', horodecki_3x3(0.5), rho_tiles, 0.7),
        ('mix(hor.50+tiles,0.3)', horodecki_3x3(0.5), rho_tiles, 0.3),
        # hor + chess
        ('mix(hor.25+chess_A)', horodecki_3x3(0.25), chess_rhos['A'], 0.5),
        ('mix(hor.50+chess_A)', horodecki_3x3(0.5), chess_rhos['A'], 0.5),
        ('mix(hor.75+chess_A)', horodecki_3x3(0.75), chess_rhos['A'], 0.5),
        # chess + tiles
        ('mix(chess_A+tiles)', chess_rhos['A'], rho_tiles, 0.5),
        ('mix(chess_A+tiles,0.7)', chess_rhos['A'], rho_tiles, 0.7),
        ('mix(chess_C+tiles)', chess_rhos['C'], rho_tiles, 0.5),
        # hor + hor
        ('mix(hor.10+hor.90)', horodecki_3x3(0.1), horodecki_3x3(0.9), 0.5),
        ('mix(hor.25+hor.75)', horodecki_3x3(0.25), horodecki_3x3(0.75), 0.5),
        ('mix(hor.30+hor.70)', horodecki_3x3(0.3), horodecki_3x3(0.7), 0.5),
        # chess + chess
        ('mix(chess_A+chess_C)', chess_rhos['A'], chess_rhos['C'], 0.5),
    ]
    for name, rho1, rho2, p in mix_defs:
        _try_add(name, p * rho1 + (1 - p) * rho2)

    # --- Three-way mixtures ---
    tri_defs = [
        ('mix3(hor.50+tiles+chess_A)',
         [horodecki_3x3(0.5), rho_tiles, chess_rhos['A']]),
        ('mix3(hor.25+tiles+chess_C)',
         [horodecki_3x3(0.25), rho_tiles, chess_rhos['C']]),
        ('mix3(hor.10+hor.90+tiles)',
         [horodecki_3x3(0.1), horodecki_3x3(0.9), rho_tiles]),
        ('mix3(chess_A+chess_C+tiles)',
         [chess_rhos['A'], chess_rhos['C'], rho_tiles]),
    ]
    for name, rhos in tri_defs:
        _try_add(name, sum(rhos) / len(rhos))

    print(f"  Derived states: {n_verified} verified, {n_skipped} skipped")
    return states


def run_theoretical(states):
    """Compute theoretical features for all states (no circuits)."""
    results = []
    for name, rho, is_be in states:
        theory = compute_features_theory(rho)
        results.append({
            'name': name, 'is_be': is_be,
            'D4': theory['D_4'],
            'S2sq_S4': theory['S2sq_S4'],
            'D2': theory['D_2'],
            'S2': theory['S_2'],
            'S4': theory['S_4'],
            'G2': theory['G_2'],
            'G4': theory['G_4'],
        })
    return results


_proc = psutil.Process()


def _resource_postfix():
    """Return a short string with CPU and memory usage for tqdm postfix."""
    cpu = psutil.cpu_percent(interval=None)
    rss_mb = _proc.memory_info().rss / (1024 * 1024)
    return f"CPU {cpu:.0f}% | {rss_mb:.0f}MB"


CHECKPOINT_FILE = Path(__file__).parent / "_sim_be_checkpoint.json"

# Global pause flag — toggled by pressing P in the console
_paused = False


def _drain_keys():
    """Consume all pending keypresses; toggle pause on 'p'/'P'."""
    global _paused
    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b'p', b'P'):
            _paused = not _paused


def _check_pause():
    """Drain keyboard buffer and block while paused."""
    _drain_keys()
    if _paused:
        tqdm.write("\n  *** PAUSED — press P to resume ***")
        while _paused:
            time.sleep(0.2)
            _drain_keys()
        tqdm.write("  *** RESUMED ***\n")


def _save_checkpoint(counts_cache, last_batch):
    """Persist raw measurement counts so a resumed run skips finished batches."""
    data = {'last_batch': last_batch, 'counts': counts_cache}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)


def _load_checkpoint():
    """Return (counts_cache, last_batch) or (None, -1) if no checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        return data['counts'], data['last_batch']
    return None, -1


def run_noisy_simulation(states, shots, max_eigvals, resume=False):
    """Run full SWAP-test pipeline on FakeMarrakesh with per-batch checkpoints."""
    if not FAKE_AVAILABLE:
        raise RuntimeError("FakeMarrakesh not available")

    fake = FakeMarrakesh()
    if AER_AVAILABLE:
        noise_model = NoiseModel.from_backend(fake)
        backend = AerSimulator(noise_model=noise_model)
        print(f"  Backend: AerSimulator + FakeMarrakesh noise model")
    else:
        backend = fake
        print(f"  Backend: FakeMarrakesh ({fake.num_qubits} qubits)")

    # optimization_level=1 is much faster than 3 and sufficient for simulation
    pm = generate_preset_pass_manager(optimization_level=1, backend=fake)
    measurer = SpectralMeasurer(backend, shots=shots, max_eigvals=max_eigvals)
    measurer.pass_manager = pm

    # Prepare all circuits
    print("  Preparing circuits...")
    all_circuits = []
    all_info = []
    state_boundaries = [0]

    for name, rho, is_be in states:
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

    n_total = len(all_circuits)
    print(f"  Total circuits: {n_total}")

    # Transpile in small batches with progress bar
    TRANSPILE_BATCH = 100
    n_tbatches = (n_total + TRANSPILE_BATCH - 1) // TRANSPILE_BATCH
    transpiled = []
    tbar = tqdm(range(n_tbatches), desc="Transpiling", unit="batch",
                ncols=100, leave=True)
    for tb in tbar:
        _check_pause()
        s = tb * TRANSPILE_BATCH
        e = min(s + TRANSPILE_BATCH, n_total)
        tbar.set_postfix_str(_resource_postfix())
        chunk = measurer.pass_manager.run(all_circuits[s:e])
        if not isinstance(chunk, list):
            chunk = [chunk]
        transpiled.extend(chunk)
    depths = [c.depth() for c in transpiled]
    print(f"  Depths: mean={np.mean(depths):.0f}, max={np.max(depths)}, "
          f"min={np.min(depths)}")

    # Execute in batches with 4 parallel workers + checkpointing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if AER_AVAILABLE:
        from qiskit_aer.primitives import SamplerV2 as Sampler
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler

    BATCH_SIZE = 100
    N_WORKERS = 4
    n_batches = (len(transpiled) + BATCH_SIZE - 1) // BATCH_SIZE

    # Try to resume from checkpoint
    counts_cache, done_batch = (None, -1)
    if resume:
        counts_cache, done_batch = _load_checkpoint()
        if counts_cache is not None:
            print(f"  Resuming from checkpoint (batches 1-{done_batch+1} done)")
        else:
            done_batch = -1

    if counts_cache is None:
        counts_cache = [None] * len(transpiled)

    def _run_batch(batch_idx):
        """Run a single batch and return (batch_idx, {global_idx: counts})."""
        s = batch_idx * BATCH_SIZE
        e = min(s + BATCH_SIZE, len(transpiled))
        if AER_AVAILABLE:
            sampler = Sampler(backend=backend)
        else:
            sampler = Sampler(mode=backend)
        job = sampler.run(transpiled[s:e], shots=shots)
        result = job.result()
        out = {}
        for i, idx in enumerate(range(s, e)):
            out[idx] = result[i].data.m.get_counts()
        return batch_idx, out

    pbar = tqdm(total=n_batches, desc="Simulating", unit="batch",
                ncols=100, leave=True)
    # Skip cached batches in progress bar
    for bi in range(n_batches):
        if bi <= done_batch:
            pbar.update(1)
            pbar.set_postfix_str("cached")

    # Collect remaining batches
    todo = [bi for bi in range(n_batches) if bi > done_batch]

    # Process in chunks of N_WORKERS
    for chunk_start in range(0, len(todo), N_WORKERS):
        _check_pause()
        chunk = todo[chunk_start:chunk_start + N_WORKERS]
        pbar.set_postfix_str(f"workers: {len(chunk)} | {_resource_postfix()}")

        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_run_batch, bi): bi for bi in chunk}
            for fut in as_completed(futures):
                bi, result_dict = fut.result()
                for idx, counts in result_dict.items():
                    counts_cache[idx] = counts
                pbar.update(1)

        # Checkpoint after each chunk of workers finishes
        _save_checkpoint(counts_cache, chunk[-1])
    pbar.close()

    # Process results per state
    results = []
    for s_idx, (name, rho, is_be) in enumerate(states):
        start, g2_end, g4_end, s4_end = state_boundaries[s_idx + 1]
        theory = compute_features_theory(rho)

        # G2 from measurements
        G2_measured = 0.0
        for idx in range(start, g2_end):
            info = all_info[idx]
            counts = counts_cache[idx]
            total = sum(counts.values())
            p0 = counts.get('0', 0) / total
            overlap_sq = max(0, 2 * p0 - 1)
            G2_measured += info['lambda_i'] * info['lambda_j'] * overlap_sq

        # G4 from measurements
        G4_measured = 0.0
        for idx in range(g2_end, g4_end):
            info = all_info[idx]
            counts = counts_cache[idx]
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
            counts = counts_cache[idx]
            total = sum(counts.values())
            p0 = counts.get('0', 0) / total
            overlap_sq = max(0, 2 * p0 - 1)
            contribution = (info['w_p'] * info['w_q']
                            * info['norm_sq_p'] * info['norm_sq_q']
                            * overlap_sq * info['multiplier'])
            S4_measured += contribution

        S2, S4_classical = measurer.compute_classical_S(rho)
        D2_mixed = S2 - G2_measured
        D4_mixed = S4_classical - G4_measured
        D4_both = S4_measured - G4_measured
        ratio_classical = S2**2 / S4_classical
        ratio_measured = S2**2 / S4_measured if S4_measured > 0 else ratio_classical

        results.append({
            'name': name, 'is_be': is_be,
            'D4_theory': theory['D_4'],
            'D4_mixed': D4_mixed,
            'D4_both': D4_both,
            'S2sq_S4_theory': theory['S2sq_S4'],
            'S2sq_S4_classical': ratio_classical,
            'S2sq_S4_measured': ratio_measured,
            'S2': S2,
            'S4_classical': S4_classical,
            'S4_measured': S4_measured,
            'D2_mixed': D2_mixed,
            'G2_theory': theory['G_2'], 'G2_measured': G2_measured,
            'G4_theory': theory['G_4'], 'G4_measured': G4_measured,
        })

    # Clean up checkpoint on success
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    return results


# ---------------------------------------------------------------------------
# RBF SVM classifier: decision_function(x) > 0  =>  BE
# Features: (D4, D2, G2, S2^2/S4), StandardScaler + RBF kernel.
# Trained on 50,000 random separable + 96 BE states.
# Test accuracy: 96/96 BE recall, 0.038% FP on 100k random separable.
# ---------------------------------------------------------------------------
_SVM_MODEL_PATH = Path(__file__).parent / 'svm_classifier.json'
_svm_cache = {}


def _load_svm():
    """Load SVM model from JSON (cached)."""
    if _svm_cache:
        return _svm_cache
    with open(_SVM_MODEL_PATH) as f:
        data = json.load(f)
    _svm_cache['mean'] = np.array(data['scaler_mean'])
    _svm_cache['scale'] = np.array(data['scaler_scale'])
    _svm_cache['sv'] = np.array(data['support_vectors'])
    _svm_cache['dual_coef'] = np.array(data['dual_coef'][0])
    _svm_cache['intercept'] = data['intercept'][0]
    _svm_cache['gamma'] = data['gamma']
    return _svm_cache


def classify(D4, D2, G2, S2sq_S4):
    """RBF SVM classifier: decision > 0 => BE.

    Uses features (D4, D2, G2, S2^2/S4) with StandardScaler + RBF kernel.
    """
    m = _load_svm()
    x = np.array([D4, D2, G2, S2sq_S4])
    x_scaled = (x - m['mean']) / m['scale']
    dists_sq = np.sum((m['sv'] - x_scaled) ** 2, axis=1)
    K = np.exp(-m['gamma'] * dists_sq)
    decision = np.dot(m['dual_coef'], K) + m['intercept']
    return decision > 0


def _state_type(name):
    """Classify state name as 'sep', 'horodecki', 'tiles', 'chess', or 'mix'."""
    if name.startswith('sep'):
        return 'sep'
    elif name.startswith('hor'):
        return 'horodecki'
    elif name.startswith('mix'):
        return 'mix'
    elif 'tiles' in name:
        return 'tiles'
    elif 'chess' in name:
        return 'chess'
    return 'sep'


def make_figure(theory_results, noisy_results, output_path):
    """Generate scatter plot with theory and noisy simulation points.

    Left panel:  S2^2/S4 vs D4  (classic 2D view with quadratic contour).
    Right panel: G2 vs D2       (shows the features that rescue chess_L).
    Markers: Horodecki = x, Tiles = diamond, Chess = square, Mix = triangle.
    Colour: BE = red, SEP = blue.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _theory_markers = {
        'sep': dict(marker='x', color=_BLUE, s=50, linewidths=1.5),
        'horodecki': dict(marker='x', color=_RED, s=50, linewidths=1.5),
        'tiles': dict(marker='D', s=55, linewidths=1.5,
                      facecolors='none', edgecolors=_RED),
        'chess': dict(marker='s', s=55, linewidths=1.5,
                      facecolors='none', edgecolors=_RED),
        'mix': dict(marker='^', s=55, linewidths=1.5,
                    facecolors='none', edgecolors=_RED),
    }

    # --- Panel (a): S2^2/S4 vs D4 with SVM decision contour ---
    ax = axes[0]
    ax.set_title('(a)', fontsize=12, loc='left')

    # SVM contour — evaluate classifier on a 2D grid of (D4, S2sq_S4)
    # projecting D2 and G2 through medians of the theory data
    d2_vals = np.array([r['D2'] for r in theory_results])
    g2_vals = np.array([r['G2'] for r in theory_results])
    d2_med = np.median(d2_vals)
    g2_med = np.median(g2_vals)

    m = _load_svm()
    S_range = np.linspace(0.8, 5.0, 150)
    D4_range = np.linspace(-0.02, 0.20, 150)
    SS, DD4 = np.meshgrid(S_range, D4_range)
    ZZ = np.zeros_like(SS)
    for i in range(SS.shape[0]):
        for j in range(SS.shape[1]):
            x = np.array([DD4[i, j], d2_med, g2_med, SS[i, j]])
            x_s = (x - m['mean']) / m['scale']
            dists_sq = np.sum((m['sv'] - x_s) ** 2, axis=1)
            K = np.exp(-m['gamma'] * dists_sq)
            ZZ[i, j] = np.dot(m['dual_coef'], K) + m['intercept']
    ax.contour(SS, DD4, ZZ, levels=[0], colors='k', linewidths=1.2,
               linestyles='--')
    ax.contourf(SS, DD4, ZZ, levels=[-1e6, 0], colors=[_BLUE], alpha=0.04)
    ax.contourf(SS, DD4, ZZ, levels=[0, 1e6], colors=[_RED], alpha=0.04)

    for r in theory_results:
        stype = _state_type(r['name'])
        ax.scatter(r['S2sq_S4'], r['D4'], zorder=4,
                   **_theory_markers.get(stype, _theory_markers['sep']))
    ax.set_xlabel(r'$S_2^2 / S_4$', fontsize=12)
    ax.set_ylabel(r'$D_4 = S_4 - G_4$', fontsize=12)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')

    # --- Panel (b): G2 vs D2 ---
    ax = axes[1]
    ax.set_title('(b)', fontsize=12, loc='left')
    for r in theory_results:
        stype = _state_type(r['name'])
        ax.scatter(r['G2'], r['D2'], zorder=4,
                   **_theory_markers.get(stype, _theory_markers['sep']))
    ax.set_xlabel(r'$G_2$', fontsize=12)
    ax.set_ylabel(r'$D_2 = S_2 - G_2$', fontsize=12)

    # Shared legend
    sep_lbl = 'Separable'
    hor_lbl = 'Horodecki BE'
    axes[0].scatter([], [], marker='x', color=_BLUE, s=50, linewidths=1.5,
                    label=sep_lbl)
    axes[0].scatter([], [], marker='x', color=_RED, s=50, linewidths=1.5,
                    label=hor_lbl)
    axes[0].scatter([], [], marker='D', s=55, linewidths=1.5,
                    facecolors='none', edgecolors=_RED, label='Tiles BE')
    axes[0].scatter([], [], marker='s', s=55, linewidths=1.5,
                    facecolors='none', edgecolors=_RED, label='Chessboard BE')
    axes[0].scatter([], [], marker='^', s=55, linewidths=1.5,
                    facecolors='none', edgecolors=_RED, label='Mixed BE')
    axes[0].legend(fontsize=8, loc='upper left')

    # Accuracy annotation
    correct = sum(
        1 for r in theory_results
        if classify(r['D4'], r['D2'], r['G2'], r['S2sq_S4']) == r['is_be']
    )
    total = len(theory_results)
    axes[1].text(0.98, 0.98, f'Accuracy: {correct}/{total}',
                 transform=axes[1].transAxes, fontsize=10, ha='right',
                 va='top', bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', edgecolor='gray',
                                     alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Simulate BE classification for many states"
    )
    parser.add_argument('--states', type=int, default=20)
    parser.add_argument('--shots', type=int, default=DEFAULT_SHOTS)
    parser.add_argument('--max-eigvals', type=int, default=MAX_EIGVALS)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--theory-only', action='store_true',
                        help='Skip noisy simulation, only compute theory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("BOUND ENTANGLEMENT CLASSIFICATION - MANY STATES")
    print("=" * 70)
    print(f"Shots: {args.shots}, Max eigenvalues: {args.max_eigvals}")
    m = _load_svm()
    print(f"Classifier: RBF SVM (C={m['gamma']}, "
          f"{len(m['sv'])} SVs, features: D4, D2, G2, S2^2/S4)")

    # Generate states
    states = generate_states(args.states, rng)
    n_sep = sum(1 for _, _, be in states if not be)
    n_be = sum(1 for _, _, be in states if be)
    print(f"\nGenerated {len(states)} states ({n_sep} SEP + {n_be} BE):")
    for name, rho, is_be in states:
        label = "BE" if is_be else "SEP"
        rank = np.sum(np.linalg.eigvalsh(rho) > 1e-10)
        print(f"  {name}: {label} (rank {rank})")

    # Theoretical features
    print("\n" + "-" * 70)
    print("THEORETICAL FEATURES")
    print("-" * 70)
    theory_results = run_theoretical(states)

    theory_correct = 0
    for r in theory_results:
        pred_be = classify(r['D4'], r['D2'], r['G2'], r['S2sq_S4'])
        ok = pred_be == r['is_be']
        if ok:
            theory_correct += 1
        label = "BE" if r['is_be'] else "SEP"
        pred = "BE" if pred_be else "SEP"
        status = "OK" if ok else "WRONG"
        print(f"  {r['name']:<18} {label:<5} D4={r['D4']:<10.6f} "
              f"S2sq_S4={r['S2sq_S4']:<8.4f} -> {pred:<5} {status}")
    print(f"  Theory accuracy: {theory_correct}/{len(theory_results)}")

    # Noisy simulation (skip with --theory-only)
    noisy_results = None
    noisy_correct = 0
    if not args.theory_only:
        print("\n" + "-" * 70)
        print("NOISY SIMULATION (FakeMarrakesh)")
        print("-" * 70)
        print(f"  Press P to pause/resume between batches")
        noisy_results = run_noisy_simulation(states, args.shots, args.max_eigvals,
                                             resume=args.resume)

        for r in noisy_results:
            pred_be = classify(r['D4_mixed'], r['D2_mixed'], r['G2_measured'], r['S2sq_S4_classical'])
            ok = pred_be == r['is_be']
            if ok:
                noisy_correct += 1
            label = "BE" if r['is_be'] else "SEP"
            pred = "BE" if pred_be else "SEP"
            status = "OK" if ok else "WRONG"
            print(f"  {r['name']:<18} {label:<5} D4_m={r['D4_mixed']:<10.6f} "
                  f"S2sq_S4={r['S2sq_S4_classical']:<8.4f} -> {pred:<5} {status}")
        print(f"  Noisy accuracy: {noisy_correct}/{len(noisy_results)}")

    # Save results
    output_json = args.output or f"sim_be_classification_{timestamp}.json"
    save_data = {
        'timestamp': timestamp,
        'config': {
            'n_states': args.states,
            'shots': args.shots,
            'max_eigvals': args.max_eigvals,
            'theory_only': args.theory_only,
        },
        'classifier': {
            'type': 'rbf_svm',
            'features': ['D4', 'D2', 'G2', 'S2sq_S4'],
            'model_file': str(_SVM_MODEL_PATH.name),
            'test_fp_rate': '0.038% (38/100000)',
        },
        'theory': theory_results,
        'noisy': noisy_results,
        'accuracy': {
            'theory': f"{theory_correct}/{len(theory_results)}",
            'noisy': f"{noisy_correct}/{len(noisy_results)}" if noisy_results else "N/A",
        },
    }
    output_path = Path(__file__).parent / output_json
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # Generate figure
    fig_path = Path(__file__).parent / "fig_simulation_be_classification.pdf"
    make_figure(theory_results, noisy_results, fig_path)

    noisy_str = f"Noisy: {noisy_correct}/{len(noisy_results)}" if noisy_results else "Noisy: skipped"
    print("\n" + "=" * 70)
    print(f"DONE - Theory: {theory_correct}/{len(theory_results)}, {noisy_str}")
    print("=" * 70)


if __name__ == '__main__':
    main()
