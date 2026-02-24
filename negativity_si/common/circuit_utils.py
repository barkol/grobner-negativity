"""
Circuit execution and measurement correction utilities.
"""

import numpy as np
from typing import Tuple

try:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def run_circuit(
    qc,
    backend,
    shots: int,
) -> Tuple[float, float, int, int]:
    """
    Run a Hadamard test circuit and return expectation value.

    Args:
        qc: QuantumCircuit to run
        backend: Qiskit backend (AerSimulator or real)
        shots: Number of measurement shots

    Returns:
        Tuple of (expectation, std, n0_counts, circuit_depth)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for circuit execution")

    compiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    n0 = counts.get('0', 0)
    n1 = counts.get('1', 0)
    total = n0 + n1

    p0 = n0 / total
    expectation = 2 * p0 - 1

    # Binomial standard error
    p0_std = np.sqrt(p0 * (1 - p0) / total)
    std = 2 * p0_std

    depth = compiled.depth()

    return expectation, std, n0, depth


def correct_measurement(measured: float, degradation_factor: float) -> float:
    """
    Correct measured value for circuit degradation.

    For Hadamard test observables:
        O_meas = f × O_ideal + (1-f) × O_noise

    For random outcomes O_noise = 0, so:
        O_meas = f × O_ideal
        O_corrected = O_meas / f

    Args:
        measured: Measured value
        degradation_factor: Estimated degradation factor f

    Returns:
        Corrected value (clipped to [-1, 1])
    """
    if degradation_factor <= 0:
        return measured
    corrected = measured / degradation_factor
    return np.clip(corrected, -1.0, 1.0)


def estimate_degradation_factor(depth: int, base_fidelity: float = 0.995) -> float:
    """
    Estimate degradation factor based on circuit depth.

    For depolarizing noise, fidelity decays exponentially:
        f = base_fidelity^depth

    Args:
        depth: Circuit depth
        base_fidelity: Per-layer fidelity (default 0.995)

    Returns:
        Estimated degradation factor
    """
    return base_fidelity ** depth
