"""
Noise model creation from IBM calibration data.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


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
    num_qubits: int = 10,
) -> 'NoiseModel':
    """
    Create noise model from calibration data.

    Uses add_all_qubit_quantum_error to avoid coupling map restrictions,
    allowing circuits of any size to run on the simulator.

    Args:
        calibrations: Dictionary from parse_calibration_csv
        num_qubits: Number of qubits to use for averaging error rates

    Returns:
        Qiskit NoiseModel (no coupling map restrictions)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for noise model creation")

    # Select best qubits by readout error for averaging
    sorted_qubits = sorted(
        [q for q, c in calibrations.items() if c['operational'] and c['t1_us'] > 10],
        key=lambda q: calibrations[q]['readout_error']
    )[:num_qubits]

    if not sorted_qubits:
        raise ValueError("No operational qubits found in calibration data")

    # Compute average error rates from calibration data
    avg_t1 = sum(calibrations[q]['t1_us'] for q in sorted_qubits) / len(sorted_qubits)
    avg_t2 = sum(calibrations[q]['t2_us'] for q in sorted_qubits) / len(sorted_qubits)
    avg_sx_err = sum(calibrations[q]['sx_error'] for q in sorted_qubits if calibrations[q]['sx_error'] > 0) / max(1, sum(1 for q in sorted_qubits if calibrations[q]['sx_error'] > 0))
    avg_x_err = sum(calibrations[q]['x_error'] for q in sorted_qubits if calibrations[q]['x_error'] > 0) / max(1, sum(1 for q in sorted_qubits if calibrations[q]['x_error'] > 0))
    avg_readout = sum(calibrations[q]['readout_error'] for q in sorted_qubits) / len(sorted_qubits)

    # Default values if averages are zero
    avg_sx_err = avg_sx_err if avg_sx_err > 0 else 0.001
    avg_x_err = avg_x_err if avg_x_err > 0 else 0.001
    avg_readout = avg_readout if avg_readout > 0 else 0.01

    noise_model = NoiseModel()
    gate_time_1q = 32e-9  # 32 ns
    gate_time_2q = 68e-9  # 68 ns

    # Single-qubit thermal relaxation (average)
    t1 = avg_t1 * 1e-6
    t2 = min(avg_t2 * 1e-6, 2 * t1)
    thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)

    # Single-qubit depolarizing errors
    dep_sx = depolarizing_error(avg_sx_err, 1)
    dep_x = depolarizing_error(avg_x_err, 1)

    # Apply to all qubits (no coupling map restriction)
    noise_model.add_all_qubit_quantum_error(thermal_1q, ['sx', 'x', 'ry', 'rz', 'h'])
    noise_model.add_all_qubit_quantum_error(dep_sx, 'sx')
    noise_model.add_all_qubit_quantum_error(dep_x, 'x')

    # Two-qubit errors (average)
    avg_cz_err = 0.005  # 0.5% average 2Q error
    thermal_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )
    dep_2q = depolarizing_error(avg_cz_err, 2)

    # Apply to all qubit pairs (no coupling map restriction)
    for gate in ['cx', 'cz', 'swap', 'cswap']:
        if gate == 'cswap':
            # 3-qubit gate
            dep_3q = depolarizing_error(avg_cz_err * 2, 3)  # Higher error for 3Q
            noise_model.add_all_qubit_quantum_error(dep_3q, gate)
        else:
            noise_model.add_all_qubit_quantum_error(thermal_2q, gate)
            noise_model.add_all_qubit_quantum_error(dep_2q, gate)

    # Readout errors - apply uniformly
    from qiskit_aer.noise import ReadoutError
    read_err = ReadoutError([[1 - avg_readout, avg_readout], [avg_readout, 1 - avg_readout]])
    noise_model.add_all_qubit_readout_error(read_err)

    return noise_model
