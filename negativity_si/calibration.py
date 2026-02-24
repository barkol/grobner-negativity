"""
Calibration utilities for creating fake backends from real device data.

Creates a fake backend using Qiskit's GenericBackendV2 with noise model
derived from IBM Kingston calibration data to simulate realistic quantum
hardware behavior without requiring access to real devices.

The FakeGeneric approach uses GenericBackendV2 which provides:
- Configurable coupling map from calibration
- Custom noise model based on measured error rates
- Realistic gate times and coherence properties
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.transpiler import CouplingMap

from .config import CALIBRATION_FILE


@dataclass
class QubitCalibration:
    """Calibration data for a single qubit."""
    qubit_id: int
    t1_us: float
    t2_us: float
    readout_error: float
    prob_meas0_prep1: float
    prob_meas1_prep0: float
    id_error: float
    sx_error: float
    x_error: float
    operational: bool
    
    # Coupling information (parsed from calibration)
    connected_qubits: List[int] = None
    cz_errors: Dict[int, float] = None


def parse_calibration_csv(
    filepath: Optional[Path] = None,
) -> Tuple[Dict[int, QubitCalibration], List[Tuple[int, int]]]:
    """
    Parse IBM Quantum calibration CSV file.
    
    Args:
        filepath: Path to calibration CSV. If None, uses default Kingston calibration.
        
    Returns:
        Tuple of (qubit_calibrations, coupling_map)
        - qubit_calibrations: Dict mapping qubit ID to QubitCalibration
        - coupling_map: List of (qubit1, qubit2) tuples for connectivity
    """
    if filepath is None:
        filepath = CALIBRATION_FILE
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Calibration file not found: {filepath}")
    
    calibrations = {}
    coupling_edges = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            qubit_id = int(row['Qubit'])
            
            # Parse basic calibration values (handle empty values)
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val else default
                except (ValueError, TypeError):
                    return default
            
            t1 = safe_float(row.get('T1 (us)', ''))
            t2 = safe_float(row.get('T2 (us)', ''))
            
            # Handle readout errors
            readout_error = safe_float(row.get('Readout assignment error', ''))
            prob_m0_p1 = safe_float(row.get('Prob meas0 prep1', ''))
            prob_m1_p0 = safe_float(row.get('Prob meas1 prep0', ''))
            
            # Gate errors
            id_error = safe_float(row.get('ID error', ''))
            sx_error = safe_float(row.get('√x (sx) error', ''))
            x_error = safe_float(row.get('Pauli-X error', ''))
            
            # Operational status
            operational = row.get('Operational', 'Yes').lower() == 'yes'
            
            # Parse connectivity and CZ errors
            # Format: "target1:gate_length;target2:gate_length"
            connected = []
            cz_errors = {}
            
            # Parse gate length field for connectivity
            gate_length_str = row.get('Gate length (ns)', '')
            if gate_length_str and ':' in gate_length_str:
                for part in gate_length_str.split(';'):
                    if ':' in part:
                        target_str = part.split(':')[0]
                        try:
                            target = int(target_str)
                            connected.append(target)
                            coupling_edges.add((min(qubit_id, target), max(qubit_id, target)))
                        except ValueError:
                            pass
            
            # Parse CZ errors
            cz_error_str = row.get('CZ error', '')
            if cz_error_str and ':' in cz_error_str:
                for part in cz_error_str.split(';'):
                    if ':' in part:
                        try:
                            target_str, error_str = part.split(':')
                            target = int(target_str)
                            cz_errors[target] = float(error_str)
                        except (ValueError, IndexError):
                            pass
            
            calibrations[qubit_id] = QubitCalibration(
                qubit_id=qubit_id,
                t1_us=t1 if t1 > 0 else 100.0,  # Default if missing
                t2_us=t2 if t2 > 0 else 50.0,
                readout_error=readout_error,
                prob_meas0_prep1=prob_m0_p1,
                prob_meas1_prep0=prob_m1_p0,
                id_error=id_error,
                sx_error=sx_error,
                x_error=x_error,
                operational=operational,
                connected_qubits=connected,
                cz_errors=cz_errors,
            )
    
    # Convert coupling edges to list
    coupling_map = list(coupling_edges)
    # Add reverse edges for bidirectional coupling
    coupling_map_full = []
    for q1, q2 in coupling_map:
        coupling_map_full.append([q1, q2])
        coupling_map_full.append([q2, q1])
    
    return calibrations, coupling_map_full


def create_noise_model_from_calibration(
    calibrations: Dict[int, QubitCalibration],
    coupling_map: List[List[int]],
    gate_time_1q: float = 32e-9,  # 32 ns single-qubit gate
    gate_time_2q: float = 68e-9,  # 68 ns two-qubit gate
) -> NoiseModel:
    """
    Create a Qiskit noise model from calibration data.
    
    Args:
        calibrations: Dictionary of qubit calibrations
        coupling_map: Coupling map as list of [q1, q2] pairs
        gate_time_1q: Single-qubit gate duration in seconds
        gate_time_2q: Two-qubit gate duration in seconds
        
    Returns:
        NoiseModel configured with calibration data
    """
    noise_model = NoiseModel()
    
    # Single-qubit errors
    for qubit_id, cal in calibrations.items():
        if not cal.operational:
            continue
            
        # Thermal relaxation for single-qubit gates
        if cal.t1_us > 0 and cal.t2_us > 0:
            t1 = cal.t1_us * 1e-6  # Convert to seconds
            t2 = min(cal.t2_us * 1e-6, 2 * t1)  # T2 ≤ 2*T1
            
            thermal_error = thermal_relaxation_error(t1, t2, gate_time_1q)
            
            # Add to all single-qubit gates
            for gate in ['id', 'rz', 'sx', 'x']:
                noise_model.add_quantum_error(thermal_error, gate, [qubit_id])
        
        # Depolarizing error based on calibration
        if cal.sx_error > 0:
            dep_error = depolarizing_error(cal.sx_error, 1)
            noise_model.add_quantum_error(dep_error, 'sx', [qubit_id])
        
        if cal.x_error > 0:
            dep_error = depolarizing_error(cal.x_error, 1)
            noise_model.add_quantum_error(dep_error, 'x', [qubit_id])
        
        # Readout error
        if cal.prob_meas0_prep1 > 0 or cal.prob_meas1_prep0 > 0:
            # Readout error probabilities
            p0_given_1 = cal.prob_meas0_prep1
            p1_given_0 = cal.prob_meas1_prep0
            
            readout_error = [
                [1 - p1_given_0, p1_given_0],  # P(measured|prepared=0)
                [p0_given_1, 1 - p0_given_1],  # P(measured|prepared=1)
            ]
            noise_model.add_readout_error(readout_error, [qubit_id])
    
    # Two-qubit errors (CZ gates)
    for q1, q2 in coupling_map:
        if q1 >= q2:  # Avoid duplicates
            continue
            
        cal1 = calibrations.get(q1)
        cal2 = calibrations.get(q2)
        
        if cal1 is None or cal2 is None:
            continue
        if not cal1.operational or not cal2.operational:
            continue
        
        # Get CZ error from calibration
        cz_error = cal1.cz_errors.get(q2, 0.0) if cal1.cz_errors else 0.0
        if cz_error == 0.0 and cal2.cz_errors:
            cz_error = cal2.cz_errors.get(q1, 0.0)
        
        if cz_error > 0:
            # Two-qubit depolarizing error
            dep_error_2q = depolarizing_error(cz_error, 2)
            noise_model.add_quantum_error(dep_error_2q, 'cz', [q1, q2])
            noise_model.add_quantum_error(dep_error_2q, 'cz', [q2, q1])
            
            # Also add to ecr and cx (common two-qubit gates)
            noise_model.add_quantum_error(dep_error_2q, 'ecr', [q1, q2])
            noise_model.add_quantum_error(dep_error_2q, 'ecr', [q2, q1])
            noise_model.add_quantum_error(dep_error_2q, 'cx', [q1, q2])
            noise_model.add_quantum_error(dep_error_2q, 'cx', [q2, q1])
        
        # Thermal relaxation for two-qubit gates
        t1_1 = cal1.t1_us * 1e-6 if cal1.t1_us > 0 else 100e-6
        t2_1 = min(cal1.t2_us * 1e-6, 2 * t1_1) if cal1.t2_us > 0 else 50e-6
        t1_2 = cal2.t1_us * 1e-6 if cal2.t1_us > 0 else 100e-6
        t2_2 = min(cal2.t2_us * 1e-6, 2 * t1_2) if cal2.t2_us > 0 else 50e-6
        
        thermal_1 = thermal_relaxation_error(t1_1, t2_1, gate_time_2q)
        thermal_2 = thermal_relaxation_error(t1_2, t2_2, gate_time_2q)
        thermal_2q = thermal_1.tensor(thermal_2)
        
        for gate in ['cz', 'ecr', 'cx']:
            noise_model.add_quantum_error(thermal_2q, gate, [q1, q2])
            noise_model.add_quantum_error(thermal_2q, gate, [q2, q1])
    
    return noise_model


def select_best_qubits(
    calibrations: Dict[int, QubitCalibration],
    coupling_map: List[List[int]],
    num_qubits: int = 10,
) -> List[int]:
    """
    Select the best qubits based on calibration quality.
    
    Selection criteria:
    1. Operational status
    2. Low readout error
    3. Low gate errors
    4. Long coherence times
    5. Connectivity (prefer connected subgraphs)
    
    Args:
        calibrations: Dictionary of qubit calibrations
        coupling_map: Coupling map
        num_qubits: Number of qubits to select
        
    Returns:
        List of selected qubit IDs
    """
    # Score each qubit
    scores = {}
    for qubit_id, cal in calibrations.items():
        if not cal.operational:
            continue
        if cal.t1_us <= 0 or cal.t2_us <= 0:
            continue
            
        # Combine metrics (lower is better for errors, higher for coherence)
        error_score = (
            cal.readout_error * 10 +  # Weight readout error heavily
            cal.sx_error * 100 +
            cal.x_error * 100
        )
        
        coherence_score = 1.0 / (cal.t1_us + cal.t2_us + 1)
        
        # Combined score (lower is better)
        scores[qubit_id] = error_score + coherence_score
    
    # Sort by score and select best
    sorted_qubits = sorted(scores.keys(), key=lambda q: scores[q])
    
    # Try to select connected qubits
    if len(sorted_qubits) <= num_qubits:
        return sorted_qubits
    
    # Build adjacency for connectivity check
    adjacency = {q: set() for q in sorted_qubits}
    for q1, q2 in coupling_map:
        if q1 in adjacency and q2 in adjacency:
            adjacency[q1].add(q2)
            adjacency[q2].add(q1)
    
    # Greedy selection maintaining connectivity
    selected = [sorted_qubits[0]]
    for qubit in sorted_qubits[1:]:
        if len(selected) >= num_qubits:
            break
        # Check if connected to any selected qubit
        if any(qubit in adjacency.get(s, set()) for s in selected):
            selected.append(qubit)
    
    # If not enough, add remaining best qubits
    for qubit in sorted_qubits:
        if len(selected) >= num_qubits:
            break
        if qubit not in selected:
            selected.append(qubit)
    
    return selected[:num_qubits]


def create_fake_kingston_backend(
    calibration_file: Optional[Path] = None,
    num_qubits: int = 20,
) -> AerSimulator:
    """
    Create a fake IBM Kingston backend using GenericBackendV2 with calibration data.
    
    This uses Qiskit's GenericBackendV2 (the "FakeGeneric" approach) combined
    with a noise model derived from real Kingston calibration data, allowing
    realistic simulation without access to the actual hardware.
    
    Args:
        calibration_file: Path to calibration CSV. If None, uses bundled data.
        num_qubits: Number of qubits to include in fake backend (default: 20)
        
    Returns:
        AerSimulator configured to mimic IBM Kingston with realistic noise
    """
    # Parse calibration data
    calibrations, coupling_map_raw = parse_calibration_csv(calibration_file)
    
    # Select best qubits for the fake backend
    selected_qubits = select_best_qubits(calibrations, coupling_map_raw, num_qubits)
    
    # Create qubit mapping (old_id -> new_id)
    qubit_map = {old: new for new, old in enumerate(selected_qubits)}
    
    # Filter and remap coupling map to selected qubits
    coupling_map_filtered = []
    for q1, q2 in coupling_map_raw:
        if q1 in qubit_map and q2 in qubit_map:
            new_q1, new_q2 = qubit_map[q1], qubit_map[q2]
            coupling_map_filtered.append([new_q1, new_q2])
    
    # Ensure we have a valid coupling map (add linear connectivity if needed)
    if len(coupling_map_filtered) < num_qubits - 1:
        # Add linear chain as fallback
        for i in range(num_qubits - 1):
            if [i, i+1] not in coupling_map_filtered:
                coupling_map_filtered.append([i, i+1])
                coupling_map_filtered.append([i+1, i])
    
    # Create GenericBackendV2 (FakeGeneric)
    coupling = CouplingMap(coupling_map_filtered)
    
    fake_backend = GenericBackendV2(
        num_qubits=num_qubits,
        coupling_map=coupling,
        basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'measure', 'reset'],
    )
    
    # Create noise model from calibration
    # Remap calibrations to new qubit indices
    remapped_calibrations = {}
    for old_id, cal in calibrations.items():
        if old_id in qubit_map:
            new_id = qubit_map[old_id]
            # Create new calibration with remapped CZ errors
            new_cz_errors = {}
            if cal.cz_errors:
                for target, error in cal.cz_errors.items():
                    if target in qubit_map:
                        new_cz_errors[qubit_map[target]] = error
            
            remapped_calibrations[new_id] = QubitCalibration(
                qubit_id=new_id,
                t1_us=cal.t1_us,
                t2_us=cal.t2_us,
                readout_error=cal.readout_error,
                prob_meas0_prep1=cal.prob_meas0_prep1,
                prob_meas1_prep0=cal.prob_meas1_prep0,
                id_error=cal.id_error,
                sx_error=cal.sx_error,
                x_error=cal.x_error,
                operational=cal.operational,
                connected_qubits=[qubit_map[q] for q in (cal.connected_qubits or []) if q in qubit_map],
                cz_errors=new_cz_errors,
            )
    
    noise_model = create_noise_model_from_calibration(
        remapped_calibrations, 
        coupling_map_filtered
    )
    
    # Create AerSimulator with fake backend properties and noise
    backend = AerSimulator.from_backend(fake_backend)
    backend.set_options(noise_model=noise_model)
    
    # Store metadata for reference
    backend._calibrations = remapped_calibrations
    backend._original_calibrations = calibrations
    backend._num_qubits = num_qubits
    backend._selected_qubits = selected_qubits
    backend._qubit_map = qubit_map
    
    return backend


def get_backend_info(backend: AerSimulator) -> Dict[str, Any]:
    """
    Get information about the fake backend.
    
    Args:
        backend: AerSimulator created by create_fake_kingston_backend
        
    Returns:
        Dictionary with backend information
    """
    info = {
        "name": "fake_kingston",
        "num_qubits": getattr(backend, '_num_qubits', 20),
        "has_noise_model": backend.options.noise_model is not None,
    }
    
    if hasattr(backend, '_selected_qubits'):
        info["selected_qubits"] = backend._selected_qubits
    
    if hasattr(backend, '_qubit_map'):
        info["qubit_mapping"] = backend._qubit_map
    
    if hasattr(backend, '_calibrations'):
        # Summary statistics
        cals = backend._calibrations
        operational = [c for c in cals.values() if c.operational]
        if operational:
            info["operational_qubits"] = len(operational)
            t1_vals = [c.t1_us for c in operational if c.t1_us > 0]
            t2_vals = [c.t2_us for c in operational if c.t2_us > 0]
            ro_vals = [c.readout_error for c in operational if c.readout_error > 0]
            
            if t1_vals:
                info["avg_t1_us"] = float(np.mean(t1_vals))
            if t2_vals:
                info["avg_t2_us"] = float(np.mean(t2_vals))
            if ro_vals:
                info["avg_readout_error"] = float(np.mean(ro_vals))
    
    return info
