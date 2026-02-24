"""
Experiment runner for qubit-qutrit (2×3) negativity measurements.

This module provides the QubitQutritExperiment class for measuring
partial transpose moments and computing negativity in 2×3 dimensional systems.

Key differences from qubit-qubit (2×2):
- 6 eigenvalues → need μ₂ through μ₆
- Qutrit encoded in 2 qubits
- Larger circuits (up to 19 qubits for μ₆)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .config import DEFAULT_SHOTS, DEFAULT_OPTIMIZATION_LEVEL
from .states_qubit_qutrit import (
    create_qubit_qutrit_state_preparation,
    get_qubit_qutrit_theoretical_values,
    QUBIT_QUTRIT_PRODUCT_STATES,
    QUBIT_QUTRIT_ENTANGLED_STATES,
)
from .circuits_qubit_qutrit import (
    create_qubit_qutrit_moment_circuits,
    QUBIT_QUTRIT_CIRCUIT_QUBITS,
)
from .analysis_qubit_qutrit import (
    compute_negativity_qubit_qutrit,
    analyze_results_qubit_qutrit,
)
from .calibration import create_fake_kingston_backend
from .circuits import extract_moment_from_counts


# Default theta values for parameterized states
DEFAULT_THETA_VALUES_23 = [
    0.0,           # Separable
    np.pi / 6,     # 30°
    np.pi / 4,     # 45°
    np.pi / 3,     # 60°
    np.pi / 2,     # 90° (maximally entangled)
]


class QubitQutritExperiment:
    """
    Experiment runner for qubit-qutrit (2×3) negativity measurements.
    
    Example usage:
        exp = QubitQutritExperiment()
        results = exp.run()
        exp.print_summary()
    
    Note:
        Circuits for higher moments (μ₅, μ₆) require many qubits (16-19).
        Use max_moment parameter to limit circuit size if needed.
    """
    
    def __init__(
        self,
        backend_name: str = "aer",
        api_key: Optional[str] = None,
        instance: str = "ibm-q/open/main",
        shots: int = DEFAULT_SHOTS,
        optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
        use_mitigation: bool = False,
        max_moment: int = 6,
    ):
        """
        Initialize the experiment.
        
        Args:
            backend_name: Backend to use ("aer", "fake_kingston", or IBM backend)
            api_key: IBM Quantum API key (required for real hardware)
            instance: IBM Quantum instance
            shots: Number of shots per circuit
            optimization_level: Transpiler optimization level
            use_mitigation: Whether to use error mitigation
            max_moment: Maximum moment to compute (2-6). Lower values use fewer qubits.
        """
        self.backend_name = backend_name
        self.shots = shots
        self.optimization_level = optimization_level
        self.use_mitigation = use_mitigation
        self.max_moment = min(max_moment, 6)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize backend
        self._init_backend(backend_name, api_key, instance)
        
        # Initialize mitigation if requested
        self.mitigator = None
        if use_mitigation and self.backend is not None:
            self._init_mitigation()
        
        # Results storage
        self.results = {
            "metadata": {
                "system": "qubit-qutrit (2x3)",
                "backend": backend_name,
                "timestamp": self.timestamp,
                "shots": shots,
                "optimization_level": optimization_level,
                "use_mitigation": use_mitigation,
                "max_moment": self.max_moment,
            },
            "states": {},
        }
        
        print(f"✓ Initialized QubitQutritExperiment")
        print(f"  System: 2×3 (qubit-qutrit)")
        print(f"  Backend: {backend_name}")
        print(f"  Shots: {shots}")
        print(f"  Max moment: μ{self.max_moment}")
        print(f"  Largest circuit: {QUBIT_QUTRIT_CIRCUIT_QUBITS[f'mu{self.max_moment}']} qubits")
    
    def _init_backend(
        self,
        backend_name: str,
        api_key: Optional[str],
        instance: str,
    ) -> None:
        """Initialize the quantum backend."""
        
        if backend_name == "fake_kingston":
            self.backend = create_fake_kingston_backend()
            self.service = None
            print(f"  Using fake Kingston backend with calibration noise")
            
        elif backend_name == "aer":
            self.backend = AerSimulator()
            self.service = None
            print(f"  Using ideal AerSimulator (no noise)")
            
        else:
            # Real IBM Quantum backend
            if api_key is None:
                raise ValueError(
                    f"API key required for backend '{backend_name}'. "
                    "Use backend_name='aer' for simulation."
                )
            
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                # Use saved credentials or create new service
                # For open plan, don't specify instance
                try:
                    self.service = QiskitRuntimeService()
                except Exception:
                    self.service = QiskitRuntimeService(
                        channel="ibm_quantum_platform",
                        token=api_key,
                    )
                self.backend = self.service.backend(backend_name)
                print(f"  Connected to IBM Quantum backend: {backend_name}")
                
            except ImportError:
                raise ImportError("qiskit_ibm_runtime required for real hardware")
    
    def _init_mitigation(self) -> None:
        """Initialize M3 error mitigation."""
        try:
            import mthree
            self.mitigator = mthree.M3Mitigation(self.backend)
            print(f"  M3 mitigation initialized")
        except ImportError:
            warnings.warn("mthree not installed. Mitigation disabled.")
            self.use_mitigation = False
            self.mitigator = None
    
    def run_circuits(
        self,
        circuits: Dict[str, QuantumCircuit],
        state_name: str,
    ) -> Dict[str, Dict]:
        """Execute circuits and collect results."""
        results = {}

        for circuit_name, circuit in circuits.items():
            print(f"    Running {circuit_name} ({circuit.num_qubits} qubits)...", end=" ")

            # Transpile
            transpiled = transpile(
                circuit,
                self.backend,
                optimization_level=self.optimization_level,
            )

            # Run using Sampler primitive (new IBM Quantum API)
            if hasattr(self, 'service') and self.service is not None:
                # Use SamplerV2 for IBM Quantum hardware
                from qiskit_ibm_runtime import SamplerV2 as Sampler
                sampler = Sampler(mode=self.backend)
                job = sampler.run([transpiled], shots=self.shots)
                result = job.result()
                # Extract counts from SamplerV2 result
                pub_result = result[0]
                data_bin = pub_result.data
                # Try different ways to get counts based on classical register name
                counts = None
                for attr_name in dir(data_bin):
                    if attr_name.startswith('_'):
                        continue
                    attr = getattr(data_bin, attr_name, None)
                    if attr is not None and hasattr(attr, 'get_counts'):
                        counts = attr.get_counts()
                        break
                if counts is None:
                    raise RuntimeError(f"Cannot extract counts from SamplerV2 result")
            else:
                # Aer simulator - use direct run
                job = self.backend.run(transpiled, shots=self.shots)
                counts = job.result().get_counts()

            # Apply mitigation if available
            if self.mitigator is not None:
                try:
                    qubits = list(range(circuit.num_qubits))
                    self.mitigator.cals_from_system(qubits)
                    quasi_probs = self.mitigator.apply_correction(counts, qubits)
                    counts = {
                        k: int(v * self.shots)
                        for k, v in quasi_probs.nearest_probability_distribution().items()
                    }
                except Exception as e:
                    warnings.warn(f"Mitigation failed: {e}")

            # Extract moment value
            moment, moment_std = extract_moment_from_counts(counts, self.shots)

            results[circuit_name] = {
                "counts": counts,
                "value": moment,
                "std": moment_std,
            }

            print(f"value = {moment:.4f} ± {moment_std:.4f}")
        
        return results
    
    def run_state(
        self,
        state_type: str,
        theta: Optional[float] = None,
        include_purity: bool = True,
    ) -> Dict:
        """
        Run complete measurement for a single qubit-qutrit state.
        """
        # Get state name
        if state_type in ["param_theta", "param_theta_12"] and theta is not None:
            state_name = f"{state_type}_{theta:.3f}"
        else:
            state_name = state_type
        
        print(f"\n  Measuring state: {state_name}")
        
        # Get theoretical values
        theoretical = get_qubit_qutrit_theoretical_values(state_type, theta)
        
        # Create and run circuits
        circuits = create_qubit_qutrit_moment_circuits(
            state_type, theta, include_purity, self.max_moment
        )
        raw_results = self.run_circuits(circuits, state_name)
        
        # Extract moments
        moments = {}
        for i in range(2, self.max_moment + 1):
            key = f"mu{i}"
            if key in raw_results:
                moments[f"mu_{i}"] = raw_results[key]["value"]
                moments[f"mu_{i}_std"] = raw_results[key]["std"]
        
        if include_purity and "purity" in raw_results:
            moments["purity"] = raw_results["purity"]["value"]
            moments["purity_std"] = raw_results["purity"]["std"]
        
        # Compute negativity (need all moments up to max_moment)
        if self.max_moment >= 6:
            negativity_raw = compute_negativity_qubit_qutrit(
                moments["mu_2"], moments["mu_3"], moments["mu_4"],
                moments["mu_5"], moments["mu_6"],
            )
        else:
            # Cannot compute full negativity without μ₆
            negativity_raw = None
            warnings.warn(f"Cannot compute negativity without μ₆ (max_moment={self.max_moment})")
        
        # Analyze results
        if negativity_raw is not None:
            analysis = analyze_results_qubit_qutrit(moments, theoretical)
        else:
            analysis = {"note": "Full reconstruction requires max_moment=6"}
        
        # Store results
        result = {
            "state_type": state_type,
            "theta": theta,
            "theoretical": theoretical,
            "measured_moments": moments,
            "negativity_raw": negativity_raw,
            "negativity": negativity_raw,  # Will be updated by MaxLik
            "analysis": analysis,
            "raw_results": {k: {"value": v["value"], "std": v["std"]} 
                          for k, v in raw_results.items()},
        }
        
        self.results["states"][state_name] = result
        
        # Print summary
        if negativity_raw is not None:
            print(f"    Negativity (raw): {negativity_raw:.4f} (theory: {theoretical['negativity']:.4f})")
        
        return result
    
    def run(
        self,
        states: Optional[List[str]] = None,
        theta_values: Optional[List[float]] = None,
        include_purity: bool = True,
    ) -> Dict:
        """
        Run complete experiment for multiple qubit-qutrit states.
        
        Args:
            states: List of state types. If None, measures all supported states.
            theta_values: List of theta values for parameterized states.
            include_purity: Whether to measure purity.
            
        Returns:
            Dictionary with all results
        """
        if theta_values is None:
            theta_values = DEFAULT_THETA_VALUES_23
        
        if states is None:
            # Default: product states + both parameterized families
            states = QUBIT_QUTRIT_PRODUCT_STATES + ["param_theta", "param_theta_12"]
        
        print(f"\n{'='*60}")
        print(f"QUBIT-QUTRIT (2×3) NEGATIVITY EXPERIMENT")
        print(f"{'='*60}")
        print(f"Backend: {self.backend_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"States to measure: {len(states)}")
        
        # Run each state
        for state_type in states:
            if state_type in ["param_theta", "param_theta_12"]:
                for theta in theta_values:
                    self.run_state(state_type, theta, include_purity)
            else:
                self.run_state(state_type, None, include_purity)
        
        # Compute summary
        self._compute_summary()
        
        # Apply MaxLik correction if we have full moments
        if self.max_moment >= 6:
            self._apply_maxlik_correction()
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        
        return self.results
    
    def _compute_summary(self) -> None:
        """Compute summary statistics."""
        states = self.results["states"]
        
        if not states:
            return
        
        raw_errors = []
        for name, data in states.items():
            if data.get("negativity_raw") is not None and "theoretical" in data:
                theory = data["theoretical"]["negativity"]
                measured = data["negativity_raw"]
                raw_errors.append(abs(theory - measured))
        
        self.results["summary"] = {
            "num_states": len(states),
            "mean_raw_negativity_error": np.mean(raw_errors) if raw_errors else None,
            "max_raw_negativity_error": np.max(raw_errors) if raw_errors else None,
            "states_measured": list(states.keys()),
        }
    
    def _apply_maxlik_correction(self) -> None:
        """Apply MaxLik degradation correction for qubit-qutrit.

        Uses physical model-based MaxLik that:
        1. Jointly fits degradation factors (f2, f3, f4, f5, f6)
        2. Fits theta parameter for each state
        3. Constrains negativity to physical values from the state family
        """
        from .maxlik_qubit_qutrit import run_maxlik_analysis_qubit_qutrit

        print(f"\n{'='*60}")
        print(f"APPLYING MAXLIK DEGRADATION CORRECTION (Physical Model)")
        print(f"{'='*60}")

        try:
            ml_results = run_maxlik_analysis_qubit_qutrit(self.results, verbose=True)
        except Exception as e:
            print(f"  MaxLik failed: {e}")
            print("  Falling back to exact calibration...")
            self._apply_exact_calibration()
            return

        # Store degradation factors
        self.results["degradation_factors"] = ml_results["degradation_factors"]

        # Update each state with ML-corrected negativity
        for state_name, ml_data in ml_results["states"].items():
            if state_name in self.results["states"]:
                # Use negativity from fitted theta (guaranteed physical)
                self.results["states"][state_name]["negativity"] = ml_data["negativity_maxlik"]
                self.results["states"][state_name]["negativity_ml_error"] = ml_data["error_maxlik"]
                self.results["states"][state_name]["theta_fit"] = ml_data["theta_fit"]

        # Update summary
        ml_errors = [d["error_maxlik"] for d in ml_results["states"].values()]
        self.results["summary"]["mean_ml_negativity_error"] = np.mean(ml_errors)
        self.results["summary"]["max_ml_negativity_error"] = np.max(ml_errors)

        raw_mean = self.results["summary"].get("mean_raw_negativity_error", 0)
        if raw_mean > 0:
            improvement = 100 * (raw_mean - np.mean(ml_errors)) / raw_mean
            self.results["summary"]["ml_improvement_percent"] = improvement

    def _apply_exact_calibration(self) -> None:
        """Fallback: Apply exact calibration using separable state."""
        # Find separable state for calibration
        calibration_state = None
        for name, data in self.results["states"].items():
            if "product" in name or (data.get("theta") == 0.0):
                calibration_state = data
                break

        if calibration_state is None:
            print("  No separable state found for calibration.")
            return

        # Get calibration factors (theoretical mu_k = 1 for separable)
        f2 = calibration_state["measured_moments"]["mu_2"]
        f3 = calibration_state["measured_moments"]["mu_3"]
        f4 = calibration_state["measured_moments"]["mu_4"]
        f5 = calibration_state["measured_moments"]["mu_5"]
        f6 = calibration_state["measured_moments"]["mu_6"]

        self.results["degradation_factors"] = {
            "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
        }

        print(f"\nExact calibration factors:")
        print(f"  f2 = {f2:.6f}, f3 = {f3:.6f}, f4 = {f4:.6f}")
        print(f"  f5 = {f5:.6f}, f6 = {f6:.6f}")
    
    def save_results(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save results to JSON file."""
        if filepath is None:
            filepath = f"negativity_23_{self.backend_name}_{self.timestamp}.json"
        
        filepath = Path(filepath)
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(v) for v in obj)
            return obj
        
        results_json = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def print_summary(self) -> None:
        """Print a summary of results."""
        print(f"\n{'='*80}")
        print(f"QUBIT-QUTRIT (2×3) RESULTS SUMMARY")
        print(f"{'='*80}")
        
        has_ml = "degradation_factors" in self.results
        
        if has_ml:
            print(f"{'State':<20} {'N_theory':>10} {'N_raw':>10} {'N_ML':>10} {'Error_ML':>10}")
        else:
            print(f"{'State':<20} {'N_theory':>10} {'N_raw':>12} {'Error':>10}")
        print(f"{'-'*80}")
        
        for name, data in self.results["states"].items():
            theory = data["theoretical"]["negativity"]
            raw = data.get("negativity_raw")
            
            if raw is None:
                print(f"{name:<20} {theory:>10.4f} {'N/A':>12}")
                continue
            
            if has_ml:
                ml = data.get("negativity", raw)
                error = abs(theory - ml)
                print(f"{name:<20} {theory:>10.4f} {raw:>10.4f} {ml:>10.4f} {error:>10.4f}")
            else:
                error = abs(theory - raw)
                print(f"{name:<20} {theory:>10.4f} {raw:>12.4f} {error:>10.4f}")
        
        if "summary" in self.results:
            summary = self.results["summary"]
            print(f"{'-'*80}")
            
            if has_ml and summary.get("mean_ml_negativity_error") is not None:
                print(f"Mean error (raw):  {summary.get('mean_raw_negativity_error', 0):.4f}")
                print(f"Mean error (ML):   {summary.get('mean_ml_negativity_error', 0):.4f}")
                if "ml_improvement_percent" in summary:
                    print(f"ML improvement:    {summary['ml_improvement_percent']:+.1f}%")
            elif summary.get("mean_raw_negativity_error") is not None:
                print(f"Mean error: {summary['mean_raw_negativity_error']:.4f}")
