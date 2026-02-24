"""
Main experiment runner for negativity measurements on IBM Quantum hardware.

This module provides the NegativityExperiment class that handles:
- Backend initialization (real IBM Quantum or fake simulator)
- State preparation and circuit creation
- Job submission and result collection
- Error mitigation via M3
- Result analysis and export
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .config import (
    DEFAULT_SHOTS,
    DEFAULT_OPTIMIZATION_LEVEL,
    DEFAULT_THETA_VALUES,
    BELL_STATES,
    PRODUCT_STATES,
)
from .states import create_state_preparation, get_theoretical_values, create_all_test_states
from .circuits import (
    create_mu2_circuit,
    create_mu3_circuit,
    create_mu4_circuit,
    create_purity_circuit,
    create_moment_circuits,
    extract_moment_from_counts,
)
from .analysis import (
    compute_negativity_newton_girard,
    analyze_results,
)
from .calibration import create_fake_kingston_backend, get_backend_info


class NegativityExperiment:
    """
    Experiment runner for entanglement negativity measurements.
    
    This class manages the complete workflow for measuring partial transpose
    moments and computing negativity on IBM Quantum hardware or simulators.
    
    Example usage:
        # Simulation mode (default)
        exp = NegativityExperiment()
        results = exp.run()
        exp.save_results("my_results.json")
        
        # Real hardware mode
        exp = NegativityExperiment(
            backend_name="ibm_kingston",
            api_key="YOUR_IBM_QUANTUM_API_KEY",
        )
        results = exp.run()
    
    Attributes:
        backend: The quantum backend (AerSimulator or IBM Quantum)
        backend_name: Name of the backend
        shots: Number of measurement shots per circuit
        use_mitigation: Whether to apply M3 error mitigation
        results: Dictionary storing all experiment results
    """
    
    def __init__(
        self,
        backend_name: str = "fake_kingston",
        api_key: Optional[str] = None,
        instance: str = "ibm-q/open/main",
        shots: int = DEFAULT_SHOTS,
        optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
        use_mitigation: bool = True,
    ):
        """
        Initialize the experiment.
        
        Args:
            backend_name: Name of backend to use. Options:
                - "fake_kingston": Simulated Kingston with calibration noise (default)
                - "aer": Ideal AerSimulator without noise
                - Any IBM backend name (e.g., "ibm_kingston", "ibm_brisbane")
            api_key: IBM Quantum API key (required for real hardware)
            instance: IBM Quantum instance (default: "ibm-q/open/main")
            shots: Number of shots per circuit (default: 100000)
            optimization_level: Transpiler optimization level (default: 3)
            use_mitigation: Whether to use M3 error mitigation (default: True)
        """
        self.backend_name = backend_name
        self.shots = shots
        self.optimization_level = optimization_level
        self.use_mitigation = use_mitigation
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
                "backend": backend_name,
                "timestamp": self.timestamp,
                "shots": shots,
                "optimization_level": optimization_level,
                "use_mitigation": use_mitigation,
            },
            "states": {},
        }
        
        print(f"✓ Initialized NegativityExperiment")
        print(f"  Backend: {backend_name}")
        print(f"  Shots: {shots}")
        print(f"  Mitigation: {use_mitigation}")
    
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
                    f"API key required for real hardware backend '{backend_name}'. "
                    "Provide api_key parameter or use backend_name='fake_kingston' for simulation."
                )
            
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                
                self.service = QiskitRuntimeService(
                    channel="ibm_cloud",
                    token=api_key,
                    instance=instance,
                )
                self.backend = self.service.backend(backend_name)
                print(f"  Connected to IBM Quantum backend: {backend_name}")
                
            except ImportError:
                raise ImportError(
                    "qiskit_ibm_runtime is required for real hardware. "
                    "Install with: pip install qiskit-ibm-runtime"
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to IBM Quantum: {e}")
    
    def _init_mitigation(self) -> None:
        """Initialize M3 error mitigation."""
        try:
            import mthree
            self.mitigator = mthree.M3Mitigation(self.backend)
            print(f"  M3 mitigation initialized")
        except ImportError:
            warnings.warn(
                "mthree not installed. Error mitigation disabled. "
                "Install with: pip install mthree"
            )
            self.use_mitigation = False
            self.mitigator = None
    
    def create_circuits_for_state(
        self,
        state_type: str,
        theta: Optional[float] = None,
        include_purity: bool = True,
    ) -> Dict[str, QuantumCircuit]:
        """
        Create all measurement circuits for a given state.
        
        Args:
            state_type: Type of state (e.g., "bell_phi_plus", "param_theta")
            theta: Rotation angle for parameterized states
            include_purity: Whether to include purity circuit
            
        Returns:
            Dictionary of circuits {"mu2": circuit, "mu3": circuit, ...}
        """
        return create_moment_circuits(state_type, theta, include_purity)
    
    def run_circuits(
        self,
        circuits: Dict[str, QuantumCircuit],
        state_name: str,
    ) -> Dict[str, Dict]:
        """
        Execute circuits and collect results.
        
        Args:
            circuits: Dictionary of circuits to run
            state_name: Name for storing results
            
        Returns:
            Dictionary with raw counts and extracted moments
        """
        results = {}
        
        for circuit_name, circuit in circuits.items():
            print(f"    Running {circuit_name}...", end=" ")
            
            # Transpile
            transpiled = transpile(
                circuit,
                self.backend,
                optimization_level=self.optimization_level,
            )
            
            # Run
            job = self.backend.run(transpiled, shots=self.shots)
            counts = job.result().get_counts()
            
            # Apply mitigation if available
            if self.mitigator is not None:
                try:
                    # Get qubits used in circuit
                    qubits = list(range(circuit.num_qubits))
                    self.mitigator.cals_from_system(qubits)
                    quasi_probs = self.mitigator.apply_correction(counts, qubits)
                    # Convert back to counts-like format
                    counts_mitigated = {
                        k: int(v * self.shots) 
                        for k, v in quasi_probs.nearest_probability_distribution().items()
                    }
                    counts = counts_mitigated
                except Exception as e:
                    warnings.warn(f"Mitigation failed: {e}. Using raw counts.")
            
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
        Run complete measurement for a single state.
        
        Args:
            state_type: Type of state to measure
            theta: Rotation angle for parameterized states
            include_purity: Whether to measure purity
            
        Returns:
            Dictionary with all results for this state
        """
        # Get state name
        if state_type == "param_theta" and theta is not None:
            state_name = f"param_theta_{theta:.3f}"
        else:
            state_name = state_type
        
        print(f"\n  Measuring state: {state_name}")
        
        # Get theoretical values
        theoretical = get_theoretical_values(state_type, theta)
        
        # Create and run circuits
        circuits = self.create_circuits_for_state(state_type, theta, include_purity)
        raw_results = self.run_circuits(circuits, state_name)
        
        # Extract moments
        moments = {
            "mu_2": raw_results["mu2"]["value"],
            "mu_2_std": raw_results["mu2"]["std"],
            "mu_3": raw_results["mu3"]["value"],
            "mu_3_std": raw_results["mu3"]["std"],
            "mu_4": raw_results["mu4"]["value"],
            "mu_4_std": raw_results["mu4"]["std"],
        }
        
        if include_purity and "purity" in raw_results:
            moments["purity"] = raw_results["purity"]["value"]
            moments["purity_std"] = raw_results["purity"]["std"]
        
        # Compute negativity (raw, before ML correction)
        negativity_raw = compute_negativity_newton_girard(
            moments["mu_2"],
            moments["mu_3"],
            moments["mu_4"],
        )
        
        # Analyze results
        analysis = analyze_results(moments, theoretical)
        
        # Store results
        # Note: "negativity" will be updated to ML-corrected value after apply_maxlik_correction()
        result = {
            "state_type": state_type,
            "theta": theta,
            "theoretical": theoretical,
            "measured_moments": moments,
            "negativity_raw": negativity_raw,  # Raw (uncorrected)
            "negativity": negativity_raw,       # Will be updated to ML-corrected
            "analysis": analysis,
            "raw_results": {k: {"value": v["value"], "std": v["std"]} 
                          for k, v in raw_results.items()},
        }
        
        self.results["states"][state_name] = result
        
        # Print summary
        print(f"    Negativity (raw): {negativity_raw:.4f} (theory: {theoretical['negativity']:.4f})")
        
        return result
    
    def run(
        self,
        states: Optional[List[str]] = None,
        theta_values: Optional[List[float]] = None,
        include_purity: bool = True,
    ) -> Dict:
        """
        Run complete experiment for multiple states.
        
        Args:
            states: List of state types to measure. If None, measures:
                    - All Bell states
                    - All product states (00, 01, 10, 11)
                    - Parameterized states at theta_values
            theta_values: List of theta values for parameterized states.
                         If None, uses DEFAULT_THETA_VALUES.
            include_purity: Whether to measure purity for each state
            
        Returns:
            Dictionary with all results
        """
        if theta_values is None:
            theta_values = DEFAULT_THETA_VALUES
        
        if states is None:
            states = BELL_STATES + PRODUCT_STATES + ["param_theta"]
        
        print(f"\n{'='*60}")
        print(f"NEGATIVITY MEASUREMENT EXPERIMENT")
        print(f"{'='*60}")
        print(f"Backend: {self.backend_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"States to measure: {len(states)}")
        
        # Run each state
        for state_type in states:
            if state_type == "param_theta":
                # Run parameterized states
                for theta in theta_values:
                    self.run_state(state_type, theta, include_purity)
            else:
                self.run_state(state_type, None, include_purity)
        
        # Add summary (raw errors)
        self._compute_summary()
        
        # Apply MaxLik degradation correction
        self.apply_maxlik_correction()
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        
        return self.results
    
    def _compute_summary(self) -> None:
        """Compute summary statistics for the experiment."""
        states = self.results["states"]
        
        if not states:
            return
        
        # Collect raw errors (before ML correction)
        raw_negativity_errors = []
        for name, data in states.items():
            if "theoretical" in data and "negativity_raw" in data:
                theory = data["theoretical"]["negativity"]
                measured = data["negativity_raw"]
                raw_negativity_errors.append(abs(theory - measured))
        
        self.results["summary"] = {
            "num_states": len(states),
            "mean_raw_negativity_error": np.mean(raw_negativity_errors) if raw_negativity_errors else None,
            "max_raw_negativity_error": np.max(raw_negativity_errors) if raw_negativity_errors else None,
            "states_measured": list(states.keys()),
        }
    
    def apply_maxlik_correction(self) -> Dict:
        """
        Apply MaxLik degradation correction to all measured states.
        
        This method:
        1. Fits degradation factors (f2, f3, f4) using separable state as calibration
        2. Corrects all measured moments
        3. Recomputes negativity from corrected moments
        
        Returns:
            Dictionary with MaxLik analysis results
        """
        from .maxlik import MaxLikEstimator, run_maxlik_analysis
        
        print(f"\n{'='*60}")
        print(f"APPLYING MAXLIK DEGRADATION CORRECTION")
        print(f"{'='*60}")
        
        # Run MaxLik analysis
        ml_results = run_maxlik_analysis(self.results, verbose=False)
        
        # Store degradation factors
        self.results["degradation_factors"] = ml_results["degradation_factors"]
        
        # Update each state with ML-corrected negativity
        for state_name, ml_data in ml_results["states"].items():
            if state_name in self.results["states"]:
                self.results["states"][state_name]["negativity"] = ml_data["negativity_maxlik"]
                self.results["states"][state_name]["negativity_ml_error"] = ml_data["error_maxlik"]
        
        # Update summary with ML-corrected errors
        ml_errors = [d["error_maxlik"] for d in ml_results["states"].values()]
        self.results["summary"]["mean_ml_negativity_error"] = np.mean(ml_errors)
        self.results["summary"]["max_ml_negativity_error"] = np.max(ml_errors)
        self.results["summary"]["ml_improvement_percent"] = ml_results["summary"]["improvement_percent"]
        
        # Print degradation factors
        df = ml_results["degradation_factors"]
        print(f"\nDegradation factors:")
        print(f"  f₂ = {df['f2']:.6f}")
        print(f"  f₃ = {df['f3']:.6f}")
        print(f"  f₄ = {df['f4']:.6f}")
        
        return ml_results
    
    def save_results(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save results to JSON file.
        
        Args:
            filepath: Output path. If None, auto-generates based on timestamp.
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = f"negativity_{self.backend_name}_{self.timestamp}.json"
        
        filepath = Path(filepath)
        
        # Convert numpy types for JSON serialization
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
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Check if ML correction was applied
        has_ml = "degradation_factors" in self.results
        
        if has_ml:
            print(f"{'State':<20} {'N_theory':>10} {'N_raw':>10} {'N_ML':>10} {'Error_ML':>10}")
        else:
            print(f"{'State':<20} {'N_theory':>10} {'N_measured':>12} {'Error':>10}")
        print(f"{'-'*80}")
        
        for name, data in self.results["states"].items():
            theory = data["theoretical"]["negativity"]
            raw = data.get("negativity_raw", data["negativity"])
            
            if has_ml:
                ml = data["negativity"]
                error = abs(theory - ml)
                print(f"{name:<20} {theory:>10.4f} {raw:>10.4f} {ml:>10.4f} {error:>10.4f}")
            else:
                error = abs(theory - raw)
                print(f"{name:<20} {theory:>10.4f} {raw:>12.4f} {error:>10.4f}")
        
        if "summary" in self.results:
            summary = self.results["summary"]
            print(f"{'-'*80}")
            
            if has_ml:
                print(f"Mean error (raw):  {summary.get('mean_raw_negativity_error', 0):.4f}")
                print(f"Mean error (ML):   {summary.get('mean_ml_negativity_error', 0):.4f}")
                print(f"ML improvement:    {summary.get('ml_improvement_percent', 0):+.1f}%")
            else:
                print(f"Mean error: {summary.get('mean_raw_negativity_error', 0):.4f}")
