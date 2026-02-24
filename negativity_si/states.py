"""
Quantum state preparation for negativity measurements.

Implements preparation circuits for:
- Parameterized states: |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
- Bell states: |Φ±⟩, |Ψ±⟩
- Product states: |00⟩, |01⟩, |10⟩, |11⟩
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from qiskit import QuantumCircuit

from .config import DEFAULT_THETA_VALUES, BELL_STATES, PRODUCT_STATES


# All supported state types
SUPPORTED_STATES = (
    BELL_STATES + 
    PRODUCT_STATES + 
    ["param_theta"]  # Parameterized family
)


def create_state_preparation(
    state_type: str,
    theta: Optional[float] = None,
    num_copies: int = 1,
) -> Callable[[QuantumCircuit, List[int]], None]:
    """
    Create a state preparation function for the specified state type.
    
    Args:
        state_type: Type of state to prepare. Options:
            - "bell_phi_plus": |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            - "bell_phi_minus": |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            - "bell_psi_plus": |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            - "bell_psi_minus": |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            - "product_00": |00⟩
            - "product_01": |01⟩
            - "product_10": |10⟩
            - "product_11": |11⟩
            - "param_theta": |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
        theta: Rotation angle for parameterized states (required if state_type="param_theta")
        num_copies: Number of state copies to prepare
        
    Returns:
        Function that applies state preparation to a circuit
    """
    
    def prepare_bell_phi_plus(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
        for i in range(0, len(qubits), 2):
            qc.h(qubits[i])
            qc.cx(qubits[i], qubits[i + 1])
    
    def prepare_bell_phi_minus(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |Φ⁻⟩ = (|00⟩ - |11⟩)/√2"""
        for i in range(0, len(qubits), 2):
            qc.h(qubits[i])
            qc.z(qubits[i])
            qc.cx(qubits[i], qubits[i + 1])
    
    def prepare_bell_psi_plus(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2"""
        for i in range(0, len(qubits), 2):
            qc.h(qubits[i])
            qc.cx(qubits[i], qubits[i + 1])
            qc.x(qubits[i + 1])
    
    def prepare_bell_psi_minus(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2"""
        for i in range(0, len(qubits), 2):
            qc.h(qubits[i])
            qc.z(qubits[i])
            qc.cx(qubits[i], qubits[i + 1])
            qc.x(qubits[i + 1])
    
    def prepare_product_00(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |00⟩ - identity (qubits start in |0⟩)"""
        pass  # Default state
    
    def prepare_product_01(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |01⟩"""
        for i in range(0, len(qubits), 2):
            qc.x(qubits[i + 1])
    
    def prepare_product_10(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |10⟩"""
        for i in range(0, len(qubits), 2):
            qc.x(qubits[i])
    
    def prepare_product_11(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |11⟩"""
        for i in range(0, len(qubits), 2):
            qc.x(qubits[i])
            qc.x(qubits[i + 1])
    
    def prepare_param_theta(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩"""
        if theta is None:
            raise ValueError("theta must be specified for param_theta state")
        for i in range(0, len(qubits), 2):
            qc.ry(theta, qubits[i])
            qc.cx(qubits[i], qubits[i + 1])
    
    # Map state types to preparation functions
    preparation_map = {
        "bell_phi_plus": prepare_bell_phi_plus,
        "bell_phi_minus": prepare_bell_phi_minus,
        "bell_psi_plus": prepare_bell_psi_plus,
        "bell_psi_minus": prepare_bell_psi_minus,
        "product_00": prepare_product_00,
        "product_01": prepare_product_01,
        "product_10": prepare_product_10,
        "product_11": prepare_product_11,
        "param_theta": prepare_param_theta,
    }
    
    if state_type not in preparation_map:
        raise ValueError(
            f"Unknown state type: {state_type}. "
            f"Supported types: {list(preparation_map.keys())}"
        )
    
    return preparation_map[state_type]


def get_theoretical_values(
    state_type: str,
    theta: Optional[float] = None,
) -> Dict[str, float]:
    """
    Get theoretical values for a given state.
    
    Args:
        state_type: Type of quantum state
        theta: Rotation angle for parameterized states
        
    Returns:
        Dictionary with theoretical values:
            - negativity: Entanglement negativity N(ρ)
            - mu_2: Second moment Tr[(ρ^{T_A})²]
            - mu_3: Third moment Tr[(ρ^{T_A})³]
            - mu_4: Fourth moment Tr[(ρ^{T_A})⁴]
            - purity: State purity Tr[ρ²]
    """
    
    if state_type in ["bell_phi_plus", "bell_phi_minus", 
                      "bell_psi_plus", "bell_psi_minus"]:
        # All Bell states have the same negativity = 0.5
        # PT eigenvalues: [-0.5, 0.5, 0.5, 0.5]
        # μₖ = Σᵢ λᵢᵏ
        return {
            "negativity": 0.5,
            "mu_2": 1.0,   # (-0.5)² + 3×(0.5)² = 0.25 + 0.75 = 1.0
            "mu_3": 0.25,  # (-0.5)³ + 3×(0.5)³ = -0.125 + 0.375 = 0.25
            "mu_4": 0.25,  # (-0.5)⁴ + 3×(0.5)⁴ = 0.0625 + 0.1875 = 0.25
            "purity": 1.0,
            "theta": np.pi / 2,
            "theta_degrees": 90.0,
        }
    
    elif state_type in ["product_00", "product_01", "product_10", "product_11"]:
        # All product states are separable with N = 0
        # PT eigenvalues: [0, 0, 0, 1] (pure product state)
        return {
            "negativity": 0.0,
            "mu_2": 1.0,
            "mu_3": 1.0,
            "mu_4": 1.0,
            "purity": 1.0,
            "theta": 0.0,
            "theta_degrees": 0.0,
        }
    
    elif state_type == "param_theta":
        if theta is None:
            raise ValueError("theta must be specified for param_theta state")
        
        # |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
        # PT eigenvalues: [-sin(θ/2)cos(θ/2), sin(θ/2)cos(θ/2), cos²(θ/2), sin²(θ/2)]
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        # PT eigenvalues
        eigs = np.array([-s * c, s * c, c**2, s**2])
        
        # Moments
        mu_2 = np.sum(eigs**2)
        mu_3 = np.sum(eigs**3)
        mu_4 = np.sum(eigs**4)
        
        # Negativity = |negative eigenvalue| = sin(θ/2)cos(θ/2) = sin(θ)/2
        negativity = np.sin(theta) / 2
        
        return {
            "negativity": negativity,
            "mu_2": mu_2,
            "mu_3": mu_3,
            "mu_4": mu_4,
            "purity": 1.0,  # Pure state
            "theta": theta,
            "theta_degrees": np.degrees(theta),
        }
    
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def get_pt_eigenvalues(
    state_type: str,
    theta: Optional[float] = None,
) -> np.ndarray:
    """
    Get partial transpose eigenvalues for a given state.
    
    Args:
        state_type: Type of quantum state
        theta: Rotation angle for parameterized states
        
    Returns:
        Array of 4 eigenvalues of ρ^{T_A}
    """
    
    if state_type in ["bell_phi_plus", "bell_phi_minus",
                      "bell_psi_plus", "bell_psi_minus"]:
        return np.array([-0.5, 0.5, 0.5, 0.5])
    
    elif state_type in ["product_00", "product_01", "product_10", "product_11"]:
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    elif state_type == "param_theta":
        if theta is None:
            raise ValueError("theta must be specified for param_theta state")
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([-s * c, s * c, c**2, s**2])
    
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def create_all_test_states(
    theta_values: Optional[List[float]] = None,
) -> Dict[str, Dict]:
    """
    Create a dictionary of all test states with their configurations.
    
    Args:
        theta_values: List of theta values for parameterized states.
                     If None, uses DEFAULT_THETA_VALUES.
    
    Returns:
        Dictionary mapping state names to their configurations:
            {
                "state_name": {
                    "state_type": str,
                    "theta": Optional[float],
                    "theoretical": Dict[str, float],
                }
            }
    """
    if theta_values is None:
        theta_values = DEFAULT_THETA_VALUES
    
    states = {}
    
    # Bell states
    for bell_state in BELL_STATES:
        states[bell_state] = {
            "state_type": bell_state,
            "theta": None,
            "theoretical": get_theoretical_values(bell_state),
        }
    
    # Product states
    for product_state in PRODUCT_STATES:
        states[product_state] = {
            "state_type": product_state,
            "theta": None,
            "theoretical": get_theoretical_values(product_state),
        }
    
    # Parameterized states
    for theta in theta_values:
        name = f"param_theta_{theta:.3f}"
        states[name] = {
            "state_type": "param_theta",
            "theta": theta,
            "theoretical": get_theoretical_values("param_theta", theta),
        }
    
    return states
