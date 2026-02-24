"""
Quantum state preparation for qubit-qutrit (2×3) systems.

Qutrit encoding:
    |0⟩_qutrit → |00⟩
    |1⟩_qutrit → |01⟩
    |2⟩_qutrit → |10⟩
    (|11⟩ is invalid and must be avoided)

Supported states:
- Parameterized: |ψ(θ)⟩ = cos(θ/2)|0⟩|0⟩ + sin(θ/2)|1⟩|1⟩
- Maximally entangled (Schmidt rank 2)
- Product states
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from qiskit import QuantumCircuit


# Qubits per subsystem
QUBITS_PER_QUBIT = 1   # A subsystem (qubit)
QUBITS_PER_QUTRIT = 2  # B subsystem (qutrit encoded in 2 qubits)
QUBITS_PER_COPY = QUBITS_PER_QUBIT + QUBITS_PER_QUTRIT  # = 3


def create_qubit_qutrit_state_preparation(
    state_type: str,
    theta: Optional[float] = None,
    phi: Optional[float] = None,
) -> Callable[[QuantumCircuit, List[int]], None]:
    """
    Create state preparation function for qubit-qutrit states.
    
    Args:
        state_type: Type of state to prepare:
            - "param_theta": |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
            - "param_theta_12": |ψ(θ)⟩ = cos(θ/2)|0,1⟩ + sin(θ/2)|1,2⟩
            - "param_theta_phi": |ψ(θ,φ)⟩ = cos(θ)|0,0⟩ + sin(θ)cos(φ)|1,1⟩ + sin(θ)sin(φ)|1,2⟩
            - "maximally_entangled": (|0,0⟩ + |1,1⟩)/√2
            - "maximally_entangled_12": (|0,1⟩ + |1,2⟩)/√2
            - "product_00", "product_01", "product_02": |0⟩|j⟩
            - "product_10", "product_11", "product_12": |1⟩|j⟩
        theta: First rotation angle
        phi: Second rotation angle (for param_theta_phi)
        
    Returns:
        Function that applies state preparation to circuit
        
    Note:
        Qubit layout per copy: [A, B0, B1]
        - A: qubit subsystem (1 qubit)
        - B0, B1: qutrit subsystem (2 qubits encoding qutrit)
        
        Qutrit encoding:
        - |0⟩_qutrit → |00⟩
        - |1⟩_qutrit → |01⟩
        - |2⟩_qutrit → |10⟩
    """
    
    def prepare_param_theta(qc: QuantumCircuit, qubits: List[int]) -> None:
        """
        Prepare |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
        
        Qutrit |1⟩ is encoded as |01⟩, so final state is:
        cos(θ/2)|0⟩|00⟩ + sin(θ/2)|1⟩|01⟩
        """
        if theta is None:
            raise ValueError("theta required for param_theta state")
        
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]      # Qubit A
            b0 = qubits[base + 1] # Qutrit B, first qubit
            b1 = qubits[base + 2] # Qutrit B, second qubit
            
            # Create entangled state
            qc.ry(theta, a)
            qc.cx(a, b1)  # |1⟩_A → |01⟩_B (qutrit |1⟩)
    
    def prepare_param_theta_12(qc: QuantumCircuit, qubits: List[int]) -> None:
        """
        Prepare |ψ(θ)⟩ = cos(θ/2)|0,1⟩ + sin(θ/2)|1,2⟩
        
        Qutrit encoding: |1⟩→|01⟩, |2⟩→|10⟩
        Final state: cos(θ/2)|0⟩|01⟩ + sin(θ/2)|1⟩|10⟩
        """
        if theta is None:
            raise ValueError("theta required for param_theta_12 state")
        
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]      # Qubit A
            b0 = qubits[base + 1] # Qutrit B, first qubit
            b1 = qubits[base + 2] # Qutrit B, second qubit
            
            # Create entangled state
            # Start with superposition on A
            qc.ry(theta, a)
            
            # When A=|0⟩: B should be |1⟩ = |01⟩
            # When A=|1⟩: B should be |2⟩ = |10⟩
            
            # Use X gate to flip logic, then controlled operations
            qc.x(a)
            qc.cx(a, b1)  # When A was |0⟩ (now |1⟩): set B to |01⟩
            qc.x(a)
            qc.cx(a, b0)  # When A=|1⟩: set B to |10⟩
    
    def prepare_param_theta_phi(qc: QuantumCircuit, qubits: List[int]) -> None:
        """
        Prepare |ψ(θ,φ)⟩ = cos(θ)|0,0⟩ + sin(θ)cos(φ)|1,1⟩ + sin(θ)sin(φ)|1,2⟩
        
        This is a Schmidt rank 2 state when φ ≠ 0, π/2.
        Qutrit encoding: |1⟩→|01⟩, |2⟩→|10⟩
        """
        if theta is None or phi is None:
            raise ValueError("theta and phi required for param_theta_phi state")
        
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]
            b0 = qubits[base + 1]
            b1 = qubits[base + 2]
            
            # First create superposition on A
            qc.ry(2 * theta, a)
            
            # When A=|1⟩, create superposition on B between |1⟩ and |2⟩
            # |1⟩_qutrit = |01⟩, |2⟩_qutrit = |10⟩
            qc.cry(2 * phi, a, b0)  # Controlled rotation for |2⟩ component
            qc.cx(a, b1)           # Flip b1 when A=1
            qc.ccx(a, b0, b1)      # Correct: if A=1 and b0=1, flip b1 back
    
    def prepare_maximally_entangled(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare (|0,0⟩ + |1,1⟩)/√2"""
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]
            b1 = qubits[base + 2]
            
            qc.h(a)
            qc.cx(a, b1)
    
    def prepare_maximally_entangled_12(qc: QuantumCircuit, qubits: List[int]) -> None:
        """Prepare (|0,1⟩ + |1,2⟩)/√2"""
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]
            b0 = qubits[base + 1]
            b1 = qubits[base + 2]
            
            qc.h(a)
            # When A=|0⟩: B = |1⟩ = |01⟩
            # When A=|1⟩: B = |2⟩ = |10⟩
            qc.x(a)
            qc.cx(a, b1)
            qc.x(a)
            qc.cx(a, b0)
    
    def prepare_product(qc: QuantumCircuit, qubits: List[int], a_val: int, b_val: int) -> None:
        """Prepare product state |a_val⟩|b_val⟩"""
        n_copies = len(qubits) // QUBITS_PER_COPY
        for i in range(n_copies):
            base = i * QUBITS_PER_COPY
            a = qubits[base]
            b0 = qubits[base + 1]
            b1 = qubits[base + 2]
            
            if a_val == 1:
                qc.x(a)
            
            # Qutrit encoding: 0→00, 1→01, 2→10
            if b_val == 1:
                qc.x(b1)
            elif b_val == 2:
                qc.x(b0)
    
    # Map state types to preparation functions
    if state_type == "param_theta":
        return prepare_param_theta
    elif state_type == "param_theta_12":
        return prepare_param_theta_12
    elif state_type == "param_theta_phi":
        return prepare_param_theta_phi
    elif state_type == "maximally_entangled":
        return prepare_maximally_entangled
    elif state_type == "maximally_entangled_12":
        return prepare_maximally_entangled_12
    elif state_type.startswith("product_"):
        # Parse product_AB where A ∈ {0,1}, B ∈ {0,1,2}
        a_val = int(state_type[8])
        b_val = int(state_type[9])
        return lambda qc, qubits: prepare_product(qc, qubits, a_val, b_val)
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def get_qubit_qutrit_theoretical_values(
    state_type: str,
    theta: Optional[float] = None,
    phi: Optional[float] = None,
) -> Dict[str, float]:
    """
    Get theoretical values for qubit-qutrit states.
    
    For |ψ(θ)⟩ = cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩:
    - Density matrix ρ = |ψ⟩⟨ψ|
    - Partial transpose ρ^{T_A} has eigenvalues that can be computed analytically
    
    For |ψ(θ)⟩ = cos(θ/2)|0,1⟩ + sin(θ/2)|1,2⟩:
    - Same Schmidt coefficients, so same PT eigenvalues and negativity
    
    Returns:
        Dictionary with negativity, moments μ₂-μ₆, purity
    """
    # Both param_theta and param_theta_12 have the same Schmidt coefficients
    # and therefore the same PT eigenvalues and negativity
    if state_type in ["param_theta", "param_theta_12", 
                      "maximally_entangled", "maximally_entangled_12"]:
        if state_type in ["maximally_entangled", "maximally_entangled_12"]:
            theta = np.pi / 2
        elif theta is None:
            raise ValueError("theta required")
        
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        # For both state families with Schmidt coefficients (c, s):
        # PT eigenvalues: [c², s², -cs, cs, 0, 0]
        # This is because both are Schmidt rank 2 with same coefficients
        eigs = np.array([c**2, s**2, -c*s, c*s, 0, 0])
        
        # Moments
        mu_2 = np.sum(eigs**2)
        mu_3 = np.sum(eigs**3)
        mu_4 = np.sum(eigs**4)
        mu_5 = np.sum(eigs**5)
        mu_6 = np.sum(eigs**6)
        
        # Negativity = sum of absolute values of negative eigenvalues
        negativity = abs(min(0, -c*s))  # = c*s = sin(θ)/2
        
        return {
            "negativity": negativity,
            "mu_2": mu_2,
            "mu_3": mu_3,
            "mu_4": mu_4,
            "mu_5": mu_5,
            "mu_6": mu_6,
            "purity": 1.0,
            "theta": theta,
            "theta_degrees": np.degrees(theta),
            "pt_eigenvalues": eigs.tolist(),
            "state_family": "01-12" if "12" in state_type else "00-11",
        }
    
    elif state_type.startswith("product_"):
        # All product states are separable with N = 0
        return {
            "negativity": 0.0,
            "mu_2": 1.0,
            "mu_3": 1.0,
            "mu_4": 1.0,
            "mu_5": 1.0,
            "mu_6": 1.0,
            "purity": 1.0,
            "theta": 0.0,
            "theta_degrees": 0.0,
            "pt_eigenvalues": [0, 0, 0, 0, 0, 1],
        }
    
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def get_qubit_qutrit_pt_eigenvalues(
    state_type: str,
    theta: Optional[float] = None,
) -> np.ndarray:
    """Get partial transpose eigenvalues for qubit-qutrit state."""
    values = get_qubit_qutrit_theoretical_values(state_type, theta)
    return np.array(values["pt_eigenvalues"])


# Supported states
QUBIT_QUTRIT_PRODUCT_STATES = [
    "product_00", "product_01", "product_02",
    "product_10", "product_11", "product_12",
]

QUBIT_QUTRIT_ENTANGLED_STATES = [
    "maximally_entangled",      # (|0,0⟩ + |1,1⟩)/√2
    "maximally_entangled_12",   # (|0,1⟩ + |1,2⟩)/√2
    "param_theta",              # cos(θ/2)|0,0⟩ + sin(θ/2)|1,1⟩
    "param_theta_12",           # cos(θ/2)|0,1⟩ + sin(θ/2)|1,2⟩
]

QUBIT_QUTRIT_STATES = QUBIT_QUTRIT_PRODUCT_STATES + QUBIT_QUTRIT_ENTANGLED_STATES
