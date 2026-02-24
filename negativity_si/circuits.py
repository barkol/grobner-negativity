"""
Quantum circuits for partial transpose moment measurements.

Implements Hermitian measurement operators Hₙ = Mₙ + Mₙ† for moments μₙ.
All circuits use controlled-SWAP (CSWAP/Fredkin) gates with Hadamard test.

Circuit structure:
- Ancilla qubit for Hadamard test
- Control qubit for Hermitian measurement (X gate toggles forward/reverse)
- Multiple copies of the two-qubit state

References:
    - Horodecki, P. Phys. Rev. Lett. 90, 167901 (2003)
    - Ekert et al. Phys. Rev. Lett. 88, 217901 (2002)
"""

from typing import Optional, List, Callable
from qiskit import QuantumCircuit

from .states import create_state_preparation


def create_mu2_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for measuring μ₂ = Tr[(ρ^{T_A})²].
    
    For μ₂, the measurement uses SWAP on BOTH subsystems (A and B).
    This is because Tr[(ρ^{T_A})²] requires the full SWAP permutation
    when n=2 (cyclic-2 permutation is symmetric).
    
    Note: For ALL pure states, μ₂ = 1 (mathematical identity).
    
    Layout:
        q0, q1: Copy 1 (qubits A₁, B₁)
        q2, q3: Copy 2 (qubits A₂, B₂)
        q4: Ancilla (Hadamard test)
    
    Measurement outcome: μ₂ = 2P(0) - 1
    
    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states
        
    Returns:
        QuantumCircuit for μ₂ measurement
    """
    qc = QuantumCircuit(5, 1, name=f"mu2_{state_type}")
    
    # Prepare 2 copies of the state
    # Copy 1: qubits 0 (A₁), 1 (B₁)
    # Copy 2: qubits 2 (A₂), 3 (B₂)
    prepare = create_state_preparation(state_type, theta)
    prepare(qc, [0, 1, 2, 3])
    
    # Hadamard test
    qc.h(4)  # Ancilla in |+⟩
    
    # Controlled-SWAP on BOTH subsystems: A₁↔A₂ and B₁↔B₂
    qc.cswap(4, 0, 2)  # A₁ ↔ A₂
    qc.cswap(4, 1, 3)  # B₁ ↔ B₂
    
    # Complete Hadamard test
    qc.h(4)
    qc.measure(4, 0)
    
    return qc


def create_mu3_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for measuring μ₃ = Tr[(ρ^{T_A})³].
    
    For μ₃, the permutation operator requires:
    - Cyclic shift FORWARD on A subsystems: A₁ → A₂ → A₃ → A₁
    - Cyclic shift BACKWARD on B subsystems: B₁ ← B₂ ← B₃ ← B₁
    
    This implements Tr[(ρ^{T_A})³] via the permutation structure.
    
    Layout:
        q0, q1: Copy 1 (qubits A₁, B₁)
        q2, q3: Copy 2 (qubits A₂, B₂)
        q4, q5: Copy 3 (qubits A₃, B₃)
        q6: Ancilla (Hadamard test)
    
    Measurement outcome: μ₃ = 2P(0) - 1
    
    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states
        
    Returns:
        QuantumCircuit for μ₃ measurement
    """
    qc = QuantumCircuit(7, 1, name=f"mu3_{state_type}")
    
    # Prepare 3 copies of the state
    # Copy 1: (A₁=0, B₁=1), Copy 2: (A₂=2, B₂=3), Copy 3: (A₃=4, B₃=5)
    prepare = create_state_preparation(state_type, theta)
    prepare(qc, [0, 1, 2, 3, 4, 5])
    
    # Hadamard test
    qc.h(6)  # Ancilla in |+⟩
    
    # Cyclic 3-shift FORWARD on A: A₁ → A₂ → A₃ → A₁
    qc.cswap(6, 0, 2)   # A₁ ↔ A₂
    qc.cswap(6, 2, 4)   # A₂ ↔ A₃
    
    # Cyclic 3-shift BACKWARD on B: B₃ → B₂ → B₁ → B₃
    qc.cswap(6, 5, 3)   # B₃ ↔ B₂
    qc.cswap(6, 3, 1)   # B₂ ↔ B₁
    
    # Complete Hadamard test
    qc.h(6)
    qc.measure(6, 0)
    
    return qc


def create_mu4_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for measuring μ₄ = Tr[(ρ^{T_A})⁴].
    
    For μ₄, the permutation operator requires:
    - Cyclic shift FORWARD on A subsystems: A₁ → A₂ → A₃ → A₄ → A₁
    - Cyclic shift BACKWARD on B subsystems: B₁ ← B₂ ← B₃ ← B₄ ← B₁
    
    This implements Tr[(ρ^{T_A})⁴] via the permutation structure.
    
    Layout:
        q0, q1: Copy 1 (qubits A₁, B₁)
        q2, q3: Copy 2 (qubits A₂, B₂)
        q4, q5: Copy 3 (qubits A₃, B₃)
        q6, q7: Copy 4 (qubits A₄, B₄)
        q8: Ancilla (Hadamard test)
    
    Measurement outcome: μ₄ = 2P(0) - 1
    
    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states
        
    Returns:
        QuantumCircuit for μ₄ measurement
    """
    qc = QuantumCircuit(9, 1, name=f"mu4_{state_type}")
    
    # Prepare 4 copies of the state
    # Copy 1: (A₁=0, B₁=1), Copy 2: (A₂=2, B₂=3)
    # Copy 3: (A₃=4, B₃=5), Copy 4: (A₄=6, B₄=7)
    prepare = create_state_preparation(state_type, theta)
    prepare(qc, [0, 1, 2, 3, 4, 5, 6, 7])
    
    # Hadamard test
    qc.h(8)  # Ancilla in |+⟩
    
    # Cyclic 4-shift FORWARD on A: A₁ → A₂ → A₃ → A₄ → A₁
    qc.cswap(8, 0, 2)   # A₁ ↔ A₂
    qc.cswap(8, 2, 4)   # A₂ ↔ A₃
    qc.cswap(8, 4, 6)   # A₃ ↔ A₄
    
    # Cyclic 4-shift BACKWARD on B: B₄ → B₃ → B₂ → B₁ → B₄
    qc.cswap(8, 7, 5)   # B₄ ↔ B₃
    qc.cswap(8, 5, 3)   # B₃ ↔ B₂
    qc.cswap(8, 3, 1)   # B₂ ↔ B₁
    
    # Complete Hadamard test
    qc.h(8)
    qc.measure(8, 0)
    
    return qc


def create_purity_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for measuring purity I₂ = Tr[ρ²].
    
    Uses SWAP test on 2 copies of the state.
    Result: I₂ = 2P(0) - 1
    
    Note: For pure states, I₂ = 1. This circuit is useful for:
    - Noise calibration (measuring effective depolarization)
    - Verifying that μ₂ and I₂ give consistent results
    
    Layout:
        q0: Ancilla (SWAP test)
        q1, q2: Copy 1 (qubits A₁, B₁)
        q3, q4: Copy 2 (qubits A₂, B₂)
    
    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states
        
    Returns:
        QuantumCircuit for purity measurement
    """
    qc = QuantumCircuit(5, 1, name=f"purity_{state_type}")
    
    # Prepare 2 copies of the state on qubits [1,2] and [3,4]
    prepare = create_state_preparation(state_type, theta)
    prepare(qc, [1, 2, 3, 4])
    
    # SWAP test
    qc.h(0)
    
    # Controlled-SWAP both qubits
    qc.cswap(0, 1, 3)  # A₁ ↔ A₂
    qc.cswap(0, 2, 4)  # B₁ ↔ B₂
    
    qc.h(0)
    qc.measure(0, 0)
    
    return qc


def create_moment_circuits(
    state_type: str,
    theta: Optional[float] = None,
    include_purity: bool = True,
) -> dict:
    """
    Create all moment measurement circuits for a given state.
    
    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states
        include_purity: Whether to include purity circuit
        
    Returns:
        Dictionary of circuits: {"mu2": circuit, "mu3": circuit, "mu4": circuit, "purity": circuit}
    """
    circuits = {
        "mu2": create_mu2_circuit(state_type, theta),
        "mu3": create_mu3_circuit(state_type, theta),
        "mu4": create_mu4_circuit(state_type, theta),
    }
    
    if include_purity:
        circuits["purity"] = create_purity_circuit(state_type, theta)
    
    return circuits


def extract_moment_from_counts(
    counts: dict,
    shots: int,
) -> tuple:
    """
    Extract moment value from measurement counts.
    
    For Hadamard test, the moment is: μ = 2P(0) - 1
    
    Args:
        counts: Dictionary of measurement outcomes {"0": n0, "1": n1}
        shots: Total number of shots
        
    Returns:
        Tuple of (moment_value, standard_deviation)
    """
    # Handle different count formats
    n0 = counts.get("0", counts.get("0x0", 0))
    n1 = counts.get("1", counts.get("0x1", 0))
    
    # Probability of measuring |0⟩
    p0 = n0 / shots
    
    # Moment value
    moment = 2 * p0 - 1
    
    # Standard deviation (binomial)
    p0_std = (p0 * (1 - p0) / shots) ** 0.5
    moment_std = 2 * p0_std
    
    return moment, moment_std


def get_circuit_depth_info(circuit: QuantumCircuit) -> dict:
    """
    Get circuit depth and gate count information.
    
    Args:
        circuit: QuantumCircuit to analyze
        
    Returns:
        Dictionary with circuit statistics
    """
    from qiskit.converters import circuit_to_dag
    
    dag = circuit_to_dag(circuit)
    
    return {
        "depth": circuit.depth(),
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "gate_counts": dict(circuit.count_ops()),
        "num_nonlocal_gates": circuit.num_nonlocal_gates(),
    }
