"""
Quantum circuits for qubit-qutrit (2×3) partial transpose moment measurements.

For 2×3 systems:
- 6 PT eigenvalues → need μ₂, μ₃, μ₄, μ₅, μ₆ for full reconstruction
- Qutrit encoded in 2 qubits
- Each copy uses 3 qubits: 1 (qubit A) + 2 (qutrit B)

Circuit structure:
- Cyclic shift on A subsystem (qubit)
- Anti-cyclic shift on B subsystem (qutrit, both encoding qubits)
"""

from typing import Optional, List, Dict
from qiskit import QuantumCircuit

from .states_qubit_qutrit import (
    create_qubit_qutrit_state_preparation,
    QUBITS_PER_COPY,
)


def _apply_cyclic_cswaps_A(
    qc: QuantumCircuit,
    ancilla: int,
    a_qubits: List[int],
) -> None:
    """
    Apply controlled cyclic shift on A subsystem qubits.
    
    Cyclic shift: A₁ → A₂ → A₃ → ... → Aₙ → A₁
    Implemented as sequence of CSWAPs.
    """
    n = len(a_qubits)
    for i in range(n - 1):
        qc.cswap(ancilla, a_qubits[i], a_qubits[i + 1])


def _apply_anticyclic_cswaps_B(
    qc: QuantumCircuit,
    ancilla: int,
    b_qubits: List[List[int]],
) -> None:
    """
    Apply controlled anti-cyclic shift on B subsystem qubits.
    
    Anti-cyclic shift: Bₙ → Bₙ₋₁ → ... → B₁ → Bₙ
    Each B is a qutrit encoded in 2 qubits, so swap both.
    """
    n = len(b_qubits)
    for i in range(n - 1, 0, -1):
        # Swap both qubits of qutrit encoding
        qc.cswap(ancilla, b_qubits[i][0], b_qubits[i - 1][0])
        qc.cswap(ancilla, b_qubits[i][1], b_qubits[i - 1][1])


def create_qubit_qutrit_mu2_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for μ₂ = Tr[(ρ^{T_A})²] in 2×3 system.
    
    Uses 2 copies × 3 qubits + 1 ancilla = 7 qubits.
    
    Layout:
        q0: A₁ (qubit)
        q1, q2: B₁ (qutrit)
        q3: A₂ (qubit)
        q4, q5: B₂ (qutrit)
        q6: Ancilla
    """
    n_copies = 2
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"mu2_23_{state_type}")
    
    # Prepare state copies
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_qubit_qutrit_state_preparation(state_type, theta)
    prepare(qc, state_qubits)
    
    ancilla = n_qubits - 1
    qc.h(ancilla)
    
    # A qubits: indices 0, 3
    a_qubits = [0, 3]
    # B qubits: indices [1,2], [4,5]
    b_qubits = [[1, 2], [4, 5]]
    
    # Cyclic on A
    _apply_cyclic_cswaps_A(qc, ancilla, a_qubits)
    # Anti-cyclic on B (same as cyclic for n=2)
    _apply_anticyclic_cswaps_B(qc, ancilla, b_qubits)
    
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    
    return qc


def create_qubit_qutrit_mu3_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for μ₃ = Tr[(ρ^{T_A})³] in 2×3 system.
    
    Uses 3 copies × 3 qubits + 1 ancilla = 10 qubits.
    """
    n_copies = 3
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"mu3_23_{state_type}")
    
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_qubit_qutrit_state_preparation(state_type, theta)
    prepare(qc, state_qubits)
    
    ancilla = n_qubits - 1
    qc.h(ancilla)
    
    # A qubits: 0, 3, 6
    a_qubits = [i * QUBITS_PER_COPY for i in range(n_copies)]
    # B qubits: [1,2], [4,5], [7,8]
    b_qubits = [[i * QUBITS_PER_COPY + 1, i * QUBITS_PER_COPY + 2] for i in range(n_copies)]
    
    _apply_cyclic_cswaps_A(qc, ancilla, a_qubits)
    _apply_anticyclic_cswaps_B(qc, ancilla, b_qubits)
    
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    
    return qc


def create_qubit_qutrit_mu4_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for μ₄ = Tr[(ρ^{T_A})⁴] in 2×3 system.
    
    Uses 4 copies × 3 qubits + 1 ancilla = 13 qubits.
    """
    n_copies = 4
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"mu4_23_{state_type}")
    
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_qubit_qutrit_state_preparation(state_type, theta)
    prepare(qc, state_qubits)
    
    ancilla = n_qubits - 1
    qc.h(ancilla)
    
    a_qubits = [i * QUBITS_PER_COPY for i in range(n_copies)]
    b_qubits = [[i * QUBITS_PER_COPY + 1, i * QUBITS_PER_COPY + 2] for i in range(n_copies)]
    
    _apply_cyclic_cswaps_A(qc, ancilla, a_qubits)
    _apply_anticyclic_cswaps_B(qc, ancilla, b_qubits)
    
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    
    return qc


def create_qubit_qutrit_mu5_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for μ₅ = Tr[(ρ^{T_A})⁵] in 2×3 system.
    
    Uses 5 copies × 3 qubits + 1 ancilla = 16 qubits.
    """
    n_copies = 5
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"mu5_23_{state_type}")
    
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_qubit_qutrit_state_preparation(state_type, theta)
    prepare(qc, state_qubits)
    
    ancilla = n_qubits - 1
    qc.h(ancilla)
    
    a_qubits = [i * QUBITS_PER_COPY for i in range(n_copies)]
    b_qubits = [[i * QUBITS_PER_COPY + 1, i * QUBITS_PER_COPY + 2] for i in range(n_copies)]
    
    _apply_cyclic_cswaps_A(qc, ancilla, a_qubits)
    _apply_anticyclic_cswaps_B(qc, ancilla, b_qubits)
    
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    
    return qc


def create_qubit_qutrit_mu6_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for μ₆ = Tr[(ρ^{T_A})⁶] in 2×3 system.
    
    Uses 6 copies × 3 qubits + 1 ancilla = 19 qubits.
    """
    n_copies = 6
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"mu6_23_{state_type}")
    
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_qubit_qutrit_state_preparation(state_type, theta)
    prepare(qc, state_qubits)
    
    ancilla = n_qubits - 1
    qc.h(ancilla)
    
    a_qubits = [i * QUBITS_PER_COPY for i in range(n_copies)]
    b_qubits = [[i * QUBITS_PER_COPY + 1, i * QUBITS_PER_COPY + 2] for i in range(n_copies)]
    
    _apply_cyclic_cswaps_A(qc, ancilla, a_qubits)
    _apply_anticyclic_cswaps_B(qc, ancilla, b_qubits)
    
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    
    return qc


def create_qubit_qutrit_purity_circuit(
    state_type: str,
    theta: Optional[float] = None,
) -> QuantumCircuit:
    """
    Create circuit for purity I₂ = Tr[ρ²] in 2×3 system.
    
    Uses full SWAP test on both subsystems.
    Uses 2 copies × 3 qubits + 1 ancilla = 7 qubits.
    """
    n_copies = 2
    n_qubits = n_copies * QUBITS_PER_COPY + 1
    qc = QuantumCircuit(n_qubits, 1, name=f"purity_23_{state_type}")
    
    state_qubits = list(range(n_copies * QUBITS_PER_COPY))
    prepare = create_qubit_qutrit_state_preparation(state_type, theta)
    prepare(qc, state_qubits)
    
    ancilla = n_qubits - 1
    qc.h(ancilla)
    
    # Full SWAP on both subsystems
    qc.cswap(ancilla, 0, 3)   # A₁ ↔ A₂
    qc.cswap(ancilla, 1, 4)   # B₁[0] ↔ B₂[0]
    qc.cswap(ancilla, 2, 5)   # B₁[1] ↔ B₂[1]
    
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    
    return qc


def create_qubit_qutrit_moment_circuits(
    state_type: str,
    theta: Optional[float] = None,
    include_purity: bool = True,
    max_moment: int = 6,
) -> Dict[str, QuantumCircuit]:
    """
    Create all moment measurement circuits for qubit-qutrit state.
    
    Args:
        state_type: Type of state to prepare
        theta: Rotation angle for parameterized states
        include_purity: Whether to include purity circuit
        max_moment: Maximum moment to compute (2-6)
        
    Returns:
        Dictionary of circuits
    """
    circuit_creators = {
        "mu2": create_qubit_qutrit_mu2_circuit,
        "mu3": create_qubit_qutrit_mu3_circuit,
        "mu4": create_qubit_qutrit_mu4_circuit,
        "mu5": create_qubit_qutrit_mu5_circuit,
        "mu6": create_qubit_qutrit_mu6_circuit,
    }
    
    circuits = {}
    for i in range(2, max_moment + 1):
        key = f"mu{i}"
        if key in circuit_creators:
            circuits[key] = circuit_creators[key](state_type, theta)
    
    if include_purity:
        circuits["purity"] = create_qubit_qutrit_purity_circuit(state_type, theta)
    
    return circuits


# Circuit resource requirements
QUBIT_QUTRIT_CIRCUIT_QUBITS = {
    "mu2": 7,   # 2 × 3 + 1
    "mu3": 10,  # 3 × 3 + 1
    "mu4": 13,  # 4 × 3 + 1
    "mu5": 16,  # 5 × 3 + 1
    "mu6": 19,  # 6 × 3 + 1
    "purity": 7,
}

QUBIT_QUTRIT_CIRCUIT_CSWAPS = {
    "mu2": 1 + 2,      # 1 A-swap + 2 B-swaps (each qutrit needs 2)
    "mu3": 2 + 4,      # 2 A-swaps + 4 B-swaps
    "mu4": 3 + 6,      # 3 A-swaps + 6 B-swaps
    "mu5": 4 + 8,      # 4 A-swaps + 8 B-swaps
    "mu6": 5 + 10,     # 5 A-swaps + 10 B-swaps
    "purity": 3,       # 1 A-swap + 2 B-swaps
}
