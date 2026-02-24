"""
Configuration and default parameters for negativity experiments.
"""

import numpy as np
from pathlib import Path

# Package data directory
DATA_DIR = Path(__file__).parent / "data"
CALIBRATION_FILE = DATA_DIR / "kingston_calibration.csv"

# Default experiment parameters
DEFAULT_SHOTS = 100000
DEFAULT_OPTIMIZATION_LEVEL = 3

# Default theta values for parameterized states
# |ψ(θ)⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩
# θ=0: separable, θ=π/2: maximally entangled (Bell state)
DEFAULT_THETA_VALUES = [
    0.0,           # |00⟩ (separable)
    np.pi / 6,     # 30° (weakly entangled)
    np.pi / 4,     # 45° (moderately entangled)
    np.pi / 3,     # 60° (strongly entangled)
    np.pi / 2,     # 90° = |Φ⁺⟩ (maximally entangled)
]

# Bell state identifiers
BELL_STATES = [
    "bell_phi_plus",   # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    "bell_phi_minus",  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    "bell_psi_plus",   # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    "bell_psi_minus",  # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
]

# Product states (computational basis)
PRODUCT_STATES = [
    "product_00",  # |00⟩
    "product_01",  # |01⟩
    "product_10",  # |10⟩
    "product_11",  # |11⟩
]

# All supported states
ALL_STATES = BELL_STATES + PRODUCT_STATES

# IBM Quantum service configuration
IBM_CHANNEL = "ibm_cloud"
DEFAULT_INSTANCE = "ibm-q/open/main"

# Qubit selection for Kingston (based on calibration quality)
# These qubits have favorable error rates and connectivity
RECOMMENDED_QUBITS_KINGSTON = {
    "mu2": [12, 13, 14, 17, 18],     # 5 qubits for μ₂
    "mu3": [11, 12, 13, 14, 17, 18, 23],  # 7 qubits for μ₃
    "mu4": [10, 11, 12, 13, 14, 17, 18, 23, 24],  # 9 qubits for μ₄
}

# Circuit qubit requirements
CIRCUIT_QUBITS = {
    "mu2": 5,   # 2 copies × 2 qubits + 1 ancilla
    "mu3": 7,   # 3 copies × 2 qubits + 1 ancilla
    "mu4": 9,   # 4 copies × 2 qubits + 1 ancilla
    "purity": 5,  # 2 copies × 2 qubits + 1 ancilla
}

# Numerical tolerances
EIGENVALUE_TOLERANCE = 1e-10
MOMENT_TOLERANCE = 1e-12
