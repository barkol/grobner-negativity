"""
Negativity and Chirality Estimation from Quantum Circuits
==========================================================

Supplementary code for:
"Resource-efficient negativity estimation from partial transpose moments"
by Patrycja Tulewicz and Karol Bartkiewicz
npj Quantum Information (2026)

This package provides tools for measuring entanglement negativity and chirality
on IBM Quantum hardware using controlled-SWAP circuits.

Package Structure:
==================
- negativity/        Negativity estimation from PT moments (μ₂, μ₃, μ₄, ...)
  - qubit_qubit/     2×2 systems (4 PT eigenvalues, moments μ₂-μ₄)
  - qubit_qutrit/    2×3 systems (6 PT eigenvalues, moments μ₂-μ₆)

- chirality/         Chirality witness Q = I₂² - M₂
  - qubit_qubit/     2×2 systems (5 qubits for I₂, 9 for M₂)
  - qubit_qutrit/    2×3 systems (extended circuits)

- common/            Shared utilities
  - noise_models.py  IBM calibration data parsing
  - states.py        Parametrized state creation
  - circuit_utils.py Circuit execution utilities

Key Features:
- Two-Stage Maximum Likelihood Estimation:
  * Stage 1: Oracle calibration of degradation factors (f₂, f₃, ..., p)
  * Stage 2: Blind state estimation using calibrated parameters
- Newton-Girard reconstruction: Machine-precision eigenvalue recovery
- KEY IDENTITY: μ₂ = R₂ = I₂ for ALL bipartite states

Quick Start:
    # Negativity estimation (2×2)
    from negativity_si.negativity.qubit_qubit import NegativityMaxLikEstimator
    estimator = NegativityMaxLikEstimator()
    estimator.add_state("theta_45", 45.0, mu2=0.95, mu3=0.85, mu4=0.75)
    results = estimator.fit_all()

    # Chirality estimation (2×2)
    from negativity_si.chirality.qubit_qubit import ChiralityMaxLikEstimator
    estimator = ChiralityMaxLikEstimator()
    estimator.add_state("theta_45", 45.0, I2=0.95, M2=0.70)
    results = estimator.fit_all()

    # Legacy API (backward compatible)
    from negativity_si import NegativityExperiment
    exp = NegativityExperiment()
    results = exp.run()
"""

__version__ = "2.0.0"
__author__ = "Patrycja Tulewicz, Karol Bartkiewicz"
__email__ = "karol.bartkiewicz@amu.edu.pl"
__citation__ = """
@article{tulewicz2026negativity,
  title={Resource-efficient negativity estimation from partial transpose moments},
  author={Tulewicz, Patrycja and Bartkiewicz, Karol},
  journal={npj Quantum Information},
  year={2026},
  publisher={Nature Publishing Group}
}
"""

# =============================================================================
# NEW MODULAR STRUCTURE (v2.0)
# =============================================================================

# Negativity estimation with two-stage ML
from .negativity import qubit_qubit as negativity_2x2
from .negativity import qubit_qutrit as negativity_2x3

# Chirality witness with two-stage ML
from .chirality import qubit_qubit as chirality_2x2
from .chirality import qubit_qutrit as chirality_2x3

# Common utilities
from .common import (
    parse_calibration_csv,
    create_noise_model_from_calibration,
)

# =============================================================================
# LEGACY API (backward compatible with v1.0)
# =============================================================================

# Qubit-qubit (2×2) system - legacy
from .experiment import NegativityExperiment
from .analysis import (
    compute_negativity_newton_girard,
    compute_negativity_from_moments,
    theoretical_moments,
    theoretical_negativity,
)
from .circuits import (
    create_mu2_circuit,
    create_mu3_circuit,
    create_mu4_circuit,
    create_purity_circuit,
)
from .states import (
    create_state_preparation,
    get_theoretical_values,
    SUPPORTED_STATES,
)
from .calibration import create_fake_kingston_backend
from .maxlik import MaxLikEstimator
from .validation import run_validation

# Simulation module (optional - requires qiskit-aer)
try:
    from .simulations import (
        run_chirality_simulation,
        run_simulation_validation,
        compute_theoretical_values,
        create_parametrized_state,
        save_results_to_csv,
    )
    SIMULATIONS_AVAILABLE = True
except ImportError:
    SIMULATIONS_AVAILABLE = False

# Qubit-qutrit (2×3) system - legacy
from .experiment_qubit_qutrit import QubitQutritExperiment
from .analysis_qubit_qutrit import (
    compute_negativity_qubit_qutrit,
    compute_negativity_from_moments_qubit_qutrit,
    theoretical_moments_qubit_qutrit,
    theoretical_negativity_qubit_qutrit,
    reconstruct_eigenvalues_6,
)
from .circuits_qubit_qutrit import (
    create_qubit_qutrit_mu2_circuit,
    create_qubit_qutrit_mu3_circuit,
    create_qubit_qutrit_mu4_circuit,
    create_qubit_qutrit_mu5_circuit,
    create_qubit_qutrit_mu6_circuit,
    create_qubit_qutrit_moment_circuits,
)
from .states_qubit_qutrit import (
    create_qubit_qutrit_state_preparation,
    get_qubit_qutrit_theoretical_values,
    QUBIT_QUTRIT_STATES,
)

__all__ = [
    # New modular structure (v2.0)
    "negativity_2x2",
    "negativity_2x3",
    "chirality_2x2",
    "chirality_2x3",

    # Common utilities
    "parse_calibration_csv",
    "create_noise_model_from_calibration",

    # Legacy: Qubit-qubit (2×2) negativity
    "NegativityExperiment",
    "compute_negativity_newton_girard",
    "compute_negativity_from_moments",
    "theoretical_moments",
    "theoretical_negativity",
    "create_mu2_circuit",
    "create_mu3_circuit",
    "create_mu4_circuit",
    "create_purity_circuit",
    "create_state_preparation",
    "get_theoretical_values",
    "SUPPORTED_STATES",
    "create_fake_kingston_backend",
    "MaxLikEstimator",
    "run_validation",

    # Legacy: Qubit-qutrit (2×3) negativity
    "QubitQutritExperiment",
    "compute_negativity_qubit_qutrit",
    "compute_negativity_from_moments_qubit_qutrit",
    "theoretical_moments_qubit_qutrit",
    "theoretical_negativity_qubit_qutrit",
    "reconstruct_eigenvalues_6",
    "create_qubit_qutrit_mu2_circuit",
    "create_qubit_qutrit_mu3_circuit",
    "create_qubit_qutrit_mu4_circuit",
    "create_qubit_qutrit_mu5_circuit",
    "create_qubit_qutrit_mu6_circuit",
    "create_qubit_qutrit_moment_circuits",
    "create_qubit_qutrit_state_preparation",
    "get_qubit_qutrit_theoretical_values",
    "QUBIT_QUTRIT_STATES",

    # Simulations (if available)
    "run_chirality_simulation",
    "run_simulation_validation",
    "compute_theoretical_values",
    "create_parametrized_state",
    "save_results_to_csv",
    "SIMULATIONS_AVAILABLE",
]
