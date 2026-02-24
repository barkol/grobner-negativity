# Spin chirality across quantum state copies detects hidden entanglement

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Supplementary code and data for:

> **"Spin chirality across quantum state copies detects hidden entanglement"**
> Patrycja Tulewicz and Karol Bartkiewicz
> *Nature Physics* (2026)

## Overview

This repository provides code for:

1. **Negativity estimation** from partial transpose moments via controlled-SWAP circuits
2. **Multi-copy chirality** measurement ($C_k = \mu_k - I_k$) as a geometrically interpretable entanglement witness
3. **Bound entanglement detection** through realignment matrix spectral features ($D_k = \Sigma_k - G_k$)
4. **Algebraic classification** using Gröbner basis conditions for adaptive measurement selection

### Key results

- **Chirality decomposition**: $C_k = 8\,\mathrm{Tr}[\Omega_A\,\Omega_B\;\rho^{\otimes k}]$ — multi-copy spin chirality as entanglement witness
- **Newton-Girard reconstruction**: 3 moments determine two-qubit negativity (5.3× efficiency gain over tomography)
- **Bound entanglement classifier**: 4-feature SVM achieving 68/68 accuracy on three PPT entangled families
- **IBM Quantum experiments**: validated on Kingston, Torino, Fez, and Marrakesh processors

## Repository structure

```
grobner-negativity/
├── negativity_si/                   # Core Python package
│   ├── states.py                    # State preparation (Bell, product, parametrized)
│   ├── circuits.py                  # Controlled-SWAP circuits for mu_k
│   ├── analysis.py                  # Newton-Girard reconstruction, negativity
│   ├── calibration.py               # Fake backend from calibration data
│   ├── experiment.py                # Main experiment runner (2x2)
│   ├── maxlik.py                    # Maximum likelihood calibration
│   ├── validation.py                # Formula validation
│   ├── simulations.py               # Chirality simulation pipeline
│   ├── *_qubit_qutrit.py            # Qubit-qutrit (2x3) variants
│   ├── chirality/                   # Modular chirality subpackage
│   ├── negativity/                  # Modular negativity subpackage
│   ├── common/                      # Shared utilities
│   └── data/                        # Calibration CSVs (Kingston, Torino)
│
├── analysis/                        # Experiment and analysis scripts
│   ├── ibmq_two_feature_batched.py  # BE classification on IBM hardware
│   ├── simulate_negativity_chirality_many.py  # Fig. S5 simulation
│   ├── simulate_be_many.py          # Fig. S6 simulation
│   ├── two_feature_svm.py           # SVM training for BE classifier
│   ├── svm_classifier.json          # Trained 4-feature RBF SVM model
│   └── *.json                       # Simulation results
│
├── verification/                    # Analytic expression verification
│   └── verify_all.py               # Verifies all manuscript equations
│
├── manuscript/                      # LaTeX source files
│   ├── manuscript_nature_physics.tex
│   ├── SI_nature_physics.tex
│   ├── graphic_story.tex
│   └── Sk_multicopy_derivation.tex
│
├── tests/                           # Unit tests
│   └── test_negativity.py
│
├── data/                            # Experimental data
│   └── SIM_qubit_qutrit_results.json
│
├── pyproject.toml                   # Package configuration
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Installation

```bash
# Basic (simulation only)
pip install .

# With IBM Quantum hardware support
pip install ".[ibm]"

# With ML classifier support
pip install ".[ml]"

# Full development install
pip install -e ".[all]"
```

## Verification

Verify all analytic expressions from the manuscript:

```bash
python verification/verify_all.py
```

This checks 24 categories of results including:
- Newton-Girard identities (Eq. 7)
- Parametrised state moments (Methods)
- Negativity formula $\mathcal{N}(\theta) = \sin\theta/2$ (Eq. 5)
- Chirality formula $C_4(\theta)$ (Eq. 5)
- $C_2 = 0$ identity (Eq. 2)
- Pure-state relation $-C_4 = 4\mathcal{N}^2(1-\mathcal{N}^2)$ (Eq. 6)
- Bell state values ($\mathcal{N}=1/2$, $C_4=-3/4$)
- Triple degeneracy condition $\mathcal{G}_2 = 0$ (Eq. 3)
- Triple root negativity formula (Eq. 4)
- Werner state chirality $C_4 = -3p^3/4$
- Separable bounds $|C_4| \leq 1/27$
- Correlation tensor identity $C_4 = \tfrac{3}{4}\det(T)$
- CCNR crossover at $a \approx 0.28$
- $D_k$ classifier decision boundary
- Horodecki PPT verification

## Quick start

### Negativity measurement (simulation)

```python
from negativity_si import NegativityExperiment

exp = NegativityExperiment()
results = exp.run()
exp.print_summary()
```

### Real hardware

```python
from negativity_si import NegativityExperiment

exp = NegativityExperiment(
    backend_name="ibm_torino",
    api_key="YOUR_IBM_QUANTUM_API_KEY",
)
results = exp.run()
```

### Bound entanglement classification

```python
from analysis.ibmq_two_feature_batched import run_experiment

# Dry-run: theoretical features only
run_experiment(mode="dry-run")

# Simulation with noise
run_experiment(mode="simulator")
```

## Circuits

| Circuit | Measures | Copies | Description |
|---------|----------|--------|-------------|
| $C_{\mu_2}$ | $\mu_2 = \mathrm{Tr}[(\rho^{T_A})^2]$ | 2 | Purity of partial transpose |
| $C_{\mu_3}$ | $\mu_3 = \mathrm{Tr}[(\rho^{T_A})^3]$ | 3 | Third moment (chirality) |
| $C_{\mu_4}$ | $\mu_4 = \mathrm{Tr}[(\rho^{T_A})^4]$ | 4 | Fourth moment (chirality) |
| $C_{I_2}$ | $I_2 = \mathrm{Tr}[\rho^2]$ | 2 | Purity |

## Citation

```bibtex
@article{tulewicz2026chirality,
  title={Spin chirality across quantum state copies detects hidden entanglement},
  author={Tulewicz, Patrycja and Bartkiewicz, Karol},
  journal={Nature Physics},
  year={2026},
  publisher={Nature Publishing Group}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Karol Bartkiewicz**: karol.bartkiewicz@amu.edu.pl
- Institute of Spintronics and Quantum Information, Adam Mickiewicz University, Poznań, Poland
