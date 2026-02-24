"""
Two-Stage Maximum Likelihood Estimation for Chirality Witness.

Physical model with depolarization:
    ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

Circuit degradation model:
    I₂_measured = f_I2 × I₂_model(p)
    M₂_measured = f_M2 × M₂_model(θ, p)

TWO-STAGE APPROACH:
    Stage 1 (Oracle Calibration): Use KNOWN θ values to fit f_I2, f_M2, p
                                  These are hardware/noise parameters
    Stage 2 (Blind Estimation): Use calibrated f_I2, f_M2, p to fit θ
                                This gives M₂^phys for unknown states
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from typing import Dict, List, Optional
from dataclasses import dataclass
from numpy import cos, sin

from .analysis import I2_model_depolarized, M2_model_depolarized


@dataclass
class StateData:
    """Container for experimental data of a single state."""
    name: str
    theta_deg: float
    theta_rad: float
    I2_meas: float
    M2_meas: float
    sigma_I2: float
    sigma_M2: float
    M2_theory: float
    Q_theory: float


class ChiralityMaxLikEstimator:
    """
    Two-Stage Maximum Likelihood Estimator for chirality witness.

    Physical model with depolarization:
        ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4

    Circuit degradation model:
        I₂_measured = f_I2 × I₂_model(p)
        M₂_measured = f_M2 × M₂_model(θ, p)

    Two-stage approach:
        Stage 1 (Oracle Calibration): Use KNOWN θ to fit f_I2, f_M2, p
        Stage 2 (Blind Estimation): Use calibrated f, p to fit θ → M₂^phys, Q^phys

    Example usage:
        estimator = ChiralityMaxLikEstimator()
        estimator.add_state("theta_30", 30.0, I2=0.95, M2=0.85)
        estimator.add_state("theta_45", 45.0, I2=0.94, M2=0.70)
        results = estimator.fit_all()
    """

    def __init__(self):
        self.states: List[StateData] = []
        self.f_I2: Optional[float] = None
        self.f_M2: Optional[float] = None
        self.p_cal: Optional[float] = None
        self.calibrated: bool = False

    def add_state(
        self,
        name: str,
        theta_deg: float,
        I2_meas: float,
        M2_meas: float,
        sigma_I2: float = 0.02,
        sigma_M2: float = 0.03,
    ) -> None:
        """
        Add a state's experimental data.

        Args:
            name: State identifier
            theta_deg: True theta value in degrees
            I2_meas: Measured I₂ (raw, not corrected)
            M2_meas: Measured M₂
            sigma_I2, sigma_M2: Measurement uncertainties
        """
        theta_rad = np.radians(theta_deg)
        c = cos(theta_rad/2)
        s = sin(theta_rad/2)
        M2_theory = (c**4 + s**4)**2
        Q_theory = 1.0 - M2_theory  # For pure state, I₂ = 1

        self.states.append(StateData(
            name=name,
            theta_deg=theta_deg,
            theta_rad=theta_rad,
            I2_meas=I2_meas,
            M2_meas=M2_meas,
            sigma_I2=sigma_I2,
            sigma_M2=sigma_M2,
            M2_theory=M2_theory,
            Q_theory=Q_theory,
        ))

    def _calibration_nll(self, params: np.ndarray) -> float:
        """
        Stage 1: Calibration NLL using KNOWN theta values.

        params = [f_I2, f_M2, p]  (shared p for all states)
        """
        f_I2, f_M2, p = params

        # Bounds check
        if f_I2 <= 0 or f_M2 <= 0 or f_I2 > 1.5 or f_M2 > 1.0:
            return 1e10
        if p < 0 or p > 0.5:
            return 1e10

        nll = 0.0

        for state in self.states:
            # Use KNOWN theta (oracle)
            theta = state.theta_rad

            # Model predictions with known theta
            I2_model = I2_model_depolarized(p)
            M2_model = M2_model_depolarized(theta, p)

            # Apply degradation
            I2_pred = f_I2 * I2_model
            M2_pred = f_M2 * M2_model

            # NLL (Gaussian likelihood)
            nll += 0.5 * ((state.I2_meas - I2_pred)**2 / state.sigma_I2**2)
            nll += 0.5 * ((state.M2_meas - M2_pred)**2 / state.sigma_M2**2)

        return nll

    def calibrate(self) -> dict:
        """
        Stage 1: Calibrate f_I2, f_M2, p using KNOWN theta values (oracle).

        Returns:
            Dictionary with calibrated parameters
        """
        if not self.states:
            raise ValueError("No states added")

        # Bounds: [f_I2, f_M2, p]
        bounds = [
            (0.8, 1.1),   # f_I2 (shallow circuit, ~no degradation)
            (0.1, 0.5),   # f_M2 (deep circuit, significant degradation)
            (0.0, 0.3),   # p (depolarization)
        ]

        result = differential_evolution(
            self._calibration_nll,
            bounds,
            seed=42,
            maxiter=1000,
            tol=1e-10,
            polish=True,
        )

        self.f_I2, self.f_M2, self.p_cal = result.x
        self.calibrated = True

        return {
            'f_I2': self.f_I2,
            'f_M2': self.f_M2,
            'p': self.p_cal,
            'success': result.success,
            'nll': result.fun,
        }

    def _fit_theta_nll(self, theta_rad: float, state: StateData) -> float:
        """NLL for fitting theta given fixed f_I2, f_M2, p."""
        if theta_rad < 0.001 or theta_rad > np.pi - 0.001:
            return 1e10

        I2_model = I2_model_depolarized(self.p_cal)
        M2_model = M2_model_depolarized(theta_rad, self.p_cal)

        I2_pred = self.f_I2 * I2_model
        M2_pred = self.f_M2 * M2_model

        nll = 0.5 * ((state.I2_meas - I2_pred)**2 / state.sigma_I2**2)
        nll += 0.5 * ((state.M2_meas - M2_pred)**2 / state.sigma_M2**2)

        return nll

    def fit_theta(self, state: StateData) -> dict:
        """
        Stage 2: Fit theta using calibrated f_I2, f_M2, p (blind estimation).

        Args:
            state: StateData with measured I₂, M₂

        Returns:
            Dictionary with fitted theta and derived M₂^phys, Q^phys
        """
        if not self.calibrated:
            raise ValueError("Must call calibrate() first")

        # Optimize theta
        result = minimize_scalar(
            lambda t: self._fit_theta_nll(t, state),
            bounds=(0.001, np.pi - 0.001),
            method='bounded',
        )

        theta_fit = result.x

        # Compute physical values using fitted theta and calibrated p
        I2_phys = I2_model_depolarized(self.p_cal)
        M2_phys = M2_model_depolarized(theta_fit, self.p_cal)
        Q_phys = I2_phys**2 - M2_phys

        return {
            'theta_fit_rad': theta_fit,
            'theta_fit_deg': np.degrees(theta_fit),
            'I2_phys': I2_phys,
            'M2_phys': M2_phys,
            'Q_phys': Q_phys,
            'nll': result.fun,
        }

    def fit_all(self) -> dict:
        """
        Run both stages: calibrate then fit all states.

        Returns:
            Complete results dictionary
        """
        # Stage 1: Calibration
        cal_results = self.calibrate()

        # Stage 2: Fit theta for each state
        results = {
            'f_I2': self.f_I2,
            'f_M2': self.f_M2,
            'p': self.p_cal,
            'calibration_success': cal_results['success'],
            'calibration_nll': cal_results['nll'],
            'states': {},
        }

        for state in self.states:
            fit_result = self.fit_theta(state)

            # Compute Q error
            Q_error = abs(fit_result['Q_phys'] - state.Q_theory)

            results['states'][state.name] = {
                'theta_true_deg': state.theta_deg,
                'theta_fit_deg': fit_result['theta_fit_deg'],
                'I2_meas': state.I2_meas,
                'M2_meas': state.M2_meas,
                'I2_phys': fit_result['I2_phys'],
                'M2_phys': fit_result['M2_phys'],
                'Q_phys': fit_result['Q_phys'],
                'Q_theory': state.Q_theory,
                'Q_error': Q_error,
            }

        # Summary
        errors = [d['Q_error'] for d in results['states'].values()]
        results['mean_error'] = np.mean(errors)

        return results

    def print_results(self, results: Optional[dict] = None) -> None:
        """Print formatted results."""
        if results is None:
            results = self.fit_all()

        print("=" * 80)
        print("TWO-STAGE MAXIMUM LIKELIHOOD ESTIMATION FOR CHIRALITY WITNESS")
        print("=" * 80)
        print("Model: ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/4")
        print("Stage 1: Oracle calibration of f_I2, f_M2, p using known θ")
        print("Stage 2: Blind θ fit using calibrated parameters")
        print()

        print(f"Calibrated parameters:")
        print(f"  f_I2 = {results['f_I2']:.4f}")
        print(f"  f_M2 = {results['f_M2']:.4f}")
        print(f"  p    = {results['p']:.4f}")
        print()

        print(f"{'State':<12} {'θ_true':>8} {'θ_fit':>8} {'M2_phys':>9} {'Q_phys':>9} {'Q_theory':>9} {'Error':>8}")
        print("-" * 75)

        for name, data in results['states'].items():
            print(f"{name:<12} {data['theta_true_deg']:>8.1f} {data['theta_fit_deg']:>8.1f} "
                  f"{data['M2_phys']:>9.4f} {data['Q_phys']:>9.4f} "
                  f"{data['Q_theory']:>9.4f} {data['Q_error']:>8.4f}")

        print("-" * 75)
        print(f"Mean error: {results['mean_error']:.4f}")


def run_chirality_maxlik(
    measurements: List[dict],
    verbose: bool = True,
) -> dict:
    """
    Convenience function to run two-stage ML analysis for chirality.

    Args:
        measurements: List of dicts with keys:
            - name: State identifier
            - theta_deg: True theta in degrees
            - I2: Measured I₂
            - M2: Measured M₂
        verbose: Print results

    Returns:
        Results dictionary
    """
    estimator = ChiralityMaxLikEstimator()

    for m in measurements:
        estimator.add_state(
            name=m['name'],
            theta_deg=m['theta_deg'],
            I2_meas=m['I2'],
            M2_meas=m['M2'],
        )

    results = estimator.fit_all()

    if verbose:
        estimator.print_results(results)

    return results
