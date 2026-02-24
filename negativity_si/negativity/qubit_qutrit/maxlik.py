"""
Two-Stage Maximum Likelihood Estimation for Qubit-Qutrit Negativity.

Physical model with depolarization:
    ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/6

Circuit degradation model:
    μₖ_measured = fₖ × μₖ_model(θ, p)

TWO-STAGE APPROACH:
    Stage 1 (Oracle Calibration): Use KNOWN θ values to fit f₂, f₃, f₄, f₅, f₆, p
                                  These are hardware/noise parameters
    Stage 2 (Blind Estimation): Use calibrated fₖ, p to fit θ
                                This gives N^phys for unknown states
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .analysis import (
    pt_eigenvalues_mixed,
    moments_from_eigenvalues,
    negativity_from_eigenvalues,
    theoretical_negativity,
    compute_negativity_qubit_qutrit,
)


@dataclass
class StateData:
    """Container for experimental data of a single qubit-qutrit state."""
    name: str
    theta_deg: float
    theta_rad: float
    mu2_meas: float
    mu3_meas: float
    mu4_meas: float
    mu5_meas: float
    mu6_meas: float
    sigma2: float
    sigma3: float
    sigma4: float
    sigma5: float
    sigma6: float
    N_theory: float


class NegativityMaxLikEstimator:
    """
    Two-Stage Maximum Likelihood Estimator for qubit-qutrit negativity.

    Physical model with depolarization:
        ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/6

    Circuit degradation model:
        μₖ_measured = fₖ × μₖ_model(θ, p)

    Two-stage approach:
        Stage 1 (Oracle Calibration): Use KNOWN θ to fit f₂, f₃, f₄, f₅, f₆, p
        Stage 2 (Blind Estimation): Use calibrated f, p to fit θ → N^phys

    Example usage:
        estimator = NegativityMaxLikEstimator()
        estimator.add_state("theta_30", 30.0, mu2=0.95, mu3=0.85, mu4=0.75, mu5=0.65, mu6=0.55)
        results = estimator.fit_all()
    """

    def __init__(self):
        self.states: List[StateData] = []
        self.f2: Optional[float] = None
        self.f3: Optional[float] = None
        self.f4: Optional[float] = None
        self.f5: Optional[float] = None
        self.f6: Optional[float] = None
        self.p_cal: Optional[float] = None
        self.calibrated: bool = False

    def add_state(
        self,
        name: str,
        theta_deg: float,
        mu2_meas: float,
        mu3_meas: float,
        mu4_meas: float,
        mu5_meas: float,
        mu6_meas: float,
        sigma2: float = 0.02,
        sigma3: float = 0.02,
        sigma4: float = 0.02,
        sigma5: float = 0.02,
        sigma6: float = 0.02,
    ) -> None:
        """
        Add a state's experimental data.

        Args:
            name: State identifier
            theta_deg: True theta value in degrees
            mu2_meas through mu6_meas: Measured moments (raw, not corrected)
            sigma2 through sigma6: Measurement uncertainties
        """
        theta_rad = np.radians(theta_deg)
        N_theory = theoretical_negativity(theta_deg)

        self.states.append(StateData(
            name=name,
            theta_deg=theta_deg,
            theta_rad=theta_rad,
            mu2_meas=mu2_meas,
            mu3_meas=mu3_meas,
            mu4_meas=mu4_meas,
            mu5_meas=mu5_meas,
            mu6_meas=mu6_meas,
            sigma2=sigma2,
            sigma3=sigma3,
            sigma4=sigma4,
            sigma5=sigma5,
            sigma6=sigma6,
            N_theory=N_theory,
        ))

    def _moments_model(self, theta_rad: float, p: float) -> Tuple[float, float, float, float, float]:
        """Compute model moments for depolarized qubit-qutrit state."""
        eigs = pt_eigenvalues_mixed(theta_rad, p)
        return moments_from_eigenvalues(eigs)

    def _calibration_nll(self, params: np.ndarray) -> float:
        """
        Stage 1: Calibration NLL using KNOWN theta values.

        params = [f2, f3, f4, f5, f6, p]  (shared p for all states)
        """
        f2, f3, f4, f5, f6, p = params

        # Bounds check
        for f in [f2, f3, f4, f5, f6]:
            if f <= 0 or f > 1.5:
                return 1e10
        if p < 0 or p > 0.5:
            return 1e10

        nll = 0.0

        for state in self.states:
            # Use KNOWN theta (oracle)
            theta = state.theta_rad

            # Model predictions with known theta
            mu2_model, mu3_model, mu4_model, mu5_model, mu6_model = self._moments_model(theta, p)

            # Apply degradation
            mu2_pred = f2 * mu2_model
            mu3_pred = f3 * mu3_model
            mu4_pred = f4 * mu4_model
            mu5_pred = f5 * mu5_model
            mu6_pred = f6 * mu6_model

            # NLL (Gaussian likelihood)
            nll += 0.5 * ((state.mu2_meas - mu2_pred)**2 / state.sigma2**2)
            nll += 0.5 * ((state.mu3_meas - mu3_pred)**2 / state.sigma3**2)
            nll += 0.5 * ((state.mu4_meas - mu4_pred)**2 / state.sigma4**2)
            nll += 0.5 * ((state.mu5_meas - mu5_pred)**2 / state.sigma5**2)
            nll += 0.5 * ((state.mu6_meas - mu6_pred)**2 / state.sigma6**2)

        return nll

    def calibrate(self) -> dict:
        """
        Stage 1: Calibrate f₂, f₃, f₄, f₅, f₆, p using KNOWN theta values (oracle).

        Returns:
            Dictionary with calibrated parameters
        """
        if not self.states:
            raise ValueError("No states added")

        # Bounds: [f2, f3, f4, f5, f6, p]
        # Deeper circuits (higher moments) typically have more degradation
        bounds = [
            (0.7, 1.2),   # f2 (shallow circuit)
            (0.5, 1.2),   # f3 (medium circuit)
            (0.3, 1.2),   # f4 (deeper circuit)
            (0.2, 1.2),   # f5 (even deeper)
            (0.1, 1.2),   # f6 (deepest circuit)
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

        self.f2, self.f3, self.f4, self.f5, self.f6, self.p_cal = result.x
        self.calibrated = True

        return {
            'f2': self.f2,
            'f3': self.f3,
            'f4': self.f4,
            'f5': self.f5,
            'f6': self.f6,
            'p': self.p_cal,
            'success': result.success,
            'nll': result.fun,
        }

    def _fit_theta_nll(self, theta_rad: float, state: StateData) -> float:
        """NLL for fitting theta given fixed fₖ, p."""
        if theta_rad < 0.001 or theta_rad > np.pi - 0.001:
            return 1e10

        mu2_model, mu3_model, mu4_model, mu5_model, mu6_model = self._moments_model(theta_rad, self.p_cal)

        mu2_pred = self.f2 * mu2_model
        mu3_pred = self.f3 * mu3_model
        mu4_pred = self.f4 * mu4_model
        mu5_pred = self.f5 * mu5_model
        mu6_pred = self.f6 * mu6_model

        nll = 0.5 * ((state.mu2_meas - mu2_pred)**2 / state.sigma2**2)
        nll += 0.5 * ((state.mu3_meas - mu3_pred)**2 / state.sigma3**2)
        nll += 0.5 * ((state.mu4_meas - mu4_pred)**2 / state.sigma4**2)
        nll += 0.5 * ((state.mu5_meas - mu5_pred)**2 / state.sigma5**2)
        nll += 0.5 * ((state.mu6_meas - mu6_pred)**2 / state.sigma6**2)

        return nll

    def fit_theta(self, state: StateData) -> dict:
        """
        Stage 2: Fit theta using calibrated fₖ, p (blind estimation).

        Args:
            state: StateData with measured moments

        Returns:
            Dictionary with fitted theta and derived N^phys
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

        # Compute physical negativity using fitted theta and calibrated p
        eigs_fit = pt_eigenvalues_mixed(theta_fit, self.p_cal)
        N_phys = negativity_from_eigenvalues(eigs_fit)

        return {
            'theta_fit_rad': theta_fit,
            'theta_fit_deg': np.degrees(theta_fit),
            'N_phys': N_phys,
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
            'f2': self.f2,
            'f3': self.f3,
            'f4': self.f4,
            'f5': self.f5,
            'f6': self.f6,
            'p': self.p_cal,
            'calibration_success': cal_results['success'],
            'calibration_nll': cal_results['nll'],
            'states': {},
        }

        for state in self.states:
            fit_result = self.fit_theta(state)

            # Compute error
            N_error = abs(fit_result['N_phys'] - state.N_theory)

            # Also compute simple correction for comparison
            mu2_corr = state.mu2_meas / self.f2 if self.f2 > 0 else state.mu2_meas
            mu3_corr = state.mu3_meas / self.f3 if self.f3 > 0 else state.mu3_meas
            mu4_corr = state.mu4_meas / self.f4 if self.f4 > 0 else state.mu4_meas
            mu5_corr = state.mu5_meas / self.f5 if self.f5 > 0 else state.mu5_meas
            mu6_corr = state.mu6_meas / self.f6 if self.f6 > 0 else state.mu6_meas
            try:
                N_simple = compute_negativity_qubit_qutrit(mu2_corr, mu3_corr, mu4_corr, mu5_corr, mu6_corr)
            except:
                N_simple = 0.0

            results['states'][state.name] = {
                'theta_true_deg': state.theta_deg,
                'theta_fit_deg': fit_result['theta_fit_deg'],
                'mu2_meas': state.mu2_meas,
                'mu3_meas': state.mu3_meas,
                'mu4_meas': state.mu4_meas,
                'mu5_meas': state.mu5_meas,
                'mu6_meas': state.mu6_meas,
                'N_phys': fit_result['N_phys'],
                'N_simple': N_simple,
                'N_theory': state.N_theory,
                'N_error_ml': N_error,
                'N_error_simple': abs(N_simple - state.N_theory),
            }

        # Summary
        ml_errors = [d['N_error_ml'] for d in results['states'].values()]
        simple_errors = [d['N_error_simple'] for d in results['states'].values()]
        results['mean_error_ml'] = np.mean(ml_errors)
        results['mean_error_simple'] = np.mean(simple_errors)

        return results

    def print_results(self, results: Optional[dict] = None) -> None:
        """Print formatted results."""
        if results is None:
            results = self.fit_all()

        print("=" * 90)
        print("TWO-STAGE MAXIMUM LIKELIHOOD ESTIMATION FOR QUBIT-QUTRIT NEGATIVITY")
        print("=" * 90)
        print("Model: ρ = (1-p)|ψ(θ)⟩⟨ψ(θ)| + p·I/6")
        print("Stage 1: Oracle calibration of f₂, f₃, f₄, f₅, f₆, p using known θ")
        print("Stage 2: Blind θ fit using calibrated parameters")
        print()

        print(f"Calibrated parameters:")
        print(f"  f₂ = {results['f2']:.4f}")
        print(f"  f₃ = {results['f3']:.4f}")
        print(f"  f₄ = {results['f4']:.4f}")
        print(f"  f₅ = {results['f5']:.4f}")
        print(f"  f₆ = {results['f6']:.4f}")
        print(f"  p  = {results['p']:.4f}")
        print()

        print(f"{'State':<12} {'θ_true':>8} {'θ_fit':>8} {'N_theory':>9} {'N_ML':>9} {'N_simple':>9} {'Err_ML':>8}")
        print("-" * 80)

        for name, data in results['states'].items():
            print(f"{name:<12} {data['theta_true_deg']:>8.1f} {data['theta_fit_deg']:>8.1f} "
                  f"{data['N_theory']:>9.4f} {data['N_phys']:>9.4f} {data['N_simple']:>9.4f} "
                  f"{data['N_error_ml']:>8.4f}")

        print("-" * 80)
        print(f"Mean error: Two-stage ML = {results['mean_error_ml']:.4f}, "
              f"Simple correction = {results['mean_error_simple']:.4f}")


def run_negativity_maxlik(
    measurements: List[dict],
    verbose: bool = True,
) -> dict:
    """
    Convenience function to run two-stage ML analysis for qubit-qutrit.

    Args:
        measurements: List of dicts with keys:
            - name: State identifier
            - theta_deg: True theta in degrees
            - mu2, mu3, mu4, mu5, mu6: Measured moments
        verbose: Print results

    Returns:
        Results dictionary
    """
    estimator = NegativityMaxLikEstimator()

    for m in measurements:
        estimator.add_state(
            name=m['name'],
            theta_deg=m['theta_deg'],
            mu2_meas=m['mu2'],
            mu3_meas=m['mu3'],
            mu4_meas=m['mu4'],
            mu5_meas=m['mu5'],
            mu6_meas=m['mu6'],
        )

    results = estimator.fit_all()

    if verbose:
        estimator.print_results(results)

    return results
