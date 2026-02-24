"""
Maximum Likelihood Estimation for Negativity from Noisy Moments.

This module implements a physical model-based maximum likelihood estimator
that jointly optimizes degradation factors and state parameters to extract
accurate negativity values from noisy quantum hardware measurements.

Physical Model (with depolarization):
    rho = (1-p) |psi(theta)><psi(theta)| + p * I/4

    - State: |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>
    - Pure PT eigenvalues: [-sin(theta/2)cos(theta/2), sin(theta/2)cos(theta/2), cos^2(theta/2), sin^2(theta/2)]
    - Mixed PT eigenvalues: lambda_i = (1-p) * lambda_i_pure + p/4
    - Circuit degradation: measured mu_k = f_k * mu_k_model

Fitted parameters per state:
    - theta: entanglement parameter
    - p: depolarization/mixing parameter

Shared parameters:
    - f2, f3, f4: degradation factors

References:
    - Hradil, Z. Phys. Rev. A 55, R1561 (1997)
    - James et al. Phys. Rev. A 64, 052312 (2001)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .analysis import (
    theoretical_moments,
    theoretical_negativity,
    compute_negativity_newton_girard,
)


def pt_eigenvalues_pure(theta: float) -> np.ndarray:
    """PT eigenvalues for pure state |psi(theta)> = cos(theta/2)|00> + sin(theta/2)|11>"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([-s * c, s * c, c**2, s**2])


def pt_eigenvalues_mixed(theta: float, p: float) -> np.ndarray:
    """
    PT eigenvalues for depolarized state:
    rho = (1-p) |psi(theta)><psi(theta)| + p * I/4

    The PT of I/4 is I/4 (maximally mixed is PPT).
    lambda_i = (1-p) * lambda_i_pure + p/4
    """
    eigs_pure = pt_eigenvalues_pure(theta)
    return (1 - p) * eigs_pure + p / 4


def moments_from_eigenvalues(eigs: np.ndarray) -> Tuple[float, float, float]:
    """Compute mu_2, mu_3, mu_4 from eigenvalues."""
    return np.sum(eigs**2), np.sum(eigs**3), np.sum(eigs**4)


def negativity_from_eigenvalues(eigs: np.ndarray) -> float:
    """Negativity = sum of |negative eigenvalues|"""
    return -np.sum(eigs[eigs < 0])


@dataclass
class StateData:
    """Container for experimental data of a single state."""
    name: str
    theta_true: float
    mu2_exp: float
    mu3_exp: float
    mu4_exp: float
    sigma2: float
    sigma3: float
    sigma4: float
    negativity_theory: float
    is_separable: bool


class MaxLikEstimator:
    """
    Maximum Likelihood Estimator for negativity from noisy moments.

    Physical model with depolarization:
        rho = (1-p) |psi(theta)><psi(theta)| + p * I/4

    Joint optimization over:
        - Shared: f2, f3, f4 (degradation factors)
        - Per state: theta (entanglement), p (depolarization)

    Example usage:
        estimator = MaxLikEstimator.from_experiment_results(results)
        ml_results = estimator.fit()

    Attributes:
        states: List of StateData objects
        f2, f3, f4: Fitted degradation factors
        fitted_params: Dict mapping state names to (theta, p) tuples
    """

    def __init__(self):
        """Initialize the estimator."""
        self.states: List[StateData] = []
        self.f2: Optional[float] = None
        self.f3: Optional[float] = None
        self.f4: Optional[float] = None
        self.fitted_params: Dict[str, Tuple[float, float]] = {}  # {name: (theta, p)}
        self._optimization_result = None
    
    def add_state(
        self,
        name: str,
        theta_true: float,
        mu2_exp: float,
        mu3_exp: float,
        mu4_exp: float,
        sigma2: float = 0.01,
        sigma3: float = 0.01,
        sigma4: float = 0.01,
        negativity_theory: Optional[float] = None,
    ) -> None:
        """
        Add a state's experimental data to the estimator.
        
        Args:
            name: State identifier
            theta_true: True/expected theta value
            mu2_exp: Measured μ₂
            mu3_exp: Measured μ₃
            mu4_exp: Measured μ₄
            sigma2: Standard deviation of μ₂
            sigma3: Standard deviation of μ₃
            sigma4: Standard deviation of μ₄
            negativity_theory: Theoretical negativity (computed if None)
        """
        if negativity_theory is None:
            negativity_theory = theoretical_negativity(theta_true)
        
        is_separable = abs(theta_true) < 1e-6  # θ ≈ 0 is separable
        
        self.states.append(StateData(
            name=name,
            theta_true=theta_true,
            mu2_exp=mu2_exp,
            mu3_exp=mu3_exp,
            mu4_exp=mu4_exp,
            sigma2=sigma2,
            sigma3=sigma3,
            sigma4=sigma4,
            negativity_theory=negativity_theory,
            is_separable=is_separable,
        ))
    
    @classmethod
    def from_experiment_results(
        cls,
        results: Dict[str, Any],
    ) -> "MaxLikEstimator":
        """
        Create estimator from NegativityExperiment results.
        
        Args:
            results: Results dictionary from NegativityExperiment.run()
            
        Returns:
            MaxLikEstimator populated with experimental data
        """
        estimator = cls()
        
        for state_name, state_data in results.get("states", {}).items():
            # Extract moments
            moments = state_data.get("measured_moments", {})
            theoretical = state_data.get("theoretical", {})
            
            theta = state_data.get("theta", theoretical.get("theta", 0.0))
            if theta is None:
                theta = theoretical.get("theta", 0.0)
            
            estimator.add_state(
                name=state_name,
                theta_true=theta,
                mu2_exp=moments.get("mu_2", 1.0),
                mu3_exp=moments.get("mu_3", 1.0),
                mu4_exp=moments.get("mu_4", 1.0),
                sigma2=moments.get("mu_2_std", 0.01),
                sigma3=moments.get("mu_3_std", 0.01),
                sigma4=moments.get("mu_4_std", 0.01),
                negativity_theory=theoretical.get("negativity", 0.0),
            )
        
        return estimator
    
    def _moments_model(self, theta: float, p: float) -> Tuple[float, float, float]:
        """
        Compute model moments for depolarized state.
        rho = (1-p) |psi(theta)><psi(theta)| + p * I/4
        """
        eigs = pt_eigenvalues_mixed(theta, p)
        return moments_from_eigenvalues(eigs)

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for optimization.

        params = [f2, f3, f4, theta_1, p_1, theta_2, p_2, ..., theta_n, p_n]
        """
        f2, f3, f4 = params[0:3]
        state_params = params[3:]  # [theta_1, p_1, theta_2, p_2, ...]

        # Bounds check for degradation factors
        if f2 <= 0 or f3 <= 0 or f4 <= 0:
            return 1e10
        if f2 > 1.5 or f3 > 1.5 or f4 > 1.5:
            return 1e10

        nll = 0.0
        for i, state in enumerate(self.states):
            theta = state_params[2 * i]
            p = state_params[2 * i + 1]

            # Bounds check for state params
            if p < 0 or p > 1:
                return 1e10
            if theta < -0.1 or theta > np.pi + 0.1:
                return 1e10

            # Model moments for depolarized state
            mu2_th, mu3_th, mu4_th = self._moments_model(theta, p)

            # Apply degradation: measured = f * model
            mu2_model = f2 * mu2_th
            mu3_model = f3 * mu3_th
            mu4_model = f4 * mu4_th

            # NLL contribution (Gaussian likelihood)
            nll += 0.5 * ((state.mu2_exp - mu2_model)**2 / max(state.sigma2**2, 1e-10))
            nll += 0.5 * ((state.mu3_exp - mu3_model)**2 / max(state.sigma3**2, 1e-10))
            nll += 0.5 * ((state.mu4_exp - mu4_model)**2 / max(state.sigma4**2, 1e-10))

        return nll
    
    def _get_calibration_from_separable(self) -> Tuple[float, float, float]:
        """Get exact calibration factors from separable state if available."""
        for state in self.states:
            if state.is_separable:
                # For |00⟩: μ₂ = μ₃ = μ₄ = 1
                return state.mu2_exp, state.mu3_exp, state.mu4_exp
        return 1.0, 1.0, 1.0
    
    def fit(
        self,
        method: str = "differential_evolution",
        maxiter: int = 2000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Perform maximum likelihood estimation with depolarization model.

        Fits: f2, f3, f4 (shared) + theta_i, p_i (per state)

        Args:
            method: Optimization method ("differential_evolution" or "SLSQP")
            maxiter: Maximum iterations
            seed: Random seed for reproducibility

        Returns:
            Dictionary with fitted parameters and results
        """
        if not self.states:
            raise ValueError("No states added. Use add_state() or from_experiment_results().")

        n_states = len(self.states)

        # Initial guess from separable state calibration
        f2_init, f3_init, f4_init = self._get_calibration_from_separable()

        # Initial state params: [theta_1, p_1, theta_2, p_2, ...]
        state_param_inits = []
        for s in self.states:
            state_param_inits.append(s.theta_true)  # theta
            state_param_inits.append(0.1)  # p (small initial depolarization)

        x0 = np.array([f2_init, f3_init, f4_init] + state_param_inits)

        # Bounds: [f2, f3, f4, theta_1, p_1, theta_2, p_2, ...]
        bounds = [
            (0.5, 1.2),  # f2
            (0.3, 1.2),  # f3
            (0.2, 1.2),  # f4
        ]
        for state in self.states:
            # theta bounds
            theta_min = max(-0.1, state.theta_true - 0.5)
            theta_max = min(np.pi + 0.1, state.theta_true + 0.5)
            bounds.append((theta_min, theta_max))
            # p bounds (depolarization)
            bounds.append((0.0, 0.8))  # p in [0, 0.8]

        # Optimize
        if method == "differential_evolution":
            result = differential_evolution(
                self._negative_log_likelihood,
                bounds,
                seed=seed,
                maxiter=maxiter,
                tol=1e-12,
                polish=True,
            )
        else:
            result = minimize(
                self._negative_log_likelihood,
                x0,
                method=method,
                bounds=bounds,
                options={"maxiter": maxiter},
            )

        self._optimization_result = result

        # Extract results
        self.f2, self.f3, self.f4 = result.x[0:3]
        state_params = result.x[3:]

        self.fitted_params = {}
        for i, state in enumerate(self.states):
            theta_fit = state_params[2 * i]
            p_fit = state_params[2 * i + 1]
            self.fitted_params[state.name] = (theta_fit, p_fit)

        # Compute negativities
        results = {
            "degradation_factors": {
                "f2": self.f2,
                "f3": self.f3,
                "f4": self.f4,
            },
            "calibration_exact": {
                "f2": f2_init,
                "f3": f3_init,
                "f4": f4_init,
            },
            "optimization": {
                "success": result.success,
                "nll": result.fun,
                "message": getattr(result, "message", ""),
            },
            "states": {},
        }

        for i, state in enumerate(self.states):
            theta_fit = state_params[2 * i]
            p_fit = state_params[2 * i + 1]

            # Negativity from fitted depolarized state
            eigs_fit = pt_eigenvalues_mixed(theta_fit, p_fit)
            neg_ml = negativity_from_eigenvalues(eigs_fit)

            # Also compute via exact correction + Newton-Girard for comparison
            mu2_corr = state.mu2_exp / f2_init if f2_init > 0 else state.mu2_exp
            mu3_corr = state.mu3_exp / f3_init if f3_init > 0 else state.mu3_exp
            mu4_corr = state.mu4_exp / f4_init if f4_init > 0 else state.mu4_exp
            neg_exact = compute_negativity_newton_girard(mu2_corr, mu3_corr, mu4_corr)

            results["states"][state.name] = {
                "theta_true": state.theta_true,
                "theta_fit": theta_fit,
                "p_fit": p_fit,
                "theta_diff_degrees": np.degrees(theta_fit - state.theta_true),
                "negativity_theory": state.negativity_theory,
                "negativity_maxlik": neg_ml,
                "negativity_exact_correction": neg_exact,
                "error_maxlik": abs(neg_ml - state.negativity_theory),
                "error_exact": abs(neg_exact - state.negativity_theory),
            }

        # Summary statistics
        ml_errors = [d["error_maxlik"] for d in results["states"].values()]
        exact_errors = [d["error_exact"] for d in results["states"].values()]

        results["summary"] = {
            "mean_error_maxlik": np.mean(ml_errors),
            "mean_error_exact": np.mean(exact_errors),
            "improvement_percent": 100 * (np.mean(exact_errors) - np.mean(ml_errors)) / max(np.mean(exact_errors), 1e-10),
        }

        return results
    
    def print_results(self, results: Optional[Dict] = None) -> None:
        """Print formatted results."""
        if results is None:
            results = self.fit()

        print("=" * 80)
        print("MAXIMUM LIKELIHOOD ESTIMATION (Depolarization Model)")
        print("Model: rho = (1-p) |psi(theta)><psi(theta)| + p * I/4")
        print("=" * 80)
        print()

        print("Degradation Factors:")
        df = results["degradation_factors"]
        cal = results["calibration_exact"]
        print(f"  f2 = {df['f2']:.6f}  (exact: {cal['f2']:.6f})")
        print(f"  f3 = {df['f3']:.6f}  (exact: {cal['f3']:.6f})")
        print(f"  f4 = {df['f4']:.6f}  (exact: {cal['f4']:.6f})")
        print()

        print(f"{'State':<15} {'theta':>8} {'p':>8} {'N_theory':>10} {'N_ML':>10} {'Err_ML':>10}")
        print("-" * 70)

        for name, data in results["states"].items():
            print(f"{name:<15} {np.degrees(data['theta_fit']):>7.1f}d {data['p_fit']:>8.3f} "
                  f"{data['negativity_theory']:>10.4f} "
                  f"{data['negativity_maxlik']:>10.4f} "
                  f"{data['error_maxlik']:>10.4f}")

        print("-" * 70)
        summary = results["summary"]
        print(f"Mean error: MaxLik = {summary['mean_error_maxlik']:.4f}, "
              f"Exact = {summary['mean_error_exact']:.4f}")
        print(f"MaxLik improvement: {summary['improvement_percent']:+.1f}%")


def run_maxlik_analysis(
    experiment_results: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run MaxLik analysis on experiment results.
    
    Args:
        experiment_results: Results from NegativityExperiment.run()
        verbose: Whether to print results
        
    Returns:
        MaxLik analysis results
    """
    estimator = MaxLikEstimator.from_experiment_results(experiment_results)
    results = estimator.fit()
    
    if verbose:
        estimator.print_results(results)
    
    return results
