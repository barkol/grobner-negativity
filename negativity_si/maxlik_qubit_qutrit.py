"""
Maximum Likelihood Estimation for Qubit-Qutrit (2x3) Negativity.

Physical Model (with depolarization):
    rho = (1-p) |psi(theta)><psi(theta)| + p * I/6

    - State: |psi(theta)> = cos(theta/2)|0,0> + sin(theta/2)|1,1>
    - Pure PT eigenvalues: [c^2, s^2, -cs, cs, 0, 0] where c=cos(theta/2), s=sin(theta/2)
    - Mixed PT eigenvalues: lambda_i = (1-p) * lambda_i_pure + p/6
    - Negativity: N = sum of |negative eigenvalues|
    - Circuit degradation: measured mu_k = f_k * mu_k_model

Joint optimization over:
    - Degradation factors (f2, f3, f4, f5, f6) - shared across all states
    - Per state: theta (entanglement), p (depolarization)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class QubitQutritStateData:
    """Container for experimental data of a single qubit-qutrit state."""
    name: str
    theta_true: float
    mu2_exp: float
    mu3_exp: float
    mu4_exp: float
    mu5_exp: float
    mu6_exp: float
    sigma2: float
    sigma3: float
    sigma4: float
    sigma5: float
    sigma6: float
    negativity_theory: float
    is_separable: bool


def pt_eigenvalues_pure_qubit_qutrit(theta: float) -> np.ndarray:
    """
    PT eigenvalues for pure state |psi(theta)> = cos(theta/2)|0,0> + sin(theta/2)|1,1>

    Returns: [c^2, s^2, -cs, cs, 0, 0] sorted
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.sort([c**2, s**2, -c*s, c*s, 0, 0])


def pt_eigenvalues_mixed_qubit_qutrit(theta: float, p: float) -> np.ndarray:
    """
    PT eigenvalues for depolarized qubit-qutrit state:
    rho = (1-p) |psi(theta)><psi(theta)| + p * I/6

    lambda_i = (1-p) * lambda_i_pure + p/6
    """
    eigs_pure = pt_eigenvalues_pure_qubit_qutrit(theta)
    return (1 - p) * eigs_pure + p / 6


def moments_from_eigenvalues_6(eigs: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Compute mu_2, mu_3, mu_4, mu_5, mu_6 from eigenvalues."""
    return (
        np.sum(eigs**2),
        np.sum(eigs**3),
        np.sum(eigs**4),
        np.sum(eigs**5),
        np.sum(eigs**6),
    )


def negativity_from_eigenvalues_6(eigs: np.ndarray) -> float:
    """Negativity = sum of |negative eigenvalues|"""
    return -np.sum(eigs[eigs < 0])


def theoretical_negativity_qubit_qutrit(theta: float) -> float:
    """Negativity for pure state = |cs| = sin(theta)/2"""
    return np.abs(np.sin(theta / 2) * np.cos(theta / 2))


class MaxLikEstimatorQubitQutrit:
    """
    Maximum Likelihood Estimator for qubit-qutrit negativity.

    Physical model with depolarization:
        rho = (1-p) |psi(theta)><psi(theta)| + p * I/6

    Joint optimization over:
        - Shared: f2, f3, f4, f5, f6 (degradation factors)
        - Per state: theta (entanglement), p (depolarization)
    """

    def __init__(self):
        self.states: List[QubitQutritStateData] = []
        self.f2: Optional[float] = None
        self.f3: Optional[float] = None
        self.f4: Optional[float] = None
        self.f5: Optional[float] = None
        self.f6: Optional[float] = None
        self.fitted_params: Dict[str, Tuple[float, float]] = {}  # {name: (theta, p)}
        self._optimization_result = None

    def add_state(
        self,
        name: str,
        theta_true: float,
        mu2_exp: float,
        mu3_exp: float,
        mu4_exp: float,
        mu5_exp: float,
        mu6_exp: float,
        sigma2: float = 0.01,
        sigma3: float = 0.01,
        sigma4: float = 0.01,
        sigma5: float = 0.01,
        sigma6: float = 0.01,
        negativity_theory: Optional[float] = None,
    ) -> None:
        """Add a state's experimental data."""
        if negativity_theory is None:
            negativity_theory = theoretical_negativity_qubit_qutrit(theta_true)

        is_separable = abs(theta_true) < 1e-6

        self.states.append(QubitQutritStateData(
            name=name,
            theta_true=theta_true,
            mu2_exp=mu2_exp,
            mu3_exp=mu3_exp,
            mu4_exp=mu4_exp,
            mu5_exp=mu5_exp,
            mu6_exp=mu6_exp,
            sigma2=sigma2,
            sigma3=sigma3,
            sigma4=sigma4,
            sigma5=sigma5,
            sigma6=sigma6,
            negativity_theory=negativity_theory,
            is_separable=is_separable,
        ))

    @classmethod
    def from_experiment_results(cls, results: Dict[str, Any]) -> "MaxLikEstimatorQubitQutrit":
        """Create estimator from QubitQutritExperiment results."""
        estimator = cls()

        for state_name, state_data in results.get("states", {}).items():
            moments = state_data.get("measured_moments", {})
            theoretical = state_data.get("theoretical", {})

            theta = state_data.get("theta", theoretical.get("theta", 0.0))
            if theta is None:
                theta = 0.0

            estimator.add_state(
                name=state_name,
                theta_true=theta,
                mu2_exp=moments.get("mu_2", 1.0),
                mu3_exp=moments.get("mu_3", 1.0),
                mu4_exp=moments.get("mu_4", 1.0),
                mu5_exp=moments.get("mu_5", 0.0),
                mu6_exp=moments.get("mu_6", 0.0),
                sigma2=moments.get("mu_2_std", 0.01),
                sigma3=moments.get("mu_3_std", 0.01),
                sigma4=moments.get("mu_4_std", 0.01),
                sigma5=moments.get("mu_5_std", 0.01),
                sigma6=moments.get("mu_6_std", 0.01),
                negativity_theory=theoretical.get("negativity", 0.0),
            )

        return estimator

    def _get_calibration_from_separable(self) -> Tuple[float, float, float, float, float]:
        """Get exact calibration factors from separable state if available."""
        for state in self.states:
            if state.is_separable:
                # For |00>: theoretical moments at theta=0, p=0
                eigs = pt_eigenvalues_pure_qubit_qutrit(0.0)
                mu2_th, mu3_th, mu4_th, mu5_th, mu6_th = moments_from_eigenvalues_6(eigs)
                return (
                    state.mu2_exp / mu2_th if mu2_th > 0 else 1.0,
                    state.mu3_exp / mu3_th if mu3_th > 0 else 1.0,
                    state.mu4_exp / mu4_th if mu4_th > 0 else 1.0,
                    state.mu5_exp / mu5_th if mu5_th > 0 else 1.0,
                    state.mu6_exp / mu6_th if mu6_th > 0 else 1.0,
                )
        return 1.0, 1.0, 1.0, 1.0, 1.0

    def _moments_model(self, theta: float, p: float) -> Tuple[float, float, float, float, float]:
        """Compute model moments for depolarized qubit-qutrit state."""
        eigs = pt_eigenvalues_mixed_qubit_qutrit(theta, p)
        return moments_from_eigenvalues_6(eigs)

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood.
        params = [f2, f3, f4, f5, f6, theta_1, p_1, theta_2, p_2, ...]
        """
        f2, f3, f4, f5, f6 = params[0:5]
        state_params = params[5:]  # [theta_1, p_1, theta_2, p_2, ...]

        # Bounds check for degradation factors
        if any(f <= 0 for f in [f2, f3, f4, f5, f6]):
            return 1e10
        if any(f > 1.5 for f in [f2, f3, f4, f5, f6]):
            return 1e10

        nll = 0.0
        for i, state in enumerate(self.states):
            theta = state_params[2 * i]
            p = state_params[2 * i + 1]

            # Bounds check
            if p < 0 or p > 1:
                return 1e10
            if theta < -0.1 or theta > np.pi + 0.1:
                return 1e10

            # Model moments for depolarized state
            mu2_th, mu3_th, mu4_th, mu5_th, mu6_th = self._moments_model(theta, p)

            # Apply degradation: measured = f * model
            mu2_model = f2 * mu2_th
            mu3_model = f3 * mu3_th
            mu4_model = f4 * mu4_th
            mu5_model = f5 * mu5_th
            mu6_model = f6 * mu6_th

            # NLL contribution (Gaussian likelihood)
            nll += 0.5 * ((state.mu2_exp - mu2_model)**2 / max(state.sigma2**2, 1e-10))
            nll += 0.5 * ((state.mu3_exp - mu3_model)**2 / max(state.sigma3**2, 1e-10))
            nll += 0.5 * ((state.mu4_exp - mu4_model)**2 / max(state.sigma4**2, 1e-10))
            nll += 0.5 * ((state.mu5_exp - mu5_model)**2 / max(state.sigma5**2, 1e-10))
            nll += 0.5 * ((state.mu6_exp - mu6_model)**2 / max(state.sigma6**2, 1e-10))

        return nll

    def fit(
        self,
        method: str = "differential_evolution",
        maxiter: int = 2000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Perform maximum likelihood estimation with depolarization model.

        Fits: f2..f6 (shared) + theta_i, p_i (per state)
        """
        if not self.states:
            raise ValueError("No states added.")

        n_states = len(self.states)

        # Initial guess from separable state calibration
        f2_init, f3_init, f4_init, f5_init, f6_init = self._get_calibration_from_separable()

        # Initial state params: [theta_1, p_1, theta_2, p_2, ...]
        state_param_inits = []
        for s in self.states:
            state_param_inits.append(s.theta_true)  # theta
            state_param_inits.append(0.1)  # p (small initial depolarization)

        x0 = np.array([f2_init, f3_init, f4_init, f5_init, f6_init] + state_param_inits)

        # Bounds: [f2..f6, theta_1, p_1, theta_2, p_2, ...]
        bounds = [
            (0.3, 1.3),  # f2
            (0.2, 1.3),  # f3
            (0.1, 1.3),  # f4
            (0.05, 1.3), # f5
            (0.02, 1.3), # f6
        ]
        for state in self.states:
            theta_min = max(-0.1, state.theta_true - 0.5)
            theta_max = min(np.pi + 0.1, state.theta_true + 0.5)
            bounds.append((theta_min, theta_max))
            bounds.append((0.0, 0.8))  # p

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
        self.f2, self.f3, self.f4, self.f5, self.f6 = result.x[0:5]
        state_params = result.x[5:]

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
                "f5": self.f5,
                "f6": self.f6,
            },
            "calibration_exact": {
                "f2": f2_init,
                "f3": f3_init,
                "f4": f4_init,
                "f5": f5_init,
                "f6": f6_init,
            },
            "optimization": {
                "success": result.success,
                "nll": result.fun,
            },
            "states": {},
        }

        for i, state in enumerate(self.states):
            theta_fit = state_params[2 * i]
            p_fit = state_params[2 * i + 1]

            # Negativity from fitted depolarized state
            eigs_fit = pt_eigenvalues_mixed_qubit_qutrit(theta_fit, p_fit)
            neg_ml = negativity_from_eigenvalues_6(eigs_fit)

            results["states"][state.name] = {
                "theta_true": state.theta_true,
                "theta_fit": theta_fit,
                "p_fit": p_fit,
                "theta_diff_degrees": np.degrees(theta_fit - state.theta_true),
                "negativity_theory": state.negativity_theory,
                "negativity_maxlik": neg_ml,
                "error_maxlik": abs(neg_ml - state.negativity_theory),
            }

        # Summary
        ml_errors = [d["error_maxlik"] for d in results["states"].values()]
        results["summary"] = {
            "mean_error_maxlik": np.mean(ml_errors),
            "max_error_maxlik": np.max(ml_errors),
        }

        return results

    def print_results(self, results: Optional[Dict] = None) -> None:
        """Print formatted results."""
        if results is None:
            results = self.fit()

        print("=" * 80)
        print("QUBIT-QUTRIT MAXIMUM LIKELIHOOD (Depolarization Model)")
        print("Model: rho = (1-p) |psi(theta)><psi(theta)| + p * I/6")
        print("=" * 80)
        print()

        print("Degradation Factors:")
        df = results["degradation_factors"]
        cal = results["calibration_exact"]
        print(f"  f2 = {df['f2']:.6f}  (exact: {cal['f2']:.6f})")
        print(f"  f3 = {df['f3']:.6f}  (exact: {cal['f3']:.6f})")
        print(f"  f4 = {df['f4']:.6f}  (exact: {cal['f4']:.6f})")
        print(f"  f5 = {df['f5']:.6f}  (exact: {cal['f5']:.6f})")
        print(f"  f6 = {df['f6']:.6f}  (exact: {cal['f6']:.6f})")
        print()

        print(f"{'State':<20} {'theta':>8} {'p':>8} {'N_theory':>10} {'N_ML':>10} {'Error':>10}")
        print("-" * 75)

        for name, data in results["states"].items():
            print(f"{name:<20} {np.degrees(data['theta_fit']):>7.1f}d {data['p_fit']:>8.3f} "
                  f"{data['negativity_theory']:>10.4f} "
                  f"{data['negativity_maxlik']:>10.4f} "
                  f"{data['error_maxlik']:>10.4f}")

        print("-" * 75)
        print(f"Mean error: {results['summary']['mean_error_maxlik']:.4f}")


def run_maxlik_analysis_qubit_qutrit(
    experiment_results: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run MaxLik analysis on qubit-qutrit experiment results.
    """
    estimator = MaxLikEstimatorQubitQutrit.from_experiment_results(experiment_results)
    results = estimator.fit()

    if verbose:
        estimator.print_results(results)

    return results
