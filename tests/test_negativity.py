"""
Tests for negativity_si package.

Run with: pytest tests/
"""

import numpy as np
import pytest


class TestAnalysis:
    """Tests for analysis module."""
    
    def test_newton_girard_elementary(self):
        """Test Newton-Girard elementary symmetric polynomials."""
        from negativity_si.analysis import newton_girard_elementary
        
        # Test with known eigenvalues
        eigs = np.array([0.25, 0.25, 0.25, 0.25])  # Maximally mixed
        mu_2 = np.sum(eigs**2)  # 0.25
        mu_3 = np.sum(eigs**3)  # 0.0625
        mu_4 = np.sum(eigs**4)  # 0.015625
        
        e1, e2, e3, e4 = newton_girard_elementary(mu_2, mu_3, mu_4)
        
        assert abs(e1 - 1.0) < 1e-10
        assert abs(e2 - 0.375) < 1e-10  # Sum of pairs
    
    def test_reconstruct_eigenvalues(self):
        """Test eigenvalue reconstruction."""
        from negativity_si.analysis import reconstruct_eigenvalues
        
        # Bell state eigenvalues: [-0.5, 0.5, 0.5, 0.5]
        mu_2 = 0.5
        mu_3 = 0.25
        mu_4 = 0.125
        
        eigs = reconstruct_eigenvalues(mu_2, mu_3, mu_4)
        
        assert len(eigs) == 4
        assert abs(eigs[0] - (-0.5)) < 1e-8
    
    def test_negativity_bell_state(self):
        """Test negativity computation for Bell state."""
        from negativity_si.analysis import compute_negativity_newton_girard
        
        # Bell state moments
        mu_2 = 0.5
        mu_3 = 0.25
        mu_4 = 0.125
        
        neg = compute_negativity_newton_girard(mu_2, mu_3, mu_4)
        
        assert abs(neg - 0.5) < 1e-8
    
    def test_negativity_product_state(self):
        """Test negativity for product state (should be zero)."""
        from negativity_si.analysis import compute_negativity_newton_girard
        
        # Product state |00⟩: μ_k = 1 for all k
        neg = compute_negativity_newton_girard(1.0, 1.0, 1.0)
        
        assert abs(neg) < 1e-10
    
    def test_theoretical_moments(self):
        """Test theoretical moment computation."""
        from negativity_si.analysis import theoretical_moments
        
        # At θ = π/2 (Bell state)
        mu_2, mu_3, mu_4 = theoretical_moments(np.pi / 2)
        
        assert abs(mu_2 - 0.5) < 1e-10
        assert abs(mu_3 - 0.25) < 1e-10
        assert abs(mu_4 - 0.125) < 1e-10
    
    def test_theoretical_negativity(self):
        """Test theoretical negativity computation."""
        from negativity_si.analysis import theoretical_negativity
        
        # At θ = π/2: N = sin(π/2)/2 = 0.5
        neg = theoretical_negativity(np.pi / 2)
        assert abs(neg - 0.5) < 1e-10
        
        # At θ = 0: N = 0
        neg = theoretical_negativity(0.0)
        assert abs(neg) < 1e-10


class TestStates:
    """Tests for states module."""
    
    def test_theoretical_values_bell(self):
        """Test theoretical values for Bell states."""
        from negativity_si.states import get_theoretical_values
        
        values = get_theoretical_values("bell_phi_plus")
        
        assert values["negativity"] == 0.5
        assert values["purity"] == 1.0
        assert values["mu_2"] == 0.5
    
    def test_theoretical_values_product(self):
        """Test theoretical values for product states."""
        from negativity_si.states import get_theoretical_values
        
        values = get_theoretical_values("product_00")
        
        assert values["negativity"] == 0.0
        assert values["purity"] == 1.0
        assert values["mu_2"] == 1.0
    
    def test_theoretical_values_parameterized(self):
        """Test theoretical values for parameterized states."""
        from negativity_si.states import get_theoretical_values
        
        # At θ = π/4
        values = get_theoretical_values("param_theta", np.pi / 4)
        
        assert 0 < values["negativity"] < 0.5
        assert values["purity"] == 1.0


class TestCircuits:
    """Tests for circuits module."""
    
    def test_mu2_circuit_creation(self):
        """Test μ₂ circuit creation."""
        from negativity_si.circuits import create_mu2_circuit
        
        qc = create_mu2_circuit("bell_phi_plus")
        
        assert qc.num_qubits == 5
        assert qc.num_clbits == 1
    
    def test_mu3_circuit_creation(self):
        """Test μ₃ circuit creation."""
        from negativity_si.circuits import create_mu3_circuit
        
        qc = create_mu3_circuit("bell_phi_plus")
        
        assert qc.num_qubits == 7
        assert qc.num_clbits == 1
    
    def test_mu4_circuit_creation(self):
        """Test μ₄ circuit creation."""
        from negativity_si.circuits import create_mu4_circuit
        
        qc = create_mu4_circuit("bell_phi_plus")
        
        assert qc.num_qubits == 9
        assert qc.num_clbits == 1
    
    def test_purity_circuit_creation(self):
        """Test purity circuit creation."""
        from negativity_si.circuits import create_purity_circuit
        
        qc = create_purity_circuit("bell_phi_plus")
        
        assert qc.num_qubits == 5
        assert qc.num_clbits == 1
    
    def test_extract_moment_from_counts(self):
        """Test moment extraction from counts."""
        from negativity_si.circuits import extract_moment_from_counts
        
        # 75% |0⟩ outcomes
        counts = {"0": 7500, "1": 2500}
        moment, std = extract_moment_from_counts(counts, 10000)
        
        # μ = 2*0.75 - 1 = 0.5
        assert abs(moment - 0.5) < 1e-10


class TestCalibration:
    """Tests for calibration module."""
    
    def test_parse_calibration(self):
        """Test calibration file parsing."""
        from negativity_si.calibration import parse_calibration_csv
        
        calibrations, coupling_map = parse_calibration_csv()
        
        assert len(calibrations) > 0
        assert len(coupling_map) > 0
    
    def test_create_fake_backend(self):
        """Test fake backend creation."""
        from negativity_si.calibration import create_fake_kingston_backend
        
        backend = create_fake_kingston_backend()
        
        assert backend is not None
        # Check noise model is attached
        assert backend.options.noise_model is not None


class TestMaxLik:
    """Tests for MaxLik estimator."""
    
    def test_maxlik_creation(self):
        """Test MaxLik estimator creation."""
        from negativity_si.maxlik import MaxLikEstimator
        
        estimator = MaxLikEstimator()
        estimator.add_state(
            name="test",
            theta_true=np.pi/4,
            mu2_exp=0.7,
            mu3_exp=0.5,
            mu4_exp=0.4,
        )
        
        assert len(estimator.states) == 1
    
    def test_maxlik_fit(self):
        """Test MaxLik fitting with synthetic data."""
        from negativity_si.maxlik import MaxLikEstimator
        from negativity_si.analysis import theoretical_moments
        
        estimator = MaxLikEstimator()
        
        # Add states with ideal moments (no noise)
        for theta in [0.0, np.pi/4, np.pi/2]:
            mu_2, mu_3, mu_4 = theoretical_moments(theta)
            estimator.add_state(
                name=f"theta_{theta:.2f}",
                theta_true=theta,
                mu2_exp=mu_2,
                mu3_exp=mu_3,
                mu4_exp=mu_4,
            )
        
        results = estimator.fit()
        
        # Degradation factors should be close to 1.0
        assert abs(results["degradation_factors"]["f2"] - 1.0) < 0.1


class TestValidation:
    """Tests for validation module."""
    
    def test_run_validation(self):
        """Test validation runner."""
        from negativity_si.validation import run_validation
        
        results = run_validation(verbose=False, full=False)
        
        assert len(results) > 0
        # All basic tests should pass
        total_failed = sum(r.failed for r in results.values())
        assert total_failed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
