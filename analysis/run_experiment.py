#!/usr/bin/env python3
"""
Main experiment runner script.

This script provides a command-line interface for running negativity
measurements on IBM Quantum hardware or simulators.

Usage:
    # Simulation mode (default - uses fake Kingston with calibration noise)
    python run_experiment.py
    
    # Ideal simulator (no noise)
    python run_experiment.py --backend aer
    
    # Real IBM Quantum hardware
    python run_experiment.py --backend ibm_kingston --api-key YOUR_KEY
    
    # Custom states
    python run_experiment.py --states bell_phi_plus product_00 param_theta
    
    # Custom theta values
    python run_experiment.py --theta 0.0 0.5 1.0 1.57

Examples:
    # Quick test with Bell states only
    python run_experiment.py --states bell_phi_plus bell_phi_minus --shots 1000
    
    # Full experiment on real hardware
    python run_experiment.py --backend ibm_kingston --api-key YOUR_KEY --shots 100000
    
    # Run validation only
    python run_experiment.py --validate-only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run negativity measurement experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Backend options
    parser.add_argument(
        "--backend",
        type=str,
        default="fake_kingston",
        help="Backend to use: 'fake_kingston' (default), 'aer', or IBM backend name",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="IBM Quantum API key (required for real hardware)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="ibm-q/open/main",
        help="IBM Quantum instance (default: ibm-q/open/main)",
    )
    
    # Experiment options
    parser.add_argument(
        "--shots",
        type=int,
        default=100000,
        help="Number of shots per circuit (default: 100000)",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="States to measure (default: all Bell, product, and parameterized)",
    )
    parser.add_argument(
        "--theta",
        nargs="+",
        type=float,
        default=None,
        help="Theta values for parameterized states (default: 0, π/6, π/4, π/3, π/2)",
    )
    parser.add_argument(
        "--no-purity",
        action="store_true",
        help="Skip purity measurements",
    )
    parser.add_argument(
        "--no-mitigation",
        action="store_true",
        help="Disable M3 error mitigation",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    # Validation
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation tests only, no experiment",
    )
    parser.add_argument(
        "--no-maxlik",
        action="store_true",
        help="Skip MaxLik degradation correction (not recommended)",
    )
    
    args = parser.parse_args()
    
    # Run validation only if requested
    if args.validate_only:
        from negativity_si.validation import run_validation
        print("Running validation tests...")
        results = run_validation(verbose=True, full=True)
        
        # Exit with error if any tests failed
        total_failed = sum(r.failed for r in results.values())
        sys.exit(1 if total_failed > 0 else 0)
    
    # Import here to avoid slow imports for --help
    from negativity_si import NegativityExperiment
    from negativity_si.maxlik import run_maxlik_analysis
    
    # Check API key for real hardware
    if args.backend not in ["fake_kingston", "aer"] and args.api_key is None:
        print(f"Error: API key required for backend '{args.backend}'")
        print("Use --api-key YOUR_KEY or use --backend fake_kingston for simulation")
        sys.exit(1)
    
    # Initialize experiment
    try:
        exp = NegativityExperiment(
            backend_name=args.backend,
            api_key=args.api_key,
            instance=args.instance,
            shots=args.shots,
            use_mitigation=not args.no_mitigation,
        )
    except Exception as e:
        print(f"Error initializing experiment: {e}")
        sys.exit(1)
    
    # Prepare theta values
    theta_values = args.theta
    if theta_values is None:
        theta_values = [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    
    # Run experiment
    try:
        results = exp.run(
            states=args.states,
            theta_values=theta_values,
            include_purity=not args.no_purity,
        )
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)
    
    # Print summary
    if not args.quiet:
        exp.print_summary()
    
    # Save results
    output_path = exp.save_results(args.output)
    
    print(f"\n✓ Experiment complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
