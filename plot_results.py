#!/usr/bin/env python3
"""
Standalone script to generate plots from saved experiment results.
This allows plotting without re-running the full experiment.
"""

import argparse
import sys
from pathlib import Path
from activation_steering import ActivationSteeringExperiment

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from saved experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_results.py                                    # Use default experiment_results.json
  python plot_results.py --input results.json              # Use custom results file
  python plot_results.py --input results.json --output my_plots  # Custom output directory
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="experiment_results.json",
        help="Path to the JSON results file (default: experiment_results.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Results file '{args.input}' not found.")
        print("Make sure you have run the experiment first to generate results.")
        sys.exit(1)
    
    print(f"Loading results from: {args.input}")
    print(f"Output directory: {args.output}")
    
    try:
        # Use the standalone plotting function
        ActivationSteeringExperiment.load_and_plot_results(
            results_file=args.input,
            save_dir=args.output
        )
        
        print(f"✓ Plots generated successfully in '{args.output}/' directory")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()