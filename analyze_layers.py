#!/usr/bin/env python3
"""
Standalone script to perform layer-wise steering analysis.
This script evaluates which transformer layers are most effective for steering.
"""

import argparse
import json
from pathlib import Path
from activation_steering import ActivationSteeringExperiment

def main():
    parser = argparse.ArgumentParser(
        description="Analyze steering effectiveness across transformer layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_layers.py                                  # Run full layer analysis
  python analyze_layers.py --challenging-size 5 --benign-size 5  # Use smaller prompt sets
  python analyze_layers.py --output layer_plots             # Save to custom directory
  python analyze_layers.py --load-existing experiment_results.json  # Use existing experiment data
"""
    )
    
    parser.add_argument(
        "--challenging-size", "-c",
        type=int,
        default=10,
        help="Number of challenging prompts to use (default: 10, reduced for efficiency)"
    )
    
    parser.add_argument(
        "--benign-size", "-b", 
        type=int,
        default=10,
        help="Number of benign prompts to use (default: 10, reduced for efficiency)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    
    parser.add_argument(
        "--load-existing", "-l",
        type=str,
        help="Load existing experiment results JSON file instead of running new experiment"
    )
    
    args = parser.parse_args()
    
    if args.load_existing:
        # Load from existing results and perform layer analysis plots only
        results_file = Path(args.load_existing)
        if not results_file.exists():
            print(f"Error: Results file '{args.load_existing}' not found.")
            return
        
        print(f"Loading existing experiment results from {args.load_existing}...")
        
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Check if layer analysis results already exist
        if 'layer_analysis_results' in saved_results:
            print("Found existing layer analysis results, generating plots...")
            experiment = ActivationSteeringExperiment()
            experiment.plot_layer_analysis(saved_results['layer_analysis_results'], args.output)
        else:
            print("No layer analysis results found in the saved file.")
            print("You need to run the full layer analysis to generate layer-specific data.")
            print("Run: python analyze_layers.py (without --load-existing)")
            return
    else:
        # Run full layer analysis
        print("=== Layer-wise Steering Analysis ===")
        print(f"Analyzing steering effectiveness across transformer layers")
        print(f"Using {args.challenging_size} challenging and {args.benign_size} benign prompts")
        print(f"Output directory: {args.output}")
        print("\nNote: This analysis is computationally intensive and may take several minutes.")
        
        try:
            # Initialize experiment
            experiment = ActivationSteeringExperiment()
            
            # Load dataset and create subsets
            print("\n1. Loading RealToxicityPrompts dataset...")
            experiment.load_rtp_dataset()
            experiment.create_subsets()
            
            # Collect activations
            print("\n2. Collecting baseline activations...")
            experiment.collect_baseline_activations()
            
            # Compute steering vectors
            print("\n3. Computing steering vectors for each layer...")
            experiment.compute_steering_vectors()
            
            # Perform layer analysis
            print("\n4. Evaluating steering effectiveness across all layers...")
            layer_analysis_results = experiment.evaluate_steering_all_layers(
                challenging_subset_size=args.challenging_size,
                benign_subset_size=args.benign_size
            )
            
            # Generate plots
            print("\n5. Generating layer analysis visualizations...")
            experiment.plot_layer_analysis(layer_analysis_results, args.output)
            
            # Save layer analysis results
            output_file = "layer_analysis_results.json"
            print(f"\n6. Saving layer analysis results to {output_file}...")
            
            results_to_save = {
                'layer_analysis_results': layer_analysis_results,
                'experiment_config': {
                    'challenging_subset_size': args.challenging_size,
                    'benign_subset_size': args.benign_size,
                    'model_name': 'microsoft/Phi-4-mini-instruct',
                    'num_layers': len(experiment.steering_vectors)
                },
                'timestamp': experiment._get_timestamp()
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            print(f"✓ Layer analysis complete!")
            print(f"✓ Plots saved to '{args.output}/' directory")
            print(f"✓ Results saved to '{output_file}'")
            print(f"\nGenerated plots:")
            print(f"  - layer_effectiveness_analysis.png/pdf: Comprehensive 6-panel layer analysis")
            print(f"  - detailed_layer_heatmap.png/pdf: Detailed layer vs alpha effectiveness heatmap")
            
        except Exception as e:
            print(f"Error during layer analysis: {e}")
            print("Layer analysis failed.")
            return

if __name__ == "__main__":
    main()