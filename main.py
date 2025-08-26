import argparse
import sys
from pathlib import Path
from activation_steering import main, ActivationSteeringExperiment

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run activation steering experiment or generate plots from results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                        # Run full experiment with layer analysis
  python main.py --plot-only                           # Plot from experiment_results.json
  python main.py --plot-only --input results.json     # Plot from custom file
  python main.py --plot-only --output my_plots        # Plot to custom directory
"""
    )
    
    parser.add_argument(
        "--plot-only", "-p",
        action="store_true",
        help="Only generate plots from existing results, don't run experiment"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="experiment_results.json",
        help="Input JSON file for plotting (default: experiment_results.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.plot_only:
        # Plot-only mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Results file '{args.input}' not found.")
            print("Make sure you have run the experiment first to generate results.")
            sys.exit(1)
        
        print(f"Loading results from: {args.input}")
        print(f"Output directory: {args.output}")
        
        try:
            ActivationSteeringExperiment.load_and_plot_results(
                results_file=args.input,
                save_dir=args.output
            )
            print(f"✓ Plots generated successfully in '{args.output}/' directory")
        except Exception as e:
            print(f"Error generating plots: {e}")
            sys.exit(1)
    else:
        # Run full experiment (now includes layer analysis by default)
        main()