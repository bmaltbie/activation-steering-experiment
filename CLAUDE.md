# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI safety research project implementing activation steering for toxicity reduction in the Qwen3-8B model. The project uses contrastive activation addition (CAA) to compute steering vectors that can reduce toxic outputs while preserving model capability on benign prompts.

## Development Environment

- **Python Version**: 3.13.1
- **Virtual Environment**: Located in `.venv/` directory
- **Key Dependencies**: transformers, torch, datasets, tqdm, numpy

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install required dependencies (already installed)
pip install datasets torch tqdm transformers
```

### Running the Experiment
```bash
# Test setup first (recommended)
source .venv/bin/activate && python test_setup.py

# Run full experiment
source .venv/bin/activate && python main.py

# Run specific components
source .venv/bin/activate && python activation_steering.py
```

### Development Tasks
```bash
# Check installed packages
source .venv/bin/activate && pip list

# Monitor GPU usage during experiments
nvidia-smi -l 1  # if using GPU
```

## Code Architecture

The project implements a comprehensive activation steering pipeline:

- **main.py**: Entry point that imports and runs the experiment
- **activation_steering.py**: Complete implementation of the steering experiment
  - `ActivationSteeringExperiment`: Main class implementing the full pipeline
  - Data loading and subset creation (RealToxicityPrompts dataset)
  - Baseline evaluation with toxicity scoring
  - Activation collection using forward hooks
  - Steering vector computation via CAA
  - Alpha-sweep evaluation across steering strengths
- **test_setup.py**: Verification script to test setup before running experiments

## Experimental Pipeline

1. **Dataset Preparation**: Load RealToxicityPrompts and create challenging/benign subsets
2. **Baseline Evaluation**: Generate completions without steering (temp=0.7, top_p=0.9)
3. **Activation Collection**: Capture residual stream activations across all layers
4. **Steering Vector Computation**: Use CAA to compute layer-wise steering directions
5. **Steering Evaluation**: Test across alpha values [-1.0, -0.5, 0, 0.25, 0.5, 1.0, 1.5, 2.0]

## Key Technical Details

- **Model**: Qwen/Qwen3-8B (8-billion parameter causal language model)
- **Toxicity Scorer**: unitary/unbiased-toxic-roberta for MeanToxicity evaluation
- **Activation Capture**: Forward hooks on transformer layers with mean-pooling over last 32 tokens
- **Steering Method**: Contrastive activation addition applied during inference
- **Hardware**: Optimized for GPU use with automatic fallback to CPU

## Project Structure

```
├── .venv/                    # Python virtual environment
├── .idea/                    # JetBrains IDE configuration  
├── .git/                     # Git repository
├── main.py                   # Main experiment entry point
├── activation_steering.py    # Core experiment implementation
├── test_setup.py            # Setup verification script
├── CLAUDE.md                # This documentation
└── README.md                # Experiment instructions
```

## Results and Output

- **experiment_results.json**: Complete experimental results with baseline and steering data
- **partial_experiment_results.json**: Saved if experiment is interrupted
- **failed_experiment_results.json**: Saved if experiment fails mid-execution

## Notes for Development

- Always activate virtual environment before running any Python commands
- The model requires significant computational resources (8B parameters)
- GPU usage is recommended but CPU fallback is implemented
- Results are automatically saved with comprehensive metrics
- The experiment implements proper error handling and partial result saving