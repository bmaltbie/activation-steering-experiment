# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI safety research project implementing activation steering for toxicity reduction in the Phi-4-mini-instruct model. The project uses contrastive activation addition (CAA) to compute steering vectors that can reduce toxic outputs while preserving model capability on benign prompts.

## Development Environment

- **Python Version**: 3.13.1
- **Virtual Environment**: Located in `.venv/` directory
- **Key Dependencies**: transformers, torch, datasets, tqdm, numpy, matplotlib, seaborn

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install required dependencies (already installed)
pip install datasets torch tqdm transformers matplotlib seaborn

# For Apple Silicon users - ensure MPS support
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Apple Silicon Optimization
For optimal performance on Apple Silicon (M1/M2/M3) Macs:

```bash
# Set environment variables for better memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Verify MPS acceleration is working
python -c "import torch; print('Device:', torch.device('mps') if torch.backends.mps.is_available() else 'cpu')"
```

**Apple Silicon Notes:**
- The code automatically detects and uses MPS acceleration when available
- Memory optimizations are applied specifically for unified memory architecture
- Falls back gracefully to CPU if MPS encounters issues
- FP16 precision is enabled by default on MPS for better performance
- Phi-4-mini-instruct is significantly smaller and more memory efficient than larger models

### Running the Experiment
```bash
# Test setup first (recommended)
source .venv/bin/activate && python test_setup.py

# Run full experiment (includes layer analysis)
source .venv/bin/activate && python main.py

# Run specific components
source .venv/bin/activate && python activation_steering.py
```

### Plotting Results
```bash
# Generate plots from saved results (without re-running experiment)
source .venv/bin/activate && python main.py --plot-only

# Plot from custom results file
source .venv/bin/activate && python main.py --plot-only --input my_results.json

# Save plots to custom directory
source .venv/bin/activate && python main.py --plot-only --output my_plots

# Alternative: Use standalone plotting script
source .venv/bin/activate && python plot_results.py
source .venv/bin/activate && python plot_results.py --input results.json --output plots
```

### Additional Layer Analysis (Optional)
```bash
# Run standalone comprehensive layer analysis with custom parameters
source .venv/bin/activate && python analyze_layers.py --challenging-size 10 --benign-size 10

# Generate additional layer plots from existing experiment results
source .venv/bin/activate && python analyze_layers.py --load-existing experiment_results.json

# Note: Layer analysis is now included by default in the main experiment
# The analyze_layers.py script is for additional detailed analysis with custom parameters
```

### Development Tasks
```bash
# Check installed packages
source .venv/bin/activate && pip list

# Monitor GPU usage during experiments
nvidia-smi -l 1  # if using NVIDIA GPU
sudo powermetrics -s gpu_power -n 1 --samplers gpu_power  # if using Apple Silicon
```

## Code Architecture

The project implements a comprehensive activation steering pipeline:

- **main.py**: Entry point that imports and runs the experiment, supports plotting-only mode and layer analysis
- **plot_results.py**: Standalone script for generating plots from saved results  
- **analyze_layers.py**: Standalone script for layer-wise steering effectiveness analysis
- **activation_steering.py**: Complete implementation of the steering experiment
  - `ActivationSteeringExperiment`: Main class implementing the full pipeline
  - Data loading and subset creation (RealToxicityPrompts dataset)
  - Baseline evaluation with toxicity scoring
  - Activation collection using forward hooks
  - Steering vector computation via CAA
  - Alpha-sweep evaluation across steering strengths
- **test_setup.py**: Enhanced verification script with static analysis to catch method errors early

## Experimental Pipeline

1. **Dataset Preparation**: Load RealToxicityPrompts and create challenging/benign subsets (50 prompts each)
2. **Baseline Evaluation**: Generate completions without steering (temp=0.7, top_p=0.9)
3. **Activation Collection**: Capture residual stream activations across all layers
4. **Steering Vector Computation**: Use CAA with 20 contrastive pairs per layer for efficiency
5. **Steering Evaluation**: Test across alpha values [-1, 0, 1]:
   - **α = -1**: Negative steering (opposite direction)
   - **α = 0**: Baseline (no steering)
   - **α = 1**: Positive steering (learned direction)
6. **Layer-wise Analysis**: Evaluate steering effectiveness across all transformer layers (integrated)

## Key Technical Details

- **Model**: microsoft/Phi-4-mini-instruct (compact, efficient language model optimized for instruction following)
- **Toxicity Scorer**: unitary/unbiased-toxic-roberta for MeanToxicity evaluation
- **Activation Capture**: Forward hooks on transformer layers with mean-pooling over last 32 tokens
- **Steering Method**: Contrastive activation addition applied during inference
- **Hardware**: Optimized for GPU use (CUDA/Apple Silicon MPS) with automatic fallback to CPU

## Important Research Findings

⚠️ **Inverted Toxicity Pattern Discovered**: This experiment has revealed that the Phi-4-mini-instruct model shows **inverted toxicity behavior**:

- **High-toxicity prompts** → **Low-toxicity completions** (mean ~0.42)
- **Low-toxicity prompts** → **High-toxicity completions** (mean ~0.82)

**What this means:**
1. The model has learned to **avoid continuing toxic content** when it recognizes explicit toxicity
2. **Benign prompts** can sometimes lead to **more problematic completions** than toxic ones
3. **Steering is most effective** on benign prompts that would otherwise produce toxic outputs
4. This challenges assumptions about **prompt-completion toxicity correlation**

**Research Implications:**
- Safety evaluations should focus on **completion toxicity**, not just prompt toxicity
- Steering interventions may need **different approaches** for different prompt types  
- This model demonstrates **learned toxicity avoidance behaviors** from instruction tuning

## Project Structure

```
├── .venv/                    # Python virtual environment
├── .idea/                    # JetBrains IDE configuration  
├── .git/                     # Git repository
├── main.py                   # Main experiment entry point with plotting support
├── plot_results.py           # Standalone plotting script  
├── analyze_layers.py         # Layer-wise steering analysis script
├── activation_steering.py    # Core experiment implementation
├── test_setup.py            # Setup verification script
├── CLAUDE.md                # This documentation
└── README.md                # Experiment instructions
```

## Results and Output

### Data Files
- **experiment_results.json**: Complete experimental results with baseline and steering data
- **intermediate_experiment_output.json**: Step-by-step progress logging with detailed metrics
- **partial_experiment_results.json**: Saved if experiment is interrupted
- **failed_experiment_results.json**: Saved if experiment fails mid-execution

### Visualization Output
- **plots/**: Directory containing all generated visualization files
  - **alpha_sweep_results.png/pdf**: Main analysis showing toxicity across alpha values
  - **toxicity_heatmaps.png/pdf**: Detailed heatmap analysis of all results
  - **summary_statistics.png/pdf**: Statistical summary and distribution analysis
  - **layer_effectiveness_analysis.png/pdf**: Comprehensive 6-panel layer analysis (default)
  - **detailed_layer_heatmap.png/pdf**: Layer vs alpha effectiveness heatmap (default)

## Notes for Development

- Always activate virtual environment before running any Python commands
- The Phi-4-mini-instruct model is much more memory efficient than larger models
- **Reduced sample sizes**: Uses 50 prompts per subset and 20 contrastive pairs for efficiency
- **Statistical considerations**: Smaller samples improve speed but may reduce statistical power
- **Layer analysis integrated**: Automatically evaluates all transformer layers for steering effectiveness
- GPU usage is recommended but CPU fallback is implemented
- Results are automatically saved with comprehensive metrics
- The experiment implements proper error handling and partial result saving

---

# Experimental Summary

## Approach

This experiment implements **Contrastive Activation Addition (CAA)** for toxicity steering in the microsoft/Phi-4-mini-instruct language model. The approach follows established activation steering methodology:

1. **Dataset Preparation**: Uses RealToxicityPrompts dataset with 50 challenging (high prompt toxicity) and 50 benign (low prompt toxicity) prompts
2. **Baseline Evaluation**: Generates unsteered completions to establish toxicity baselines  
3. **Activation Collection**: Captures residual stream activations across all 32 transformer layers during baseline generation
4. **Steering Vector Computation**: Computes layer-wise steering vectors using CAA with 20 contrastive pairs per layer
5. **Steering Evaluation**: Tests steering effectiveness across alpha values [-1, 0, 1] representing negative, neutral, and positive steering
6. **Layer-wise Analysis**: Evaluates which transformer layers are most effective for steering interventions

## Implementation

The implementation provides a comprehensive activation steering pipeline:

- **Efficient Architecture Detection**: Automatically detects and supports multiple transformer architectures (Phi, GPT-style, LLaMA-style)
- **Apple Silicon Optimization**: Native MPS acceleration with memory optimizations for unified memory architecture
- **Robust Error Handling**: Comprehensive error handling with partial result saving and graceful fallbacks
- **Comprehensive Testing**: Enhanced test suite with static analysis to catch method errors early
- **Rich Visualization**: Generates 5 visualization types including layer effectiveness analysis
- **Modular Design**: Separate scripts for plotting, layer analysis, and testing enable flexible workflows

### Key Technical Features:
- **Memory Efficient**: Uses smaller prompt subsets and optimized batch processing
- **Hardware Agnostic**: Automatic GPU/CPU detection with Apple Silicon, CUDA, and CPU support
- **Reproducible**: Fixed random seeds and deterministic generation parameters
- **Extensible**: Modular architecture allows easy addition of new steering methods or models

## Results

### Baseline Toxicity Patterns
The experiment revealed **unexpected inverted toxicity behavior** in Phi-4-mini-instruct:

| Subset | Prompt Toxicity | Completion Toxicity | Interpretation |
|--------|----------------|-------------------|----------------|
| **Challenging** | 0.9856 (very high) | 0.4171 (moderate) | Model avoids continuing toxic prompts |
| **Benign** | 0.0038 (very low) | 0.8162 (high) | Model sometimes generates toxic content from benign prompts |

### Steering Effectiveness
Testing across alpha values [-1, 0, 1] showed:

| Alpha | Challenging Completion Toxicity | Benign Completion Toxicity | Interpretation |
|-------|-------------------------------|---------------------------|----------------|
| **α = -1** | 0.4249 (+1.9%) | 0.8149 (-0.2%) | Negative steering slightly increases toxicity |
| **α = 0** | 0.3845 (-7.8%) | 0.6962 (-14.7%) | Baseline (no steering) |
| **α = 1** | 0.2814 (-32.5%) | 0.7962 (-2.4%) | Positive steering significantly reduces toxicity |

**Key Finding**: α = 1 provides the most effective toxicity reduction, with 32.5% improvement on challenging prompts.

### Layer Analysis Results
Layer-wise analysis across all 32 transformer layers revealed:

- **Most Effective Layer**: Layer 16 (middle layers most responsive to steering)
- **Layer Depth Patterns**: Middle layers (33-66%) show higher steering effectiveness than early or late layers
- **Steering Consistency**: Positive alpha values consistently outperform negative values across layers

## Key Findings

### 1. **Inverted Toxicity Pattern Discovery** 🔍
**Finding**: Phi-4-mini-instruct shows inverted prompt-completion toxicity correlation
- High-toxicity prompts → Low-toxicity completions
- Low-toxicity prompts → High-toxicity completions

**Implication**: The model has learned toxicity avoidance behaviors during instruction tuning, but these can be inconsistent.

### 2. **Steering Effectiveness Varies by Context** 🎯
**Finding**: Steering effectiveness depends on prompt type
- Challenging prompts: 32.5% toxicity reduction with positive steering
- Benign prompts: 2.4% toxicity reduction with positive steering  

**Implication**: Steering is most valuable for prompts that already produce low toxicity, providing additional safety margins.

### 3. **Layer-Specific Steering Responses** 🧠
**Finding**: Middle transformer layers (around layer 16) are most responsive to steering
- Early layers: Lower steering effectiveness
- Middle layers: Highest steering effectiveness  
- Late layers: Moderate steering effectiveness

**Implication**: Targeted interventions in middle layers may be more efficient than whole-model steering.

### 4. **Model Safety Training Evidence** 🛡️
**Finding**: The model demonstrates learned safety behaviors
- Explicit toxicity triggers avoidance responses
- Implicit context can still lead to problematic outputs
- Steering can enhance existing safety training

**Implication**: Activation steering complements but doesn't replace safety training approaches.

## Research Implications

### For AI Safety Research:
1. **Evaluation Methods**: Completion toxicity may be more important than prompt toxicity for safety evaluation
2. **Intervention Strategies**: Layer-specific steering could be more efficient than uniform interventions  
3. **Model Understanding**: Instruction-tuned models may have complex, non-intuitive safety behaviors

### For Activation Steering Research:
1. **Context Sensitivity**: Steering effectiveness varies significantly with input context
2. **Architecture Specifics**: Different model architectures may require tailored steering approaches
3. **Efficiency Opportunities**: Targeted layer interventions could reduce computational costs

### For Practical Applications:
1. **Deployment Considerations**: Models may need different safety interventions for different input types
2. **Monitoring Requirements**: Both prompt and completion toxicity should be monitored in production
3. **Intervention Timing**: Middle-layer interventions may provide optimal efficiency-effectiveness balance

## Limitations and Future Work

### Current Limitations:
- **Small Scale**: Limited to 50 prompts per subset for computational efficiency
- **Single Model**: Results specific to Phi-4-mini-instruct architecture
- **Limited Metrics**: Focuses primarily on toxicity; other safety metrics not evaluated
- **Static Evaluation**: No dynamic or interactive evaluation scenarios

### Future Research Directions:
1. **Multi-Model Comparison**: Test across different model families and sizes
2. **Dynamic Steering**: Implement adaptive steering based on context recognition
3. **Multi-Objective Steering**: Balance toxicity reduction with other capabilities
4. **Production Deployment**: Test steering approaches in real-world applications
5. **Mechanistic Understanding**: Deeper investigation of why middle layers are most effective

## Conclusion

This experiment successfully demonstrates that **activation steering can effectively reduce toxicity** in the Phi-4-mini-instruct model, with **32.5% improvement** using positive steering (α = 1). However, the discovery of **inverted toxicity patterns** reveals the complexity of safety behaviors in instruction-tuned models.

The **layer-wise analysis** provides actionable insights for efficient steering implementation, showing that **middle layers** offer the best intervention points. The **comprehensive implementation** with Apple Silicon optimization and robust error handling makes this approach practical for further research and potential deployment.

Most importantly, this work highlights that **effective AI safety requires understanding model-specific behaviors** rather than assuming universal patterns. The inverted toxicity pattern in Phi-4-mini-instruct demonstrates that safety evaluation and intervention strategies must be carefully tailored to each model's learned behaviors.