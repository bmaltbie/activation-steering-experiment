My first attempt at applying mechanistic interpretability. I implemented activation steering for toxicity reduction in the microsoft/Phi-4-mini-instruct model using Contrastive Activation Addition (CAA). Experiments on the RealToxicityPrompts dataset achieved 35.4% toxicity reduction on challenging prompts and 50.0% reduction on benign prompts, with Layer 29 identified as the most effective intervention point (70.8% effectiveness). Results include a layer-wise analysis across all 32 transformer layers to see the impact at a layer-level.

## Experimental Results

### 1. **Normal Toxicity Pattern Observed** 
**Baseline Toxicity Behavior**: Phi-4-mini-instruct shows expected prompt-completion correlation:

| Subset | Prompt Toxicity | Completion Toxicity | Interpretation |
|--------|----------------|-------------------|----------------|
| **Challenging** | 0.986 (very high) | 0.602 (high) | High-toxicity prompts → High-toxicity completions |
| **Benign** | 0.004 (very low) | 0.006 (very low) | Low-toxicity prompts → Low-toxicity completions |

The model follows expected toxicity propagation patterns, making it a suitable target for steering interventions.

### 2. **Steering Effectiveness Results**
**Optimal Performance**: α = 1 (positive steering) achieved significant toxicity reduction:

| Alpha Value | Challenging Toxicity | Benign Toxicity | Improvement |
|-------------|---------------------|----------------|-------------|
| **α = -1** | 0.622 (+3.3%) | 0.013 (+116.7%) | Negative steering increases toxicity |
| **α = 0** | 0.602 (baseline) | 0.006 (baseline) | No intervention |
| **α = 1** | 0.389 (-35.4%) | 0.003 (-50.0%) | **Optimal toxicity reduction** |

**Key Result**: **35.4% toxicity reduction** on challenging prompts and **50.0% reduction** on benign prompts.

### 3. **Layer-Wise Effectiveness Analysis**
**Most Effective Layers**: Comprehensive analysis across all 32 transformer layers:

| Layer Group | Layers | Average Effectiveness | Best Performance |
|-------------|--------|---------------------|------------------|
| **Early** | 0-10 | 43.3% | Layer 6 (8.8% final toxicity) |
| **Middle** | 11-21 | 45.8% | Layer 13 (16.7% final toxicity) |
| **Late** | 22-31 | 41.9% | **Layer 29 (70.8% effectiveness)** |

**Layer 29** provides highest steering effectiveness at **70.8%**.

### 4. **Steering Consistency Across Contexts** ⚖️
**Universal Effectiveness**: Positive steering (α = 1) consistently outperforms across all conditions:

- **Challenging Prompts**: 35.5% best improvement, 16.1% average improvement
- **Benign Prompts**: 50.9% best improvement, already very low baseline
- **Layer Consistency**: Positive alpha values effective across all 32 layers

## Statistical Summary

**Baseline Toxicity Distribution**:
- **Challenging**: Mean 0.538, Std 0.106, Range 0.389-0.622
- **Benign**: Mean 0.007, Std 0.004, Range 0.003-0.013

**Best Steering Results (α = 1)**:
- **Challenging**: Mean 0.389 (-27.7% from baseline)
- **Benign**: Mean 0.003 (-57.1% from baseline)