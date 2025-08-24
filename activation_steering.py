#!/usr/bin/env python3
"""
Activation Steering for Toxicity Reduction in Qwen3-8B
AI Safety Research Implementation

This script implements a contrastive activation addition (CAA) method 
to reduce toxicity in the Qwen3-8B language model using activation steering.
"""

import os
import json
import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    set_seed
)

# Set random seeds for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256  # Using upper bound of 128-256 range
    do_sample: bool = True
    pad_token_id: int = None
    eos_token_id: int = None

@dataclass
class SteeringConfig:
    """Configuration for activation steering"""
    k_examples: int = 50  # Number of top-K toxic/non-toxic examples
    mean_pool_tokens: int = 32  # Number of last tokens to mean-pool
    alpha_values: List[float] = None  # Alpha values for steering strength
    
    def __post_init__(self):
        if self.alpha_values is None:
            self.alpha_values = [-1.0, -0.5, 0, 0.25, 0.5, 1.0, 1.5, 2.0]

class ActivationSteeringExperiment:
    """
    Main class for the activation steering experiment.
    Implements toxicity reduction using contrastive activation addition.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize toxicity classifier
        print("Loading toxicity classifier...")
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-roberta",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Configuration
        self.gen_config = GenerationConfig()
        self.gen_config.pad_token_id = self.tokenizer.pad_token_id
        self.gen_config.eos_token_id = self.tokenizer.eos_token_id
        
        self.steering_config = SteeringConfig()
        
        # Data storage
        self.rtp_data = None
        self.challenging_subset = None
        self.benign_subset = None
        self.baseline_results = None
        self.activations = {}
        self.steering_vectors = {}
        
    def load_rtp_dataset(self) -> None:
        """Load the RealToxicityPrompts dataset"""
        print("Loading RealToxicityPrompts dataset...")
        
        # Load the dataset
        dataset = datasets.load_dataset("allenai/real-toxicity-prompts")
        self.rtp_data = dataset['train']
        
        print(f"Loaded {len(self.rtp_data)} prompts from RealToxicityPrompts")
        
    def create_subsets(self) -> None:
        """Create challenging (high toxicity) and benign (low toxicity) subsets"""
        if self.rtp_data is None:
            raise ValueError("RTP dataset not loaded. Call load_rtp_dataset() first.")
            
        print("Creating challenging and benign subsets...")
        
        # Filter out prompts without toxicity scores
        valid_prompts = [
            item for item in self.rtp_data 
            if item['prompt'] and item['prompt'].get('toxicity') is not None
        ]
        
        # Sort by toxicity score
        sorted_prompts = sorted(
            valid_prompts, 
            key=lambda x: x['prompt']['toxicity'], 
            reverse=True
        )
        
        # Create subsets
        self.challenging_subset = sorted_prompts[:100]  # Top 100 most toxic
        self.benign_subset = sorted_prompts[-100:]      # Bottom 100 least toxic
        
        print(f"Created challenging subset: {len(self.challenging_subset)} prompts")
        print(f"Mean toxicity: {np.mean([p['prompt']['toxicity'] for p in self.challenging_subset]):.3f}")
        
        print(f"Created benign subset: {len(self.benign_subset)} prompts")
        print(f"Mean toxicity: {np.mean([p['prompt']['toxicity'] for p in self.benign_subset]):.3f}")
        
    def score_toxicity(self, texts: List[str]) -> List[float]:
        """Score toxicity for a list of texts using the classifier"""
        scores = []
        
        # Process in batches to avoid memory issues
        batch_size = 16
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring toxicity"):
            batch = texts[i:i+batch_size]
            batch_scores = self.toxicity_classifier(batch)
            
            for score_result in batch_scores:
                # Get toxicity score (1.0 - TOXIC label probability)
                if isinstance(score_result, list):
                    # Handle case where multiple labels are returned
                    toxic_score = next(
                        (item['score'] for item in score_result if item['label'] == 'TOXIC'), 
                        0.0
                    )
                else:
                    # Handle single result
                    toxic_score = score_result['score'] if score_result['label'] == 'TOXIC' else 1.0 - score_result['score']
                
                scores.append(toxic_score)
        
        return scores
    
    def generate_completions(self, prompts: List[str], alpha: float = 0.0, layer: int = None) -> List[str]:
        """Generate completions for a list of prompts with optional steering"""
        completions = []
        
        for prompt in tqdm(prompts, desc=f"Generating completions (alpha={alpha})"):
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with optional steering
            with torch.no_grad():
                if alpha != 0.0 and layer is not None and layer in self.steering_vectors:
                    # Apply steering
                    outputs = self._generate_with_steering(inputs, alpha, layer)
                else:
                    # Standard generation
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        temperature=self.gen_config.temperature,
                        top_p=self.gen_config.top_p,
                        max_new_tokens=self.gen_config.max_new_tokens,
                        do_sample=self.gen_config.do_sample,
                        pad_token_id=self.gen_config.pad_token_id,
                        eos_token_id=self.gen_config.eos_token_id
                    )
            
            # Decode completion (remove input prompt)
            input_length = inputs['input_ids'].shape[1]
            completion = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            completions.append(completion)
            
        return completions
    
    def _generate_with_steering(self, inputs: Dict[str, torch.Tensor], alpha: float, layer: int) -> torch.Tensor:
        """Generate text with activation steering applied at specified layer"""
        if layer not in self.steering_vectors:
            raise ValueError(f"No steering vector available for layer {layer}")
        
        steering_vector = self.steering_vectors[layer]
        applied_steering = False
        
        def steering_hook(module, input, output):
            nonlocal applied_steering
            if not applied_steering:
                # Apply steering to the residual stream
                # output[0] shape: [batch_size, seq_len, hidden_dim]
                batch_size, seq_len, hidden_dim = output[0].shape
                
                # Add steering vector to all positions
                steering_addition = alpha * steering_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                steering_addition = steering_addition.expand(batch_size, seq_len, hidden_dim)
                
                output = (output[0] + steering_addition.to(output[0].device),) + output[1:]
                applied_steering = True
                
            return output
        
        # Register hook on the target layer
        hook = self.model.transformer.h[layer].register_forward_hook(steering_hook)
        
        try:
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                max_new_tokens=self.gen_config.max_new_tokens,
                do_sample=self.gen_config.do_sample,
                pad_token_id=self.gen_config.pad_token_id,
                eos_token_id=self.gen_config.eos_token_id
            )
        finally:
            hook.remove()
        
        return outputs
    
    def run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run baseline evaluation without steering"""
        print("Running baseline evaluation...")
        
        if self.challenging_subset is None or self.benign_subset is None:
            raise ValueError("Subsets not created. Call create_subsets() first.")
        
        # Extract prompts
        challenging_prompts = [item['prompt']['text'] for item in self.challenging_subset]
        benign_prompts = [item['prompt']['text'] for item in self.benign_subset]
        
        # Generate completions
        challenging_completions = self.generate_completions(challenging_prompts)
        benign_completions = self.generate_completions(benign_prompts)
        
        # Score completions
        challenging_scores = self.score_toxicity(challenging_completions)
        benign_scores = self.score_toxicity(benign_completions)
        
        # Calculate metrics
        results = {
            'challenging': {
                'prompts': challenging_prompts,
                'completions': challenging_completions,
                'toxicity_scores': challenging_scores,
                'mean_toxicity': np.mean(challenging_scores)
            },
            'benign': {
                'prompts': benign_prompts,
                'completions': benign_completions,
                'toxicity_scores': benign_scores,
                'mean_toxicity': np.mean(benign_scores)
            }
        }
        
        self.baseline_results = results
        
        print(f"Baseline Results:")
        print(f"  Challenging subset mean toxicity: {results['challenging']['mean_toxicity']:.3f}")
        print(f"  Benign subset mean toxicity: {results['benign']['mean_toxicity']:.3f}")
        
        return results
    
    def collect_activations(self) -> None:
        """Collect activations from all layers during baseline generation"""
        if self.baseline_results is None:
            raise ValueError("Baseline results not available. Run run_baseline_evaluation() first.")
        
        print("Collecting activations from model layers...")
        
        # Get all transformer layers
        num_layers = len(self.model.transformer.h)
        print(f"Model has {num_layers} transformer layers")
        
        # Storage for activations
        layer_activations = {i: [] for i in range(num_layers)}
        current_activations = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Store the activation (residual stream output)
                current_activations[layer_idx] = output[0].detach().cpu()
            return hook_fn
        
        # Register hooks for all layers
        hooks = []
        for i, layer in enumerate(self.model.transformer.h):
            hook = layer.register_forward_hook(make_hook(i))
            hooks.append(hook)
        
        try:
            # Re-generate completions with hooks to collect activations
            all_prompts = []
            all_labels = []
            
            # Challenging prompts (will be labeled as potentially toxic)
            challenging_prompts = [item['prompt']['text'] for item in self.challenging_subset]
            all_prompts.extend(challenging_prompts)
            all_labels.extend(['challenging'] * len(challenging_prompts))
            
            # Benign prompts
            benign_prompts = [item['prompt']['text'] for item in self.benign_subset]
            all_prompts.extend(benign_prompts)
            all_labels.extend(['benign'] * len(benign_prompts))
            
            # Generate completions with activation collection
            for idx, (prompt, label) in enumerate(tqdm(zip(all_prompts, all_labels), 
                                                    desc="Collecting activations", 
                                                    total=len(all_prompts))):
                # Clear current activations
                current_activations.clear()
                
                # Tokenize and generate
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        temperature=self.gen_config.temperature,
                        top_p=self.gen_config.top_p,
                        max_new_tokens=self.gen_config.max_new_tokens,
                        do_sample=self.gen_config.do_sample,
                        pad_token_id=self.gen_config.pad_token_id,
                        eos_token_id=self.gen_config.eos_token_id
                    )
                
                # Get the completion text
                input_length = inputs['input_ids'].shape[1]
                completion = self.tokenizer.decode(
                    outputs[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
                
                # Store activations for each layer with metadata
                for layer_idx in current_activations:
                    # Mean pool over the last min(32, sequence_length) tokens
                    activation = current_activations[layer_idx]
                    seq_len = activation.shape[1]
                    pool_tokens = min(self.steering_config.mean_pool_tokens, seq_len)
                    
                    # Take the last pool_tokens and mean pool
                    pooled_activation = activation[:, -pool_tokens:, :].mean(dim=1)  # [1, hidden_dim]
                    
                    layer_activations[layer_idx].append({
                        'activation': pooled_activation.squeeze(0),  # [hidden_dim]
                        'completion': completion,
                        'prompt_type': label,
                        'prompt': prompt
                    })
        
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()
        
        self.activations = layer_activations
        print(f"Collected activations for {len(all_prompts)} completions across {num_layers} layers")
    
    def compute_steering_vectors(self) -> Dict[int, torch.Tensor]:
        """Compute steering vectors for each layer using CAA"""
        if not self.activations:
            raise ValueError("Activations not collected. Run collect_activations() first.")
        
        print("Computing steering vectors using contrastive activation addition...")
        
        steering_vectors = {}
        
        for layer_idx, layer_data in self.activations.items():
            print(f"Processing layer {layer_idx}...")
            
            # Score all completions for this layer
            completions = [item['completion'] for item in layer_data]
            toxicity_scores = self.score_toxicity(completions)
            
            # Add toxicity scores to the data
            for item, score in zip(layer_data, toxicity_scores):
                item['toxicity_score'] = score
            
            # Sort by toxicity score (descending)
            sorted_data = sorted(layer_data, key=lambda x: x['toxicity_score'], reverse=True)
            
            # Select top-K toxic and top-K non-toxic examples
            k = self.steering_config.k_examples
            toxic_examples = sorted_data[:k]  # Most toxic
            non_toxic_examples = sorted_data[-k:]  # Least toxic
            
            print(f"  Layer {layer_idx}: Top-{k} toxic examples have mean toxicity {np.mean([x['toxicity_score'] for x in toxic_examples]):.3f}")
            print(f"  Layer {layer_idx}: Top-{k} non-toxic examples have mean toxicity {np.mean([x['toxicity_score'] for x in non_toxic_examples]):.3f}")
            
            # Compute mean activations
            toxic_activations = torch.stack([item['activation'] for item in toxic_examples])
            non_toxic_activations = torch.stack([item['activation'] for item in non_toxic_examples])
            
            toxic_mean = toxic_activations.mean(dim=0)
            non_toxic_mean = non_toxic_activations.mean(dim=0)
            
            # Compute steering vector (difference)
            steering_vector = non_toxic_mean - toxic_mean  # Points towards non-toxic
            
            # Sanity check: verify the direction makes sense
            # The steering vector should point from toxic to non-toxic representations
            dot_product_check = torch.dot(steering_vector, (non_toxic_mean - toxic_mean))
            print(f"  Layer {layer_idx}: Sanity check - dot product with expected direction: {dot_product_check.item():.3f}")
            
            if dot_product_check < 0:
                print(f"  WARNING: Layer {layer_idx} steering vector may have wrong sign!")
            
            steering_vectors[layer_idx] = steering_vector.to(self.device)
        
        self.steering_vectors = steering_vectors
        print(f"Computed steering vectors for {len(steering_vectors)} layers")
        
        return steering_vectors
    
    def run_steering_evaluation(self, target_layer: int = None) -> Dict[str, Any]:
        """Run full steering evaluation across all alpha values"""
        if not self.steering_vectors:
            raise ValueError("Steering vectors not computed. Run compute_steering_vectors() first.")
        
        print("Running steering evaluation...")
        
        # Select target layer (use middle layer if not specified)
        if target_layer is None:
            target_layer = len(self.steering_vectors) // 2
            print(f"Using layer {target_layer} for steering (middle layer)")
        else:
            print(f"Using layer {target_layer} for steering")
        
        if target_layer not in self.steering_vectors:
            raise ValueError(f"No steering vector available for layer {target_layer}")
        
        # Extract prompts for evaluation
        challenging_prompts = [item['prompt']['text'] for item in self.challenging_subset]
        benign_prompts = [item['prompt']['text'] for item in self.benign_subset]
        
        results = {}
        
        for alpha in tqdm(self.steering_config.alpha_values, desc="Alpha sweep"):
            print(f"\nEvaluating with alpha = {alpha}")
            
            # Generate completions with steering
            challenging_completions = self.generate_completions(
                challenging_prompts, alpha=alpha, layer=target_layer
            )
            benign_completions = self.generate_completions(
                benign_prompts, alpha=alpha, layer=target_layer  
            )
            
            # Score completions
            challenging_scores = self.score_toxicity(challenging_completions)
            benign_scores = self.score_toxicity(benign_completions)
            
            # Store results
            results[alpha] = {
                'challenging': {
                    'completions': challenging_completions,
                    'toxicity_scores': challenging_scores,
                    'mean_toxicity': np.mean(challenging_scores)
                },
                'benign': {
                    'completions': benign_completions,
                    'toxicity_scores': benign_scores,
                    'mean_toxicity': np.mean(benign_scores)
                }
            }
            
            print(f"  Challenging subset mean toxicity: {results[alpha]['challenging']['mean_toxicity']:.3f}")
            print(f"  Benign subset mean toxicity: {results[alpha]['benign']['mean_toxicity']:.3f}")
        
        # Add summary analysis
        print("\n=== Steering Evaluation Summary ===")
        print("Alpha\tChallenging\tBenign")
        for alpha in self.steering_config.alpha_values:
            challenging_mean = results[alpha]['challenging']['mean_toxicity']
            benign_mean = results[alpha]['benign']['mean_toxicity']
            print(f"{alpha:6.2f}\t{challenging_mean:.3f}\t\t{benign_mean:.3f}")
        
        return {
            'results': results,
            'target_layer': target_layer,
            'baseline_challenging': self.baseline_results['challenging']['mean_toxicity'] if self.baseline_results else None,
            'baseline_benign': self.baseline_results['benign']['mean_toxicity'] if self.baseline_results else None
        }
    
    def save_results(self, filepath: str, steering_results: Dict[str, Any] = None) -> None:
        """Save all experimental results to file"""
        results = {
            'baseline_results': self.baseline_results,
            'steering_results': steering_results,
            'steering_vectors_shapes': {k: list(v.shape) for k, v in self.steering_vectors.items()},
            'config': {
                'model_name': self.model_name,
                'generation_config': self.gen_config.__dict__,
                'steering_config': self.steering_config.__dict__
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")

def main():
    """Main execution function"""
    print("=== Activation Steering Experiment for Toxicity Reduction ===")
    
    # Initialize experiment
    experiment = ActivationSteeringExperiment()
    
    try:
        # Step 1: Load dataset and create subsets
        experiment.load_rtp_dataset()
        experiment.create_subsets()
        
        # Step 2: Run baseline evaluation
        print("\n--- Step 2: Baseline Evaluation ---")
        baseline_results = experiment.run_baseline_evaluation()
        
        # Step 3: Collect activations
        print("\n--- Step 3: Collecting Activations ---")
        experiment.collect_activations()
        
        # Step 4: Compute steering vectors
        print("\n--- Step 4: Computing Steering Vectors ---")
        experiment.compute_steering_vectors()
        
        # Step 5: Run steering evaluation
        print("\n--- Step 5: Steering Evaluation ---")
        steering_results = experiment.run_steering_evaluation()
        
        # Save all results
        print("\n--- Saving Results ---")
        experiment.save_results('experiment_results.json', steering_results)
        
        print("\n=== Experiment completed successfully ===")
        
        # Print final summary
        print("\n=== Final Summary ===")
        print(f"Baseline - Challenging: {baseline_results['challenging']['mean_toxicity']:.3f}")
        print(f"Baseline - Benign: {baseline_results['benign']['mean_toxicity']:.3f}")
        
        if steering_results:
            best_alpha = min(
                steering_results['results'].keys(), 
                key=lambda a: steering_results['results'][a]['challenging']['mean_toxicity']
            )
            print(f"Best alpha for toxicity reduction: {best_alpha}")
            print(f"  Challenging: {steering_results['results'][best_alpha]['challenging']['mean_toxicity']:.3f}")
            print(f"  Benign: {steering_results['results'][best_alpha]['benign']['mean_toxicity']:.3f}")
        
    except KeyboardInterrupt:
        print("\n=== Experiment interrupted by user ===")
        # Save partial results if available
        if hasattr(experiment, 'baseline_results') and experiment.baseline_results:
            experiment.save_results('partial_experiment_results.json')
            print("Partial results saved to partial_experiment_results.json")
    except Exception as e:
        print(f"\n=== Experiment failed with error: {e} ===")
        # Save partial results if available
        if hasattr(experiment, 'baseline_results') and experiment.baseline_results:
            experiment.save_results('failed_experiment_results.json')
            print("Partial results saved to failed_experiment_results.json")
        raise

if __name__ == "__main__":
    main()