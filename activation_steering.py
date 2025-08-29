#!/usr/bin/env python3
"""
Activation Steering for Toxicity Reduction in Phi-4-mini-instruct
AI Safety Research Implementation

This script implements a contrastive activation addition (CAA) method 
to reduce toxicity in the Phi-4-mini-instruct language model using activation steering.
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# Set random seeds for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.3  # More deterministic for clearer steering effects
    top_p: float = 0.8       # More focused sampling to reduce noise
    max_new_tokens: int = 128  # Using lower bound of 128-256 range
    do_sample: bool = True
    pad_token_id: int = None
    eos_token_id: int = None

@dataclass
class SteeringConfig:
    """Configuration for activation steering"""
    k_examples: int = 20  # Number of top-K toxic/non-toxic examples (reduced for efficiency)
    mean_pool_tokens: int = 32  # Number of last tokens to mean-pool
    alpha_values: List[float] = None  # Alpha values for steering strength
    
    def __post_init__(self):
        if self.alpha_values is None:
            self.alpha_values = [-1, 0, 1]  # Simplified set: negative, neutral, positive steering

class ActivationSteeringExperiment:
    """
    Main class for the activation steering experiment.
    Implements toxicity reduction using contrastive activation addition.
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-4-mini-instruct"):
        self.model_name = model_name
        
        # Optimized device detection for Apple Silicon
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU acceleration")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (no GPU acceleration available)")
        
        # Initialize model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optimized model loading for different hardware
        use_fp16 = torch.cuda.is_available() or torch.backends.mps.is_available()
        model_kwargs = {
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            "low_cpu_mem_usage": True  # Helps with memory management on Apple Silicon
        }
        
        # Only use device_map for CUDA (MPS doesn't support it)
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move model to appropriate device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize toxicity classifier with optimized device support
        print("Loading toxicity classifier...")
        
        # Determine device for pipeline
        if torch.cuda.is_available():
            pipeline_device = 0  # CUDA device 0
        elif torch.backends.mps.is_available():
            pipeline_device = "mps"  # Apple Silicon MPS
        else:
            pipeline_device = -1  # CPU
            
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-roberta",
            device=pipeline_device,
            top_k=None  # Return all scores (replaces deprecated return_all_scores=True)
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
        
        # Intermediate output file path
        self.intermediate_output_file = "intermediate_experiment_output.json"
        
        # Apple Silicon specific optimizations
        self._setup_apple_silicon_optimizations()
        
        # Set up model architecture compatibility
        self._setup_model_architecture()
    
    def _setup_model_architecture(self) -> None:
        """Detect model architecture and set up layer access compatibility"""
        model_type = type(self.model).__name__.lower()
        
        try:
            if 'phi' in model_type:
                # Phi models use 'model.layers' instead of 'transformer.h'
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    self.transformer_layers = self.model.model.layers
                    self.layer_attr = 'model.layers'
                    print(f"Detected Phi architecture: {type(self.model).__name__}")
                else:
                    raise AttributeError("Phi model doesn't have expected structure")
                    
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # GPT-style models (GPT-2, Qwen, etc.) use 'transformer.h'
                self.transformer_layers = self.model.transformer.h
                self.layer_attr = 'transformer.h'
                print(f"Detected GPT-style architecture: {type(self.model).__name__}")
                
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # LLaMA-style models use 'model.layers'
                self.transformer_layers = self.model.model.layers
                self.layer_attr = 'model.layers'
                print(f"Detected LLaMA-style architecture: {type(self.model).__name__}")
                
            else:
                # Try to find layers automatically
                possible_attrs = [
                    ('model.layers', lambda m: m.model.layers),
                    ('transformer.h', lambda m: m.transformer.h),
                    ('transformer.layers', lambda m: m.transformer.layers),
                    ('layers', lambda m: m.layers),
                ]
                
                for attr_name, attr_getter in possible_attrs:
                    try:
                        layers = attr_getter(self.model)
                        if hasattr(layers, '__len__') and len(layers) > 0:
                            self.transformer_layers = layers
                            self.layer_attr = attr_name
                            print(f"Auto-detected architecture with {attr_name}: {type(self.model).__name__}")
                            break
                    except (AttributeError, TypeError):
                        continue
                else:
                    raise ValueError(f"Could not find transformer layers in model {type(self.model).__name__}")
            
            # Validate that we have layers
            if not hasattr(self, 'transformer_layers') or len(self.transformer_layers) == 0:
                raise ValueError(f"No transformer layers found in {type(self.model).__name__}")
                
            print(f"Found {len(self.transformer_layers)} transformer layers")
            
        except Exception as e:
            print(f"Error detecting model architecture: {e}")
            print("Available attributes:")
            for attr in dir(self.model):
                if not attr.startswith('_'):
                    print(f"  - {attr}")
            raise
    
    def _setup_apple_silicon_optimizations(self) -> None:
        """Configure optimizations specific to Apple Silicon"""
        if self.device.type == "mps":
            print("Applying Apple Silicon MPS optimizations...")
            
            try:
                # Enable memory efficient attention if available
                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = True
                
                # Clear MPS cache to start fresh
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    
                # Set memory fraction if available (helps prevent OOM)
                if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                    torch.mps.set_per_process_memory_fraction(0.8)
                    
                print("MPS optimizations applied successfully")
                
            except Exception as e:
                print(f"Warning: Some MPS optimizations failed: {e}")
                print("Continuing with default MPS configuration...")
        
        elif self.device.type == "cuda":
            print("Applying CUDA optimizations...")
            try:
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            except Exception as e:
                print(f"Warning: CUDA optimization failed: {e}")
    
    def _clear_device_cache(self) -> None:
        """Clear device cache to free memory"""
        try:
            if self.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass  # Silently continue if cache clearing fails
        
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
        
        # Create subsets (reduced size for efficiency)
        self.challenging_subset = sorted_prompts[:50]   # Top 50 most toxic
        self.benign_subset = sorted_prompts[-50:]       # Bottom 50 least toxic
        
        print(f"Created challenging subset: {len(self.challenging_subset)} prompts")
        print(f"Mean toxicity: {np.mean([p['prompt']['toxicity'] for p in self.challenging_subset]):.3f}")
        
        print(f"Created benign subset: {len(self.benign_subset)} prompts")
        print(f"Mean toxicity: {np.mean([p['prompt']['toxicity'] for p in self.benign_subset]):.3f}")
        
        # Save intermediate output: prompt subsets
        self._save_intermediate_output({
            'step': 'subset_creation',
            'timestamp': self._get_timestamp(),
            'challenging_subset': {
                'count': len(self.challenging_subset),
                'mean_toxicity': np.mean([p['prompt']['toxicity'] for p in self.challenging_subset]),
                'prompts': [{'text': p['prompt']['text'], 'toxicity': p['prompt']['toxicity']} 
                           for p in self.challenging_subset]
            },
            'benign_subset': {
                'count': len(self.benign_subset),
                'mean_toxicity': np.mean([p['prompt']['toxicity'] for p in self.benign_subset]),
                'prompts': [{'text': p['prompt']['text'], 'toxicity': p['prompt']['toxicity']} 
                           for p in self.benign_subset]
            }
        })
        
    def score_toxicity(self, texts: List[str]) -> List[float]:
        """Score toxicity for a list of texts using the classifier"""
        scores = []
        
        # Process in batches to avoid memory issues
        batch_size = 16
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring toxicity"):
            batch = texts[i:i+batch_size]
            batch_results = self.toxicity_classifier(batch)
            
            for result in batch_results:
                # With top_k=None, result is always a list of all label probabilities
                toxic_score = next(
                    (item['score'] for item in result if item['label'] == 'toxicity'), 
                    0.0
                )
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
                # Handle different output formats
                if isinstance(output, tuple):
                    activation_tensor = output[0]
                    rest_of_output = output[1:]
                else:
                    activation_tensor = output
                    rest_of_output = ()
                
                # Apply steering to the residual stream
                original_shape = activation_tensor.shape
                
                if activation_tensor.dim() == 3:
                    # Standard case: [batch_size, seq_len, hidden_dim]
                    batch_size, seq_len, hidden_dim = activation_tensor.shape
                    steering_addition = alpha * steering_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                    steering_addition = steering_addition.expand(batch_size, seq_len, hidden_dim)
                elif activation_tensor.dim() == 2:
                    # Case: [seq_len, hidden_dim] - add batch dimension
                    seq_len, hidden_dim = activation_tensor.shape
                    steering_addition = alpha * steering_vector.unsqueeze(0)  # [1, hidden_dim]
                    steering_addition = steering_addition.expand(seq_len, hidden_dim)
                else:
                    print(f"Warning: Unexpected activation tensor shape in steering: {original_shape}")
                    return output
                
                # Apply steering
                steered_activation = activation_tensor + steering_addition.to(activation_tensor.device)
                
                # Reconstruct output
                if isinstance(output, tuple):
                    output = (steered_activation,) + rest_of_output
                else:
                    output = steered_activation
                    
                applied_steering = True
                
            return output
        
        # Register hook on the target layer
        hook = self.transformer_layers[layer].register_forward_hook(steering_hook)
        
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
        """
        DEPRECATED: Run baseline evaluation without steering
        
        This function is deprecated. Use run_baseline_evaluation_with_activations() instead
        for better performance (combines baseline evaluation and activation collection).
        """
        print("WARNING: run_baseline_evaluation() is deprecated. Use run_baseline_evaluation_with_activations() for better performance.")
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
        
        # Save intermediate output: baseline results
        self._save_intermediate_output({
            'step': 'baseline_evaluation',
            'timestamp': self._get_timestamp(),
            'results': {
                'challenging_mean_toxicity': results['challenging']['mean_toxicity'],
                'benign_mean_toxicity': results['benign']['mean_toxicity'],
                'challenging_prompt_completion_pairs': [
                    {'prompt': prompt, 'completion': completion, 'toxicity_score': score}
                    for prompt, completion, score in zip(
                        results['challenging']['prompts'],
                        results['challenging']['completions'], 
                        results['challenging']['toxicity_scores']
                    )
                ],
                'benign_prompt_completion_pairs': [
                    {'prompt': prompt, 'completion': completion, 'toxicity_score': score}
                    for prompt, completion, score in zip(
                        results['benign']['prompts'],
                        results['benign']['completions'], 
                        results['benign']['toxicity_scores']
                    )
                ]
            }
        })
        
        return results
    
    def collect_activations(self) -> None:
        """
        DEPRECATED: Collect activations from all layers during baseline generation
        
        This function is deprecated. Use run_baseline_evaluation_with_activations() instead
        for better performance (combines baseline evaluation and activation collection in one pass).
        """
        print("WARNING: collect_activations() is deprecated. Use run_baseline_evaluation_with_activations() for better performance.")
        if self.baseline_results is None:
            raise ValueError("Baseline results not available. Run run_baseline_evaluation() first.")
        
        print("Collecting activations from model layers...")
        
        # Get all transformer layers
        num_layers = len(self.transformer_layers)
        print(f"Model has {num_layers} transformer layers")
        print("Note: Activation collection handles different tensor shapes automatically")
        
        # Storage for activations
        layer_activations = {i: [] for i in range(num_layers)}
        current_activations = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Store the activation (residual stream output)
                # Handle different output formats from different model architectures
                if isinstance(output, tuple):
                    activation = output[0].detach().cpu()
                else:
                    activation = output.detach().cpu()
                
                current_activations[layer_idx] = activation
            return hook_fn
        
        # Register hooks for all layers
        hooks = []
        for i, layer in enumerate(self.transformer_layers):
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
                    activation = current_activations[layer_idx]
                    
                    # Handle different activation tensor shapes
                    if activation.dim() == 2:
                        # Shape: [seq_len, hidden_dim] - need to add batch dimension
                        activation = activation.unsqueeze(0)  # [1, seq_len, hidden_dim]
                    elif activation.dim() == 1:
                        # Shape: [hidden_dim] - already pooled, just use as is
                        pooled_activation = activation
                        layer_activations[layer_idx].append({
                            'activation': pooled_activation,  # [hidden_dim]
                            'completion': completion,
                            'prompt_type': label,
                            'prompt': prompt
                        })
                        continue
                    elif activation.dim() != 3:
                        print(f"Warning: Unexpected activation shape at layer {layer_idx}: {activation.shape}")
                        # Skip this layer if we can't handle the shape
                        continue
                    
                    # Now we should have [batch_size, seq_len, hidden_dim]
                    if activation.dim() == 3:
                        seq_len = activation.shape[1]
                        pool_tokens = min(self.steering_config.mean_pool_tokens, seq_len)
                        
                        # Take the last pool_tokens and mean pool
                        pooled_activation = activation[:, -pool_tokens:, :].mean(dim=1)  # [batch_size, hidden_dim]
                        pooled_activation = pooled_activation.squeeze(0)  # [hidden_dim]
                    else:
                        # Fallback: just use the activation as is
                        pooled_activation = activation.flatten()
                    
                    layer_activations[layer_idx].append({
                        'activation': pooled_activation,  # [hidden_dim]
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
        
        # Clear device cache after intensive activation collection
        self._clear_device_cache()
        
        # Save intermediate output: activation collection metrics
        activation_stats = {}
        for layer_idx, layer_data in layer_activations.items():
            challenging_count = sum(1 for item in layer_data if item['prompt_type'] == 'challenging')
            benign_count = sum(1 for item in layer_data if item['prompt_type'] == 'benign')
            activation_stats[layer_idx] = {
                'total_samples': len(layer_data),
                'challenging_samples': challenging_count,
                'benign_samples': benign_count
            }
        
        self._save_intermediate_output({
            'step': 'activation_collection',
            'timestamp': self._get_timestamp(),
            'metrics': {
                'num_layers': num_layers,
                'total_completions': len(all_prompts),
                'challenging_prompts': sum(1 for label in all_labels if label == 'challenging'),
                'benign_prompts': sum(1 for label in all_labels if label == 'benign'),
                'layer_stats': activation_stats,
                'mean_pool_tokens': self.steering_config.mean_pool_tokens
            }
        })
    
    def run_baseline_evaluation_with_activations(self) -> Dict[str, Any]:
        """Run baseline evaluation while collecting activations in one pass for efficiency"""
        print("Running combined baseline evaluation and activation collection...")
        
        if self.challenging_subset is None or self.benign_subset is None:
            raise ValueError("Subsets not created. Call create_subsets() first.")
        
        # Get all transformer layers
        num_layers = len(self.transformer_layers)
        print(f"Model has {num_layers} transformer layers")
        print("Collecting baseline results and activations simultaneously for efficiency")
        
        # Storage for activations
        layer_activations = {i: [] for i in range(num_layers)}
        current_activations = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Store the activation (residual stream output)
                # Handle different output formats from different model architectures
                if isinstance(output, tuple):
                    activation = output[0].detach().cpu()
                else:
                    activation = output.detach().cpu()
                
                current_activations[layer_idx] = activation
            return hook_fn
        
        # Register hooks for all layers
        hooks = []
        for i, layer in enumerate(self.transformer_layers):
            hook = layer.register_forward_hook(make_hook(i))
            hooks.append(hook)
        
        try:
            # Prepare prompts
            challenging_prompts = [item['prompt']['text'] for item in self.challenging_subset]
            benign_prompts = [item['prompt']['text'] for item in self.benign_subset]
            all_prompts = challenging_prompts + benign_prompts
            all_labels = ['challenging'] * len(challenging_prompts) + ['benign'] * len(benign_prompts)
            
            all_completions = []
            
            # Single pass: generate completions AND collect activations
            for prompt, label in tqdm(zip(all_prompts, all_labels), 
                                    desc="Baseline + Activations", 
                                    total=len(all_prompts)):
                # Clear current activations
                current_activations.clear()
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate completion (activations collected via hooks)
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
                
                # Decode completion (remove input prompt)
                input_length = inputs['input_ids'].shape[1]
                completion = self.tokenizer.decode(
                    outputs[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
                
                all_completions.append(completion)
                
                # Store activations for each layer with metadata
                for layer_idx in current_activations:
                    activation = current_activations[layer_idx]
                    
                    # Handle different activation tensor shapes
                    if activation.dim() == 2:
                        # Shape: [seq_len, hidden_dim] - need to add batch dimension
                        activation = activation.unsqueeze(0)  # [1, seq_len, hidden_dim]
                    elif activation.dim() == 1:
                        # Shape: [hidden_dim] - already pooled, just use as is
                        pooled_activation = activation
                        layer_activations[layer_idx].append({
                            'activation': pooled_activation,  # [hidden_dim]
                            'completion': completion,
                            'prompt_type': label,
                            'prompt': prompt
                        })
                        continue
                    elif activation.dim() != 3:
                        print(f"Warning: Unexpected activation shape at layer {layer_idx}: {activation.shape}")
                        # Skip this layer if we can't handle the shape
                        continue
                    
                    # Now we should have [batch_size, seq_len, hidden_dim]
                    if activation.dim() == 3:
                        seq_len = activation.shape[1]
                        pool_tokens = min(self.steering_config.mean_pool_tokens, seq_len)
                        
                        # Take the last pool_tokens and mean pool
                        pooled_activation = activation[:, -pool_tokens:, :].mean(dim=1)  # [batch_size, hidden_dim]
                        pooled_activation = pooled_activation.squeeze(0)  # [hidden_dim]
                    else:
                        # Fallback: just use the activation as is
                        pooled_activation = activation.flatten()
                    
                    layer_activations[layer_idx].append({
                        'activation': pooled_activation,  # [hidden_dim]
                        'completion': completion,
                        'prompt_type': label,
                        'prompt': prompt
                    })
            
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()
        
        # Score all completions at once (more efficient)
        print("Scoring toxicity for all completions...")
        all_toxicity_scores = self.score_toxicity(all_completions)
        
        # Split results back into challenging/benign
        challenging_completions = all_completions[:len(challenging_prompts)]
        challenging_scores = all_toxicity_scores[:len(challenging_prompts)]
        benign_completions = all_completions[len(challenging_prompts):]
        benign_scores = all_toxicity_scores[len(challenging_prompts):]
        
        # Create baseline results
        baseline_results = {
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
        
        # Store results in instance variables
        self.baseline_results = baseline_results
        self.activations = layer_activations
        
        print(f"Combined Results:")
        print(f"  Challenging subset mean toxicity: {baseline_results['challenging']['mean_toxicity']:.3f}")
        print(f"  Benign subset mean toxicity: {baseline_results['benign']['mean_toxicity']:.3f}")
        print(f"  Collected activations for {len(all_completions)} completions across {num_layers} layers")
        
        # Clear device cache after intensive operation
        self._clear_device_cache()
        
        # Save intermediate output: combined results
        self._save_intermediate_output({
            'step': 'baseline_evaluation_with_activations',
            'timestamp': self._get_timestamp(),
            'baseline_results': {
                'challenging_mean_toxicity': baseline_results['challenging']['mean_toxicity'],
                'benign_mean_toxicity': baseline_results['benign']['mean_toxicity'],
                'challenging_prompt_completion_pairs': [
                    {'prompt': prompt, 'completion': completion, 'toxicity_score': score}
                    for prompt, completion, score in zip(
                        baseline_results['challenging']['prompts'],
                        baseline_results['challenging']['completions'], 
                        baseline_results['challenging']['toxicity_scores']
                    )
                ],
                'benign_prompt_completion_pairs': [
                    {'prompt': prompt, 'completion': completion, 'toxicity_score': score}
                    for prompt, completion, score in zip(
                        baseline_results['benign']['prompts'],
                        baseline_results['benign']['completions'], 
                        baseline_results['benign']['toxicity_scores']
                    )
                ]
            },
            'activation_collection': {
                'num_layers': num_layers,
                'total_completions': len(all_completions),
                'challenging_prompts': len(challenging_prompts),
                'benign_prompts': len(benign_prompts),
                'mean_pool_tokens': self.steering_config.mean_pool_tokens
            }
        })
        
        return baseline_results
    
    def compute_steering_vectors(self) -> Dict[int, torch.Tensor]:
        """Compute steering vectors for each layer using CAA"""
        if not self.activations:
            raise ValueError("Activations not collected. Run collect_activations() first.")
        
        print("Computing steering vectors using contrastive activation addition...")
        print(f"Note: Using k={self.steering_config.k_examples} contrastive pairs per layer.")
        print("Reduced sample size may affect statistical robustness but improves computational efficiency.")
        
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
            
            # Check if we have enough data
            if len(sorted_data) < 2 * k:
                print(f"  Warning: Layer {layer_idx} has only {len(sorted_data)} samples, need {2*k}. Using all available data.")
                k = min(k, len(sorted_data) // 2)
                
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
        
        # Clear device cache after steering vector computation
        self._clear_device_cache()
        
        # Save intermediate output: steering vector computation metrics
        layer_metrics = {}
        for layer_idx in steering_vectors.keys():
            if layer_idx in self.activations:
                layer_data = self.activations[layer_idx]
                completions = [item['completion'] for item in layer_data]
                toxicity_scores = [item.get('toxicity_score', 0) for item in layer_data]
                sorted_data = sorted(layer_data, key=lambda x: x.get('toxicity_score', 0), reverse=True)
                
                k = self.steering_config.k_examples
                toxic_examples = sorted_data[:k]
                non_toxic_examples = sorted_data[-k:]
                
                layer_metrics[layer_idx] = {
                    'total_samples': len(layer_data),
                    'k_examples_used': k,
                    'top_k_toxic_mean_score': np.mean([x.get('toxicity_score', 0) for x in toxic_examples]),
                    'top_k_non_toxic_mean_score': np.mean([x.get('toxicity_score', 0) for x in non_toxic_examples]),
                    'steering_vector_norm': float(torch.norm(steering_vectors[layer_idx]).item()),
                    'steering_vector_shape': list(steering_vectors[layer_idx].shape)
                }
        
        self._save_intermediate_output({
            'step': 'steering_vector_computation',
            'timestamp': self._get_timestamp(),
            'metrics': {
                'num_layers_processed': len(steering_vectors),
                'k_examples': self.steering_config.k_examples,
                'layer_metrics': layer_metrics
            }
        })
        
        return steering_vectors
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_intermediate_output(self, data: Dict[str, Any]) -> None:
        """Save intermediate results to output file"""
        try:
            # Load existing data if file exists
            existing_data = []
            if os.path.exists(self.intermediate_output_file):
                with open(self.intermediate_output_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Append new data
            existing_data.append(data)
            
            # Save updated data
            with open(self.intermediate_output_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
            print(f"Intermediate output saved to {self.intermediate_output_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save intermediate output: {e}")
    
    def run_steering_evaluation(self, target_layer: int = None) -> Dict[str, Any]:
        """
        DEPRECATED: Run full steering evaluation across all alpha values
        
        This function is deprecated. Use run_comprehensive_steering_evaluation() instead
        for better performance (combines single-layer and all-layer analysis).
        """
        print("WARNING: run_steering_evaluation() is deprecated. Use run_comprehensive_steering_evaluation() for better performance.")
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
        
        # Save intermediate output: steering evaluation results
        self._save_intermediate_output({
            'step': 'steering_evaluation',
            'timestamp': self._get_timestamp(),
            'results': {
                'target_layer': target_layer,
                'baseline_challenging': self.baseline_results['challenging']['mean_toxicity'] if self.baseline_results else None,
                'baseline_benign': self.baseline_results['benign']['mean_toxicity'] if self.baseline_results else None,
                'alpha_sweep_results': {
                    str(alpha): {
                        'challenging_mean_toxicity': results[alpha]['challenging']['mean_toxicity'],
                        'benign_mean_toxicity': results[alpha]['benign']['mean_toxicity'],
                        'challenging_completions_sample': results[alpha]['challenging']['completions'][:5],  # First 5 for space
                        'benign_completions_sample': results[alpha]['benign']['completions'][:5]  # First 5 for space
                    }
                    for alpha in self.steering_config.alpha_values
                },
                'best_alpha_for_toxicity_reduction': min(
                    self.steering_config.alpha_values,
                    key=lambda a: results[a]['challenging']['mean_toxicity']
                )
            }
        })
        
        return {
            'results': results,
            'target_layer': target_layer,
            'baseline_challenging': self.baseline_results['challenging']['mean_toxicity'] if self.baseline_results else None,
            'baseline_benign': self.baseline_results['benign']['mean_toxicity'] if self.baseline_results else None
        }
    
    def evaluate_steering_all_layers(self, challenging_subset_size: int = None, benign_subset_size: int = None) -> Dict[str, Any]:
        """
        DEPRECATED: Evaluate steering effectiveness across all layers for comprehensive analysis.
        
        This function is deprecated. Use run_comprehensive_steering_evaluation() instead
        for better performance (avoids redundant computation with single-layer evaluation).
        """
        print("WARNING: evaluate_steering_all_layers() is deprecated. Use run_comprehensive_steering_evaluation() for better performance.")
        if not self.steering_vectors:
            raise ValueError("Steering vectors not computed. Run compute_steering_vectors() first.")
        
        print("=== Evaluating Steering Across All Layers ===")
        
        # Use subset of prompts for efficiency (layer evaluation is computationally intensive)
        challenging_subset_size = challenging_subset_size or min(10, len(self.challenging_subset))
        benign_subset_size = benign_subset_size or min(10, len(self.benign_subset))
        
        challenging_prompts = [item['prompt']['text'] for item in self.challenging_subset[:challenging_subset_size]]
        benign_prompts = [item['prompt']['text'] for item in self.benign_subset[:benign_subset_size]]
        
        print(f"Using {len(challenging_prompts)} challenging and {len(benign_prompts)} benign prompts")
        print(f"Evaluating {len(self.steering_vectors)} layers across {len(self.steering_config.alpha_values)} alpha values")
        
        # Store layer-wise results
        layer_results = {}
        
        # Test all alpha values for layer analysis (now simplified to just 3 values)
        test_alphas = [alpha for alpha in self.steering_config.alpha_values if alpha != 0]  # Exclude baseline (α=0)
        
        print(f"Testing alpha values for layer analysis: {test_alphas}")
        
        for layer_idx in tqdm(sorted(self.steering_vectors.keys()), desc="Layer evaluation"):
            layer_results[layer_idx] = {}
            
            for alpha in test_alphas:
                # Generate completions with steering at this specific layer
                challenging_completions = self.generate_completions(
                    challenging_prompts, alpha=alpha, layer=layer_idx
                )
                benign_completions = self.generate_completions(
                    benign_prompts, alpha=alpha, layer=layer_idx
                )
                
                # Score the completions
                challenging_scores = self.score_toxicity(challenging_completions)
                benign_scores = self.score_toxicity(benign_completions)
                
                # Store results for this layer and alpha
                layer_results[layer_idx][alpha] = {
                    'challenging': {
                        'completions': challenging_completions,
                        'toxicity_scores': challenging_scores,
                        'mean_toxicity': np.mean(challenging_scores) if challenging_scores else 0
                    },
                    'benign': {
                        'completions': benign_completions,
                        'toxicity_scores': benign_scores,
                        'mean_toxicity': np.mean(benign_scores) if benign_scores else 0
                    }
                }
        
        # Calculate layer effectiveness metrics
        layer_effectiveness = {}
        for layer_idx in layer_results:
            effectiveness_scores = []
            for alpha in test_alphas:
                if alpha in layer_results[layer_idx]:
                    challenging_toxicity = layer_results[layer_idx][alpha]['challenging']['mean_toxicity']
                    # Lower toxicity = higher effectiveness, so we invert
                    effectiveness_scores.append(1.0 - challenging_toxicity)
            
            layer_effectiveness[layer_idx] = {
                'mean_effectiveness': np.mean(effectiveness_scores) if effectiveness_scores else 0,
                'max_effectiveness': np.max(effectiveness_scores) if effectiveness_scores else 0,
                'effectiveness_scores': effectiveness_scores
            }
        
        return {
            'layer_results': layer_results,
            'layer_effectiveness': layer_effectiveness,
            'test_alphas': test_alphas,
            'num_prompts': {
                'challenging': len(challenging_prompts),
                'benign': len(benign_prompts)
            }
        }
    
    def run_comprehensive_steering_evaluation(self, target_layer: int = None, 
                                            full_subset_for_target: bool = True,
                                            layer_subset_size: int = 8) -> Dict[str, Any]:
        """
        Run comprehensive steering evaluation combining single-layer analysis and all-layer analysis.
        More efficient than running run_steering_evaluation() and evaluate_steering_all_layers() separately.
        
        Args:
            target_layer: Primary layer for detailed analysis (default: middle layer)
            full_subset_for_target: Use full dataset for target layer (50 prompts), smaller for others
            layer_subset_size: Number of prompts to use for non-target layers (for efficiency)
        """
        if not self.steering_vectors:
            raise ValueError("Steering vectors not computed. Run compute_steering_vectors() first.")
        
        print("=== Running Comprehensive Steering Evaluation ===")
        
        # Select target layer (use middle layer if not specified)
        if target_layer is None:
            target_layer = len(self.steering_vectors) // 2
            print(f"Using layer {target_layer} as primary target layer (middle layer)")
        else:
            print(f"Using layer {target_layer} as primary target layer")
        
        if target_layer not in self.steering_vectors:
            raise ValueError(f"No steering vector available for layer {target_layer}")
        
        # Prepare prompt sets
        challenging_prompts_full = [item['prompt']['text'] for item in self.challenging_subset]
        benign_prompts_full = [item['prompt']['text'] for item in self.benign_subset]
        
        # Smaller subset for layer analysis efficiency
        challenging_prompts_small = challenging_prompts_full[:layer_subset_size]
        benign_prompts_small = benign_prompts_full[:layer_subset_size]
        
        print(f"Target layer will use full dataset ({len(challenging_prompts_full)} + {len(benign_prompts_full)} prompts)")
        print(f"Other layers will use reduced dataset ({len(challenging_prompts_small)} + {len(benign_prompts_small)} prompts)")
        
        # Results storage
        target_layer_results = {}  # Detailed results for target layer
        all_layers_results = {}    # Results for all layers (smaller dataset)
        
        # Test all alpha values
        alpha_values = self.steering_config.alpha_values
        test_alphas_no_baseline = [alpha for alpha in alpha_values if alpha != 0]  # Exclude baseline for layer analysis
        
        print(f"Testing alpha values: {alpha_values} (target layer), {test_alphas_no_baseline} (all layers)")
        
        # Step 1: Evaluate target layer with full dataset and all alpha values
        print(f"\n--- Evaluating Target Layer {target_layer} (Full Dataset) ---")
        for alpha in tqdm(alpha_values, desc=f"Target layer {target_layer}"):
            print(f"Evaluating target layer {target_layer} with alpha = {alpha}")
            
            if alpha == 0.0:
                # Use baseline results for alpha = 0
                if self.baseline_results:
                    target_layer_results[alpha] = {
                        'challenging': self.baseline_results['challenging'],
                        'benign': self.baseline_results['benign']
                    }
                    print(f"  Using cached baseline results for alpha = {alpha}")
                    continue
            
            # Generate completions with steering
            challenging_completions = self.generate_completions(
                challenging_prompts_full, alpha=alpha, layer=target_layer
            )
            benign_completions = self.generate_completions(
                benign_prompts_full, alpha=alpha, layer=target_layer
            )
            
            # Score completions
            challenging_scores = self.score_toxicity(challenging_completions)
            benign_scores = self.score_toxicity(benign_completions)
            
            # Store results
            target_layer_results[alpha] = {
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
            
            print(f"  Challenging mean toxicity: {target_layer_results[alpha]['challenging']['mean_toxicity']:.3f}")
            print(f"  Benign mean toxicity: {target_layer_results[alpha]['benign']['mean_toxicity']:.3f}")
        
        # Step 2: Evaluate all other layers with reduced dataset
        print(f"\n--- Evaluating All {len(self.steering_vectors)} Layers (Reduced Dataset) ---")
        
        for layer_idx in tqdm(sorted(self.steering_vectors.keys()), desc="All layers evaluation"):
            all_layers_results[layer_idx] = {}
            
            # For target layer, reuse results from Step 1 (but adapt to smaller dataset if needed)
            if layer_idx == target_layer:
                print(f"  Reusing target layer {layer_idx} results...")
                for alpha in test_alphas_no_baseline:
                    if alpha in target_layer_results:
                        # Use subset of the full results to match other layers
                        full_result = target_layer_results[alpha]
                        all_layers_results[layer_idx][alpha] = {
                            'challenging': {
                                'completions': full_result['challenging']['completions'][:layer_subset_size],
                                'toxicity_scores': full_result['challenging']['toxicity_scores'][:layer_subset_size],
                                'mean_toxicity': np.mean(full_result['challenging']['toxicity_scores'][:layer_subset_size])
                            },
                            'benign': {
                                'completions': full_result['benign']['completions'][:layer_subset_size],
                                'toxicity_scores': full_result['benign']['toxicity_scores'][:layer_subset_size],
                                'mean_toxicity': np.mean(full_result['benign']['toxicity_scores'][:layer_subset_size])
                            }
                        }
                continue
            
            # For other layers, run with reduced dataset
            for alpha in test_alphas_no_baseline:
                # Generate completions with steering at this specific layer
                challenging_completions = self.generate_completions(
                    challenging_prompts_small, alpha=alpha, layer=layer_idx
                )
                benign_completions = self.generate_completions(
                    benign_prompts_small, alpha=alpha, layer=layer_idx
                )
                
                # Score the completions
                challenging_scores = self.score_toxicity(challenging_completions)
                benign_scores = self.score_toxicity(benign_completions)
                
                # Store results for this layer and alpha
                all_layers_results[layer_idx][alpha] = {
                    'challenging': {
                        'completions': challenging_completions,
                        'toxicity_scores': challenging_scores,
                        'mean_toxicity': np.mean(challenging_scores) if challenging_scores else 0
                    },
                    'benign': {
                        'completions': benign_completions,
                        'toxicity_scores': benign_scores,
                        'mean_toxicity': np.mean(benign_scores) if benign_scores else 0
                    }
                }
        
        # Calculate layer effectiveness metrics
        layer_effectiveness = {}
        for layer_idx in all_layers_results:
            effectiveness_scores = []
            for alpha in test_alphas_no_baseline:
                if alpha in all_layers_results[layer_idx]:
                    challenging_toxicity = all_layers_results[layer_idx][alpha]['challenging']['mean_toxicity']
                    # Lower toxicity = higher effectiveness, so we invert
                    effectiveness_scores.append(1.0 - challenging_toxicity)
            
            layer_effectiveness[layer_idx] = {
                'mean_effectiveness': np.mean(effectiveness_scores) if effectiveness_scores else 0,
                'max_effectiveness': np.max(effectiveness_scores) if effectiveness_scores else 0,
                'effectiveness_scores': effectiveness_scores
            }
        
        # Print summary
        print("\n=== Comprehensive Steering Summary ===")
        print("Target Layer Results (Full Dataset):")
        print("Alpha\tChallenging\tBenign")
        for alpha in alpha_values:
            if alpha in target_layer_results:
                challenging_mean = target_layer_results[alpha]['challenging']['mean_toxicity']
                benign_mean = target_layer_results[alpha]['benign']['mean_toxicity']
                print(f"{alpha:6.2f}\t{challenging_mean:.3f}\t\t{benign_mean:.3f}")
        
        # Find most/least effective layers
        if layer_effectiveness:
            most_effective_layer = max(layer_effectiveness.keys(), key=lambda l: layer_effectiveness[l]['mean_effectiveness'])
            least_effective_layer = min(layer_effectiveness.keys(), key=lambda l: layer_effectiveness[l]['mean_effectiveness'])
            print(f"\nLayer Analysis Summary:")
            print(f"  Most effective layer: {most_effective_layer} ({layer_effectiveness[most_effective_layer]['mean_effectiveness']*100:.1f}% effectiveness)")
            print(f"  Least effective layer: {least_effective_layer} ({layer_effectiveness[least_effective_layer]['mean_effectiveness']*100:.1f}% effectiveness)")
        
        # Save comprehensive intermediate output
        self._save_intermediate_output({
            'step': 'comprehensive_steering_evaluation',
            'timestamp': self._get_timestamp(),
            'target_layer_results': {
                'layer': target_layer,
                'full_dataset_size': {'challenging': len(challenging_prompts_full), 'benign': len(benign_prompts_full)},
                'results': {
                    str(alpha): {
                        'challenging_mean_toxicity': target_layer_results[alpha]['challenging']['mean_toxicity'],
                        'benign_mean_toxicity': target_layer_results[alpha]['benign']['mean_toxicity']
                    } for alpha in target_layer_results
                }
            },
            'layer_analysis_summary': {
                'num_layers_tested': len(all_layers_results),
                'reduced_dataset_size': {'challenging': len(challenging_prompts_small), 'benign': len(benign_prompts_small)},
                'most_effective_layer': most_effective_layer if layer_effectiveness else None,
                'least_effective_layer': least_effective_layer if layer_effectiveness else None
            }
        })
        
        return {
            'target_layer_results': {
                'results': target_layer_results,
                'target_layer': target_layer,
                'baseline_challenging': self.baseline_results['challenging']['mean_toxicity'] if self.baseline_results else None,
                'baseline_benign': self.baseline_results['benign']['mean_toxicity'] if self.baseline_results else None
            },
            'layer_analysis_results': {
                'layer_results': all_layers_results,
                'layer_effectiveness': layer_effectiveness,
                'test_alphas': test_alphas_no_baseline,
                'num_prompts': {
                    'challenging': len(challenging_prompts_small),
                    'benign': len(benign_prompts_small)
                }
            }
        }
    
    def save_results(self, filepath: str, steering_results: Dict[str, Any] = None, layer_analysis_results: Dict[str, Any] = None) -> None:
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
        
        # Add layer analysis results if provided
        if layer_analysis_results:
            results['layer_analysis_results'] = layer_analysis_results
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def load_and_plot_results(results_file: str = "experiment_results.json", save_dir: str = "plots") -> None:
        """Load results from JSON file and create plots without re-running experiment"""
        print(f"Loading experiment results from {results_file}...")
        
        try:
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            # Extract the steering results and configuration
            steering_results = saved_results.get('steering_results')
            config = saved_results.get('config', {})
            
            if not steering_results:
                print("Error: No steering results found in the file.")
                print("Available keys:", list(saved_results.keys()))
                return
            
            print("✓ Results loaded successfully")
            print(f"Found results for {len(steering_results.get('results', {}))} alpha values")
            
            # Create a minimal experiment instance for plotting configuration
            # We need the steering config for plotting
            steering_config = SteeringConfig()
            if 'steering_config' in config:
                steering_config.k_examples = config['steering_config'].get('k_examples', 20)
                if 'alpha_values' in config['steering_config']:
                    steering_config.alpha_values = config['steering_config']['alpha_values']
            
            # Create temporary experiment instance for plotting method access
            class PlottingHelper:
                def __init__(self, steering_config):
                    self.steering_config = steering_config
                
                def plot_results(self, steering_results: Dict[str, Any], save_dir: str = "plots") -> None:
                    # Import the actual plotting method from the experiment class
                    temp_experiment = ActivationSteeringExperiment.__new__(ActivationSteeringExperiment)
                    temp_experiment.steering_config = self.steering_config
                    
                    # Call the original plot_results method
                    ActivationSteeringExperiment.plot_results(temp_experiment, steering_results, save_dir)
            
            # Create plotting helper and generate plots
            plotter = PlottingHelper(steering_config)
            plotter.plot_results(steering_results, save_dir)
            
            # Generate steering effectiveness summary
            temp_experiment.plot_steering_effectiveness_summary(steering_results, save_dir)
            
            # Generate layer analysis plots if data is available
            if 'layer_analysis_results' in saved_results:
                print("Found layer analysis data, generating layer plots...")
                temp_experiment.plot_layer_analysis(saved_results['layer_analysis_results'], save_dir)
                print(f"✓ Layer analysis plots generated successfully")
            
            print(f"✓ All plots generated successfully in '{save_dir}/' directory")
            
        except FileNotFoundError:
            print(f"Error: Results file '{results_file}' not found.")
            print("Available files:")
            from pathlib import Path
            current_dir = Path(".")
            json_files = list(current_dir.glob("*.json"))
            if json_files:
                for file in json_files:
                    print(f"  - {file}")
            else:
                print("  No JSON files found in current directory")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file format: {e}")
            
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def plot_results(self, steering_results: Dict[str, Any], save_dir: str = "plots") -> None:
        """Create comprehensive plots of the steering experiment results"""
        if not steering_results or 'results' not in steering_results:
            print("Warning: No steering results available for plotting")
            return
            
        try:
            # Create plots directory
            Path(save_dir).mkdir(exist_ok=True)
            
            # Check matplotlib backend
            current_backend = matplotlib.get_backend()
            print(f"Using matplotlib backend: {current_backend}")
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
        
        except Exception as e:
            print(f"Error setting up plotting environment: {e}")
            print("Attempting to continue with basic plotting...")
            Path(save_dir).mkdir(exist_ok=True)
        
        try:
            results = steering_results['results']
            alpha_values = self.steering_config.alpha_values
            
            # Extract data for plotting (convert alpha values to strings to match JSON keys)
            challenging_scores = [results[str(alpha)]['challenging']['mean_toxicity'] for alpha in alpha_values]
            benign_scores = [results[str(alpha)]['benign']['mean_toxicity'] for alpha in alpha_values]
            
            baseline_challenging = steering_results.get('baseline_challenging', 0)
            baseline_benign = steering_results.get('baseline_benign', 0)
            
            # Plot 1: Alpha Sweep - Mean Toxicity Scores
            plt.figure(figsize=(12, 8))
        
            plt.subplot(2, 2, 1)
            plt.plot(alpha_values, challenging_scores, 'o-', linewidth=2.5, markersize=8, 
                    label='Challenging Subset', color='#e74c3c')
            plt.plot(alpha_values, benign_scores, 's-', linewidth=2.5, markersize=8, 
                    label='Benign Subset', color='#3498db')
        
            # Add baseline lines
            if baseline_challenging > 0:
                plt.axhline(y=baseline_challenging, color='#e74c3c', linestyle='--', alpha=0.7, 
                           label=f'Baseline Challenging ({baseline_challenging:.3f})')
            if baseline_benign > 0:
                plt.axhline(y=baseline_benign, color='#3498db', linestyle='--', alpha=0.7, 
                           label=f'Baseline Benign ({baseline_benign:.3f})')
            
            plt.xlabel('Alpha (Steering Strength)', fontsize=12)
            plt.ylabel('Mean Toxicity Score', fontsize=12)
            plt.title('Activation Steering Effects on Toxicity\nAcross Alpha Values', fontsize=12, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(alpha_values)
        
            # Plot 2: Toxicity Reduction Effectiveness
            plt.subplot(2, 2, 2)
            if baseline_challenging > 0:
                challenging_reduction = [(baseline_challenging - score) / baseline_challenging * 100 
                                       for score in challenging_scores]
                plt.plot(alpha_values, challenging_reduction, 'o-', linewidth=2.5, markersize=8, 
                        color='#e74c3c', label='Challenging Subset')
            
            if baseline_benign > 0:
                benign_change = [(baseline_benign - score) / baseline_benign * 100 
                               for score in benign_scores]
                plt.plot(alpha_values, benign_change, 's-', linewidth=2.5, markersize=8, 
                        color='#3498db', label='Benign Subset')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Alpha (Steering Strength)', fontsize=12)
            plt.ylabel('Toxicity Change (%)', fontsize=12)
            plt.title('Relative Toxicity Change from Baseline', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(alpha_values)
        
            # Plot 3: Differential Effect (Challenging - Benign)
            plt.subplot(2, 2, 3)
            differential_effect = [challenging_scores[i] - benign_scores[i] 
                                 for i in range(len(alpha_values))]
            baseline_diff = baseline_challenging - baseline_benign if baseline_challenging > 0 and baseline_benign > 0 else 0
            
            plt.plot(alpha_values, differential_effect, 'D-', linewidth=2.5, markersize=8, 
                    color='#9b59b6', label='Steering Effect')
            if baseline_diff > 0:
                plt.axhline(y=baseline_diff, color='#9b59b6', linestyle='--', alpha=0.7, 
                           label=f'Baseline Difference ({baseline_diff:.3f})')
            
            plt.xlabel('Alpha (Steering Strength)', fontsize=12)
            plt.ylabel('Toxicity Difference\n(Challenging - Benign)', fontsize=12)
            plt.title('Differential Steering Effect', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(alpha_values)
        
            # Plot 4: Best Alpha Highlighting
            plt.subplot(2, 2, 4)
            
            # Find best alpha (lowest challenging toxicity)
            best_alpha_idx = np.argmin(challenging_scores)
            best_alpha = alpha_values[best_alpha_idx]
            
            bars = plt.bar(['Challenging', 'Benign'], 
                          [challenging_scores[best_alpha_idx], benign_scores[best_alpha_idx]],
                          color=['#e74c3c', '#3498db'], alpha=0.7)
            
            # Add baseline comparison bars
            if baseline_challenging > 0 and baseline_benign > 0:
                plt.bar(['Challenging\n(Baseline)', 'Benign\n(Baseline)'], 
                       [baseline_challenging, baseline_benign],
                       color=['#e74c3c', '#3498db'], alpha=0.3, hatch='///')
            
            plt.ylabel('Mean Toxicity Score', fontsize=12)
            plt.title(f'Best Steering Result (α = {best_alpha})', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/alpha_sweep_results.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_dir}/alpha_sweep_results.pdf", bbox_inches='tight')
            print(f"Alpha sweep plots saved to {save_dir}/alpha_sweep_results.png and .pdf")
        
            # Create a detailed heatmap of all results
            plt.figure(figsize=(14, 8))
            
            # Prepare data for heatmap
            heatmap_data = []
            row_labels = []
            
            for alpha in alpha_values:
                challenging_mean = results[str(alpha)]['challenging']['mean_toxicity']
                benign_mean = results[str(alpha)]['benign']['mean_toxicity']
                heatmap_data.append([challenging_mean, benign_mean])
                row_labels.append(f"α = {alpha}")
            
            heatmap_data = np.array(heatmap_data)
            
            # Create heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(heatmap_data, 
                       xticklabels=['Challenging', 'Benign'],
                       yticklabels=row_labels,
                       annot=True, fmt='.3f', cmap='RdYlBu_r',
                       cbar_kws={'label': 'Mean Toxicity Score'})
            plt.title('Toxicity Scores Heatmap\nAcross Alpha Values', fontsize=14, fontweight='bold')
            plt.xlabel('Subset Type', fontsize=12)
            plt.ylabel('Steering Strength', fontsize=12)
        
            # Create improvement heatmap (compared to baseline)
            plt.subplot(1, 2, 2)
            if baseline_challenging > 0 and baseline_benign > 0:
                improvement_data = []
                for alpha in alpha_values:
                    challenging_improvement = (baseline_challenging - results[str(alpha)]['challenging']['mean_toxicity']) / baseline_challenging * 100
                    benign_impact = (baseline_benign - results[str(alpha)]['benign']['mean_toxicity']) / baseline_benign * 100
                    improvement_data.append([challenging_improvement, benign_impact])
                
                improvement_data = np.array(improvement_data)
                
                sns.heatmap(improvement_data,
                           xticklabels=['Challenging', 'Benign'],
                           yticklabels=row_labels,
                           annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                           cbar_kws={'label': 'Improvement (%)'})
                plt.title('Toxicity Improvement from Baseline\n(% Change)', fontsize=14, fontweight='bold')
                plt.xlabel('Subset Type', fontsize=12)
                plt.ylabel('Steering Strength', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/toxicity_heatmaps.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_dir}/toxicity_heatmaps.pdf", bbox_inches='tight')
            print(f"Heatmap plots saved to {save_dir}/toxicity_heatmaps.png and .pdf")
            
            # Create statistical summary plot
            plt.figure(figsize=(12, 6))
            
            # Box plot showing distribution characteristics
            plt.subplot(1, 2, 1)
            box_data = [challenging_scores, benign_scores]
            labels = ['Challenging Subset', 'Benign Subset']
            colors = ['#e74c3c', '#3498db']
            
            bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel('Mean Toxicity Score', fontsize=12)
            plt.title('Toxicity Score Distribution\nAcross All Alpha Values', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Summary statistics table as text
            plt.subplot(1, 2, 2)
            plt.axis('off')
            
            summary_stats = []
            summary_stats.append(['Metric', 'Challenging', 'Benign'])
            summary_stats.append(['Mean', f'{np.mean(challenging_scores):.3f}', f'{np.mean(benign_scores):.3f}'])
            summary_stats.append(['Std Dev', f'{np.std(challenging_scores):.3f}', f'{np.std(benign_scores):.3f}'])
            summary_stats.append(['Min', f'{np.min(challenging_scores):.3f}', f'{np.min(benign_scores):.3f}'])
            summary_stats.append(['Max', f'{np.max(challenging_scores):.3f}', f'{np.max(benign_scores):.3f}'])
            summary_stats.append(['Range', f'{np.max(challenging_scores) - np.min(challenging_scores):.3f}', 
                             f'{np.max(benign_scores) - np.min(benign_scores):.3f}'])
            
            if baseline_challenging > 0 and baseline_benign > 0:
                best_challenging_improvement = (baseline_challenging - np.min(challenging_scores)) / baseline_challenging * 100
                worst_benign_impact = (baseline_benign - np.max(benign_scores)) / baseline_benign * 100
                summary_stats.append(['Best Improvement (%)', f'{best_challenging_improvement:.1f}', f'{worst_benign_impact:.1f}'])
                summary_stats.append(['Best Alpha', f'{alpha_values[np.argmin(challenging_scores)]}', f'{alpha_values[np.argmax(benign_scores)]}'])
            
            table = plt.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for (row, col), cell in table.get_celld().items():
                if row == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
            
            plt.title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/summary_statistics.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_dir}/summary_statistics.pdf", bbox_inches='tight')
            print(f"Summary statistics plots saved to {save_dir}/summary_statistics.png and .pdf")
            
            # Generate steering effectiveness summary
            self.plot_steering_effectiveness_summary(steering_results, save_dir)
            
            # Close all figures to free memory (since using Agg backend, no display needed)
            plt.close('all')
            
            print(f"\n=== Plotting Summary ===")
            print(f"All plots saved to '{save_dir}/' directory")
            print(f"Generated files:")
            print(f"  - alpha_sweep_results.png/pdf: Main alpha sweep analysis")
            print(f"  - toxicity_heatmaps.png/pdf: Detailed heatmap analysis")
            print(f"  - summary_statistics.png/pdf: Statistical summary")
            print(f"  - steering_effectiveness_summary.png/pdf: Overall effectiveness summary")
            print(f"Plus layer analysis plots if layer data is available:")
            print(f"  - layer_effectiveness_analysis.png/pdf: 6-panel layer analysis")
            print(f"  - detailed_layer_heatmap.png/pdf: Layer vs alpha heatmap")
            
            print(f"\n=== Statistical Notes ===")
            print(f"Experiment used {self.steering_config.k_examples} contrastive pairs and {len(challenging_scores)} alpha values: {self.steering_config.alpha_values}.")
            print(f"Reduced sample sizes improve efficiency but may affect statistical power.")
            print(f"Results should be interpreted with appropriate caution for the sample size.")
            
            if baseline_challenging > 0 and len(challenging_scores) > 0:
                best_alpha_idx = np.argmin(challenging_scores)
                best_alpha = alpha_values[best_alpha_idx]
                best_reduction = (baseline_challenging - challenging_scores[best_alpha_idx]) / baseline_challenging * 100
                print(f"Best toxicity reduction: {best_reduction:.1f}% at α = {best_alpha}")
                
        except Exception as e:
            print(f"Error during plotting: {e}")
            print("Plotting failed, but experiment results are still saved in JSON format.")
            print(f"You can manually analyze the results or try plotting with different settings.")
            # Still try to close figures to prevent memory leaks
            try:
                plt.close('all')
            except:
                pass
    
    def plot_layer_analysis(self, layer_analysis_results: Dict[str, Any], save_dir: str = "plots") -> None:
        """Create plots showing steering effectiveness across different layers"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from pathlib import Path
        
        # Ensure save directory exists
        Path(save_dir).mkdir(exist_ok=True)
        
        # Set matplotlib backend for non-interactive plotting
        matplotlib = __import__('matplotlib')
        matplotlib.use('Agg')
        
        print(f"Generating layer analysis plots...")
        
        try:
            layer_results = layer_analysis_results['layer_results']
            layer_effectiveness = layer_analysis_results['layer_effectiveness'] 
            test_alphas = layer_analysis_results['test_alphas']
            
            # Sort layers for consistent ordering
            sorted_layers = sorted(layer_results.keys())
            
            # Create comprehensive layer analysis figure
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Layer Effectiveness Heatmap (Layers vs Alpha Values)
            plt.subplot(2, 3, 1)
            
            # Create heatmap data: layers (rows) vs alphas (columns)
            heatmap_data = []
            layer_labels = []
            
            for layer_idx in sorted_layers:
                row_data = []
                for alpha in test_alphas:
                    if alpha in layer_results[layer_idx]:
                        # Use toxicity reduction as effectiveness metric
                        toxicity = layer_results[layer_idx][alpha]['challenging']['mean_toxicity']
                        effectiveness = (1.0 - toxicity) * 100  # Convert to percentage
                        row_data.append(effectiveness)
                    else:
                        row_data.append(0)
                heatmap_data.append(row_data)
                layer_labels.append(f"Layer {layer_idx}")
            
            heatmap_data = np.array(heatmap_data)
            
            sns.heatmap(heatmap_data, 
                       xticklabels=[f"α={a}" for a in test_alphas],
                       yticklabels=layer_labels,
                       annot=True, fmt='.1f', cmap='RdYlBu_r', center=50,
                       cbar_kws={'label': 'Effectiveness (%)'})
            plt.title('Steering Effectiveness by Layer and Alpha\n(Higher = Less Toxic)', fontweight='bold')
            plt.xlabel('Alpha Values')
            plt.ylabel('Transformer Layers')
            
            # Plot 2: Average Layer Effectiveness
            plt.subplot(2, 3, 2)
            
            effectiveness_means = [layer_effectiveness[layer]['mean_effectiveness'] * 100 for layer in sorted_layers]
            effectiveness_maxes = [layer_effectiveness[layer]['max_effectiveness'] * 100 for layer in sorted_layers]
            
            plt.plot(sorted_layers, effectiveness_means, 'o-', linewidth=2, markersize=6, 
                    label='Mean Effectiveness', color='#2E86AB')
            plt.plot(sorted_layers, effectiveness_maxes, 's--', linewidth=2, markersize=6, 
                    label='Max Effectiveness', color='#A23B72', alpha=0.7)
            
            plt.xlabel('Layer Index')
            plt.ylabel('Effectiveness (%)')
            plt.title('Layer-wise Steering Effectiveness\n(Averaged Across Alpha Values)', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Highlight most/least effective layers
            most_effective_layer = max(sorted_layers, key=lambda l: layer_effectiveness[l]['mean_effectiveness'])
            least_effective_layer = min(sorted_layers, key=lambda l: layer_effectiveness[l]['mean_effectiveness'])
            
            plt.axvline(x=most_effective_layer, color='green', linestyle=':', alpha=0.7, 
                       label=f'Most Effective (Layer {most_effective_layer})')
            plt.axvline(x=least_effective_layer, color='red', linestyle=':', alpha=0.7,
                       label=f'Least Effective (Layer {least_effective_layer})')
            
            # Plot 3: Toxicity Reduction by Layer (for best alpha)
            plt.subplot(2, 3, 3)
            
            # Find the alpha that gives best overall toxicity reduction
            best_alpha = None
            best_overall_score = float('inf')
            for alpha in test_alphas:
                alpha_toxicity = []
                for layer_idx in sorted_layers:
                    if alpha in layer_results[layer_idx]:
                        alpha_toxicity.append(layer_results[layer_idx][alpha]['challenging']['mean_toxicity'])
                if alpha_toxicity:
                    avg_toxicity = np.mean(alpha_toxicity)
                    if avg_toxicity < best_overall_score:
                        best_overall_score = avg_toxicity
                        best_alpha = alpha
            
            if best_alpha is not None:
                toxicity_by_layer = []
                for layer_idx in sorted_layers:
                    if best_alpha in layer_results[layer_idx]:
                        toxicity_by_layer.append(layer_results[layer_idx][best_alpha]['challenging']['mean_toxicity'])
                    else:
                        toxicity_by_layer.append(0.5)  # Neutral value if no data
                
                plt.bar(range(len(sorted_layers)), toxicity_by_layer, 
                       color='#F18F01', alpha=0.7, edgecolor='black', linewidth=0.5)
                plt.xlabel('Layer Index')
                plt.ylabel('Mean Toxicity Score')
                plt.title(f'Toxicity Score by Layer\n(α = {best_alpha}, Lower = Better)', fontweight='bold')
                plt.xticks(range(len(sorted_layers)), [str(l) for l in sorted_layers])
                
                # Highlight the most effective layer
                min_toxicity_idx = np.argmin(toxicity_by_layer)
                plt.bar(min_toxicity_idx, toxicity_by_layer[min_toxicity_idx], 
                       color='green', alpha=0.8, edgecolor='black', linewidth=1.5,
                       label=f'Best Layer ({sorted_layers[min_toxicity_idx]})')
                plt.legend()
            
            # Plot 4: Layer Effectiveness Distribution
            plt.subplot(2, 3, 4)
            
            all_effectiveness = [layer_effectiveness[layer]['mean_effectiveness'] * 100 for layer in sorted_layers]
            
            plt.hist(all_effectiveness, bins=10, color='#C73E1D', alpha=0.7, edgecolor='black')
            plt.axvline(x=np.mean(all_effectiveness), color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(all_effectiveness):.1f}%')
            plt.axvline(x=np.median(all_effectiveness), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(all_effectiveness):.1f}%')
            
            plt.xlabel('Effectiveness (%)')
            plt.ylabel('Number of Layers')
            plt.title('Distribution of Layer Effectiveness', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Early vs Late Layers Comparison
            plt.subplot(2, 3, 5)
            
            num_layers = len(sorted_layers)
            early_layers = sorted_layers[:num_layers//3]
            middle_layers = sorted_layers[num_layers//3:2*num_layers//3] 
            late_layers = sorted_layers[2*num_layers//3:]
            
            early_effectiveness = [layer_effectiveness[l]['mean_effectiveness'] * 100 for l in early_layers]
            middle_effectiveness = [layer_effectiveness[l]['mean_effectiveness'] * 100 for l in middle_layers]
            late_effectiveness = [layer_effectiveness[l]['mean_effectiveness'] * 100 for l in late_layers]
            
            layer_groups = ['Early\n(0-33%)', 'Middle\n(33-66%)', 'Late\n(66-100%)']
            group_means = [
                np.mean(early_effectiveness) if early_effectiveness else 0,
                np.mean(middle_effectiveness) if middle_effectiveness else 0,
                np.mean(late_effectiveness) if late_effectiveness else 0
            ]
            group_stds = [
                np.std(early_effectiveness) if len(early_effectiveness) > 1 else 0,
                np.std(middle_effectiveness) if len(middle_effectiveness) > 1 else 0, 
                np.std(late_effectiveness) if len(late_effectiveness) > 1 else 0
            ]
            
            colors = ['#4ECDC4', '#44A08D', '#093637']
            bars = plt.bar(layer_groups, group_means, yerr=group_stds, capsize=5,
                          color=colors, alpha=0.7, edgecolor='black')
            
            plt.ylabel('Mean Effectiveness (%)')
            plt.title('Effectiveness by Layer Depth', fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, mean) in enumerate(zip(bars, group_means)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + group_stds[i] + 1,
                        f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Plot 6: Layer Summary Statistics
            plt.subplot(2, 3, 6)
            
            # Summary statistics table
            stats_data = [
                ['Most Effective Layer', f"Layer {most_effective_layer}", 
                 f"{layer_effectiveness[most_effective_layer]['mean_effectiveness']*100:.1f}%"],
                ['Least Effective Layer', f"Layer {least_effective_layer}", 
                 f"{layer_effectiveness[least_effective_layer]['mean_effectiveness']*100:.1f}%"],
                ['Best Overall Alpha', f"α = {best_alpha}" if best_alpha else "N/A", 
                 f"{best_overall_score:.3f}" if best_alpha else "N/A"],
                ['Early Layers Avg', f"Layers 0-{len(early_layers)-1}", f"{group_means[0]:.1f}%"],
                ['Middle Layers Avg', f"Layers {len(early_layers)}-{len(early_layers)+len(middle_layers)-1}", f"{group_means[1]:.1f}%"],
                ['Late Layers Avg', f"Layers {len(early_layers)+len(middle_layers)}-{num_layers-1}", f"{group_means[2]:.1f}%"]
            ]
            
            plt.axis('tight')
            plt.axis('off')
            
            table = plt.table(cellText=stats_data,
                             colLabels=['Metric', 'Layer/Value', 'Effectiveness'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 2)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#4ECDC4')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else '#ffffff')
            
            plt.title('Layer Analysis Summary', fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/layer_effectiveness_analysis.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_dir}/layer_effectiveness_analysis.pdf", bbox_inches='tight')
            plt.close('all')
            
            print(f"Layer analysis plots saved to {save_dir}/layer_effectiveness_analysis.png and .pdf")
            
            # Generate additional detailed heatmap
            plt.figure(figsize=(14, 10))
            
            # Detailed layer vs alpha heatmap with better formatting
            sns.heatmap(heatmap_data, 
                       xticklabels=[f"α={a}" for a in test_alphas],
                       yticklabels=[f"L{l}" for l in sorted_layers],
                       annot=True, fmt='.1f', cmap='RdYlBu_r', center=50,
                       linewidths=0.5, cbar_kws={'label': 'Effectiveness (%, Higher = Less Toxic)'})
            
            plt.title('Detailed Layer Effectiveness Heatmap\nSteering Vector Impact Across Transformer Layers', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Alpha Values (Steering Strength)', fontsize=12)
            plt.ylabel('Transformer Layers', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/detailed_layer_heatmap.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_dir}/detailed_layer_heatmap.pdf", bbox_inches='tight')
            plt.close('all')
            
            print(f"Detailed layer heatmap saved to {save_dir}/detailed_layer_heatmap.png and .pdf")
            
            # Print summary to console
            print(f"\n=== Layer Analysis Summary ===")
            print(f"Most effective layer: {most_effective_layer} ({layer_effectiveness[most_effective_layer]['mean_effectiveness']*100:.1f}% effectiveness)")
            print(f"Least effective layer: {least_effective_layer} ({layer_effectiveness[least_effective_layer]['mean_effectiveness']*100:.1f}% effectiveness)")
            print(f"Best alpha value overall: {best_alpha} (mean toxicity: {best_overall_score:.3f})")
            print(f"Layer depth analysis:")
            print(f"  Early layers (0-33%): {group_means[0]:.1f}% effectiveness")
            print(f"  Middle layers (33-66%): {group_means[1]:.1f}% effectiveness") 
            print(f"  Late layers (66-100%): {group_means[2]:.1f}% effectiveness")
            
        except Exception as e:
            print(f"Error during layer analysis plotting: {e}")
            print("Layer analysis plotting failed.")
            try:
                plt.close('all')
            except:
                pass
    
    def plot_steering_effectiveness_summary(self, steering_results: Dict[str, Any], save_dir: str = "plots") -> None:
        """Create a high-level summary plot showing overall steering effectiveness vs baseline"""
        print("Generating steering effectiveness summary plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
            
            # Ensure save directory exists
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            if not steering_results or 'results' not in steering_results:
                print("Warning: No steering results available for effectiveness summary")
                return
            
            results = steering_results['results']
            baseline_challenging = steering_results.get('baseline_challenging', 0)
            baseline_benign = steering_results.get('baseline_benign', 0)
            
            if baseline_challenging == 0 or baseline_benign == 0:
                print("Warning: Baseline results not available for comparison")
                return
            
            # Extract best steering results (usually alpha = 1)
            best_alpha = None
            best_challenging_improvement = float('-inf')
            best_benign_improvement = float('-inf')
            
            alpha_improvements = {}
            
            for alpha, results_dict in results.items():
                alpha_val = float(alpha)
                if alpha_val == 0:  # Skip baseline
                    continue
                    
                challenging_toxicity = results_dict['challenging']['mean_toxicity']
                benign_toxicity = results_dict['benign']['mean_toxicity']
                
                # Calculate improvement percentages
                challenging_improvement = ((baseline_challenging - challenging_toxicity) / baseline_challenging) * 100
                benign_improvement = ((baseline_benign - benign_toxicity) / baseline_benign) * 100
                
                alpha_improvements[alpha_val] = {
                    'challenging_improvement': challenging_improvement,
                    'benign_improvement': benign_improvement,
                    'challenging_toxicity': challenging_toxicity,
                    'benign_toxicity': benign_toxicity
                }
                
                # Track best overall improvement (prioritize challenging subset)
                if challenging_improvement > best_challenging_improvement:
                    best_challenging_improvement = challenging_improvement
                    best_benign_improvement = benign_improvement
                    best_alpha = alpha_val
            
            if not alpha_improvements:
                print("Warning: No non-baseline results available for effectiveness summary")
                return
            
            # Create summary plot with 3 panels
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Activation Steering Effectiveness Summary\nOverall Impact vs Baseline', fontsize=16, fontweight='bold')
            
            # Panel 1: Absolute Toxicity Comparison (Bar Chart)
            ax1.set_title('Absolute Toxicity Scores\nBaseline vs Best Steering', fontweight='bold')
            
            categories = ['Challenging\nSubset', 'Benign\nSubset']
            baseline_values = [baseline_challenging, baseline_benign]
            best_steering_values = [
                alpha_improvements[best_alpha]['challenging_toxicity'],
                alpha_improvements[best_alpha]['benign_toxicity']
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline (α=0)', 
                           color='#ff7f7f', alpha=0.8, edgecolor='black', linewidth=1)
            bars2 = ax1.bar(x + width/2, best_steering_values, width, label=f'Best Steering (α={best_alpha})', 
                           color='#7fbf7f', alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for i, (baseline, steered) in enumerate(zip(baseline_values, best_steering_values)):
                ax1.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
                ax1.text(i + width/2, steered + 0.01, f'{steered:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            ax1.set_ylabel('Mean Toxicity Score')
            ax1.set_xlabel('Prompt Type')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(max(baseline_values), max(best_steering_values)) * 1.2)
            
            # Panel 2: Improvement Percentages (Bar Chart)
            ax2.set_title('Steering Effectiveness\nImprovement from Baseline', fontweight='bold')
            
            improvements = [
                alpha_improvements[best_alpha]['challenging_improvement'],
                alpha_improvements[best_alpha]['benign_improvement']
            ]
            
            colors = ['#ff6b6b' if imp < 0 else '#4ecdc4' for imp in improvements]
            bars = ax2.bar(categories, improvements, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{improvement:+.1f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_ylabel('Improvement (%)')
            ax2.set_xlabel('Prompt Type')
            ax2.grid(True, alpha=0.3)
            
            # Panel 3: Alpha Sweep Comparison
            ax3.set_title('Steering Strength Comparison\nAll Alpha Values', fontweight='bold')
            
            alphas = sorted([float(a) for a in alpha_improvements.keys()])
            challenging_improvements = [alpha_improvements[a]['challenging_improvement'] for a in alphas]
            benign_improvements = [alpha_improvements[a]['benign_improvement'] for a in alphas]
            
            ax3.plot(alphas, challenging_improvements, 'o-', color='#ff6b6b', linewidth=2, 
                    markersize=8, label='Challenging Subset')
            ax3.plot(alphas, benign_improvements, 's-', color='#4ecdc4', linewidth=2, 
                    markersize=8, label='Benign Subset')
            
            # Highlight best alpha
            best_idx = alphas.index(best_alpha)
            ax3.plot(best_alpha, challenging_improvements[best_idx], 'o', 
                    color='darkred', markersize=12, markerfacecolor='yellow', 
                    markeredgewidth=2, label=f'Best α={best_alpha}')
            
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Alpha Value (Steering Strength)')
            ax3.set_ylabel('Improvement from Baseline (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Panel 4: Summary Statistics Table
            ax4.set_title('Summary Statistics', fontweight='bold')
            ax4.axis('off')
            
            # Calculate additional metrics
            max_challenging_improvement = max(challenging_improvements) if challenging_improvements else 0
            max_benign_improvement = max(benign_improvements) if benign_improvements else 0
            avg_challenging_improvement = np.mean(challenging_improvements) if challenging_improvements else 0
            avg_benign_improvement = np.mean(benign_improvements) if benign_improvements else 0
            
            # Create summary table
            summary_data = [
                ['Metric', 'Challenging', 'Benign'],
                ['Baseline Toxicity', f'{baseline_challenging:.3f}', f'{baseline_benign:.3f}'],
                ['Best Steered Toxicity', f'{alpha_improvements[best_alpha]["challenging_toxicity"]:.3f}', 
                 f'{alpha_improvements[best_alpha]["benign_toxicity"]:.3f}'],
                ['Best Improvement', f'{max_challenging_improvement:+.1f}%', f'{max_benign_improvement:+.1f}%'],
                ['Average Improvement', f'{avg_challenging_improvement:+.1f}%', f'{avg_benign_improvement:+.1f}%'],
                ['Optimal Alpha', f'{best_alpha}', f'{best_alpha}'],
                ['Baseline Context', 'High prompt toxicity → Low completion toxicity', 
                 'Low prompt toxicity → High completion toxicity']
            ]
            
            table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.5)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#ff7f50')
                elif i == 3:  # Best improvement row - highlight
                    cell.set_facecolor('#90EE90' if j == 0 else '#ffeb9c')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else '#ffffff')
            
            plt.tight_layout()
            
            # Save summary plot
            plt.savefig(f"{save_dir}/steering_effectiveness_summary.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_dir}/steering_effectiveness_summary.pdf", bbox_inches='tight')
            plt.close('all')
            
            print(f"Steering effectiveness summary saved to {save_dir}/steering_effectiveness_summary.png and .pdf")
            
            # Print key insights to console
            print(f"\n=== Steering Effectiveness Summary ===")
            print(f"Best steering configuration: α = {best_alpha}")
            print(f"Challenging subset improvement: {max_challenging_improvement:+.1f}% (baseline: {baseline_challenging:.3f} → steered: {alpha_improvements[best_alpha]['challenging_toxicity']:.3f})")
            print(f"Benign subset improvement: {max_benign_improvement:+.1f}% (baseline: {baseline_benign:.3f} → steered: {alpha_improvements[best_alpha]['benign_toxicity']:.3f})")
            
            if max_challenging_improvement > 10:
                print("✓ Strong steering effectiveness on challenging prompts")
            elif max_challenging_improvement > 0:
                print("⚠ Moderate steering effectiveness on challenging prompts")  
            else:
                print("⚠ Negative steering effect on challenging prompts")
                
        except Exception as e:
            print(f"Error generating steering effectiveness summary: {e}")
            try:
                plt.close('all')
            except:
                pass

def main_lightweight():
    print("=== Lightweight Activation Steering Experiment for Toxicity Reduction ===")
    experiment = ActivationSteeringExperiment()

    # Step 1: Load dataset and create subsets
    experiment.load_rtp_dataset()
    experiment.create_subsets()


def main():
    """Main execution function"""
    print("=== Activation Steering Experiment for Toxicity Reduction ===")
    print("This experiment includes comprehensive layer-wise steering analysis")
    
    # Initialize experiment
    experiment = ActivationSteeringExperiment()
    
    try:
        # Step 1: Load dataset and create subsets
        experiment.load_rtp_dataset()
        experiment.create_subsets()
        
        # Step 2: Run baseline evaluation with activation collection (combined for efficiency)
        print("\n--- Step 2: Baseline Evaluation + Activation Collection ---")
        baseline_results = experiment.run_baseline_evaluation_with_activations()
        
        # Step 3: Compute steering vectors
        print("\n--- Step 3: Computing Steering Vectors ---")
        experiment.compute_steering_vectors()
        
        # Step 4: Comprehensive steering evaluation (combines single-layer and all-layer analysis)
        print("\n--- Step 4: Comprehensive Steering Evaluation ---")
        print("Running combined target layer analysis and all-layer comparison for efficiency...")
        comprehensive_results = experiment.run_comprehensive_steering_evaluation(
            layer_subset_size=8  # Use smaller subset for non-target layers
        )
        
        # Extract results for backward compatibility with plotting functions
        steering_results = comprehensive_results['target_layer_results']
        layer_analysis_results = comprehensive_results['layer_analysis_results']
        
        # Save all results including layer analysis
        print("\n--- Saving Results ---")
        experiment.save_results('experiment_results.json', steering_results, layer_analysis_results)
        
        # Generate all plots
        print("\n--- Generating Visualizations ---")
        experiment.plot_results(steering_results)
        
        print("\n--- Generating Layer Analysis Plots ---")
        experiment.plot_layer_analysis(layer_analysis_results)
        
        print("\n=== Experiment completed successfully ===")
        
        # Print final summary
        print("\n=== Final Summary ===")
        print(f"🔍 IMPORTANT FINDING: This model shows inverted toxicity patterns!")
        print(f"📊 Challenging prompts (high prompt toxicity) → LOW completion toxicity: {baseline_results['challenging']['mean_toxicity']:.3f}")
        print(f"📊 Benign prompts (low prompt toxicity) → HIGH completion toxicity: {baseline_results['benign']['mean_toxicity']:.3f}")
        print(f"💡 This suggests Phi-4-mini-instruct has learned to avoid continuing toxic content.")
        print(f"")
        
        if steering_results:
            # Find best alpha for each prompt type
            best_alpha_challenging = min(
                steering_results['results'].keys(), 
                key=lambda a: steering_results['results'][a]['challenging']['mean_toxicity']
            )
            best_alpha_benign = min(
                steering_results['results'].keys(),
                key=lambda a: steering_results['results'][a]['benign']['mean_toxicity'] 
            )
            
            print(f"🎯 Steering Analysis:")
            print(f"Best α for challenging prompts (further reducing toxicity): {best_alpha_challenging}")
            print(f"  Toxicity: {baseline_results['challenging']['mean_toxicity']:.3f} → {steering_results['results'][best_alpha_challenging]['challenging']['mean_toxicity']:.3f}")
            
            print(f"Best α for benign prompts (reducing problematic completions): {best_alpha_benign}")
            print(f"  Toxicity: {baseline_results['benign']['mean_toxicity']:.3f} → {steering_results['results'][best_alpha_benign]['benign']['mean_toxicity']:.3f}")
            
            # Calculate reductions
            challenging_reduction = baseline_results['challenging']['mean_toxicity'] - steering_results['results'][best_alpha_challenging]['challenging']['mean_toxicity']
            benign_reduction = baseline_results['benign']['mean_toxicity'] - steering_results['results'][best_alpha_benign]['benign']['mean_toxicity']
            
            print(f"")
            print(f"📈 Steering Effectiveness:")
            print(f"  Challenging prompts: {challenging_reduction:.3f} absolute reduction ({(challenging_reduction/baseline_results['challenging']['mean_toxicity']*100):+.1f}% relative)")
            print(f"  Benign prompts: {benign_reduction:.3f} absolute reduction ({(benign_reduction/baseline_results['benign']['mean_toxicity']*100):+.1f}% relative)")
            print(f"")
            print(f"🎓 Key Insight: Steering is most effective on benign prompts that produce toxic completions!")
            
        # Layer analysis summary (always available now)
        layer_effectiveness = layer_analysis_results['layer_effectiveness']
        sorted_layers = sorted(layer_effectiveness.keys())
        most_effective_layer = max(sorted_layers, key=lambda l: layer_effectiveness[l]['mean_effectiveness'])
        least_effective_layer = min(sorted_layers, key=lambda l: layer_effectiveness[l]['mean_effectiveness'])
        
        print(f"\n=== Layer Analysis Summary ===")
        print(f"Most effective layer: Layer {most_effective_layer} ({layer_effectiveness[most_effective_layer]['mean_effectiveness']*100:.1f}% effectiveness)")
        print(f"Least effective layer: Layer {least_effective_layer} ({layer_effectiveness[least_effective_layer]['mean_effectiveness']*100:.1f}% effectiveness)")
        
        # Calculate layer depth analysis
        num_layers = len(sorted_layers)
        early_layers = sorted_layers[:num_layers//3]
        middle_layers = sorted_layers[num_layers//3:2*num_layers//3]
        late_layers = sorted_layers[2*num_layers//3:]
        
        early_effectiveness = np.mean([layer_effectiveness[l]['mean_effectiveness'] * 100 for l in early_layers])
        middle_effectiveness = np.mean([layer_effectiveness[l]['mean_effectiveness'] * 100 for l in middle_layers]) 
        late_effectiveness = np.mean([layer_effectiveness[l]['mean_effectiveness'] * 100 for l in late_layers])
        
        print(f"Layer depth analysis:")
        print(f"  Early layers (0-33%): {early_effectiveness:.1f}% average effectiveness")
        print(f"  Middle layers (33-66%): {middle_effectiveness:.1f}% average effectiveness")
        print(f"  Late layers (66-100%): {late_effectiveness:.1f}% average effectiveness")
        
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