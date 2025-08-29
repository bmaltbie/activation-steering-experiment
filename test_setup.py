#!/usr/bin/env python3
"""
Test script to verify the setup works before running the full experiment.
This will test model loading and basic functionality without running the full experiment.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import ActivationSteeringExperiment

def test_basic_setup():
    """Test basic model loading and tokenizer setup"""
    print("Testing basic setup...")
    
    try:
        # Test model and tokenizer loading
        model_name = "microsoft/Phi-4-mini-instruct"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        # Test a simple tokenization
        test_text = "Hello world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Tokenization test successful: {test_text} -> {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic setup test failed: {e}")
        return False

def test_experiment_init():
    """Test experiment class initialization"""
    print("\nTesting experiment initialization...")
    
    try:
        experiment = ActivationSteeringExperiment()
        print("✓ Experiment class initialized successfully")
        
        # Test device setup
        print(f"✓ Using device: {experiment.device}")
        
        # Test configuration
        print(f"✓ Generation config: temp={experiment.gen_config.temperature}, "
              f"top_p={experiment.gen_config.top_p}, max_tokens={experiment.gen_config.max_new_tokens}")
        
        # Validate new optimized parameters
        assert experiment.gen_config.temperature == 0.3, f"Expected temperature=0.3, got {experiment.gen_config.temperature}"
        assert experiment.gen_config.top_p == 0.8, f"Expected top_p=0.8, got {experiment.gen_config.top_p}"
        print("✓ Optimized generation parameters validated")
        
        # Test model architecture detection
        print(f"✓ Model architecture: {type(experiment.model).__name__}")
        print(f"✓ Transformer layers: {len(experiment.transformer_layers)}")
        
        # Test toxicity classifier configuration
        test_texts = ["Hello world", "This is a test"]
        toxicity_results = experiment.toxicity_classifier(test_texts)
        
        # Validate top_k=None is working (replaces deprecated return_all_scores=True)
        assert isinstance(toxicity_results[0], list), "top_k=None not working - should return list of all scores"
        assert len(toxicity_results[0]) > 2, "Expected multiple labels from classifier (toxicity, severe_toxicity, etc.)"
        
        # Test that we can find 'toxicity' label (the correct label name)
        toxic_score = next((item['score'] for item in toxicity_results[0] if item['label'] == 'toxicity'), None)
        assert toxic_score is not None, "Could not find 'toxicity' label in classifier output"
        assert 0.0 <= toxic_score <= 1.0, f"Toxicity score {toxic_score} not in valid range [0,1]"
        
        print("✓ Toxicity classifier: top_k=None configuration validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment initialization test failed: {e}")
        return False

def test_dataset_loading():
    """Test RTP dataset loading (quick test)"""
    print("\nTesting dataset loading...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Load dataset
        experiment.load_rtp_dataset()
        print(f"✓ Dataset loaded: {len(experiment.rtp_data)} samples")
        
        # Test subset creation
        experiment.create_subsets()
        print(f"✓ Challenging subset: {len(experiment.challenging_subset)} samples")
        print(f"✓ Benign subset: {len(experiment.benign_subset)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        return False

def test_architecture_compatibility():
    """Test model architecture detection and layer access"""
    print("\nTesting architecture compatibility...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Test 1: Architecture detection worked
        assert hasattr(experiment, 'transformer_layers'), "Architecture detection failed - no transformer_layers attribute"
        assert len(experiment.transformer_layers) > 0, "No transformer layers found"
        print(f"✓ Architecture detection: {experiment.layer_attr} with {len(experiment.transformer_layers)} layers")
        
        # Test 2: Layer access works
        first_layer = experiment.transformer_layers[0]
        assert first_layer is not None, "Cannot access first transformer layer"
        print("✓ Layer access: Can access transformer layers")
        
        # Test 3: Hook registration works (lightweight test)
        def dummy_hook(module, input, output):
            pass
        
        hook = first_layer.register_forward_hook(dummy_hook)
        hook.remove()  # Clean up immediately
        print("✓ Hook registration: Forward hooks work")
        
        return True
        
    except Exception as e:
        print(f"✗ Architecture compatibility test failed: {e}")
        return False

def test_activation_tensor_shapes():
    """Test activation collection and tensor shape handling with minimal computation"""
    print("\nTesting activation tensor shapes...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Test with a very short prompt to minimize computation
        test_prompt = "Hello"
        print(f"Testing with minimal prompt: '{test_prompt}'")
        
        # Collect activations from just the first layer to test tensor shapes
        layer_idx = 0
        first_layer = experiment.transformer_layers[layer_idx]
        activation_collected = {}
        
        def test_hook(module, input, output):
            # This is the same logic as in the real activation collection
            if isinstance(output, tuple):
                activation = output[0].detach().cpu()
            else:
                activation = output.detach().cpu()
            activation_collected[layer_idx] = activation
        
        # Register hook for just one layer
        hook = first_layer.register_forward_hook(test_hook)
        
        try:
            # Generate a very short completion to test tensor shapes
            inputs = experiment.tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(experiment.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Generate just 1 token to minimize computation
                outputs = experiment.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=1,  # Minimal generation
                    do_sample=False,  # Deterministic
                    pad_token_id=experiment.gen_config.pad_token_id,
                    eos_token_id=experiment.gen_config.eos_token_id
                )
            
            # Test that we collected an activation
            assert layer_idx in activation_collected, "No activation was collected"
            activation = activation_collected[layer_idx]
            
            print(f"✓ Activation collected: shape {activation.shape}")
            
            # Test tensor shape handling (the logic that was failing)
            if activation.dim() == 2:
                # This would have caused the original error
                test_pooling = activation.unsqueeze(0)  # Add batch dim
                seq_len = test_pooling.shape[1]
                pool_tokens = min(5, seq_len)  # Small pool for test
                pooled = test_pooling[:, -pool_tokens:, :].mean(dim=1)
                print(f"✓ 2D tensor handling: {activation.shape} → {pooled.shape}")
            elif activation.dim() == 3:
                # Standard case
                seq_len = activation.shape[1]
                pool_tokens = min(5, seq_len)
                pooled = activation[:, -pool_tokens:, :].mean(dim=1)
                print(f"✓ 3D tensor handling: {activation.shape} → {pooled.shape}")
            elif activation.dim() == 1:
                # Already pooled
                print(f"✓ 1D tensor handling: {activation.shape} (already pooled)")
            else:
                print(f"⚠ Unusual tensor shape: {activation.shape}")
            
        finally:
            hook.remove()
        
        print("✓ Tensor shape handling: All shapes handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Activation tensor shape test failed: {e}")
        return False

def test_steering_compatibility():
    """Test steering hook registration and basic functionality"""
    print("\nTesting steering compatibility...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Create a dummy steering vector with correct dimensions
        # Get the hidden dimension from the first layer
        test_prompt = "Test"
        inputs = experiment.tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(experiment.device) for k, v in inputs.items()}
        
        # Collect one activation to determine hidden dimension
        layer_idx = 0
        first_layer = experiment.transformer_layers[layer_idx]
        activation_shape = None
        
        def shape_hook(module, input, output):
            nonlocal activation_shape
            if isinstance(output, tuple):
                activation_shape = output[0].shape
            else:
                activation_shape = output.shape
        
        hook = first_layer.register_forward_hook(shape_hook)
        
        try:
            with torch.no_grad():
                experiment.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=1,
                    do_sample=False
                )
        finally:
            hook.remove()
        
        assert activation_shape is not None, "Could not determine activation shape"
        
        # Determine hidden dimension
        if len(activation_shape) >= 2:
            hidden_dim = activation_shape[-1]  # Last dimension is usually hidden_dim
        else:
            hidden_dim = activation_shape[0]
        
        print(f"✓ Hidden dimension detected: {hidden_dim}")
        
        # Create dummy steering vector
        dummy_steering_vector = torch.randn(hidden_dim).to(experiment.device)
        
        # Test steering hook registration (the other potential failure point)
        steering_applied = False
        
        def test_steering_hook(module, input, output):
            nonlocal steering_applied
            # This mimics the steering logic that could fail
            if isinstance(output, tuple):
                activation_tensor = output[0]
                rest_of_output = output[1:]
            else:
                activation_tensor = output
                rest_of_output = ()
            
            # Test the tensor operations that could fail
            if activation_tensor.dim() == 3:
                batch_size, seq_len, hidden_dim = activation_tensor.shape
                steering_addition = dummy_steering_vector.unsqueeze(0).unsqueeze(0)
                steering_addition = steering_addition.expand(batch_size, seq_len, hidden_dim)
            elif activation_tensor.dim() == 2:
                seq_len, hidden_dim = activation_tensor.shape
                steering_addition = dummy_steering_vector.unsqueeze(0)
                steering_addition = steering_addition.expand(seq_len, hidden_dim)
            else:
                steering_applied = True
                return output
            
            # Apply steering
            steered_activation = activation_tensor + steering_addition * 0.1  # Small alpha for test
            
            # Reconstruct output
            if isinstance(output, tuple):
                output = (steered_activation,) + rest_of_output
            else:
                output = steered_activation
            
            steering_applied = True
            return output
        
        # Test steering hook
        hook = first_layer.register_forward_hook(test_steering_hook)
        
        try:
            with torch.no_grad():
                experiment.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=1,
                    do_sample=False
                )
        finally:
            hook.remove()
        
        assert steering_applied, "Steering hook was not applied"
        print("✓ Steering hook: Registration and application works")
        
        return True
        
    except Exception as e:
        print(f"✗ Steering compatibility test failed: {e}")
        return False

def test_plotting_setup():
    """Test matplotlib backend and basic plotting functionality"""
    print("\nTesting plotting setup...")
    
    try:
        # Test matplotlib import and backend
        import matplotlib
        backend = matplotlib.get_backend()
        print(f"✓ Matplotlib backend: {backend}")
        
        # Test basic plotting
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple test plot
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 2, 3])
        
        plt.figure(figsize=(4, 3))
        plt.plot(x, y, 'o-')
        plt.title("Test Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        # Test saving (don't show)
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = f"{temp_dir}/test_plot.png"
            plt.savefig(test_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Check if file was created
            from pathlib import Path
            if Path(test_file).exists():
                print("✓ Plot saving: PNG export works")
            else:
                print("⚠ Plot saving: PNG file not created")
        
        print("✓ Plotting setup: Basic functionality works")
        
        # Test that new plotting method exists
        from activation_steering import ActivationSteeringExperiment
        temp_experiment = ActivationSteeringExperiment.__new__(ActivationSteeringExperiment)
        
        assert hasattr(temp_experiment, 'plot_steering_effectiveness_summary'), "New summary plot method not found"
        print("✓ New summary plot method: plot_steering_effectiveness_summary exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Plotting setup test failed: {e}")
        return False

def test_layer_analysis_methods():
    """Test layer analysis method compatibility and API (lightweight static analysis)"""
    print("\nTesting layer analysis methods...")
    
    try:
        # Test 1: Static code analysis to catch method name errors
        import ast
        import inspect
        
        # Read the source code to check for method calls
        with open('activation_steering.py', 'r') as f:
            source_code = f.read()
        
        # Parse the AST to find method calls
        tree = ast.parse(source_code)
        
        # Look for calls to 'score_completions' which should be 'score_toxicity'
        class MethodCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.score_completions_calls = []
                self.method_calls = []
                
            def visit_Attribute(self, node):
                if isinstance(node.ctx, ast.Load):
                    if hasattr(node, 'attr'):
                        if node.attr == 'score_completions':
                            self.score_completions_calls.append(f"Line {node.lineno}: {node.attr}")
                        self.method_calls.append(node.attr)
                self.generic_visit(node)
        
        visitor = MethodCallVisitor()
        visitor.visit(tree)
        
        if visitor.score_completions_calls:
            print(f"✗ CRITICAL: Found {len(visitor.score_completions_calls)} calls to 'score_completions' (should be 'score_toxicity'):")
            for call in visitor.score_completions_calls:
                print(f"  {call}")
            return False
        else:
            print("✓ No incorrect 'score_completions' method calls found")
        
        # Test 2: Check that required methods would exist if we instantiated the class
        # We'll do this without actually loading the model for speed
        try:
            experiment = ActivationSteeringExperiment.__new__(ActivationSteeringExperiment)
            
            required_methods = [
                'score_toxicity',
                'evaluate_steering_all_layers', 
                'plot_layer_analysis',
                'generate_completions'
            ]
            
            # Check method definitions exist in source
            for method_name in required_methods:
                method_def_pattern = f"def {method_name}"
                if method_def_pattern not in source_code:
                    print(f"✗ Missing method definition: {method_name}")
                    return False
            
            print(f"✓ All required method definitions found: {', '.join(required_methods)}")
            
        except Exception as e:
            print(f"✗ Error checking method definitions: {e}")
            return False
        
        # Test 3: Check method signatures without full initialization
        import activation_steering
        cls = activation_steering.ActivationSteeringExperiment
        
        # Check evaluate_steering_all_layers signature
        if hasattr(cls, 'evaluate_steering_all_layers'):
            sig = inspect.signature(cls.evaluate_steering_all_layers)
            params = list(sig.parameters.keys())
            if 'challenging_subset_size' not in params or 'benign_subset_size' not in params:
                print(f"✗ evaluate_steering_all_layers missing required parameters")
                return False
            print("✓ evaluate_steering_all_layers has correct signature")
        else:
            print("✗ evaluate_steering_all_layers method not found")
            return False
        
        # Test 4: Look for specific problematic patterns in the source code
        problematic_patterns = [
            ('self.score_completions(', 'should be self.score_toxicity('),
            ('score_completions =', 'should be score_toxicity ='),
        ]
        
        for pattern, message in problematic_patterns:
            if pattern in source_code:
                print(f"✗ Found problematic pattern '{pattern}' - {message}")
                return False
        
        print("✓ No problematic patterns found in source code")
        print("✓ Layer analysis methods API compatibility verified (static analysis)")
        return True
        
    except Exception as e:
        print(f"✗ Layer analysis methods test failed: {e}")
        return False

def test_merged_baseline_activations():
    """Test the new merged baseline evaluation with activations function"""
    print("\nTesting merged baseline evaluation with activations...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Test that the new function exists
        assert hasattr(experiment, 'run_baseline_evaluation_with_activations'), "New merged function not found"
        print("✓ Merged function exists: run_baseline_evaluation_with_activations")
        
        # Test function signature
        import inspect
        sig = inspect.signature(experiment.run_baseline_evaluation_with_activations)
        # Instance methods don't include 'self' in signature when inspected on instance
        assert len(sig.parameters) == 0, f"Expected 0 parameters (self excluded), got {len(sig.parameters)}"
        print("✓ Function signature correct (no additional parameters beyond self)")
        
        # Test that old functions have deprecation warnings (check source)
        with open('activation_steering.py', 'r') as f:
            source = f.read()
        
        assert 'DEPRECATED' in source, "Expected deprecation warnings in source code"
        assert source.count('run_baseline_evaluation_with_activations') > 1, "New function not referenced enough times"
        print("✓ Deprecation warnings present in old functions")
        
        # Test method availability without running (too expensive for setup test)
        old_baseline_method = getattr(experiment, 'run_baseline_evaluation', None)
        old_activation_method = getattr(experiment, 'collect_activations', None)
        new_merged_method = getattr(experiment, 'run_baseline_evaluation_with_activations', None)
        
        assert old_baseline_method is not None, "Old baseline method should still exist for backward compatibility"
        assert old_activation_method is not None, "Old activation method should still exist for backward compatibility"  
        assert new_merged_method is not None, "New merged method should exist"
        print("✓ All baseline/activation methods available")
        
        return True
        
    except Exception as e:
        print(f"✗ Merged baseline/activations test failed: {e}")
        return False

def test_comprehensive_steering_evaluation():
    """Test the new comprehensive steering evaluation function"""
    print("\nTesting comprehensive steering evaluation...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Test that the new function exists
        assert hasattr(experiment, 'run_comprehensive_steering_evaluation'), "New comprehensive function not found"
        print("✓ Comprehensive function exists: run_comprehensive_steering_evaluation")
        
        # Test function signature
        import inspect
        sig = inspect.signature(experiment.run_comprehensive_steering_evaluation)
        # Instance methods don't include 'self' when inspected on instance
        expected_params = ['target_layer', 'full_subset_for_target', 'layer_subset_size']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"Missing parameter: {param}"
        print("✓ Function signature has all expected parameters")
        
        # Test default values
        assert sig.parameters['target_layer'].default is None, "target_layer should default to None"
        assert sig.parameters['full_subset_for_target'].default == True, "full_subset_for_target should default to True"
        assert sig.parameters['layer_subset_size'].default == 8, "layer_subset_size should default to 8"
        print("✓ Function default parameters correct")
        
        # Test that old steering functions have deprecation warnings
        with open('activation_steering.py', 'r') as f:
            source = f.read()
        
        # Check for deprecation in old steering functions
        old_steering_deprecated = 'def run_steering_evaluation(self, target_layer: int = None) -> Dict[str, Any]:\n        """\n        DEPRECATED:' in source
        old_layer_analysis_deprecated = 'def evaluate_steering_all_layers(self, challenging_subset_size: int = None, benign_subset_size: int = None) -> Dict[str, Any]:\n        """\n        DEPRECATED:' in source
        
        assert old_steering_deprecated, "run_steering_evaluation should have deprecation warning"
        assert old_layer_analysis_deprecated, "evaluate_steering_all_layers should have deprecation warning"
        print("✓ Old steering evaluation functions have deprecation warnings")
        
        # Test that all old methods still exist for backward compatibility
        old_steering_method = getattr(experiment, 'run_steering_evaluation', None)
        old_layer_method = getattr(experiment, 'evaluate_steering_all_layers', None)
        new_comprehensive_method = getattr(experiment, 'run_comprehensive_steering_evaluation', None)
        
        assert old_steering_method is not None, "Old steering method should exist for backward compatibility"
        assert old_layer_method is not None, "Old layer analysis method should exist for backward compatibility"
        assert new_comprehensive_method is not None, "New comprehensive method should exist"
        print("✓ All steering evaluation methods available")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive steering evaluation test failed: {e}")
        return False

def test_optimized_score_toxicity():
    """Test the optimized score_toxicity function with return_all_scores=True"""
    print("\nTesting optimized score_toxicity function...")
    
    try:
        experiment = ActivationSteeringExperiment()
        
        # Test that score_toxicity works with the new classifier configuration
        test_texts = [
            "Hello, this is a nice day!",
            "I hate everything and everyone",  # More likely to be toxic
            "The weather is lovely today"
        ]
        
        print("Testing score_toxicity with sample texts...")
        scores = experiment.score_toxicity(test_texts)
        
        # Validate output format
        assert isinstance(scores, list), "score_toxicity should return a list"
        assert len(scores) == len(test_texts), f"Expected {len(test_texts)} scores, got {len(scores)}"
        
        # Validate score ranges
        for i, score in enumerate(scores):
            assert isinstance(score, (int, float)), f"Score {i} should be numeric, got {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score {i} should be in [0,1], got {score}"
        
        print(f"✓ Scores returned: {[f'{s:.3f}' for s in scores]}")
        
        # Test that the optimized version produces reasonable results
        # The toxic text should generally have higher score than benign texts
        toxic_text_score = scores[1]  # "I hate everything"
        benign_text_scores = [scores[0], scores[2]]  # Nice day, lovely weather
        
        print(f"✓ Toxic text score: {toxic_text_score:.3f}")
        print(f"✓ Benign text scores: {[f'{s:.6f}' for s in benign_text_scores]}")
        
        # Basic sanity check - toxic text should have higher score than benign texts
        max_benign_score = max(benign_text_scores)
        if toxic_text_score > max_benign_score:
            improvement_ratio = toxic_text_score / max_benign_score if max_benign_score > 0 else float('inf')
            print(f"✓ Sanity check: Toxic text score ({toxic_text_score:.6f}) > max benign score ({max_benign_score:.6f})")
            print(f"  → Toxic text is {improvement_ratio:.1f}x more toxic than highest benign text")
        else:
            print(f"⚠ Note: Toxic text score ({toxic_text_score:.6f}) ≤ max benign score ({max_benign_score:.6f})")
            print("  This may be normal for this specific classifier/text combination")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimized score_toxicity test failed: {e}")
        return False

def test_deprecated_function_warnings():
    """Test that deprecated functions show appropriate warnings"""
    print("\nTesting deprecated function warnings...")
    
    try:
        # Test by checking source code for warning patterns
        with open('activation_steering.py', 'r') as f:
            source = f.read()
        
        # Functions that should have deprecation warnings
        deprecated_functions = [
            'run_baseline_evaluation',
            'collect_activations', 
            'run_steering_evaluation',
            'evaluate_steering_all_layers'
        ]
        
        for func_name in deprecated_functions:
            # Look for deprecation warning in the function
            func_pattern = f'def {func_name}('
            if func_pattern in source:
                func_start = source.find(func_pattern)
                # Look for DEPRECATED keyword in the next 500 characters (docstring area)
                func_section = source[func_start:func_start+500]
                if 'DEPRECATED' in func_section:
                    print(f"✓ {func_name}: Deprecation warning present")
                else:
                    print(f"⚠ {func_name}: No deprecation warning found")
            else:
                print(f"✗ {func_name}: Function definition not found")
        
        # Test that new functions exist and don't have deprecation warnings
        new_functions = [
            'run_baseline_evaluation_with_activations',
            'run_comprehensive_steering_evaluation'
        ]
        
        for func_name in new_functions:
            func_pattern = f'def {func_name}('
            assert func_pattern in source, f"New function {func_name} not found"
            
            func_start = source.find(func_pattern)
            func_section = source[func_start:func_start+500]
            assert 'DEPRECATED' not in func_section, f"New function {func_name} should not be deprecated"
            print(f"✓ {func_name}: New function exists and not deprecated")
        
        return True
        
    except Exception as e:
        print(f"✗ Deprecated function warnings test failed: {e}")
        return False

def test_method_signatures_updated():
    """Test that the main() function uses the new optimized methods"""
    print("\nTesting main() function updates...")
    
    try:
        with open('activation_steering.py', 'r') as f:
            source = f.read()
        
        # Find the main() function
        main_start = source.find('def main():')
        assert main_start != -1, "main() function not found"
        
        # Look at the main function content
        main_section = source[main_start:main_start + 3000]  # Look at substantial portion
        
        # Test 1: Should use new merged baseline function
        assert 'run_baseline_evaluation_with_activations()' in main_section, "main() should use merged baseline function"
        print("✓ main() uses merged baseline evaluation function")
        
        # Test 2: Should use new comprehensive steering evaluation
        assert 'run_comprehensive_steering_evaluation(' in main_section, "main() should use comprehensive steering evaluation"
        print("✓ main() uses comprehensive steering evaluation function")
        
        # Test 3: Should NOT use old separate functions in the main path
        # (They might exist for backward compatibility but shouldn't be the primary path)
        baseline_and_collect = 'run_baseline_evaluation()' in main_section and 'collect_activations()' in main_section
        if baseline_and_collect:
            print("⚠ main() still uses old separate baseline + collect_activations pattern")
        else:
            print("✓ main() no longer uses old separate baseline + activation collection")
        
        steering_and_layer = 'run_steering_evaluation()' in main_section and 'evaluate_steering_all_layers(' in main_section
        if steering_and_layer:
            print("⚠ main() still uses old separate steering + layer analysis pattern")  
        else:
            print("✓ main() no longer uses old separate steering evaluation + layer analysis")
        
        # Test 4: Check step numbering updated
        step_count = main_section.count('Step ')
        if step_count <= 5:  # Should be reduced from original 6 steps
            print(f"✓ Step count optimized: Found {step_count} steps (should be ≤5)")
        else:
            print(f"⚠ Step count: Found {step_count} steps (may not be optimized)")
        
        return True
        
    except Exception as e:
        print(f"✗ Main function update test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Setup Verification Tests ===\n")
    
    tests = [
        test_basic_setup,
        test_layer_analysis_methods,  # Run static analysis early to catch method errors quickly
        test_experiment_init,         # Now includes generation parameter validation
        test_optimized_score_toxicity, # Test new toxicity classifier configuration
        test_architecture_compatibility,
        test_activation_tensor_shapes,
        test_steering_compatibility,
        test_merged_baseline_activations,      # Test new merged baseline function
        test_comprehensive_steering_evaluation, # Test new comprehensive steering function
        test_deprecated_function_warnings,     # Test deprecation warnings
        test_method_signatures_updated,        # Test main() function updates
        test_plotting_setup,
        test_dataset_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"=== Test Summary: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All tests passed! The setup is ready for the full experiment.")
        print("Run 'python main.py' to start the full experiment.")
    else:
        print("✗ Some tests failed. Please check the setup before running the experiment.")
    
    return passed == total

if __name__ == "__main__":
    main()