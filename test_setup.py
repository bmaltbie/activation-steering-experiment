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

def main():
    """Run all tests"""
    print("=== Setup Verification Tests ===\n")
    
    tests = [
        test_basic_setup,
        test_experiment_init,
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