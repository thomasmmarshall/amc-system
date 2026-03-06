#!/usr/bin/env python3
"""
Test script for AMC system to verify functionality
"""
import sys
import numpy as np
from amc import amcnet

def test_amc_initialization():
    """Test AMC system initialization"""
    print("="*60)
    print("Testing AMC System Initialization")
    print("="*60)
    
    try:
        amc = amcnet('amcnet', 2.4e6, 90e6, 0.0)
        print("✓ AMC initialized successfully")
        print(f"  - Model: {amc.modelname}")
        print(f"  - Classes: {amc.classlist}")
        print(f"  - Weights loaded: {len(amc.weights)} layers")
        print(f"  - Filters loaded: {len(amc.filters)} layers")
        return amc, True
    except Exception as e:
        print(f"✗ Initialization failed: {str(e)}")
        return None, False

def test_forward_pass(amc):
    """Test forward pass with random data"""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    try:
        # Generate random complex signal
        dummy_input = np.random.randn(256).astype(np.float32) * 0.1
        
        # Run forward pass
        output = amc.forwardpass(dummy_input)
        
        # Validate output
        assert output.shape == (13,), f"Expected shape (13,), got {output.shape}"
        assert np.abs(np.sum(output) - 1.0) < 0.001, f"Output probabilities should sum to 1.0, got {np.sum(output)}"
        
        predicted_class = amc.classlist[np.argmax(output)]
        confidence = np.max(output)
        
        print("✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Probability sum: {np.sum(output):.6f}")
        print(f"  - Predicted class: {predicted_class}")
        print(f"  - Confidence: {confidence:.4f}")
        print(f"  - Full distribution:")
        for i, cls in enumerate(amc.classlist):
            print(f"    {cls:6s}: {output[i]:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_snr_estimation(amc):
    """Test SNR estimation"""
    print("\n" + "="*60)
    print("Testing SNR Estimation")
    print("="*60)
    
    try:
        # Generate signal with known characteristics
        t = np.linspace(0, 1, 1024)
        signal = np.exp(1j * 2 * np.pi * 10 * t)
        noise = 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        noisy_signal = signal + noise
        
        snr = amc.SNR(noisy_signal)
        
        print("✓ SNR estimation successful")
        print(f"  - Estimated SNR: {snr:.2f} dB")
        
        return True
    except Exception as e:
        print(f"✗ SNR estimation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_classifications(amc):
    """Test multiple classifications to ensure consistency"""
    print("\n" + "="*60)
    print("Testing Multiple Classifications")
    print("="*60)
    
    try:
        results = []
        for i in range(5):
            dummy_input = np.random.randn(256).astype(np.float32) * 0.1
            output = amc.forwardpass(dummy_input)
            predicted = amc.classlist[np.argmax(output)]
            confidence = np.max(output)
            results.append((predicted, confidence))
            print(f"  Run {i+1}: {predicted} (confidence: {confidence:.4f})")
        
        print("✓ Multiple classifications completed successfully")
        return True
    except Exception as e:
        print(f"✗ Multiple classifications failed: {str(e)}")
        return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" AMC System Test Suite")
    print("="*60 + "\n")
    
    results = {}
    
    # Test 1: Initialization
    amc, success = test_amc_initialization()
    results['initialization'] = success
    
    if not success:
        print("\n✗ Cannot proceed with tests - initialization failed")
        sys.exit(1)
    
    # Test 2: Forward pass
    results['forward_pass'] = test_forward_pass(amc)
    
    # Test 3: SNR estimation
    results['snr_estimation'] = test_snr_estimation(amc)
    
    # Test 4: Multiple classifications
    results['multiple_classifications'] = test_multiple_classifications(amc)
    
    # Summary
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! AMC system is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        sys.exit(1)
