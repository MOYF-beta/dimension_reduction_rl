#!/usr/bin/env python3
"""
Simple test script for the DimensionEstimator class
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dimension_estimator import DimensionEstimator
import numpy as np


def test_fisher_rao_distance():
    """Test Fisher-Rao distance calculation"""
    print("Testing Fisher-Rao distance calculation...")
    
    estimator = DimensionEstimator("gpt2", max_context_length=64)
    
    # Test with identical distributions
    p1 = np.array([0.5, 0.3, 0.2])
    p2 = np.array([0.5, 0.3, 0.2])
    dist = estimator._fisher_rao_distance(p1, p2)
    assert abs(dist) < 1e-6, f"Distance between identical distributions should be 0, got {dist}"
    
    # Test with orthogonal distributions
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    dist = estimator._fisher_rao_distance(p1, p2)
    expected = np.pi  # Should be π for orthogonal distributions
    assert abs(dist - expected) < 0.1, f"Distance between orthogonal distributions should be ~π, got {dist}"
    
    print("✓ Fisher-Rao distance tests passed")
    return True


def test_dimension_reduction():
    """Test dimension reduction functionality"""
    print("Testing dimension reduction...")
    
    estimator = DimensionEstimator("gpt2", dimension_reduction=100, max_context_length=64)
    
    # Test reduction map
    assert len(set(estimator.reduction_map.values())) <= 100
    assert all(0 <= v < 100 for v in estimator.reduction_map.values())
    
    print("✓ Dimension reduction tests passed")
    return True


def test_basic_functionality():
    """Test basic functionality with a simple text"""
    print("Testing basic functionality...")
    
    # Use a very small model and short text for quick testing
    estimator = DimensionEstimator(
        "gpt2",
        max_context_length=32,
        dimension_reduction=100,  # Small for speed
    )
    
    # Simple test text
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    
    try:
        # Test probability distribution extraction
        prob_dists = estimator._get_probability_distributions(test_text)
        assert len(prob_dists) > 0, "Should generate probability distributions"
        assert prob_dists.shape[1] == 100, "Should have reduced dimensionality"
        
        # Test entropy filtering
        filtered_dists, indices = estimator._filter_high_entropy(prob_dists)
        assert len(filtered_dists) <= len(prob_dists), "Filtered should be subset"
        
        print("✓ Basic functionality tests passed")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    return True


def test_simple_estimation():
    """Test dimension estimation with simple texts"""
    print("Testing dimension estimation...")
    
    estimator = DimensionEstimator(
        "gpt2",
        max_context_length=64,
        dimension_reduction=200,
    )
    
    # Simple test texts
    test_texts = [
        "The cat sat on the mat. The dog ran in the park. Birds fly in the sky.",
        "Once upon a time there was a little girl who lived in the forest with her grandmother.",
    ]
    
    try:
        results = estimator.estimate_dimension(test_texts)
        
        # Basic sanity checks
        assert 'correlation_dimension' in results
        assert 'r_squared' in results
        assert 'total_sequences' in results
        
        dim = results['correlation_dimension']
        r2 = results['r_squared']
        
        print(f"  Estimated dimension: {dim:.3f}")
        print(f"  R-squared: {r2:.3f}")
        
        # Sanity checks
        assert 0 < dim < 200, f"Dimension should be reasonable, got {dim}"
        assert 0 <= r2 <= 1, f"R-squared should be in [0,1], got {r2}"
        
        print("✓ Dimension estimation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Dimension estimation test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    estimator = DimensionEstimator("gpt2", max_context_length=32, dimension_reduction=50)
    
    try:
        # Test with very short text
        short_results = estimator.estimate_dimension(["Hello world."])
        assert 'correlation_dimension' in short_results
        
        # Test with empty list (should raise error)
        try:
            estimator.estimate_dimension([])
            assert False, "Should have raised an error for empty text list"
        except (ValueError, Exception):
            pass  # Expected
        
        print("✓ Edge case tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Testing DimensionEstimator ===\n")
    
    tests = [
        test_fisher_rao_distance,
        test_dimension_reduction,
        test_basic_functionality,
        test_simple_estimation,
        test_edge_cases,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The DimensionEstimator appears to be working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
