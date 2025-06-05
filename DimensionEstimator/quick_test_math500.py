#!/usr/bin/env python3
"""
Quick test script for MATH-500 dimension estimation.
This script runs a fast test with a small sample to verify everything works.
"""

import time
from datasets import load_dataset
from dimension_estimator import DimensionEstimator


def quick_test():
    """Run a quick test with a small sample of MATH-500 data."""
    
    print("Quick MATH-500 Dimension Estimation Test")
    print("="*45)
    
    # Load a small sample of the dataset
    print("Loading MATH-500 dataset sample...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Take first 10 examples for quick test
    sample_size = 10
    texts = []
    
    for i in range(min(sample_size, len(dataset))):
        item = dataset[i]
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        
        # Combine problem and solution
        combined_text = f"Problem: {problem}\n\nSolution: {solution}"
        texts.append(combined_text)
    
    print(f"Loaded {len(texts)} problem-solution pairs")
    
    # Show sample content
    print(f"\nSample problem length: {len(texts[0])} characters")
    print(f"Sample problem preview: {texts[0][:200]}...")
    
    # Initialize estimator with small model for quick test
    print("\nInitializing DimensionEstimator...")
    estimator = DimensionEstimator(
        model_name_or_path="gpt2",  # Small model for speed
        device="auto",
        max_context_length=256,     # Smaller context for speed
        eta_threshold=0.5,
        batch_size=4               # Small batch for memory efficiency
    )
    
    print(f"Using device: {estimator.device}")
    
    # Run estimation
    print("\nEstimating correlation dimension...")
    start_time = time.time()
    
    try:
        results = estimator.estimate_dimension(texts)
        processing_time = time.time() - start_time
        
        print("\n" + "="*45)
        print("RESULTS")
        print("="*45)
        print(f"Correlation Dimension: {results['correlation_dimension']:.4f}")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Total sequences: {results['total_sequences']:,}")
        print(f"High-entropy sequences: {results['high_entropy_sequences']:,}")
        print(f"High-entropy fraction: {results['high_entropy_fraction']:.4f}")
        
        # Performance
        sequences_per_second = results['total_sequences'] / processing_time
        print(f"Sequences per second: {sequences_per_second:.1f}")
        
        print("\n✅ Quick test completed successfully!")
        print(f"The sample of MATH-500 data has dimension ≈ {results['correlation_dimension']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during estimation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n" + "="*45)
        print("Next steps:")
        print("1. Run full analysis: python estimate_math500_dimension.py")
        print("2. Try different models: python estimate_math500_dimension.py --model microsoft/DialoGPT-medium")
        print("3. Use larger sample: python estimate_math500_dimension.py --sample-size 100")
    else:
        print("\n" + "="*45)
        print("Please check the error messages and ensure all dependencies are installed.")
        print("Try: pip install -r requirements.txt")
