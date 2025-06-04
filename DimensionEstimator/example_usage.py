#!/usr/bin/env python3
"""
Example usage of the DimensionEstimator class

This script demonstrates how to use the correlation dimension estimator
to analyze text sequences using different language models.
"""

from dimension_estimator import DimensionEstimator
import torch


def main():
    # Sample texts for testing (you can replace with your own texts)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing the dimension estimator.",
        "In the beginning was the Word, and the Word was with God, and the Word was God. All things were made through him.",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
        "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
        "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse.",
    ]
    
    # You can also load longer texts from files
    # with open('your_text_file.txt', 'r', encoding='utf-8') as f:
    #     long_text = f.read()
    #     sample_texts = [long_text]
    
    print("=== Correlation Dimension Estimation for Natural Language ===\n")
    
    # Initialize the dimension estimator 
    # You can use different models like "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    model_name = "gpt2"  # Start with smaller model for testing
    
    print(f"Initializing DimensionEstimator with model: {model_name}")
    
    # Create estimator with reasonable settings
    estimator = DimensionEstimator(
        model_name_or_path=model_name,
        device="auto",  # Will use CUDA if available, otherwise CPU
        max_context_length=256,  # Smaller context for faster processing
        eta_threshold=0.5,  # Filter threshold from paper
        dimension_reduction=1000,  # Reduce vocabulary size for efficiency
        batch_size=8
    )
    
    print("\nEstimating correlation dimension...")
    
    # Estimate dimension
    try:
        results = estimator.estimate_dimension(sample_texts)
        
        print("\n=== Results ===")
        print(f"Correlation Dimension: {results['correlation_dimension']:.4f}")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"Total sequences: {results['total_sequences']}")
        print(f"High-entropy sequences: {results['high_entropy_sequences']}")
        print(f"High-entropy fraction: {results['high_entropy_fraction']:.3f}")
        print(f"Eta threshold: {results['eta_threshold']}")
        
        # According to the paper, natural language should have dimension around 6.5
        expected_dim = 6.5
        print(f"\nExpected dimension (from paper): ~{expected_dim}")
        
        if results['r_squared'] > 0.8:
            print("✓ Good linear fit in correlation integral")
        else:
            print("⚠ Poor linear fit - results may be unreliable")
            
        if results['high_entropy_fraction'] > 0.1:
            print("✓ Sufficient high-entropy sequences")
        else:
            print("⚠ Few high-entropy sequences - consider longer texts")
        
        # Optionally plot the correlation integral curve
        try:
            print("\nPlotting correlation integral curve...")
            estimator.plot_correlation_integral(results)
        except:
            print("Could not plot (matplotlib may not be available)")
            
    except Exception as e:
        print(f"Error during estimation: {e}")
        return
    
    print("\n=== Interpretation ===")
    print("The correlation dimension quantifies the self-similarity in language.")
    print("Values around 6.5 are typical for natural language (from the paper).")
    print("Higher values indicate more randomness, lower values more structure.")
    print("\nComparison with other processes (from paper):")
    print("- Natural language: ~6.5")
    print("- Shuffled text: ~13.0") 
    print("- Random weights: ~80")
    print("- Barabási-Albert networks: ~2.0")
    print("- White noise: >100")


if __name__ == "__main__":
    main()
