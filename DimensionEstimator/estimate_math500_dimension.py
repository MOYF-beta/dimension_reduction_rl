#!/usr/bin/env python3
"""
Script to estimate the correlation dimension of the MATH-500 dataset using DimensionEstimator.

The MATH-500 dataset contains 500 mathematical problems and solutions from various topics
including Algebra, Geometry, Number Theory, Precalculus, and more.
"""

import time
import json
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from datasets import load_dataset
from dimension_estimator import DimensionEstimator


def load_math500_dataset() -> List[str]:
    """
    Load the MATH-500 dataset from Hugging Face and extract text content.
    
    Returns:
        List of text strings combining problems and solutions
    """
    print("Loading MATH-500 dataset from Hugging Face...")
    
    # Load the dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    texts = []
    categories = set()
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    for item in dataset:
        # Extract fields - the dataset has 'problem', 'solution', 'answer', 'subject', 'level'
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        subject = item.get('subject', 'Unknown')
        level = item.get('level', 0)
        
        categories.add(subject)
        
        # Combine problem and solution for dimension estimation
        # We'll analyze the mathematical language structure
        combined_text = f"Problem: {problem}\n\nSolution: {solution}"
        texts.append(combined_text)
    
    print(f"Extracted {len(texts)} problem-solution pairs")
    print(f"Subject areas: {sorted(categories)}")
    
    return texts


def analyze_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze basic statistics of the text corpus.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text statistics
    """
    total_chars = sum(len(text) for text in texts)
    total_words = sum(len(text.split()) for text in texts)
    
    text_lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    stats = {
        'num_texts': len(texts),
        'total_characters': total_chars,
        'total_words': total_words,
        'avg_chars_per_text': total_chars / len(texts),
        'avg_words_per_text': total_words / len(texts),
        'min_chars': min(text_lengths),
        'max_chars': max(text_lengths),
        'min_words': min(word_counts),
        'max_words': max(word_counts)
    }
    
    return stats


def estimate_math_dimension(
    model_name: str = "gpt2",
    max_context_length: int = 512,
    eta_threshold: float = 0.5,
    batch_size: int = 8,
    dimension_reduction: int = None,
    sample_size: int = None
) -> Dict[str, Any]:
    """
    Estimate the correlation dimension of the MATH-500 dataset.
    
    Args:
        model_name: Hugging Face model name or path
        max_context_length: Maximum context length for the model
        eta_threshold: Entropy threshold for filtering
        batch_size: Batch size for processing
        dimension_reduction: If provided, reduce vocabulary to this size
        sample_size: If provided, use only first N texts (for testing)
        
    Returns:
        Dictionary with dimension estimates and analysis results
    """
    # Load dataset
    texts = load_math500_dataset()
    
    if sample_size and sample_size < len(texts):
        print(f"Using sample of {sample_size} texts for faster processing")
        texts = texts[:sample_size]
    
    # Analyze text statistics
    print("\n" + "="*50)
    print("TEXT STATISTICS")
    print("="*50)
    
    stats = analyze_text_statistics(texts)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Initialize dimension estimator
    print("\n" + "="*50)
    print("DIMENSION ESTIMATION")
    print("="*50)
    
    estimator = DimensionEstimator(
        model_name_or_path=model_name,
        device="auto",
        max_context_length=max_context_length,
        eta_threshold=eta_threshold,
        dimension_reduction=dimension_reduction,
        batch_size=batch_size
    )
    
    print(f"Using model: {model_name}")
    print(f"Device: {estimator.device}")
    print(f"Max context length: {max_context_length}")
    print(f"Eta threshold: {eta_threshold}")
    print(f"Batch size: {batch_size}")
    if dimension_reduction:
        print(f"Dimension reduction: {dimension_reduction}")
    
    # Estimate dimension
    start_time = time.time()
    
    try:
        results = estimator.estimate_dimension(texts)
        processing_time = time.time() - start_time
        
        # Add additional information
        results.update({
            'model_name': model_name,
            'processing_time': processing_time,
            'text_statistics': stats,
            'parameters': {
                'max_context_length': max_context_length,
                'eta_threshold': eta_threshold,
                'dimension_reduction': dimension_reduction,
                'batch_size': batch_size
            }
        })
        
        return results
        
    except Exception as e:
        print(f"Error during dimension estimation: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_results(results: Dict[str, Any]):
    """
    Print formatted results.
    
    Args:
        results: Results dictionary from dimension estimation
    """
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "="*50)
    print("DIMENSION ESTIMATION RESULTS")
    print("="*50)
    
    print(f"Correlation Dimension: {results['correlation_dimension']:.4f}")
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    
    print(f"\nSequence Statistics:")
    print(f"  Total sequences: {results['total_sequences']:,}")
    print(f"  High-entropy sequences: {results['high_entropy_sequences']:,}")
    print(f"  High-entropy fraction: {results['high_entropy_fraction']:.4f}")
    
    # Performance metrics
    sequences_per_second = results['total_sequences'] / results['processing_time']
    print(f"\nPerformance:")
    print(f"  Sequences per second: {sequences_per_second:.1f}")
    
    print(f"\nParameters used:")
    for key, value in results['parameters'].items():
        print(f"  {key}: {value}")


def save_results(results: Dict[str, Any], output_file: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    if not results:
        print("No results to save.")
        return
    
    # Convert numpy arrays to lists for JSON serialization
    results_copy = results.copy()
    if 'epsilons' in results_copy:
        results_copy['epsilons'] = results_copy['epsilons'].tolist()
    if 'correlation_integrals' in results_copy:
        results_copy['correlation_integrals'] = results_copy['correlation_integrals'].tolist()
    
    with open(output_file, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def plot_results(results: Dict[str, Any], save_plot: str = None):
    """
    Create visualization plots.
    
    Args:
        results: Results dictionary
        save_plot: If provided, save plot to this file
    """
    if not results:
        print("No results to plot.")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation integral curve
    epsilons = results['epsilons']
    correlations = results['correlation_integrals']
    nonzero_mask = correlations > 0
    
    ax1.loglog(epsilons[nonzero_mask], correlations[nonzero_mask], 'bo-', alpha=0.7)
    ax1.set_xlabel('ε (epsilon)')
    ax1.set_ylabel('C(ε) (correlation integral)')
    ax1.set_title(f'Correlation Integral Curve\nDimension: {results["correlation_dimension"]:.3f}, R²: {results["r_squared"]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Text length distribution
    text_lengths = results['text_statistics']['word_counts'] if 'word_counts' in results['text_statistics'] else []
    if text_lengths:
        ax2.hist(text_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Words per Text')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Text Lengths')
        ax2.grid(True, alpha=0.3)
    
    # 3. Entropy filtering visualization
    total_seq = results['total_sequences']
    high_entropy_seq = results['high_entropy_sequences']
    filtered_seq = total_seq - high_entropy_seq
    
    ax3.pie([high_entropy_seq, filtered_seq], 
            labels=['High Entropy\n(Used)', 'Low Entropy\n(Filtered)'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightgreen', 'lightcoral'])
    ax3.set_title(f'Sequence Filtering\n(η threshold: {results["eta_threshold"]})')
    
    # 4. Processing statistics
    stats = results['text_statistics']
    categories = ['Texts', 'Total Chars', 'Total Words', 'Avg Words/Text']
    values = [stats['num_texts'], stats['total_characters']/1000, 
              stats['total_words']/1000, stats['avg_words_per_text']]
    
    ax4.bar(categories, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    ax4.set_ylabel('Count (thousands for chars/words)')
    ax4.set_title('Dataset Statistics')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_plot}")
    
    plt.show()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Estimate correlation dimension of MATH-500 dataset")
    
    parser.add_argument("--model", default="gpt2", 
                       help="Hugging Face model name (default: gpt2)")
    parser.add_argument("--max-context", type=int, default=512,
                       help="Maximum context length (default: 512)")
    parser.add_argument("--eta-threshold", type=float, default=0.5,
                       help="Entropy threshold for filtering (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing (default: 8)")
    parser.add_argument("--dimension-reduction", type=int, default=None,
                       help="Reduce vocabulary to this size (optional)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Use only first N texts for testing (optional)")
    parser.add_argument("--output", default="math500_dimension_results.json",
                       help="Output file for results (default: math500_dimension_results.json)")
    parser.add_argument("--plot", default="math500_dimension_analysis.png",
                       help="Output file for plots (default: math500_dimension_analysis.png)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("MATH-500 Dataset Correlation Dimension Estimation")
    print("="*60)
    
    # Run dimension estimation
    results = estimate_math_dimension(
        model_name=args.model,
        max_context_length=args.max_context,
        eta_threshold=args.eta_threshold,
        batch_size=args.batch_size,
        dimension_reduction=args.dimension_reduction,
        sample_size=args.sample_size
    )
    
    if results:
        # Print results
        print_results(results)
        
        # Save results
        save_results(results, args.output)
        
        # Create plots
        if not args.no_plot:
            plot_results(results, args.plot)
        
        # Summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"The MATH-500 dataset has an estimated correlation dimension of {results['correlation_dimension']:.4f}")
        print(f"This suggests the mathematical language has a {results['correlation_dimension']:.1f}-dimensional structure")
        print(f"in the probability manifold when using the {args.model} language model.")
        
        if results['r_squared'] > 0.9:
            print("High R² value indicates a reliable linear relationship in the log-log plot.")
        elif results['r_squared'] > 0.7:
            print("Moderate R² value suggests reasonable but not perfect linear scaling.")
        else:
            print("Low R² value indicates potential issues with the power-law assumption.")
            
    else:
        print("Dimension estimation failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
