#!/usr/bin/env python3
"""
Comprehensive analysis script for MATH-500 dataset dimension estimation.
This script runs multiple experiments with different parameters to understand
the dimensional characteristics of mathematical language.
"""

import json
import time
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from dimension_estimator import DimensionEstimator


def run_experiment(
    sample_size: int,
    eta_threshold: float, 
    model_name: str = "gpt2",
    max_context: int = 512,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Run a single dimension estimation experiment.
    
    Args:
        sample_size: Number of texts to analyze
        eta_threshold: Entropy threshold for filtering
        model_name: Model to use
        max_context: Maximum context length
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {sample_size} texts, η={eta_threshold}, model={model_name}")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    texts = []
    
    for i in range(min(sample_size, len(dataset))):
        item = dataset[i]
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        combined_text = f"Problem: {problem}\n\nSolution: {solution}"
        texts.append(combined_text)
    
    # Initialize estimator
    estimator = DimensionEstimator(
        model_name_or_path=model_name,
        device="auto",
        max_context_length=max_context,
        eta_threshold=eta_threshold,
        batch_size=batch_size
    )
    
    # Run estimation
    start_time = time.time()
    try:
        results = estimator.estimate_dimension(texts)
        processing_time = time.time() - start_time
        
        # Add experiment parameters
        results.update({
            'experiment_params': {
                'sample_size': sample_size,
                'eta_threshold': eta_threshold,
                'model_name': model_name,
                'max_context': max_context,
                'batch_size': batch_size
            },
            'processing_time': processing_time,
            'success': True
        })
        
        print(f"✅ SUCCESS: Dimension = {results['correlation_dimension']:.4f}, R² = {results['r_squared']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {
            'experiment_params': {
                'sample_size': sample_size,
                'eta_threshold': eta_threshold,
                'model_name': model_name,
                'max_context': max_context,
                'batch_size': batch_size
            },
            'success': False,
            'error': str(e)
        }


def run_parameter_sweep():
    """Run experiments with different parameter combinations."""
    
    experiments = [
        # Vary sample size with fixed eta=0.5
        {'sample_size': 25, 'eta_threshold': 0.5, 'name': '25_samples_eta05'},
        {'sample_size': 50, 'eta_threshold': 0.5, 'name': '50_samples_eta05'},
        {'sample_size': 100, 'eta_threshold': 0.5, 'name': '100_samples_eta05'},
        
        # Vary eta threshold with fixed sample size=50
        {'sample_size': 50, 'eta_threshold': 0.3, 'name': '50_samples_eta03'},
        {'sample_size': 50, 'eta_threshold': 0.4, 'name': '50_samples_eta04'},
        {'sample_size': 50, 'eta_threshold': 0.6, 'name': '50_samples_eta06'},
        {'sample_size': 50, 'eta_threshold': 0.7, 'name': '50_samples_eta07'},
    ]
    
    results = {}
    
    print("MATH-500 Parameter Sweep Analysis")
    print("="*60)
    
    for exp in experiments:
        result = run_experiment(
            sample_size=exp['sample_size'],
            eta_threshold=exp['eta_threshold']
        )
        results[exp['name']] = result
        
        # Short break between experiments
        time.sleep(2)
    
    return results


def analyze_results(results: Dict[str, Dict]) -> None:
    """Analyze and visualize the experimental results."""
    
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Extract successful results
    successful = {name: result for name, result in results.items() if result.get('success', False)}
    
    if not successful:
        print("No successful experiments to analyze.")
        return
    
    # Create summary table
    print(f"\n{'Experiment':<20} {'Dimension':<12} {'R²':<8} {'Sequences':<12} {'High-Entropy':<12} {'Time(s)':<8}")
    print("-" * 80)
    
    for name, result in successful.items():
        dim = result['correlation_dimension']
        r2 = result['r_squared']
        total_seq = result['total_sequences']
        high_ent = result['high_entropy_sequences']
        proc_time = result['processing_time']
        
        print(f"{name:<20} {dim:<12.4f} {r2:<8.4f} {total_seq:<12,} {high_ent:<12,} {proc_time:<8.1f}")
    
    # Create visualizations
    create_analysis_plots(successful)
    
    # Statistical analysis
    analyze_trends(successful)


def create_analysis_plots(results: Dict[str, Dict]) -> None:
    """Create visualization plots for the analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    names = list(results.keys())
    dimensions = [results[name]['correlation_dimension'] for name in names]
    r_squared = [results[name]['r_squared'] for name in names]
    eta_thresholds = [results[name]['experiment_params']['eta_threshold'] for name in names]
    sample_sizes = [results[name]['experiment_params']['sample_size'] for name in names]
    high_entropy_fractions = [results[name]['high_entropy_fraction'] for name in names]
    
    # 1. Dimension vs Sample Size
    sample_size_results = [(size, dim) for size, dim in zip(sample_sizes, dimensions) if eta_thresholds[sample_sizes.index(size)] == 0.5]
    if sample_size_results:
        sizes, dims = zip(*sample_size_results)
        ax1.plot(sizes, dims, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Correlation Dimension')
        ax1.set_title('Dimension vs Sample Size (η=0.5)')
        ax1.grid(True, alpha=0.3)
    
    # 2. Dimension vs Eta Threshold
    eta_results = [(eta, dim) for eta, dim in zip(eta_thresholds, dimensions) if sample_sizes[eta_thresholds.index(eta)] == 50]
    if eta_results:
        etas, dims = zip(*eta_results)
        ax2.plot(etas, dims, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Eta Threshold')
        ax2.set_ylabel('Correlation Dimension')
        ax2.set_title('Dimension vs Eta Threshold (50 samples)')
        ax2.grid(True, alpha=0.3)
    
    # 3. R² quality assessment
    colors = ['green' if r2 > 0.95 else 'orange' if r2 > 0.90 else 'red' for r2 in r_squared]
    ax3.bar(range(len(names)), r_squared, color=colors, alpha=0.7)
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('R² Value')
    ax3.set_title('Fit Quality (R²)')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45)
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='R²=0.95')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. High-entropy fraction vs Eta threshold
    ax4.scatter(eta_thresholds, high_entropy_fractions, s=100, alpha=0.7, c=dimensions, cmap='viridis')
    ax4.set_xlabel('Eta Threshold')
    ax4.set_ylabel('High-Entropy Fraction')
    ax4.set_title('High-Entropy Fraction vs Eta Threshold')
    cbar = plt.colorbar(ax4.scatter(eta_thresholds, high_entropy_fractions, s=100, alpha=0.7, c=dimensions, cmap='viridis'), ax=ax4)
    cbar.set_label('Correlation Dimension')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('math500_parameter_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: math500_parameter_analysis.png")
    plt.show()


def analyze_trends(results: Dict[str, Dict]) -> None:
    """Analyze trends in the experimental results."""
    
    print(f"\n{'='*60}")
    print("TREND ANALYSIS")
    print(f"{'='*60}")
    
    # Extract data
    dimensions = [results[name]['correlation_dimension'] for name in results.keys()]
    eta_thresholds = [results[name]['experiment_params']['eta_threshold'] for name in results.keys()]
    sample_sizes = [results[name]['experiment_params']['sample_size'] for name in results.keys()]
    high_entropy_fractions = [results[name]['high_entropy_fraction'] for name in results.keys()]
    
    # Basic statistics
    print(f"Dimension Range: {min(dimensions):.3f} - {max(dimensions):.3f}")
    print(f"Average Dimension: {np.mean(dimensions):.3f} ± {np.std(dimensions):.3f}")
    
    # Sample size effect (with eta=0.5)
    sample_effect = [(size, dim) for size, dim in zip(sample_sizes, dimensions) 
                    if eta_thresholds[sample_sizes.index(size)] == 0.5]
    if len(sample_effect) > 1:
        sizes, dims = zip(*sample_effect)
        correlation = np.corrcoef(sizes, dims)[0, 1]
        print(f"\nSample Size Effect (η=0.5):")
        print(f"  Correlation with dimension: {correlation:.3f}")
        if correlation > 0.3:
            print("  → Dimension tends to increase with more data")
        elif correlation < -0.3:
            print("  → Dimension tends to decrease with more data")
        else:
            print("  → Dimension relatively stable across sample sizes")
    
    # Eta threshold effect (with 50 samples)
    eta_effect = [(eta, dim) for eta, dim in zip(eta_thresholds, dimensions) 
                  if sample_sizes[eta_thresholds.index(eta)] == 50]
    if len(eta_effect) > 1:
        etas, dims = zip(*eta_effect)
        correlation = np.corrcoef(etas, dims)[0, 1]
        print(f"\nEta Threshold Effect (50 samples):")
        print(f"  Correlation with dimension: {correlation:.3f}")
        if correlation > 0.3:
            print("  → Higher η (more filtering) increases dimension")
        elif correlation < -0.3:
            print("  → Higher η (more filtering) decreases dimension")
        else:
            print("  → Dimension relatively stable across η values")
    
    # Quality assessment
    r_squared_values = [results[name]['r_squared'] for name in results.keys()]
    high_quality = sum(1 for r2 in r_squared_values if r2 > 0.95)
    print(f"\nFit Quality:")
    print(f"  High quality (R² > 0.95): {high_quality}/{len(r_squared_values)}")
    print(f"  Average R²: {np.mean(r_squared_values):.3f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive MATH-500 dimension analysis")
    parser.add_argument("--run-sweep", action="store_true", 
                       help="Run parameter sweep experiments")
    parser.add_argument("--analyze-existing", type=str,
                       help="Analyze existing results from JSON file")
    
    args = parser.parse_args()
    
    if args.run_sweep:
        # Run new experiments
        results = run_parameter_sweep()
        
        # Save results
        output_file = "math500_parameter_sweep_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_for_json = {}
        for name, result in results.items():
            result_copy = result.copy()
            if 'epsilons' in result_copy:
                result_copy['epsilons'] = result_copy['epsilons'].tolist()
            if 'correlation_integrals' in result_copy:
                result_copy['correlation_integrals'] = result_copy['correlation_integrals'].tolist()
            results_for_json[name] = result_copy
        
        with open(output_file, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Analyze results
        analyze_results(results)
        
    elif args.analyze_existing:
        # Load and analyze existing results
        with open(args.analyze_existing, 'r') as f:
            results = json.load(f)
        
        analyze_results(results)
        
    else:
        print("Please specify --run-sweep to run new experiments or --analyze-existing <file> to analyze existing results")


if __name__ == "__main__":
    main()
