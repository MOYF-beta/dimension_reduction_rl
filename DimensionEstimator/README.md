# Correlation Dimension Estimator for Natural Language

This implementation is based on the paper ["Correlation Dimension of Natural Language in a Statistical Manifold"](https://arxiv.org/abs/2405.06321) by Xin Du and Kumiko Tanaka-Ishii (2024).

## Overview

This tool estimates the correlation dimension of natural language texts using the Grassberger-Procaccia algorithm reformulated in a statistical manifold with Fisher-Rao distance. The method uses large language models to obtain probability distributions for text sequences and measures their fractal dimension.

### Key Features

- **Statistical Manifold Analysis**: Uses Fisher-Rao distance instead of Euclidean distance
- **HuggingFace Integration**: Works with any HuggingFace causal language model
- **Entropy Filtering**: Filters out low-entropy regions to focus on global fractals
- **Dimension Reduction**: Optional vocabulary reduction for computational efficiency
- **Batch Processing**: Efficient processing of multiple texts

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- torch>=1.9.0
- transformers>=4.20.0
- numpy>=1.21.0
- scipy>=1.7.0
- tqdm>=4.62.0
- matplotlib>=3.3.0 (optional, for plotting)

## Usage

### Basic Usage

```python
from dimension_estimator import DimensionEstimator

# Initialize with a HuggingFace model
estimator = DimensionEstimator(
    model_name_or_path="gpt2",
    max_context_length=512,
    eta_threshold=0.5,
    dimension_reduction=1000
)

# Estimate dimension for a list of texts
texts = ["Your text here...", "Another text..."]
results = estimator.estimate_dimension(texts)

print(f"Correlation Dimension: {results['correlation_dimension']:.3f}")
print(f"R-squared: {results['r_squared']:.3f}")
```

### Advanced Usage

```python
# Use a larger model for better results
estimator = DimensionEstimator(
    model_name_or_path="gpt2-xl",
    device="cuda",  # Use GPU if available
    max_context_length=1024,
    eta_threshold=0.5,
    dimension_reduction=2000,
    batch_size=16
)

# Load texts from files
with open('book.txt', 'r') as f:
    long_text = f.read()

results = estimator.estimate_dimension([long_text])

# Plot correlation integral curve
estimator.plot_correlation_integral(results)
```

## Method Details

### The Algorithm

1. **Probability Extraction**: For each position t in a text, extract the probability distribution p_t over the vocabulary using a language model
2. **Entropy Filtering**: Filter out distributions where max(p_t) >= η (default η=0.5) to focus on high-entropy, global fractal patterns
3. **Distance Calculation**: Compute Fisher-Rao distances between probability distributions: d_FR(p_i, p_j) = 2 * arccos(Σ√(p_i(w) * p_j(w)))
4. **Correlation Integral**: Calculate C(ε) = fraction of pairs with distance < ε
5. **Dimension Estimation**: Find the slope ν in the power law C(ε) ∼ ε^ν using linear regression on log-log plot

### Parameters

- **model_name_or_path**: HuggingFace model identifier (e.g., "gpt2", "gpt2-xl")
- **max_context_length**: Maximum context length for the language model (default: 512)
- **eta_threshold**: Maximum probability threshold for entropy filtering (default: 0.5)
- **dimension_reduction**: Reduce vocabulary to this size for efficiency (optional)
- **device**: Device to run on ("auto", "cuda", "cpu")

### Expected Results

According to the paper, natural language typically exhibits:
- **Correlation dimension**: ~6.5
- **Universal across languages**: English (6.39±0.40), Chinese (6.81±0.58), Japanese (7.30±0.41), German (5.84±0.70)
- **Comparison with other processes**:
  - Shuffled text: ~13.0
  - Random model weights: ~80
  - Barabási-Albert networks: ~2.0
  - White noise: >100

## Implementation Notes

### Computational Complexity

- Time complexity: O(M × N²) where M is vocabulary size and N is sequence length
- Memory: Linear in sequence length
- The dimension reduction feature can reduce complexity from O(|V| × N²) to O(M × N²) where M ≪ |V|

### Numerical Considerations

- Uses Fisher-Rao distance for statistical manifold analysis
- Includes numerical stability measures (epsilon for zero probabilities)
- Filters correlation integral values for reliable regression

### Limitations

- Requires sufficient text length for reliable estimation (>1000 tokens recommended)
- Computational cost scales quadratically with sequence length
- Quality depends on the underlying language model
- Results are approximate due to finite context length and vocabulary reduction

## Example Output

```
Processing 1 texts...
Processing text 1/1
Total sequence length: 2547
High-entropy sequences: 1823 (η < 0.5)
Computing correlation integrals...
100%|██████████| 20/20 [02:15<00:00,  6.78s/it]
Estimated correlation dimension: 6.234
R-squared: 0.892

=== Results ===
Correlation Dimension: 6.2340
R-squared: 0.8920
Total sequences: 2547
High-entropy sequences: 1823
High-entropy fraction: 0.716
```

## References

```bibtex
@article{du2024correlation,
  title={Correlation Dimension of Natural Language in a Statistical Manifold},
  author={Du, Xin and Tanaka-Ishii, Kumiko},
  journal={arXiv preprint arXiv:2405.06321},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Please cite the original paper if you use this code in your research.
