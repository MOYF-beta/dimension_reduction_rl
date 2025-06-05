# MATH-500 Dataset Correlation Dimension Analysis

This directory contains scripts to estimate the correlation dimension of the MATH-500 dataset using the DimensionEstimator based on the Du & Tanaka-Ishii (2024) paper.

## Dataset

The [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) contains 500 mathematical competition problems from various subjects:
- Algebra
- Geometry  
- Number Theory
- Precalculus
- Probability & Counting
- Intermediate Algebra

Each problem includes:
- Problem statement
- Detailed solution
- Final answer
- Subject category
- Difficulty level

## Scripts

### 1. Quick Test (`quick_test_math500.py`)

Run a quick test with 10 sample problems to verify everything works:

```bash
python quick_test_math500.py
```

This script:
- Loads a small sample (10 problems) 
- Uses GPT-2 model for fast processing
- Estimates dimension on the sample
- Completes in ~1-2 minutes

### 2. Full Analysis (`estimate_math500_dimension.py`)

Run complete dimension estimation on the full dataset:

```bash
# Basic usage
python estimate_math500_dimension.py

# With custom parameters
python estimate_math500_dimension.py --model microsoft/DialoGPT-medium --sample-size 100 --eta-threshold 0.4
```

#### Command Line Options:

- `--model`: Hugging Face model name (default: `gpt2`)
- `--max-context`: Maximum context length (default: 512)  
- `--eta-threshold`: Entropy threshold for filtering (default: 0.5)
- `--batch-size`: Batch size for processing (default: 8)
- `--dimension-reduction`: Reduce vocabulary size (optional)
- `--sample-size`: Use subset of N texts (optional)
- `--output`: JSON output file (default: `math500_dimension_results.json`)
- `--plot`: Plot output file (default: `math500_dimension_analysis.png`)
- `--no-plot`: Skip generating plots

## Example Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Quick test
python quick_test_math500.py

# Full analysis with small sample
python estimate_math500_dimension.py --sample-size 50
```

### Advanced Usage
```bash
# Use a more sophisticated model
python estimate_math500_dimension.py --model microsoft/DialoGPT-medium

# Reduce vocabulary dimension for faster processing
python estimate_math500_dimension.py --dimension-reduction 1000

# Higher entropy threshold to get more sequences
python estimate_math500_dimension.py --eta-threshold 0.7

# Process with larger batches (requires more GPU memory)
python estimate_math500_dimension.py --batch-size 16
```

## Expected Results

Based on the Du & Tanaka-Ishii (2024) paper, we expect:

- **Natural Language**: Correlation dimension around 2-4
- **Mathematical Text**: Potentially higher dimension due to more structured/formal language
- **Competition Problems**: May show different characteristics than general mathematical text

The actual results will depend on:
- Choice of language model (affects probability distributions)
- Entropy threshold (affects which sequences are analyzed)
- Context length (affects local vs global patterns)

## Output Files

### JSON Results (`math500_dimension_results.json`)

Contains:
- `correlation_dimension`: Estimated dimension value
- `r_squared`: Quality of linear fit
- `total_sequences`: Number of sequences processed
- `high_entropy_sequences`: Number used in analysis
- `processing_time`: Time taken
- `text_statistics`: Dataset statistics
- `epsilons` & `correlation_integrals`: Raw correlation data

### Visualization (`math500_dimension_analysis.png`)

Four-panel plot showing:
1. **Correlation Integral Curve**: Log-log plot for dimension estimation
2. **Text Length Distribution**: Histogram of problem lengths  
3. **Entropy Filtering**: Pie chart showing sequence filtering
4. **Dataset Statistics**: Bar chart of corpus statistics

## Technical Details

### Method
1. **Text Processing**: Combine problem + solution text
2. **Tokenization**: Use language model tokenizer
3. **Probability Extraction**: Get next-token probability distributions
4. **Entropy Filtering**: Keep high-entropy (informative) sequences
5. **Distance Calculation**: Fisher-Rao distance between distributions
6. **Correlation Integral**: Count close pairs for different distance thresholds
7. **Dimension Estimation**: Linear regression on log-log plot

### Performance
- **GPU Recommended**: Significantly faster with CUDA
- **Memory Usage**: ~2-8GB GPU memory depending on batch size
- **Processing Time**: 
  - Quick test (10 problems): ~1-2 minutes
  - Full dataset (500 problems): ~30-60 minutes
  - Sample (50-100 problems): ~5-15 minutes

### Memory Optimization
The implementation includes automatic memory management:
- Chunked processing for large datasets
- Batch size adaptation based on available memory
- Automatic fallback to CPU if GPU memory insufficient

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Reduce batch size
   python estimate_math500_dimension.py --batch-size 4
   
   # Use dimension reduction
   python estimate_math500_dimension.py --dimension-reduction 500
   ```

2. **Too Few High-Entropy Sequences**:
   ```bash
   # Lower entropy threshold
   python estimate_math500_dimension.py --eta-threshold 0.3
   ```

3. **Slow Processing**:
   ```bash
   # Use smaller sample
   python estimate_math500_dimension.py --sample-size 100
   
   # Reduce context length
   python estimate_math500_dimension.py --max-context 256
   ```

### Dependencies
Make sure all required packages are installed:
```bash
pip install torch transformers datasets numpy scipy matplotlib tqdm
```

## Research Applications

This analysis can help understand:
- **Mathematical Language Structure**: How mathematical text differs from natural language
- **Problem Complexity**: Whether dimension correlates with problem difficulty
- **Subject Differences**: Dimensional characteristics across mathematical subjects
- **Model Comparison**: How different language models represent mathematical content

## Citation

If you use this analysis in research, please cite:

```bibtex
@article{du2024correlation,
  title={Correlation Dimension of Natural Language in a Statistical Manifold},
  author={Du, Xin and Tanaka-Ishii, Kumiko},
  journal={arXiv preprint arXiv:2401.xxxxx},
  year={2024}
}
```
