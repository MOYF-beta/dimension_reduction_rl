import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional, Dict
import warnings
from scipy.stats import linregress
from tqdm import tqdm
import math


class DimensionEstimator:
    """
    Correlation Dimension Estimator for Natural Language in a Statistical Manifold
    
    Based on the paper "Correlation Dimension of Natural Language in a Statistical Manifold"
    by Xin Du and Kumiko Tanaka-Ishii (2024)
    
    This implementation uses the Grassberger-Procaccia algorithm with Fisher-Rao distance
    to estimate the correlation dimension of text sequences.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        max_context_length: int = 512,
        eta_threshold: float = 0.5,
        dimension_reduction: Optional[int] = None,
        batch_size: int = 16
    ):
        """
        Initialize the dimension estimator.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            device: Device to run the model on ("auto", "cuda", "cpu")
            max_context_length: Maximum context length for the model
            eta_threshold: Entropy threshold for filtering (default 0.5 from paper)
            dimension_reduction: If provided, reduce vocabulary to this size
            batch_size: Batch size for processing
        """
        self.device = self._get_device(device)
        self.max_context_length = max_context_length
        self.eta_threshold = eta_threshold
        self.dimension_reduction = dimension_reduction
        self.batch_size = batch_size
        
        # Load model and tokenizer
        print(f"Loading model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32
        ).to(self.device)
        
        # Set padding token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.vocab_size = len(self.tokenizer)
        
        # Create dimension reduction mapping if specified
        if self.dimension_reduction:
            self.reduction_map = self._create_reduction_map()
        else:
            self.reduction_map = None
            
        print(f"Model loaded. Vocabulary size: {self.vocab_size}")
        if self.dimension_reduction:
            print(f"Using dimension reduction to {self.dimension_reduction}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _create_reduction_map(self) -> Dict[int, int]:
        """Create mapping for dimension reduction using modulo function."""
        return {i: i % self.dimension_reduction for i in range(self.vocab_size)}
    
    def _get_probability_distributions(self, text: str) -> np.ndarray:
        """
        Get probability distributions for each position in the text.
        
        Args:
            text: Input text
            
        Returns:
            Array of shape (sequence_length, vocab_size) containing probability distributions
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 2:
            raise ValueError("Text too short, need at least 2 tokens")
        
        probability_distributions = []
        
        # Process in batches to handle memory efficiently
        with torch.no_grad():
            for i in range(1, len(tokens)):  # Start from 1 since we need context
                # Get context (limited by max_context_length)
                start_idx = max(0, i - self.max_context_length)
                context_tokens = tokens[start_idx:i]
                
                # Convert to tensor
                input_ids = torch.tensor([context_tokens]).to(self.device)
                
                # Get model outputs
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position logits
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                
                # Apply dimension reduction if specified
                if self.reduction_map:
                    reduced_probs = np.zeros(self.dimension_reduction)
                    for orig_idx, reduced_idx in self.reduction_map.items():
                        reduced_probs[reduced_idx] += probs[orig_idx]
                    probs = reduced_probs
                
                probability_distributions.append(probs)
        
        return np.array(probability_distributions)
    
    def _fisher_rao_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate Fisher-Rao distance between two probability distributions.
        
        Formula: d_FR(p1, p2) = 2 * arccos(sum(sqrt(p1(w) * p2(w))))
        """
        # Ensure probabilities are positive (add small epsilon for numerical stability)
        eps = 1e-12
        p1 = np.maximum(p1, eps)
        p2 = np.maximum(p2, eps)
        
        # Calculate Bhattacharyya coefficient
        bc = np.sum(np.sqrt(p1 * p2))
        
        # Clamp to [0, 1] to avoid numerical issues with arccos
        bc = np.clip(bc, 0.0, 1.0)
        
        # Fisher-Rao distance
        distance = 2 * np.arccos(bc)
        
        return distance
    
    def _filter_high_entropy(self, prob_distributions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter probability distributions to keep only high-entropy ones.
        
        Args:
            prob_distributions: Array of probability distributions
            
        Returns:
            Tuple of (filtered_distributions, indices)
        """
        max_probs = np.max(prob_distributions, axis=1)
        high_entropy_mask = max_probs < self.eta_threshold
        
        filtered_distributions = prob_distributions[high_entropy_mask]
        indices = np.where(high_entropy_mask)[0]
        
        return filtered_distributions, indices
    
    def _compute_correlation_integral(self, prob_distributions: np.ndarray, epsilon: float) -> float:
        """
        Compute the correlation integral C(ε) for a given epsilon.
        
        Args:
            prob_distributions: Array of probability distributions
            epsilon: Distance threshold
            
        Returns:
            Correlation integral value
        """
        N = len(prob_distributions)
        if N < 2:
            return 0.0
        
        count = 0
        total_pairs = 0
        
        # Calculate pairwise distances
        for i in range(N):
            for j in range(i + 1, N):
                distance = self._fisher_rao_distance(prob_distributions[i], prob_distributions[j])
                total_pairs += 1
                if distance < epsilon:
                    count += 1
        
        # Return correlation integral (normalized by total pairs)
        return count / total_pairs if total_pairs > 0 else 0.0
    
    def _estimate_correlation_dimension(
        self, 
        prob_distributions: np.ndarray,
        epsilon_range: Optional[Tuple[float, float]] = None,
        num_epsilons: int = 20
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Estimate correlation dimension using linear regression on log-log plot.
        
        Args:
            prob_distributions: Filtered probability distributions
            epsilon_range: Range of epsilon values (min, max)
            num_epsilons: Number of epsilon values to test
            
        Returns:
            Tuple of (correlation_dimension, r_squared, epsilons, correlation_integrals)
        """
        if epsilon_range is None:
            # Estimate reasonable epsilon range based on typical distances
            distances = []
            N = min(len(prob_distributions), 100)  # Sample for efficiency
            for i in range(N):
                for j in range(i + 1, min(i + 10, N)):
                    dist = self._fisher_rao_distance(prob_distributions[i], prob_distributions[j])
                    distances.append(dist)
            
            if distances:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                epsilon_range = (min_dist * 0.1, max_dist * 0.9)
            else:
                epsilon_range = (0.01, 1.0)
        
        # Generate epsilon values (log-spaced)
        epsilons = np.logspace(
            np.log10(epsilon_range[0]), 
            np.log10(epsilon_range[1]), 
            num_epsilons
        )
        
        correlation_integrals = []
        
        print("Computing correlation integrals...")
        for epsilon in tqdm(epsilons):
            c_eps = self._compute_correlation_integral(prob_distributions, epsilon)
            correlation_integrals.append(c_eps)
        
        correlation_integrals = np.array(correlation_integrals)
        
        # Filter out zero values for log-log regression
        nonzero_mask = correlation_integrals > 0
        if np.sum(nonzero_mask) < 3:
            warnings.warn("Too few non-zero correlation integral values for reliable dimension estimation")
            return 0.0, 0.0, epsilons, correlation_integrals
        
        log_epsilons = np.log10(epsilons[nonzero_mask])
        log_correlations = np.log10(correlation_integrals[nonzero_mask])
        
        # Linear regression to find slope (correlation dimension)
        slope, intercept, r_value, p_value, std_err = linregress(log_epsilons, log_correlations)
        
        correlation_dimension = slope
        r_squared = r_value ** 2
        
        return correlation_dimension, r_squared, epsilons, correlation_integrals
    
    def estimate_dimension(self, texts: List[str]) -> Dict[str, float]:
        """
        Estimate correlation dimension for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary containing dimension estimates and statistics
        """
        print(f"Processing {len(texts)} texts...")
        
        all_prob_distributions = []
        text_lengths = []
        
        # Process each text
        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}")
            try:
                prob_dists = self._get_probability_distributions(text)
                all_prob_distributions.append(prob_dists)
                text_lengths.append(len(prob_dists))
            except Exception as e:
                print(f"Warning: Failed to process text {i+1}: {e}")
                continue
        
        if not all_prob_distributions:
            raise ValueError("No texts could be processed successfully")
        
        # Concatenate all probability distributions
        combined_prob_distributions = np.vstack(all_prob_distributions)
        
        print(f"Total sequence length: {len(combined_prob_distributions)}")
        
        # Filter for high-entropy distributions
        filtered_dists, filtered_indices = self._filter_high_entropy(combined_prob_distributions)
        
        print(f"High-entropy sequences: {len(filtered_dists)} (η < {self.eta_threshold})")
        
        if len(filtered_dists) < 10:
            warnings.warn("Very few high-entropy sequences found. Results may be unreliable.")
        
        # Estimate correlation dimension
        correlation_dim, r_squared, epsilons, correlations = self._estimate_correlation_dimension(filtered_dists)
        
        # Calculate additional statistics
        results = {
            'correlation_dimension': correlation_dim,
            'r_squared': r_squared,
            'total_sequences': len(combined_prob_distributions),
            'high_entropy_sequences': len(filtered_dists),
            'high_entropy_fraction': len(filtered_dists) / len(combined_prob_distributions),
            'text_lengths': text_lengths,
            'eta_threshold': self.eta_threshold,
            'epsilons': epsilons,
            'correlation_integrals': correlations
        }
        
        print(f"Estimated correlation dimension: {correlation_dim:.3f}")
        print(f"R-squared: {r_squared:.3f}")
        
        return results
    
    def plot_correlation_integral(self, results: Dict) -> None:
        """
        Plot the correlation integral curve (requires matplotlib).
        
        Args:
            results: Results dictionary from estimate_dimension
        """
        try:
            import matplotlib.pyplot as plt
            
            epsilons = results['epsilons']
            correlations = results['correlation_integrals']
            
            # Filter non-zero values
            nonzero_mask = correlations > 0
            
            plt.figure(figsize=(10, 6))
            plt.loglog(epsilons[nonzero_mask], correlations[nonzero_mask], 'bo-', alpha=0.7)
            plt.xlabel('ε (epsilon)')
            plt.ylabel('C(ε) (correlation integral)')
            plt.title(f'Correlation Integral Curve\nCorrelation Dimension: {results["correlation_dimension"]:.3f}, R²: {results["r_squared"]:.3f}')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("matplotlib not available. Install it to plot correlation integral curves.")