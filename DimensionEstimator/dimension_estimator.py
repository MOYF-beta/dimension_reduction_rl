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
    
    def _get_probability_distributions(self, text: str) -> torch.Tensor:
        """
        Get probability distributions for each position in the text using KV cache optimization.
        
        Args:
            text: Input text
            
        Returns:
            Tensor of shape (sequence_length, vocab_size) containing probability distributions
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 2:
            raise ValueError("Text too short, need at least 2 tokens")
        
        all_probs = []
        
        # Use KV cache for efficient sequential processing
        with torch.no_grad():
            past_key_values = None
            input_ids = torch.tensor([tokens[0]], device=self.device).unsqueeze(0)  # Start with first token
            
            # Process tokens sequentially with KV cache
            for i in range(1, len(tokens)):
                # For the first iteration, we need to process the initial context
                if i == 1:
                    # Process initial context if longer than 1 token
                    if len(tokens) > 2:
                        initial_context_end = min(len(tokens), self.max_context_length)
                        input_ids = torch.tensor([tokens[:initial_context_end]], device=self.device)
                        outputs = self.model(input_ids=input_ids, use_cache=True)
                        past_key_values = outputs.past_key_values
                        
                        # Extract probabilities for each position from 1 to initial_context_end-1
                        for pos in range(1, initial_context_end):
                            logits = outputs.logits[0, pos - 1, :]  # pos-1 because logits are shifted
                            probs = F.softmax(logits, dim=-1)
                            
                            # Apply dimension reduction if specified
                            if self.reduction_map:
                                reduced_probs = torch.zeros(self.dimension_reduction, device=self.device)
                                for orig_idx, reduced_idx in self.reduction_map.items():
                                    reduced_probs[reduced_idx] += probs[orig_idx]
                                probs = reduced_probs
                            
                            all_probs.append(probs)
                        
                        # Update position to continue from where we left off
                        i = initial_context_end
                        if i >= len(tokens):
                            break
                    else:
                        # Simple case: just 2 tokens
                        outputs = self.model(input_ids=input_ids, use_cache=True)
                        past_key_values = outputs.past_key_values
                        
                        logits = outputs.logits[0, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        
                        if self.reduction_map:
                            reduced_probs = torch.zeros(self.dimension_reduction, device=self.device)
                            for orig_idx, reduced_idx in self.reduction_map.items():
                                reduced_probs[reduced_idx] += probs[orig_idx]
                            probs = reduced_probs
                        
                        all_probs.append(probs)
                        continue
                
                # For subsequent tokens, use cached key-values and only process the new token
                if past_key_values is not None:
                    # Check if we need to truncate cache due to max_context_length
                    if past_key_values[0][0].shape[2] >= self.max_context_length:
                        # Truncate the oldest entries from cache
                        truncate_length = past_key_values[0][0].shape[2] - self.max_context_length + 1
                        new_past_key_values = []
                        for layer_past in past_key_values:
                            new_layer_past = []
                            for kv in layer_past:
                                new_layer_past.append(kv[:, :, truncate_length:, :])
                            new_past_key_values.append(tuple(new_layer_past))
                        past_key_values = tuple(new_past_key_values)
                    
                    # Process only the current token
                    current_token = torch.tensor([[tokens[i]]], device=self.device)
                    outputs = self.model(
                        input_ids=current_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    
                    # Get probability distribution for the current position
                    logits = outputs.logits[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    
                    # Apply dimension reduction if specified
                    if self.reduction_map:
                        reduced_probs = torch.zeros(self.dimension_reduction, device=self.device)
                        for orig_idx, reduced_idx in self.reduction_map.items():
                            reduced_probs[reduced_idx] += probs[orig_idx]
                        probs = reduced_probs
                    
                    all_probs.append(probs)
        
        return torch.stack(all_probs)
    
    def _fisher_rao_distance_batch(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Fisher-Rao distance between batches of probability distributions.
        
        Args:
            p1: Tensor of shape (..., vocab_size)
            p2: Tensor of shape (..., vocab_size)
            
        Returns:
            Tensor of Fisher-Rao distances
        """
        # Ensure probabilities are positive (add small epsilon for numerical stability)
        eps = 1e-12
        p1 = torch.clamp(p1, min=eps)
        p2 = torch.clamp(p2, min=eps)
        
        # Calculate Bhattacharyya coefficient
        bc = torch.sum(torch.sqrt(p1 * p2), dim=-1)
        
        # Clamp to [0, 1] to avoid numerical issues with arccos
        bc = torch.clamp(bc, min=0.0, max=1.0)
        
        # Fisher-Rao distance
        distance = 2 * torch.arccos(bc)
        
        return distance

    def _fisher_rao_distance(self, p1: torch.Tensor, p2: torch.Tensor) -> float:
        """
        Calculate Fisher-Rao distance between two probability distributions.
        
        Formula: d_FR(p1, p2) = 2 * arccos(sum(sqrt(p1(w) * p2(w))))
        """
        # Ensure probabilities are positive (add small epsilon for numerical stability)
        eps = 1e-12
        p1 = torch.clamp(p1, min=eps)
        p2 = torch.clamp(p2, min=eps)
        
        # Calculate Bhattacharyya coefficient
        bc = torch.sum(torch.sqrt(p1 * p2))
        
        # Clamp to [0, 1] to avoid numerical issues with arccos
        bc = torch.clamp(bc, min=0.0, max=1.0)
        
        # Fisher-Rao distance
        distance = 2 * torch.arccos(bc)
        
        return distance.item()
    
    def _filter_high_entropy(self, prob_distributions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter probability distributions to keep only high-entropy ones.
        
        Args:
            prob_distributions: Tensor of probability distributions
            
        Returns:
            Tuple of (filtered_distributions, indices)
        """
        max_probs = torch.max(prob_distributions, dim=1)[0]
        high_entropy_mask = max_probs < self.eta_threshold
        
        filtered_distributions = prob_distributions[high_entropy_mask]
        indices = torch.where(high_entropy_mask)[0]
        
        return filtered_distributions, indices
    
    def _compute_correlation_integral_batch(self, prob_distributions: torch.Tensor, epsilons: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation integrals for multiple epsilon values using vectorized operations.
        
        Args:
            prob_distributions: Tensor of probability distributions [N, vocab_size]
            epsilons: Tensor of epsilon values [num_epsilons]
            
        Returns:
            Tensor of correlation integral values [num_epsilons]
        """
        N = prob_distributions.shape[0]
        if N < 2:
            return torch.zeros_like(epsilons)
        
        # Estimate memory usage and decide on processing strategy
        vocab_size = prob_distributions.shape[1]
        memory_required = N * N * vocab_size * 4  # More accurate memory estimate
        
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(self.device).total_memory * 0.8  # Use 80% of available memory
        else:
            available_memory = 4e9  # 4GB for CPU
        
        # Use conservative thresholds
        if memory_required < available_memory * 0.1 and N < 200:
            # Use full batch processing only for very small datasets
            return self._compute_correlation_integral_batch_full(prob_distributions, epsilons)
        else:
            # Use chunked processing for larger datasets
            # Calculate conservative chunk size based on available memory
            max_pairs_per_chunk = int(available_memory * 0.05 / (vocab_size * 4))  # Very conservative
            chunk_size = max(50, int(np.sqrt(max_pairs_per_chunk)))
            chunk_size = min(chunk_size, 200)  # Cap at 200 for safety
            print(f"Using chunked processing with chunk size: {chunk_size}")
            return self._compute_correlation_integral_chunked(prob_distributions, epsilons, chunk_size)

    def _compute_correlation_integral_batch_full(self, prob_distributions: torch.Tensor, epsilons: torch.Tensor) -> torch.Tensor:
        """
        Full batch computation for smaller datasets.
        """
        N = prob_distributions.shape[0]
        
        # Calculate all pairwise distances using broadcasting
        prob_expanded_i = prob_distributions.unsqueeze(1)  # [N, 1, vocab_size]
        prob_expanded_j = prob_distributions.unsqueeze(0)  # [1, N, vocab_size]
        
        # Calculate Fisher-Rao distances for all pairs
        distances = self._fisher_rao_distance_batch(prob_expanded_i, prob_expanded_j)  # [N, N]
        
        # Only consider upper triangular part (i < j)
        triu_mask = torch.triu(torch.ones(N, N, device=self.device), diagonal=1).bool()
        valid_distances = distances[triu_mask]  # [N*(N-1)/2]
        
        # For each epsilon, count distances less than epsilon
        correlation_integrals = []
        for eps in epsilons:
            count = torch.sum(valid_distances < eps).float()
            total_pairs = valid_distances.shape[0]
            correlation_integrals.append(count / total_pairs if total_pairs > 0 else 0.0)
        
        return torch.tensor(correlation_integrals, device=self.device)

    def _compute_correlation_integral_chunked(self, prob_distributions: torch.Tensor, epsilons: torch.Tensor, chunk_size: int = 1000) -> torch.Tensor:
        """
        Memory-efficient computation of correlation integrals using chunked processing.
        
        Args:
            prob_distributions: Tensor of probability distributions [N, vocab_size]
            epsilons: Tensor of epsilon values [num_epsilons]
            chunk_size: Size of chunks for processing
            
        Returns:
            Tensor of correlation integral values [num_epsilons]
        """
        N = prob_distributions.shape[0]
        if N < 2:
            return torch.zeros_like(epsilons)
        
        # Initialize counters for each epsilon
        total_counts = torch.zeros(len(epsilons), device=self.device)
        total_pairs = 0
        
        # Process in chunks to manage memory
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_i = prob_distributions[i:end_i]
            
            for j in range(i, N, chunk_size):
                end_j = min(j + chunk_size, N)
                chunk_j = prob_distributions[j:end_j]
                
                # Calculate distances for this chunk pair
                chunk_i_expanded = chunk_i.unsqueeze(1)  # [chunk_i_size, 1, vocab_size]
                chunk_j_expanded = chunk_j.unsqueeze(0)  # [1, chunk_j_size, vocab_size]
                
                distances = self._fisher_rao_distance_batch(chunk_i_expanded, chunk_j_expanded)
                
                # Only count upper triangular pairs
                if i == j:
                    # Same chunk - use upper triangular mask
                    chunk_size_i, chunk_size_j = distances.shape
                    triu_mask = torch.triu(torch.ones(chunk_size_i, chunk_size_j, device=self.device), diagonal=1).bool()
                    valid_distances = distances[triu_mask]
                elif i < j:
                    # Different chunks where i < j - use all pairs
                    valid_distances = distances.flatten()
                else:
                    # Skip when i > j to avoid double counting
                    continue
                
                # Count pairs for each epsilon
                for eps_idx, eps in enumerate(epsilons):
                    total_counts[eps_idx] += torch.sum(valid_distances < eps)
                
                total_pairs += valid_distances.numel()
        
        # Calculate correlation integrals
        correlation_integrals = total_counts.float() / total_pairs if total_pairs > 0 else torch.zeros_like(total_counts)
        
        return correlation_integrals

    def _estimate_correlation_dimension(
        self, 
        prob_distributions: torch.Tensor,
        epsilon_range: Optional[Tuple[float, float]] = None,
        num_epsilons: int = 20
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
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
            N = min(prob_distributions.shape[0], 100)  # Sample for efficiency
            with torch.no_grad():
                sample_dists = prob_distributions[:N]
                # Calculate sample distances
                prob_expanded_i = sample_dists.unsqueeze(1)  # [N, 1, vocab_size]
                prob_expanded_j = sample_dists.unsqueeze(0)  # [1, N, vocab_size]
                distances = self._fisher_rao_distance_batch(prob_expanded_i, prob_expanded_j)
                
                # Get upper triangular distances
                triu_mask = torch.triu(torch.ones(N, N, device=self.device), diagonal=1).bool()
                valid_distances = distances[triu_mask]
                
                if len(valid_distances) > 0:
                    min_dist = torch.min(valid_distances).item()
                    max_dist = torch.max(valid_distances).item()
                    epsilon_range = (min_dist * 0.1, max_dist * 0.9)
                else:
                    epsilon_range = (0.01, 1.0)
        
        # Generate epsilon values (log-spaced)
        epsilons = torch.logspace(
            math.log10(epsilon_range[0]), 
            math.log10(epsilon_range[1]), 
            num_epsilons,
            device=self.device
        )
        
        print("Computing correlation integrals with batch processing...")
        
        # Use batch processing for correlation integrals
        correlation_integrals = self._compute_correlation_integral_batch(prob_distributions, epsilons)
        
        # Filter out zero values for log-log regression
        nonzero_mask = correlation_integrals > 0
        if torch.sum(nonzero_mask) < 3:
            warnings.warn("Too few non-zero correlation integral values for reliable dimension estimation")
            return 0.0, 0.0, epsilons.cpu(), correlation_integrals.cpu()
        
        # Convert to numpy for linear regression (scipy only supports numpy)
        log_epsilons = torch.log10(epsilons[nonzero_mask]).cpu().numpy()
        log_correlations = torch.log10(correlation_integrals[nonzero_mask]).cpu().numpy()
        
        # Linear regression to find slope (correlation dimension)
        slope, intercept, r_value, p_value, std_err = linregress(log_epsilons, log_correlations)
        
        correlation_dimension = slope
        r_squared = r_value ** 2
        
        return correlation_dimension, r_squared, epsilons.cpu(), correlation_integrals.cpu()
    
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
                text_lengths.append(prob_dists.shape[0])
            except Exception as e:
                print(f"Warning: Failed to process text {i+1}: {e}")
                continue
        
        if not all_prob_distributions:
            raise ValueError("No texts could be processed successfully")
        
        # Concatenate all probability distributions
        combined_prob_distributions = torch.cat(all_prob_distributions, dim=0)
        
        print(f"Total sequence length: {combined_prob_distributions.shape[0]}")
        
        # Filter for high-entropy distributions
        filtered_dists, filtered_indices = self._filter_high_entropy(combined_prob_distributions)
        
        print(f"High-entropy sequences: {filtered_dists.shape[0]} (η < {self.eta_threshold})")
        
        if filtered_dists.shape[0] < 10:
            warnings.warn("Very few high-entropy sequences found. Results may be unreliable.")
        
        # Estimate correlation dimension
        correlation_dim, r_squared, epsilons, correlations = self._estimate_correlation_dimension(filtered_dists)
        
        # Calculate additional statistics
        results = {
            'correlation_dimension': correlation_dim,
            'r_squared': r_squared,
            'total_sequences': combined_prob_distributions.shape[0],
            'high_entropy_sequences': filtered_dists.shape[0],
            'high_entropy_fraction': filtered_dists.shape[0] / combined_prob_distributions.shape[0],
            'text_lengths': text_lengths,
            'eta_threshold': self.eta_threshold,
            'epsilons': epsilons.numpy() if isinstance(epsilons, torch.Tensor) else epsilons,
            'correlation_integrals': correlations.numpy() if isinstance(correlations, torch.Tensor) else correlations
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