# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
import math
from typing import Optional

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Implement scaled dot-product attention as described in Vaswani et al. [2017].
    
    Computes Attention(Q, K, V) = softmax(Q^T K / sqrt(d_k)) V
    
    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k) 
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Optional boolean mask of shape (..., n_queries, n_keys)
              True means attend, False means mask out (set to -inf before softmax)
              
    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    # Get the dimension of keys for scaling
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q^T @ K / sqrt(d_k)
    # Q shape: (..., n_queries, d_k)
    # K shape: (..., n_keys, d_k)
    # We want (..., n_queries, n_keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Convert boolean mask to float mask
        # True -> 0 (attend), False -> -inf (mask out)
        float_mask = torch.where(mask, 0.0, float('-inf'))
        scores = scores + float_mask
    
    # Apply softmax to get attention weights
    # Softmax over the last dimension (keys dimension)
    attention_weights = softmax(scores, dim=-1)
    
    # Apply attention weights to values
    # attention_weights shape: (..., n_queries, n_keys)
    # V shape: (..., n_keys, d_v)
    # Output shape: (..., n_queries, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output


def softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax operation on the specified dimension with numerical stability.
    
    Uses the trick of subtracting the maximum value in the specified dimension
    to avoid numerical overflow issues where exp(vi) can become inf.
    
    Args:
        input: Input tensor of arbitrary shape
        dim: Dimension to apply softmax over
        
    Returns:
        Tensor of same shape as input with softmax applied over the specified dimension
    """
    # For numerical stability, subtract the max value along the specified dimension
    # This keeps the largest exponent at 0, preventing overflow
    input_max = torch.max(input, dim=dim, keepdim=True)[0]
    input_stable = input - input_max
    
    # Elementwise compute exp of the numerically stable input
    exp_values = torch.exp(input_stable)
    
    # Compute the sum along the specified dimension
    exp_sum = torch.sum(exp_values, dim=dim, keepdim=True)
    
    # Return the normalized probabilities
    return exp_values / exp_sum

