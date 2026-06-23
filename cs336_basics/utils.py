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


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with numerical stability.
    
    Computes the cross-entropy loss ℓ = -log(softmax(logits)[targets])
    for each example, then returns the average across the batch.
    
    Uses numerical stability tricks:
    - Subtracts the maximum logit value to prevent overflow
    - Cancels log and exp where possible to avoid computing full softmax
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size) containing unnormalized logits
        targets: Tensor of shape (batch_size,) containing target class indices
        
    Returns:
        Scalar tensor containing the average cross-entropy loss across the batch
    """
    batch_size, vocab_size = logits.shape
    
    # For numerical stability, subtract the max logit from each row
    # This prevents overflow when computing exp
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    logits_stable = logits - max_logits
    
    # Compute log-sum-exp for the denominator
    # log(sum(exp(logits_stable))) = log(sum(exp(logits - max_logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=1))
    
    # Get the logits for the target classes
    # logits_stable[i, targets[i]] gives the numerator logit for example i
    target_logits = logits_stable[torch.arange(batch_size), targets]
    
    # Compute cross-entropy loss for each example
    # loss = -log(softmax(logits)[target]) = -log(exp(logit_target) / sum(exp(logits)))
    # = -logit_target + log(sum(exp(logits)))
    # Since we subtracted max_logits, this becomes:
    # = -(logit_target - max_logit) + log(sum(exp(logits - max_logit)))
    # = -logit_target + max_logit + log_sum_exp
    # But max_logit cancels out since target_logits already has max_logit subtracted
    losses = -target_logits + log_sum_exp
    
    # Return the average loss across the batch
    return torch.mean(losses)


def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (α_max)
        min_learning_rate: Minimum learning rate (α_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Number of cosine annealing iterations (T_c)
    
    Returns:
        Learning rate at iteration it
    """
    if it < warmup_iters:
        # Warmup phase: linear increase from 0 to max_learning_rate
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(progress * math.pi)
        )
    else:
        # Post-annealing phase: constant minimum learning rate
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients to have L2 norm at most max_l2_norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum L2 norm for gradients
    """
    eps = 1e-6  # PyTorch default for numerical stability
    
    # Collect all gradients
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    
    if not gradients:
        return
    
    # Compute total L2 norm of all gradients
    total_norm = 0.0
    for grad in gradients:
        total_norm += grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    # Apply clipping if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in gradients:
            grad.mul_(clip_coef)
