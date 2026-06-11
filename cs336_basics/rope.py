# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.linear import Linear


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotary position embeddings to query and key vectors by rotating
    pairs of dimensions according to their position in the sequence.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the RoPE module.
        
        Args:
            theta: Base value for the rotation angles (Θ)
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length to precompute for
            device: Device to store buffers
        """
        # Ensure d_k is even (required for pair-wise rotations)
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute the rotation angles for efficiency
        # θ_i,k = i * θ^(2k/d) for k ∈ {1, ..., d/2}
        self._precompute_rotations(device)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)
            
        Returns:
            Rotated tensor of same shape as input
        """
        # Get the sequence length
        seq_len = x.shape[-2]
        
        # Extract cos and sin values for the specified positions
        # token_positions shape: (..., seq_len)
        # cos_cached/sin_cached shape: (max_seq_len, d_k//2)
        
        # Index into our precomputed cos/sin arrays using token_positions
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)
        
        # Apply the rotary position embedding
        return self._apply_rotary_pos_emb(x, cos, sin)


    def _precompute_rotations(self, device: Optional[torch.device]):
        """Precompute cos and sin values for all positions and dimensions."""
        
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        
        # Create dimension indices for pairs: [0, 1, 2, ..., d_k/2-1]
        dim_indices = torch.arange(self.d_k // 2, dtype=torch.float32, device=device)
        
        # Compute θ^(2k/d) for each dimension pair k
        # This gives us the base rotation angles for each dimension pair
        inv_freq = 1.0 / (self.theta ** (2 * dim_indices / self.d_k))
        
        # Compute all rotation angles: positions[:, None] * inv_freq[None, :]
        # Shape: (max_seq_len, d_k//2)
        angles = torch.outer(positions, inv_freq)
        
        # Precompute cos and sin values
        # Shape: (max_seq_len, d_k//2)
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        
        # Register as buffers (not parameters, since we don't want to learn them)
        self.register_buffer('cos_cached', cos_values, persistent=False)
        self.register_buffer('sin_cached', sin_values, persistent=False)


    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            cos: Cosine values of shape (..., seq_len, d_k//2)
            sin: Sine values of shape (..., seq_len, d_k//2)
            
        Returns:
            Rotated tensor of same shape as input
        """
        # Split x into pairs: x_1, x_2, x_3, x_4, ... -> (x_1, x_2), (x_3, x_4), ...
        # Shape: (..., seq_len, d_k//2)
        x1 = x[..., 0::2]  # Even indices: 0, 2, 4, ...
        x2 = x[..., 1::2]  # Odd indices: 1, 3, 5, ...
        
        # Apply rotation:
        # [x1']   [cos -sin] [x1]   [x1*cos - x2*sin]
        # [x2'] = [sin  cos] [x2] = [x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave the rotated pairs back together
        # We need to combine (x1', x2') back into the original format
        result = torch.empty_like(x)
        result[..., 0::2] = rotated_x1  # Even indices
        result[..., 1::2] = rotated_x2  # Odd indices
        
        return result

