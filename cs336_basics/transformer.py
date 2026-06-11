# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.mha import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with multi-head self-attention and feed-forward network.
    
    Implements the architecture:
    1. y1 = x + MultiHeadSelfAttention(RMSNorm(x))
    2. y2 = y1 + FeedForward(RMSNorm(y1))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Dimensionality of the model embeddings
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Layer normalization for attention sublayer
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention with RoPE
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
            device=device,
            dtype=dtype
        )
        
        # Layer normalization for feed-forward sublayer
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Transformer block.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position tensor for RoPE
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # First sublayer: Multi-head self-attention with residual connection
        # y1 = x + MultiHeadSelfAttention(RMSNorm(x))
        attn_input = self.ln1(x)
        attn_output = self.attn(attn_input, token_positions)
        y1 = x + attn_output
        
        # Second sublayer: Feed-forward with residual connection
        # y2 = y1 + FeedForward(RMSNorm(y1))
        ffn_input = self.ln2(y1)
        ffn_output = self.ffn(ffn_input)
        y2 = y1 + ffn_output
        
        return y2
