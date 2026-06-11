# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.rope import RoPE
from cs336_basics.linear import Linear
from cs336_basics.utils import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with causal masking and optional RoPE.
    
    Implements the multi-head attention mechanism from "Attention Is All You Need"
    (Vaswani et al., 2017) with causal masking for autoregressive language modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        use_rope: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Dimensionality of the model embeddings
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE precomputation
            theta: RoPE theta parameter
            use_rope: Whether to apply rotary position embeddings
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v = d_model / h
        self.d_v = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Projection layers for Q, K, V
        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype) 
        self.v_proj = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        
        # Output projection
        self.o_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        
        # RoPE if enabled
        if use_rope:
            self.rope = RoPE(
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device
            )
        else:
            self.rope = None
            
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position tensor of shape (..., seq_len)
                           If not provided and RoPE is enabled, uses default positions
                           
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # Project to Q, K, V
        # Each has shape (..., seq_len, num_heads * d_k/d_v)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape and transpose to get (..., num_heads, seq_len, d_k/d_v)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
        
        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            if token_positions is None:
                # Default to sequential positions
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
                # Expand to match batch dimensions
                for _ in range(len(batch_shape)):
                    token_positions = token_positions.unsqueeze(0)
                token_positions = token_positions.expand(*batch_shape, seq_len)
            
            # Apply RoPE to Q and K (not V)
            # RoPE expects (..., seq_len, d_k), so we need to handle the head dimension
            # Reshape to treat heads as part of batch dimension
            q_shape = Q.shape
            k_shape = K.shape
            
            Q_for_rope = Q.reshape(-1, seq_len, self.d_k)
            K_for_rope = K.reshape(-1, seq_len, self.d_k)
            
            # Expand token_positions to match the head dimension
            pos_for_rope = token_positions.unsqueeze(-2).expand(*batch_shape, self.num_heads, seq_len)
            pos_for_rope = pos_for_rope.reshape(-1, seq_len)
            
            Q_for_rope = self.rope(Q_for_rope, pos_for_rope)
            K_for_rope = self.rope(K_for_rope, pos_for_rope)
            
            Q = Q_for_rope.reshape(q_shape)
            K = K_for_rope.reshape(k_shape)
        
        # Create causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        # Convert to attention mask format (True = attend, False = mask out)
        causal_mask = ~mask  # Invert: False where we should mask, True where we should attend
        
        # Expand mask to match batch and head dimensions
        # Start with shape (seq_len, seq_len)
        # Add dimensions for batch and head: (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Now expand to match the actual batch and head dimensions
        # Target shape: (*batch_shape, num_heads, seq_len, seq_len)
        target_shape = (*batch_shape, self.num_heads, seq_len, seq_len)
        causal_mask = causal_mask.expand(target_shape)
        
        # Apply scaled dot-product attention
        # Q, K, V shapes: (..., num_heads, seq_len, d_k/d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        
        # Reshape back to (..., seq_len, num_heads * d_v)
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_shape, seq_len, self.num_heads * self.d_v)
        
        # Apply output projection
        output = self.o_proj(attn_output)
        
        return output
