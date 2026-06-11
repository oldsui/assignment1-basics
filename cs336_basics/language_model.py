# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.transformer import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.
    
    Implements the full architecture from token embeddings through multiple
    Transformer blocks to final language modeling head.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Transformer Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            context_length: Maximum context length
            d_model: Model dimensionality
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward dimensionality
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        # Token embeddings
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Language modeling head (output projection to vocabulary)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer Language Model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            token_positions: Optional position indices for RoPE
            
        Returns:
            Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create default token positions if not provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
