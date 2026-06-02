# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional

class Embedding(nn.Module):
    """
    Token embedding module.
    
    Maps token IDs to dense vectors using manual indexing (no nn.Embedding).
    """


    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the Embedding module.
        
        Args:
            num_embeddings: Size of the Vocabulary
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding weight parameter with shape (num_embeddings, embedding_dim)
        # Store with d_model (embedding_dim) as the final dimension
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Embedding: N(μ=0, σ²=1) truncated at [-3, 3]
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: Token ID tensor of shape (...,)
            
        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        # Manual embedding lookup without using nn.functional.embedding
        # weight[token_ids] uses PyTorch's advanced indexing: efficiently retrieves multiple rows at once
        return self.weight[token_ids]