# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm normalizes activations using the root mean square and applies
    learnable scaling parameters. This is used in pre-norm Transformer blocks.
    """
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # Learnable gain parameters (one per model dimension)
        self.weight = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        
        # RMSNorm: Initialize to 1
        nn.init.ones_(self.weight)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Store original dtype for later restoration
        in_dtype = x.dtype
        
        # Upcast to float32 to prevent overflow when squaring
        x = x.to(torch.float32)
        
        # Calculate RMS along the last dimension (d_model)
        # RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        
        # Apply RMSNorm: x_i / RMS(x) * g_i
        normalized = x / rms
        
        # Apply learnable gain parameters
        result = normalized * self.weight
        
        # Return result in original dtype
        return result.to(in_dtype)

