# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    """
    SwiGLU position-wise feed-forward network.
    
    Combines SiLU (Swish) activation with Gated Linear Units (GLU).
    FFN(x) = W2(SiLU(W1x) ⊙ W3x)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the SwiGLU module.
        
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension of feed-forward network
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear transformations as per SwiGLU formula
        # W1: d_model -> d_ff (for SiLU path)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # W2: d_ff -> d_model (output projection)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # W3: d_model -> d_ff (for gating path)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        # Apply the three linear transformations
        w1_x = self.w1(x)    # (..., d_ff)
        w3_x = self.w3(x)    # (..., d_ff)
        
        # Apply SiLU (Swish) activation: SiLU(x) = x * sigmoid(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        
        # Element-wise multiplication (gating)
        gated = silu_w1_x * w3_x  # (..., d_ff)
        
        # Final linear transformation
        output = self.w2(gated)   # (..., d_model)
        
        return output
    