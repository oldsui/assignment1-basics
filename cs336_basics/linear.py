# ref: 
# - https://github.com/DhyeyMavani2003/stanford-cs336-assignment1-basics-solution/blob/main/cs336_basics/nn_utils.py

import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    """
    Linear transformation module without bias.
    
    Performs y = W @ x where W is a learnable weight matrix.
    This follows the column vector convention used in the assignment.
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module. 
        Args:
            in_features: Size of input features (din)
            out_features: Size of output features (dout)  
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Create weight parameter W of shape (out_features, in_features)
        # This stores W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Linear weights: N(μ=0, σ²=2/(din+dout)) truncated at [-3σ, 3σ]
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Use @ operator for matrix multiplication
        # x is a row vector
        # x @ W^T gives us the desired transformation

        return x @ self.weight.T