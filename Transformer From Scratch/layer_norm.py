class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: Optional[float] = 1e-6,
        dtype: Optional[torch.dtype] = torch.float32
      ):
        """
        Initialize Layer Normalization module.

        Args:
            normalized_shape (int): Shape of the input tensor's normalized dimension(s).
                For example, if the input is (batch, seq_len, features), this would be `features` or
                a tuple of the last dimensions.
            eps (Optional[float]): Small constant added to the variance to prevent division by zero.
                Defaults to 1e-6.
            dtype (Optional[torch.dtype]): Data type for the learnable parameters. Defaults to torch.float32.

        Attributes:
            gamma (torch.nn.Parameter): Learnable scaling factor, initialized to ones.
            beta (torch.nn.Parameter): Learnable shift factor, initialized to zeros.
            eps (float): Small constant for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.eps = eps # Small value for numerical stability
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype).to(device)) # Scaling factor
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape, dtype=dtype).to(device)) # Shifiting factor

    def forward(
        self,
        x: torch.Tensor
      ):
        """
        Perform Layer Normalization on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., normalized_shape).

        Returns:
            x (torch.Tensor): Normalized and transformed tensor of the same shape as the input,
                computed as `gamma * normalized_x + beta`.
        """
        x = x.to(device) # Ensure x is on same device
        mean = x.mean(dim=-1, keepdim=True) # Compute mean over the last dimension
        var = x.var(dim=-1, unbiased=False, keepdim=True) # Compute variance over the last dimension
        normalized_x = (x - mean) / torch.sqrt(var + self.eps) # Normalize the input
        x = self.gamma * normalized_x + self.beta # Apply scaling and shifting
        return x
