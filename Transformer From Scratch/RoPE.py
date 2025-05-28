class RoPE(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0
      ):
        """
        Initialize rotary positional embeddings.

          Attributes:
              head_dim (int): Dimensionality of each attention head.
                  Must be even for pair splitting in RoPE.
              base (int): Denominator raised to the power of 2i/d.

          Raises:
              ValueError if `head_dim % 2 != 0`
        """
        super(RoPE, self).__init__()

        # Ensure evenness for splitting
        if head_dim % 2 != 0:
            logger.debug(f"head_dim must be divisible by 2, head_dim: {head_dim}") # Log error
            raise ValueError(f"head_dim must be divisible by 2, head_dim: {head_dim}") # End program
        self.head_dim = head_dim

        # Calculate inverse frequency (occurs in log space) and store it as a buffer
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def compute_sine_cosine(
        self,
        T: int,
        offset: int = 0
      ):
        """
        Compute Sine and Cosine Rotation Matrices.

          Args:
              T (int): Sequence length; number of tokens in the input sequence.

          Returns:
              sin (torch.Tensor): Sine values of shape [1, 1, T, head_dim//2] for rotary positional embeddings.
              cos (torch.Tensor): Cosine values of shape [1, 1, T, head_dim//2] for rotary positional embeddings.
        """
        # Compute position indicies
        pos = torch.arange(offset, offset + T, dtype=self.inv_freq.dtype, device=self.inv_freq.device).unsqueeze(1)

        # Compute position angles
        theta = pos * self.inv_freq # [T, head_dim//2]

        # Compute sine and cosine tensors
        sin = torch.sin(theta).unsqueeze(0).unsqueeze(0) # [1, 1, T, head_dim//2]
        cos = torch.cos(theta).unsqueeze(0).unsqueeze(0) # [1, 1, T, head_dim//2]

        # Ensure tensors on same device
        return sin.to(device), cos.to(device)

    def create_rotary(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor
      ):
        """
        Create rotary positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, num_heads, head_dim].
            sin (torch.Tensor): Sine values of shape [1, 1, T, head_dim//2] for rotary positional embeddings.
            cos (torch.Tensor): Cosine values of shape [1, 1, T, head_dim//2] for rotary positional embeddings.

        Returns:
            torch.Tensor: Rotated tensor with shape: [B, T, num_heads, head_dim].
        """
        # Split head_dim into two parts for rotation
        x_reshape = x.reshape(*x.shape[:-1], -1, 2) # [B, T, num_heads, head_dim//2, 2]

        # Split into even and odd indices
        x1 = x_reshape[..., 0] # [B, T, num_heads, head_dim//2]
        x2 = x_reshape[..., 1] # [B, T, num_heads, head_dim//2]

        # Permute to ensure correct broadcasting
        sin = sin.permute(0, 2, 1, 3).expand(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])
        cos = cos.permute(0, 2, 1, 3).expand(x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3])

        # Apply rotary embeddings
        x_rotated_even = x1 * cos - x2 * sin # [B, T, num_heads, head_dim//2]
        x_rotated_odd = x1 * sin + x2 * cos # [B, T, num_heads, head_dim//2]

        # Interleave even and odd rotations together
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1) # [B, T, num_heads, head_dim//2, 2]
        return x_rotated.reshape(*x.shape) # [B, T, num_heads, head_dim]

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
      ):
        """
        Apply rotary positional embeddings to input tensor, x.

        Args:
              x (torch.Tensor): Input tensor of shape: [B, T, d_model]

        Returns:
            (torch.Tensor): Tensor with applied rotary positional embeddings of shape: [B, T, d_model].

        Raises:
            ValueError if `d_model % self.head_dim != 0`
        """
        B, T, d_model = x.size()
        if d_model % self.head_dim != 0:
            logger.error(f"d_model ({d_model}) must be divisible by head_dim ({self.head_dim})") # Log error
            raise ValueError(f"d_model ({d_model}) must be divisible by head_dim ({self.head_dim})") # End program
        num_heads = d_model // self.head_dim

        x = x.view(B, T, num_heads, self.head_dim) # [B, T, num_heads, head_dim]
        sin, cos = self.compute_sine_cosine(T, offset) # [1, 1, T, head_dim//2]

        x = self.create_rotary(x, sin, cos) # [B, T, num_heads, head_dim]
        return x.view(B, T, d_model) # [B, T, d_model]
