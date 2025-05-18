class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1
      ):
        super(PositionalEncoding, self).__init__()
        """
        Initialize sinusoidal positional encoding layer.

        Args:
            d_model (int): Dimensionality of the model's input/output representations.
                Must be even for sine/cosine splitting.

        Attributes:
            PE (torch.Tensor): Positional encoding tensor, registered as a buffer (non-learnable).
        """
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor
      ):
        """
        Apply sinusoidal positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            x (torch.Tensor): Input tensor with positional encodings added, using element-wise addition
            as well as dropout to avoid overfitting.
        """
        # Ensure x is on same device
        x = x.to(device)

        # Dynamically calculate sequence length
        T = x.size(1)

        # Create position indices for the current sequence length
        position = torch.arange(0, T).unsqueeze(1).float().to(device) # [T, 1]

        # Compute the denominator for the sinusoidal encoding
        divisor = torch.exp(torch.arange(0, self.d_model, 2).float().to(device) * -(math.log(10000.0) / self.d_model)) # [d_model//2]

        # Create Sine and Cosine encodings
        PE = torch.zeros(T, self.d_model).to(device) # [T, d_model]

        # Fill tensor with Sine and Cosine
        PE[:, 0::2] = torch.sin(position * divisor) # Even indices --> Sine (2i)
        PE[:, 1::2] = torch.cos(position * divisor) # Odd indices --> Cosine (2i+1)

        # Add positional encodings to the input tensor and apply dropout
        x = x + PE[:T, :] # Add positional encoding for the current sequence length
        x = self.dropout(x) # Apply dropout to prevent overfitting
        return x
