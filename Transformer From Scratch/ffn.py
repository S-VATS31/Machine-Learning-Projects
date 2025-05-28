class FeedForwardMLP(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1
      ):
        """
        Feed-Forward Neural Network (FFN) module for Transformer models.

        Args:
            d_model (int): Dimensionality of the model's input/output representations.
            d_ffn (int): Dimensionality of the hidden layer in the feed-forward network.
                = d_model * 4.
            dropout (float, optional): Dropout probability for the output of the second linear layer.
                Defaults to 0.1.

        Attributes:
            dropout (torch.nn.Dropout): Dropout layer for regularization.
        """
        super(FeedForwardMLP, self).__init__()

        # Linear Layer 1
        self.linear1 = torch.nn.Linear(d_model, d_ffn)

        # Linear Layer 2
        self.linear2 = torch.nn.Linear(d_ffn, d_model)

        # Dropout & LayerNorm
        self.dropout = torch.nn.Dropout(dropout).to(device)

    def forward(
        self,
        x: torch.Tensor
      ):
        """
        Apply the feed-forward neural network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = x.to(device) # Ensure x is on same device

        # First linear transformation (GELU + matmul)
        x = F.gelu(self.linear1(x), approximate="tanh")

        # Second linear transformation (matmul)
        x = self.linear2(x)

        # Apply Dropout
        x = self.dropout(x)

        return x
