class FeedForwardMLP(torch.nn.Module):
    def __init__(
        self,
        d_ffn: int,
        d_model: int,
        dropout: float =0.1
      ):
        """
        Feed-Forward Neural Network (FFN) module for Transformer models.

        Args:
            d_ffn (int): Dimensionality of the hidden layer in the feed-forward network.
            d_model (int): Dimensionality of the model's input/output representations.
            dropout (float, optional): Dropout probability for the output of the second linear layer.
                Defaults to 0.1.

        Attributes:
            weight_matrix1 (torch.nn.Parameter): Weight matrix for the first linear layer, shape (d_model, d_ffn).
            bias1 (torch.nn.Parameter): Bias for the first linear layer, shape (d_ffn).
            weight_matrix2 (torch.nn.Parameter): Weight matrix for the second linear layer, shape (d_ffn, d_model).
            bias2 (torch.nn.Parameter): Bias for the second linear layer, shape (d_model).
            dropout (torch.nn.Dropout): Dropout layer for regularization.
            layer_norm (LayerNorm): Layer normalization module.
        """
        super(FeedForwardMLP, self).__init__()

        # Linear layer 1
        self.weight_matrix1 = torch.nn.Parameter(torch.randn(d_model, d_ffn).to(device) * math.sqrt(2.0 / d_model)) # [d_model, d_ffn]
        self.bias1 = torch.nn.Parameter(torch.zeros(d_ffn).to(device)) # [d_ffn]

        # Linear layer 2
        self.weight_matrix2 = torch.nn.Parameter(torch.randn(d_ffn, d_model).to(device) * math.sqrt(2.0 / d_ffn)) # [d_ffn, d_model]
        self.bias2 = torch.nn.Parameter(torch.zeros(d_model).to(device)) # [d_model]

        # Dropout & LayerNorm
        self.dropout = torch.nn.Dropout(dropout).to(device)
        self.layer_norm = LayerNorm(d_model).to(device)

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

        # Dynamically calculate the batch size and sequence length
        B, T, _ = x.shape

        # Residual connection
        residual = x

        # PreNorm
        x = self.layer_norm(x)

        # First linear transformation (GELU + matmul)
        if x.shape[-1] != self.weight_matrix1.shape[0]:
            logger.error(f"Dimensionality Mismatch: x shape: {x.shape}, weight_matrix1 shape: {self.weight_matrix1.shape}") # Log error
            raise ValueError(f"Dimensionality Mismatch: x shape: {x.shape}, weight_matrix1 shape: {self.weight_matrix1.shape}") # End program
        x = F.gelu(torch.matmul(x, self.weight_matrix1) + self.bias1)

        # Second linear transformation (matmul)
        if x.shape[-1] != self.weight_matrix2.shape[0]:
            logger.error(f"Dimensionality Mismatch: x shape: {x.shape}, weight_matrix2 shape: {self.weight_matrix2.shape}") # Log error
            raise ValueError(f"Dimensionality Mismatch: x shape: {x.shape}, weight_matrix2 shape: {self.weight_matrix2.shape}") # End program
        x = torch.matmul(x, self.weight_matrix2) + self.bias2

        # Apply Dropout
        x = self.dropout(x)

        # Apply residual connection
        x = residual + x

        return x
