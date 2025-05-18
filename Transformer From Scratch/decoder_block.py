class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super(DecoderBlock, self).__init__()

        # Multi-Head Attention
        self.MHA = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed Forward MLP
        self.MLP = FeedForwardMLP(d_ffn, d_model, dropout)

        # Layer Normalization (Pre-Norm)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        # Dropout for Regularization
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ):
        # Causal Attention + Residuals
        residual = x
        x = self.layer_norm1(x) # Pre-Norm
        attn, _ = self.MHA(
            x, # Q, K, V
            causal = True # Apply autoregressive masking
        )
        x = residual + self.dropout1(attn)

        # Feed Forward MLP + Residuals
        residual = x
        x = self.layer_norm2(x) # Pre-Norm
        ffn = self.MLP(x) # Pass through Feed Forward MLP
        x = residual + self.dropout2(ffn)

        return x
