class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize Decoder Block.

        Args:
            d_model (int): Dimensionality of the entire model.
            d_ffn (int): Dimensionality of the feedforward network (4 * d_model).
            num_heads (int): Number of attention heads.
            dropout (float): Probability of model elements being dropped out.
                Regularization to prevent overfitting.
        """
        super(DecoderBlock, self).__init__()

        # Multi-Head Attention
        self.MHA = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed Forward MLP
        self.MLP = FeedForwardMLP(d_model, d_ffn, dropout)

        # Layer Normalization (Pre-Norm)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        # Dropout for Regularization
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Per-layer KV cache
        padding_mask: Optional[torch.Tensor] = None, # Full padding mask for current total sequence
        kv_cache_offset: int = 0 # Offset for RoPE and causal mask
    ):
        """
        Forward pass through the Decoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape: [B, T, d_model].
            padding (Optional, torch.Tensor): Padding mask of shape: [B, T].
                1 = Valid tokens, 0 = Invalid tokens.
        
        Returns:
            x (torch.Tensor) = Transformed input tensor of shape: [B, T, d_model].
        """
        # Causal Attention + Residuals
        residual = x
        x_norm = self.layer_norm1(x) # Pre-Norm

        # Multi-Head Attention + Residuals
        attn, _, present_kv = self.MHA(
            x_norm,
            past_kv=past_kv,
            causal=True, # Apply autoregressive masking
            padding_mask=padding_mask, # Apply padding masking
            kv_cache_offset=kv_cache_offset
        )
        x = residual + self.dropout1(attn) # Apply residual

        # Feed Forward MLP + Residuals
        residual = x
        x_norm = self.layer_norm2(x) # Pre-Norm
        ffn = self.MLP(x_norm) # Forward pass through MLP
        x = residual + self.dropout2(ffn) # Apply residual

        return x, present_kv
