class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
      ):
        """
        Initialize Multi-Head Attention module.

        Args:
            d_model (int): Dimensionality of the model's input/output representations.
                Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability for attention weights and output.
                Defaults to 0.1.

        Attributes:
            d_k (int): Dimensionality of each attention head (d_model // num_heads).
            W_Q (torch.nn.Linear): Linear projection for queries.
            W_K (torch.nn.Linear): Linear projection for keys.
            W_V (torch.nn.Linear): Linear projection for values.
            W_O (torch.nn.Linear): Linear projection for output.
            positional_encoding (PositionalEncoding): Positional encoding layer.
            dropout (torch.nn.Dropout): Dropout layer for regularization.
            layer_norm (LayerNorm): Layer normalization module.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if d_model % num_heads != 0:
            logger.error(f"Invalid Configuration: d_model ({d_model}) must be divisible by num_heads ({num_heads})") # Log error
            raise ValueError(f"Invalid Configuration: d_model ({d_model}) must be divisible by num_heads ({num_heads})") # End program

        # Weight Matrices
        self.W_Q = torch.nn.Linear(d_model, d_model).to(device)
        self.W_K = torch.nn.Linear(d_model, d_model).to(device)
        self.W_V = torch.nn.Linear(d_model, d_model).to(device)
        self.W_O = torch.nn.Linear(d_model, d_model).to(device)

        # Check weight matrices for infinite values
        if torch.isinf(self.W_Q.weight).any():
            logger.warning("Inf found in Query Weight Matrix")
        if torch.isinf(self.W_K.weight).any():
            logger.warning("Inf found in Key Weight Matrix")
        if torch.isinf(self.W_V.weight).any():
            logger.warning("Inf found in Value Weight Matrix")
        if torch.isinf(self.W_O.weight).any():
            logger.warning("Inf found in Output Weight Matrix")

        self.positional_encoding = PositionalEncoding(d_model).to(device) # Give tokens respective positions
        self.dropout = torch.nn.Dropout(p=dropout).to(device) # Dropout to prevent overfitting
        self.layer_norm = LayerNorm(d_model).to(device) # Normalize and transform input tensor x

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True
      ):
        """
        Apply multi-head attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            padding (torch.Tensor, optional): Padding mask of shape (batch_size, seq_len).
                Zeros indicate padded positions; ones indicate valid positions.
                Defaults to None.
            causal (bool): If causal is true, apply autoregressive masking.

        Returns:
            tuple:
                - torch.Tensor: Attention output of shape (batch_size, seq_len, d_model).
                - torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        x = x.to(device) # Ensure x is on same device
        residual = x # Residual connection
        x = self.layer_norm(x) # Apply LayerNorm

        # Check for infinite values after PreNorm
        if torch.isinf(x).any():
            logger.warning("Inf found in x after PreNorm")

        # Dynamically calculate batch size and sequence length
        B, T, _ = x.size()

        # Apply Positional Encoding
        x = self.positional_encoding(x) # Apply Positional encoding

        # Check for infinite values after Positional Encoding
        if torch.isinf(x).any():
            logger.warning(f"Inf found in x applying Positional Encodings")

        # Linear projections
        Q = self.W_Q(x) # Query: [B, T, d_model]
        K = self.W_K(x) # Key:   [B, T, d_model]
        V = self.W_V(x) # Value: [B, T, d_model]

        # Check for infinite values after being linearly projected
        if torch.isinf(Q).any():
            logger.warning("Inf found in Query Vectors after linear projection")
        if torch.isinf(K).any():
            logger.warning("Inf found in Key Vectors after linear projection")
        if torch.isinf(V).any():
            logger.warning("Inf found in Value Vectors after linear projection")

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2) # [B, num_heads, T, d_k]
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2) # [B, num_heads, T, d_k]
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2) # [B, num_heads, T, d_k]

        # Check for infinite values after reshaping Q, K, V Vectors
        if torch.isinf(Q).any():
            logger.warning(f"Inf found in Query Vectors after reshaping")
        if torch.isinf(K).any():
            logger.warning(f"Inf found in Key Vectors after reshaping")
        if torch.isinf(V).any():
            logger.warning(f"Inf found in Value Vectors after reshaping")

        # Log after reshaping
        logger.debug(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        # Scaled Dot Product Attention
        if Q.shape[-1] != K.shape[-1]:
            logger.error(f"Dimension Mismatch: Q shape: {Q.shape}, K shape {K.shape}") # Log error
            raise ValueError(f"Dimension Mismatch: Q shape: {Q.shape}, K shape {K.shape}") # End program
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # [B, num_heads, T, T]

        # Check for infinite values calculating attention_scores
        if torch.isinf(attention_scores).any():
            logger.warning("Inf found in attention_scores")

        # Log attention score
        logger.debug(f"attention_scores shape: {attention_scores.shape}") # Shape
        logger.debug(f"attention_scores minimum: {attention_scores.min().item():.4f}, maximum: {attention_scores.max().item():.4f}, mean: {attention_scores.mean().item():.4f}") # Statistics

        # Autoregressive Masking
        if causal:
            autoregressive_mask = torch.tril(torch.ones(T, T, device=device)) # Lower triangular matrix: [T, T]
            autoregressive_mask = autoregressive_mask.unsqueeze(0).unsqueeze(0) # Add batch and head dimensions: [1, 1, T, T]
            autoregressive_mask = autoregressive_mask.expand(B, self.num_heads, T, T) # Expand for batch size and heads: [B, num_heads, T, T]

            # Apply autoregressive mask (setting future positions to -inf)
            attention_scores = attention_scores.masked_fill(autoregressive_mask == 0, float('-inf'))

            # Log details of the autoregressive mask application
            logger.debug(f"Autoregressive mask applied. Non-masked positions: {autoregressive_mask.sum().item()}")

            # Check for infinite values after applying autoregressive mask
            if torch.isinf(attention_scores).any():
                logger.warning("Inf found in attention_scores after applying autoregressive mask")

        # Probability Distribution
        attention_weights = F.softmax(attention_scores, dim=-1) # [B, num_heads, T, T]

        # Log attention weights
        logger.debug(f"attention_weights shape: {attention_weights.shape}") # Shape
        logger.debug(f"attention_weights minimum: {attention_weights.min().item():.4f}, maximum: {attention_weights.max().item():.4f}, sum (≈1): {attention_weights.sum().item():.4f}") # Statistics

        # Ensure matrix multiplication is compatible
        if attention_weights.shape[-1] != V.shape[-2]:
            logger.error(f"Dimension Mismatch: attention_weights shape: {attention_weights.shape}, V shape: {V.shape}") # Log error
            raise ValueError(f"Dimension Mismatch: attention_weights shape: {attention_weights.shape}, V shape: {V.shape}") # End program
        attention_output = torch.matmul(attention_weights, V) # [B, num_heads, T, d_k]
        attention_output = self.dropout(attention_output)

        # Concatenate the attention heads and apply final projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.d_model) # [B, T, d_model]
        attention_output = self.dropout(attention_output) # Apply Dropout

        attention_output = self.W_O(attention_output) # Final output projection

        # Apply residual connection
        final_output = attention_output + residual

        return final_output, attention_weights
