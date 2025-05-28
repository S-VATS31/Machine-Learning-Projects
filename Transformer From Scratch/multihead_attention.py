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
            rope (RoPE): Rotary Positional Embeddings layer.
            dropout (torch.nn.Dropout): Dropout layer for regularization.
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

        self.rope = RoPE(self.d_k).to(device) # Give tokens respective positions
        self.dropout = torch.nn.Dropout(p=dropout).to(device) # Dropout to prevent overfitting

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
        kv_cache_offset: int = 0
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

        # Dynamically calculate batch size and sequence length
        B, T_current, _ = x.size()

        # Handle if sequence length is 0
        if T_current == 0:
            logger.error(f"T_current {T_current} cannot be equal to 0. Returning empty weights") # Log error
            empty_output = torch.empty(B, 0, self.d_model).to(device)
            empty_weights = torch.empty(B, self.num_heads, 0, 0).to(device)
            return empty_output, empty_weights, (None, None) # Same MHA output but with empty tensors

        # Linear projections
        Q = self.W_Q(x) # Query: [B, T, d_model]
        K = self.W_K(x) # Key:   [B, T, d_model]
        V = self.W_V(x) # Value: [B, T, d_model]

        Q = self.rope(Q, offset=kv_cache_offset) # Apply RoPE to Query Vectors with offset
        K = self.rope(K, offset=kv_cache_offset) # Apply RoPE to Key Vectors with offset

        # Reshape for multi-head attention
        Q = Q.view(B, T_current, self.num_heads, self.d_k).transpose(1, 2) # [B, num_heads, T, d_k]
        K = K.view(B, T_current, self.num_heads, self.d_k).transpose(1, 2) # [B, num_heads, T, d_k]
        V = V.view(B, T_current, self.num_heads, self.d_k).transpose(1, 2) # [B, num_heads, T, d_k]

        # Set up KV Cache
        if past_kv is not None:
            past_key, past_value = past_kv # Previous accumulated K, V
            K_full = torch.cat((past_key, K), dim=2)
            V_full = torch.cat((past_value, V), dim=2)
            # Update T for attention calculation
            current_sequence_length = K_full.size(2) # Total sequence length for K and V
        else:
            K_full = K
            V_full = V
            current_sequence_length = T_current # Initial sequence length is just T

        # Store present key value
        present_kv = (K_full, V_full) # Store full K and V

        # Log after reshaping
        logger.debug(f"Q shape: {Q.shape}, K_full shape: {K_full.shape}, V_full shape: {V_full.shape}")

        # Scaled Dot Product Attention
        if Q.shape[-1] != K_full.shape[-1]:
            logger.error(f"Dimension Mismatch: Q shape: {Q.shape}, K_full shape {K_full.shape}") # Log error
            raise ValueError(f"Dimension Mismatch: Q shape: {Q.shape}, K_full shape {K_full.shape}") # End program
        attention_scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_k) # [B, num_heads, T, T_K_full]
        
        # Autoregressive Masking
        if causal:
            # Create a mask for Q (length T_current) attending to K_full (length current_sequence_length)
            causal_mask = torch.triu(torch.ones(T_current, current_sequence_length, dtype=torch.bool, device=device), diagonal=kv_cache_offset + 1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            attention_scores.masked_fill_(causal_mask, float('-inf'))
            logger.debug(f"Autoregressive mask applied. Non-masked positions: {causal_mask.numel() - causal_mask.sum().item()}")

        # Apply padding masking
        if padding_mask is not None:
            if padding_mask.shape != (B, current_sequence_length):
                logger.error(f"padding_mask must have shape: ({B}, {current_sequence_length}), got: ({padding_mask.shape})") # Log error
                raise ValueError(f"padding_mask must have shape: ({B}, {current_sequence_length}), got: ({padding_mask.shape})") # End program
            padding_mask_expanded = padding_mask.bool().unsqueeze(1).unsqueeze(2) # [B, 1, 1, current_sequence_length]
            attention_scores = attention_scores.masked_fill(~padding_mask_expanded, float('-inf'))
            logger.debug(f"Padding mask applied. Number of masked tokens: {(~padding_mask_expanded).sum().item()}")

        # Probability Distribution
        attention_weights = F.softmax(attention_scores, dim=-1) # [B, num_heads, T, current_sequence_length]

       # Log attention weights
        logger.debug(f"attention_weights shape: {attention_weights.shape}") # Shape
        weights_sum = attention_weights.sum(dim=-1) # [B, num_heads, T_current]
        logger.debug(f"attention_weights sum shape: {weights_sum.shape}, sample sums (batch 0, head 0): {weights_sum[0, 0, :].tolist()}")
        logger.debug(f"attention_weights minimum: {attention_weights.min().item():.4f}, maximum: {attention_weights.max().item():.4f}, sum per position (min/mean/max): {weights_sum.min().item():.4f}/{weights_sum.mean().item():.4f}/{weights_sum.max().item():.4f}")

        # Ensure matrix multiplication is compatible
        if attention_weights.shape[-1] != V_full.shape[-2]:
            logger.error(f"Dimension Mismatch: attention_weights shape: {attention_weights.shape}, V_full shape: {V_full.shape}") # Log error
            raise ValueError(f"Dimension Mismatch: attention_weights shape: {attention_weights.shape}, V_full shape: {V_full.shape}") # End program
        
        # Get weighted sum of values
        attention_output = torch.matmul(attention_weights, V_full) # [B, num_heads, T, d_k]
        attention_output = self.dropout(attention_output)

        # Concatenate the attention heads and apply final projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T_current, self.d_model) # [B, T, d_model]
        attention_output = self.dropout(attention_output) # Apply Dropout

        attention_output = self.W_O(attention_output) # Final output projection

        return attention_output, attention_weights, present_kv
