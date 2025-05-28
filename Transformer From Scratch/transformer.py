class Transformer(torch.nn.Module):
    def __init__(
            self,
            config: Config
    ):
        """
        Complete Transformer architecture.

        Args:
            config (Config): Class containing the model's parameters.
        """
        super(Transformer, self).__init__()
        
        self.config = config
        
        # Model Architecture
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.d_model).to(device)
        self.dropout = torch.nn.Dropout(config.dropout).to(device)

        # Stack decoder blocks
        self.blocks = torch.nn.ModuleList([
            DecoderBlock(config.d_model, config.d_ffn, config.num_heads, config.dropout).to(device)
            for _ in range(config.num_layers)
        ])

        # Apply Layer Norm
        self.layer_norm = LayerNorm(config.d_model).to(device)

        # Final linear projection
        self.head = torch.nn.Linear(config.d_model, config.vocab_size).to(device)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(
            self,
            module: torch.nn.Linear
        ):
        """"
        Initialize weights for various components in transformer.

        Args:
            module (torch.nn.Linear): Each component in the transformer architecture to be intialized.
        """
        # If specific component is a linear layer
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 0.02 is the normal amount for transformers (GPT, BERT)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # If specific component is an embedding layer
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: torch.Tensor,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # List of (K, V) tuples
            padding_mask: Optional[torch.Tensor] = None, # Full padding mask for current total sequence
            kv_cache_offset: int = 0 # Offset for RoPE and causal mask
    ):
        """
        Forward pass of the complete transformer.

        Args:
            input_ids (torch.Tensor): Input tokens of shape: [B, T]
            padding_mask (Optional, torch.Tensor): Padding mask of shape [B, T]
                1 = Valid tokens, 0 = Invalid tokens
        
        Returns:
            torch.Tensor: Logits tensor of shape: [B, T, vocab_size]
        """

        # Token embeddings
        x = self.token_embedding(input_ids) # [B, T, d_model]

        # Apply dropout for regularization
        x = self.dropout(x)

        # Store new kv as list
        new_past_key_values = [] # Initialize empty list to collect new KVs

        # Apply decoder blocks
        for i, block in enumerate(self.blocks):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, layer_past_kv, padding_mask, kv_cache_offset) # Pass offset
            new_past_key_values.append(present_kv) # Append to build up the cache

        # Apply final Layer Norm
        x = self.layer_norm(x)

        # Apply final linear projection
        logits = self.head(x) # [B, T, vocab_size]

        return logits, new_past_key_values

    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None
    ):
        """
        Autoregressively generate using next token prediction and softmax with temperature.

        Args:
            input_ids (torch.Tensor): Input tokens of shape: [B, T].
            max_length (int): Maximum sequence length that can be generated.
            temperature (Optional, float): Controls the randomness of the output.
                Lower values make the model answers straight forward, while higher values increase creativity.
                Typical range is between 0.0 and 1.0.
                Defaults to 1.0.
                temperature equal to or less than 1e-4 will cause the softmax to converge towards argmax behavior.

        Returns:
            torch.Tensor: Generated tokens with shape: [B, T].
        """
        self.temperature = temperature

        # Ensure termperature >= 0
        if temperature <= 0:
            logger.error(f"Temperature ({temperature}) cannot be less or equal to 0.") # Log error
            raise ValueError(f"Temperature ({temperature}) cannot be less than or equal to 0.") # End program

        # Ensure input is on same device
        input_ids = input_ids.to(device) # [B, T]

        # Model generate tokens
        generated = input_ids

        # Initialize KV cache
        past_key_values = None
        kv_cache_offset = 0

        # Generate tokens autoregressively
        for _ in range(max_length - generated.size(1)): # Loop for remaining tokens to generate
            if past_key_values is None:
                current_input = generated
                full_padding_mask = (current_input != self.config.pad_token_id).int()
            else: 
                # Subsequent passes (decoding) - only process the last token
                current_input = generated[:, -1:]
                full_padding_mask = (generated != self.config.pad_token_id).int()

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                # Pass past_kv to the forward pass
                logits, past_key_values = self.forward(
                    current_input, 
                    past_key_values=past_key_values, 
                    padding_mask=full_padding_mask, 
                    kv_cache_offset=kv_cache_offset
                )

            next_token_logits = logits[:, -1, :] # [B, vocab_size]

            # Apply Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # Apply Nucleus Sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply temperature and sample
            if temperature <= 1e-4:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                prob_dist = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(prob_dist, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            kv_cache_offset = generated.size(1) - 1

        return generated
