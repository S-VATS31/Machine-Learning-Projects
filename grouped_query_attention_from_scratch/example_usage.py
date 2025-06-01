# Hyperparameters
batch_size = 2
seq_len = 5
d_model = 64
num_heads = 8
query_groups = 4
dropout = 0.1

# Input tensor
x = torch.randn(batch_size, seq_len, d_model)

# Instantiate GQA layer
gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, query_groups=query_groups, dropout=dropout)

# Initialize weights
gqa.init_weights()

# Move to device
gqa.to(device)
x = x.to(device)

# Forward pass
output = gqa(x)

print(f"Input shape: {x.shape}") # [B, T, d_model]
print(f"Output shape: {output.shape}") # [B, T, d_model]
