# Example usage
B, T, d_model = 4, 8, 32

# Common input tensor shape
x = torch.randn(B, T, d_model)

# Instantiate
rms_norm = RMSNorm(d_model)

# Forward pass
rms_norm(x)
