# Example usage
img_size, patch_size, C_in, D = 224, 16, 3, 64
patch_embeddings = PatchEmbeddings(img_size, patch_size, C_in, D)
patch_embeddings = patch_embeddings.to(device)

# Forward pass
B, H_in, W_in = 4, 225, 224 # H_in value should raise ValueError.
x = torch.randn(B, C_in, H_in, W_in)
output = patch_embeddings(x)
print(output.shape) # Prints shape
print(output) # Prints tensor
