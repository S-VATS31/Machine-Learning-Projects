import torch

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class PatchEmbeddings(torch.nn.Module):
    def __init__(self, img_size: int, patch_size: int, C_in: int, D: int):
        """
        Initialize patch embeddings layer.

        Args:
            img_size (int): Represents the height and width of the image.
                E.g., img_size = 224 means image_width, image_height = 224, 224.
            patch_size (int): Height and width of each patch. Creates square patches.
                E.g., patch_size = 16 means patch_width, patch_height = 16, 16.
            C_in (int): Number of input channels. E.g, C_in = 3 for RGB.
            D (int): Dimensionality of the model's input/output representations.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = C_in * patch_size ** 2

        # Linear projection for 1D patches
        self.proj = torch.nn.Linear(self.patch_dim, D)

    def forward(self, x: torch.Tensor):
        """
        Perform forward pass of the Patch Embeddings layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H_in, W_in].
        
        Returns:
            x (torch.Tensor): Output tensor of [B, num_patches, D].
        """
        x = x.to(device)
        B, C_in, H_in, W_in = x.shape
        if H_in != self.img_size or W_in != self.img_size:
            raise ValueError(f"H_in ({H_in}), W_in ({W_in}) must match img_size ({self.img_size})")
        
        # Divide image into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # [B, C_in, H_in//P, P, W_in//P, P]

        # Reshape patches
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous() # [B, H_in//P, W_in//P, C_in, P, P]
        x = x.view(B, -1, self.patch_dim) # [B, num_patches, patch_dim]
        x = self.proj(x) # Linear projection
        return x
