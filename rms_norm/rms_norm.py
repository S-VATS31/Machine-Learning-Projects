import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
      """Initialize RMSNorm layer."""
      super().__init__()
      self.eps = eps
      self.gamma = torch.nn.Parameter(torch.ones(d_model)) # Scaling factor
      self.beta = torch.nn.Parameter(torch.zeros(d_model)) # Optional bias

    def forward(self, x: torch.Tensor):
        """Perform forward pass of RMSNorm layer."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # RMSNorm formula
        x = x / rms # Divide tensor by RMS
        return self.gamma * x + self.beta # Apply scaling shifting
