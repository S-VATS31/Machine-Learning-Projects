def ReLU(x: torch.Tensor):
    """
    Apply ReLU activation element-wise to input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape [out_channels, in_channels, height_in, width_in]

    Returns:
        x (torch.Tensor): Output tensor with ReLU applied element-wise with same shape as earlier.
    """
    return torch.clamp(x, min=0)
