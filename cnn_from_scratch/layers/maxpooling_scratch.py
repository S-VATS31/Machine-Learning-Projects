class MaxPool2D:
    def __init__(self, kernel_size: int, stride: int, padding: int = 0):
        """
        Initialize MaxPool2D layer.

        Args:
            kernel_size (int): Size of pooling window.
            stride (int): stride for convolution (applied both height and width)
            padding (int): padding applied to input (applied both height and width)
        """
        # Set up kernel_size, stride, and padding
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

    def forward(self, x: torch.Tensor):
        """
        Perform forward pass of the MaxPool2D layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height_in, width_in]

        Returns:
            x (torch.Tensor): Output tensor with max pooling applied.
        """
        # Unpack input tensor
        batch_size, in_channels, height_in, width_in = x.shape

        # Unpack kernel_size, stride, and padding
        kernel_height, kernel_width = self.kernel_size
        S_H, S_W = self.stride
        P_H, P_W = self.padding

        # Apply padding; padding = (left, right, top, bottom)
        x = F.pad(x, (P_W, P_W, P_H, P_H))

        # Unfold into patches
        x_unf = x.unfold(2, kernel_height, S_H).unfold(3, kernel_width, S_W) # [batch_size, in_channels, height_out, width_out, kernel_height, kernel_width]
        x_unf = x_unf.contiguous().view(batch_size, in_channels, -1, kernel_height * kernel_width)

        # Max over the patch dimension
        out, _ = x_unf.max(dim=-1) # [batch_size, in_channels, height_out*width_out]
        H_out = (height_in + 2 * P_H - kernel_height) // S_H + 1
        W_out = (width_in + 2 * P_W - kernel_width) // S_W + 1
        out = out.view(batch_size, in_channels, H_out, W_out)

        return out
