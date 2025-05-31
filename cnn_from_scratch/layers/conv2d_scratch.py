class Conv2D(torch.nn.Module):
    def __init__(self, weights: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0):
        """
        Initialize Conv2D layer.

        Args:
            weights (torch.Tensor): kernels of shape [out_channels, in_channels, kernel_height, kernel_width].
            bias (torch.Tensor): bias tensor of shape [out_channels,] or None.
            stride (int): stride for convolution (applied both height and width).
            padding (int): padding to apply on input (applied both height and width).
        """
        super().__init__()
        self.weights = weights.to(device)
        self.bias = bias.to(device) if bias is not None else None

        # Set up stride and padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

    @staticmethod
    def init_weights(weights: torch.Tensor, bias: torch.Tensor = None, uniform: bool = False):
        """
        Initialize weights using Xavier initialization.

        Args:
            weights (torch.Tensor): weights tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
            bias (torch.Tensor): bias tensor of shape [out_channels,] to be initialized to 0.
            uniform (bool): True = apply uniform distribution, False = normal distribution.

        Returns:
            weights (torch.Tensor): Initialized weights tensor with previous shape.
            bias (torch.Tensor): Initialized bias tensor with previous shape.
        """
        # Calculate fan_in, fan_out
        out_channels, in_channels, kernel_height, kernel_width = weights.shape
        fan_in = in_channels * kernel_height * kernel_width
        fan_out = out_channels * kernel_height * kernel_width

        # Apply uniform distribution
        if uniform:
            limit = math.sqrt(6 / (fan_in + fan_out))
            with torch.no_grad():
                weights.uniform_(-limit, limit)
        
        # Apply normal distribution
        else:
            std = math.sqrt(2 / (fan_in + fan_out))
            with torch.no_grad():
                weights.normal_(0, std)

        # Apply zero initialization
        if bias is not None:
            with torch.no_grad():
                bias.zero_()
        
        return weights, bias
        
    def forward(self, x: torch.Tensor):
        """
        Perform forward pass of Conv2D.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height_in, width_in]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height_out, width_out]
        """
        batch_size, _, height_in, width_in = x.shape
        out_channels, in_channels, kernel_height, kernel_width = self.weights.shape

        # Ensure input tensor and weights have 4 dimensions
        if x.dim() != 4 or self.weights.dim() != 4:
            raise ValueError(f"Input tensor, x, and kernel weights, weights, must have dimension 4. x dim: {x.dim()}, weights dim: {self.weights.dim()}")

        # Unpack stride and padding height/width
        S_H, S_W = self.stride
        P_H, P_W = self.padding

        # Apply padding; padding = (left, right, top, bottom)
        x = F.pad(x, (P_W, P_W, P_H, P_H))

        # Use unfold to extract sliding local blocks
        x_unf = x.unfold(2, kernel_height, S_H).unfold(3, kernel_width, S_W) # [batch_size, in_channels, height_out, width_out, kernel_height, kernel_width]
        x_unf = x_unf.contiguous().view(batch_size, in_channels, -1, kernel_height * kernel_width) # [batch_size, in_channels, output, kernel_height*kernel_width]
        x_unf = x_unf.permute(0, 2, 1, 3).reshape(batch_size * x_unf.shape[2], in_channels * kernel_height * kernel_width) # [batch_size*output, in_channels*kernel_height*kernel_width]

        w = self.weights.view(out_channels, -1) # [out_channels, in_channels*kernel_height*kernel_width]
        out = torch.matmul(x_unf, w.T) # [batch_size*output, out_channels]
        out = out.view(batch_size, -1, out_channels).permute(0, 2, 1)

        # Compute spatial size and update out tensor
        height_out = (height_in + 2 * P_H - kernel_height) // S_H + 1
        width_out = (width_in + 2 * P_W - kernel_width) // S_W + 1
        out = out.view(batch_size, out_channels, height_out, width_out)

        # Add bias if given
        if self.bias is not None:
            out += self.bias[None, :, None, None]

        return out
