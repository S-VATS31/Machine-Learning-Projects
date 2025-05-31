class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self):
        """
        Initialize Convolutional Neural Network.
        """
        super().__init__()
        # Initialize Conv2D
        weights = torch.empty(16, 1, 3, 3)
        bias = torch.empty(16)
        weights, bias = Conv2D.init_weights(weights, bias)
        self.conv1 = Conv2D(weights, bias, stride=1, padding=1)

        # Initialize Conv2D
        weights = torch.empty(32, 16, 3, 3)
        bias = torch.empty(32)
        
        # Conv2D layer
        weights, bias = Conv2D.init_weights(weights, bias)
        self.conv2 = Conv2D(weights, bias, stride=1, padding=1)

        # BatchNorm layer
        self.bn1 = BatchNorm2D(16)
        self.bn2 = BatchNorm2D(32)

        # MaxPool layer
        self.maxpool1 = MaxPool2D(kernel_size=2, stride=2)
        self.maxpool2 = MaxPool2D(kernel_size=2, stride=2)

        # Fully connected layer to 10 classes
        self.linear = torch.nn.Linear(32 * 7 * 7, 10)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Perform forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]

        Returns:
            out (torch.Tensor): Output tensor of shape [batch_size, 10]
        """
        # Forward through conv1, batchnorm, ReLU, and maxpool1
        x = self.maxpool1.forward(ReLU(self.bn1(self.conv1.forward(x))))

        # Forward through conv2, batchnorm, ReLU, and maxpool2
        x = self.maxpool2.forward(ReLU(self.bn2(self.conv2.forward(x))))

        # Flatten output for fully connected layer
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Apply dropout
        x = self.dropout(x)

        # Fully connected layer
        out = self.linear(x)

        return out
