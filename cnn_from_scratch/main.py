# Load MNIST dataset
train_dataset = datasets.MNIST(root='./mnist', train=True, download=True, 
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

test_dataset = datasets.MNIST(root='./mnist', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, loss function, optimizer
cnn = ConvolutionalNeuralNetwork().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

def train_cnn(model, train_loader, criterion, optimizer):
    """
    Train looping for CNN.

    Args:
        model (ConvolutionalNeuralNetwork): Model to be trained.
        train_loader (DataLoader): Loader to be iterated through during training.
        criterion (nn.CrossEntropyLoss): Loss function to be minimized.
        optimizer (optim.Adam): Optimization algorithm to minimize loss.
    
    Returns:
        avg_loss (float): Calculated by dividing total loss by number of samples.
        accuracy (float): Calculated by dividing total correct predictions by number of samples.
    """
    model.train()
    train_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        # Ensure on same device
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model.forward(images)

        # Compute loss and backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        train_loss += loss.item() * images.size(0)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

    # Calculate average loss and accuracy
    avg_loss = train_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def test(model, test_loader):
    """
    Testing loop for CNN.

    Args:
        model (ConvolutionalNeuralNetwork): Model to be tested.
        test_loader (DataLoader): Loader to be iterated through during testing.

    Returns:
        accuracy (float): Calculated by dividing total correct predictions by number of samples.
    """
    model.eval()
    test_loss = 0.0
    total_correct = 0
    total_samples = 0
    # Turn off gradient computation
    with torch.no_grad():
        for images, labels in test_loader:
            # Ensure images, labels on same device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model.forward(images)

            # Compute test loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
        
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    # Calculate accuracy
    avg_loss = test_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
