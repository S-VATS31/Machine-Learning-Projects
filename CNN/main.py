import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # First convolution block
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution block
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolution block
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(num_features=128)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout to reduce overfitting
        self.dropout = torch.nn.Dropout(p=0.2)

        # Linear layers
        self.linear1 = torch.nn.Linear(128 * 3 * 3, 256)
        self.linear2 = torch.nn.Linear(256, 10) # 10 outputs for MNIST dataset

    def forward(self, x):
        # Convolution blocks
        x = self.pool1(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.batch_norm3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x) # Apply dropout

        # Pass through linear layers
        x = F.relu(self.linear1(x))
        x = self.dropout(x) # Apply dropout
        x = self.linear2(x)
        return x

# Define image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model, loss, optimizer
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)

def train(cnn, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]
        cnn.train()
        train_loss = 0.0

        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = cnn(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Calculate loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Create predictions on test data
        cnn.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = cnn(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)

                # Predict most likely class
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        accuracy = 100 * correct / len(test_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Test Loss: {test_loss:.4f} - "
              f"Test Accuracy: {accuracy:.2f}% - "
              f"Learning Rate: {lr:.4f}")
        
    return train_losses, test_losses

# Example run for 5 epochs
train_losses, test_losses = train(cnn, criterion, optimizer, train_loader, test_loader, epochs=5)

# Get epochs for plotting
epochs = range(1, len(train_losses) + 1)

# Graph training vs test loss
plt.plot(epochs, train_losses, label="Training Loss", c='red')
plt.plot(epochs, test_losses, label="Testing Loss", c='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.grid(True)
plt.legend()
plt.show()

# Run prediction on one batch from test_loader
cnn.eval()

images, labels = next(iter(test_loader)) # Get a batch of test images and labels

with torch.no_grad():
    outputs = cnn(images) # Forward pass
    preds = outputs.argmax(dim=1) # Predicted classes

# Convert list to arrays for accuracy
y_hat = np.array(preds.tolist()) # Predictions
y = np.array(labels.tolist()) # Ground truth labels

# Compute accuracy
accuracy = np.mean(y_hat == y)
print(f"Accuracy: {accuracy * 100}%")
