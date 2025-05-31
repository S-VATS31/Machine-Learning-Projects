# Get losses
train_losses, test_losses = [], []

# Training loop
epochs = 20
for epoch in range(epochs):
    train_loss, train_acc = train_cnn(cnn, train_loader, criterion, optimizer)
    test_loss, test_acc = test(cnn, test_loader)

    # Append to losses lists
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Display loss
    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Plot training vs. test loss
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", c="red") # Train loss
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", c="blue") # Test loss
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs. Testing loss over epochs")
plt.grid(True)
plt.legend()
plt.show()
