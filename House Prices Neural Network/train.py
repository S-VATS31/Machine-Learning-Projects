torch.manual_seed(42)

# Create an instance of the model
house_prices_nn = HousePricesNN(input_size=8, hidden_layer1=64, hidden_layer2=32, hidden_layer3=16)

# Set up the criterion
criterion = nn.MSELoss()

# Set up optimization algorithm
optimizer = optim.Adam(house_prices_nn.parameters(), lr=0.0011)

# Function to train and evaluate the model
def train_eval_neural_net(house_prices_nn, criterion, optimizer, train_loader, test_loader, epochs=10000):

    # Make sure house_prices_nn is on the same device
    house_prices_nn.to(device)

    # Set the model to training mode
    house_prices_nn.train()

    # Empty lists to store average losses for plotting
    train_loss_list = []
    test_loss_list = []

    # Global loop
    for epoch in range(epochs):

        # Initialize total train loss
        total_train_loss = 0.0

        # Training-Specific Loop
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Turns y_train into a column vector
            y_train = y_train.view(-1, 1)

            optimizer.zero_grad() # Zero the gradients
            y_pred = house_prices_nn(X_train) # Forward pass
            loss = criterion(y_pred, y_train) # Measure loss function
            loss.backward() # Perform backpropagation
            optimizer.step() # Update weights

            total_train_loss += loss.item() # item(): tensor -> scalar

        # Calculate, store, & print training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Set the model to evaluation mode
        house_prices_nn.eval()

        # Initialize total test loss
        total_test_loss = 0.0

        with torch.no_grad(): # Turn off gradient calculation for evaluation mode

            # Testing-Specific Loop
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)

                # Turns y_train into a column vector
                y_test = y_test.view(-1, 1)

                y_pred = house_prices_nn(X_test) # Forward pass
                loss = criterion(y_pred, y_test) # Measure loss function
                total_test_loss += loss.item() # item(): tensor -> scalar

        # Calculate, store, & print testing loss
        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_list.append(avg_test_loss)

        # Print losses every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Testing Loss: {avg_test_loss:.4f}")

    # Plot the training loss over epochs
    plt.plot(range(epochs), train_loss_list, label='Training Loss', color='orange')
    plt.plot(range(epochs), test_loss_list, label='Testing Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Testing Loss over Epochs')
    plt.legend()
    plt.show()

# Call training and evaluation function
train_eval_neural_net(house_prices_nn, criterion, optimizer, train_loader, test_loader, epochs=1500)

# Create a function to evaluate the model on testing data
def eval_test_data(house_prices_nn, test_loader, device):

    # Set the model to evaluation mode
    house_prices_nn.eval()

    # Make sure house_prices_nn is on the same device
    house_prices_nn.to(device)

    # Empty lists to store predictions and true values
    predictions = []
    true_values = []

    with torch.no_grad(): # Turn off gradient calculation for prediction creation
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_pred = house_prices_nn(X_test) # Forward pass

            # Remove dimensions
            if y_pred.dim() > 1 and y_pred.size(1) == 1:
                y_pred = y_pred.squeeze()
            if y_test.dim() > 1 and y_test.size(1) == 1:
                y_test = y_test.squeeze()

            # Convert to numpy arrays and store in CPU
            predictions.append(y_pred.cpu().numpy())
            true_values.append(y_test.cpu().numpy())

    # Concatenate all batches in a large array of values
    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)

    return predictions, true_values # Return both predictions and true values

def evaluate_metrics(predictions, true_values):
    # Calculate MAE, RMSE, and R^2
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)

    return mae, rmse, r2

# Call evaluation function
predicted_prices, actual_prices = eval_test_data(house_prices_nn, test_loader, device)

# Calculate metrics function
mae, rmse, r2 = evaluate_metrics(predicted_prices, actual_prices)

# Print predicted vs actual prices for 50 houses
print("Predicted vs Actual Housing Prices:")
for i in range(min(50, len(predicted_prices))):
    print(f"Predicted: ${predicted_prices[i] * 100000:.2f}, Actual: ${actual_prices[i] * 100000:.2f}")

print(f"Mean Absolute Error (MAE): {mae * 100:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R²: {r2:.4f}")

