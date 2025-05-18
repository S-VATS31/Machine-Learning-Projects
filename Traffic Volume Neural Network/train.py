# Create Model Instances
torch.manual_seed(42) # Seed for Reproducibility

# Model
neural_network = NeuralNetwork(input_layer=24, hidden_layer1=200, hidden_layer2=150, hidden_layer3=100, hidden_layer4=50)

# Cost Function
criterion = nn.MSELoss()

# Optimization Algorithm
lr=5e-6
weight_decay=4e-5
optimizer = torch.optim.AdamW(neural_network.parameters(), lr=lr, weight_decay=weight_decay)

# Learning Rate Scheduler
eta_min = 1e-6
T_max = 1000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=T_max)

# Backpropagtion, Gradient Descent, Testing vs. Training Loss
def train_nn(neural_network, criterion, optimizer, testloader, trainloader, device, max_iters):
    # Make sure model is on same device as X/y
    neural_network.to(device)

    # Set the model to training mode
    neural_network.train()

    # Empty lists for training and testing loss for plotting
    total_train_loss_list = []
    total_test_loss_list = []

    # Testing/Training Loop
    for iter in range(max_iters):

        # Initialize total loss for the epoch
        total_train_loss = 0.0
        total_test_loss = 0.0

        # L1 Regularization Strength
        l1_lambda = 1e-3

        # Training Loop
        for X_train, y_train in trainloader:

            # Ensure X/y are on the same device as the model
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Column vector
            y_train = y_train.view(-1, 1)

            optimizer.zero_grad() # Zero gradients to prevent accumulation
            y_pred = neural_network(X_train) # Forward pass
            cost = criterion(y_pred, y_train) # MSE cost function

            # Calculate cost after applying L1 penalty
            l1_normalization = sum(p.abs().sum() for p in neural_network.parameters())

            # Add L1 Regularization to cost
            cost += l1_lambda * l1_normalization

            cost.backward() # Backpropagation
            optimizer.step() # Update parameters

            total_train_loss += cost.item() # Accumulate training loss

        # Apply learning rate scheduler
        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate average training loss for this epoch
        avg_train_loss = total_train_loss / len(trainloader)
        total_train_loss_list.append(avg_train_loss)

        # Turn off gradient calculation for testing
        with torch.no_grad():

            # Testing Loop
            for X_test, y_test in testloader:
                X_test, y_test = X_test.to(device), y_test.to(device)

                # Column vector
                y_test = y_test.view(-1, 1)

                y_pred = neural_network(X_test) # Forward pass
                cost = criterion(y_pred, y_test) # MSE cost function

                total_test_loss += cost.item() # Accumulate test loss

        # Calculate average test loss for this epoch
        avg_test_loss = total_test_loss / len(testloader)
        total_test_loss_list.append(avg_test_loss)

        # Display training vs test loss
        if (iter + 1) % 100 == 0 or iter == 0:
            print(f"Iteration: {iter+1} - Training Loss: {avg_train_loss:.4f}, Testing Loss: {avg_test_loss:.4f}, Learning Rate: {current_lr:.6f}")

    # Plot Training Loss
    plt.plot(total_train_loss_list, label='Training Loss', c='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost Function over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot Testing Loss
    plt.plot(total_test_loss_list, label='Testing Loss', c='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost Function over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Call training function
train_nn(neural_network, criterion, optimizer, testloader, trainloader, device, max_iters=6000)

# Predictions on Test Data and Model Evaluation
def eval_nn(neural_network, testloader, device):
    # Make sure model is on same device as X/y
    neural_network.to(device)

    # Set model to evaluation mode
    neural_network.eval()

    # Empty lists to store predicted and actual values
    predicted_vals = []
    true_vals = []

    # Turn off gradient calculation for evaluation
    with torch.no_grad():
        # Testing/Predictions Loop
        for X_test, y_test in testloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = neural_network(X_test) # Forward Pass

            # Only squeeze if dimensions > 1
            if y_pred.dim() > 1 and y_pred.size(1) == 1:
                y_pred = y_pred.squeeze(1)
            if y_test.dim() > 1 and y_test.size(1) == 1:
                y_test = y_test.squeeze(1)

            # Convert tensors to NumPy arrays and store to CPU
            predicted_vals.append(y_pred.cpu().numpy())
            true_vals.append(y_test.cpu().numpy())

    # Concatenate all batches in a large array of values
    predicted_vals = np.concatenate(predicted_vals)
    true_vals = np.concatenate(true_vals)

    return predicted_vals, true_vals

def eval_metrics(predicted_vals, true_vals):
    # Make sure arrays have the same shape
    if predicted_vals.shape != true_vals.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted_vals.shape}, true {true_vals.shape}")

    # Calculate MAE, RMSE, and R^2
    mae = mean_absolute_error(true_vals, predicted_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, predicted_vals))
    r2 = r2_score(true_vals, predicted_vals)

    return mae, rmse, r2

# Call evaluation function
predicted_vals, true_vals = eval_nn(neural_network, testloader, device)

# Inverse scaling to original values
predicted_vals = y_scaled.inverse_transform(predicted_vals.reshape(-1, 1)).flatten()
true_vals = y_scaled.inverse_transform(true_vals.reshape(-1, 1)).flatten()

# Call evaluation metrics function
mae, rmse, r2 = eval_metrics(predicted_vals, true_vals)

# Print sample predictions
print("\nSample Predictions (Predicted vs Actual):")
for i in range(min(50, len(predicted_vals))):
    print(f"Predicted: {predicted_vals[i]:.2f}, Actual: {true_vals[i]:.2f}")

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Ideal line for points to pass through
plt.figure(figsize=(8, 6))
plt.scatter(true_vals, predicted_vals, label='(Actual Value, Predicted Value)', alpha=0.6, c='b', edgecolors='k')
plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], label='y = x', c='r', linestyle='--')
plt.title('Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Check total parameters
total_params = sum(p.numel() for p in neural_network.parameters())
print(f"Total parameters: {total_params}")

# Check trainable parameters
trainable_params = sum(p.numel() for p in neural_network.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
