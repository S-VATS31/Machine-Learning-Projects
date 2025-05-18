# Import Libraries
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Class and Parameter Initialization
class NeuralNetwork(nn.Module):
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4):
        super(NeuralNetwork, self).__init__()

        # Feedforward network

        # Input Layer
        self.layer1 = nn.Linear(input_layer, hidden_layer1) # Input Layer --> Hidden Layer 1
        self.activate_layer1 = nn.ReLU()

        # Hidden Layer 1
        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2) # Hidden Layer 1 --> Hidden Layer 2
        self.activate_layer2 = nn.ReLU()

        # Hidden Layer 2
        self.layer3 = nn.Linear(hidden_layer2, hidden_layer3) # Hidden Layer 2 --> Hidden Layer 3
        self.activate_layer3 = nn.ReLU()

        # Hidden Layer 3
        self.layer4 = nn.Linear(hidden_layer3, hidden_layer4) # Hidden Layer 3 --> Hidden Layer 4
        self.activate_layer4 = nn.ReLU()

        # Output Layer
        self.layer5 = nn.Linear(hidden_layer4, 1) # Hidden Layer 4 --> Output Layer

        # Initialize Weights using Xavier for ReLU activation function
        self.apply(self.init_weights_xavier)

    def init_weights_xavier(self, layer):
        # Initialize Weight using Xavier Initialization
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            # Set bias = 0
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # Forward pass
    def forward(self, x):
        x = self.activate_layer1(self.layer1(x))
        x = self.activate_layer2(self.layer2(x))
        x = self.activate_layer3(self.layer3(x))
        x = self.activate_layer4(self.layer4(x))
        x = self.layer5(x)

        return x

# Metro Dataset Class
class MetroDataset(Dataset):
    # Convert X and y to FloatTensors
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    # Loop through all instances
    def __len__(self):
        return len(self.X)

    # Return specific instances
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Data Preprocessing
metro_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz" # Fetch data

# Convert data to a DataFrame
df = pd.read_csv(metro_dataset)

# Extract datetime features
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['dayofweek'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Drop columns
df.drop(columns=['date_time', 'weather_description'], inplace=True)

# Handle missing values
df = df.dropna()

# Encode text/categorical values
df = pd.get_dummies(df, columns=['holiday', 'weather_main'], drop_first=True)

# Features (X) and outputs (y)
X = df.drop(['traffic_volume'], axis=1).to_numpy()
y = df['traffic_volume'].to_numpy()

# Normalize X and y
X_scaled = StandardScaler()
y_scaled = StandardScaler()
X_normal = X_scaled.fit_transform(X)
y_normal = y_scaled.fit_transform(y.reshape(-1, 1)) # Column vector (output)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=42)

# Create instances of MetroDataset
train_dataset = MetroDataset(X_train, y_train)
test_dataset = MetroDataset(X_test, y_test)

# Create instances of DataLoader
trainloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

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
print()

