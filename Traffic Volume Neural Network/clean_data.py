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
