import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HousePricesNN(nn.Module):
    def __init__(self, input_size, hidden_layer1, hidden_layer2, hidden_layer3):
        super(HousePricesNN, self).__init__()

        # First Layer (Input Layer)
        self.layer1 = nn.Linear(input_size, hidden_layer1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second Layer (1st Hidden Layer)
        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(p=0.4)

        # Third Layer (2nd Hidden Layer)
        self.layer3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.relu3 = nn.LeakyReLU(0.1)
        self.dropout3 = nn.Dropout(p=0.3)

        # Fourth Layer (Output Layer)
        self.layer4 = nn.Linear(hidden_layer3, 1)

        # Initialize the weight parameter
        self.apply(self.init_weights_kaiming)

    def init_weights_kaiming(self, layer):
    # Checks if the layer is Linear (dense)
        if isinstance(layer, nn.Linear):
            init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')

        # Checks if the layer has a bias term
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias) # Initialize bias at 0

    def forward(self, x):
        x = self.relu1(self.layer1(x)) # Input Layer -> 1st Hidden Layer
        x= self.dropout1(x)
        x = self.relu2(self.layer2(x)) # 1st Hidden Layer -> 2nd Hidden Layer
        x= self.dropout2(x)
        x = self.relu3(self.layer3(x)) # 2nd Hidden Layer -> 3rd Hidden Layer
        x= self.dropout3(x)
        x = self.layer4(x) # 3rd Hidden Layer -> Output Layer

        return x
class CaliforniaHousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    # Return the number of datapoints
    def __len__(self):
        return len(self.X)

    # Return individual instances
    def __getitem__(self, index):
        return self.X[index], self.y[index]

  # Fetch data
california_housing = fetch_california_housing()

# Raw data -> DataFrame
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

# Add target variable
df['Price'] = california_housing.target

# Define X and y
X = df.drop('Price', axis=1).to_numpy()
y = df['Price'].to_numpy()

# Scale X and y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split into training/testing data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Create instances of training and testing dataset
train_dataset = CaliforniaHousingDataset(X_train, y_train)
test_dataset = CaliforniaHousingDataset(X_test, y_test)

# Create instances of training and test dataloaders
train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=96, shuffle=True)
