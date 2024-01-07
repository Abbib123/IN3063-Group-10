# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:27:53 2023
"""
# Atif Riaz (2023) IN3063: Programming and Mathematics for Artificial Intelligence Lab08_4. City University of London. Accessed 06/01/2024.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset (must be extracted from dataset zip file first)
data = pd.read_csv("brewery_data_complete_extended.csv")

# Drop unnecessary columns that we don't need
data = data.drop(columns = ['Beer_Style', 'SKU', 'Location', 'Ingredient_Ratio', 'Batch_ID', 'Loss_During_Brewing',
                     'Loss_During_Fermentation', 'Loss_During_Bottling_Kegging', 'Brew_Date'])

# X (factors we're interested in that can affect the beer quality)
columns = list({'Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 'Alcohol_Content',
                'Bitterness', 'Color', 'Volume_Produced', 'Total_Sales', 'Brewhouse_Efficiency'})
X = data[columns]

# y (beer quality)
y = data["Quality_Score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to Pytorch tensors with type float32 and reshape if needed
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# Neural Network Class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # ReLU layer
        self.relu = nn.ReLU()
        
    # Foward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Parameters to be used for the Neural Network class
input_size = X_train.shape[1]
hidden_size = 5
output_size = 1

# Parameters to be used for training the Neural Network
num_of_epochs = 5
learning_rate = 0.001
L1_lambda = 0.0
batch_size = 5
loss_across_epochs = []

# Initialise Neural Network model and optimiser
model = NeuralNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

# Train model
for epoch in range(num_of_epochs):
    model.train()
    train_loss= 0.0
    
    for i in range(0, X_train.shape[0], batch_size):
        input_data = X_train[i : min(X_train.shape[0], i + batch_size)]
        target = y_train[i : min(X_train.shape[0], i + batch_size)]
                
        # clear existing gradients
        optimizer.zero_grad()
        
        # forward pass
        output = model(input_data)
    
        # calculate loss with mean squared error for regression
        loss_function = nn.MSELoss()
        loss = loss_function(output, target)
            
        # add l1 regularisation
        L1_loss = 0.0
        for param in model.parameters():
                L1_loss += torch.sum(torch.abs(param))
        loss = loss + L1_lambda * L1_loss

        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        train_loss += loss.item() * input_data.size(0)
        
    # add to loss across epochs array
    loss_across_epochs.append(train_loss/X_train.size(0))
        
# Evaluate model
model.eval()

y_test_prob = model(X_test)
y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
y_train_prob = model(X_train)
y_train_pred = np.where(y_train_prob > 0.5, 1, 0)

# using mean squared error because this is a regression problem
print("y_train mean squared error: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("y_test mean squared error: ", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Loss Curve
plt.plot(loss_across_epochs)
plt.title('Loss across epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
