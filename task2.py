# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:27:53 2023
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Task 2

data = pd.read_csv("brewery_data_complete_extended.csv").dropna()
print(data)

X = ...
y = ...

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert data to numpy and then PyTorch tensors
X_train = torch.from_numpy(X_train)
X_test  = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test  = torch.from_numpy(y_test)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
input_size = X.shape
hidden_size = ...
output_size = ...

num_of_epochs = 1000
learning_rate = 0.01
L1_lambda = 0.01

model = NeuralNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_of_epochs):
    model.train()
    train_loss= 0.0
    
    for i in range(0, X_train.shape[0]):        
        # forward pass
        output = model(input_data)
    
        # calculate loss with mean squared error for regression
        loss_function = nn.MSELoss()
        loss = loss_function(output, target)
            
        # add l1 regularisation
        L1_loss = torch.tensor(0.0)
        for param in model.parameters():
                L1_loss += torch.sum(torch.abs(param))
        loss = loss + L1_lambda * L1_loss
        
        # clear existing gradients
        optimizer.zero_grad()
        
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        train_loss += loss.item() * input_data.size(0)