# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:27:53 2023
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, auc, roc_curve

# Task 2
data = pd.read_csv("brewery_data_complete_extended.csv")

# X
columns = list({'Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 'Alcohol_Content',
                'Bitterness', 'Color', 'Volume_Produced', 'Total_Sales', 'Brewhouse_Efficiency'})
X = data[columns]

# y
y = data["Quality_Score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy and then PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

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
    
input_size = X_train.shape[1]
hidden_size = 5
output_size = 1

num_of_epochs = 1
learning_rate = 0.001
L1_lambda = 0.0
batch_size = 1
loss_across_epochs = []

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
        
    loss_across_epochs.append(train_loss/X_train.size(0))
        
# Evaluate model
model.eval()

y_test_prob = model(X_test)
y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
y_train_prob = model(X_train)
y_train_pred = np.where(y_train_prob > 0.5, 1, 0)

print("y_train accuracy score: ", round(accuracy_score(y_train, y_train_pred), 3))
print("y_train precision score: ", round(precision_score(y_train, y_train_pred), 3))
print("y_train score: ", round(recall_score(y_train, y_train_pred), 3))
print("y_train roc auc score: ", round(roc_auc_score(y_train, y_train_prob.detach().numpy()), 3))
print("y_test accuracy score: ", round(accuracy_score(y_test, y_test_pred), 3))
print("y_test precision score: ", round(precision_score(y_test, y_test_pred), 3))
print("y_test recall score: ", round(recall_score(y_test, y_test_pred), 3))
print("y_test roc auc score: ", round(roc_auc_score(y_test, y_test_prob.detach().numpy()), 3))
