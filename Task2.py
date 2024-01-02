#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:32:52 2023

@author: anson
"""

#import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable 
#from torchsummary import summary 

'''
# create dataframe
original_df = pd.read_csv('brewery_data_complete_extended.csv')

# function to check original_df
def original_df_info():
    # overview of original_df
    print('overview:')
    print(original_df, '\n')
    
    # data types of original_df
    print('data types:')
    print(original_df.dtypes, '\n')
    
    # missing values of original_df
    print('missing values:')
    print(original_df.isnull().sum(), '\n')

# check original_df
original_df_info()
'''

# create dataframe
brewery_df = pd.read_csv('brewery_data_complete_extended.csv', index_col = 'Brew_Date', parse_dates=True)

plt.style.use('ggplot')
brewery_df['Quality_Score'].plot(label='CLOSE', title='Quality_Score', marker='o', markersize=0.5)

# function to check brewery_df
def brewery_df_info():
    # overview of brewery_df
    print('overview:')
    print(brewery_df, '\n')
    
    # data types of brewery_df
    print('data types:')
    print(brewery_df.dtypes, '\n')
    
    # missing values of brewery_df
    print('missing values:')
    print(brewery_df.isnull().sum(), '\n')

# check brewery_df
brewery_df_info()

# remove columns with non-numerical values
brewery_df = brewery_df.drop(columns = ['Beer_Style', 'SKU', 'Location',
                                 'Ingredient_Ratio'])

# remove columns with irrelevant features
brewery_df = brewery_df.drop(columns = ['Batch_ID', 'Loss_During_Brewing',
                                        'Loss_During_Fermentation', 'Loss_During_Bottling_Kegging'
                                        ])

# check brewery_df
brewery_df_info()

# check for GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device:",device)

# create predictor variable X
X = brewery_df[['Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 'Alcohol_Content',
                'Bitterness', 'Color', 'Volume_Produced', 'Total_Sales', 'Brewhouse_Efficiency'
                ]]

# create target variable y
y = brewery_df[['Quality_Score']]

# scale data
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_ss = ss.fit_transform(y)

# split data and print shpae
X_train, X_test, y_train, y_test = train_test_split(X_ss, y_ss, test_size=0.20, random_state=42)
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

# prepare for the PyTorch
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))
y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 
print(X_train_tensors.shape)

#reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

class LSTM_sb(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_sb, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

# hyperparameters
num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr
input_size = 10 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes

lstm_model = LSTM_sb(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
'''
lstm_model.to(device)
for epoch in range(num_epochs):
    outputs = lstm_model(X_train_tensors_final) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
     
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)
     
    loss.backward() #calculates the loss of the loss function
     
    optimizer.step() #improve from loss, i.e backprop
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}")
'''



# convert string to datetime64
#brewery_df["Brew_Date"] = brewery_df["Brew_Date"].apply(pd.to_datetime)
#brewery_df.set_index("Brew_Date", inplace=True)

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# scale data
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.fit_transform(X_test)
y_train_ss = ss.fit_transform(y_train)
y_test_ss = ss.fit_transform(y_test)


print("Training Shape", X_train_ss.shape, y_train_ss.shape)
print("Testing Shape", X_test_ss.shape, y_train_ss.shape) 
'''

'''
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 

print(X_train_tensors.shape)
'''

'''
# create line plot of sales data
plt.plot(brewery_df["Brew_Date"], brewery_df["Quality_Score"])
plt.xlabel("Brew_Date")
plt.ylabel("Quality_Score")
plt.show()
'''

'''
plt.style.use('ggplot')
brewery_df['Quality_Score'].plot(label='CLOSE', title='Quality_Score')
'''

'''
class BreweryDataset(dataset):
    def __innit__(self, path)'''
    