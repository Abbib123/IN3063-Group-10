#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:51:29 2024

@author: abbibakram
"""

import torch.nn as nn
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score,roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

path = "."  

filename_read = os.path.join(path, "brewery_data_complete_extended.csv")
df = pd.read_csv(filename_read)
print(df[0:10000000])

df = pd.read_csv(filename_read, na_values=['NA', '?'])




if 'Brew_Date' in df.columns:
    df = df.drop('Brew_Date', axis=1)
    
 
if 'Batch_ID' in df.columns:
    df = df.drop('Batch_ID', axis=1)
    p
    
    
if 'Bitterness' in df.columns:
    df = df.drop('Bitterness', axis=1)
    
    
    
if 'pH_Level' in df.columns:
    
    
    
if 'Loss_During_Fermentation' in df.columns:
    df = df.drop('Loss_During_Fermentation', axis=1)
    
     
      
     

if 'Alcohol_Content' in df.columns:
     df = df.drop('Alcohol_Content', axis=1)
     
     

if 'Color' in df.columns:
     df = df.drop('Color', axis=1)
        
     

if 'Ingredient_Ratio' in df.columns:
     df = df.drop('Ingredient_Ratio', axis=1)
     
     
     
if 'Loss_During_Brewing' in df.columns:
     df = df.drop('Loss_During_Brewing', axis=1)
     
     
if 'Loss_During_Bottling_Kegging' in df.columns:
     df = df.drop('Loss_During_Bottling_Kegging', axis=1)

     

if 'Location' in df.columns:
     df = df.drop('Location', axis=1)
  
     

if 'SKU' in df.columns:
     df = df.drop('SKU', axis=1)
        
     
if 'Gravity' in df.columns:
     df = df.drop('Gravity', axis=1)
     
     
#Reference to IN3062 Introduction to AI Tutorial 2,Howe,Jacob,Date accessed:06.07.2024
df = df.select_dtypes(include=['int', 'float'])

headers = list(df.columns.values)
fields = []

for field in headers:
    fields.append({
        'name' : field,
        'mean': df[field].mean(),
        'var': df[field].var(),
        'sdev': df[field].std()
    })

for field in fields:
    print(field)



# 1.Create a fully connected neural network 
# 2.Check for a gpu within the cpu
# 3.Load and preprocess the data(Already done)
# 4.Turning our data into tensors and splitting them for training and testing
# 5.Implementing data loaders
# 6.


#Persson, A. (2021). Pytorch Neural Network Tutorial. [online] Medium. Available at: https://aladdinpersson.medium.com/pytorch-neural-network-tutorial-7e871d6be7c4 [Accessed 5 Jan. 2024].
‌

training_loader = Dataloader (dataset=training_dataset, size_of_batch=size_of_batch, shuffle=True)


testing_loader = DataLoader(
    dataset=testing_dataset, size_of_batch=size_of_batch, shuffle=True
)


class Neural_network_cnn_py(nn.Module):
    def __init__(self, input, classes):
        super(Neural_network_cnn_py, self).__init__()
        self.fc3 = nn.Linear(input_size, 1000)
        self.fc4 = nn.Linear(1000, classes)

    def forward(self, input):
        input = F.relu(self.fc3(input))
        input = self.fc4(input)
        return input

     



     
     

    
    


#print(f"Before drop: {df.columns}")
#df.drop('Brew_Date')
#print(f"After drop: {df.columns}")

#df.drop('Beer_style')




