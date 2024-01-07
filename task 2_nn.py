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
    print(f"Before drop: {df.columns}")
    print("Column 'Brew_Date' successfully dropped.")
    print(f"After drop: {df.columns}")
else:
    print("Column 'Brew_Date' not found in the DataFrame.")
    
 
if 'Batch_ID' in df.columns:
    df = df.drop('Batch_ID', axis=1)
    print(f"Before drop: {df.columns}")
    print("Column 'Batch_ID' successfully dropped.")
    print(f"Before drop: {df.columns}")
else:
    print("Column 'Batch_ID' not found in the DataFrame.")
    
    
if 'Bitterness' in df.columns:
    df = df.drop('Bitterness', axis=1)
    print(f"Before drop: {df.columns}")
    print("Column 'Bitterness' successfully dropped.")
    print(f"Before drop: {df.columns}")
else:
    print("Column 'Bitterness' not found in the DataFrame.")
    
    
if 'pH_Level' in df.columns:
    df = df.drop('pH_Level', axis=1)
    print(f"Before drop: {df.columns}")
    print("Column 'pH_Level' successfully dropped.")
    print(f"Before drop: {df.columns}")
else:
    print("Column 'pH_Level' not found in the DataFrame.")
    
    
if 'Loss_During_Fermentation' in df.columns:
     df = df.drop('Loss_During_Fermentation', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Loss_During_Fermentation' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'Loss_During_Fermentation' not found in the DataFrame.")  
     

if 'Alcohol_Content' in df.columns:
     df = df.drop('Alcohol_Content', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Alcohol_Content' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'Alcohol_Content' not found in the DataFrame.")
     

if 'Color' in df.columns:
     df = df.drop('Color', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Color' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column Color' not found in the DataFrame.")     
     

if 'Ingredient_Ratio' in df.columns:
     df = df.drop('Ingredient_Ratio', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Ingredient_Ratio' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'Ingredient_Ratio' not found in the DataFrame.")  
     
     
if 'Loss_During_Brewing' in df.columns:
     df = df.drop('Loss_During_Brewing', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Loss_During_Brewing' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'Loss_During_Brewing' not found in the DataFrame.") 
     
if 'Loss_During_Bottling_Kegging' in df.columns:
     df = df.drop('Loss_During_Bottling_Kegging', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Loss_During_Bottling_Kegging' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column Loss_During_Bottling_Kegging' not found in the DataFrame.") 
     

if 'Location' in df.columns:
     df = df.drop('Location', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Location' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'Location' not found in the DataFrame.")   
     

if 'SKU' in df.columns:
     df = df.drop('SKU', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'SKU' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'SKU' not found in the DataFrame.")     
     
if 'Gravity' in df.columns:
     df = df.drop('Gravity', axis=1)
     print(f"Before drop: {df.columns}")
     print("Column 'Gravity' successfully dropped.")
     print(f"Before drop: {df.columns}")
else:
     print("Column 'Gravity' not found in the DataFrame.")  
     

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

     



     
     

    
    


#print(f"Before drop: {df.columns}")
#df.drop('Brew_Date')
#print(f"After drop: {df.columns}")

#df.drop('Beer_style')




