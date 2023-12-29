#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:32:52 2023

@author: anson
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt

original_df = pd.read_csv('brewery_data_complete_extended.csv', index_col = 'Brew_Date', parse_dates=True)
#original_df = pd.read_csv('brewery_data_complete_extended.csv')

# function to check numerical_df_info
def original_df_info():
    # overview of numerical_df
    print('overview:')
    print(original_df.head(5), '\n')
    
    # data types of numerical_df
    print('data types:')
    print(original_df.dtypes, '\n')
    
    # missing values of numerical_df
    print('missing values:')
    print(original_df.isnull().sum(), '\n')

original_df_info()

brewery_df = original_df.copy()

# remove columns with non-numerical values, except 'Brew_Date'
brewery_df = brewery_df.drop(columns = ['Beer_Style', 'SKU', 'Location',
                                 'Ingredient_Ratio'])

# remove columns with irrelevant features
brewery_df = brewery_df.drop(columns = ['Batch_ID'])

# convert string to datetime64
#brewery_df["Brew_Date"] = brewery_df["Brew_Date"].apply(pd.to_datetime)
#brewery_df.set_index("Brew_Date", inplace=True)

# function to check numerical_df_info
def brewery_df_info():
    # overview of numerical_df
    print('overview:')
    print(brewery_df.head(5), '\n')
    
    # data types of numerical_df
    print('data types:')
    print(brewery_df.dtypes, '\n')
    
    # missing values of numerical_df
    print('missing values:')
    print(brewery_df.isnull().sum(), '\n')
    
brewery_df_info()

'''
# create line plot of sales data
plt.plot(brewery_df["Brew_Date"], brewery_df["Quality_Score"])
plt.xlabel("Brew_Date")
plt.ylabel("Quality_Score")
plt.show()
'''


plt.style.use('ggplot')
brewery_df['Quality_Score'].plot(label='CLOSE', title='Quality_Score')

'''
class BreweryDataset(dataset):
    def __innit__(self, path)'''
    