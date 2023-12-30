#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:47:44 2023

@author: abbibakram
"""

from tensorflow import keras
import numpy as np
import random
from keras.datsets import mnist
import matplotlib.pyplot as plt

# Reference to:https://www.analyticsvidhya.com/blog/2021/06/mnist-dataset-prediction-using-keras/#h-step1-importing-dataset
data = mnist.load_data() 
(X_train, y_train), (X_test, y_test) = data
X_train[0].shape
X_train.shape

class neural_network_mnist :
    
    

