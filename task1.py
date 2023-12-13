# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:24:02 2023

"""

import numpy as np
from tensorflow.keras.datasets import mnist

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size, output_size, no_of_hidden_layers):
        self.input_size = ...
        self.hidden_size = ...
        self.output_size = ...
        self.no_of_hidden_layers = ...
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        sigmoid_function = self.sigmoid(x)
        return sigmoid_function * (1 - sigmoid_function)
        
    def relu(x):
        return np.maximum(0, x)
        
    def relu_derivative(x):
        return np.where(x > 0, 0, 1)
        
    def forward_pass(self, activation_function, x):
        ...
            
    def backward_pass(self, activation_function):
        ...
        
    def softmax():
        ...
        
    def dropout():
        ...