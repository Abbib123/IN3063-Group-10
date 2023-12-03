# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:56:29 2023

@author: Tameem Al Azam
"""
import numpy as np

# Sigmoid Activation Layer
class SigmoidLayer:
    def forward(self, x):
        self.cache = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dA):
        x = self.cache
        sigmoid_x = 1 / (1 + np.exp(-x))
        return dA * sigmoid_x * (1 - sigmoid_x)

# ReLU Activation Layer
class ReLULayer:
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dA):
        x = self.cache
        return dA * (x > 0)

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.sigmoid = SigmoidLayer()
        self.relu = ReLULayer()
