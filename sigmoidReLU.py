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

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu.forward(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid.forward(self.z2)
        return self.a2

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu.backward(self.z1)
        dW1 = np.dot(x.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1