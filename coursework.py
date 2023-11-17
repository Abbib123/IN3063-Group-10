# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:27:11 2023
"""

from tensorflow import keras
import numpy as np

# Load MNIST dataset and split into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1.0 - x)


# Neural Network Class
class NeuralNetwork:

    # Implement forward and backward pass for Sigmoid
    def forward_pass_backward_pass_sigmoid(inputs, weights):
        epochs = 10
        for epoch in range(epochs):
            np.dot()
    
    # Implement forward and backward pass for ReLU
    
    # Implement softmax layer