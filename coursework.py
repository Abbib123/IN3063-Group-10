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

# ReLU
def relu(x):
    return np.maximum(0, x)

# ReLU derivative
def relu_derivative(x):
    return (x >= 0) * 1

# Neural Network Class
class NeuralNetwork:
    
    # Initialise random weight matrices
    w1 = np.random.random(())
    w2 = np.random.random(())

    # Implement forward and backward pass for Sigmoid
    def forward_pass_sigmoid(inputs, weights):
        epochs = ...
        
        for epoch in range(epochs):
            z1 = np.dot()
            a1 = sigmoid(z1)
            
            z2 = np.dot()
            a2 = sigmoid(z2)
            
    def backward_pass_sigmoid(inputs, weights):
        epochs = ...
        
        for epoch in range(epochs):
            z1 = np.dot()
            a1 = sigmoid_derivative(z1)
            
            z2 = np.dot()
            a2 = sigmoid_derivative(z2)
    
    # Implement forward and backward pass for ReLU
    def forward_pass_relu(inputs, weights):
        epochs = ...
        
        for epoch in range(epochs):
            z1 = np.dot()
            a1 = relu(z1)
            
            z2 = np.dot()
            a2 = relu(z2)
            
    def backward_pass_relu(inputs, weights):
        epochs = ...
        
        for epoch in range(epochs):
            z1 = np.dot()
            a1 = relu_derivative(z1)
            
            z2 = np.dot()
            a2 = relu_derivative(z2)
    
    # Implement softmax layer