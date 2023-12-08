# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:27:11 2023
"""

from tensorflow import keras
import numpy as np

# Load MNIST dataset and split into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

# Implement sigmoid and ReLU layers

class SigmoidLayer:
    # Sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Derivative of Sigmoid
    def sigmoid_derivative(x):
        return x * (1.0 - x)
    
    # Implement forward and backward pass for Sigmoid
    def forward_pass_sigmoid(self, x, weights, biases):
        activation_input = np.dot(x, weights) + biases
        activation_output = self.sigmoid(activation_input)
        return activation_output

    def backward_pass_sigmoid(self, y):
        activation_output = self.sigmoid_derivative()
        
        # Update weights and biases

class ReLULayer:
    # ReLU function
    def relu(x):
        return np.maximum(0, x)
    # ReLU derivative
    def relu_derivative(x):
        return (x >= 0) * 1
    
    # Implement forward and backward pass for ReLU
    def forward_pass_relu(self, x, weights, biases):
        activation_input = np.dot(x, weights) + biases
        activation_output = self.relu(activation_input)
        return activation_output
    
    def backward_pass_relu(self, y):
        activation_output = self.relu_derivative()
        
        # Update weights and biases
            
# Implement softmax layer
class SoftmaxLayer:
    def softmax(x):
        e = np.exp(x)
        return e / e.sum()
    
    def forward_pass_softmax(self, x):
        activation_output = self.softmax(x)
        return activation_output
        
    def backward_pass_softmax(self, x):
        ...

# Implement a fully parametrizable neural network class
class NeuralNetwork:
    
    def __init__(self):
        # Initialise random weight matrices
        w1 = np.random.random(())
        w2 = np.random.random(())
        
        # Initialise biases
        b1 = np.zeros()
        b2 = np.zeros()
        
    def train(self):
        ...
    
# Implement optimizer