# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:27:11 2023
"""

from tensorflow import keras
import numpy as np
import random

# Load MNIST dataset and split into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

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

    def backward_pass_sigmoid(self, x, targets):
        activation_output = (targets - self.sigmoid(x)) * self.sigmoid_derivative(x) 
        return activation_output

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
    
    def backward_pass_relu(self, x, targets):
        activation_output = (targets - self.relu(x)) * self.relu_derivative(x) 
        return activation_output

            
# Implement softmax layer
class SoftmaxLayer:
    def softmax(x):
        e = np.exp(x)
        return e / e.sum()
    
    def softmax_derivative(self, x):
        e = self.softmax(x) * (1 - self.softmax(x))
        return e
    
    def forward_pass_softmax(self, x):
        activation_output = self.softmax(x)
        return activation_output
        
    def backward_pass_softmax(self, x, targets):
        activation_output = (targets - self.softmax(x)) * self.softmax_derivative(x) 
        return activation_output
        
# Implement dropout

# Implement a fully parametrizable neural network class
class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size, output_size, layer_size, activation_functions):
        # Initialise random weight matrices
        self.w1 = np.random.rand(input_size, hidden_size)
        self.w2 = np.random.rand(hidden_size, output_size)
        
        # Initialise biases
        self.b1 = np.zeros((1, output_size))
        self.b2 = np.zeros((1, output_size))
        
        self.layers = layer_size
        self.activation_functions_list = [activation_functions]
        
    def train(self):
        ...
