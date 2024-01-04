# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:25:45 2023

@author: Tameem Al Azam
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)

def softmax_derivative(Y, softmax_output):
    dZ = softmax_output - Y
    return dZ

# Neural Network Class
class NeuralNetwork:
    def __init__(self, layers, activation_funcs, learning_rate=0.01, reg_lambda=0, dropout_keep_prob=1.0):
        self.layers = layers
        self.activation_funcs = activation_funcs
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.parameters = self.initialize_parameters()
        self.velocities = self.initialize_velocities()

    def initialize_parameters(self):
        parameters = {}
        for l in range(1, len(self.layers)):
            parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layers[l], 1))
        return parameters

    def initialize_velocities(self):
        velocities = {}
        for l in range(1, len(self.layers)):
            velocities['dW' + str(l)] = np.zeros_like(self.parameters['W' + str(l)])
            velocities['db' + str(l)] = np.zeros_like(self.parameters['b' + str(l)])
        return velocities

    def forward_pass(self, X, training=True):
        cache = {'A0': X}  # Storing input layer activation
        A = X
        L = len(self.parameters) // 2  # number of layers in the neural network
    
        for l in range(1, L):
            A_prev = A 
            Z = np.dot(self.parameters['W' + str(l)], A_prev) + self.parameters['b' + str(l)]
            
            if self.activation_funcs[l-1] == "relu":
                A = relu(Z)
            elif self.activation_funcs[l-1] == "sigmoid":
                A = sigmoid(Z)
        
            cache['Z' + str(l)] = Z
            cache['A' + str(l)] = A  # Store the activation for this layer

            # Dropout (if applicable)
            if training and self.dropout_keep_prob < 1.0:
                D = np.random.rand(A.shape[0], A.shape[1]) < self.dropout_keep_prob
                A = np.multiply(A, D)
                A /= self.dropout_keep_prob
                cache['D' + str(l)] = D

        # Output layer
        ZL = np.dot(self.parameters['W' + str(L)], A) + self.parameters['b' + str(L)]
        AL = softmax(ZL)
        cache['Z' + str(L)] = ZL
        cache['A' + str(L)] = AL  # Store the activation for the output layer

        return AL, cache


    def backward_pass(self, X, Y, cache):
        grads = {}
        L = len(self.parameters) // 2  # number of layers
        m = X.shape[1]
        Y = Y.reshape(cache['A' + str(L)].shape)
        #print(f"Backward Pass: Cache Keys: {list(cache.keys())}")

        # Initializing the backpropagation
        dZL = softmax_derivative(Y, cache['A' + str(L)])
        grads["dW" + str(L)] = np.dot(dZL, cache['A' + str(L-1)].T) / m
        grads["db" + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m

        for l in reversed(range(L-1)):
            dA = np.dot(self.parameters['W' + str(l + 2)].T, dZL)
            if 'D' + str(l + 1) in cache:
                dA = np.multiply(dA, cache['D' + str(l + 1)])
                dA /= self.dropout_keep_prob
            if self.activation_funcs[l] == "relu":
                dZL = np.multiply(dA, relu_derivative(cache['Z' + str(l + 1)]))
            elif self.activation_funcs[l] == "sigmoid":
                dZL = np.multiply(dA, sigmoid_derivative(cache['Z' + str(l + 1)]))
            grads["dW" + str(l + 1)] = np.dot(dZL, cache['A' + str(l)].T) / m
            grads["db" + str(l + 1)] = np.sum(dZL, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads):
        L = len(self.parameters) // 2  # number of layers
        for l in range(L):
            self.parameters["W" + str(l+1)] -= self.learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= self.learning_rate * grads["db" + str(l+1)]

    def update_parameters_sgd(self, grads):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters['W' + str(l+1)] -= self.learning_rate * grads['dW' + str(l+1)]
            self.parameters['b' + str(l+1)] -= self.learning_rate * grads['db' + str(l+1)]

    def update_parameters_with_momentum(self, grads, beta=0.9):
        L = len(self.parameters) // 2
        for l in range(L):
            self.velocities['dW' + str(l+1)] = beta * self.velocities['dW' + str(l+1)] + (1 - beta) * grads['dW' + str(l+1)]
            self.velocities['db' + str(l+1)] = beta * self.velocities['db' + str(l+1)] + (1 - beta) * grads['db' + str(l+1)]
        
            self.parameters['W' + str(l+1)] -= self.learning_rate * self.velocities['dW' + str(l+1)]
            self.parameters['b' + str(l+1)] -= self.learning_rate * self.velocities['db' + str(l+1)]

    def compute_cost(self, Y, Y_hat):
        m = Y.shape[1]
        L = len(self.parameters) // 2
        regularized_sum = sum([np.linalg.norm(self.parameters[f'W{i+1}'])**2 for i in range(L)])
        cost = (-np.sum(Y * np.log(Y_hat + 1e-15)) / m) + (self.reg_lambda / (2 * m)) * regularized_sum
        return np.squeeze(cost)

    def train(self, X_train, Y_train, epochs, optimizer="momentum"):
        training_losses = []
        test_accuracies = []
        for i in range(epochs):
            # Forward pass
            Y_hat, cache = self.forward_pass(X_train)

            # Compute cost
            cost = self.compute_cost(Y_train, Y_hat)
            training_losses.append(cost)

            # Backward pass
            grads = self.backward_pass(X_train, Y_train, cache)
            
            # Update parameters
            if optimizer == "sgd":
                self.update_parameters_sgd(grads)
            elif optimizer == "momentum":
                self.update_parameters_with_momentum(grads)

            if i % 10 == 0 or i == epochs - 1:
                test_accuracy = self.evaluate(X_test, y_test_one_hot)
                test_accuracies.append(test_accuracy)
                
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")
            
        return training_losses, test_accuracies
        
    def predict(self, X):
        Y_hat, _ = self.forward_pass(X, training=False)
        #print(f"Shape of Y_hat before argmax: {Y_hat.shape}")  # Debugging line
        predictions = np.argmax(Y_hat, axis=0)
        return predictions


    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        labels = np.argmax(Y_test, axis=0)
        
        accuracy = np.mean(predictions == labels)
        return accuracy



# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

# Convert y_train and y_test to one-hot encoding
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train].T
y_test_one_hot = np.eye(num_classes)[y_test].T

# Training using standard SGD
nn_sgd = NeuralNetwork(layers=[784, 128, 10], activation_funcs=["relu", "softmax"], learning_rate=0.1, dropout_keep_prob=0.5)
losses_sgd, accuracies_sgd = nn_sgd.train(X_train, y_train_one_hot, epochs=1000, optimizer="sgd")

# Evaluate the network on the test set
accuracy_sgd = nn_sgd.evaluate(X_test, y_test_one_hot)
print(f"Test Accuracy: {accuracy_sgd * 100:.2f}%")

# Training using SGD with momentum
nn_momentum = NeuralNetwork(layers=[784, 128, 10], activation_funcs=["relu", "softmax"], learning_rate=0.1, dropout_keep_prob=0.5)
losses_momentum, accuracies_momentum = nn_momentum.train(X_train, y_train_one_hot, epochs=1000, optimizer="sgd")

# Evaluate the network on the test set
accuracy_momentum = nn_momentum.evaluate(X_test, y_test_one_hot)
print(f"Test Accuracy: {accuracy_momentum * 100:.2f}%")

epochs_loss = range(1, 1001)  # Assuming 1000 epochs for loss data
epochs_accuracy = range(0, 1001, 10)  # For accuracy data recorded every 10 epochs

plt.figure(figsize=(12, 5))

# Plotting training loss for different optimizers
plt.subplot(1, 2, 1)
plt.plot(epochs_loss, losses_sgd, label='Standard SGD')
plt.plot(epochs_loss, losses_momentum, label='SGD with Momentum')
plt.title("Training Loss for Different Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting test accuracy for different optimizers
plt.subplot(1, 2, 2)
plt.plot(epochs_accuracy, accuracies_sgd, label='Standard SGD')
plt.plot(epochs_accuracy, accuracies_momentum, label='SGD with Momentum')
plt.title("Test Accuracy for Different Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
