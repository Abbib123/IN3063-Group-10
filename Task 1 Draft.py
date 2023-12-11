#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:00:21 2023

@author: anson
"""
# IN3063 Mathematics and Programming for AI
# Coursework

from keras.datasets import mnist

import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(len(train_y))

# 1a. Implement sigmoid and ReLU layers
# sigmoid
def sigmoid(x):
    e = np.exp(-x)
    s = 1 / ( 1 + e )
    return s

# sigmoid derivative
def d_sigmoid(x):
    s = sigmoid(x)
    ds = s * ( 1 - s )
    return ds

# sigmoid forward
def forward_sigmoid(inputs):
    outputs = sigmoid(inputs)
    return outputs

# sigmoid backward
def backward_sigmoid(output, d_cost_output):
    d_output_input = d_sigmoid(output)
    d_cost_input = d_cost_output * d_output_input
    return d_cost_input

# relu
def relu(x):
    r = np.maximum(0, x)
    return r

# relu derivative
def d_relu(x):
    dr = np.where(x > 0, 0, 1)
    return dr

# relu forward
def forward_relu(inputs):
    output = relu(inputs)
    return output

# sigmoid backward
def backward_relu(output, d_cost_output):
    d_output_input = d_relu(output)
    d_cost_input = d_cost_output * d_output_input
    return d_cost_input

# 1b. Implement softmax layer
# softmax
def softmax(v):
    e = np.exp(v)
    sum_e = np.sum(e)
    s = e / sum_e
    return s

# softmax derivative
def d_softmax(v):
    s = softmax(v)
    ds = s * (1 - s)
    return ds

# softmax forward
def forward_softmax(inputs):
    output = softmax(inputs)
    return output

# softmax backward
def backward_softmax(output, d_cost_output):
    d_output_input = d_softmax(output)
    d_cost_input = d_cost_output * d_output_input
    return d_cost_input

# softmax cross entropy loss
'''
def cross_entropy_loss_softmax(x):
    num_samples = y_pred.shape[0]
    loss = -np.sum(np.log(y_pred[np.arange(num_samples), y_true])) / num_samples
    return loss
'''
# 1c. Implement dropout
# inverted dropout