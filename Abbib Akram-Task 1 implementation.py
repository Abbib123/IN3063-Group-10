#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:47:44 2023

@author: abbibakram
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from keras.datsets import mnist

#from tensorflow import keras
#from keras.datsets import mnist

# Reference to:https://www.scaler.com/topics/deep-learning/how-to-build-a-neural-network/
# Reference to:https://www.analyticsvidhya.com/blog/2021/06/mnist-dataset-prediction-using-keras/#h-step1-importing-dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# One-hot encode labels
y_train = to_categorical(y_train, 8)#10
y_test = to_categorical(y_test, 8)#10


# This is the implementation of the Sigmoid activation layer.This will include the programmatic
# This will include the programmatic representation of the mathematical sigmoid function,
# The mathematical derivative of the sigmoid function
# The forward propagation of the Sigmoid function
# The backward propagation of the Sigmoid function

class sigmoid_layer:
    
    def sigmoid_formula(x):
        l = 1 / (1+np.exp(-x))
        return l
    
    def derivative_sigmoid_formula(x):
        h = x * (1-x)
        return h
    
    def forward_pass_sigmoid(self,x,bias,weight):
        feeding_input_activation = np.dot(x,weight) + bias
        feeding_output_activation = self.sigmoid_formula(feeding_input_activation)
        return feeding_output_activation
        
    def backward_pass_sigmoid(self,x,bias,weight,aims):
        feeding_output_activation = (aims-self.sigmoid_formula(x)) *self.derivative_sigmoid_formula
        return feeding_output_activation
    
# This is the implementation of the RELU activation layer.This will include the programmatic
# This will include the programmatic representation of the mathematical RELU function,
# The mathematical derivative of the RELU function
# The forward propagation of the RELU function
# The backward propagation of the RELU function    
    
class ReLU_layer:
    
    def ReLU_formula(x):
        g = np.maximum(0, x)
        return g
    
    def derivative_ReLU_formula(x):
        j = (x >= 0) * 1
        return j
    
    def forward_pass_ReLU(self,x,bias,weight):
        feeding_input_activation = np.dot(x,weight) + bias
        feeding_output_activation = self.ReLU_formula(feeding_input_activation)
        return feeding_output_activation
        
    
def backward_pass_ReLU(self,x,bias,weight,aims):
         feeding_output_activation = (aims-self.ReLU_formula(x)) *self.derivative_sigmoid_formula
         return feeding_output_activation
        
     
     
        
        
      
        
        
        
        
        
        
    
















#Code for creating and training the neural network
#Reference to:https://www.youtube.com/watch?v=vtx1iwmOx10

class neural_network_mnist:
    def __init__(self,
                 inputnodes,outputnodes,
                 hiddennodes,function_of_activation,
                 rate_of_learning,epochs):
        
        self.inputnodes = inputnodes
        self.outputnodes = outputnodes
        self.hiddennodes = hiddennodes
        self.function_of_activation = function_of_activation
        self.rate_of_learning = rate_of_learning
        self.epochs = epochs
        
def initialise_architecture(self,inputnodes,outputnodes,hiddennodes):
    
    self.weight_1 = np.random.randint(inputnodes,hiddennodes)*0.01#This is to make the terms smaller
    self.bias_1 = np.zeroes((hiddennodes,1))
    self.weight_2 = np.random.randint(outputnodes,hiddennodes)*0.01
    self.bias_2 = np.zeroes((outputnodes,1))
    
    architecture = {
        
        "weight_1":self.weight_1,
        
        "bias_1":self.bias_1,
        
        "weight_2":self.weight_2,
        
        "bias_2":self.bias_2
        
        }
    return architecture

def forward_propagation_nn(inputnodes,architecture):
    
    initialise_architecture.weight_1 = architecture ['weight_1']
    initialise_architecture.bias_1 = architecture['bias_1']
    initialise_architecture.weight_2 = architecture['weight_2']
    initialise_architecture.bias_1 = architecture['bias_2']
    
    
    form_weight_1 = np.dot(initialise_architecture.weight_1,inputnodes + initialise_architecture.bias_1)
    
    form_bias_1 = tanh(form_weight_1)
    
    form_weight_2 = np.dot(initialise_architecture.weight_2,form_bias_1)+initialise_architecture.bias1
                           
    form_bias_2 = softMax(form_weight_2)
    
    
    forwardprop_value_storage = {
        "form_weight_1":form_weight_1,
        "form_bias_1 ":form_bias_2,
        "form_weight_2":form_weight_2,
        "form_bias_2":form_bias_2
        
        }
    return forwardprop_value_storage

def cost_function_nn(form_bias_2,outputnodes):
    
    h = outputnodes.shape[1]
    
    cost = -(1/h)*np.sum(outputnodes*np.log(form_bias_2))
    
    return cost

def backpropagation_nn(inputnodes,outputnodes,architecture,forwardprop_value_storage):
    
    initialise_architecture.weight_1 = architecture ['weight_1']
    initialise_architecture.bias_1 = architecture ['bias_1']
    initialise_architecture.weight_2 = architecture ['weight_2']
    initialise_architecture.bias_2 = architecture ['bias_2']
    
    form_bias_1 = forwardprop_value_storage ['form_bias_1']
    form_bias_2 = forward_propagation_nn ['form_bias_2']
    
    h = inputnodes.shape[1]
    
    form_com2 = (form_bias_2 - outputnodes)
    form_weight_2 = (1/h)*np.dot(form_com2,form_bias_1.T)
    form_bias_2 = (1/h)*np.sum(form_com2,axis=1,keepdims=True)
    
    form_com1 = (1/h)*np.dot(initialise_architecture.weight_2.T,form_com2)*derivative_tanh(form_bias_1)
    form_weight_1 = (1/h)*np.dot(form_com1,inputnodes.T)
    form_bias_1 = (1/h)*np.sum(form_com1,axis=1,keepdims=True)
    
    gradients = {
        
        "form_weight_1":form_weight_1,
        "form_bias_1": form_bias_1,
        "form_weight_2":form_weight_2,
        "form_bias_2":form_bias_2
        
        }
    return gradients

def update_architecture(architecture,gradients,rate_of_learning):
    
    weight_1 = architecture ['weight_1']
    bias_1 = architecture ['bias_1']
    weight_2 = architecture['weight_2']
    bias_2 = architecture['weight_2']
    
    form_weight_1 = gradients['form_weight_1']
    form_bias_1 = gradients['form_bias_1']
    form_weight_2 = gradients['form_bias_2']
    form_bias_2 = gradients ['form_bias_2']
    
    
    weight_1 = weight_1 - rate_of_learning *form_bias_1
    bias_1 = bias_1 -rate_of_learning*form_weight_2
    weight_2 = weight_2 - rate_of_learning*form_bias_2
    bias_2 = bias_2 - rate_of_learning*form_bias_2
    
    architecture = {
        "weight_1" : weight_1,
        "bias_1" : bias_1,
        "weight_2" : weight_2,
        "bias_2" : bias_2
        
        }
    
    return architecture



neural_network_mnist.inputnodes = 784
neural_network_mnist.hiddennodes = 128
neural_network_mnist.outputnodes = 10 
neural_network_mnist.epochs = 1000
neural_network_mnist.rate_of_learning = 0.01 # Set to the number of classes in your problem


def final_nn_model(inputnodes, outputnodes,hiddennodes, rate_of_learning, epochs):
    
    inputnodes = inputnodes.shape[0]
    outputnodes = outputnodes.shape[0]
    
    list_of_cost_value = []
    
    architecture = initialise_architecture(initialise_architecture.inputnodes,initialise_architecture.hiddennodes,
                                           initialise_architecture.outputnodes)
    
    for i in range(epochs):
        
        forwardprop_value_storage = forward_propagation_nn(forward_propagation_nn.x, forward_propagation_nn.parameters)
        
        
        cost =  cost_function_nn(forwardprop_value_storage['form_bias_2'], outputnodes)
        
        
        gradients = backpropagation_nn(inputnodes, outputnodes, architecture, forwardprop_value_storage)
        
        architecture = update_architecture(architecture, gradients, rate_of_learning)
        
        list_of_cost_value.append(cost)
        
        if(i%(epochs/10) == 0):
            print("Cost after", i, "iterations is :", cost)
        
    return architecture, cost_list
                 
    
    
    
    
    
    
                           
                           
    
    
    
    
        
        

        
        
    






# Reference to:https://www.analyticsvidhya.com/blog/2021/06/mnist-dataset-prediction-using-keras/#h-step1-importing-dataset
#data = mnist.load_data() 
#(X_train, y_train), (X_test, y_test) = data
#X_train[0].shape
#X_train.shape


    
    
                  
    
       
       
       
       
       
       
       
