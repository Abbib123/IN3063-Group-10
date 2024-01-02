#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:47:44 2023

@author: abbibakram
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from keras.datsets import mnist

#from tensorflow import keras
#from keras.datsets import mnist

# Reference to:https://www.scaler.com/topics/deep-learning/how-to-build-a-neural-network/

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
        "weight_1" = weight_1,
        "bias_1" = bias_1,
        "weight_2"=weight_2,
        "bias_2" = bias_2
        
        }
    return architecture


    
                 
    
    
    
    
    
    
                           
                           
    
    
    
    
        
        

        
        
    






# Reference to:https://www.analyticsvidhya.com/blog/2021/06/mnist-dataset-prediction-using-keras/#h-step1-importing-dataset
#data = mnist.load_data() 
#(X_train, y_train), (X_test, y_test) = data
#X_train[0].shape
#X_train.shape

class neural_network_mnist:
    def __init__(self,inputnodes,outputnodes,hiddennodes,rate_of_learning):
        
        self.inputnodes = inputnodes
        self.outputnodes = outputnodes
        self.hiddennodes = hiddennodes
        self.rate_of_learning = rate_of_learning
        self.Create_weighted_nodes()
    
   
#Code for creating and training the neural network
#Reference to:https://www.youtube.com/watch?v=vtx1iwmOx10
def initialise_weights(inputnodes,hiddennodes,outputnodes):
    
    h1 = np.random.randn(hiddennodes,inputnodes)*0.01
    c1 = np.zeros((hiddennodes,1))
    
    h2 = np.random.randn(outputnodes,hiddennodes)
    c2 = np.zeros((hiddennodes,1))
    
    weights = {
        "h1" : h1,
        "c1" : c1,
        "h2" :h2,
        "c2" : c2
        
        }
    
    return weights


    #Reference to:https://www.youtube.com/watch?v=vtx1iwmOx10
def forward_prop_NN(inputnodes,weights):
       
       h1 = weights['h1']
       c1 = weights['c1']
       h2 = weights['h2']
       c2 = weights['c2']
       
       #Matrix multiplication
       k1=np.dot(h1,inputnodes) + c1
       v1 = reLu(k1)  #Implementing relu later
       
       k2=np.dot(h2,v1)+ c2
       v2=softMax(c2) #Implementing sigmoid later
       
       forward_storage = {
           "k1" : k1,
           "v1" : v2,
           "k2" : k2,
           "v2" : v2
           
           }  
       return forward_storage
       
       
def cost_function (v2, outputnodes):     
    
    m = outputnodes.shape[1]
    
    cost = -(1/m)*np.sum(outputnodes*np.log(v2))
    
    return cost

def backward_prop_NN(inputnodes,outputnodes,weights,forward_storage):
    
    h1 = weights ['h1']
    c1 = weights['c1']
    h2 = weights['h2']
    c2 = weights['c2']
    
    k1 = forward_storage['k1']
    k2 = forward_storage['k2']
    
    m = inputnodes.shape[1]
    
    #Activation functions for backpropagation
    
    ak2 = (v2 - outputnodes)
    ac2 = (1/m)*np.dot(ah2.T)
    ah2 = (1/m)*np.sum(ah2,axis= 1,keepdims= 1)
    
   
    
    ah1 = (1/m)*np.dot(w2.T, ah2)*derivative_tanh(k1) #Implementing tanh later
    ac1 = (1/m)*np.dot(ah1, x.T)
    ah1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True) 
    
    
    gradients = {
        }
    
    
    
    
    
                  
    
       
       
       
       
       
       
       
