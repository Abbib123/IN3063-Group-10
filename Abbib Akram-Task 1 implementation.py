#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:47:44 2023

@author: abbibakram
"""

from tensorflow import keras
import numpy as np
import random
from keras.datsets import mnist
import matplotlib.pyplot as plt

# Reference to:https://www.analyticsvidhya.com/blog/2021/06/mnist-dataset-prediction-using-keras/#h-step1-importing-dataset
data = mnist.load_data() 
(X_train, y_train), (X_test, y_test) = data
X_train[0].shape
X_train.shape

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
    
    m = y.shape[1]
    
    cost = -(1/m)*np.sum(outputnodes*np.log(a2))
    
    return cost

def backward_prop_NN(inputnodes,outputnodes,weights,forward_storage):
    
    
    
       
       
       
       
       
       
       
