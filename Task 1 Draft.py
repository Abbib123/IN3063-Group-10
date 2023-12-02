#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:00:21 2023

@author: anson
"""
# IN3063 Mathematics and Programming for AI
# Coursework

import numpy as np

# 1a. Implement sigmoid and ReLU layers
# sigmoid
def sigmoid(x):
    e = np.exp(-x)
    s = 1 / ( 1 + e )
    return s

# derivative of sigmoid
def d_sigmoid(x):
    s = sigmoid(x)
    ds = s * ( 1 - s )
    return ds

# relu
def relu(x):
    r = max(0, x)
    return r

# derivative of relu
def d_relu(x):
    dr = np.where(x <= 0, 0, 1)
    '''
    if x >= 0:
        dr = 1
    else:
        dr = 0
        '''
    return dr
    
# 1b. Implement softmax layer
# softmax
def softmax(x):
    e = np.exp(x)
    sum_e = np.sum(e)
    s = e / sum_e
    return s
'''
# derivative of softmax
def d_softmax(x):
'''
# 1c. Implement dropout
# inverted dropout