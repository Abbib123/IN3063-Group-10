# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:56:29 2023

@author: Tameem Al Azam
"""
import numpy as np

# Sigmoid Activation Layer
class SigmoidLayer:
    def forward(self, x):
        self.cache = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dA):
        x = self.cache
        sigmoid_x = 1 / (1 + np.exp(-x))
        return dA * sigmoid_x * (1 - sigmoid_x)


