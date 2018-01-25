#HANDLE YOUR IMPORTS HERE
import sys
import argparse
import numpy
import math
import pdb
import random
import pickle
import numpy as np

def load_data(filePath):
	X = []
	Y = []
	with open(filePath,'r') as file:
        	for line in file:
                	dataPoint = list(map(float,line.split(',')))
                	X.append(dataPoint[0:len(dataPoint)-1])
                	Y.append(int(dataPoint[-1]))    
	return X, Y

def softmax(z):
    """compute softmax of z"""
    shiftedExp = np.exp(z-np.max(z))
    return shiftedExp/np.sum(shiftedExp)

#PROBLEM 1
def update_weights_perceptron(X, Y, weights, bias, lr):
    """
    update the weight and bias of a NN with a single sigmoid activation layer
    and softmax output.
    Divergence measure: cross entropy
    """
    updated_weights = [] 
    updated_bias = []
    
    gradW1 = np.zeros((784,10))
    gradb1 = np.zeros(10)
    
    T = len(Y)#batch size
    loss = 0
    
    for t in range(T):
        #feed forward
        Z1 = np.dot(weights[0],X[t]) + bias[0]
        Y1 = np.maximum(0,Z1)#applying relu activation
        output = softmax(Y1)#apply softmax on last layer
        loss  += -np.log(output[Y[t]])
        oneHotEncoding = np.zeros(10)
        oneHotEncoding[Y[t]] = 1
        #back prop
        gradDivY1 = output-Y[t]
        reluJacobian = np.zeros(10)
        reluJacobian[Z1>=0]=1
        reluJacobian = np.eye(10)*reluJacobian 
        gradDivZ1 = np.dot(reluJacobian,gradDivY1)
        gradb1 += gradDivZ1
        gradW1 += np.outer(X[t],gradDivZ1)
        
    #updating weights and bias   
    updated_weights.append(weights[0]-lr*(1/T)*gradW1.T)
    updated_bias.append(bias[0]-lr*(1/T)*gradb1.T )
    print(loss)
        
    return updated_weights, updated_bias

#PROBLEM 2
def update_weights_single_layer(X, Y, weights, bias, lr):
	#INSERT YOUR CODE HERE
	return updated_weights, updated_bias

#PROBLEM 3
def update_weights_single_layer_mean(X, Y, weights, bias, lr):
	#INSERT YOUR CODE HERE
	return updated_weights, updated_bias

#PROBLEM 4
def update_weights_double_layer(X, Y, weights, bias, lr):
	#INSERT YOUR CODE HERE
	return updated_weights, updated_bias

#PROBLEM 5
def update_weights_double_layer_batch(X, Y, weights, bias, lr, batch_size):
	#INSERT YOUR CODE HERE
	return updated_weights, updated_bias

#PROBLEM 6
def update_weights_double_layer_batch_act(X, Y, weights, bias, lr, batch_size, activation):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'tanh':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'relu':
        pass
        #INSERT YOUR CODE HERE
        #INSERT YOUR CODE HERE
    return updated_weights, updated_bias

#PROBLEM 7
def update_weights_double_layer_batch_act_mom(X, Y, weights, bias, lr, batch_size, activation, momentum):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'tanh':
        pass
        #INSERT YOUR CODE HERE
    if activation == 'relu':
        pass
        #INSERT YOUR CODE HERE
        #INSERT YOUR CODE HERE
    return updated_weights, updated_bias
